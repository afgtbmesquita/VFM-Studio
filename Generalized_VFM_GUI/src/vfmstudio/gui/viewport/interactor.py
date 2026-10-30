"""PyVista / VTK viewport embedded in Qt."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QObject, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QGridLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from vfmstudio.core import bridge, mesh_convert
from vfmstudio.core.document import ExperimentModel, SessionDocument
from vfmstudio.core.free_mesh import (
    inspect_free_points,
    nearest_point_index,
    offset_label_display,
    snap_coordinate,
)
from vfmstudio.core.drag_mesh import (
    apply_grid_line_drag,
    axis_nodes_from_mesh,
    choose_drag_axis,
    dic_axis_targets,
    element_edge_records,
    grid_line_segments,
    nodes_on_grid_line,
    pick_drag_target,
    restore_mesh_axes,
    snap_axis_value,
    snapshot_mesh_axes,
)
from vfmstudio.core.fields import (
    experiment_alignment_angle,
    has_deformed_configuration,
    load_step_fields,
    referenced_step_dataset,
    specimen_points_xyz,
)
from vfmstudio.core.step_series import resolve_step_data_file, viewport_caption
from vfmstudio.gui.session import AppSession
from vfmstudio.gui.viewport import overlays
from vfmstudio.gui.viewport.load_plot import LoadPlotOverlay
from vfmstudio.gui.viewport.picking import (
    nearest_mesh_edge,
    nearest_node_index,
    parse_bc_actor_name,
)


def viewport_enabled() -> bool:
    if os.environ.get("VFMSTUDIO_NO_VISTA", "").strip().lower() in {"1", "true", "yes"}:
        return False
    if os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen":
        return False
    if "PYTEST_CURRENT_TEST" in os.environ:
        return False
    return True


class ViewportWidget(QWidget):
    error_logged = Signal(str)
    request_add_experiment = Signal()
    request_open_project = Signal()
    picked = Signal(object)
    step_delta = Signal(int)
    free_mesh_changed = Signal()
    drag_mesh_committed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._plotter = None
        self._scene_key = None
        self._mesh_key = None
        self._camera_key = None
        self._mesh_pkg = None
        self._dic_cache: dict = {}
        self._bc_key = None
        self._last_dic_xyz = None
        self._xy_axes_on = False
        self._element_counts_on = False
        self._showing_deformed = False
        self._last_applied_force = None
        self._ref_xyz = None
        self._uvw = None
        self._warp_mesh = None
        self._session: AppSession | None = None
        self._load_plot_visible = False
        self._last_field_name = ""
        self._last_field_values = None
        self._last_field_xyz = None
        self._pick_filter = None
        self._resize_frozen = False
        self._allow_resize_freeze = False
        self._xy_locked = False
        self._saved_camera = None
        self._saved_style = None
        self._drag_index = None
        self._drag_mesh_state = None
        self._drag_mesh_hover = None
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(80)
        self._resize_timer.timeout.connect(self.resume_live_render)
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._stack = QStackedWidget()
        layout.addWidget(self._stack, 0, 0)

        self._vista_page = QWidget()
        vista_layout = QGridLayout(self._vista_page)
        vista_layout.setContentsMargins(0, 0, 0, 0)
        vista_layout.setSpacing(0)
        self._resize_cover = QLabel(self._vista_page)
        self._resize_cover.setScaledContents(True)
        self._resize_cover.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._resize_cover.hide()
        self._stack.addWidget(self._vista_page)

        self._empty_page = QWidget()
        empty = QVBoxLayout(self._empty_page)
        empty.setContentsMargins(40, 40, 40, 40)
        hint = QLabel(
            "This project has no experiments.\n"
            "Add an experiment or open an existing project."
        )
        hint.setWordWrap(True)
        add_exp = QPushButton("Add experiment")
        add_exp.clicked.connect(self.request_add_experiment.emit)
        open_proj = QPushButton("Open…")
        open_proj.clicked.connect(self.request_open_project.emit)
        empty.addStretch(1)
        empty.addWidget(hint)
        empty.addWidget(add_exp)
        empty.addWidget(open_proj)
        empty.addStretch(2)
        self._stack.addWidget(self._empty_page)

        self._load_plot = LoadPlotOverlay()
        self._load_plot.closed.connect(self.hide_load_plot)
        self._stack.addWidget(self._load_plot)

        if not viewport_enabled():
            self._pv = None
            fallback = QLabel("3D viewport disabled in this session.")
            fallback.setWordWrap(True)
            vista_layout.addWidget(fallback, 0, 0)
            return
        try:
            import pyvista as pv
            from pyvistaqt import QtInteractor

            self._pv = pv
            try:
                self._plotter = QtInteractor(
                    self._vista_page,
                    auto_update=False,
                    line_smoothing=False,
                    point_smoothing=False,
                    polygon_smoothing=False,
                )
            except TypeError:
                self._plotter = QtInteractor(self._vista_page)
            if hasattr(self._plotter, "set_multi_samples"):
                try:
                    self._plotter.set_multi_samples(0)
                except Exception:
                    pass
            window = None
            try:
                window = getattr(self._plotter, "ren_win", None) or getattr(
                    self._plotter, "render_window", None
                )
                if window is not None:
                    window.SetDesiredUpdateRate(0.0001)
            except Exception:
                pass
            interactor = self._plotter.interactor
            interactor.setMinimumSize(0, 0)
            interactor.setSizePolicy(
                QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored
            )
            vista_layout.addWidget(interactor, 0, 0)
            vista_layout.addWidget(self._resize_cover, 0, 0)
            self._bc_key = None
            self._install_pick_filter()
        except Exception as exc:
            self._pv = None
            fallback = QLabel(
                "3D viewport unavailable.\n"
                f"{exc}\nInstall pyvista, pyvistaqt and vtk."
            )
            fallback.setWordWrap(True)
            vista_layout.addWidget(fallback, 0, 0)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._plotter is not None:
            try:
                self._plotter.render()
            except Exception:
                pass
        self._allow_resize_freeze = True

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._allow_resize_freeze:
            self.pause_live_render()

    def pause_live_render(self) -> None:
        """Swap the VTK view for a still image until the size stops changing."""
        if self._plotter is None or not self._allow_resize_freeze:
            return
        if not self.isVisible():
            return
        if self._stack.currentWidget() is not self._vista_page:
            return
        self._resize_timer.start()
        if self._resize_frozen:
            return
        self._resize_frozen = True
        try:
            self._plotter.suppress_rendering = True
            pixmap = self._plotter.grab()
            if pixmap.isNull():
                return
            self._resize_cover.setPixmap(pixmap)
            self._resize_cover.show()
            self._resize_cover.raise_()
            self._plotter.hide()
        except Exception:
            self._resize_frozen = False
            try:
                self._plotter.suppress_rendering = False
            except Exception:
                pass

    def resume_live_render(self) -> None:
        """Show VTK again at the final size and render once."""
        self._resize_timer.stop()
        if not self._resize_frozen:
            return
        self._resize_frozen = False
        if self._plotter is None:
            self._resize_cover.hide()
            return
        try:
            self._plotter.suppress_rendering = False
            self._plotter.show()
            self._resize_cover.hide()
            self._plotter.render()
        except Exception:
            pass

    def clear(self) -> None:
        if self._plotter is not None:
            self._plotter.clear()

    def reset_camera(self) -> None:
        if self._plotter is not None:
            self._plotter.reset_camera()

    def show_load_plot(self, session: AppSession) -> None:
        self._session = session
        self._load_plot_visible = True
        self._load_plot.set_session(session)
        self._stack.setCurrentWidget(self._load_plot)

    def hide_load_plot(self) -> None:
        self._load_plot_visible = False
        if self._session is not None and not self._session.document.experiments:
            self._stack.setCurrentWidget(self._empty_page)
        else:
            self._stack.setCurrentWidget(self._vista_page)
        if self._session is not None:
            self.refresh(self._session, force=True)

    def load_plot_visible(self) -> bool:
        return self._load_plot_visible

    def _update_caption(self, session: AppSession) -> None:
        if self._plotter is None:
            return
        experiment = session.selected_experiment()
        if experiment is None:
            overlays.add_viewport_caption(self._plotter, "")
            return
        step = session.selected_step()
        step_index = 0
        if experiment.steps:
            if session.selection.step_index is not None:
                step_index = int(session.selection.step_index)
            elif step is not None:
                try:
                    step_index = experiment.steps.index(step)
                except ValueError:
                    step_index = 0
        overlays.add_viewport_caption(
            self._plotter,
            viewport_caption(
                experiment,
                step,
                step_index=step_index,
                applied_force=self._last_applied_force,
            ),
        )

    def apply_deformation_scale(
        self, session: AppSession, scale: float, *, live: bool = True
    ) -> None:
        """Warp the existing DIC surface without retessellating or rebuilding BCs."""
        self._session = session
        if self._plotter is None or self._ref_xyz is None or self._uvw is None:
            return
        if not self._showing_deformed:
            return
        factor = max(0.0, float(scale))
        pts = self._ref_xyz + factor * self._uvw
        self._last_dic_xyz = pts
        overlays.set_mesh_points(self._warp_mesh, pts)
        if not live and self._mesh_pkg is not None:
            experiment = session.selected_experiment()
            if experiment is not None:
                overlays.update_bc_glyphs(
                    self._plotter,
                    self._pv,
                    experiment,
                    self._mesh_pkg,
                    session.selection,
                    surface_xyz=pts,
                )
        try:
            self._plotter.render()
        except Exception:
            pass
        self._scene_key = _scene_key(session)

    def refresh(self, session: AppSession, force: bool = False) -> None:
        self._session = session
        if self._load_plot_visible:
            self._load_plot.set_session(session)
            return
        if not session.document.experiments:
            self._stack.setCurrentWidget(self._empty_page)
            session.status_n_points = 0
            self._scene_key = None
            return
        self._stack.setCurrentWidget(self._vista_page)
        key = _scene_key(session)
        if not force and key == self._scene_key and self._plotter is not None:
            return
        if self._plotter is None:
            return
        experiment = session.selected_experiment()
        mesh_key = _mesh_key(session)
        camera_key = _camera_key(session)
        bc_key = _bc_select_key(session)
        rebuild_mesh = force or mesh_key != self._mesh_key or self._mesh_pkg is None
        reset_camera = camera_key != self._camera_key
        if self._xy_tool_active(session) and self._xy_locked:
            reset_camera = False
        if experiment is None:
            self._plotter.clear()
            overlays.add_plain_text(self._plotter, "No experiment", font_size=8)
            session.status_n_points = 0
            self._scene_key = key
            self._mesh_key = mesh_key
            self._bc_key = bc_key
            self._mesh_pkg = None
            self._last_dic_xyz = None
            self._xy_axes_on = False
            self._element_counts_on = False
            self._showing_deformed = False
            self._last_applied_force = None
            self._ref_xyz = None
            self._uvw = None
            self._warp_mesh = None
            overlays.remove_node_highlights(self._plotter)
            self._update_caption(session)
            return
        try:
            if rebuild_mesh:
                mesh_pkg = self._build_mesh(session.document, experiment)
                self._plotter.clear()
                overlays.add_mesh_wireframe(self._plotter, self._pv, mesh_pkg)
                if not self._xy_tool_active(session):
                    overlays.add_global_axes(self._plotter, self._pv, mesh_pkg)
                overlays.add_bc_glyphs(
                    self._plotter, self._pv, experiment, mesh_pkg, session.selection
                )
                overlays.add_material_axes(self._plotter, experiment, mesh_pkg)
                self._mesh_pkg = mesh_pkg
                self._mesh_key = mesh_key
                self._bc_key = bc_key
            else:
                overlays.remove_named(self._plotter, "dic", "contour", "cid", "deformed_msg", "dic_error")
                if bc_key != self._bc_key and self._mesh_pkg is not None:
                    overlays.update_bc_glyphs(
                        self._plotter, self._pv, experiment, self._mesh_pkg, session.selection
                    )
                    self._bc_key = bc_key
        except Exception as exc:
            overlays.add_plain_text(self._plotter, f"Mesh error:\n{exc}", font_size=9)
            return

        n_points = self._update_dic_cloud(session, experiment)
        if self._mesh_pkg is not None and self._pv is not None:
            if self._showing_deformed and not bool(
                getattr(session, "drag_mesh_mode", False)
            ):
                overlays.set_named_visible(self._plotter, "mesh", False)
            else:
                overlays.add_mesh_wireframe(self._plotter, self._pv, self._mesh_pkg)
            overlays.update_bc_glyphs(
                self._plotter,
                self._pv,
                experiment,
                self._mesh_pkg,
                session.selection,
                surface_xyz=self._last_dic_xyz,
            )
            self._bc_key = bc_key
        axes_were_on = self._xy_axes_on
        self._sync_xy_axes(session)
        self._sync_element_point_counts(session)
        self._sync_node_highlights(session)
        try:
            self._plotter.render()
        except Exception:
            pass
        if reset_camera or rebuild_mesh or (session.show_xy_axes and not axes_were_on):
            if self._xy_tool_active(session):
                self._lock_xy_view(fit=not self._xy_locked)
            else:
                self._fit_camera(deformed=self._showing_deformed)
            self._camera_key = camera_key
        if bool(getattr(session, "free_mesh_mode", False)):
            self._lock_xy_view(fit=False)
            self.sync_free_mesh_overlay(session)
        if bool(getattr(session, "drag_mesh_mode", False)):
            self._lock_xy_view(fit=False)
            self.sync_drag_mesh_overlay(session)
        self._update_caption(session)
        session.status_n_points = n_points
        self._scene_key = key

    def set_node_highlights(self, node_ids, *, show_all: bool = False) -> None:
        """Update node markers without rebuilding the mesh or resetting the camera."""
        self._apply_node_highlights(node_ids, show_all=show_all)
        if self._plotter is None:
            return
        try:
            self._plotter.render()
        except Exception:
            pass

    def _sync_node_highlights(self, session: AppSession) -> None:
        ids, show_all = _node_highlight_spec(session)
        self._apply_node_highlights(ids, show_all=show_all)

    def _apply_node_highlights(self, node_ids, *, show_all: bool = False) -> None:
        if self._plotter is None:
            return
        if self._pv is None or self._mesh_pkg is None:
            overlays.remove_node_highlights(self._plotter)
            return
        overlays.update_node_highlights(
            self._plotter,
            self._pv,
            self._mesh_pkg,
            node_ids,
            show_all=show_all,
        )

    def show_virtual_field(self, session: AppSession, parameter_index: int = 0) -> str | None:
        """Warp the mesh by the stored optimal virtual field. Returns an error string or None."""
        if session.result is None:
            return "Run Identify first. Optimal virtual fields are computed with FGLS."
        from vfmstudio.core.virtual_fields import virtual_fields_for_target

        payload = virtual_fields_for_target(
            session.document,
            session.result,
            experiment_id=session.vf_experiment_id,
            step_index=session.vf_step_index,
        )
        if payload is None or not payload.u_list:
            return "No virtual fields on this result. Re-run Identify."
        if self._plotter is None or self._pv is None:
            return "3D viewport is not available in this session."
        index = min(max(int(parameter_index), 0), len(payload.u_list) - 1)
        u = np.asarray(payload.u_list[index], dtype=float).ravel()
        mesh_pkg = payload.mesh_pkg
        experiment = None
        if session.vf_experiment_id is not None:
            experiment = session.document.experiment_by_id(session.vf_experiment_id)
        if mesh_pkg is None:
            experiment, mesh_pkg = self._mesh_for_virtual_field(session, u)
        elif experiment is None:
            experiment = session.selected_experiment()
        if experiment is None or mesh_pkg is None:
            return (
                f"No experiment mesh matches the stored virtual field "
                f"({u.size} DOFs)."
            )
        try:
            post = bridge.vfm_virtual_field_post()
            nodal = getattr(post, "nodal_field", None)
            if nodal is None:
                return "Virtual-field post-processor is missing nodal_field."
            dof = _virtual_dof_label(mesh_pkg)
            w = np.asarray(nodal(mesh_pkg, u, dof), dtype=float)
            grid = overlays.unstructured_quad_grid(
                self._pv, mesh_pkg, z=w * float(session.warp_scale)
            )
            name = f"{dof.lower()}*"
            grid[name] = w
            self._plotter.clear()
            self._plotter.add_mesh(grid, scalars=name, cmap="coolwarm", show_edges=True)
            overlays.add_material_axes(self._plotter, experiment, mesh_pkg)
            self._plotter.reset_camera()
            self._scene_key = None
            self._mesh_key = None
            self._camera_key = None
            self._mesh_pkg = mesh_pkg
            self._last_dic_xyz = None
            self._xy_axes_on = False
        except Exception as exc:
            return str(exc)
        return None

    def _mesh_for_virtual_field(self, session: AppSession, u: np.ndarray):
        ordered = []
        seen = set()
        selected = session.selected_experiment()
        if selected is not None:
            ordered.append(selected)
            seen.add(selected.experiment_id)
        for experiment in session.document.experiments:
            if experiment.experiment_id in seen:
                continue
            seen.add(experiment.experiment_id)
            ordered.append(experiment)
        n_dof = int(np.asarray(u).size)
        for experiment in ordered:
            try:
                mesh_pkg = self._build_mesh(session.document, experiment)
            except Exception:
                continue
            if int(getattr(mesh_pkg, "n_dofs", -1)) == n_dof:
                return experiment, mesh_pkg
        return None, None

    def _sync_xy_axes(self, session: AppSession) -> None:
        overlays.remove_xy_axes(self._plotter)
        force = self._xy_tool_active(session)
        if (not session.show_xy_axes and not force) or self._mesh_pkg is None or self._pv is None:
            self._xy_axes_on = False
            return
        overlays.add_specimen_xy_axes(
            self._plotter, self._pv, self._mesh_pkg, self._last_dic_xyz
        )
        self._xy_axes_on = True

    def _sync_element_point_counts(self, session: AppSession) -> None:
        overlays.remove_element_point_counts(self._plotter)
        self._element_counts_on = False
        if (
            not session.show_element_point_counts
            or self._mesh_pkg is None
            or self._plotter is None
        ):
            return
        xy = None
        if self._last_dic_xyz is not None and self._last_dic_xyz.shape[0]:
            xy = np.asarray(self._last_dic_xyz[:, :2], dtype=float)
        if xy is None or xy.shape[0] == 0:
            return
        try:
            counts = mesh_convert.count_dic_points_per_element(self._mesh_pkg, xy)
            overlays.add_element_point_counts(self._plotter, self._mesh_pkg, counts)
            self._element_counts_on = True
        except Exception:
            return

    def _fit_camera(self, *, deformed: bool) -> None:
        """Fit the scene; use an isometric view when the plate is warped."""
        if self._xy_locked:
            self._lock_xy_view(fit=True)
            return
        try:
            if deformed:
                self._plotter.view_isometric()
        except Exception:
            pass
        try:
            self._plotter.reset_camera()
        except Exception:
            pass

    def _update_dic_cloud(self, session: AppSession, experiment: ExperimentModel) -> int:
        overlays.remove_named(self._plotter, "dic", "contour", "cid", "deformed_msg", "dic_error")
        self._last_dic_xyz = None
        self._showing_deformed = False
        self._last_applied_force = None
        self._ref_xyz = None
        self._uvw = None
        self._warp_mesh = None
        scalar_name = session.current_scalar
        if scalar_name.lower() == "cid":
            overlays.set_named_visible(self._plotter, "mesh", True)
            try:
                return overlays.add_cid_cloud(
                    self._plotter,
                    self._pv,
                    session.identifiability,
                    experiment_name=getattr(experiment, "name", None),
                )
            except Exception as exc:
                self._show_dic_error(f"Could not display CID:\n{exc}")
                return 0
        data_path = _step_data_path(session)
        if data_path is None:
            overlays.set_named_visible(self._plotter, "mesh", True)
            step = session.selected_step()
            if step is not None:
                message = f"Measurement file not found:\n{step.data_file}"
                self._show_dic_error(message)
            return 0
        try:
            cache_key = (
                str(data_path),
                str(session.document.material.analysis_type),
                experiment.sample_area_total,
                bool(experiment.fit_data_to_system_axes),
            )
            dataset = self._dic_cache.get(cache_key)
            if dataset is None:
                angle = experiment_alignment_angle(
                    experiment,
                    session.document.material.analysis_type,
                    session.document.source_path,
                )
                dataset = load_step_fields(
                    data_path,
                    session.document.material.analysis_type,
                    experiment.sample_area_total,
                    align_to_axes=bool(experiment.fit_data_to_system_axes),
                    alignment_angle_deg=angle if experiment.fit_data_to_system_axes else None,
                )
                self._dic_cache[cache_key] = dataset
            dataset = referenced_step_dataset(
                dataset,
                experiment,
                analysis_type=session.document.material.analysis_type,
                source_path=session.document.source_path,
                cache=self._dic_cache,
            )
            want_deformed = bool(experiment.show_deformed_configuration)
            can_deform = has_deformed_configuration(dataset)
            self._last_applied_force = dataset.attrs.get("applied_force")
            if want_deformed and not can_deform:
                try:
                    overlays.add_plain_text(
                        self._plotter,
                        "Deformed configuration not available",
                        font_size=9,
                        name="deformed_msg",
                    )
                except Exception:
                    pass
            showing = want_deformed and can_deform
            warp = 1.0
            if showing:
                warp = float(getattr(experiment, "deformation_scale", 1.0) or 1.0)
            # Tessellate / undeformed view: flat plate (Z = 0). Measured Z is
            # only the warp origin once deformed configuration is on.
            ref = specimen_points_xyz(dataset, deformed=showing, scale=0.0)
            uvw = None
            if can_deform:
                n = int(ref.shape[0])
                uvw = np.column_stack(
                    [
                        np.asarray(dataset["u"].values, dtype=float).ravel()[:n],
                        np.asarray(dataset["v"].values, dtype=float).ravel()[:n],
                        np.asarray(dataset["w"].values, dtype=float).ravel()[:n],
                    ]
                )
                n = min(n, uvw.shape[0])
                ref = ref[:n]
                uvw = uvw[:n]
            self._ref_xyz = ref
            self._uvw = uvw
            pts = ref if not showing or uvw is None else ref + warp * uvw
            self._last_dic_xyz = pts
            self._showing_deformed = showing
            n_points = pts.shape[0]
            overlays.set_named_visible(self._plotter, "mesh", not showing)
            available = [str(name) for name in dataset.data_vars]
            session.available_scalars = tuple(available)
            if session.display_mode == "points":
                self._warp_mesh = overlays.add_dic_points(
                    self._plotter, self._pv, pts, color="#B8B8B8"
                )
                return n_points
            if scalar_name not in dataset:
                self._warp_mesh = overlays.add_dic_points(
                    self._plotter, self._pv, pts, color="#B8B8B8"
                )
                return n_points
            self._warp_mesh = overlays.add_field_contour(
                self._plotter,
                self._pv,
                ref,
                dataset[scalar_name].values,
                scalar_name,
                deformed=showing,
                clim=getattr(session, "scalar_clim", None),
            )
            self._last_field_name = str(scalar_name)
            self._last_field_values = np.asarray(dataset[scalar_name].values, dtype=float)
            self._last_field_xyz = np.asarray(ref, dtype=float)
            if showing and uvw is not None:
                overlays.set_mesh_points(self._warp_mesh, pts)
            return n_points
        except FileNotFoundError as exc:
            overlays.set_named_visible(self._plotter, "mesh", True)
            self._showing_deformed = False
            self._show_dic_error(f"Measurement file not found:\n{exc}")
            return 0
        except Exception as exc:
            overlays.set_named_visible(self._plotter, "mesh", True)
            self._showing_deformed = False
            path_text = str(data_path) if data_path is not None else ""
            self._show_dic_error(
                f"Could not read measurement file:\n{path_text}\n{exc}"
            )
            return 0

    def _show_dic_error(self, message: str) -> None:
        text = str(message).strip()
        if self._plotter is not None:
            try:
                overlays.add_plain_text(self._plotter, text, font_size=9, name="dic_error")
            except Exception:
                pass
        self.error_logged.emit(text.replace("\n", " "))

    def _build_mesh(self, document: SessionDocument, experiment: ExperimentModel):
        return mesh_convert.build_experiment_mesh(document, experiment)

    def displayed_point_data(self) -> dict | None:
        if self._last_field_values is None or not self._last_field_name:
            return None
        values = np.asarray(self._last_field_values, dtype=float).ravel()
        return {self._last_field_name: values}

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key.Key_Escape and bool(
            getattr(self._session, "drag_mesh_mode", False)
        ):
            self._cancel_in_progress_drag()
            self._drag_mesh_hover = None
            self.sync_drag_mesh_overlay(self._session)
            event.accept()
            return
        if key == Qt.Key.Key_PageDown:
            self.step_delta.emit(1)
            event.accept()
            return
        if key == Qt.Key.Key_PageUp:
            self.step_delta.emit(-1)
            event.accept()
            return
        super().keyPressEvent(event)

    def _install_pick_filter(self) -> None:
        interactor = getattr(self._plotter, "interactor", None)
        if interactor is None:
            return
        watcher = _ViewportPickFilter(self)
        interactor.installEventFilter(watcher)
        self._pick_filter = watcher

    def enter_free_mesh(self, session: AppSession) -> None:
        self._session = session
        if self._plotter is None:
            return
        self._ensure_xy_lock(fit=not self._xy_locked)

    def leave_free_mesh(self, session: AppSession | None = None) -> None:
        if session is not None:
            self._session = session
        self._drag_index = None
        overlays.remove_free_mesh_overlay(self._plotter)
        self._release_xy_lock_if_idle()

    def enter_drag_mesh(self, session: AppSession) -> None:
        self._session = session
        self._drag_mesh_state = None
        self._drag_mesh_hover = None
        if self._plotter is None:
            return
        self._ensure_xy_lock(fit=not self._xy_locked)

    def leave_drag_mesh(self, session: AppSession | None = None) -> None:
        if session is not None:
            self._session = session
        self._cancel_in_progress_drag()
        self._drag_mesh_hover = None
        overlays.remove_drag_mesh_overlay(self._plotter)
        self._release_xy_lock_if_idle()

    def _xy_tool_active(self, session: AppSession | None = None) -> bool:
        session = session if session is not None else self._session
        if session is None:
            return False
        return bool(
            getattr(session, "free_mesh_mode", False)
            or getattr(session, "drag_mesh_mode", False)
        )

    def _ensure_xy_lock(self, *, fit: bool) -> None:
        if self._plotter is None:
            return
        if not self._xy_locked:
            self._saved_camera = self._capture_camera()
            try:
                self._saved_style = self._plotter.interactor.GetInteractorStyle()
            except Exception:
                self._saved_style = None
        self._lock_xy_view(fit=fit)

    def _release_xy_lock_if_idle(self) -> None:
        if self._xy_tool_active():
            return
        overlays.remove_drag_mesh_overlay(self._plotter)
        overlays.remove_free_mesh_overlay(self._plotter)
        self._restore_3d_view()

    def sync_free_mesh_overlay(self, session: AppSession | None = None) -> None:
        session = session or self._session
        if session is None or self._plotter is None or self._pv is None:
            return
        if not bool(getattr(session, "free_mesh_mode", False)):
            overlays.remove_free_mesh_overlay(self._plotter)
            return
        points = list(session.free_mesh_points or [])
        status = inspect_free_points(points)
        bounds = None
        if self._mesh_pkg is not None:
            try:
                bounds = overlays.specimen_xy_extents(self._mesh_pkg, self._last_dic_xyz)
            except Exception:
                bounds = None
        drag_xyz = None
        drag_text = None
        if self._drag_index is not None and 0 <= self._drag_index < len(points):
            x, y = points[self._drag_index]
            drag_text = f"({x:.4g}, {y:.4g})"
            drag_xyz = self._label_world_beside(x, y)
        overlays.add_free_mesh_overlay(
            self._plotter,
            self._pv,
            points,
            status.quad_xy,
            bounds=bounds,
            drag_index=self._drag_index,
            drag_label_xyz=drag_xyz,
            drag_text=drag_text,
        )
        try:
            self._plotter.render()
        except Exception:
            pass

    def sync_drag_mesh_overlay(self, session: AppSession | None = None) -> None:
        session = session or self._session
        if session is None or self._plotter is None or self._pv is None:
            return
        if not bool(getattr(session, "drag_mesh_mode", False)):
            overlays.remove_drag_mesh_overlay(self._plotter)
            return
        target = None
        if self._drag_mesh_state is not None:
            target = self._drag_mesh_state.get("target")
            axis = self._drag_mesh_state.get("axis")
        else:
            target = self._drag_mesh_hover
            axis = None if target is None else target.axis
        if target is None or self._mesh_pkg is None:
            overlays.remove_drag_mesh_overlay(self._plotter)
            try:
                self._plotter.render()
            except Exception:
                pass
            return
        xyz = mesh_convert.mesh_xyz(self._mesh_pkg)
        quads = mesh_convert.mesh_quads(self._mesh_pkg)
        xs = [float(v) for v in np.asarray(self._mesh_pkg.x_nodes, dtype=float).ravel()]
        ys = [float(v) for v in np.asarray(self._mesh_pkg.y_nodes, dtype=float).ravel()]
        segs = []
        node_pts = []
        axes = []
        if axis in (None, "x"):
            axes.append(("x", int(target.ix)))
        if axis in (None, "y"):
            axes.append(("y", int(target.iy)))
        seen = set()
        for line_axis, index in axes:
            key = (line_axis, index)
            if key in seen:
                continue
            seen.add(key)
            line = grid_line_segments(xyz, quads, xs, ys, line_axis, index)
            if line.size:
                segs.append(line)
            ids = nodes_on_grid_line(xyz, xs, ys, line_axis, index)
            if ids.size:
                node_pts.append(xyz[ids, :2])
        segments = np.concatenate(segs, axis=0) if segs else np.zeros((0, 2, 2))
        nodes = np.concatenate(node_pts, axis=0) if node_pts else np.zeros((0, 2))
        drag_xyz = None
        drag_text = None
        if self._drag_mesh_state is not None and axis in {"x", "y"}:
            label_xy = self._drag_mesh_label_point(xyz, target, axis)
            if label_xy is not None:
                drag_text = f"{axis} = {label_xy[0 if axis == 'x' else 1]:.4g}"
                drag_xyz = self._label_world_beside(label_xy[0], label_xy[1])
        overlays.add_drag_mesh_overlay(
            self._plotter,
            self._pv,
            segments=segments,
            nodes=nodes,
            drag_label_xyz=drag_xyz,
            drag_text=drag_text,
        )
        try:
            self._plotter.render()
        except Exception:
            pass

    def _drag_mesh_label_point(self, xyz, target, axis: str):
        if target.node_id is not None and 0 <= int(target.node_id) < xyz.shape[0]:
            pt = xyz[int(target.node_id)]
            return float(pt[0]), float(pt[1])
        ids = nodes_on_grid_line(
            xyz,
            np.asarray(self._mesh_pkg.x_nodes, dtype=float),
            np.asarray(self._mesh_pkg.y_nodes, dtype=float),
            axis,
            int(target.ix if axis == "x" else target.iy),
        )
        if ids.size:
            pt = xyz[int(ids[0])]
            return float(pt[0]), float(pt[1])
        return None

    def _preview_drag_mesh(self) -> None:
        session = self._session
        if session is None or self._plotter is None or self._pv is None:
            return
        experiment = session.selected_experiment()
        if experiment is None:
            return
        try:
            mesh_pkg = self._build_mesh(session.document, experiment)
        except Exception as exc:
            self.error_logged.emit(f"Drag mesh: {exc}")
            return
        self._mesh_pkg = mesh_pkg
        self._mesh_key = _mesh_key(session)
        overlays.add_mesh_wireframe(self._plotter, self._pv, mesh_pkg)
        overlays.set_named_visible(self._plotter, "mesh", True)
        try:
            overlays.update_bc_glyphs(
                self._plotter, self._pv, experiment, mesh_pkg, session.selection
            )
        except Exception:
            pass
        overlays.remove_element_point_counts(self._plotter)
        self.sync_drag_mesh_overlay(session)

    def _cancel_in_progress_drag(self) -> None:
        state = self._drag_mesh_state
        session = self._session
        self._drag_mesh_state = None
        if state is None or session is None or not state.get("moved"):
            return
        experiment = session.selected_experiment()
        if experiment is None:
            return
        snap = state.get("snapshot") or {}
        restore_mesh_axes(
            experiment.mesh,
            snap.get("x_nodes"),
            snap.get("y_nodes"),
            snap.get("bounds"),
            snap.get("free_nodes"),
            n_elements_x=snap.get("n_elements_x"),
            n_elements_y=snap.get("n_elements_y"),
        )
        session.free_mesh_points = list(state.get("start_free_points") or [])
        self._preview_drag_mesh()

    def handle_drag_mesh_event(self, event) -> bool:
        try:
            return self._handle_drag_mesh_event(event)
        except Exception as exc:
            self.error_logged.emit(f"Drag mesh: {exc}")
            return True

    def _handle_drag_mesh_event(self, event) -> bool:
        if self._plotter is None or self._session is None:
            return False
        etype = event.type()
        if etype == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
            self._cancel_in_progress_drag()
            self._drag_mesh_hover = None
            self.sync_drag_mesh_overlay(self._session)
            return True
        if etype == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.RightButton:
                if self._drag_mesh_state is not None:
                    self._cancel_in_progress_drag()
                    self.sync_drag_mesh_overlay(self._session)
                    return True
                return False
            if event.button() != Qt.MouseButton.LeftButton:
                return False
            display = self._qt_to_vtk_display(event)
            target = self._pick_drag_target(display)
            if target is None:
                self._drag_mesh_hover = None
                self._drag_mesh_state = None
                self.sync_drag_mesh_overlay(self._session)
                return True
            world = self._world_from_event(event)
            if world is None:
                return True
            experiment = self._session.selected_experiment()
            if experiment is None:
                return True
            xs, ys = axis_nodes_from_mesh(experiment.mesh)
            self._drag_mesh_hover = target
            self._drag_mesh_state = {
                "target": target,
                "ix": int(target.ix),
                "iy": int(target.iy),
                "preferred": target.axis,
                "axis": target.axis,
                "start_display": display,
                "start_world": world,
                "snapshot": snapshot_mesh_axes(experiment.mesh),
                "start_free_points": list(self._session.free_mesh_points or []),
                "start_value": {
                    "x": float(xs[min(max(target.ix, 0), len(xs) - 1)]),
                    "y": float(ys[min(max(target.iy, 0), len(ys) - 1)]),
                },
                "moved": False,
            }
            self.sync_drag_mesh_overlay(self._session)
            return True
        if etype == QEvent.Type.MouseMove:
            if self._drag_mesh_state is not None:
                self._apply_drag_mesh_move(event)
                return True
            display = self._qt_to_vtk_display(event)
            hover = self._pick_drag_target(display)
            if hover != self._drag_mesh_hover:
                self._drag_mesh_hover = hover
                self.sync_drag_mesh_overlay(self._session)
            return False
        if etype == QEvent.Type.MouseButtonRelease:
            if self._drag_mesh_state is None:
                return False
            if event.button() != Qt.MouseButton.LeftButton:
                return False
            state = self._drag_mesh_state
            moved = bool(state.get("moved"))
            if moved:
                experiment = self._session.selected_experiment()
                if experiment is not None and experiment.mesh.free_nodes:
                    self._session.free_mesh_points = [
                        (float(p[0]), float(p[1])) for p in experiment.mesh.free_nodes
                    ]
            self._drag_mesh_hover = state.get("target")
            self._drag_mesh_state = None
            self.sync_drag_mesh_overlay(self._session)
            if moved:
                self.drag_mesh_committed.emit()
            return True
        return False

    def _apply_drag_mesh_move(self, event) -> None:
        state = self._drag_mesh_state
        session = self._session
        if state is None or session is None:
            return
        experiment = session.selected_experiment()
        world = self._world_from_event(event)
        display = self._qt_to_vtk_display(event)
        if experiment is None or world is None:
            return
        sx, sy = state["start_display"]
        dx = float(display[0]) - float(sx)
        dy = float(display[1]) - float(sy)
        xs, ys = axis_nodes_from_mesh(experiment.mesh)
        axis = choose_drag_axis(
            int(state["ix"]),
            int(state["iy"]),
            len(xs),
            len(ys),
            dx,
            dy,
            preferred=state.get("preferred"),
        )
        if axis is None:
            return
        state["axis"] = axis
        value = float(world[0] if axis == "x" else world[1])
        index = int(state["ix"] if axis == "x" else state["iy"])
        outer = (axis == "x" and index in {0, len(xs) - 1}) or (
            axis == "y" and index in {0, len(ys) - 1}
        )
        if outer:
            targets = dic_axis_targets(self._last_dic_xyz, axis)
            target_display = []
            for target_val in targets:
                if axis == "x":
                    loc = self._world_to_display(float(target_val), float(world[1]))
                    target_display.append(
                        float(loc[0]) if loc is not None else 1.0e300
                    )
                else:
                    loc = self._world_to_display(float(world[0]), float(target_val))
                    target_display.append(
                        float(loc[1]) if loc is not None else 1.0e300
                    )
            display_value = float(display[0] if axis == "x" else display[1])
            value = snap_axis_value(value, display_value, targets, target_display)
        new_val = apply_grid_line_drag(experiment.mesh, axis, index, value)
        start = float(state["start_value"][axis])
        state["moved"] = abs(new_val - start) > 1.0e-12
        if experiment.mesh.free_nodes:
            session.free_mesh_points = [
                (float(p[0]), float(p[1])) for p in experiment.mesh.free_nodes
            ]
        self._preview_drag_mesh()

    def _pick_drag_target(self, display_xy):
        if self._mesh_pkg is None:
            return None
        xyz = mesh_convert.mesh_xyz(self._mesh_pkg)
        quads = mesh_convert.mesh_quads(self._mesh_pkg)
        xs = [float(v) for v in np.asarray(self._mesh_pkg.x_nodes, dtype=float).ravel()]
        ys = [float(v) for v in np.asarray(self._mesh_pkg.y_nodes, dtype=float).ravel()]
        node_display = []
        for pt in xyz:
            loc = self._world_to_display(float(pt[0]), float(pt[1]))
            if loc is None:
                return None
            node_display.append(loc)
        edges, axes = element_edge_records(xyz, quads, xs, ys)
        edge_display = []
        for seg in edges:
            p0 = self._world_to_display(float(seg[0, 0]), float(seg[0, 1]))
            p1 = self._world_to_display(float(seg[1, 0]), float(seg[1, 1]))
            if p0 is None or p1 is None:
                return None
            edge_display.append([p0, p1])
        edge_arr = (
            np.asarray(edge_display, dtype=float)
            if edge_display
            else np.zeros((0, 2, 2), dtype=float)
        )
        return pick_drag_target(
            display_xy,
            np.asarray(node_display, dtype=float),
            xyz,
            edge_arr,
            axes,
            xs,
            ys,
        )

    def _lock_xy_view(self, *, fit: bool) -> None:
        if self._plotter is None:
            return
        try:
            self._plotter.view_xy()
        except Exception:
            pass
        try:
            self._plotter.enable_parallel_projection()
        except Exception:
            pass
        try:
            import vtk

            style = vtk.vtkInteractorStyleImage()
            self._plotter.interactor.SetInteractorStyle(style)
        except Exception:
            pass
        self._xy_locked = True
        if fit:
            try:
                self._plotter.reset_camera()
            except Exception:
                pass

    def _capture_camera(self):
        if self._plotter is None:
            return None
        try:
            cam = self._plotter.camera
        except Exception:
            return None
        state = {}
        try:
            state["position"] = tuple(cam.position)
            state["focal_point"] = tuple(cam.focal_point)
            up = getattr(cam, "up", None)
            state["view_up"] = tuple(up if up is not None else cam.GetViewUp())
        except Exception:
            return None
        try:
            state["parallel_projection"] = bool(cam.parallel_projection)
        except Exception:
            try:
                state["parallel_projection"] = bool(cam.GetParallelProjection())
            except Exception:
                state["parallel_projection"] = False
        try:
            state["parallel_scale"] = float(cam.parallel_scale)
        except Exception:
            pass
        return state

    def _restore_3d_view(self) -> None:
        if self._plotter is None:
            self._xy_locked = False
            return
        try:
            import vtk

            style = self._saved_style
            if style is None:
                style = vtk.vtkInteractorStyleTrackballCamera()
            self._plotter.interactor.SetInteractorStyle(style)
        except Exception:
            pass
        cam_state = self._saved_camera
        self._saved_camera = None
        self._saved_style = None
        self._xy_locked = False
        if cam_state:
            try:
                cam = self._plotter.camera
                cam.position = cam_state["position"]
                cam.focal_point = cam_state["focal_point"]
                if hasattr(cam, "up"):
                    cam.up = cam_state["view_up"]
                else:
                    cam.SetViewUp(*cam_state["view_up"])
                if cam_state.get("parallel_scale") is not None:
                    try:
                        cam.parallel_scale = cam_state["parallel_scale"]
                    except Exception:
                        pass
                if cam_state.get("parallel_projection"):
                    self._plotter.enable_parallel_projection()
                else:
                    self._plotter.disable_parallel_projection()
                self._plotter.reset_camera_clipping_range()
                return
            except Exception:
                pass
        try:
            self._plotter.disable_parallel_projection()
        except Exception:
            pass
        try:
            self._plotter.view_isometric()
            self._plotter.reset_camera()
        except Exception:
            pass

    def _qt_to_vtk_display(self, event):
        pos = event.position() if hasattr(event, "position") else event.pos()
        x = float(pos.x())
        y = float(pos.y())
        qt_w, qt_h = self._qt_size()
        vtk_w, vtk_h = self._display_size()
        sx = vtk_w / max(qt_w, 1.0)
        sy = vtk_h / max(qt_h, 1.0)
        return x * sx, (qt_h - y - 1.0) * sy

    def _world_from_event(self, event):
        dx, dy = self._qt_to_vtk_display(event)
        return self._world_from_display(dx, dy)

    def _vtk_renderer(self):
        if self._plotter is None:
            return None
        renderer = getattr(self._plotter, "renderer", None)
        if renderer is None:
            return None
        getter = getattr(renderer, "GetVTKRenderer", None)
        if callable(getter):
            try:
                inner = getter()
                if inner is not None:
                    return inner
            except Exception:
                pass
        return renderer

    def _world_from_display(self, display_x: float, display_y: float):
        renderer = self._vtk_renderer()
        if renderer is None:
            return None
        try:
            import vtk

            coord = vtk.vtkCoordinate()
            coord.SetCoordinateSystemToDisplay()
            coord.SetValue(float(display_x), float(display_y), 0.0)
            world = coord.GetComputedWorldValue(renderer)
            return float(world[0]), float(world[1])
        except Exception:
            pass
        try:
            import vtk

            world = [0.0, 0.0, 0.0, 0.0]
            vtk.vtkInteractorObserver.ComputeDisplayToWorld(
                renderer, float(display_x), float(display_y), 0.0, world
            )
            w = float(world[3]) if abs(float(world[3])) > 1.0e-12 else 1.0
            return float(world[0] / w), float(world[1] / w)
        except Exception:
            return None

    def _world_to_display(self, x: float, y: float, z: float = 0.0):
        renderer = self._vtk_renderer()
        if renderer is None:
            return None
        try:
            import vtk

            coord = vtk.vtkCoordinate()
            coord.SetCoordinateSystemToWorld()
            coord.SetValue(float(x), float(y), float(z))
            display = coord.GetComputedDisplayValue(renderer)
            return float(display[0]), float(display[1])
        except Exception:
            pass
        try:
            import vtk

            display = [0.0, 0.0, 0.0]
            vtk.vtkInteractorObserver.ComputeWorldToDisplay(
                renderer, float(x), float(y), float(z), display
            )
            return float(display[0]), float(display[1])
        except Exception:
            return None

    def _qt_size(self):
        interactor = getattr(self._plotter, "interactor", None)
        if interactor is None:
            return 1.0, 1.0
        return float(max(interactor.width(), 1)), float(max(interactor.height(), 1))

    def _display_size(self):
        try:
            window = getattr(self._plotter, "render_window", None) or getattr(
                self._plotter, "ren_win", None
            )
            if window is None:
                renderer = self._vtk_renderer()
                getter = getattr(renderer, "GetRenderWindow", None) if renderer is not None else None
                window = getter() if callable(getter) else None
            if window is not None:
                size = window.GetSize()
                return float(max(size[0], 1)), float(max(size[1], 1))
        except Exception:
            pass
        return self._qt_size()

    def _label_world_beside(self, x: float, y: float):
        display = self._world_to_display(x, y)
        if display is None:
            return (x, y, 0.0)
        width, height = self._display_size()
        lx, ly = offset_label_display(display[0], display[1], width, height)
        world = self._world_from_display(lx, ly)
        if world is None:
            return (x, y, 0.0)
        return (float(world[0]), float(world[1]), 0.0)

    def _hit_free_point(self, display_xy) -> int | None:
        session = self._session
        if session is None:
            return None
        points = list(session.free_mesh_points or [])
        displayed = []
        for x, y in points:
            loc = self._world_to_display(float(x), float(y))
            if loc is None:
                return None
            displayed.append(loc)
        return nearest_point_index(display_xy, displayed, pixel_tol=10.0)

    def _snap_world(self, x: float, y: float, *, skip_index: int | None = None):
        session = self._session
        points = list(session.free_mesh_points or []) if session is not None else []
        xs = []
        ys = []
        for index, (px, py) in enumerate(points):
            if skip_index is not None and index == skip_index:
                continue
            xs.append(float(px))
            ys.append(float(py))
        display = self._world_to_display(x, y)
        if display is None or (not xs and not ys):
            return float(x), float(y)
        x_disp = []
        for value in xs:
            loc = self._world_to_display(value, y)
            if loc is None:
                return float(x), float(y)
            x_disp.append(loc[0])
        y_disp = []
        for value in ys:
            loc = self._world_to_display(x, value)
            if loc is None:
                return float(x), float(y)
            y_disp.append(loc[1])
        snapped_x = (
            snap_coordinate(
                x, xs, display_value=display[0], axis_display=x_disp, pixel_tol=10.0
            )
            if xs
            else float(x)
        )
        snapped_y = (
            snap_coordinate(
                y, ys, display_value=display[1], axis_display=y_disp, pixel_tol=10.0
            )
            if ys
            else float(y)
        )
        return float(snapped_x), float(snapped_y)

    def _commit_free_points(self) -> None:
        session = self._session
        if session is None:
            return
        experiment = session.selected_experiment()
        if experiment is not None:
            experiment.mesh.free_nodes = [
                [float(x), float(y)] for x, y in session.free_mesh_points
            ]
            session.mark_dirty()
        self.sync_free_mesh_overlay(session)
        self.free_mesh_changed.emit()

    def handle_free_mesh_event(self, event) -> bool:
        try:
            return self._handle_free_mesh_event(event)
        except Exception as exc:
            self.error_logged.emit(f"Free mesh: {exc}")
            return True

    def _handle_free_mesh_event(self, event) -> bool:
        if self._plotter is None or self._session is None:
            return False
        etype = event.type()
        if etype == QEvent.Type.MouseButtonPress:
            display = self._qt_to_vtk_display(event)
            if event.button() == Qt.MouseButton.RightButton:
                hit = self._hit_free_point(display)
                if hit is None:
                    return False
                del self._session.free_mesh_points[hit]
                self._drag_index = None
                self._commit_free_points()
                return True
            if event.button() != Qt.MouseButton.LeftButton:
                return False
            world = self._world_from_event(event)
            if world is None:
                return True
            hit = self._hit_free_point(display)
            if hit is not None:
                self._drag_index = hit
                self.sync_free_mesh_overlay(self._session)
                return True
            x, y = self._snap_world(world[0], world[1])
            snapped_display = self._world_to_display(x, y)
            if snapped_display is not None:
                hit = self._hit_free_point(snapped_display)
                if hit is not None:
                    self._drag_index = hit
                    self.sync_free_mesh_overlay(self._session)
                    return True
            self._session.free_mesh_points.append((x, y))
            self._drag_index = len(self._session.free_mesh_points) - 1
            self._commit_free_points()
            return True
        if etype == QEvent.Type.MouseMove and self._drag_index is not None:
            world = self._world_from_event(event)
            if world is None:
                return True
            x, y = self._snap_world(
                world[0], world[1], skip_index=self._drag_index
            )
            points = self._session.free_mesh_points
            if 0 <= self._drag_index < len(points):
                points[self._drag_index] = (x, y)
                self.sync_free_mesh_overlay(self._session)
            return True
        if etype == QEvent.Type.MouseButtonRelease and self._drag_index is not None:
            if event.button() in {Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton}:
                self._drag_index = None
                self._commit_free_points()
                return True
        return False

    def handle_pick(self, display_x: float, display_y: float) -> None:
        session = self._session
        if session is None or self._plotter is None or self._mesh_pkg is None:
            return
        picking_nodes = bool(getattr(session, "pick_nodes", False)) or bool(
            getattr(session, "preview_all_nodes", False)
        )
        hit = self._actor_at(display_x, display_y)
        if hit and not picking_nodes:
            selection = parse_bc_actor_name(hit)
            if selection is not None:
                self.picked.emit(selection)
                return
        point = self._world_xy(display_x, display_y)
        if point is None:
            return
        mesh_xyz = mesh_convert.mesh_xyz(self._mesh_pkg)
        if picking_nodes:
            index = nearest_node_index(point, mesh_xyz)
            self.picked.emit(("node", index))
            return
        edge = nearest_mesh_edge(point, mesh_xyz)
        self.picked.emit(("edge", edge))

    def _actor_at(self, display_x: float, display_y: float) -> str | None:
        try:
            import vtk
        except Exception:
            return None
        renderer = getattr(self._plotter, "renderer", None)
        if renderer is None:
            return None
        picker = vtk.vtkPropPicker()
        if not picker.Pick(float(display_x), float(display_y), 0, renderer):
            return None
        actor = picker.GetActor()
        actors = getattr(self._plotter, "actors", None) or {}
        for name, mapped in actors.items():
            try:
                if mapped is actor or getattr(mapped, "GetAddressAsString", lambda *a: None)() == getattr(
                    actor, "GetAddressAsString", lambda *a: None
                )():
                    return str(name)
            except Exception:
                continue
            try:
                if getattr(mapped, "GetMapper", lambda: None)() is actor:
                    return str(name)
            except Exception:
                continue
        for name, mapped in actors.items():
            try:
                vtk_actor = getattr(mapped, "GetActor", lambda: mapped)()
            except Exception:
                vtk_actor = mapped
            if vtk_actor is actor:
                return str(name)
        return None

    def _world_xy(self, display_x: float, display_y: float):
        try:
            picker = self._plotter.pick_mouse_position()
        except Exception:
            picker = None
        if picker is not None:
            try:
                return float(picker[0]), float(picker[1])
            except Exception:
                pass
        return self._world_from_display(display_x, display_y)


class _ViewportPickFilter(QObject):
    def __init__(self, viewport: ViewportWidget) -> None:
        super().__init__(viewport)
        self._viewport = viewport

    def eventFilter(self, watched, event) -> bool:
        session = self._viewport._session
        if session is not None and bool(getattr(session, "drag_mesh_mode", False)):
            return self._viewport.handle_drag_mesh_event(event)
        if session is not None and bool(getattr(session, "free_mesh_mode", False)):
            return self._viewport.handle_free_mesh_event(event)
        if event.type() != QEvent.Type.MouseButtonPress:
            return False
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        pick = bool(
            session
            and (
                session.pick_mode
                or session.pick_nodes
                or bool(getattr(session, "preview_all_nodes", False))
            )
        )
        ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        if not (pick or ctrl):
            return False
        pos = event.position() if hasattr(event, "position") else event.pos()
        self._viewport.handle_pick(float(pos.x()), float(pos.y()))
        return True


def _node_highlight_spec(session: AppSession):
    preview = tuple(int(item) for item in (getattr(session, "preview_node_ids", ()) or ()))
    show_all = bool(getattr(session, "preview_all_nodes", False))
    if show_all or preview:
        return preview, show_all
    sel = session.selection
    if sel.kind != "bc" or sel.bc_kind is None or sel.bc_index is None:
        return (), False
    experiment = session.document.experiment_by_id(sel.experiment_id or 0)
    if experiment is None:
        return (), False
    try:
        entry = experiment.boundary_entries(sel.bc_kind)[sel.bc_index]
    except (KeyError, IndexError):
        return (), False
    nodes = tuple(int(item) for item in (entry.nodes or ()) if int(item) >= 0)
    return nodes, False


def _virtual_dof_label(mesh_pkg) -> str:
    analysis = str(getattr(mesh_pkg, "analysis_type", "out_of_plane")).lower()
    if analysis in {"in_plane", "inplane"}:
        return "U"
    return "W"


def _step_data_path(session: AppSession) -> Path | None:
    step = session.selected_step()
    if step is None:
        return None
    return resolve_step_data_file(step.data_file, session.document.source_path)


def _existing_data_file(session: AppSession):
    path = _step_data_path(session)
    return str(path) if path is not None else None


def _mesh_key(session: AppSession):
    experiment = session.selected_experiment()
    if experiment is None:
        return ("none",)
    bounds = experiment.mesh.bounds
    return (
        experiment.experiment_id,
        experiment.mesh.n_elements_x,
        experiment.mesh.n_elements_y,
        round(float(experiment.material_orientation_deg), 3),
        round(bounds.x_min, 6),
        round(bounds.x_max, 6),
        round(bounds.y_min, 6),
        round(bounds.y_max, 6),
        tuple(round(float(v), 6) for v in (experiment.mesh.x_nodes or [])),
        tuple(round(float(v), 6) for v in (experiment.mesh.y_nodes or [])),
        tuple((int(cell[0]), int(cell[1])) for cell in (experiment.mesh.keep_cells or [])),
        bool(experiment.fit_data_to_system_axes),
        _existing_data_file(session),
        tuple(
            (kind, entry.dof, entry.edge, entry.load_type, tuple(entry.nodes or ()))
            for kind, entries in (
                ("prescribed", experiment.prescribed),
                ("linking", experiment.linking),
                ("external", experiment.external),
            )
            for entry in entries
        ),
    )


def _camera_key(session: AppSession):
    experiment = session.selected_experiment()
    if experiment is None:
        return ("none",)
    bounds = experiment.mesh.bounds
    return (
        experiment.experiment_id,
        experiment.mesh.n_elements_x,
        experiment.mesh.n_elements_y,
        round(bounds.x_min, 6),
        round(bounds.x_max, 6),
        round(bounds.y_min, 6),
        round(bounds.y_max, 6),
        bool(experiment.show_deformed_configuration),
    )


def _bc_select_key(session: AppSession):
    sel = session.selection
    if sel.kind == "bc":
        return (sel.experiment_id, sel.bc_kind, sel.bc_index)
    return None


def _scene_key(session: AppSession):
    experiment = session.selected_experiment()
    return (
        _mesh_key(session),
        _existing_data_file(session),
        session.current_scalar,
        session.display_mode,
        getattr(session, "scalar_clim", None),
        bool(session.show_xy_axes),
        bool(session.show_element_point_counts),
        bool(getattr(session, "free_mesh_mode", False)),
        bool(getattr(session, "drag_mesh_mode", False)),
        round(float(getattr(experiment, "deformation_scale", 1.0) or 1.0), 4),
        session.selection.step_index,
        None if experiment is None else experiment.reference_step,
        _bc_select_key(session),
        bool(getattr(experiment, "show_deformed_configuration", False))
        if experiment is not None
        else False,
    )
