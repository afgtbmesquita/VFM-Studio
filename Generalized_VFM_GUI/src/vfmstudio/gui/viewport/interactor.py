"""PyVista / VTK viewport embedded in Qt."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QGridLayout, QLabel, QStackedWidget, QWidget

from vfmstudio.core import bridge, mesh_convert
from vfmstudio.core.document import ExperimentModel, SessionDocument
from vfmstudio.core.fields import experiment_alignment_angle, load_step_fields
from vfmstudio.core.step_series import resolve_step_data_file
from vfmstudio.gui.session import AppSession
from vfmstudio.gui.viewport import overlays
from vfmstudio.gui.viewport.load_plot import LoadPlotOverlay


def viewport_enabled() -> bool:
    if os.environ.get("VFMSTUDIO_NO_VISTA", "").strip().lower() in {"1", "true", "yes"}:
        return False
    if os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen":
        return False
    if "PYTEST_CURRENT_TEST" in os.environ:
        return False
    return True


class ViewportWidget(QWidget):
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
        self._session: AppSession | None = None
        self._load_plot_visible = False
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._stack = QStackedWidget()
        layout.addWidget(self._stack, 0, 0)

        self._vista_page = QWidget()
        vista_layout = QGridLayout(self._vista_page)
        vista_layout.setContentsMargins(0, 0, 0, 0)
        vista_layout.setSpacing(0)
        self._stack.addWidget(self._vista_page)

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
            vista_layout.addWidget(self._plotter.interactor, 0, 0)
            self._bc_key = None
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
        self._stack.setCurrentWidget(self._vista_page)
        if self._session is not None:
            self.refresh(self._session, force=True)

    def load_plot_visible(self) -> bool:
        return self._load_plot_visible

    def refresh(self, session: AppSession, force: bool = False) -> None:
        self._session = session
        if self._load_plot_visible:
            self._load_plot.set_session(session)
            return
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
        if experiment is None:
            self._plotter.clear()
            self._plotter.add_text("No experiment", font_size=8)
            session.status_n_points = 0
            self._scene_key = key
            self._mesh_key = mesh_key
            self._bc_key = bc_key
            self._mesh_pkg = None
            self._last_dic_xyz = None
            self._xy_axes_on = False
            self._element_counts_on = False
            return
        try:
            if rebuild_mesh:
                mesh_pkg = self._build_mesh(session.document, experiment)
                self._plotter.clear()
                overlays.add_mesh_wireframe(self._plotter, self._pv, mesh_pkg)
                overlays.add_global_axes(self._plotter, self._pv, mesh_pkg)
                overlays.add_bc_glyphs(
                    self._plotter, self._pv, experiment, mesh_pkg, session.selection
                )
                overlays.add_material_axes(self._plotter, experiment, mesh_pkg)
                self._mesh_pkg = mesh_pkg
                self._mesh_key = mesh_key
                self._bc_key = bc_key
                try:
                    self._plotter.render()
                except Exception:
                    pass
            else:
                overlays.remove_named(self._plotter, "dic", "contour", "cid")
                if bc_key != self._bc_key and self._mesh_pkg is not None:
                    overlays.update_bc_glyphs(
                        self._plotter, self._pv, experiment, self._mesh_pkg, session.selection
                    )
                    self._bc_key = bc_key
                    try:
                        self._plotter.render()
                    except Exception:
                        pass
        except Exception as exc:
            self._plotter.add_text(f"Mesh error:\n{exc}", font_size=9)
            return

        n_points = self._update_dic_cloud(session, experiment)
        if self._mesh_pkg is not None and self._pv is not None:
            overlays.add_mesh_wireframe(self._plotter, self._pv, self._mesh_pkg)
        axes_were_on = self._xy_axes_on
        self._sync_xy_axes(session)
        self._sync_element_point_counts(session)
        try:
            self._plotter.render()
        except Exception:
            pass
        if reset_camera or rebuild_mesh or (session.show_xy_axes and not axes_were_on):
            self._plotter.reset_camera()
            self._camera_key = camera_key
        session.status_n_points = n_points
        self._scene_key = key

    def show_virtual_field(self, session: AppSession, parameter_index: int = 0) -> str | None:
        """Warp the mesh by the stored optimal virtual field. Returns an error string or None."""
        if session.result is None:
            return "Run Identify first. Optimal virtual fields are computed with FGLS."
        fields = getattr(session.result, "optimal_field_results", None) or []
        if not fields:
            return "No virtual fields on this result. Re-run Identify."
        if self._plotter is None or self._pv is None:
            return "3D viewport is not available in this session."
        field = fields[min(max(parameter_index, 0), len(fields) - 1)]
        u = np.asarray(getattr(field, "nodal_variables_k"), dtype=float).ravel()
        experiment, mesh_pkg = self._mesh_for_virtual_field(session, u)
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
        if not session.show_xy_axes or self._mesh_pkg is None or self._pv is None:
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

    def _update_dic_cloud(self, session: AppSession, experiment: ExperimentModel) -> int:
        overlays.remove_named(self._plotter, "dic", "contour", "cid")
        self._last_dic_xyz = None
        scalar_name = session.current_scalar
        if scalar_name.lower() == "cid":
            overlays.set_named_visible(self._plotter, "mesh", True)
            return overlays.add_cid_cloud(self._plotter, self._pv, session.identifiability)
        data_path = _step_data_path(session)
        if data_path is None:
            overlays.set_named_visible(self._plotter, "mesh", True)
            return 0
        try:
            cache_key = (
                str(data_path),
                session.document.material.analysis_type,
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
            pts = mesh_convert.points_xyz(dataset["x"].values, dataset["y"].values)
            self._last_dic_xyz = pts
            n_points = pts.shape[0]
            warp = float(session.warp_scale)
            if "w" in dataset:
                pts[:, 2] = np.asarray(dataset["w"].values, dtype=float) * warp
            overlays.set_named_visible(self._plotter, "mesh", True)
            available = [str(name) for name in dataset.data_vars]
            session.available_scalars = tuple(available)
            if session.display_mode == "points":
                overlays.add_dic_points(
                    self._plotter, self._pv, pts, color="#B8B8B8"
                )
                return n_points
            if scalar_name not in dataset:
                overlays.add_dic_points(
                    self._plotter, self._pv, pts, color="#B8B8B8"
                )
                return n_points
            overlays.add_field_contour(
                self._plotter,
                self._pv,
                pts,
                dataset[scalar_name].values,
                scalar_name,
            )
            return n_points
        except Exception:
            overlays.set_named_visible(self._plotter, "mesh", True)
            return 0

    def _build_mesh(self, document: SessionDocument, experiment: ExperimentModel):
        return mesh_convert.build_experiment_mesh(document, experiment)


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
        bool(experiment.fit_data_to_system_axes),
        len(experiment.prescribed),
        len(experiment.linking),
        len(experiment.external),
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
    )


def _bc_select_key(session: AppSession):
    sel = session.selection
    if sel.kind == "bc":
        return (sel.experiment_id, sel.bc_kind, sel.bc_index)
    return None


def _scene_key(session: AppSession):
    return (
        _mesh_key(session),
        _existing_data_file(session),
        session.current_scalar,
        session.display_mode,
        bool(session.show_xy_axes),
        bool(session.show_element_point_counts),
        round(float(session.warp_scale), 4),
        _bc_select_key(session),
    )
