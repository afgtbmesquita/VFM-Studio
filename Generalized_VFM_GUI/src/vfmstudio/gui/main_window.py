"""VFM Studio main window."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, QEventLoop, Qt, QSize, QThreadPool, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QColor, QKeySequence, QPalette, QShortcut
from PySide6.QtWidgets import (
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QSplitter,
    QSizePolicy,
    QToolBar,
    QToolButton,
)

from vfmstudio import __version__
from vfmstudio.branding import (
    APP_AUTHOR,
    APP_COPYRIGHT_LINE,
    APP_LICENSE_LINE,
    APP_NAME,
    APP_TRADEMARK_LINE,
)
from vfmstudio.core.document import ExperimentModel, MeshModel
from vfmstudio.core.fields import sync_mesh_bounds_from_file
from vfmstudio.core.jobs import validate_document
from vfmstudio.core.mesh_convert import build_experiment_mesh, mesh_quads, mesh_xyz
from vfmstudio.core.parameter_space import selected_experiment_ids
from vfmstudio.core.paths import campaign_project_path, samples_ab_all_project_path
from vfmstudio.gui.actions import ActionRegistry
from vfmstudio.gui.dialogs import (
    AddBcDialog,
    AddLoadStepsDialog,
    ExperimentEditorDialog,
    ImportMatDialog,
    NewProjectDialog,
    SolverSettingsDialog,
    WorkflowDialog,
)
from vfmstudio.gui.docks.job_monitor import JobMonitorDock
from vfmstudio.gui.docks.message_log import MessageLogDock
from vfmstudio.gui.docks.model_tree import ModelTreeDock
from vfmstudio.gui.docks.property_editor import PropertyEditorDock
from vfmstudio.gui.module_bar import ModuleBar
from vfmstudio.gui.post_processing import PostProcessingWorkspace
from vfmstudio.gui.prefs import (
    load_geometry,
    load_window_state,
    make_settings,
    recent_files,
    remember_file,
    save_window,
    ui_persist_enabled,
)
from vfmstudio.gui.results.plot_export import (
    FILE_FILTER,
    default_export_stem,
    resolve_export_path,
    save_figure,
)
from vfmstudio.gui.session import AppSession, Selection, is_post_kind
from vfmstudio.gui.undo import DocumentHistory, document_from_dump
from vfmstudio.gui.viewport.interactor import ViewportWidget
from vfmstudio.gui.viewport.picking import first_bc_on_edge
from vfmstudio.gui.widgets import DisplayModeCombo, FieldCombo
from vfmstudio.gui.workers import JobProcess
from vfmstudio.io import load_project, save_any
from vfmstudio.io.hdf5_project import (
    is_scratch_project_path,
    normalize_project_save_path,
)
from vfmstudio.io.export_vtk import export_mesh_vtk
from vfmstudio.resources import app_icon, icon


def _toolbar_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("toolLabel")
    toolbar = QColor("#D4D4D4")
    pal = label.palette()
    pal.setColor(QPalette.ColorRole.Window, toolbar)
    pal.setColor(QPalette.ColorRole.WindowText, QColor("#222222"))
    pal.setColor(QPalette.ColorRole.Light, toolbar)
    pal.setColor(QPalette.ColorRole.Midlight, toolbar)
    label.setPalette(pal)
    label.setAutoFillBackground(True)
    return label


def _mark_toolbar_buttons(bar: QToolBar) -> None:
    """Keep toolbar actions looking like buttons, not auto-raising text."""
    for child in bar.findChildren(QToolButton):
        child.setAutoRaise(False)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.session = AppSession()
        self.pool = QThreadPool.globalInstance()
        self._jobs = []
        self._job_busy = False
        self._job_token = 0
        self._job_process = None
        self._active_bc_dialog = None
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(app_icon())
        self.resize(1400, 900)
        self.actions = ActionRegistry(self)

        self.viewport = ViewportWidget(self)
        self.post = PostProcessingWorkspace(self)
        self._splitter = QSplitter(Qt.Orientation.Vertical)
        self._splitter.setObjectName("workspaceSplitter")
        self._splitter.setChildrenCollapsible(True)
        self._splitter.setOpaqueResize(False)
        self._splitter.addWidget(self.viewport)
        self._splitter.addWidget(self.post)
        self._splitter.setStretchFactor(0, 2)
        self._splitter.setStretchFactor(1, 3)
        ignored = QSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.viewport.setSizePolicy(ignored)
        self.post.setSizePolicy(ignored)
        self.post.setMinimumSize(0, 0)
        self._results_sizes = [400, 600]
        self._central = self._splitter
        self.setCentralWidget(self._splitter)
        self.post.hide()
        handle = self._splitter.handle(1)
        if handle is not None:
            handle.installEventFilter(self)

        self.tree = ModelTreeDock()
        self.props = PropertyEditorDock()
        self.log = MessageLogDock()
        self.jobs = JobMonitorDock()
        self._history = DocumentHistory(self._restore_document)
        self._history.capture(self.session.document)

        self.dock_model = self._add_dock("Model", self.tree, Qt.DockWidgetArea.LeftDockWidgetArea)
        self.dock_props = self._add_dock(
            "Properties", self.props, Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.dock_log = self._add_dock("Log", self.log, Qt.DockWidgetArea.BottomDockWidgetArea)
        self.dock_jobs = self._add_dock("Jobs", self.jobs, Qt.DockWidgetArea.BottomDockWidgetArea)
        self.tabifyDockWidget(self.dock_log, self.dock_jobs)
        self.dock_log.raise_()
        self.dock_model.setMinimumWidth(280)
        self.dock_model.setMaximumWidth(420)
        self.dock_props.setMinimumWidth(260)
        self.dock_props.setMaximumWidth(420)
        self.resizeDocks([self.dock_model], [320], Qt.Orientation.Horizontal)
        self.setDockNestingEnabled(True)
        self.resizeDocks([self.dock_log], [140], Qt.Orientation.Vertical)

        self.tree.selection_changed.connect(self._on_tree_selection)
        self.props.document_changed.connect(self._on_document_edited)
        self.props.view_changed.connect(self._on_view_option_changed)
        self.props.deformation_scale_changed.connect(self._on_deformation_scale)
        self.props.request_add_bc.connect(self._add_bc)
        self.tree.request_add_bc.connect(self._add_bc)
        self.tree.request_delete_bc.connect(self._delete_bc)
        self.props.request_delete_bc.connect(self._delete_bc)
        self.props.request_add_step.connect(self._add_step)
        self.props.request_visualize_steps.connect(self._show_step_load_plot)
        self.props.request_uncertainty.connect(self.run_uncertainty)
        self.props.request_loo_refit.connect(self.run_loo_refit)
        self.props.request_export_plot.connect(self.export_post_plot)
        self.tree.request_add_experiment.connect(self.add_experiment)
        self.tree.request_add_step.connect(self._add_step)
        self.tree.request_delete_experiment.connect(self._delete_experiment_id)
        self.tree.request_duplicate_experiment.connect(self.duplicate_experiment)
        self.tree.request_delete_step.connect(self._delete_step)
        self.tree.request_move_step.connect(self._move_step)
        self.viewport.error_logged.connect(self.log.log)
        self.viewport.request_add_experiment.connect(self.add_experiment)
        self.viewport.request_open_project.connect(self.open_project)
        self.viewport.picked.connect(self._on_viewport_pick)
        self.viewport.step_delta.connect(self._step_delta)
        self.viewport.free_mesh_changed.connect(self.props._refresh_free_mesh_hint)
        self.viewport.drag_mesh_committed.connect(self._on_drag_mesh_committed)
        self.jobs.cancel_requested.connect(self.cancel_job)
        self.props.request_pick_nodes.connect(self._on_pick_nodes)
        self.props.request_clear_bc_nodes.connect(self._clear_bc_nodes)
        self.props.request_free_mesh_mode.connect(self._on_free_mesh_mode)
        self.props.request_drag_mesh_mode.connect(self._on_drag_mesh_mode)
        self.props.request_clear_free_mesh.connect(self._on_clear_free_mesh)
        self.props.request_clear_mesh.connect(self._on_clear_mesh)

        self._build_menus()
        self._build_toolbars()
        self.statusBar().showMessage("Ready")
        self._restore_ui()
        self._refresh_all()
        self._sync_bottom_docks()
        QTimer.singleShot(0, self._rebuild_recent_menu)

    def nativeEvent(self, eventType, message):
        result = super().nativeEvent(eventType, message)
        try:
            import sys

            if sys.platform != "win32":
                return result
            kind = bytes(eventType).decode("latin1", errors="ignore")
            if "windows" not in kind:
                return result
            import ctypes
            from ctypes import wintypes

            msg = wintypes.MSG.from_address(int(message))
            if msg.message == 0x0231:  # WM_ENTERSIZEMOVE
                self.viewport.pause_live_render()
            elif msg.message == 0x0232:  # WM_EXITSIZEMOVE
                self.viewport.resume_live_render()
        except Exception:
            pass
        return result

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.MouseButtonDblClick:
            handle = self._splitter.handle(1)
            if watched is handle:
                self.actions.get("plots_fill").toggle()
                return True
        return super().eventFilter(watched, event)

    def _add_dock(self, title: str, widget, area) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(title)
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def _exec_dialog(self, dialog, name: str) -> int:
        mins = {
            "NewProject": (460, 220),
            "SolverSettings": (480, 280),
            "AddStep": (560, 320),
            "AddBc:prescribed": (520, 420),
            "AddBc:linking": (520, 420),
            "AddBc:external": (520, 420),
            "ExperimentEditor": (420, 180),
        }
        min_w, min_h = mins.get(name, (460, 220))
        if name.startswith("AddBc:"):
            min_w, min_h = 520, 460
            dialog.setMinimumSize(min_w, min_h)
            dialog.adjustSize()
            hint = dialog.sizeHint()
            dialog.resize(max(min_w, hint.width() + 40), max(min_h, hint.height() + 24))
            return self._exec_add_bc_dialog(dialog)
        dialog.setMinimumSize(min_w, min_h)
        dialog.adjustSize()
        hint = dialog.sizeHint()
        dialog.resize(max(min_w, hint.width() + 40), max(min_h, hint.height() + 24))
        return dialog.exec()

    def _place_add_bc_dialog(self, dialog) -> None:
        geo = self.geometry()
        dialog.move(geo.x() + 16, geo.y() + 88)

    def _exec_add_bc_dialog(self, dialog) -> int:
        """Keep the viewport interactive so node picks and highlights can update."""
        dialog.setWindowModality(Qt.WindowModality.NonModal)
        flags = dialog.windowFlags()
        dialog.setWindowFlags(
            flags | Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint
        )
        self._place_add_bc_dialog(dialog)
        loop = QEventLoop(self)
        dialog.finished.connect(loop.quit)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        loop.exec()
        try:
            dialog.finished.disconnect(loop.quit)
        except (RuntimeError, TypeError):
            pass
        return int(dialog.result())

    def _restore_document(self, payload: dict) -> None:
        self.session.document = document_from_dump(payload)
        self.session.mark_dirty()
        self._history.capture(self.session.document)
        self._refresh_all()

    def _commit_document_edit(self, label: str, coalesce_key: str | None = None) -> None:
        if self._history.commit(self.session.document, label, coalesce_key):
            self.session.mark_dirty()

    def _reset_history(self) -> None:
        self._history.reset(self.session.document)

    def set_module(self, module: str) -> None:
        name = str(module or "model")
        if name not in {"model", "mesh", "bcs", "job", "results"}:
            name = "model"
        if self.session.module == name:
            self._apply_module_layout()
            self._rebuild_context_toolbar()
            return
        self.session.module = name
        self.module_bar.set_current(name)
        self._focus_tree_for_module(name)
        self._apply_module_layout()
        self._rebuild_context_toolbar()
        self._apply_selection()
        self._sync_bottom_docks()

    def _focus_tree_for_module(self, module: str) -> None:
        selection = self.session.selection
        experiment = self.session.targeted_experiment() or self.session.selected_experiment()
        exp_id = experiment.experiment_id if experiment is not None else None
        if module == "mesh":
            if exp_id is not None:
                self.session.selection = Selection(kind="mesh", experiment_id=exp_id)
        elif module == "bcs":
            if exp_id is not None:
                self.session.selection = Selection(kind="bcs", experiment_id=exp_id)
        elif module == "job":
            self.session.selection = Selection(kind="solver")
        elif module == "results":
            if not is_post_kind(selection.kind):
                self.session.selection = Selection(
                    kind=self.session.last_post_kind or "post_plots"
                )
        elif module == "model":
            if is_post_kind(selection.kind):
                if exp_id is not None:
                    self.session.selection = Selection(
                        kind="experiment", experiment_id=exp_id
                    )
                else:
                    self.session.selection = Selection(kind="project")
        self.tree.select(self.session.selection)

    def _apply_module_layout(self) -> None:
        results = self.session.module == "results"
        if results:
            self.post.show()
            if self.session.plots_fill_central:
                sizes = self._splitter.sizes()
                if self.viewport.isVisible() and len(sizes) > 1 and sizes[0] > 40:
                    self._results_sizes = sizes
                self.viewport.hide()
            else:
                self.viewport.show()
                sizes = self._splitter.sizes()
                if len(sizes) > 1 and sizes[1] < 40:
                    self._splitter.setSizes(self._results_sizes)
        else:
            sizes = self._splitter.sizes()
            if self.post.isVisible() and len(sizes) > 1 and sizes[1] > 40:
                self._results_sizes = sizes
            self.post.hide()
            self.viewport.show()
        self._sync_bottom_docks()

    def _sync_bottom_docks(self) -> None:
        """Keep Log/Jobs out of the plot area unless a job is running."""
        busy = bool(self._job_busy)
        results = self.session.module == "results"
        job_module = self.session.module == "job"
        if busy or job_module:
            self.dock_jobs.show()
            self.dock_jobs.raise_()
            if not results:
                self.dock_log.show()
            return
        self.dock_jobs.hide()
        if results:
            self.dock_log.hide()
            return
        self.dock_log.show()

    def _on_plots_fill_toggled(self, checked: bool) -> None:
        self.session.plots_fill_central = bool(checked)
        self._apply_module_layout()

    def _carry_view_prefs(self, session: AppSession) -> None:
        try:
            session.plots_fill_central = self.actions.get("plots_fill").isChecked()
        except KeyError:
            pass

    def _rebuild_context_toolbar(self) -> None:
        bar = self._context
        bar.clear()
        module = self.session.module
        if module == "model":
            bar.addAction(self.actions.get("add_exp"))
        elif module == "mesh":
            bar.addAction("Previous step", lambda: self._step_delta(-1))
            bar.addAction("Next step", lambda: self._step_delta(1))
            bar.addSeparator()
            bar.addAction(self._pick_action)
        elif module == "bcs":
            bar.addAction(self.actions.get("add_prescribed"))
            bar.addAction(self.actions.get("add_linking"))
            bar.addAction(self.actions.get("add_external"))
            bar.addSeparator()
            bar.addAction(self._pick_action)
        elif module == "job":
            bar.addAction(self.actions.get("run"))
            bar.addAction(self.actions.get("cancel_job"))
        elif module == "results":
            bar.addAction(self.actions.get("export_plot"))
            bar.addAction(self.actions.get("plots_fill"))
            bar.addAction(self.actions.get("show_vf"))
        _mark_toolbar_buttons(bar)

    def _on_clim_edited(self) -> None:
        low = float(self._clim_min.value())
        high = float(self._clim_max.value())
        if high < low:
            low, high = high, low
        self.session.scalar_clim = (low, high)
        self.viewport.refresh(self.session, force=True)

    def _reset_clim(self) -> None:
        self.session.scalar_clim = None
        self.viewport.refresh(self.session, force=True)

    def _on_pick_mode(self, checked: bool) -> None:
        self.session.pick_mode = bool(checked)

    def _on_pick_nodes(self, checked: bool) -> None:
        self.session.pick_nodes = bool(checked)
        if checked:
            self.session.pick_mode = True
            self._pick_action.setChecked(True)

    def _on_bc_node_preview(self, node_ids, show_all: bool = False) -> None:
        ids = tuple(int(item) for item in (node_ids or []) if int(item) >= 0)
        self.session.preview_node_ids = ids
        self.session.preview_all_nodes = bool(show_all)
        self.viewport.set_node_highlights(ids, show_all=bool(show_all))

    def _selection_allows_free_mesh(self) -> bool:
        return (
            self.session.module == "mesh"
            and self.session.selection.kind == "mesh"
        )

    def _on_free_mesh_mode(self, enabled: bool) -> None:
        enabled = bool(enabled)
        experiment = self.session.selected_experiment()
        if enabled and experiment is None:
            self.props._sync_free_mesh_controls(self.session)
            return
        if enabled == bool(self.session.free_mesh_mode) and not (
            enabled and self.session.drag_mesh_mode
        ):
            return
        if enabled:
            if self.session.module != "mesh" and experiment is not None:
                self.session.module = "mesh"
                self.module_bar.set_current("mesh")
                self.session.selection = Selection(
                    kind="mesh", experiment_id=experiment.experiment_id
                )
            if self.session.drag_mesh_mode:
                self.session.free_mesh_mode = True
                self.session.drag_mesh_mode = False
                self.viewport.leave_drag_mesh(self.session)
            if not self.session.free_mesh_points and experiment is not None:
                nodes = experiment.mesh.free_nodes or []
                self.session.free_mesh_points = [
                    (float(item[0]), float(item[1])) for item in nodes if item is not None
                ]
            self.session.free_mesh_mode = True
            self.viewport.enter_free_mesh(self.session)
        else:
            self.session.free_mesh_mode = False
            self.viewport.leave_free_mesh(self.session)
        self.props._sync_free_mesh_controls(self.session)
        self.viewport.refresh(self.session, force=True)

    def _on_drag_mesh_mode(self, enabled: bool) -> None:
        enabled = bool(enabled)
        experiment = self.session.selected_experiment()
        if enabled and experiment is None:
            self.session.drag_mesh_mode = False
            self.props._sync_free_mesh_controls(self.session)
            return
        if enabled == bool(self.session.drag_mesh_mode) and not (
            enabled and self.session.free_mesh_mode
        ):
            return
        if enabled:
            if self.session.module != "mesh" and experiment is not None:
                self.session.module = "mesh"
                self.module_bar.set_current("mesh")
                self.session.selection = Selection(
                    kind="mesh", experiment_id=experiment.experiment_id
                )
            if self.session.free_mesh_mode:
                self.session.drag_mesh_mode = True
                self.session.free_mesh_mode = False
                self.viewport.leave_free_mesh(self.session)
            self.session.drag_mesh_mode = True
            self.viewport.enter_drag_mesh(self.session)
        else:
            self.session.drag_mesh_mode = False
            self.viewport.leave_drag_mesh(self.session)
        self.props._sync_free_mesh_controls(self.session)
        self.viewport.refresh(self.session, force=True)

    def _on_drag_mesh_committed(self) -> None:
        self._commit_document_edit("Drag mesh")
        experiment = self.session.selected_experiment()
        if experiment is not None:
            self.props._fill_mesh(self.session, experiment.experiment_id)
        self.viewport.refresh(self.session, force=True)
        self._update_title()
        self._update_status()

    def _on_clear_free_mesh(self) -> None:
        self.session.free_mesh_points = []
        experiment = self.session.selected_experiment()
        if experiment is not None:
            experiment.mesh.free_nodes = None
            self.session.mark_dirty()
        self.viewport.sync_free_mesh_overlay(self.session)
        self.props._refresh_free_mesh_hint()

    def _on_clear_mesh(self) -> None:
        experiment = self.session.targeted_experiment() or self.session.selected_experiment()
        if experiment is None:
            return
        n_bc = (
            len(experiment.prescribed)
            + len(experiment.linking)
            + len(experiment.external)
        )
        extra = ""
        if self.session.result is not None:
            extra = (
                "\n\nExisting identification results were computed on the "
                "current mesh and will no longer match. Re-run Identify after "
                "you rebuild the mesh and BCs."
            )
        if not self._confirm_delete(
            "Clear mesh",
            "This will delete this experiment's mesh and everything that "
            "depends on it:\n"
            f"- the current mesh (including free-mesh nodes)\n"
            f"- all prescribed, linking, and external boundary conditions"
            f" ({n_bc} total)\n"
            "- per-step external loads attached to those BCs"
            f"{extra}\n\n"
            "This cannot be undone except with Undo. Continue?",
        ):
            return
        if self.session.free_mesh_mode:
            self.session.free_mesh_mode = False
            self.session.free_mesh_points = []
            self.viewport.leave_free_mesh(self.session)
        if self.session.drag_mesh_mode:
            self.session.drag_mesh_mode = False
            self.viewport.leave_drag_mesh(self.session)
        experiment.clear_mesh_and_dependencies()
        self.session.free_mesh_points = []
        selection = self.session.selection
        if (
            selection.experiment_id == experiment.experiment_id
            and selection.kind in {"bc", "bc_group", "bcs"}
        ):
            self.session.selection = Selection(
                kind="mesh", experiment_id=experiment.experiment_id
            )
        self._commit_document_edit("Clear mesh")
        self.log.log(
            f"Cleared mesh and boundary conditions for "
            f"{experiment.name or f'experiment {experiment.experiment_id}'}."
        )
        self._refresh_all()

    def _clear_bc_nodes(self) -> None:
        selection = self.session.selection
        if selection.kind != "bc":
            return
        experiment = self.session.document.experiment_by_id(selection.experiment_id or 0)
        if experiment is None or selection.bc_kind is None or selection.bc_index is None:
            return
        try:
            entry = experiment.boundary_entries(selection.bc_kind)[selection.bc_index]
        except (KeyError, IndexError):
            return
        entry.nodes = None
        self._commit_document_edit("Clear BC nodes")
        self._refresh_all()

    def _on_viewport_pick(self, payload) -> None:
        if isinstance(payload, Selection):
            self.session.selection = payload
            self.tree.select(payload)
            self._apply_selection()
            return
        if not isinstance(payload, tuple) or len(payload) != 2:
            return
        kind, value = payload
        if kind == "node":
            dialog = self._active_bc_dialog
            if dialog is not None and hasattr(dialog, "toggle_node"):
                dialog.toggle_node(int(value))
                return
            self._assign_picked_node(int(value))
            return
        if kind == "edge":
            experiment = self.session.selected_experiment()
            if experiment is None:
                return
            hit = first_bc_on_edge(experiment, str(value))
            if hit is None:
                self.session.selection = Selection(
                    kind="mesh", experiment_id=experiment.experiment_id
                )
            else:
                group, index = hit
                self.session.selection = Selection(
                    kind="bc",
                    experiment_id=experiment.experiment_id,
                    bc_kind=group,
                    bc_index=index,
                )
            self.tree.select(self.session.selection)
            self._apply_selection()

    def _assign_picked_node(self, node_index: int) -> None:
        selection = self.session.selection
        if selection.kind != "bc" or selection.bc_kind is None or selection.bc_index is None:
            return
        experiment = self.session.document.experiment_by_id(selection.experiment_id or 0)
        if experiment is None:
            return
        try:
            entry = experiment.boundary_entries(selection.bc_kind)[selection.bc_index]
        except (KeyError, IndexError):
            return
        nodes = list(entry.nodes or [])
        if node_index not in nodes:
            nodes.append(node_index)
        entry.nodes = nodes
        self._commit_document_edit("Pick BC node", coalesce_key="pick-node")
        self._apply_selection()

    def _step_delta(self, delta: int) -> None:
        experiment = self.session.selected_experiment()
        if experiment is None or not experiment.steps:
            return
        current = self.session.selection.step_index
        if current is None:
            current = 0
        index = max(0, min(len(experiment.steps) - 1, int(current) + int(delta)))
        self.session.selection = Selection(
            kind="step",
            experiment_id=experiment.experiment_id,
            step_index=index,
        )
        self.tree.select(self.session.selection)
        self._apply_selection()

    def _show_workflow(self) -> None:
        dialog = WorkflowDialog(self)
        self._exec_dialog(dialog, "Workflow")

    def _rebuild_recent_menu(self) -> None:
        menu = getattr(self, "_recent_menu", None)
        if menu is None:
            return
        menu.clear()
        paths = recent_files()
        if not paths:
            empty = menu.addAction("(none)")
            empty.setEnabled(False)
            return
        for path in paths:
            action = menu.addAction(path.name)
            action.setData(str(path))
            action.triggered.connect(lambda checked=False, p=path: self._open_recent(p))

    def _open_recent(self, path: Path) -> None:
        if not self._maybe_save():
            return
        self.open_project_path(Path(path))

    def _remember_path(self, path: Path) -> None:
        remember_file(Path(path))
        self._rebuild_recent_menu()

    def _restore_ui(self) -> None:
        if not ui_persist_enabled():
            return
        geometry = load_geometry()
        state = load_window_state()
        if geometry is not None:
            self.restoreGeometry(geometry)
        if state is not None:
            self.restoreState(state)
        fill = make_settings().value("plotsFillCentral", False)
        if isinstance(fill, str):
            fill = fill.lower() in {"1", "true", "yes"}
        else:
            fill = bool(fill)
        self.session.plots_fill_central = fill
        try:
            action = self.actions.get("plots_fill")
            action.blockSignals(True)
            action.setChecked(fill)
            action.blockSignals(False)
        except KeyError:
            pass

    def _persist_ui(self) -> None:
        if not ui_persist_enabled():
            return
        save_window(self.saveGeometry(), self.saveState())
        make_settings().setValue("plotsFillCentral", self.session.plots_fill_central)

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        self._file_menu = file_menu
        self.actions.add("new", "New", self.new_project, QKeySequence.StandardKey.New, menu=file_menu)
        self.actions.add(
            "open", "Open…", self.open_project, QKeySequence.StandardKey.Open, menu=file_menu
        )
        self._recent_menu = QMenu("Open recent", file_menu)
        file_menu.addMenu(self._recent_menu)
        self.actions.add(
            "save", "Save", self.save_project, QKeySequence.StandardKey.Save, menu=file_menu
        )
        self.actions.add("save_as", "Save as…", self.save_project_as, menu=file_menu)
        file_menu.addSeparator()
        self.actions.add("exit", "Exit", self.close, menu=file_menu)

        edit_menu = self.menuBar().addMenu("&Edit")
        undo = self._history.stack.createUndoAction(self, "Undo")
        undo.setShortcut(QKeySequence.StandardKey.Undo)
        redo = self._history.stack.createRedoAction(self, "Redo")
        redo.setShortcut(QKeySequence.StandardKey.Redo)
        edit_menu.addAction(undo)
        edit_menu.addAction(redo)
        edit_menu.addSeparator()
        self.actions.add("delete_exp", "Delete selected experiment", self.delete_experiment, menu=edit_menu)
        self.actions.get("delete_exp").setEnabled(False)
        self.actions.add(
            "duplicate_exp",
            "Duplicate selected experiment",
            self.duplicate_selected_experiment,
            menu=edit_menu,
        )
        self.actions.get("duplicate_exp").setEnabled(False)
        self.actions.add(
            "delete_step",
            "Delete selected load step",
            self.delete_selected_step,
            menu=edit_menu,
        )
        self.actions.get("delete_step").setEnabled(False)
        self.actions.add(
            "delete_bc",
            "Delete selected BC",
            self.delete_selected_bc,
            menu=edit_menu,
        )
        self.actions.get("delete_bc").setEnabled(False)

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.dock_model.toggleViewAction())
        view_menu.addAction(self.dock_props.toggleViewAction())
        view_menu.addAction(self.dock_log.toggleViewAction())
        view_menu.addAction(self.dock_jobs.toggleViewAction())
        view_menu.addSeparator()
        self.actions.add(
            "xy_axes",
            "Show X/Y axes",
            self._on_show_xy_axes,
            menu=view_menu,
            checkable=True,
            checked=False,
        )
        self.actions.add("reset_cam", "Reset camera", self.viewport.reset_camera, menu=view_menu)
        view_menu.addSeparator()
        plots_fill = self.actions.add(
            "plots_fill",
            "Plots fill central area",
            self._on_plots_fill_toggled,
            menu=view_menu,
            checkable=True,
            checked=False,
        )
        plots_fill.setToolTip(
            "Hide the 3D view so post-processing uses the whole central window. "
            "Double-click the splitter between 3D and plots to toggle."
        )
        plots_fill.setStatusTip(
            "Give plots the full central area (like before the Results splitter)."
        )

        model_menu = self.menuBar().addMenu("&Model")
        self.actions.add("add_exp", "Add experiment", self.add_experiment, menu=model_menu)
        self.actions.add("solver", "Solver settings…", self.edit_solver, menu=model_menu)

        exp_menu = self.menuBar().addMenu("&Experiment")
        self.actions.add("add_step", "Add load steps…", self._add_step_current, menu=exp_menu)
        self.actions.add("import_mat", "Import MAT files…", self._import_mat_current, menu=exp_menu)

        vfm_menu = self.menuBar().addMenu("&VFM")
        self.actions.add(
            "add_prescribed",
            "Add prescribed BC…",
            lambda: self._add_bc_current("prescribed"),
            menu=vfm_menu,
        )
        self.actions.add(
            "add_linking",
            "Add linking BC…",
            lambda: self._add_bc_current("linking"),
            menu=vfm_menu,
        )
        self.actions.add(
            "add_external",
            "Add external BC…",
            lambda: self._add_bc_current("external"),
            menu=vfm_menu,
        )
        vfm_menu.addSeparator()
        self.actions.add("check", "Check consistency", self.check_consistency, menu=vfm_menu)

        identify = self.menuBar().addMenu("&Identify")
        self.actions.add(
            "run", "Run FGLS", self.run_identification, QKeySequence("F5"), menu=identify
        )
        self.actions.add("ident", "Identifiability", self.run_identifiability, menu=identify)
        self.actions.add(
            "uncertainty",
            "Uncertainty quantification",
            self.run_uncertainty,
            menu=identify,
        )
        self.actions.add("cancel_job", "Cancel job", self.cancel_job, menu=identify)
        self.actions.get("cancel_job").setEnabled(False)
        self.actions.add("check2", "Check consistency", self.check_consistency, menu=identify)

        results = self.menuBar().addMenu("&Results")
        self.actions.add("show_vf", "Show virtual field", self.show_virtual_field, menu=results)
        self.actions.add("export_vtk", "Export VTK…", self.export_vtk, menu=results)
        self.actions.add("export_plot", "Export plot…", self.export_post_plot, menu=results)

        help_menu = self.menuBar().addMenu("&Help")
        self.actions.add("workflow", "Workflow", self._show_workflow, menu=help_menu)
        examples = help_menu.addMenu("Examples")
        self.actions.add("open_campaign", "Open campaign example", self.open_campaign, menu=examples)
        self.actions.add(
            "open_samples_ab",
            "Open Samples A/B (all)",
            self.open_samples_ab,
            menu=examples,
        )
        help_menu.addSeparator()
        self.actions.add("about", "About", self._about, menu=help_menu)

    def _build_toolbars(self) -> None:
        self.module_bar = ModuleBar(self)
        self.module_bar.module_changed.connect(self.set_module)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.module_bar)
        _mark_toolbar_buttons(self.module_bar)

        bar = QToolBar("Main")
        bar.setObjectName("mainBar")
        bar.setMovable(False)
        bar.setIconSize(QSize(20, 20))
        bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBarBreak(Qt.ToolBarArea.TopToolBarArea)
        self.addToolBar(bar)
        bar.addAction(icon("open"), "Open", self.open_project)
        bar.addAction(icon("save"), "Save", self.save_project)
        bar.addAction(icon("run"), "Identify", self.run_identification)
        bar.addSeparator()
        self.field_combo = FieldCombo()
        self.field_combo.currentIndexChanged.connect(self._on_field_changed)
        bar.addWidget(self.field_combo)
        self.display_combo = DisplayModeCombo()
        self.display_combo.currentIndexChanged.connect(self._on_display_mode_changed)
        bar.addWidget(self.display_combo)
        bar.addWidget(_toolbar_label("min"))
        self._clim_min = QDoubleSpinBox()
        self._clim_max = QDoubleSpinBox()
        for spin in (self._clim_min, self._clim_max):
            spin.setDecimals(4)
            spin.setRange(-1e12, 1e12)
            spin.setMaximumWidth(92)
            spin.editingFinished.connect(self._on_clim_edited)
        bar.addWidget(self._clim_min)
        bar.addWidget(_toolbar_label("max"))
        bar.addWidget(self._clim_max)
        bar.addAction("Reset range", self._reset_clim)
        bar.setMaximumHeight(40)
        self._main_bar = bar
        _mark_toolbar_buttons(bar)

        self._context = QToolBar("Context")
        self._context.setObjectName("contextBar")
        self._context.setMovable(False)
        self._context.setIconSize(QSize(20, 20))
        self._context.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(self._context)
        self._context.setMaximumHeight(40)

        self._pick_action = QAction("Pick", self)
        self._pick_action.setCheckable(True)
        self._pick_action.toggled.connect(self._on_pick_mode)
        self._rebuild_context_toolbar()

        down = QShortcut(QKeySequence(Qt.Key.Key_PageDown), self)
        down.activated.connect(lambda: self._step_delta(1))
        up = QShortcut(QKeySequence(Qt.Key.Key_PageUp), self)
        up.activated.connect(lambda: self._step_delta(-1))

    def new_project(self) -> None:
        if not self._maybe_save():
            return
        dialog = NewProjectDialog(self)
        if self._exec_dialog(dialog, "NewProject"):
            self.session = AppSession(document=dialog.document(), dirty=True)
            self._carry_view_prefs(self.session)
            self._reset_history()
            self.log.log("Created new project.")
            self._refresh_all()

    def open_project(self) -> None:
        if not self._maybe_save():
            return
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            "Open project",
            "",
            "VFM projects (*.vfmproj.json *.json *.vfm);;All files (*)",
        )
        if chosen:
            self.open_project_path(Path(chosen))

    def open_campaign(self) -> None:
        path = campaign_project_path()
        if path.is_file():
            self.open_project_path(path)
        else:
            QMessageBox.warning(self, APP_NAME, f"Campaign project not found:\n{path}")

    def open_samples_ab(self) -> None:
        path = samples_ab_all_project_path()
        if path.is_file():
            self.open_project_path(path)
        else:
            QMessageBox.warning(
                self,
                APP_NAME,
                f"Samples A/B project not found:\n{path}\n\n"
                "Rebuild with Samples_A_B/VFM_analysis/make_gui_project.py",
            )

    def open_project_path(self, path: Path) -> None:
        try:
            loaded = load_project(Path(path))
        except Exception as exc:
            QMessageBox.critical(self, "Open failed", str(exc))
            self.log.log(f"Open failed: {exc}")
            return
        self.session = AppSession(document=loaded.document, dirty=False)
        self._carry_view_prefs(self.session)
        self.session.result = loaded.result
        self.session.identifiability = loaded.identifiability
        self.session.apply_view_state(getattr(loaded, "view_state", None))
        self.session.uncertainty = getattr(loaded, "uncertainty", None)
        if self.session.uncertainty is None and loaded.result is not None:
            try:
                from vfmstudio.core.jobs import run_uncertainty as restore_uncertainty

                self.session.uncertainty = restore_uncertainty(
                    self.session.document,
                    loaded.result,
                    kind=self.session.uncertainty_kind,
                )
            except Exception as exc:
                self.log.log(f"Uncertainty could not be restored: {exc}")
        self.session.results_unsaved = False
        self._reset_history()
        self._remember_path(Path(path))
        self.log.log(f"Opened {path}")
        self._refresh_all()

    def save_project(self) -> bool:
        path = self._user_save_path()
        if path is None:
            return self.save_project_as()
        return self._write(path)

    def save_project_as(self) -> bool:
        has_results = self.session.result is not None
        if has_results:
            filters = "VFM HDF5 with results (*.vfm);;VFM project JSON (*.vfmproj.json)"
        else:
            filters = "VFM project JSON (*.vfmproj.json);;VFM HDF5 (*.vfm)"
        current = self._user_save_path()
        start = str(current) if current is not None else ""
        chosen, selected = QFileDialog.getSaveFileName(
            self,
            "Save project",
            start,
            filters,
        )
        if not chosen:
            return False
        return self._write(normalize_project_save_path(Path(chosen), selected))

    def _user_save_path(self) -> Path | None:
        path = self.session.path
        if path is None or is_scratch_project_path(path):
            return None
        return Path(path)

    def _write(self, path: Path) -> bool:
        try:
            save_any(
                self.session.document,
                path,
                result=self.session.result,
                identifiability=self.session.identifiability,
                uncertainty=self.session.uncertainty,
                view_state=self.session.view_state(),
            )
            self.session.dirty = False
            self.session.document.source_path = Path(path).resolve()
            self.log.log(f"Saved {path}")
            stored = path.suffix.lower() in {".vfm", ".h5", ".hdf5"}
            if stored:
                self.session.results_unsaved = False
            elif self.session.result is not None or self.session.identifiability is not None:
                self.log.log(
                    "Identification results are stored only in .vfm; "
                    "this JSON file is the solver project."
                )
            self._update_title()
            self._update_status()
            self._remember_path(Path(path))
            return True
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return False

    def _maybe_save(self) -> bool:
        if not self.session.dirty and not self.session.results_unsaved:
            return True
        prompt = "Save the current project?"
        if self.session.results_unsaved:
            prompt = "Identification results are not saved. Save as .vfm to keep them?"
        answer = QMessageBox.question(
            self,
            "Unsaved changes",
            prompt,
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Cancel:
            return False
        if answer == QMessageBox.StandardButton.Save:
            return self.save_project()
        return True

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._maybe_save():
            self._persist_ui()
            event.accept()
        else:
            event.ignore()

    def add_experiment(self) -> None:
        dialog = ExperimentEditorDialog(
            self, name=f"Experiment {self.session.document.next_experiment_id()}"
        )
        if not self._exec_dialog(dialog, "ExperimentEditor"):
            return
        document = self.session.document
        exp_id = document.next_experiment_id()
        document.experiments.append(
            ExperimentModel(
                experiment_id=exp_id,
                name=dialog.experiment_name(),
                mesh=MeshModel(),
            )
        )
        self.session.selection = Selection(kind="experiment", experiment_id=exp_id)
        self._commit_document_edit("Add experiment")
        self._refresh_all()

    def _confirm_delete(self, title: str, text: str) -> bool:
        return (
            QMessageBox.question(
                self,
                title,
                text,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            == QMessageBox.StandardButton.Yes
        )

    def _require_targeted_experiment(self, action: str):
        experiment = self.session.targeted_experiment()
        if experiment is not None:
            return experiment
        if not self.session.document.experiments:
            answer = QMessageBox.question(
                self,
                action,
                "This project has no experiments. Add one now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer == QMessageBox.StandardButton.Yes:
                self.add_experiment()
                return self.session.targeted_experiment()
            return None
        QMessageBox.information(
            self, action, "Select an experiment in the model tree first."
        )
        return None

    def delete_experiment(self) -> None:
        experiment = self.session.targeted_experiment()
        if experiment is None:
            return
        self._delete_experiment_id(experiment.experiment_id)

    def _delete_experiment_id(self, experiment_id: int) -> None:
        experiment = self.session.document.experiment_by_id(experiment_id)
        if experiment is None:
            return
        label = experiment.name or f"Experiment {experiment.experiment_id}"
        if not self._confirm_delete(
            "Delete experiment", f'Delete experiment "{label}" and all of its steps?'
        ):
            return
        self.session.document.experiments = [
            item
            for item in self.session.document.experiments
            if item.experiment_id != experiment.experiment_id
        ]
        self.session.selection = Selection()
        self._commit_document_edit("Delete experiment")
        self.log.log(f"Deleted experiment '{label}'.")
        self._refresh_all()

    def duplicate_selected_experiment(self) -> None:
        experiment = self.session.targeted_experiment()
        if experiment is None:
            return
        self.duplicate_experiment(experiment.experiment_id)

    def duplicate_experiment(self, experiment_id: int) -> None:
        clone = self.session.document.duplicate_experiment(experiment_id)
        if clone is None:
            return
        self.session.selection = Selection(
            kind="experiment", experiment_id=clone.experiment_id
        )
        self._commit_document_edit("Duplicate experiment")
        self.log.log(f"Duplicated experiment as '{clone.name}'.")
        self._refresh_all()

    def edit_solver(self) -> None:
        dialog = SolverSettingsDialog(self, self.session.document.solver)
        if self._exec_dialog(dialog, "SolverSettings"):
            dialog.apply_to(self.session.document.solver)
            self._commit_document_edit("Solver settings")
            self._refresh_all()

    def _add_step_current(self) -> None:
        experiment = self._require_targeted_experiment("Add load steps")
        if experiment is None:
            return
        self._add_step(experiment.experiment_id)

    def _import_mat_current(self) -> None:
        experiment = self._require_targeted_experiment("Import MAT files")
        if experiment is None:
            return
        self._import_mat(experiment.experiment_id)

    def _start_directory_for(self, experiment) -> str:
        if experiment.steps:
            try:
                return str(Path(experiment.steps[-1].data_file).parent)
            except Exception:
                return ""
        return ""

    def _add_step(self, experiment_id: int) -> None:
        experiment = self.session.document.experiment_by_id(experiment_id)
        if experiment is None:
            return
        dialog = AddLoadStepsDialog(
            self,
            start_directory=self._start_directory_for(experiment),
            source_path=self.session.document.source_path,
        )
        if not self._exec_dialog(dialog, "AddStep"):
            return
        self._append_imported_steps(experiment, dialog, "Add load steps")

    def _import_mat(self, experiment_id: int) -> None:
        experiment = self.session.document.experiment_by_id(experiment_id)
        if experiment is None:
            return
        dialog = ImportMatDialog(
            self, start_directory=self._start_directory_for(experiment)
        )
        if not self._exec_dialog(dialog, "ImportMat"):
            return
        self._append_imported_steps(experiment, dialog, "Import MAT files")

    def _append_imported_steps(self, experiment, dialog, title: str) -> None:
        chosen = dialog.steps()
        if not chosen:
            return
        steps = dialog.steps(skip=[step.data_file for step in experiment.steps])
        if not steps:
            QMessageBox.information(
                self,
                title,
                "Those measurement files are already on this experiment.",
            )
            return
        first_index = len(experiment.steps)
        analysis_type = self.session.document.material.analysis_type
        source_path = self.session.document.source_path
        for step in steps:
            experiment.steps.append(step)
            sync_mesh_bounds_from_file(
                experiment,
                step.data_file,
                source_path=source_path,
                analysis_type=analysis_type,
            )
        self.session.selection = Selection(
            kind="step",
            experiment_id=experiment.experiment_id,
            step_index=first_index,
        )
        self._commit_document_edit(title)
        n = len(steps)
        self.log.log(
            f"Added {n} load step{'s' if n != 1 else ''} to "
            f"{experiment.name or f'experiment {experiment.experiment_id}'}."
        )
        self._refresh_all()

    def _add_bc_current(self, kind: str) -> None:
        experiment = self._require_targeted_experiment(f"Add {kind} BC")
        if experiment is None:
            return
        self._add_bc(experiment.experiment_id, kind)

    def _add_bc(self, experiment_id: int, kind: str) -> None:
        experiment = self.session.document.experiment_by_id(experiment_id)
        if experiment is None or kind not in {"prescribed", "linking", "external"}:
            return
        dialog = AddBcDialog(self, kind=kind, node_coords=_bc_dialog_nodes(self.session, experiment))
        if hasattr(dialog, "preview_nodes_changed"):
            dialog.preview_nodes_changed.connect(self._on_bc_node_preview)
        self._active_bc_dialog = dialog
        accepted = False
        try:
            accepted = bool(self._exec_dialog(dialog, f"AddBc:{kind}"))
        finally:
            self._active_bc_dialog = None
            if hasattr(dialog, "preview_nodes_changed"):
                try:
                    dialog.preview_nodes_changed.disconnect(self._on_bc_node_preview)
                except (RuntimeError, TypeError):
                    pass
            self._on_bc_node_preview([], False)
        if accepted:
            new_entries = _dialog_bc_entries(dialog)
            if not new_entries:
                return
            entries = getattr(experiment, kind)
            start = len(entries)
            entries.extend(new_entries)
            if kind == "external":
                for step in experiment.steps:
                    values = list(step.external_values)
                    while len(values) < len(experiment.external):
                        values.append(None)
                    step.external_values = values
            self.session.selection = Selection(
                kind="bc_group" if kind == "external" else "bc",
                experiment_id=experiment.experiment_id,
                bc_kind=kind,
                bc_index=None if kind == "external" else start,
            )
            self._commit_document_edit(f"Add {kind} BC")
            names = ", ".join(entry.dof for entry in new_entries)
            self.log.log(
                f"Added {len(new_entries)} {kind} BC"
                f"{'s' if len(new_entries) != 1 else ''} ({names}) to "
                f"{experiment.name or f'experiment {experiment.experiment_id}'}."
            )
            self._refresh_all()

    def delete_selected_bc(self) -> None:
        selection = self.session.selection
        if selection.kind != "bc" or selection.experiment_id is None:
            return
        if selection.bc_kind is None or selection.bc_index is None:
            return
        self._delete_bc(selection.experiment_id, selection.bc_kind, selection.bc_index)

    def _delete_bc(self, experiment_id: int, kind: str, index: int) -> None:
        experiment = self.session.document.experiment_by_id(experiment_id)
        if experiment is None:
            return
        try:
            entries = experiment.boundary_entries(kind)
        except KeyError:
            return
        if index < 0 or index >= len(entries):
            return
        entry = entries[index]
        label = entry.label or f"{entry.dof} @{entry.edge or 'nodes'}"
        if not self._confirm_delete(
            "Delete boundary condition", f'Delete {kind} BC "{label}"?'
        ):
            return
        removed = experiment.remove_boundary_entry(kind, index)
        if removed is None:
            return
        self.session.selection = Selection(
            kind="bc_group",
            experiment_id=experiment.experiment_id,
            bc_kind=kind,
        )
        self._commit_document_edit("Delete BC")
        self.log.log(
            f"Deleted {kind} BC '{label}' from "
            f"{experiment.name or f'experiment {experiment.experiment_id}'}."
        )
        self._refresh_all()

    def delete_selected_step(self) -> None:
        experiment = self.session.targeted_experiment()
        index = self.session.targeted_step_index()
        if experiment is None or index is None:
            return
        self._delete_step(experiment.experiment_id, index)

    def _delete_step(self, experiment_id: int, index: int) -> None:
        experiment = self.session.document.experiment_by_id(experiment_id)
        if experiment is None:
            return
        if index < 0 or index >= len(experiment.steps):
            return
        step = experiment.steps[index]
        label = step.label or Path(step.data_file).name
        if not self._confirm_delete(
            "Delete load step", f'Delete load step "{label}"?'
        ):
            return
        removed = experiment.remove_step(index)
        if removed is None:
            return
        if experiment.steps:
            next_index = min(index, len(experiment.steps) - 1)
            self.session.selection = Selection(
                kind="step",
                experiment_id=experiment.experiment_id,
                step_index=next_index,
            )
        else:
            self.session.selection = Selection(
                kind="steps", experiment_id=experiment.experiment_id
            )
        self._commit_document_edit("Delete load step")
        self.log.log(
            f"Deleted load step '{label}' from "
            f"{experiment.name or f'experiment {experiment.experiment_id}'}."
        )
        self._refresh_all()

    def _move_step(self, experiment_id: int, index: int, delta: int) -> None:
        experiment = self.session.document.experiment_by_id(experiment_id)
        if experiment is None:
            return
        if not experiment.move_step(index, delta):
            return
        self.session.selection = Selection(
            kind="step",
            experiment_id=experiment.experiment_id,
            step_index=index + int(delta),
        )
        self._commit_document_edit("Reorder load step")
        self._refresh_all()

    def _on_document_edited(self) -> None:
        selection = self.session.selection
        key = (
            f"{selection.kind}:{selection.experiment_id}:"
            f"{selection.step_index}:{selection.bc_kind}:{selection.bc_index}"
        )
        self._commit_document_edit("Edit properties", coalesce_key=key)
        self.tree.rebuild(self.session)
        self._apply_selection()
        self._update_title()
        self._update_status()

    def _on_tree_selection(self, selection: Selection) -> None:
        self.session.selection = selection
        self._apply_selection()

    def _apply_selection(self) -> None:
        selection = self.session.selection
        if is_post_kind(selection.kind):
            self.session.last_post_kind = selection.kind
            if self.session.module != "results":
                self.session.module = "results"
                self.module_bar.set_current("results")
                self._apply_module_layout()
                self._rebuild_context_toolbar()
        if self.session.free_mesh_mode and not self._selection_allows_free_mesh():
            self.session.free_mesh_mode = False
            self.viewport.leave_free_mesh(self.session)
        if self.session.drag_mesh_mode and not self._selection_allows_free_mesh():
            self.session.drag_mesh_mode = False
            self.viewport.leave_drag_mesh(self.session)
        self.props.show_selection(self.session, selection)
        post = is_post_kind(selection.kind)
        if hasattr(self, "field_combo"):
            self.field_combo.setEnabled(True)
        if hasattr(self, "display_combo"):
            self.display_combo.setEnabled(True)
        try:
            self.actions.get("delete_bc").setEnabled(selection.kind == "bc")
            targeted = self.session.targeted_experiment() is not None
            self.actions.get("delete_exp").setEnabled(targeted)
            self.actions.get("duplicate_exp").setEnabled(targeted)
            self.actions.get("add_step").setEnabled(
                targeted or not self.session.document.experiments
            )
            self.actions.get("import_mat").setEnabled(
                targeted or not self.session.document.experiments
            )
            for name in ("add_prescribed", "add_linking", "add_external"):
                self.actions.get(name).setEnabled(targeted)
            self.actions.get("delete_step").setEnabled(
                selection.kind == "step" and targeted
            )
            self.actions.get("export_vtk").setEnabled(
                targeted or self.session.selected_experiment() is not None
            )
        except KeyError:
            pass
        if post:
            if self.viewport.load_plot_visible():
                self.viewport.hide_load_plot()
            self.post.show_kind(selection.kind)
            if selection.kind == "post_stress_space":
                self._apply_stress_space_view()
        self._apply_module_layout()
        self.viewport.refresh(self.session)
        self._update_status()

    def _show_step_load_plot(self) -> None:
        experiment = self.session.targeted_experiment()
        if experiment is None or not experiment.steps:
            QMessageBox.information(
                self, "Load data", "Select an experiment with load steps first."
            )
            return
        self._apply_module_layout()
        self.viewport.show()
        self.viewport.show_load_plot(self.session)

    def _on_field_changed(self, _index: int = 0) -> None:
        name = self.field_combo.currentData()
        self.session.current_scalar = str(name or self.field_combo.currentText() or "kappa_yy")
        self.viewport.refresh(self.session)
        self._update_status()

    def _on_display_mode_changed(self, _index: int = 0) -> None:
        mode = self.display_combo.currentData()
        self.session.display_mode = str(mode or "contour")
        self.viewport.refresh(self.session)

    def _on_show_xy_axes(self, checked: bool) -> None:
        self.session.show_xy_axes = bool(checked)
        self.viewport.refresh(self.session)

    def _on_deformation_scale(self, scale: float, live: bool = True) -> None:
        try:
            self.viewport.apply_deformation_scale(
                self.session, float(scale), live=bool(live)
            )
        except Exception:
            pass

    def _on_view_option_changed(self) -> None:
        kind = self.session.selection.kind
        if kind in {"post_plots", "post_processing"}:
            self.post.plots.apply_combination(self.session.document)
            self._update_title()
            return
        if kind == "post_parameter_space":
            self._apply_combination_view()
            self._update_title()
            return
        self._apply_combination_view()
        try:
            self.viewport.refresh(self.session)
        except Exception:
            pass
        self.post.experiment_check.apply_display(
            visible=self.session.experiment_check_visible,
            colors=self.session.experiment_check_colors,
        )
        self.post.ident_strength.apply_baseline(
            self.session.identifiability_baseline,
            shift_scale=self.session.identifiability_shift_scale,
        )
        self.post.vf_panel.update_from(
            self.session.document,
            self.session.result,
            warp_scale=self.session.warp_scale,
            experiment_id=self.session.vf_experiment_id,
            step_index=self.session.vf_step_index,
            view_mode=self.session.vf_view_mode,
            quantity=self.session.vf_quantity,
        )
        if self.session.show_virtual_field:
            try:
                self.viewport.show_virtual_field(
                    self.session, parameter_index=self.session.vf_parameter_index
                )
            except Exception:
                pass
        self._apply_post_plot_modes()
        self._apply_stress_space_view()
        self._update_title()
        self._update_status()

    def _apply_post_plot_modes(self) -> None:
        modes = self.session.plot_modes or {}
        mapping = {
            "post_experiment_check": self.post.experiment_check,
            "post_uncertainty_cov": self.post.uq_cov,
            "post_uncertainty_experiments": self.post.uq_experiments,
            "post_uncertainty_ab": self.post.uq_ab,
            "post_uncertainty_info": self.post.uq_info,
            "post_uncertainty_loo": self.post.uq_loo,
        }
        for kind, panel in mapping.items():
            text = modes.get(kind)
            setter = getattr(panel, "set_plot_mode", None)
            if text and callable(setter):
                setter(text)

    def _sync_field_combo(self) -> None:
        available = list(self.session.available_scalars)
        if self.session.identifiability is not None and "CID" not in available:
            available.append("CID")
        if not available:
            return
        self.field_combo.set_fields(available, current=self.session.current_scalar)
        selected = self.field_combo.currentData()
        if selected:
            self.session.current_scalar = str(selected)

    def _refresh_all(self) -> None:
        self.tree.rebuild(self.session)
        self.post.set_results(
            self.session.document,
            self.session.result,
            self.session.identifiability,
            warp_scale=self.session.warp_scale,
            uncertainty=self.session.uncertainty,
            vf_experiment_id=self.session.vf_experiment_id,
            vf_step_index=self.session.vf_step_index,
            vf_view_mode=self.session.vf_view_mode,
            vf_quantity=self.session.vf_quantity,
        )
        self.post.experiment_check.apply_display(
            visible=self.session.experiment_check_visible,
            colors=self.session.experiment_check_colors,
        )
        self.post.ident_strength.apply_baseline(
            self.session.identifiability_baseline,
            shift_scale=self.session.identifiability_shift_scale,
        )
        self._apply_combination_view()
        self._apply_stress_space_view()
        self._apply_selection()
        self._update_title()
        self._update_status()

    def _update_title(self) -> None:
        name = APP_NAME
        path = self._user_save_path()
        if path is not None:
            name += f" — {path.name}"
        if self.session.dirty:
            name += " *"
        self.setWindowTitle(name)

    def _update_status(self) -> None:
        experiment = self.session.selected_experiment()
        exp_name = (
            experiment.name or f"Experiment {experiment.experiment_id}"
            if experiment is not None
            else "no experiment"
        )
        step = self.session.selected_step()
        frame = "—"
        n_steps = 0
        if experiment is not None and experiment.steps:
            n_steps = len(experiment.steps)
            index = self.session.selection.step_index
            if index is None:
                index = 0
            frame = f"{index + 1}/{n_steps}"
        n_pts = self.session.status_n_points
        self._sync_field_combo()
        self.statusBar().showMessage(
            f"Ready | {n_pts} points | Frame {frame} | {exp_name}"
        )

    def check_consistency(self) -> None:
        report = validate_document(self.session.document)
        for item in report.errors:
            self.log.log(f"ERROR: {item}")
        for item in report.warnings:
            self.log.log(f"WARNING: {item}")
        if report.errors:
            QMessageBox.warning(self, "Consistency", "\n".join(report.errors[:12]))
        elif report.warnings:
            QMessageBox.information(self, "Consistency", "\n".join(report.warnings[:12]))
        else:
            QMessageBox.information(self, "Consistency", "No GLS consistency problems.")

    def _apply_combination_view(self) -> None:
        ids = selected_experiment_ids(self.session.document)
        self.post.plots.apply_combination(self.session.document)
        self.post.parameter_space.set_selected_ids(ids)
        self.post.parameter_space.set_groups(self.session.parameter_space_groups)
        self.post.parameter_space.apply_axes(
            self.session.parameter_space_left_x,
            self.session.parameter_space_left_y,
            self.session.parameter_space_right_x,
            self.session.parameter_space_right_y,
        )

    def _apply_stress_space_view(self) -> None:
        self.post.stress_space.apply_view(
            experiment_id=self.session.stress_space_experiment_id,
            step_index=self.session.stress_space_step_index,
            x2d=self.session.stress_space_2d_x,
            y2d=self.session.stress_space_2d_y,
            x3d=self.session.stress_space_3d_x,
            y3d=self.session.stress_space_3d_y,
            z3d=self.session.stress_space_3d_z,
            color=self.session.stress_space_color,
        )

    def run_identification(self) -> None:
        if not self._begin_job("Identifying…"):
            return
        report = validate_document(self.session.document)
        if report.errors:
            self._finish_job("Cannot identify")
            QMessageBox.warning(self, "Cannot identify", "\n".join(report.errors[:12]))
            for item in report.errors:
                self.log.log(f"ERROR: {item}")
            return
        for item in report.warnings:
            self.log.log(f"WARNING: {item}")
            if QMessageBox.question(self, "Load orientation", item + "\n\nContinue?") != (
                QMessageBox.StandardButton.Yes
            ):
                self._finish_job("Ready")
                return
        self._start_job_process("identify", "FGLS identification")

    def _on_identify_finished(self, payload, token: int = 0) -> None:
        if not self._job_is_live(token):
            self.log.log("Cancelled; ignoring stale result.")
            return
        ident = None
        uq = None
        if isinstance(payload, tuple):
            if len(payload) >= 3:
                result, ident, uq = payload[0], payload[1], payload[2]
            else:
                result, ident = payload[0], payload[1]
        else:
            result = payload
        self.session.result = result
        if ident is not None:
            self.session.identifiability = ident
        self.session.uncertainty = uq
        self.session.results_unsaved = True
        self.session.mark_dirty()
        n_fields = len(getattr(result, "optimal_field_results", None) or [])
        self.session.selection = Selection(kind="post_plots")
        self.log.log(
            f"Identification finished ({n_fields} optimal virtual field(s) stored)."
        )
        if uq is None:
            self.log.log("Uncertainty quantification was not computed.")
        self._finish_job("Identification finished")
        self._refresh_all()

    def run_identifiability(self) -> None:
        if self.session.result is None:
            QMessageBox.information(self, "Identifiability", "Run identification first.")
            return
        if not self._begin_job("Computing identifiability…"):
            return
        self._start_job_process("identifiability", "Identifiability")

    def _on_ident_finished(self, diagnostics, token: int = 0) -> None:
        if not self._job_is_live(token):
            self.log.log("Cancelled; ignoring stale result.")
            return
        self.session.identifiability = diagnostics
        self.session.results_unsaved = True
        self.session.mark_dirty()
        self.session.selection = Selection(kind="post_identifiability")
        self.log.log("Identifiability finished.")
        self._finish_job("Identifiability finished")
        self._refresh_all()

    def run_uncertainty(self) -> None:
        if self.session.result is None:
            QMessageBox.information(
                self, "Uncertainty quantification", "Run identification first."
            )
            return
        if not self._begin_job("Computing uncertainty…"):
            return
        self._start_job_process("uncertainty", "Uncertainty quantification")

    def _on_uncertainty_finished(self, payload, token: int = 0) -> None:
        if not self._job_is_live(token):
            self.log.log("Cancelled; ignoring stale result.")
            return
        self.session.uncertainty = payload
        self.session.results_unsaved = True
        self.session.mark_dirty()
        kind = self.session.selection.kind
        if not str(kind).startswith("post_uncertainty"):
            self.session.selection = Selection(kind="post_uncertainty_cov")
        self.log.log("Uncertainty quantification finished.")
        self._finish_job("Uncertainty quantification finished")
        self._refresh_all()

    def run_loo_refit(self) -> None:
        if self.session.uncertainty is None:
            QMessageBox.information(
                self,
                "Leave-one-out refit",
                "Compute uncertainty quantification first.",
            )
            return
        if not self._begin_job("Leave-one-out FGLS refits…"):
            return
        self._start_job_process("loo", "Leave-one-out refit")

    def _start_job_process(self, kind: str, name: str) -> None:
        token = self._job_token
        self.jobs.start_job(name, token)
        self._sync_bottom_docks()
        worker = JobProcess(self)
        worker.log.connect(self.log.log)
        worker.log.connect(lambda text, t=token: self.jobs.log_line(text, t))
        worker.failed.connect(lambda message, t=token: self._on_job_failed(message, t))
        worker.finished.connect(
            lambda loaded, t=token, k=kind: self._on_job_process_finished(loaded, t, k)
        )
        self._job_process = worker
        worker.start(
            kind,
            self.session.document,
            result=self.session.result,
            identifiability=self.session.identifiability,
            uncertainty=self.session.uncertainty,
            view_state=self.session.view_state(),
            uq_kind=self.session.uncertainty_kind,
        )
        self.log.log(f"Queued {name}.")

    def _on_job_process_finished(self, loaded, token: int, kind: str) -> None:
        if kind == "identify":
            self._on_identify_finished(
                (loaded.result, loaded.identifiability, loaded.uncertainty), token
            )
        elif kind == "identifiability":
            self._on_ident_finished(loaded.identifiability, token)
        else:
            self._on_uncertainty_finished(loaded.uncertainty, token)

    def _begin_job(self, status: str) -> bool:
        if self._job_busy:
            QMessageBox.information(
                self, APP_NAME, "A job is already running. Cancel it first."
            )
            return False
        self._job_busy = True
        self._job_token += 1
        self._set_job_actions_enabled(False)
        self.statusBar().showMessage(status)
        self._sync_bottom_docks()
        return True

    def _job_is_live(self, token: int) -> bool:
        return self._job_busy and int(token) == int(self._job_token)

    def _finish_job(self, status: str = "Ready") -> None:
        token = self._job_token
        self.jobs.finish_job(status if status != "Ready" else "Completed", token)
        self._job_busy = False
        self._jobs = []
        self._job_process = None
        self._set_job_actions_enabled(True)
        self.statusBar().showMessage(status)
        self._sync_bottom_docks()

    def _set_job_actions_enabled(self, enabled: bool) -> None:
        for name in ("run", "ident", "uncertainty"):
            try:
                self.actions.get(name).setEnabled(enabled)
            except KeyError:
                pass
        try:
            self.actions.get("cancel_job").setEnabled(not enabled)
        except KeyError:
            pass

    def cancel_job(self) -> None:
        if not self._job_busy:
            return
        token = self._job_token
        self._job_token += 1
        self._job_busy = False
        self._jobs = []
        proc = self._job_process
        self._job_process = None
        if proc is not None:
            proc.kill()
        self._set_job_actions_enabled(True)
        self.jobs.mark_cancelled(token)
        self.log.log("Cancelled; solver process stopped.")
        self.statusBar().showMessage("Cancelled")
        self._sync_bottom_docks()

    def _on_job_failed(self, message: str, token: int = 0) -> None:
        if not self._job_is_live(token):
            self.log.log("Cancelled; ignoring stale result.")
            return
        self.log.log(f"FAILED: {message}")
        self._finish_job("Job failed")
        QMessageBox.critical(self, "Job failed", message)

    def show_virtual_field(self) -> None:
        self.set_module("results")
        self.session.selection = Selection(kind="post_virtual_fields")
        self.tree.select(self.session.selection)
        self._apply_selection()
        if self.session.result is None or not getattr(
            self.session.result, "optimal_field_results", None
        ):
            QMessageBox.information(
                self,
                "Virtual fields",
                "Run Identify first. Optimal virtual fields are computed "
                "together with FGLS.",
            )
            return
        self.session.show_virtual_field = True
        error = self.viewport.show_virtual_field(
            self.session, parameter_index=self.session.vf_parameter_index
        )
        if error:
            QMessageBox.information(self, "Virtual fields", error)

    def export_vtk(self) -> None:
        experiment = (
            self.session.targeted_experiment() or self.session.selected_experiment()
        )
        if experiment is None:
            QMessageBox.information(self, "Export", "Select an experiment first.")
            return
        chosen, _ = QFileDialog.getSaveFileName(self, "Export VTK", "", "VTK (*.vtk)")
        if not chosen:
            return
        try:
            mesh_pkg = self.viewport._build_mesh(self.session.document, experiment)
            point_data = self.viewport.displayed_point_data()
            export_mesh_vtk(
                Path(chosen),
                mesh_xyz(mesh_pkg),
                mesh_quads(mesh_pkg),
                point_data=point_data,
            )
            self.log.log(f"Wrote {chosen}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    def export_post_plot(self) -> None:
        kind = self.session.selection.kind
        figure = self.post.figure_for_kind(kind)
        if figure is None:
            QMessageBox.information(self, "Export", "This page has no plot to export.")
            return
        stem = default_export_stem(kind)
        start = f"{stem}.pdf"
        if self.session.path is not None:
            start = str(self.session.path.parent / f"{stem}.pdf")
        chosen, selected = QFileDialog.getSaveFileName(
            self, "Export plot", start, FILE_FILTER
        )
        if not chosen:
            return
        path = resolve_export_path(chosen, selected)
        try:
            save_figure(figure, path)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.log.log(f"Wrote {path}")
        self.statusBar().showMessage(f"Exported {path.name}", 4000)

    def _about(self) -> None:
        QMessageBox.information(
            self,
            f"About {APP_NAME}",
            f"{APP_NAME} {__version__} — CAE frontend for the VFM plate solver.\n"
            f"Created by {APP_AUTHOR}, Ph.D. candidate, Caltech.\n\n"
            f"{APP_COPYRIGHT_LINE} {APP_TRADEMARK_LINE}\n"
            f"{APP_LICENSE_LINE}\n",
        )


def _bc_dialog_nodes(session, experiment):
    try:
        mesh_pkg = build_experiment_mesh(session.document, experiment)
        return mesh_xyz(mesh_pkg)
    except Exception:
        return None


def _dialog_bc_entries(dialog):
    if hasattr(dialog, "entries"):
        try:
            found = list(dialog.entries())
        except Exception:
            found = []
        if found:
            return found
    entry = getattr(dialog, "entry", None)
    if callable(entry):
        value = entry()
        if value is not None:
            return [value]
    return []

