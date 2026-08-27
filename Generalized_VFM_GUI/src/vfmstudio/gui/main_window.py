"""VFM Studio main window."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QSize, QThreadPool
from PySide6.QtGui import QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QTabWidget,
    QToolBar,
)

from vfmstudio.core.document import ExperimentModel, MeshModel
from vfmstudio.core.fields import sync_mesh_bounds_from_file
from vfmstudio.core.jobs import validate_document
from vfmstudio.core.mesh_convert import mesh_quads, mesh_xyz
from vfmstudio.core.paths import campaign_project_path
from vfmstudio.gui.actions import ActionRegistry
from vfmstudio.gui.dialogs import (
    AddBcDialog,
    ExperimentEditorDialog,
    NewProjectDialog,
    SolverSettingsDialog,
    ensure_experiment,
)
from vfmstudio.gui.dialogs.import_mat import load_steps_from_paths
from vfmstudio.gui.docks.message_log import MessageLogDock
from vfmstudio.gui.docks.model_tree import ModelTreeDock
from vfmstudio.gui.docks.plot_dock import PlotDock
from vfmstudio.gui.docks.property_editor import PropertyEditorDock
from vfmstudio.gui.results import (
    CoefficientPanel,
    ConvergencePanel,
    IdentifiabilityPanel,
    VirtualFieldPanel,
)
from vfmstudio.gui.session import AppSession, Selection
from vfmstudio.gui.viewport.interactor import ViewportWidget
from vfmstudio.gui.widgets import DisplayModeCombo, FieldCombo
from vfmstudio.gui.workers import IdentifiabilityRunnable, IdentifyRunnable
from vfmstudio.io import load_any, save_any
from vfmstudio.io.export_vtk import export_mesh_vtk
from vfmstudio.resources import icon


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.session = AppSession()
        self.pool = QThreadPool.globalInstance()
        self._jobs = []
        self.setWindowTitle("VFM Studio")
        self.resize(1400, 900)
        self.actions = ActionRegistry(self)

        self.viewport = ViewportWidget(self)
        self.setCentralWidget(self.viewport)

        self.tree = ModelTreeDock()
        self.props = PropertyEditorDock()
        self.log = MessageLogDock()
        self.plots = PlotDock()
        self.coeff = CoefficientPanel()
        self.conv = ConvergencePanel()
        self.ident_panel = IdentifiabilityPanel()
        self.vf_panel = VirtualFieldPanel()

        results = QTabWidget()
        results.addTab(self.coeff, "Coefficients")
        results.addTab(self.conv, "Convergence")
        results.addTab(self.ident_panel, "Identifiability")
        results.addTab(self.vf_panel, "Virtual fields")

        self.dock_model = self._add_dock("Model", self.tree, Qt.DockWidgetArea.LeftDockWidgetArea)
        self.dock_props = self._add_dock(
            "Properties", self.props, Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.dock_plots = self._add_dock("Plots", self.plots, Qt.DockWidgetArea.BottomDockWidgetArea)
        self.dock_results = self._add_dock(
            "Results", results, Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.dock_log = self._add_dock("Log", self.log, Qt.DockWidgetArea.BottomDockWidgetArea)
        self.dock_model.setMinimumWidth(280)
        self.dock_model.setMaximumWidth(420)
        self.dock_props.setMinimumWidth(260)
        self.dock_props.setMaximumWidth(360)
        self.resizeDocks([self.dock_model], [320], Qt.Orientation.Horizontal)
        self.setDockNestingEnabled(True)
        self.tabifyDockWidget(self.dock_plots, self.dock_results)
        self.tabifyDockWidget(self.dock_plots, self.dock_log)
        self.dock_plots.raise_()
        self.resizeDocks([self.dock_plots], [200], Qt.Orientation.Vertical)

        self.tree.selection_changed.connect(self._on_tree_selection)
        self.props.document_changed.connect(self._on_document_edited)
        self.props.view_changed.connect(self._on_view_option_changed)
        self.props.request_add_bc.connect(self._add_bc)
        self.props.request_add_step.connect(self._add_step)
        self.props.request_visualize_steps.connect(self._show_step_load_plot)

        self._build_menus()
        self._build_toolbar()
        self.statusBar().showMessage("Ready")
        self._refresh_all()

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
            if msg.message == 0x0232:  # WM_EXITSIZEMOVE
                plotter = getattr(self.viewport, "_plotter", None)
                if plotter is not None:
                    plotter.render()
        except Exception:
            pass
        return result

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
            "AddBc:prescribed": (460, 240),
            "AddBc:linking": (460, 240),
            "AddBc:external": (460, 240),
            "ExperimentEditor": (420, 180),
        }
        min_w, min_h = mins.get(name, (460, 220))
        if name.startswith("AddBc:"):
            min_w, min_h = 460, 240
        dialog.setMinimumSize(min_w, min_h)
        dialog.adjustSize()
        hint = dialog.sizeHint()
        dialog.resize(max(min_w, hint.width() + 40), max(min_h, hint.height() + 24))
        return dialog.exec()

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        self.actions.add("new", "New", self.new_project, QKeySequence.StandardKey.New, menu=file_menu)
        self.actions.add(
            "open", "Open…", self.open_project, QKeySequence.StandardKey.Open, menu=file_menu
        )
        self.actions.add("open_campaign", "Open campaign example", self.open_campaign, menu=file_menu)
        self.actions.add(
            "save", "Save", self.save_project, QKeySequence.StandardKey.Save, menu=file_menu
        )
        self.actions.add("save_as", "Save as…", self.save_project_as, menu=file_menu)
        file_menu.addSeparator()
        self.actions.add("exit", "Exit", self.close, menu=file_menu)

        edit_menu = self.menuBar().addMenu("&Edit")
        self.actions.add("delete_exp", "Delete selected experiment", self.delete_experiment, menu=edit_menu)

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.dock_model.toggleViewAction())
        view_menu.addAction(self.dock_props.toggleViewAction())
        view_menu.addAction(self.dock_plots.toggleViewAction())
        view_menu.addAction(self.dock_results.toggleViewAction())
        view_menu.addAction(self.dock_log.toggleViewAction())
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

        model_menu = self.menuBar().addMenu("&Model")
        self.actions.add("add_exp", "Add experiment", self.add_experiment, menu=model_menu)
        self.actions.add("solver", "Solver settings…", self.edit_solver, menu=model_menu)

        exp_menu = self.menuBar().addMenu("&Experiment")
        self.actions.add("add_step", "Add load steps…", self._add_step_current, menu=exp_menu)
        self.actions.add("import_mat", "Import MAT files…", self._add_step_current, menu=exp_menu)

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
            "Add EXTERNAL group…",
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
        self.actions.add("check2", "Check consistency", self.check_consistency, menu=identify)

        results = self.menuBar().addMenu("&Results")
        self.actions.add("show_vf", "Show virtual field", self.show_virtual_field, menu=results)
        self.actions.add("export_vtk", "Export mesh VTK…", self.export_vtk, menu=results)

        help_menu = self.menuBar().addMenu("&Help")
        self.actions.add("about", "About", self._about, menu=help_menu)

    def _build_toolbar(self) -> None:
        bar = QToolBar("Main")
        bar.setMovable(False)
        bar.setIconSize(QSize(20, 20))
        bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
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
        bar.setMaximumHeight(40)

    def new_project(self) -> None:
        if not self._maybe_save():
            return
        dialog = NewProjectDialog(self)
        if self._exec_dialog(dialog, "NewProject"):
            self.session = AppSession(document=dialog.document(), dirty=True)
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
            QMessageBox.warning(self, "VFM Studio", f"Campaign project not found:\n{path}")

    def open_project_path(self, path: Path) -> None:
        try:
            document = load_any(Path(path))
        except Exception as exc:
            QMessageBox.critical(self, "Open failed", str(exc))
            self.log.log(f"Open failed: {exc}")
            return
        self.session = AppSession(document=document, dirty=False)
        self.log.log(f"Opened {path}")
        self._refresh_all()

    def save_project(self) -> None:
        path = self.session.path
        if path is None:
            self.save_project_as()
            return
        self._write(path)

    def save_project_as(self) -> None:
        chosen, _ = QFileDialog.getSaveFileName(
            self,
            "Save project",
            "",
            "VFM project JSON (*.vfmproj.json);;VFM HDF5 (*.vfm)",
        )
        if chosen:
            self._write(Path(chosen))

    def _write(self, path: Path) -> None:
        try:
            save_any(self.session.document, path, result=self.session.result)
            self.session.dirty = False
            self.session.document.source_path = Path(path)
            self.log.log(f"Saved {path}")
            self._update_title()
            self._update_status()
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def _maybe_save(self) -> bool:
        if not self.session.dirty:
            return True
        answer = QMessageBox.question(
            self,
            "Unsaved changes",
            "Save the current project?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Cancel:
            return False
        if answer == QMessageBox.StandardButton.Save:
            self.save_project()
        return True

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._maybe_save():
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
        self.session.mark_dirty()
        self._refresh_all()

    def delete_experiment(self) -> None:
        experiment = self.session.selected_experiment()
        if experiment is None:
            return
        self.session.document.experiments = [
            item
            for item in self.session.document.experiments
            if item.experiment_id != experiment.experiment_id
        ]
        self.session.selection = Selection()
        self.session.mark_dirty()
        self._refresh_all()

    def edit_solver(self) -> None:
        dialog = SolverSettingsDialog(self, self.session.document.solver)
        if self._exec_dialog(dialog, "SolverSettings"):
            dialog.apply_to(self.session.document.solver)
            self.session.mark_dirty()
            self._refresh_all()

    def _add_step_current(self) -> None:
        experiment = self.session.selected_experiment()
        if experiment is None:
            experiment = ensure_experiment(self.session.document)
        self._add_step(experiment.experiment_id)

    def _add_step(self, experiment_id: int) -> None:
        experiment = self.session.document.experiment_by_id(experiment_id)
        if experiment is None:
            return
        start_directory = ""
        if experiment.steps:
            try:
                start_directory = str(Path(experiment.steps[-1].data_file).parent)
            except Exception:
                start_directory = ""
        chosen, _ = QFileDialog.getOpenFileNames(
            self,
            "Add load steps",
            start_directory,
            "MAT (*.mat);;All files (*)",
        )
        if not chosen:
            return
        steps = load_steps_from_paths(
            chosen,
            skip=[step.data_file for step in experiment.steps],
        )
        if not steps:
            QMessageBox.information(
                self,
                "Add load steps",
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
        self.session.mark_dirty()
        n = len(steps)
        self.log.log(
            f"Added {n} load step{'s' if n != 1 else ''} to "
            f"{experiment.name or f'experiment {experiment.experiment_id}'}."
        )
        self._refresh_all()

    def _add_bc_current(self, kind: str) -> None:
        experiment = self.session.selected_experiment() or ensure_experiment(
            self.session.document
        )
        self._add_bc(experiment.experiment_id, kind)

    def _add_bc(self, experiment_id: int, kind: str) -> None:
        experiment = self.session.document.experiment_by_id(experiment_id)
        if experiment is None:
            return
        dialog = AddBcDialog(self, kind=kind)
        if self._exec_dialog(dialog, f"AddBc:{kind}"):
            getattr(experiment, kind).append(dialog.entry())
            self.session.mark_dirty()
            self._refresh_all()

    def _on_document_edited(self) -> None:
        self.tree.rebuild(self.session)
        self.viewport.refresh(self.session)
        self._update_title()
        self._update_status()

    def _on_tree_selection(self, selection: Selection) -> None:
        self.session.selection = selection
        self.props.show_selection(self.session, selection)
        self.viewport.refresh(self.session)
        self._update_status()

    def _show_step_load_plot(self) -> None:
        experiment = self.session.selected_experiment()
        if experiment is None or not experiment.steps:
            QMessageBox.information(
                self, "Load data", "Select an experiment with load steps first."
            )
            return
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

    def _on_view_option_changed(self) -> None:
        self.viewport.refresh(self.session)
        self._update_status()

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
        self.props.show_selection(self.session, self.session.selection)
        self.viewport.refresh(self.session)
        self.plots.set_results(self.session.result, self.session.identifiability)
        self.coeff.update_from(self.session.document, self.session.result)
        self.conv.update_from(self.session.result)
        self.ident_panel.update_from(self.session.identifiability)
        self.vf_panel.update_from(self.session.result)
        self._update_title()
        self._update_status()

    def _update_title(self) -> None:
        name = "VFM Studio"
        path = self.session.path
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

    def run_identification(self) -> None:
        report = validate_document(self.session.document)
        if report.errors:
            QMessageBox.warning(self, "Cannot identify", "\n".join(report.errors[:12]))
            for item in report.errors:
                self.log.log(f"ERROR: {item}")
            return
        for item in report.warnings:
            self.log.log(f"WARNING: {item}")
            if QMessageBox.question(self, "Load orientation", item + "\n\nContinue?") != (
                QMessageBox.StandardButton.Yes
            ):
                return
        worker = IdentifyRunnable(self.session.document)
        worker.signals.log.connect(self.log.log)
        worker.signals.failed.connect(self._on_job_failed)
        worker.signals.finished.connect(self._on_identify_finished)
        self._jobs.append(worker)
        self.statusBar().showMessage("Identifying…")
        self.log.log("Queued FGLS identification.")
        self.pool.start(worker)

    def _on_identify_finished(self, result) -> None:
        self.session.result = result
        self.log.log("Identification finished.")
        self.statusBar().showMessage("Identification finished")
        self._refresh_all()

    def run_identifiability(self) -> None:
        if self.session.result is None:
            QMessageBox.information(self, "Identifiability", "Run identification first.")
            return
        worker = IdentifiabilityRunnable(self.session.document, self.session.result)
        worker.signals.log.connect(self.log.log)
        worker.signals.failed.connect(self._on_job_failed)
        worker.signals.finished.connect(self._on_ident_finished)
        self._jobs.append(worker)
        self.pool.start(worker)

    def _on_ident_finished(self, diagnostics) -> None:
        self.session.identifiability = diagnostics
        self.log.log("Identifiability finished.")
        self._refresh_all()

    def _on_job_failed(self, message: str) -> None:
        self.log.log(f"FAILED: {message}")
        self.statusBar().showMessage("Job failed")
        QMessageBox.critical(self, "Job failed", message)

    def show_virtual_field(self) -> None:
        self.viewport.show_virtual_field(self.session, 0)

    def export_vtk(self) -> None:
        experiment = self.session.selected_experiment()
        if experiment is None:
            QMessageBox.information(self, "Export", "Select an experiment first.")
            return
        chosen, _ = QFileDialog.getSaveFileName(self, "Export VTK", "", "VTK (*.vtk)")
        if not chosen:
            return
        try:
            mesh_pkg = self.viewport._build_mesh(self.session.document, experiment)
            export_mesh_vtk(Path(chosen), mesh_xyz(mesh_pkg), mesh_quads(mesh_pkg))
            self.log.log(f"Wrote {chosen}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    def _about(self) -> None:
        QMessageBox.information(
            self,
            "VFM Studio",
            "VFM Studio 0.1 — CAE frontend for the Generalized VFM plate solver.\n"
            "The solver folder is not modified; this application wraps it.",
        )
