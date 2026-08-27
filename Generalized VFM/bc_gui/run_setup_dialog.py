# =============================================================================
#  bc_gui / run_setup_dialog.py  (plate)
# -----------------------------------------------------------------------------
#  First-window run setup: choose measurement MAT file(s), constitutive
#  model (material_type + analysis_type), and initial stiffness guesses.
#  Seeded from ``vfm_user_inputs.build_user_inputs()`` defaults so that
#  Continue-with-no-edits preserves the previous pipeline behaviour.
#
#  Called by ``run_vfm.py`` before ``launch_bc_editor``.
# =============================================================================

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from vfm_user_inputs import (
    ANALYSIS_TYPES,
    PARAMETER_LABELS_BY_MODEL,
    UserInputs,
    parameter_labels_for,
)

_MATERIAL_TYPES = ("iso", "orthotropic", "anisotropic")

#  Item data roles for the file list: the path, and the experiment it belongs
#  to.  The visible text is derived from both.
_PATH_ROLE = int(Qt.UserRole)
_EXPERIMENT_ROLE = int(Qt.UserRole) + 1

_MATERIAL_HELP = (
    "<b>Material type</b> selects the constitutive law:<br>"
    "<ul>"
    "<li><b>iso</b> — isotropic plate; unknowns reduce via "
    "A22=A11, A66=(A11−A12)/2 (and the same for D).</li>"
    "<li><b>orthotropic</b> — independent A11/A12/A22/A66 "
    "(or D equivalents); supported only for in_plane / out_of_plane.</li>"
    "<li><b>anisotropic</b> — full in-plane or bending coupling "
    "(adds A16/A26 or D16/D26); not supported with analysis "
    "<i>both</i>.</li>"
    "</ul>"
)

_ANALYSIS_HELP = (
    "<b>Analysis type</b> selects the physics scope:<br>"
    "<ul>"
    "<li><b>in_plane</b> — identify the A matrix only (membrane). "
    "Mesh has 2 DOFs/node [u, v]. MAT file must provide strains "
    "Eps1, Eps2, Eps6.</li>"
    "<li><b>out_of_plane</b> — identify the D matrix only (bending). "
    "Mesh has 4 DOFs/node [w, w_x, w_y, w_xy]. MAT file must provide "
    "curvatures K1, K2, K6.</li>"
    "<li><b>both</b> — full ABD (B = 0). Mesh has 6 DOFs/node. "
    "Requires both strain and curvature fields. Only available for "
    "<b>iso</b>.</li>"
    "</ul>"
)

_FILES_HELP = (
    "Select one or more experimental <b>.mat</b> measurement files. "
    "Order is preserved and becomes the stacked order for identification.<br><br>"
    "<b>Experiment #</b> groups the files for the generalized-least-squares "
    "noise model. Files sharing a number are <b>load steps of one test</b>: "
    "their measurement noise is allowed to be correlated across steps, and "
    "they must share a virtual mesh, a BC file and a measurement ROI. "
    "Different numbers are treated as <b>statistically independent "
    "experiments</b>. Leave everything at 1 if each file is its own test."
)

_PROJECT_HELP = (
    "A <b>.vfmproj.json</b> holds a whole run: every data file with its "
    "experiment number, its virtual mesh, its boundary conditions and its "
    "sample area, plus the constitutive model, the initial guess and the "
    "solver settings.<br>"
    "Loading one fills in everything below <i>and</i> supplies the boundary "
    "conditions, so the BC editor is skipped and the run reproduces the "
    "published result exactly. Build the campaign one with "
    "<code>python -m workflows.make_gui_project</code>."
)

_PARAMS_HELP = (
    "Initial guess for the unknown stiffness coefficients. "
    "Length and labels follow the selected (material, analysis) pair. "
    "Default is <b>1</b> on every unknown when you change the model."
)


@dataclass
class RunSetupResult:
    """User choices from the run-setup dialog."""
    data_files: List[Path]
    material_type: str
    analysis_type: str
    initial_parameters: List[float]
    #  One entry per data file.  Files sharing a number are load steps of one
    #  experiment and get a dense cross-time noise covariance in the GLS.
    experiment_ids: List[int] = field(default_factory=list)
    #  Set when the user loaded a .vfmproj.json.  When present it carries the
    #  virtual meshes, boundary conditions and sample areas for every test, so
    #  run_vfm.py can skip the BC editor entirely and reproduce a published
    #  run exactly.
    project: Optional[Any] = None
    project_path: Optional[Path] = None
    skip_bc_editor: bool = False


class RunSetupDialog(QDialog):
    """Modal dialog to configure data files and constitutive model."""

    def __init__(self, defaults: UserInputs, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("VFM run setup (plate)")
        self.resize(720, 640)
        self.setModal(True)

        self._defaults = defaults
        self._seed_material = str(defaults.material_type).strip().lower()
        self._seed_analysis = str(defaults.analysis_type).strip().lower()
        self._seed_parameters = [float(v) for v in defaults.initial_parameters]
        self._param_spins: List[QDoubleSpinBox] = []
        self._last_valid_analysis = self._seed_analysis
        self._project = None
        self._project_path: Optional[Path] = None

        root = QVBoxLayout(self)

        # ---- Project ----------------------------------------------------
        project_box = QGroupBox("Project file (optional)")
        project_layout = QVBoxLayout(project_box)
        project_help = QLabel(_PROJECT_HELP)
        project_help.setWordWrap(True)
        project_layout.addWidget(project_help)

        project_row = QHBoxLayout()
        load_project_btn = QPushButton("Load project…")
        load_project_btn.clicked.connect(self._load_project)
        clear_project_btn = QPushButton("Clear")
        clear_project_btn.clicked.connect(self._clear_project)
        project_row.addWidget(load_project_btn)
        project_row.addWidget(clear_project_btn)
        project_row.addStretch(1)
        project_layout.addLayout(project_row)

        self._project_label = QLabel("No project loaded — "
                                     "configure the run manually below.")
        self._project_label.setWordWrap(True)
        self._project_label.setStyleSheet("color: #52514e;")
        project_layout.addWidget(self._project_label)
        root.addWidget(project_box)

        # ---- Data files -------------------------------------------------
        files_box = QGroupBox("Measurement data files")
        files_layout = QVBoxLayout(files_box)
        files_help = QLabel(_FILES_HELP)
        files_help.setWordWrap(True)
        files_layout.addWidget(files_help)

        self._file_list = QListWidget()
        self._file_list.setSelectionMode(QListWidget.ExtendedSelection)
        seed_experiments = (list(defaults.experiment_ids)
                            if getattr(defaults, "experiment_ids", None)
                            else [1] * len(defaults.data_files))
        for path, experiment in zip(defaults.data_files, seed_experiments):
            self._add_file_item(path, experiment)
        files_layout.addWidget(self._file_list)

        file_buttons = QHBoxLayout()
        add_btn = QPushButton("Add files…")
        add_btn.clicked.connect(self._add_files)
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._remove_selected_files)
        up_btn = QPushButton("Move up")
        up_btn.clicked.connect(lambda: self._move_selected(-1))
        down_btn = QPushButton("Move down")
        down_btn.clicked.connect(lambda: self._move_selected(1))
        experiment_btn = QPushButton("Set experiment #…")
        experiment_btn.setToolTip(
            "Files sharing an experiment number are load steps of ONE test.\n"
            "The GLS gives them a dense cross-time noise covariance; separate\n"
            "experiments are assumed statistically independent.")
        experiment_btn.clicked.connect(self._set_experiment_for_selection)
        file_buttons.addWidget(add_btn)
        file_buttons.addWidget(remove_btn)
        file_buttons.addWidget(up_btn)
        file_buttons.addWidget(down_btn)
        file_buttons.addWidget(experiment_btn)
        file_buttons.addStretch(1)
        files_layout.addLayout(file_buttons)
        root.addWidget(files_box)

        # ---- Constitutive model ----------------------------------------
        model_box = QGroupBox("Constitutive model")
        model_layout = QVBoxLayout(model_box)

        form = QFormLayout()
        self._material_combo = QComboBox()
        for key in _MATERIAL_TYPES:
            self._material_combo.addItem(key, key)
        seed_mat_idx = self._material_combo.findData(self._seed_material)
        if seed_mat_idx >= 0:
            self._material_combo.setCurrentIndex(seed_mat_idx)
        form.addRow("Material type", self._material_combo)

        self._analysis_combo = QComboBox()
        for key in ANALYSIS_TYPES:
            self._analysis_combo.addItem(key, key)
        seed_an_idx = self._analysis_combo.findData(self._seed_analysis)
        if seed_an_idx >= 0:
            self._analysis_combo.setCurrentIndex(seed_an_idx)
        form.addRow("Analysis type", self._analysis_combo)
        model_layout.addLayout(form)

        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml(
            f"{_MATERIAL_HELP}<br>{_ANALYSIS_HELP}"
        )
        help_text.setMaximumHeight(180)
        model_layout.addWidget(help_text)
        root.addWidget(model_box)

        # ---- Initial parameters ----------------------------------------
        params_box = QGroupBox("Initial parameters")
        params_layout = QVBoxLayout(params_box)
        params_help = QLabel(_PARAMS_HELP)
        params_help.setWordWrap(True)
        params_layout.addWidget(params_help)

        self._params_form_host = QWidget()
        self._params_form = QFormLayout(self._params_form_host)
        params_layout.addWidget(self._params_form_host)
        root.addWidget(params_box)

        # ---- Buttons ----------------------------------------------------
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.button(QDialogButtonBox.Ok).setText("Continue")
        buttons.accepted.connect(self._on_continue)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._material_combo.currentIndexChanged.connect(
            self._on_model_changed)
        self._analysis_combo.currentIndexChanged.connect(
            self._on_model_changed)

        self._update_analysis_availability(reset_if_invalid=False)
        self._rebuild_parameter_fields(use_seed_if_matching=True)

    # ------------------------------------------------------------------
    #  File list helpers
    # ------------------------------------------------------------------

    def _default_browse_dir(self) -> str:
        if self._file_list.count() > 0:
            first = Path(self._file_list.item(0).text())
            if first.parent.is_dir():
                return str(first.parent)
        loading = Path(__file__).resolve().parent.parent / "Loading_Data"
        if loading.is_dir():
            return str(loading)
        return str(Path.cwd())

    def _add_file_item(self, path, experiment: int) -> None:
        """One row: the file, tagged with the experiment it belongs to."""
        item = QListWidgetItem()
        item.setData(_PATH_ROLE, str(Path(path)))
        item.setData(_EXPERIMENT_ROLE, int(experiment))
        self._refresh_item_text(item)
        self._file_list.addItem(item)

    @staticmethod
    def _refresh_item_text(item) -> None:
        item.setText(f"[exp {int(item.data(_EXPERIMENT_ROLE))}]  "
                     f"{item.data(_PATH_ROLE)}")

    def _add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select measurement MAT files",
            self._default_browse_dir(),
            "MATLAB data (*.mat);;All files (*)",
        )
        existing = {
            self._file_list.item(i).data(_PATH_ROLE)
            for i in range(self._file_list.count())
        }
        #  New files join the last experiment in the list, which is what you
        #  want when adding load steps to a test one at a time.
        experiment = max(self._current_experiment_ids(), default=1)
        for path in paths:
            text = str(Path(path))
            if text not in existing:
                self._add_file_item(text, experiment)
                existing.add(text)

    def _load_project(self) -> None:
        """Fill the whole dialog from a .vfmproj.json."""
        from vfm_project import describe_project, load_project

        start = str(self._project_path.parent) if self._project_path \
            else self._default_browse_dir()
        path, _ = QFileDialog.getOpenFileName(
            self, "Load VFM project", start,
            "VFM project (*.json);;All files (*)")
        if not path:
            return

        try:
            project = load_project(Path(path))
        except (ValueError, FileNotFoundError, OSError) as error:
            QMessageBox.critical(self, "Could not load project", str(error))
            return

        missing = [t.data_file for t in project.tests
                   if not t.data_file.is_file()]
        if missing:
            QMessageBox.critical(
                self, "Missing measurement files",
                "The project references files that are not on disk:\n\n"
                + "\n".join(str(p) for p in missing[:8])
                + ("\n…" if len(missing) > 8 else ""))
            return

        without_bc = [t.label or t.data_file.name for t in project.tests
                      if not t.external and not t.prescribed]
        if without_bc:
            QMessageBox.warning(
                self, "Project has tests without boundary conditions",
                "These tests carry no PRESCRIBED or EXTERNAL entries, so the "
                "BC editor will still open for them:\n\n"
                + "\n".join(without_bc[:8]))

        self._project = project
        self._project_path = Path(path)

        self._file_list.clear()
        for test in project.tests:
            self._add_file_item(test.data_file, test.experiment_id)

        #  Seed FIRST: changing the combos fires the rebuild handler, which
        #  reads these.  Then rebuild explicitly so the project's initial
        #  guess survives regardless of signal order.
        self._seed_material = project.material_type
        self._seed_analysis = project.analysis_type
        self._seed_parameters = [float(v) for v in project.initial_parameters]

        index = self._material_combo.findData(project.material_type)
        if index >= 0:
            self._material_combo.setCurrentIndex(index)
        index = self._analysis_combo.findData(project.analysis_type)
        if index >= 0:
            self._analysis_combo.setCurrentIndex(index)
        self._rebuild_parameter_fields(use_seed_if_matching=True)

        self._project_label.setText(
            f"<b>{self._project_path.name}</b> — {describe_project(project)}"
            "<br>Boundary conditions and virtual meshes come from the "
            "project; the BC editor will be skipped.")
        self._project_label.setStyleSheet("color: #12805a;")

    def _clear_project(self) -> None:
        """Go back to configuring the run by hand."""
        self._project = None
        self._project_path = None
        self._project_label.setText("No project loaded — "
                                    "configure the run manually below.")
        self._project_label.setStyleSheet("color: #52514e;")

    def _set_experiment_for_selection(self) -> None:
        items = self._file_list.selectedItems()
        if not items:
            QMessageBox.information(
                self, "No files selected",
                "Select the load steps that belong to one experiment, then "
                "set their experiment number.")
            return
        current = int(items[0].data(_EXPERIMENT_ROLE))
        value, accepted = QInputDialog.getInt(
            self, "Experiment number",
            "All files sharing a number are LOAD STEPS of one test.\n"
            "They get a dense cross-time noise covariance, and must share a\n"
            "virtual mesh, a BC file and a measurement ROI.\n\n"
            "Different numbers are treated as independent experiments.",
            current, 1, 999, 1)
        if not accepted:
            return
        for item in items:
            item.setData(_EXPERIMENT_ROLE, int(value))
            self._refresh_item_text(item)

    def _remove_selected_files(self) -> None:
        for item in self._file_list.selectedItems():
            row = self._file_list.row(item)
            self._file_list.takeItem(row)

    def _move_selected(self, direction: int) -> None:
        rows = sorted(
            {self._file_list.row(item)
             for item in self._file_list.selectedItems()}
        )
        if not rows:
            return
        if direction < 0:
            if rows[0] == 0:
                return
            for row in rows:
                item = self._file_list.takeItem(row)
                self._file_list.insertItem(row - 1, item)
                item.setSelected(True)
        else:
            if rows[-1] >= self._file_list.count() - 1:
                return
            for row in reversed(rows):
                item = self._file_list.takeItem(row)
                self._file_list.insertItem(row + 1, item)
                item.setSelected(True)

    def _current_data_files(self) -> List[Path]:
        return [
            Path(self._file_list.item(i).data(_PATH_ROLE))
            for i in range(self._file_list.count())
        ]

    def _current_experiment_ids(self) -> List[int]:
        return [
            int(self._file_list.item(i).data(_EXPERIMENT_ROLE))
            for i in range(self._file_list.count())
        ]

    # ------------------------------------------------------------------
    #  Model / parameter helpers
    # ------------------------------------------------------------------

    def _current_material(self) -> str:
        return str(self._material_combo.currentData())

    def _current_analysis(self) -> str:
        return str(self._analysis_combo.currentData())

    def _update_analysis_availability(
            self, *, reset_if_invalid: bool) -> None:
        material = self._current_material()
        both_allowed = material == "iso"
        both_idx = self._analysis_combo.findData("both")
        if both_idx >= 0:
            model = self._analysis_combo.model()
            if isinstance(model, QStandardItemModel):
                item = model.item(both_idx)
                if item is not None:
                    flags = item.flags()
                    if both_allowed:
                        item.setFlags(flags | Qt.ItemIsEnabled)
                    else:
                        item.setFlags(flags & ~Qt.ItemIsEnabled)

        current = self._current_analysis()
        key = (material, current)
        if key not in PARAMETER_LABELS_BY_MODEL and reset_if_invalid:
            fallback = self._last_valid_analysis
            if (material, fallback) not in PARAMETER_LABELS_BY_MODEL:
                fallback = "out_of_plane"
            idx = self._analysis_combo.findData(fallback)
            if idx >= 0:
                self._analysis_combo.blockSignals(True)
                self._analysis_combo.setCurrentIndex(idx)
                self._analysis_combo.blockSignals(False)

        current = self._current_analysis()
        if (material, current) in PARAMETER_LABELS_BY_MODEL:
            self._last_valid_analysis = current

    def _on_model_changed(self) -> None:
        self._update_analysis_availability(reset_if_invalid=True)
        self._rebuild_parameter_fields(use_seed_if_matching=True)

    def _clear_parameter_fields(self) -> None:
        while self._params_form.rowCount():
            self._params_form.removeRow(0)
        self._param_spins = []

    def _rebuild_parameter_fields(
            self, *, use_seed_if_matching: bool) -> None:
        material = self._current_material()
        analysis = self._current_analysis()
        key = (material, analysis)
        if key not in PARAMETER_LABELS_BY_MODEL:
            self._clear_parameter_fields()
            return

        labels = parameter_labels_for(material, analysis)
        if (use_seed_if_matching
                and material == self._seed_material
                and analysis == self._seed_analysis
                and len(self._seed_parameters) == len(labels)):
            values = list(self._seed_parameters)
        else:
            values = [1.0] * len(labels)

        self._clear_parameter_fields()
        for label, value in zip(labels, values):
            spin = QDoubleSpinBox()
            spin.setRange(-1.0e30, 1.0e30)
            spin.setDecimals(8)
            spin.setValue(float(value))
            self._params_form.addRow(label, spin)
            self._param_spins.append(spin)

    def _current_parameters(self) -> List[float]:
        return [float(spin.value()) for spin in self._param_spins]

    def _on_continue(self) -> None:
        data_files = self._current_data_files()
        if not data_files:
            QMessageBox.warning(
                self,
                "Missing data files",
                "Select at least one measurement .mat file before continuing.",
            )
            return

        material = self._current_material()
        analysis = self._current_analysis()
        key = (material, analysis)
        if key not in PARAMETER_LABELS_BY_MODEL:
            QMessageBox.warning(
                self,
                "Unsupported model",
                f"The combination material_type={material!r}, "
                f"analysis_type={analysis!r} is not supported.\n"
                "Orthotropic and anisotropic require in_plane or "
                "out_of_plane (not both).",
            )
            return

        labels = parameter_labels_for(material, analysis)
        params = self._current_parameters()
        if len(params) != len(labels):
            QMessageBox.warning(
                self,
                "Invalid initial parameters",
                f"Expected {len(labels)} initial parameters "
                f"({', '.join(labels)}), got {len(params)}.",
            )
            return

        self.accept()

    def result_values(self) -> RunSetupResult:
        return RunSetupResult(
            data_files=self._current_data_files(),
            material_type=self._current_material(),
            analysis_type=self._current_analysis(),
            initial_parameters=self._current_parameters(),
            experiment_ids=self._current_experiment_ids(),
            project=self._project,
            project_path=self._project_path,
            #  Only skip the editor when every test actually carries BCs;
            #  a half-populated project must not silently run with none.
            skip_bc_editor=bool(
                self._project is not None
                and all(t.external or t.prescribed
                        for t in self._project.tests)),
        )


def launch_run_setup(defaults: UserInputs) -> Optional[RunSetupResult]:
    """Show the run-setup dialog; return choices or ``None`` on cancel.

    Creates a ``QApplication`` if none exists, but does not quit it, so
    the subsequent BC editor can reuse the same Qt event loop.
    """
    existing_app = QApplication.instance()
    if existing_app is None:
        QApplication(sys.argv)

    dialog = RunSetupDialog(defaults)
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    accepted = dialog.exec() == QDialog.Accepted
    if not accepted:
        return None
    return dialog.result_values()
