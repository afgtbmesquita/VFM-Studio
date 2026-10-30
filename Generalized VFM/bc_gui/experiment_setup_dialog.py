# =============================================================================
#  bc_gui / experiment_setup_dialog.py
# -----------------------------------------------------------------------------
#  One dialog for the whole run, organised the way the method actually is:
#
#      EXPERIMENT  -- one physical test.  Owns the boundary conditions, the
#                     virtual mesh and the sample area.
#        TIME STEP -- one load step of that test.  Owns a .mat file and the
#                     applied force/moment on each external group.
#
#  That two-level structure is not a presentation choice.  It is what the GLS
#  requires:
#
#    * The boundary-condition topology must be the same at every time step.
#      The optimal virtual field is built to annihilate specific unknown
#      boundary terms; change the constraint set between steps and the steps
#      are no longer measuring the same quantity.
#    * The virtual mesh must be the same at every time step, because
#      C_{i,ts} = U_t^T G(Gamma_ts) U_s couples the virtual fields of steps t
#      and s.  Two meshes with different DOF counts make that block undefined.
#    * Only the applied load may vary between steps -- which is the entire
#      point of having several of them.
#
#  So the UI offers exactly that freedom and no more: BCs and mesh are edited
#  ONCE per experiment, loads are edited PER time step.  A user cannot build an
#  inconsistent run by accident, which is the main thing a published tool
#  should guarantee.
# =============================================================================

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from vfm_project import (
    ProjectTest,
    VfmProject,
    apply_experiment_definition,
    check_experiment_consistency,
    check_load_orientation,
    describe_project,
    group_by_experiment,
    load_project,
    save_project,
)
from vfm_user_inputs import (
    ANALYSIS_TYPES,
    PARAMETER_LABELS_BY_MODEL,
    parameter_labels_for,
)

from bc_gui.noise_model_help import show_noise_model_help

_MATERIAL_TYPES = ("iso", "orthotropic", "anisotropic")


def _format_load(value) -> str:
    """How a per-step applied load is DISPLAYED in the time-step table.

    Deliberately rounded so the table stays readable.  The exact float rides
    along in the item's ``Qt.UserRole`` and is what actually reaches the
    solver -- see ``commit_step_loads``.
    """
    return "" if value is None else f"{float(value):g}"

_INTRO = (
    "<b>An experiment is one physical test; its time steps are that test's "
    "load steps.</b><br>"
    "Boundary conditions and the virtual mesh belong to the <i>experiment</i> "
    "and are shared by all of its time steps — the cross-time covariance "
    "couples the virtual fields of different steps, so they must live on the "
    "same mesh and the same constraint set. Only the applied force/moment may "
    "change from step to step."
)


# =============================================================================
#  One experiment
# =============================================================================

class ExperimentTab(QWidget):
    """Time steps, mesh, BCs and per-step loads for a single experiment."""

    def __init__(self, experiment_id: int, analysis_type: str,
                 parent_dialog: "ExperimentSetupDialog") -> None:
        super().__init__()
        self.experiment_id = experiment_id
        self._dialog = parent_dialog
        self._analysis_type = analysis_type

        #  The experiment-level definition, shared by every time step.
        self.prescribed: List[Dict] = []
        self.linking: List[Dict] = []
        self.external: List[Dict] = []
        self.mesh_settings: Optional[Dict] = None
        self.sample_area_total: Optional[float] = None
        self.material_orientation_deg: float = 0.0
        #  Per-time-step: data file, and one magnitude per external group.
        self.tests: List[ProjectTest] = []

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit(f"Experiment {experiment_id}")
        self.name_edit.setMaximumWidth(280)
        name_row.addWidget(self.name_edit)
        name_row.addStretch(1)
        layout.addLayout(name_row)

        layout.addWidget(self._build_steps_box(), 1)
        layout.addWidget(self._build_definition_box())

    # ------------------------------------------------------------------
    def _build_steps_box(self) -> QGroupBox:
        box = QGroupBox("Time steps  (load steps of this experiment)")
        layout = QVBoxLayout(box)

        buttons = QHBoxLayout()
        add = QPushButton("Add .mat files…")
        add.clicked.connect(self._add_files)
        remove = QPushButton("Remove selected")
        remove.clicked.connect(self._remove_selected)
        up = QPushButton("Move up")
        up.clicked.connect(lambda: self._move(-1))
        down = QPushButton("Move down")
        down.clicked.connect(lambda: self._move(1))
        for widget in (add, remove, up, down):
            buttons.addWidget(widget)
        buttons.addStretch(1)
        self.count_label = QLabel("0 time steps")
        self.count_label.setStyleSheet("color: #52514e;")
        buttons.addWidget(self.count_label)
        layout.addLayout(buttons)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["#", "measurement file"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.itemChanged.connect(self._on_load_cell_changed)
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch)
        layout.addWidget(self.table)

        #  Where this experiment's applied load comes from.  Without this the
        #  per-step columns below are read, stored -- and then ignored at solve
        #  time on any dataset that also ships an F field.
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Applied load from:"))
        self.load_source_combo = QComboBox()
        self.load_source_combo.addItem(
            "the .mat file's F field", "mat")
        self.load_source_combo.addItem(
            "the values typed below (one per step, per group)", "manual")
        self.load_source_combo.setToolTip(
            "<p><b>.mat F field</b> — each step's load is read from the "
            "<code>F</code> vector inside its own measurement file. Anything "
            "typed in the table is ignored.</p>"
            "<p><b>values typed below</b> — the table wins. Every cell must be "
            "filled: a half-filled table would mix a typed load on one edge "
            "with the file's on another, which is a different experiment from "
            "the one you meant to describe.</p>"
            "<p>This is a per-experiment choice; one experiment takes its "
            "loads from one place.</p>"
        )
        self.load_source_combo.setToolTipDuration(60000)
        self.load_source_combo.currentIndexChanged.connect(
            self._on_load_source_changed)
        source_row.addWidget(self.load_source_combo, 1)
        layout.addLayout(source_row)

        self.load_note = QLabel(
            "Applied force/moment per step: the .mat file's <code>F</code> "
            "field is used when present. Columns appear here once the "
            "experiment has EXTERNAL groups.")
        self.load_note.setWordWrap(True)
        self.load_note.setStyleSheet("color: #52514e;")
        layout.addWidget(self.load_note)
        return box

    @property
    def applied_load_source(self) -> str:
        return str(self.load_source_combo.currentData() or "mat")

    def _set_applied_load_source(self, source: str) -> None:
        index = self.load_source_combo.findData(
            "manual" if str(source) == "manual" else "mat")
        if index >= 0:
            self.load_source_combo.blockSignals(True)
            self.load_source_combo.setCurrentIndex(index)
            self.load_source_combo.blockSignals(False)

    def _on_load_source_changed(self, _index: int) -> None:
        #  Redraw so the columns switch between editable and greyed, and the
        #  note explains which source is live.
        self.commit_step_loads()
        self.refresh()

    def _build_definition_box(self) -> QGroupBox:
        box = QGroupBox("Shared by every time step of this experiment")
        layout = QVBoxLayout(box)

        row = QHBoxLayout()
        self.bc_summary = QLabel("Boundary conditions: none defined")
        self.bc_summary.setWordWrap(True)
        row.addWidget(self.bc_summary, 1)
        edit = QPushButton("Edit boundary conditions && mesh…")
        edit.clicked.connect(self._edit_definition)
        row.addWidget(edit)
        layout.addLayout(row)

        self.mesh_summary = QLabel("Virtual mesh: none defined")
        self.mesh_summary.setWordWrap(True)
        self.mesh_summary.setStyleSheet("color: #52514e;")
        layout.addWidget(self.mesh_summary)

        #  The frame this experiment's fields were measured in, relative to the
        #  material frame the identified D lives in.  Editable here and in the
        #  BC editor's Solver Settings tab; the two are kept in sync through
        #  self.material_orientation_deg.
        orientation_row = QHBoxLayout()
        orientation_row.addWidget(QLabel("Material frame orientation:"))
        self.orientation_spin = QDoubleSpinBox()
        self.orientation_spin.setRange(-180.0, 180.0)
        self.orientation_spin.setDecimals(3)
        self.orientation_spin.setSingleStep(5.0)
        self.orientation_spin.setSuffix(" deg")
        self.orientation_spin.setValue(0.0)
        self.orientation_spin.setMaximumWidth(140)
        self.orientation_spin.setToolTip(
            "<p>Angle from this experiment's measurement frame (+<i>x</i>) to "
            "the material 1-axis, counter-clockwise positive.</p>"
            "<p>The fields, mesh, boundary conditions and applied load all "
            "stay in the measurement frame; the <b>D matrix</b> is rotated by "
            "this angle instead. Do <b>not</b> also rotate the fields.</p>"
            "<p>0 = this coupon was measured in the material frame. Open the "
            "BC editor to see the axes drawn on the mesh.</p>"
        )
        self.orientation_spin.setToolTipDuration(60000)
        self.orientation_spin.valueChanged.connect(
            self._on_orientation_changed)
        orientation_row.addWidget(self.orientation_spin)
        orientation_row.addStretch(1)
        layout.addLayout(orientation_row)
        return box

    def _on_orientation_changed(self, value: float) -> None:
        self.material_orientation_deg = float(value)
        self._refresh_summaries()

    # ------------------------------------------------------------------
    #  Time steps
    # ------------------------------------------------------------------
    def _add_files(self) -> None:
        self.commit_step_loads()
        paths, _ = QFileDialog.getOpenFileNames(
            self, f"Time steps for {self.name_edit.text()}",
            self._dialog.browse_dir(), "MATLAB data (*.mat);;All files (*)")
        existing = {t.data_file for t in self.tests}
        for path in paths:
            resolved = Path(path).resolve()
            if resolved in existing:
                continue
            self.tests.append(ProjectTest(
                data_file=resolved, experiment_id=self.experiment_id,
                label=f"{self.name_edit.text()}:{resolved.stem}"))
            existing.add(resolved)
        self.refresh()

    def _remove_selected(self) -> None:
        self.commit_step_loads()
        rows = sorted({index.row()
                       for index in self.table.selectedIndexes()},
                      reverse=True)
        for row in rows:
            if 0 <= row < len(self.tests):
                self.tests.pop(row)
        self.refresh()

    def _move(self, direction: int) -> None:
        self.commit_step_loads()
        rows = sorted({index.row() for index in self.table.selectedIndexes()})
        if not rows:
            return
        if direction < 0 and rows[0] == 0:
            return
        if direction > 0 and rows[-1] >= len(self.tests) - 1:
            return
        for row in (rows if direction < 0 else reversed(rows)):
            target = row + direction
            self.tests[row], self.tests[target] = \
                self.tests[target], self.tests[row]
        self.refresh()
        self.table.clearSelection()
        for row in rows:
            self.table.selectRow(row + direction)

    def refresh(self) -> None:
        """Redraw the step table, including one column per external group."""
        manual = self.applied_load_source == "manual"
        group_names = [str(entry.get("label", f"group {i + 1}"))
                       for i, entry in enumerate(self.external)]
        suffix = "  [load]" if manual else "  (from .mat F)"
        headers = ["#", "measurement file"] + [f"{n}{suffix}"
                                               for n in group_names]
        self.table.blockSignals(True)
        self.table.clear()
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(self.tests))
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch)

        for row, test in enumerate(self.tests):
            index_item = QTableWidgetItem(str(row + 1))
            index_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table.setItem(row, 0, index_item)

            name_item = QTableWidgetItem(test.data_file.name)
            name_item.setToolTip(str(test.data_file))
            name_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if not test.data_file.is_file():
                name_item.setForeground(Qt.red)
                name_item.setToolTip(f"MISSING: {test.data_file}")
            self.table.setItem(row, 1, name_item)

            for column, _ in enumerate(group_names):
                value = None
                if column < len(test.external):
                    value = test.external[column].get("value")
                item = QTableWidgetItem(_format_load(value))
                #  The DISPLAY is rounded for readability, so parsing the text
                #  back would quietly truncate the load -- 6 significant digits
                #  costs ~2e-6 relative on every redraw, and the table is
                #  redrawn constantly.  Keep the exact float alongside it and
                #  prefer it whenever the user has not actually typed.
                if value is not None:
                    item.setData(Qt.UserRole, float(value))
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if manual:
                    item.setToolTip(
                        "This experiment takes its applied load from this "
                        "table, so this cell IS the load for this step and "
                        "group. It must not be blank.")
                    if value is None:
                        #  A blank cell in manual mode is the one thing that
                        #  will stop the run, so it is shown, not just
                        #  reported when Run is pressed.
                        item.setBackground(QColor("#ffe0e0"))
                        item.setToolTip(
                            "Required: this experiment takes its applied load "
                            "from the table, and this cell is empty.")
                else:
                    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    item.setForeground(QColor("#9a9a9a"))
                    item.setToolTip(
                        "Not in use: this experiment reads its load from the "
                        ".mat file's F field. Switch 'Applied load from' to "
                        "the typed values to make this column live.")
                self.table.setItem(row, 2 + column, item)
        self.table.blockSignals(False)

        self.count_label.setText(f"{len(self.tests)} time step"
                                 + ("" if len(self.tests) == 1 else "s"))
        if not group_names:
            self.load_note.setText(
                "Applied force/moment per step: one column appears here for "
                "each EXTERNAL group, once this experiment has boundary "
                "conditions defined.")
        elif manual:
            blanks = sum(
                1 for test in self.tests
                for column in range(len(group_names))
                if column >= len(test.external)
                or test.external[column].get("value") is None)
            self.load_note.setText(
                f"<b>The table is live.</b> Every cell is this step's applied "
                f"load for that group; the .mat <code>F</code> field is "
                f"ignored."
                + ("" if not blanks
                   else f" <span style='color:#b00020;'><b>{blanks} cell(s) "
                        f"still blank</b> — the run will be refused until they "
                        f"are filled.</span>"))
        else:
            self.load_note.setText(
                "Each step's load is read from the <code>F</code> field of its "
                "own .mat file. The columns below are shown greyed because "
                "they are not in use.")
        self.load_note.setVisible(True)
        self._refresh_summaries()

    def _on_load_cell_changed(self, item: QTableWidgetItem) -> None:
        """Persist a load edit immediately so later redraws are lossless."""
        row = item.row()
        group = item.column() - 2
        if row < 0 or row >= len(self.tests) or group < 0:
            return
        test = self.tests[row]
        if group >= len(test.external):
            return

        def decorate(color: str, tooltip: str) -> None:
            blocked = self.table.blockSignals(True)
            try:
                item.setBackground(QColor(color))
                item.setToolTip(tooltip)
            finally:
                self.table.blockSignals(blocked)

        text = item.text().strip()
        if not text:
            test.external[group].pop("value", None)
            decorate(
                "#ffe0e0",
                "Required in manual mode: enter this step's applied force "
                "or moment for this external-work group.")
            return

        try:
            value = float(text)
        except ValueError:
            value = float("nan")
        if not math.isfinite(value):
            test.external[group].pop("value", None)
            decorate(
                "#ffe0e0",
                "Invalid load. Enter a finite number, for example 1250 or "
                "-3.5e4.")
            return

        test.external[group]["value"] = value
        #  The user typed this, so it IS the exact value now; refresh the
        #  shadow copy or the next redraw would restore the old one.
        blocked = self.table.blockSignals(True)
        try:
            item.setData(Qt.UserRole, value)
        finally:
            self.table.blockSignals(blocked)
        decorate(
            "#ffffff",
            "Applied force or moment for this load step and external-work "
            "group.")

    def _refresh_summaries(self) -> None:
        if self.prescribed or self.external or self.linking:
            summary = (f"<b>Boundary conditions:</b> {len(self.prescribed)} "
                       f"prescribed, {len(self.linking)} linking, "
                       f"{len(self.external)} external")
            if self.external:
                groups = ", ".join(str(entry.get("label", "?"))
                                   for entry in self.external)
                summary += f" ({groups})"
            self.bc_summary.setText(summary)
        else:
            self.bc_summary.setText(
                "<b>Boundary conditions:</b> none defined — "
                "click Edit to set them up.")
        if self.mesh_settings:
            bounds = self.mesh_settings.get("bounds", {})
            self.mesh_summary.setText(
                f"Virtual mesh: "
                f"{self.mesh_settings.get('number_of_elements_x')} x "
                f"{self.mesh_settings.get('number_of_elements_y')} elements, "
                f"x [{bounds.get('x_min', 0):.3f}, {bounds.get('x_max', 0):.3f}]"
                f", y [{bounds.get('y_min', 0):.3f}, "
                f"{bounds.get('y_max', 0):.3f}]"
                + (f", sample area {self.sample_area_total:.3f} mm² (typed)"
                   if self.sample_area_total
                   else ", sample area auto-detected from the data")
                + (f", material frame {self.material_orientation_deg:+.1f}°"
                   if self.material_orientation_deg else ""))
        else:
            self.mesh_summary.setText("Virtual mesh: none defined")

    def commit_step_loads(self) -> None:
        """Read the per-step magnitudes back out of the table."""
        for row, test in enumerate(self.tests):
            #  The orientation is experiment-level, so it lands on every step.
            #  Done here rather than only in _edit_definition, because the user
            #  may set the angle from this tab without opening the BC editor.
            test.material_orientation_deg = self.material_orientation_deg
            test.applied_load_source = self.applied_load_source
            for column in range(len(self.external)):
                item = self.table.item(row, 2 + column)
                if item is None or column >= len(test.external):
                    continue
                text = item.text().strip()
                if not text:
                    test.external[column].pop("value", None)
                    continue
                #  Untouched cell: the text is a ROUNDED rendering of the
                #  stored value, so re-parsing it would lose precision the
                #  user never asked to lose.  Only trust the text when it no
                #  longer matches what we drew.
                stored = item.data(Qt.UserRole)
                if stored is not None and _format_load(stored) == text:
                    test.external[column]["value"] = float(stored)
                    continue
                try:
                    test.external[column]["value"] = float(text)
                except ValueError:
                    test.external[column].pop("value", None)

    # ------------------------------------------------------------------
    #  BC + mesh editor, once per experiment
    # ------------------------------------------------------------------
    def _edit_definition(self) -> None:
        self.commit_step_loads()
        if not self.tests:
            QMessageBox.information(
                self, "Add a time step first",
                "Add at least one .mat file so the editor can show the "
                "measurement cloud and fit the virtual mesh to it.")
            return

        from .app import launch_bc_editor
        from .bc_model import BcDocument, coerce_mesh_settings

        seed = None
        if self.prescribed or self.external or self.linking:
            seed = BcDocument.from_payload(
                prescribed=self.prescribed, linking=self.linking,
                external=self.external,
                mesh_settings=self.mesh_settings,
                solver_settings=(
                    {"sample_area_total": self.sample_area_total}
                    if self.sample_area_total else None),
                analysis_type=self._analysis_type,
                description=f"{self.name_edit.text()} (all time steps)")

        result = launch_bc_editor(
            mesh_pkg=None,
            mesh_editor_seed_settings=coerce_mesh_settings(self.mesh_settings),
            initial_bc_file=None,
            initial_document=seed,
            output_dir=self._dialog.output_dir,
            #  ONE representative step: the topology is shared, so editing it
            #  against the first step's cloud is enough and avoids implying
            #  that each step could have its own.
            data_file=self.tests[0].data_file,
            data_files=[self.tests[0].data_file],
            analysis_type=self._analysis_type,
            initial_material_orientation_deg=self.material_orientation_deg,
            external_values_managed_by_parent=True,
            window_title=(f"Boundary conditions — {self.name_edit.text()} "
                          f"(applies to all {len(self.tests)} time steps)"),
            #  This dialog is application-modal; without a parent the editor
            #  would open behind it and refuse every click.
            parent=self._dialog,
        )
        if result is None or not getattr(result, "test_results", None):
            return
        first = result.test_results[0]
        document = getattr(first, "document", None)
        if document is None:
            return

        from .bc_model import mesh_settings_to_payload
        mesh_payload = mesh_settings_to_payload(
            first.mesh_settings or document.mesh_settings)
        area = first.sample_area_total
        if area is None and document.solver_settings:
            area = document.solver_settings.get("sample_area_total")

        returned_external = [e.to_python_dict() for e in document.external]
        #  Losing the EXTERNAL groups silently is the one edit that produces a
        #  run with no right-hand side, and it used to surface much later as a
        #  confusing size-mismatch from the solver.  If the editor comes back
        #  with none but this experiment had some, ask before discarding them.
        if self.external and not returned_external:
            keep = QMessageBox.question(
                self, "Keep the external work groups?",
                f"The boundary-condition editor returned no EXTERNAL work "
                f"groups, but this experiment currently has "
                f"{len(self.external)}.\n\n"
                "The applied moment does its virtual work through an EXTERNAL "
                "group. Without one the run has no right-hand side and cannot "
                "identify anything.\n\n"
                "Keep the existing groups?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if keep == QMessageBox.Yes:
                returned_external = self.external

        self.prescribed = [e.to_python_dict() for e in document.prescribed]
        self.linking = [e.to_python_dict() for e in document.linking]
        self.external = returned_external
        if mesh_payload:
            self.mesh_settings = mesh_payload
        if area is not None:
            self.sample_area_total = float(area)

        #  The BC editor owns the same setting; whatever the user left it on
        #  in there wins, and the tab's own spin box follows it.
        orientation = getattr(first, "material_orientation_deg", None)
        if orientation is None:
            orientation = result.material_orientation_deg
        if orientation is not None:
            self.material_orientation_deg = float(orientation)
            self.orientation_spin.blockSignals(True)
            self.orientation_spin.setValue(self.material_orientation_deg)
            self.orientation_spin.blockSignals(False)

        #  Push onto every time step, keeping any magnitudes already typed.
        #  One shared helper so the GUI and a headless caller cannot apply the
        #  rule differently.
        holder = VfmProject(tests=self.tests)
        apply_experiment_definition(
            holder, self.experiment_id,
            prescribed=self.prescribed, linking=self.linking,
            external=self.external, mesh_settings=self.mesh_settings,
            sample_area_total=self.sample_area_total,
            material_orientation_deg=self.material_orientation_deg,
            applied_load_source=self.applied_load_source)
        self.refresh()

    # ------------------------------------------------------------------
    def adopt(self, tests: List[ProjectTest], name: str) -> None:
        """Populate from a loaded project."""
        self.tests = list(tests)
        self.name_edit.setText(name)
        if tests:
            reference = tests[0]
            self.prescribed = [dict(e) for e in reference.prescribed]
            self.linking = [dict(e) for e in reference.linking]
            self.external = [{k: v for k, v in e.items() if k != "value"}
                             for e in reference.external]
            self.mesh_settings = (dict(reference.mesh_settings)
                                  if reference.mesh_settings else None)
            self.sample_area_total = reference.sample_area_total
            self.material_orientation_deg = float(
                reference.material_orientation_deg)
            self.orientation_spin.blockSignals(True)
            self.orientation_spin.setValue(self.material_orientation_deg)
            self.orientation_spin.blockSignals(False)
            self._set_applied_load_source(reference.applied_load_source)
        self.refresh()


# =============================================================================
#  The dialog
# =============================================================================

class ExperimentSetupDialog(QDialog):
    """Number of experiments, their time steps, and the constitutive model."""

    def __init__(self, defaults, output_dir: Path, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("VFM run setup")
        self.resize(940, 780)
        self.setModal(True)
        self.output_dir = Path(output_dir)
        self._project_path: Optional[Path] = None
        self._param_spins: List[QDoubleSpinBox] = []
        self._tabs_by_id: Dict[int, ExperimentTab] = {}

        root = QVBoxLayout(self)

        intro = QLabel(_INTRO)
        intro.setWordWrap(True)
        intro.setStyleSheet(
            "background:#f6f5f1; border:1px solid #dcdbd5; border-radius:4px; "
            "padding:8px;")
        root.addWidget(intro)

        # ---- experiments ------------------------------------------------
        header = QHBoxLayout()
        header.addWidget(QLabel("Number of experiments:"))
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 99)
        self.count_spin.setValue(1)
        self.count_spin.valueChanged.connect(self._sync_experiment_count)
        header.addWidget(self.count_spin)
        header.addSpacing(18)
        load = QPushButton("Load project…")
        load.clicked.connect(self._load_project)
        save = QPushButton("Save project…")
        save.clicked.connect(self._save_project)
        header.addWidget(load)
        header.addWidget(save)
        header.addStretch(1)
        root.addLayout(header)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)

        # ---- model ------------------------------------------------------
        root.addWidget(self._build_model_box())

        self.status = QLabel("")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok
                                   | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Run identification")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._seed_from_defaults(defaults)

    # ------------------------------------------------------------------
    def _build_model_box(self) -> QGroupBox:
        box = QGroupBox("Constitutive model and solver")
        outer = QHBoxLayout(box)

        form = QFormLayout()
        self.material_combo = QComboBox()
        for key in _MATERIAL_TYPES:
            self.material_combo.addItem(key, key)
        self.material_combo.currentIndexChanged.connect(self._rebuild_params)
        form.addRow("Material", self.material_combo)

        self.analysis_combo = QComboBox()
        for key in ANALYSIS_TYPES:
            self.analysis_combo.addItem(key, key)
        self.analysis_combo.currentIndexChanged.connect(self._rebuild_params)
        form.addRow("Analysis", self.analysis_combo)

        help_row = QHBoxLayout()
        help_row.addWidget(QLabel("<b>Noise model and weighting (Γ)</b>"))
        help_row.addStretch(1)
        self.noise_help_btn = QToolButton()
        self.noise_help_btn.setText("?")
        self.noise_help_btn.setToolTip(
            "Explain the noise model: how Γ is built, what each setting "
            "below does, and what happens when you change them.")
        self.noise_help_btn.setFixedWidth(24)
        self.noise_help_btn.clicked.connect(
            lambda: show_noise_model_help(self))
        help_row.addWidget(self.noise_help_btn)
        form.addRow(help_row)

        self.basis_spin = QSpinBox()
        self.basis_spin.setRange(0, 20)
        self.basis_spin.setValue(6)
        self.basis_spin.setToolTip(
            "Degree of the smooth spatial basis used to separate the physical "
            "field from the measurement fluctuation when estimating Gamma. "
            "Not part of the VFM; sweep it to check it is not carrying the "
            "result.")
        form.addRow("Noise basis degree", self.basis_spin)

        self.spatial_check = QCheckBox(
            "apply the measured spatial correlation (kappa)")
        self.spatial_check.setChecked(False)
        self.spatial_check.setToolTip(
            "OFF reproduces eq. (3) exactly, which is what is reported. ON is "
            "a labelled sensitivity check only: kappa approximates eq. (1) "
            "and should not set a published error bar.")
        form.addRow("Spatial correction", self.spatial_check)

        #  --- two-axis noise weighting -----------------------------------
        #  Because H_t = I, every load step already gives a complete estimate
        #  of all parameters, so identification is "combine N estimates of the
        #  same vector".  Load steps of one test are repeat measurements of
        #  the SAME specimen and are 0.96-0.999 correlated, so a GLS
        #  combination of them DIFFERENCES them, leaning entirely on a
        #  cross-step covariance estimated from the same data.  Separate tests
        #  are independent and carry different information, so a GLS
        #  combination of THEM is ordinary inverse-variance weighting.  One
        #  rule cannot be right for both, hence two controls.
        self.load_step_combo = QComboBox()
        for text, key in (
            ("Single rule below (legacy)", None),
            ("OLS - plain average  (recommended)", "ols"),
            ("GLS - full cross-step covariance", "gls"),
            ("Block-diagonal - per step only", "block_diagonal"),
            ("Diagonal - per step, per parameter", "diagonal"),
        ):
            self.load_step_combo.addItem(text, key)
        self.load_step_combo.setToolTip(
            "<p>How the <b>load steps of one test</b> are combined.</p>"
            "<p>They are repeat measurements of the same specimen and are "
            "0.96-0.999 correlated here, so <b>GLS</b> ends up differencing "
            "them: its weights go negative and the answer can land outside "
            "the range of the per-step estimates. <b>OLS</b> is a plain "
            "average, which cannot do that.</p>"
            "<p>With one load step per experiment this choice has no effect "
            "at all - there is nothing to combine.</p>")
        self.load_step_combo.setToolTipDuration(60000)
        self.load_step_combo.currentIndexChanged.connect(
            self._refresh_weighting_hint)
        form.addRow("Load-step estimator", self.load_step_combo)

        self.experiment_combo = QComboBox()
        for text, key in (
            ("Single rule below (legacy)", None),
            ("GLS - inverse covariance  (recommended)", "gls"),
            ("Diagonal - inverse variance per parameter", "diagonal"),
            ("OLS - every experiment equal", "ols"),
        ):
            self.experiment_combo.addItem(text, key)
        self.experiment_combo.setToolTip(
            "<p>How the <b>separate experiments</b> are combined once each "
            "has been reduced to one estimate.</p>"
            "<p>Different tests are statistically independent and carry "
            "genuinely different information, so <b>GLS</b> is on solid "
            "ground here: it stops a coupon that knows nothing about a "
            "parameter from voting on it at full weight. <b>OLS</b> lets it, "
            "which is why equal weighting across tests fails badly.</p>"
            "<p>'Block-diagonal' is not offered: C is already block diagonal "
            "between experiments, so it would mean the same as GLS.</p>"
            "<p>With anything other than GLS the reported +/- stops being a "
            "covariance - quote the jackknife instead.</p>")
        self.experiment_combo.setToolTipDuration(60000)
        self.experiment_combo.currentIndexChanged.connect(
            self._refresh_weighting_hint)
        form.addRow("Experiment estimator", self.experiment_combo)

        self.weighting_hint = QLabel()
        self.weighting_hint.setWordWrap(True)
        form.addRow("", self.weighting_hint)

        outer.addLayout(form)
        self._refresh_weighting_hint()

        self.params_form = QFormLayout()
        params_box = QGroupBox("Initial guess")
        params_box.setLayout(self.params_form)
        outer.addWidget(params_box)
        outer.addStretch(1)
        return box

    # ------------------------------------------------------------------
    @staticmethod
    def _set_combo_key(combo, key) -> None:
        """Select the entry whose userData is `key`, falling back to legacy."""
        index = combo.findData(key if key else None)
        combo.setCurrentIndex(index if index >= 0 else 0)

    def _load_step_weighting(self) -> Optional[str]:
        return self.load_step_combo.currentData()

    def _experiment_weighting(self) -> Optional[str]:
        return self.experiment_combo.currentData()

    def _refresh_weighting_hint(self) -> None:
        """Spell out, in one line, what the current pair actually does."""
        load_step = self._load_step_weighting()
        experiment = self._experiment_weighting()
        if load_step is None and experiment is None:
            self.weighting_hint.setText(
                "One rule for both jobs (weighting='gls' unless a loaded "
                "project says otherwise). This is the historical behaviour "
                "and is reproduced bit-for-bit.")
            return
        resolved_load = load_step or "gls"
        resolved_experiment = experiment or "gls"
        text = (f"Two-stage: load steps combined by "
                f"<b>{resolved_load}</b>, then experiments combined by "
                f"<b>{resolved_experiment}</b>.")
        if resolved_load == "ols" and resolved_experiment == "gls":
            text += (" This is the recommended pairing - average the repeat "
                     "measurements, weight the independent ones.")
        if resolved_experiment != "gls":
            text += (" The reported +/- is no longer a covariance; use the "
                     "jackknife.")
        self.weighting_hint.setText(text)

    def _seed_from_defaults(self, defaults) -> None:
        index = self.material_combo.findData(
            str(defaults.material_type).lower())
        if index >= 0:
            self.material_combo.setCurrentIndex(index)
        index = self.analysis_combo.findData(
            str(defaults.analysis_type).lower())
        if index >= 0:
            self.analysis_combo.setCurrentIndex(index)
        self._rebuild_params()
        self._sync_experiment_count(1)

    # ------------------------------------------------------------------
    def browse_dir(self) -> str:
        for tab in self._tabs_by_id.values():
            if tab.tests:
                return str(tab.tests[0].data_file.parent)
        return str(Path.cwd())

    def _sync_experiment_count(self, count: int) -> None:
        while self.tabs.count() > count:
            index = self.tabs.count() - 1
            tab = self.tabs.widget(index)
            self._tabs_by_id.pop(tab.experiment_id, None)
            self.tabs.removeTab(index)
        while self.tabs.count() < count:
            experiment_id = self.tabs.count() + 1
            tab = ExperimentTab(experiment_id, self._analysis(), self)
            tab.name_edit.textChanged.connect(
                lambda text, i=experiment_id: self._rename_tab(i, text))
            self._tabs_by_id[experiment_id] = tab
            self.tabs.addTab(tab, f"Experiment {experiment_id}")
            tab.refresh()

    def _rename_tab(self, experiment_id: int, text: str) -> None:
        tab = self._tabs_by_id.get(experiment_id)
        if tab is None:
            return
        index = self.tabs.indexOf(tab)
        if index >= 0:
            self.tabs.setTabText(index, text or f"Experiment {experiment_id}")

    def _material(self) -> str:
        return str(self.material_combo.currentData())

    def _analysis(self) -> str:
        return str(self.analysis_combo.currentData())

    def _rebuild_params(self) -> None:
        key = (self._material(), self._analysis())
        while self.params_form.rowCount():
            self.params_form.removeRow(0)
        self._param_spins = []
        if key not in PARAMETER_LABELS_BY_MODEL:
            return
        for label in parameter_labels_for(*key):
            spin = QDoubleSpinBox()
            spin.setRange(-1.0e30, 1.0e30)
            spin.setDecimals(6)
            spin.setValue(1.0)
            self.params_form.addRow(label, spin)
            self._param_spins.append(spin)
        for tab in self._tabs_by_id.values():
            tab._analysis_type = self._analysis()

    # ------------------------------------------------------------------
    #  Project I/O
    # ------------------------------------------------------------------
    def _load_project(self) -> None:
        start = (str(self._project_path.parent) if self._project_path
                 else self.browse_dir())
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
        self.adopt_project(project, Path(path))

    def adopt_project(self, project: VfmProject,
                      path: Optional[Path] = None) -> None:
        self._project_path = path
        index = self.material_combo.findData(project.material_type)
        if index >= 0:
            self.material_combo.setCurrentIndex(index)
        index = self.analysis_combo.findData(project.analysis_type)
        if index >= 0:
            self.analysis_combo.setCurrentIndex(index)
        self.basis_spin.setValue(project.basis_degree)
        self.spatial_check.setChecked(project.spatial_correction)
        self._set_combo_key(self.load_step_combo, project.load_step_weighting)
        self._set_combo_key(self.experiment_combo,
                            project.experiment_weighting)
        self._refresh_weighting_hint()
        self._rebuild_params()
        for spin, value in zip(self._param_spins, project.initial_parameters):
            spin.setValue(float(value))

        groups = group_by_experiment(project)
        self.count_spin.setValue(max(1, len(groups)))
        self._sync_experiment_count(max(1, len(groups)))
        for position, (exp_id, name, tests) in enumerate(groups, start=1):
            tab = self._tabs_by_id.get(position)
            if tab is None:
                continue
            for test in tests:
                test.experiment_id = position
            tab.adopt(tests, name)
        self.status.setText(
            f"Loaded {path.name if path else 'project'} — "
            f"{describe_project(project)}")
        self.status.setStyleSheet("color: #12805a;")

    def _save_project(self) -> None:
        project = self.build_project()
        if project is None:
            return
        start = str(self._project_path or (Path.cwd() / "run.vfmproj.json"))
        path, _ = QFileDialog.getSaveFileName(
            self, "Save VFM project", start, "VFM project (*.json)")
        if not path:
            return
        try:
            save_project(project, Path(path))
        except OSError as error:
            QMessageBox.critical(self, "Could not save", str(error))
            return
        self._project_path = Path(path)
        self.status.setText(f"Saved {Path(path).name}")
        self.status.setStyleSheet("color: #12805a;")

    # ------------------------------------------------------------------
    def build_project(self) -> Optional[VfmProject]:
        """Assemble the project, or show why it cannot be assembled."""
        tests: List[ProjectTest] = []
        names: Dict[int, str] = {}
        for position in range(1, self.tabs.count() + 1):
            tab = self._tabs_by_id.get(position)
            if tab is None:
                continue
            tab.commit_step_loads()
            names[position] = tab.name_edit.text().strip() or \
                f"Experiment {position}"
            for order, test in enumerate(tab.tests, start=1):
                test.experiment_id = position
                test.label = f"{names[position]}:{test.data_file.stem}"
                tests.append(test)

        if not tests:
            QMessageBox.warning(self, "Nothing to run",
                                "Add at least one .mat time step.")
            return None

        project = VfmProject(
            tests=tests,
            material_type=self._material(),
            analysis_type=self._analysis(),
            initial_parameters=[s.value() for s in self._param_spins],
            basis_degree=self.basis_spin.value(),
            spatial_correction=self.spatial_check.isChecked(),
            load_step_weighting=self._load_step_weighting(),
            experiment_weighting=self._experiment_weighting(),
            experiment_names=names,
        )
        problems = check_experiment_consistency(project)
        missing = [t.data_file.name for t in tests if not t.data_file.is_file()]
        if missing:
            problems.append("Missing measurement files: "
                            + ", ".join(missing[:6]))
        if problems:
            QMessageBox.warning(self, "The run is not complete yet",
                                "\n\n".join(problems))
            return None
        return project

    def _on_accept(self) -> None:
        project = self.build_project()
        if project is None:
            return

        #  Advisory, not blocking: a genuinely biaxial coupon has no dominant
        #  curvature and the check stays silent, so this must never be a hard
        #  gate.  But a load DOF on the wrong axis produces a run that
        #  completes happily and reports the WRONG coefficient, which is worth
        #  one confirmation.
        warnings = check_load_orientation(project)
        if warnings:
            answer = QMessageBox.warning(
                self, "Check the applied-load direction",
                "The declared load direction does not match the measured "
                "bending for:\n\n" + "\n\n".join(f"• {w}" for w in warnings)
                + "\n\nRun anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return

        self._project = project
        self.accept()

    def result_project(self) -> Optional[VfmProject]:
        return getattr(self, "_project", None)

    def project_path(self) -> Optional[Path]:
        return self._project_path


def launch_experiment_setup(defaults, output_dir: Path):
    """Show the dialog; return (project, path) or (None, None) on cancel."""
    owns = QApplication.instance() is None
    app = QApplication.instance() or QApplication([])
    dialog = ExperimentSetupDialog(defaults, output_dir)
    accepted = dialog.exec() == QDialog.Accepted
    if not accepted:
        return None, None
    return dialog.result_project(), dialog.project_path()
