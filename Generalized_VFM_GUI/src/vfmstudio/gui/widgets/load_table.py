"""Experiment-wide table of load-step force / moment magnitudes."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from vfmstudio.core.document import ExperimentModel, LoadStepModel
from vfmstudio.core.fields import (
    conjugate_moment_plain,
    force_component_text,
    force_index_for_group,
    mapped_force_component,
    mat_force_component,
    read_applied_force_from_file,
    read_force_labels_from_file,
)
from vfmstudio.core.step_series import resolve_step_data_file, step_display_label

_BLANK = QColor("#ffe0e0")
_WHITE = QColor("#ffffff")
_MUTED = QColor("#9a9a9a")
_HIGHLIGHT = QColor("#d6e6f7")

ReadForce = Callable[[Path], Optional[object]]


def format_load(value) -> str:
    """Readable cell text. Exact float lives in ``Qt.UserRole``."""
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if number != number or abs(number) == float("inf"):
        return ""
    return f"{number:g}"


def group_header(entry) -> str:
    label = str(getattr(entry, "label", "") or "").strip()
    if label:
        return label
    dof = str(getattr(entry, "dof", "") or "")
    edge = str(getattr(entry, "edge", "") or "") or None
    plain = conjugate_moment_plain(dof, edge)
    return f"{plain} ({edge})" if edge else plain


def pad_external_values(step: LoadStepModel, n_groups: int) -> list:
    values = list(step.external_values or [])
    if n_groups <= 0:
        step.external_values = []
        return []
    if len(values) < n_groups:
        values.extend([None] * (n_groups - len(values)))
    step.external_values = values[:n_groups]
    return step.external_values


def fill_blank_external_values(
    experiment: ExperimentModel,
    source_path: Optional[Path] = None,
    read_force: Optional[ReadForce] = None,
) -> int:
    """Copy MAT ``F`` into empty per-step slots. Returns how many cells filled."""
    reader = read_force or read_applied_force_from_file
    n_groups = len(experiment.external)
    filled = 0
    for step in experiment.steps:
        values = pad_external_values(step, n_groups)
        force = _force_for_step(step, source_path, reader)
        for index, entry in enumerate(experiment.external):
            if values[index] is not None:
                continue
            value = mapped_force_component(force, entry, index)
            if value is None:
                continue
            values[index] = value
            filled += 1
        step.external_values = values
    return filled


def displayed_load(
    step: LoadStepModel,
    index: int,
    n_groups: int,
    *,
    manual: bool,
    force,
    entry=None,
) -> Optional[float]:
    if manual:
        values = list(step.external_values or [])
        if index < len(values) and values[index] is not None:
            try:
                number = float(values[index])
            except (TypeError, ValueError):
                return None
            if number == number and abs(number) != float("inf"):
                return number
        return None
    if entry is not None:
        return mapped_force_component(force, entry, index)
    return mat_force_component(force, index, n_groups)


def _force_for_step(step: LoadStepModel, source_path: Optional[Path], reader: ReadForce):
    resolved = resolve_step_data_file(Path(step.data_file), source_path)
    if resolved is None:
        return None
    try:
        return reader(resolved)
    except Exception:
        return None


def _referenced_force_for_step(
    step: LoadStepModel,
    experiment: Optional[ExperimentModel],
    source_path: Optional[Path],
    reader: ReadForce,
):
    """MAT F for display: current minus the experiment reference step when set."""
    force = _force_for_step(step, source_path, reader)
    if experiment is None or experiment.reference_step is None:
        return force
    ref_index = int(experiment.reference_step) - 1
    if ref_index < 0 or ref_index >= len(experiment.steps):
        return force
    reference = _force_for_step(experiment.steps[ref_index], source_path, reader)
    if force is None or reference is None:
        return force
    current = np.atleast_1d(np.asarray(force, dtype=float)).ravel()
    baseline = np.atleast_1d(np.asarray(reference, dtype=float)).ravel()
    if current.shape != baseline.shape:
        return force
    return current - baseline


class AppliedLoadEditor(QWidget):
    """Rows = load steps, columns = EXTERNAL groups. Magnitudes are N·mm."""

    loads_changed = Signal()
    mapping_changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._experiment: Optional[ExperimentModel] = None
        self._source_path: Optional[Path] = None
        self._read_force: ReadForce = read_applied_force_from_file
        self._highlight_row: Optional[int] = None
        self._map_combos: list = []

        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setProperty("role", "hint")

        self.map_caption = QLabel(
            "Assign each .mat F component to an EXTERNAL group. "
            "Default is list order (F[0] → first group)."
        )
        self.map_caption.setWordWrap(True)
        self.map_caption.setProperty("role", "hint")
        self.map_host = QWidget()
        self._map_form = QFormLayout(self.map_host)
        self._map_form.setContentsMargins(0, 0, 0, 0)
        self._map_form.setSpacing(4)

        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setProperty("role", "hint")

        self.table = QTableWidget(0, 2)
        self.table.setObjectName("appliedLoadTable")
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.SelectedClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.AnyKeyPressed
        )
        self.table.setMinimumHeight(140)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.table.itemChanged.connect(self._on_item_changed)

        self.fill_button = QPushButton("Fill blanks from .mat F")
        self.fill_button.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed
        )
        self.fill_button.setProperty("role", "compact")
        self.fill_button.setToolTip(
            "Copy each step's F vector from its measurement file into empty "
            "table cells, then use this table as the applied-load source."
        )
        self.fill_button.clicked.connect(self._on_fill_blanks)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.addWidget(self.fill_button)
        buttons.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.note)
        layout.addWidget(self.map_caption)
        layout.addWidget(self.map_host)
        layout.addWidget(self.table, 1)
        layout.addLayout(buttons)

    def populate(
        self,
        experiment: Optional[ExperimentModel],
        *,
        source_path: Optional[Path] = None,
        highlight_row: Optional[int] = None,
        read_force: Optional[ReadForce] = None,
    ) -> None:
        self._experiment = experiment
        self._source_path = source_path
        self._highlight_row = highlight_row
        if read_force is not None:
            self._read_force = read_force
        self._rebuild()

    def _rebuild(self) -> None:
        experiment = self._experiment
        table = self.table
        table.blockSignals(True)
        table.clear()
        if experiment is None:
            table.setRowCount(0)
            table.setColumnCount(2)
            table.setHorizontalHeaderLabels(["#", "Step"])
            self.note.setText("Select an experiment to edit applied loads.")
            self.fill_button.setEnabled(False)
            self._rebuild_mapping([])
            table.blockSignals(False)
            return

        groups: Sequence = list(experiment.external or [])
        n_groups = len(groups)
        headers = ["#", "Step"] + [group_header(group) for group in groups]
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(experiment.steps))
        self._layout_group_columns(n_groups)
        header = table.horizontalHeader()
        for column in range(2, len(headers)):
            header_item = table.horizontalHeaderItem(column)
            if header_item is not None:
                header_item.setToolTip(
                    f"{headers[column]}  (N·mm, total couple — not per unit width)"
                )

        manual = str(experiment.applied_load_source or "mat") == "manual"
        blanks = 0
        for row, step in enumerate(experiment.steps):
            pad_external_values(step, n_groups)
            force = (
                _force_for_step(step, self._source_path, self._read_force)
                if manual
                else _referenced_force_for_step(
                    step, experiment, self._source_path, self._read_force
                )
            )
            index_item = QTableWidgetItem(str(row + 1))
            index_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            table.setItem(row, 0, index_item)

            name_item = QTableWidgetItem(step_display_label(step))
            name_item.setToolTip(str(step.data_file))
            name_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            table.setItem(row, 1, name_item)

            highlight = self._highlight_row is not None and row == int(self._highlight_row)
            if highlight:
                index_item.setBackground(_HIGHLIGHT)
                name_item.setBackground(_HIGHLIGHT)

            for column, group in enumerate(groups):
                value = displayed_load(
                    step, column, n_groups, manual=manual, force=force, entry=group
                )
                item = QTableWidgetItem(format_load(value))
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                if value is not None:
                    item.setData(Qt.ItemDataRole.UserRole, float(value))
                if manual:
                    item.setFlags(
                        Qt.ItemFlag.ItemIsEnabled
                        | Qt.ItemFlag.ItemIsSelectable
                        | Qt.ItemFlag.ItemIsEditable
                    )
                    if value is None:
                        blanks += 1
                        item.setBackground(_BLANK)
                        item.setToolTip(
                            "Required: this experiment takes its applied load "
                            "from this table, and this cell is empty."
                        )
                    else:
                        item.setToolTip(
                            "Applied force or moment for this load step and "
                            "external-work group (N·mm)."
                        )
                else:
                    item.setFlags(
                        Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                    )
                    item.setForeground(_MUTED)
                    if value is None:
                        item.setToolTip(
                            "No F in this .mat file. Switch Applied load from "
                            "to “this table” to type a value."
                        )
                    else:
                        item.setToolTip(
                            "From this step's .mat F field. Switch Applied "
                            "load from to “this table” to override it."
                        )
                table.setItem(row, 2 + column, item)

        n_steps = len(experiment.steps)
        if not groups:
            self.note.setText(
                "Add an EXTERNAL boundary condition to create a force/moment "
                "column for every load step."
            )
        elif not n_steps:
            self.note.setText("Add load steps to enter applied force/moment values.")
        elif manual:
            extra = (
                ""
                if not blanks
                else f" {blanks} cell(s) still blank — Identify will refuse until they are filled."
            )
            self.note.setText(
                "The table is live. Each cell is that step's applied force or "
                "moment (N·mm); the .mat F field is ignored."
                + extra
            )
        else:
            extra = ""
            if experiment.reference_step is not None:
                extra = (
                    f" Values are current minus reference step "
                    f"{int(experiment.reference_step)}."
                )
            self.note.setText(
                "Loads are read from each step's .mat F field (shown greyed). "
                "Use the dropdowns to choose which F component loads each group. "
                "Switch Applied load from to “this table” to type values instead."
                + extra
            )
        self.fill_button.setEnabled(bool(groups) and n_steps > 0)
        self._rebuild_mapping(groups)
        table.blockSignals(False)

    def _layout_group_columns(self, n_groups: int) -> None:
        """Keep every EXTERNAL group on screen in the narrow properties dock."""
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(56)
        header.setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        if n_groups <= 0:
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            return
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        for column in range(2, 2 + n_groups):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)

    def _preview_force(self):
        experiment = self._experiment
        if experiment is None or not experiment.steps:
            return None
        row = 0
        if self._highlight_row is not None:
            row = max(0, min(int(self._highlight_row), len(experiment.steps) - 1))
        return _referenced_force_for_step(
            experiment.steps[row],
            experiment,
            self._source_path,
            self._read_force,
        )

    def _preview_labels(self) -> Optional[list]:
        experiment = self._experiment
        if experiment is None or not experiment.steps:
            return None
        row = 0
        if self._highlight_row is not None:
            row = max(0, min(int(self._highlight_row), len(experiment.steps) - 1))
        resolved = resolve_step_data_file(
            Path(experiment.steps[row].data_file), self._source_path
        )
        if resolved is None:
            return None
        try:
            return read_force_labels_from_file(resolved)
        except Exception:
            return None

    def _force_width(self) -> int:
        experiment = self._experiment
        n_groups = len(experiment.external) if experiment is not None else 0
        width = 0
        if experiment is not None:
            for step in experiment.steps:
                force = _force_for_step(step, self._source_path, self._read_force)
                if force is None:
                    continue
                width = max(width, int(np.atleast_1d(np.asarray(force, dtype=float)).size))
        return width or n_groups

    def _rebuild_mapping(self, groups: Sequence) -> None:
        # Called from a combo's own currentIndexChanged: deleting it synchronously crashes Qt.
        while self._map_form.rowCount():
            row = self._map_form.takeRow(0)
            for item in (row.labelItem, row.fieldItem):
                widget = item.widget() if item is not None else None
                if widget is not None:
                    widget.blockSignals(True)
                    widget.hide()
                    widget.deleteLater()
        self._map_combos = []
        n_groups = len(groups)
        show = n_groups > 0
        self.map_caption.setVisible(show)
        self.map_host.setVisible(show)
        if not show:
            return
        preview = self._preview_force()
        labels = self._preview_labels()
        n_force = self._force_width()
        preview_vals = []
        if preview is not None:
            preview_vals = [
                float(v)
                for v in np.atleast_1d(np.asarray(preview, dtype=float)).ravel()
            ]
        for group_index, entry in enumerate(groups):
            combo = QComboBox()
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
            for f_index in range(max(n_force, 1)):
                extra = ""
                if f_index < len(preview_vals):
                    extra = f"  ({format_load(preview_vals[f_index])} N·mm)"
                combo.addItem(
                    f"{force_component_text(f_index, labels)}{extra}", f_index
                )
            current = force_index_for_group(entry, group_index)
            match = combo.findData(current)
            combo.setCurrentIndex(match if match >= 0 else min(group_index, combo.count() - 1))
            combo.currentIndexChanged.connect(
                lambda _i, g=group_index, box=combo: self._on_map_changed(g, box)
            )
            self._map_form.addRow(group_header(entry), combo)
            self._map_combos.append(combo)

    def _on_map_changed(self, group_index: int, combo: QComboBox) -> None:
        experiment = self._experiment
        if experiment is None or group_index >= len(experiment.external):
            return
        data = combo.currentData()
        if data is None:
            return
        experiment.external[group_index].force_index = int(data)
        self.mapping_changed.emit()
        self._rebuild()

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        experiment = self._experiment
        if experiment is None:
            return
        row = item.row()
        group = item.column() - 2
        if row < 0 or row >= len(experiment.steps) or group < 0:
            return
        if str(experiment.applied_load_source or "mat") != "manual":
            return
        n_groups = len(experiment.external)
        if group >= n_groups:
            return
        step = experiment.steps[row]
        values = pad_external_values(step, n_groups)
        text = item.text().strip()
        if not text:
            values[group] = None
            self._decorate(item, None, required=True)
            experiment.steps[row].external_values = values
            self.loads_changed.emit()
            return
        stored = item.data(Qt.ItemDataRole.UserRole)
        if stored is not None and format_load(stored) == text:
            values[group] = float(stored)
            experiment.steps[row].external_values = values
            self._decorate(item, float(stored), required=False)
            return
        try:
            number = float(text)
        except ValueError:
            number = float("nan")
        if number != number or abs(number) == float("inf"):
            values[group] = None
            self._decorate(item, None, required=True, invalid=True)
            experiment.steps[row].external_values = values
            self.loads_changed.emit()
            return
        values[group] = number
        experiment.steps[row].external_values = values
        blocked = self.table.blockSignals(True)
        try:
            item.setData(Qt.ItemDataRole.UserRole, number)
        finally:
            self.table.blockSignals(blocked)
        self._decorate(item, number, required=False)
        self.loads_changed.emit()

    def _decorate(
        self,
        item: QTableWidgetItem,
        value: Optional[float],
        *,
        required: bool,
        invalid: bool = False,
    ) -> None:
        blocked = self.table.blockSignals(True)
        try:
            if invalid:
                item.setBackground(_BLANK)
                item.setToolTip("Invalid load. Enter a finite number, for example 1250 or -3.5e4.")
            elif required and value is None:
                item.setBackground(_BLANK)
                item.setToolTip(
                    "Required: this experiment takes its applied load from this "
                    "table, and this cell is empty."
                )
            else:
                item.setBackground(_WHITE)
                item.setToolTip(
                    "Applied force or moment for this load step and "
                    "external-work group (N·mm)."
                )
        finally:
            self.table.blockSignals(blocked)

    def _on_fill_blanks(self) -> None:
        experiment = self._experiment
        if experiment is None:
            return
        fill_blank_external_values(
            experiment, self._source_path, self._read_force
        )
        experiment.applied_load_source = "manual"
        self.loads_changed.emit()
