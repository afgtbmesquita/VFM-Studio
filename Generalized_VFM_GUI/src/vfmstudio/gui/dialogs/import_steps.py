"""Add load steps from measurement MAT files or DIC + machine CSVs."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from vfmstudio.core.document import LoadStepModel
from vfmstudio.gui.dialogs.import_mat import load_steps_from_paths, natural_path_key
from vfmstudio.io.csv_import import (
    JOIN_AUTO,
    JOIN_DIC_NAME,
    JOIN_DIC_NUMBER,
    JOIN_ROW,
    UNIT_AUTO,
    UNIT_CUSTOM,
    UNIT_N,
    UNIT_NM,
    UNIT_NMM,
    VIC3D_CONVENTION_NOTE,
    CsvImportError,
    LoadColumn,
    StepMatch,
    available_join_keys,
    collect_load_columns,
    convert_csv_steps,
    default_output_folder,
    force_preview,
    format_match_row,
    is_vic3d_csv,
    load_frames_from_paths,
    match_dic_steps,
    path_identity,
)

_SOURCE_MAT = "mat"
_SOURCE_CSV = "csv"

_JOIN_LABELS = {
    JOIN_AUTO: "Auto",
    JOIN_DIC_NUMBER: "DIC file number",
    JOIN_DIC_NAME: "DIC file name",
    JOIN_ROW: "Row index",
}

_UNMATCHED = QColor("#ffe0e0")
_UNIT_LABELS = (
    (UNIT_AUTO, "Auto"),
    (UNIT_NMM, "N·mm"),
    (UNIT_NM, "N·m (×1000)"),
    (UNIT_N, "N"),
    (UNIT_CUSTOM, "Custom scale"),
)


def _add_unique_paths(existing: List[Path], chosen: Sequence[str]) -> List[Path]:
    identities = {path_identity(path) for path in existing}
    merged = list(existing)
    for text in chosen:
        path = Path(text)
        identity = path_identity(path)
        if identity in identities:
            continue
        identities.add(identity)
        merged.append(path)
    merged.sort(key=natural_path_key)
    return merged


class _FileList(QWidget):
    """Browse / clear list of files."""

    changed = Signal()

    def __init__(
        self,
        caption: str,
        file_filter: str,
        *,
        start_directory: str = "",
        dialog_title: str = "Select files",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._filter = file_filter
        self._dialog_title = dialog_title
        self._start_directory = start_directory or ""
        self._paths: List[Path] = []

        self._list = QListWidget()
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        clear = QPushButton("Clear")
        clear.clicked.connect(self.clear)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(browse)
        row.addWidget(clear)
        row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        if caption:
            label = QLabel(caption)
            label.setWordWrap(True)
            layout.addWidget(label)
        layout.addWidget(self._list, 1)
        layout.addLayout(row)

    def paths(self) -> List[Path]:
        return list(self._paths)

    def set_paths(self, paths: Sequence[Path | str]) -> None:
        self._paths = _add_unique_paths([], [str(path) for path in paths])
        if self._paths:
            self._start_directory = str(self._paths[0].parent)
        self._rebuild()

    def clear(self) -> None:
        self._paths = []
        self._rebuild()

    def _browse(self) -> None:
        chosen, _ = QFileDialog.getOpenFileNames(
            self,
            self._dialog_title,
            self._start_directory,
            self._filter,
        )
        if not chosen:
            return
        self._start_directory = str(Path(chosen[0]).parent)
        self._paths = _add_unique_paths(self._paths, chosen)
        self._rebuild()

    def _rebuild(self) -> None:
        self._list.clear()
        for path in self._paths:
            self._list.addItem(str(path))
        self.changed.emit()


class AddLoadStepsDialog(QDialog):
    """MAT files, or DIC CSV + synchronized machine CSV converted to Layout A MAT."""

    def __init__(
        self,
        parent=None,
        *,
        start_directory: str = "",
        source_path: Optional[Path] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add load steps")
        self.resize(820, 560)
        self._start_directory = start_directory or ""
        self._source_path = Path(source_path) if source_path else None
        self._result_steps: Optional[List[LoadStepModel]] = None
        self._load_frames = {}
        self._columns: List[LoadColumn] = []
        self._matches: List[StepMatch] = []

        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_files_page())
        self._stack.addWidget(self._build_mapping_page())

        self._back = QPushButton("Back")
        self._back.clicked.connect(self._go_files)
        self._next = QPushButton("Next")
        self._next.clicked.connect(self._on_next)
        self._ok = QPushButton("Add steps")
        self._ok.clicked.connect(self._on_finish)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self._back)
        buttons.addWidget(self._next)
        buttons.addWidget(self._ok)
        buttons.addWidget(cancel)

        layout = QVBoxLayout(self)
        layout.addWidget(self._stack, 1)
        layout.addLayout(buttons)
        self._sync_buttons()
        self._mat_list.changed.connect(self._sync_buttons)
        self._dic_list.changed.connect(self._sync_buttons)
        self._load_list.changed.connect(self._sync_buttons)

    def _build_files_page(self) -> QWidget:
        page = QWidget()
        hint = QLabel(
            "Add one measurement file per load step. MAT files are used as-is. "
            "CSV files are Vic-3D facet clouds plus already-synchronized machine "
            "tables — time-sync of raw logs is not done here."
        )
        hint.setWordWrap(True)
        hint.setProperty("role", "hint")

        self._mat_radio = QRadioButton("Measurement MAT files")
        self._csv_radio = QRadioButton("DIC CSV + machine CSV")
        self._mat_radio.setChecked(True)
        group = QButtonGroup(self)
        group.addButton(self._mat_radio)
        group.addButton(self._csv_radio)
        self._mat_radio.toggled.connect(self._on_source_toggled)

        radios = QHBoxLayout()
        radios.addWidget(self._mat_radio)
        radios.addWidget(self._csv_radio)
        radios.addStretch(1)

        self._source_stack = QStackedWidget()
        mat_page = QWidget()
        mat_layout = QVBoxLayout(mat_page)
        mat_layout.setContentsMargins(0, 0, 0, 0)
        self._mat_list = _FileList(
            "Select one or more measurement MAT files (natural filename order).",
            "MAT (*.mat);;All files (*)",
            start_directory=self._start_directory,
            dialog_title="Measurement MAT files",
        )
        mat_layout.addWidget(self._mat_list, 1)
        self._source_stack.addWidget(mat_page)

        csv_page = QWidget()
        csv_layout = QVBoxLayout(csv_page)
        csv_layout.setContentsMargins(0, 0, 0, 0)
        csv_layout.setSpacing(8)
        self._dic_list = _FileList(
            "DIC files — one Vic-3D facet cloud (exp-*.csv) per load step.",
            "CSV (*.csv);;All files (*)",
            start_directory=self._start_directory,
            dialog_title="DIC CSV files",
        )
        self._load_list = _FileList(
            "Load / machine tables — already synchronized (one or more CSVs).",
            "CSV (*.csv);;All files (*)",
            start_directory=self._start_directory,
            dialog_title="Machine / load CSV files",
        )
        csv_layout.addWidget(self._dic_list, 1)
        csv_layout.addWidget(self._load_list, 1)
        self._source_stack.addWidget(csv_page)

        layout = QVBoxLayout(page)
        layout.addWidget(hint)
        layout.addLayout(radios)
        layout.addWidget(self._source_stack, 1)
        return page

    def _build_mapping_page(self) -> QWidget:
        page = QWidget()
        col_hint = QLabel(
            "Tick the machine channels to store as F. Rename them for the "
            "applied-load mapping. Units default from the column name "
            "(N·m → N·mm). Sign is applied after the unit scale."
        )
        col_hint.setWordWrap(True)
        col_hint.setProperty("role", "hint")

        self._join = QComboBox()
        self._join.currentIndexChanged.connect(self._rematch)

        self._column_table = QTableWidget(0, 7)
        self._column_table.setHorizontalHeaderLabels(
            ["Use", "Source", "Column", "Import as", "Unit", "Sign", "Scale"]
        )
        self._column_table.verticalHeader().setVisible(False)
        self._column_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        header = self._column_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        self._column_table.itemChanged.connect(self._on_column_item_changed)

        match_hint = QLabel(
            "Each DIC file must match a row in the selected load tables. "
            "Edit the step label to change how it appears in the tree. "
            "Leave fields in the measurement frame; set the Analysis window "
            "reference step after import so Identify subtracts it."
        )
        match_hint.setWordWrap(True)
        match_hint.setProperty("role", "hint")

        self._match_table = QTableWidget(0, 4)
        self._match_table.setHorizontalHeaderLabels(
            ["Label", "DIC file", "Matched row", "F preview"]
        )
        self._match_table.verticalHeader().setVisible(False)
        match_header = self._match_table.horizontalHeader()
        match_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        match_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        match_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        match_header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self._match_table.itemChanged.connect(self._on_match_item_changed)

        self._output = QLineEdit()
        self._output.textChanged.connect(self._sync_buttons)
        browse_out = QPushButton("Browse…")
        browse_out.clicked.connect(self._browse_output)
        out_row = QHBoxLayout()
        out_row.addWidget(self._output, 1)
        out_row.addWidget(browse_out)

        self._vic3d = QCheckBox("Convert Vic-3D strains/curvatures to plate theory")
        self._vic3d.setChecked(True)
        self._vic3d.setToolTip(
            "On: apply the Vic-3D → Kirchhoff plate-theory factors below.\n"
            "Off: copy cxx/cyy/cxy and exx/eyy/exy into the MAT file unchanged."
        )
        self._vic3d_note = QLabel(VIC3D_CONVENTION_NOTE)
        self._vic3d_note.setWordWrap(True)
        self._vic3d_note.setProperty("role", "hint")
        self._vic3d.toggled.connect(self._vic3d_note.setVisible)
        self._intersect = QCheckBox("Keep only facets present in every DIC file")
        self._intersect.setChecked(True)

        form = QFormLayout()
        form.addRow("Join DIC to load rows", self._join)
        form.addRow("Write MAT files to", out_row)

        layout = QVBoxLayout(page)
        layout.addWidget(col_hint)
        layout.addWidget(self._column_table, 1)
        layout.addWidget(match_hint)
        layout.addWidget(self._match_table, 1)
        layout.addLayout(form)
        layout.addWidget(self._vic3d)
        layout.addWidget(self._vic3d_note)
        layout.addWidget(self._intersect)
        return page

    def source_mode(self) -> str:
        return _SOURCE_CSV if self._csv_radio.isChecked() else _SOURCE_MAT

    def set_source_mode(self, mode: str) -> None:
        if mode == _SOURCE_CSV:
            self._csv_radio.setChecked(True)
        else:
            self._mat_radio.setChecked(True)

    def _on_source_toggled(self, _checked: bool = False) -> None:
        self._source_stack.setCurrentIndex(1 if self._csv_radio.isChecked() else 0)
        self._sync_buttons()

    def _page(self) -> int:
        return self._stack.currentIndex()

    def _go_files(self) -> None:
        self._stack.setCurrentIndex(0)
        self._sync_buttons()

    def _go_mapping(self) -> None:
        try:
            self._prepare_mapping()
        except CsvImportError as exc:
            QMessageBox.warning(self, "Add load steps", str(exc))
            return
        self._stack.setCurrentIndex(1)
        self._sync_buttons()

    def _on_next(self) -> None:
        if self.source_mode() == _SOURCE_MAT:
            self._on_finish()
            return
        self._go_mapping()

    def _files_ready(self) -> bool:
        if self.source_mode() == _SOURCE_MAT:
            return bool(self._mat_list.paths())
        return bool(self._dic_list.paths())

    def _mapping_ready(self) -> bool:
        if not self._dic_list.paths():
            return False
        if not str(self._output.text()).strip():
            return False
        used = [item for item in self._columns if item.used]
        if not used:
            return True
        return bool(self._matches) and all(match.is_ready(used) for match in self._matches)

    def _sync_buttons(self) -> None:
        csv = self.source_mode() == _SOURCE_CSV
        on_map = self._page() == 1
        self._back.setVisible(on_map)
        self._next.setVisible(not on_map)
        if not on_map:
            self._next.setText("Add steps" if not csv else "Next")
            self._next.setEnabled(self._files_ready())
            self._ok.setVisible(False)
        else:
            self._ok.setVisible(True)
            self._ok.setEnabled(self._mapping_ready())

    def _prepare_mapping(self) -> None:
        dic_paths = self._dic_list.paths()
        if not dic_paths:
            raise CsvImportError("Select at least one DIC CSV file.")
        load_paths = self._load_list.paths()
        self._load_frames = load_frames_from_paths(load_paths)
        self._columns = collect_load_columns(load_paths, self._load_frames)
        self._output.setText(
            str(default_output_folder(self._source_path, dic_paths))
        )
        try:
            self._vic3d.setChecked(is_vic3d_csv(dic_paths[0]))
        except Exception:
            self._vic3d.setChecked(True)
        self._rebuild_join_combo()
        self._rebuild_column_table()
        self._rematch()

    def _rebuild_join_combo(self) -> None:
        self._join.blockSignals(True)
        self._join.clear()
        frames = list(self._load_frames.values())
        for key in available_join_keys(frames) if frames else [JOIN_AUTO, JOIN_ROW]:
            self._join.addItem(_JOIN_LABELS.get(key, key), key)
        self._join.setCurrentIndex(0)
        self._join.blockSignals(False)

    def _rebuild_column_table(self) -> None:
        table = self._column_table
        table.blockSignals(True)
        table.setRowCount(len(self._columns))
        for row, column in enumerate(self._columns):
            use = QTableWidgetItem()
            use.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsUserCheckable
            )
            use.setCheckState(
                Qt.CheckState.Checked if column.used else Qt.CheckState.Unchecked
            )
            table.setItem(row, 0, use)

            source = QTableWidgetItem(column.source_file.name)
            source.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            table.setItem(row, 1, source)

            name = QTableWidgetItem(column.source_column)
            name.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            table.setItem(row, 2, name)

            rename = QTableWidgetItem(column.import_as)
            table.setItem(row, 3, rename)

            unit = QComboBox()
            for value, label in _UNIT_LABELS:
                unit.addItem(label, value)
            match = unit.findData(column.unit)
            unit.setCurrentIndex(match if match >= 0 else 0)
            unit.currentIndexChanged.connect(
                lambda _i, r=row, box=unit: self._on_unit_changed(r, box)
            )
            table.setCellWidget(row, 4, unit)

            sign = QComboBox()
            sign.addItem("+1", 1.0)
            sign.addItem("−1", -1.0)
            sign.setCurrentIndex(0 if column.sign >= 0 else 1)
            sign.currentIndexChanged.connect(
                lambda _i, r=row, box=sign: self._on_sign_changed(r, box)
            )
            table.setCellWidget(row, 5, sign)
            self._set_scale_item(row, column)
        table.blockSignals(False)

    def _set_scale_item(self, row: int, column: LoadColumn) -> None:
        custom = column.unit == UNIT_CUSTOM
        value = column.scale if custom else column.resolved_scale()
        item = QTableWidgetItem(f"{value:g}")
        if custom:
            item.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEditable
            )
        else:
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        self._column_table.setItem(row, 6, item)

    def _on_column_item_changed(self, item: QTableWidgetItem) -> None:
        row = item.row()
        if row < 0 or row >= len(self._columns):
            return
        column = self._columns[row]
        if item.column() == 0:
            column.used = item.checkState() == Qt.CheckState.Checked
            self._rematch()
            return
        if item.column() == 3:
            text = item.text().strip()
            column.import_as = text or column.source_column
            self._rebuild_match_table()
            return
        if item.column() == 6:
            try:
                column.scale = float(item.text().strip())
            except ValueError:
                column.scale = 1.0
            column.unit = UNIT_CUSTOM
            combo = self._column_table.cellWidget(row, 4)
            if isinstance(combo, QComboBox):
                index = combo.findData(UNIT_CUSTOM)
                combo.blockSignals(True)
                if index >= 0:
                    combo.setCurrentIndex(index)
                combo.blockSignals(False)
            self._rebuild_match_table()

    def _on_unit_changed(self, row: int, combo: QComboBox) -> None:
        if row < 0 or row >= len(self._columns):
            return
        data = combo.currentData()
        column = self._columns[row]
        previous = column.resolved_scale()
        column.unit = str(data or UNIT_AUTO)
        if column.unit == UNIT_CUSTOM:
            column.scale = previous
        self._column_table.blockSignals(True)
        self._set_scale_item(row, column)
        self._column_table.blockSignals(False)
        self._rebuild_match_table()

    def _on_sign_changed(self, row: int, combo: QComboBox) -> None:
        if row < 0 or row >= len(self._columns):
            return
        data = combo.currentData()
        self._columns[row].sign = float(data if data is not None else 1.0)
        self._rebuild_match_table()

    def _join_key(self) -> str:
        data = self._join.currentData()
        return str(data or JOIN_AUTO)

    def _rematch(self) -> None:
        dic_paths = self._dic_list.paths()
        self._matches = match_dic_steps(
            dic_paths, self._load_frames, join_key=self._join_key()
        )
        self._rebuild_match_table()
        self._sync_buttons()

    def _rebuild_match_table(self) -> None:
        table = self._match_table
        table.blockSignals(True)
        table.setRowCount(len(self._matches))
        used = [item for item in self._columns if item.used]
        for row, match in enumerate(self._matches):
            ready = match.is_ready(used)
            label = QTableWidgetItem(match.label)
            table.setItem(row, 0, label)

            dic = QTableWidgetItem(match.dic_path.name)
            dic.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            table.setItem(row, 1, dic)

            status = QTableWidgetItem(
                format_match_row(match, self._load_frames, self._columns)
            )
            status.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            table.setItem(row, 2, status)

            preview = force_preview(match, self._columns, self._load_frames)
            bits = []
            for name, value in preview.items():
                if value is None:
                    bits.append(f"{name}=?")
                else:
                    bits.append(f"{name}={value:g}")
            preview_item = QTableWidgetItem(", ".join(bits) if bits else "—")
            preview_item.setFlags(
                Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            )
            table.setItem(row, 3, preview_item)
            for col in range(4):
                cell = table.item(row, col)
                if cell is None:
                    continue
                if not ready:
                    cell.setBackground(_UNMATCHED)
                else:
                    cell.setBackground(QColor())
        table.blockSignals(False)
        self._sync_buttons()

    def _on_match_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != 0:
            return
        row = item.row()
        if row < 0 or row >= len(self._matches):
            return
        text = item.text().strip()
        self._matches[row].label = text or self._matches[row].dic_path.stem

    def _browse_output(self) -> None:
        start = str(self._output.text()).strip() or self._start_directory
        chosen = QFileDialog.getExistingDirectory(self, "MAT output folder", start)
        if chosen:
            self._output.setText(chosen)
            self._sync_buttons()

    def _on_finish(self) -> None:
        if self.source_mode() == _SOURCE_MAT:
            steps = load_steps_from_paths(self._mat_list.paths())
            if not steps:
                return
            self._result_steps = steps
            self.accept()
            return
        if self._page() == 0:
            self._go_mapping()
            return
        if not self._mapping_ready():
            QMessageBox.information(
                self,
                "Add load steps",
                "Match every DIC file to a load-table row, or uncheck the "
                "load columns if you only want kinematics.",
            )
            return
        output = Path(str(self._output.text()).strip())
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._result_steps = convert_csv_steps(
                self._matches,
                self._load_frames,
                self._columns,
                output,
                vic3d_conventions=self._vic3d.isChecked(),
                intersect_facets=self._intersect.isChecked(),
            )
        except CsvImportError as exc:
            QMessageBox.warning(self, "Add load steps", str(exc))
            return
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Add load steps",
                f"Could not convert the CSV files:\n{exc}",
            )
            return
        finally:
            QApplication.restoreOverrideCursor()
        if not self._result_steps:
            QMessageBox.information(
                self,
                "Add load steps",
                "No new measurement files were written.",
            )
            return
        self.accept()

    def steps(self, skip: Optional[Sequence[Path | str]] = None) -> List[LoadStepModel]:
        if self._result_steps is not None:
            skip_ids = {path_identity(Path(item)) for item in (skip or [])}
            return [
                step
                for step in self._result_steps
                if path_identity(Path(step.data_file)) not in skip_ids
            ]
        if self.source_mode() == _SOURCE_MAT:
            return load_steps_from_paths(self._mat_list.paths(), skip=skip)
        return []
