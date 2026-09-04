"""Add PRESCRIBED / LINKING / EXTERNAL groups."""

from __future__ import annotations

from typing import List

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from vfmstudio.core.bc_entries import (
    DOF_ORDER,
    default_dofs,
    make_boundary_entries,
)
from vfmstudio.core.document import BoundaryEntry


class AddBcDialog(QDialog):
    preview_nodes_changed = Signal(object, bool)

    def __init__(
        self,
        parent=None,
        kind: str = "prescribed",
        node_coords=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Add {kind} condition")
        self.kind = kind
        self._node_coords = _as_xy(node_coords)

        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText("Optional. Each DOF gets its own tree entry.")

        self._dof_boxes = {}
        dof_row = QWidget()
        dof_layout = QHBoxLayout(dof_row)
        dof_layout.setContentsMargins(0, 0, 0, 0)
        checked = set(default_dofs(kind))
        for name in DOF_ORDER:
            box = QCheckBox(name)
            box.setChecked(name in checked)
            self._dof_boxes[name] = box
            dof_layout.addWidget(box)
        dof_layout.addStretch(1)

        self._edge_radio = QRadioButton("Entire edge")
        self._nodes_radio = QRadioButton("Individual nodes")
        self._edge_radio.setChecked(True)
        target_group = QButtonGroup(self)
        target_group.addButton(self._edge_radio)
        target_group.addButton(self._nodes_radio)
        target_row = QWidget()
        target_layout = QHBoxLayout(target_row)
        target_layout.setContentsMargins(0, 0, 0, 0)
        target_layout.addWidget(self._edge_radio)
        target_layout.addWidget(self._nodes_radio)
        target_layout.addStretch(1)

        self.edge = QComboBox()
        self.edge.addItems(["top", "bottom", "left", "right"])
        self._edge_caption = QLabel("Edge")

        self.nodes = QListWidget()
        self.nodes.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.nodes.setMinimumHeight(140)
        self._fill_nodes()
        self._nodes_caption = QLabel(
            "Select one or more mesh nodes (Ctrl-click or shift-click). "
            "The 3D viewport highlights the selection and shows node ids. "
            "Click a mesh node in the viewport to add or remove it."
        )
        self._nodes_caption.setWordWrap(True)
        self._nodes_caption.setProperty("role", "hint")

        self._load = QComboBox()
        self._load.addItems(["distributed", "concentrated"])
        self._load_caption = QLabel("Load type")

        form = QFormLayout()
        form.addRow("Label", self.label_edit)
        form.addRow("DOFs", dof_row)
        form.addRow("Apply to", target_row)
        form.addRow(self._edge_caption, self.edge)
        form.addRow(self._load_caption, self._load)
        form.addRow(self._nodes_caption)
        form.addRow(self.nodes)

        is_external = kind == "external"
        self._load.setVisible(is_external)
        self._load_caption.setVisible(is_external)

        self._edge_radio.toggled.connect(self._sync_target)
        self.nodes.itemSelectionChanged.connect(self._emit_preview)
        self.finished.connect(self._clear_preview)
        self._sync_target()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _fill_nodes(self) -> None:
        self.nodes.clear()
        xy = self._node_coords
        if xy.size == 0:
            item = QListWidgetItem("No mesh nodes. Build a mesh first.")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.nodes.addItem(item)
            return
        for index, (x, y) in enumerate(xy):
            item = QListWidgetItem(f"{index}    ({x:.3f}, {y:.3f})")
            item.setData(Qt.ItemDataRole.UserRole, int(index))
            self.nodes.addItem(item)

    def _sync_target(self) -> None:
        use_edge = self._edge_radio.isChecked()
        self.edge.setVisible(use_edge)
        self._edge_caption.setVisible(use_edge)
        self.nodes.setVisible(not use_edge)
        self._nodes_caption.setVisible(not use_edge)
        self._emit_preview()

    def _emit_preview(self) -> None:
        if self._nodes_radio.isChecked():
            self.preview_nodes_changed.emit(self.selected_node_ids(), True)
        else:
            self.preview_nodes_changed.emit([], False)

    def _clear_preview(self, _result: int = 0) -> None:
        self.preview_nodes_changed.emit([], False)

    def toggle_node(self, node_id: int) -> None:
        """Select or deselect a mesh node from a viewport pick."""
        if not self._nodes_radio.isChecked():
            return
        target = int(node_id)
        for row in range(self.nodes.count()):
            item = self.nodes.item(row)
            value = item.data(Qt.ItemDataRole.UserRole)
            if value is None or int(value) != target:
                continue
            item.setSelected(not item.isSelected())
            self.nodes.scrollToItem(item)
            return

    def selected_dofs(self) -> List[str]:
        return [name for name, box in self._dof_boxes.items() if box.isChecked()]

    def selected_node_ids(self) -> List[int]:
        ids = []
        for item in self.nodes.selectedItems():
            value = item.data(Qt.ItemDataRole.UserRole)
            if value is None:
                continue
            ids.append(int(value))
        return ids

    def _on_accept(self) -> None:
        try:
            self.entries()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid boundary condition", str(exc))
            return
        self.accept()

    def entries(self) -> List[BoundaryEntry]:
        if self._edge_radio.isChecked():
            return make_boundary_entries(
                self.kind,
                self.selected_dofs(),
                edge=self.edge.currentText(),
                label=self.label_edit.text(),
                load_type=self._load.currentText(),
            )
        return make_boundary_entries(
            self.kind,
            self.selected_dofs(),
            nodes=self.selected_node_ids(),
            label=self.label_edit.text(),
            load_type=self._load.currentText(),
        )

    def entry(self) -> BoundaryEntry:
        """First entry; kept so older callers still compile."""
        return self.entries()[0]


def _as_xy(node_coords) -> np.ndarray:
    if node_coords is None:
        return np.zeros((0, 2), dtype=float)
    arr = np.asarray(node_coords, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((0, 2), dtype=float)
    return arr[:, :2]
