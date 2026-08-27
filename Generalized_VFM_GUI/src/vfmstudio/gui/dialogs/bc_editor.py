"""Add PRESCRIBED / LINKING / EXTERNAL groups."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QVBoxLayout,
)

from vfmstudio.core.document import BoundaryEntry


class AddBcDialog(QDialog):
    def __init__(self, parent=None, kind: str = "prescribed") -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Add {kind} condition")
        self.kind = kind
        self.label_edit = QLineEdit()
        self.dof = QComboBox()
        self.dof.addItems(["W", "W_X", "W_Y", "W_XY", "U", "V"])
        if kind == "external":
            self.dof.setCurrentText("W_Y")
        self.edge = QComboBox()
        self.edge.addItems(["top", "bottom", "left", "right"])
        form = QFormLayout()
        form.addRow("Label", self.label_edit)
        form.addRow("DOF", self.dof)
        form.addRow("Edge", self.edge)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def entry(self) -> BoundaryEntry:
        label = self.label_edit.text().strip() or (
            f"{self.dof.currentText()} @{self.edge.currentText()}"
        )
        payload = {
            "label": label,
            "dof": self.dof.currentText(),
            "edge": self.edge.currentText(),
            "value": 0.0 if self.kind != "external" else None,
        }
        if self.kind == "external":
            payload["loadType"] = "distributed"
            payload["sortDirection"] = (
                "x" if self.edge.currentText() in {"top", "bottom"} else "y"
            )
        return BoundaryEntry.model_validate(payload)
