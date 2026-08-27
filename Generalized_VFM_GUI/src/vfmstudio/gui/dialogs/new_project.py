"""New-project dialog."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QVBoxLayout,
)

from vfmstudio.core.document import SessionDocument, empty_document


class NewProjectDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New project")
        self.name = QLineEdit("Untitled campaign")
        self.material = QComboBox()
        self.material.addItems(["iso", "orthotropic", "anisotropic"])
        self.material.setCurrentText("orthotropic")
        self.analysis = QComboBox()
        self.analysis.addItems(["in_plane", "out_of_plane", "both"])
        self.analysis.setCurrentText("out_of_plane")
        form = QFormLayout()
        form.addRow("Description", self.name)
        form.addRow("Material", self.material)
        form.addRow("Analysis", self.analysis)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def document(self) -> SessionDocument:
        document = empty_document()
        document.description = self.name.text()
        document.material.material_type = self.material.currentText()
        document.material.analysis_type = self.analysis.currentText()
        return document
