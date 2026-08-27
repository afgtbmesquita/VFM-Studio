"""Helpers and a small dialog for creating experiments."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QVBoxLayout,
)

from vfmstudio.core.document import ExperimentModel, MeshModel, SessionDocument


def ensure_experiment(document: SessionDocument) -> ExperimentModel:
    if document.experiments:
        return document.experiments[0]
    experiment = ExperimentModel(
        experiment_id=document.next_experiment_id(),
        name="Experiment 1",
        mesh=MeshModel(),
    )
    document.experiments.append(experiment)
    return experiment


class ExperimentEditorDialog(QDialog):
    def __init__(self, parent=None, name: str = "Experiment") -> None:
        super().__init__(parent)
        self.setWindowTitle("Add experiment")
        self.name_edit = QLineEdit(name)
        form = QFormLayout()
        form.addRow("Name", self.name_edit)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def experiment_name(self) -> str:
        return self.name_edit.text().strip() or "Experiment"
