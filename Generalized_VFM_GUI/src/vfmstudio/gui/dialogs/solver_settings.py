"""Solver-settings dialog (mirrors the Solver property page)."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QSpinBox,
    QVBoxLayout,
)

from vfmstudio.core.document import SolverModel


class SolverSettingsDialog(QDialog):
    def __init__(self, parent=None, solver: SolverModel | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Solver settings")
        solver = solver or SolverModel()
        self.iters = QSpinBox()
        self.iters.setRange(1, 500)
        self.iters.setValue(solver.max_iterations)
        self.tol = QDoubleSpinBox()
        self.tol.setDecimals(12)
        self.tol.setRange(0.0, 1.0)
        self.tol.setValue(solver.convergence_tolerance)
        self.degree = QSpinBox()
        self.degree.setRange(0, 20)
        self.degree.setValue(solver.basis_degree)
        self.weight = QComboBox()
        self.weight.addItems(["gls", "block_diagonal", "diagonal", "ols"])
        self.weight.setCurrentText(solver.weighting)
        self.spatial = QCheckBox("Spatial correction of Γ")
        self.spatial.setChecked(solver.spatial_correction)
        self.cov_tol = QDoubleSpinBox()
        self.cov_tol.setDecimals(12)
        self.cov_tol.setRange(0.0, 1.0)
        self.cov_tol.setValue(solver.covariance_tolerance)
        form = QFormLayout()
        form.addRow("Max iterations", self.iters)
        form.addRow("Tolerance", self.tol)
        form.addRow("Γ basis degree", self.degree)
        form.addRow("Weighting", self.weight)
        form.addRow("Covariance tolerance", self.cov_tol)
        form.addRow(self.spatial)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def apply_to(self, solver: SolverModel) -> None:
        solver.max_iterations = int(self.iters.value())
        solver.convergence_tolerance = float(self.tol.value())
        solver.basis_degree = int(self.degree.value())
        solver.weighting = self.weight.currentText()
        solver.spatial_correction = self.spatial.isChecked()
        solver.covariance_tolerance = float(self.cov_tol.value())
