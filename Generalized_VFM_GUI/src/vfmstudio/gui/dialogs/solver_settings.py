"""Solver-settings dialog (mirrors the Solver property page)."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
)

from vfmstudio.core.document import SolverModel
from vfmstudio.gui.dialogs.noise_model_help import show_noise_model_help


def _select(combo: QComboBox, key: str) -> None:
    """Select the entry whose userData is `key`, else the first one."""
    index = combo.findData(key or "")
    combo.setCurrentIndex(index if index >= 0 else 0)


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
        #  Two-axis noise weighting; "" keeps the single rule above.  Load
        #  steps of one test are repeat measurements of the same specimen and
        #  are strongly correlated, so combining them is a different job from
        #  combining independent tests.
        self.load_step = QComboBox()
        for text, key in (
            ("use Weighting above", ""),
            ("ols  (average - recommended)", "ols"),
            ("gls  (full cross-step covariance)", "gls"),
            ("block_diagonal", "block_diagonal"),
            ("diagonal", "diagonal"),
        ):
            self.load_step.addItem(text, key)
        _select(self.load_step, solver.load_step_weighting)
        self.load_step.setToolTip(
            "How the LOAD STEPS of one experiment are combined. They are "
            "repeat measurements of the same specimen, so 'gls' differences "
            "them; 'ols' averages them. No effect with one step per test.")
        self.experiment = QComboBox()
        for text, key in (
            ("use Weighting above", ""),
            ("gls  (inverse covariance - recommended)", "gls"),
            ("diagonal  (inverse variance)", "diagonal"),
            ("ols  (every experiment equal)", "ols"),
        ):
            self.experiment.addItem(text, key)
        _select(self.experiment, solver.experiment_weighting)
        self.experiment.setToolTip(
            "How the EXPERIMENTS are combined. They are independent and carry "
            "different information, so 'gls' belongs here. With anything else "
            "the reported +/- stops being a covariance.")
        self.spatial = QCheckBox("Spatial correction of Γ")
        self.spatial.setChecked(solver.spatial_correction)
        self.cov_tol = QDoubleSpinBox()
        self.cov_tol.setDecimals(12)
        self.cov_tol.setRange(0.0, 1.0)
        self.cov_tol.setValue(solver.covariance_tolerance)
        #  Two groups, because these answer two unrelated questions.
        #  The noise model decides WHAT ANSWER you get; the convergence
        #  settings only decide whether the iteration is allowed to stop.
        noise_form = QFormLayout()
        noise_form.addRow("Γ basis degree", self.degree)
        noise_form.addRow(self.spatial)
        noise_form.addRow("Weighting", self.weight)
        noise_form.addRow("Load-step estimator", self.load_step)
        noise_form.addRow("Experiment estimator", self.experiment)
        self.gamma_role = QLabel()
        self.gamma_role.setWordWrap(True)
        noise_form.addRow(self.gamma_role)
        noise_box = QGroupBox("Noise model and weighting  (Γ)")
        noise_box.setLayout(noise_form)
        self.noise_help_btn = QToolButton(noise_box)
        self.noise_help_btn.setText("?")
        self.noise_help_btn.setToolTip(
            "Explain the noise model: how Γ is built, what each setting "
            "below does, and what happens when you change them.")
        self.noise_help_btn.setFixedWidth(24)
        self.noise_help_btn.clicked.connect(
            lambda: show_noise_model_help(self))
        help_row = QHBoxLayout()
        help_row.addStretch(1)
        help_row.addWidget(self.noise_help_btn)
        noise_form.insertRow(0, help_row)

        convergence_form = QFormLayout()
        convergence_form.addRow("Max iterations", self.iters)
        convergence_form.addRow("Parameter tolerance", self.tol)
        convergence_form.addRow("Covariance tolerance", self.cov_tol)
        note = QLabel(
            "The loop stops only when BOTH are met. Neither changes the "
            "answer — only whether you reached it.")
        note.setWordWrap(True)
        convergence_form.addRow(note)
        convergence_box = QGroupBox("Convergence")
        convergence_box.setLayout(convergence_form)

        for combo in (self.weight, self.load_step, self.experiment):
            combo.currentIndexChanged.connect(self._refresh_gamma_role)
        self._refresh_gamma_role()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(noise_box)
        layout.addWidget(convergence_box)
        layout.addWidget(buttons)

    def _refresh_gamma_role(self) -> None:
        """Say whether Γ is actually doing anything with these choices."""
        load_step = self.load_step.currentData() or ""
        experiment = self.experiment.currentData() or ""
        two_stage = bool(load_step or experiment)
        resolved_load = load_step or self.weight.currentText()
        resolved_experiment = experiment or "gls"
        self.weight.setEnabled(not two_stage)

        if not two_stage and self.weight.currentText() == "ols":
            text = ("Γ is NOT USED: weighting = 'ols' sets C = I, so the "
                    "basis degree and the spatial correction change nothing.")
            active = False
        elif (two_stage and resolved_load == "ols"
              and resolved_experiment == "ols"):
            text = ("Γ is NOT USED for θ: load steps averaged and every "
                    "experiment weighted equally. It still scales the ±.")
            active = False
        elif two_stage and resolved_load == "ols":
            text = ("Γ does NOT affect the load-step average, but it sets "
                    "how the experiments are weighted against each other. "
                    "With one experiment it does nothing to θ.")
            active = True
        else:
            text = ("Γ sets the load-step weights AND the experiment "
                    "weights. Sweep the degree: if θ moves, the noise model "
                    "is carrying the result.")
            active = True
        if two_stage:
            text += " (Weighting is ignored in two-stage mode.)"
        self.gamma_role.setText(text)
        self.degree.setEnabled(active)
        self.spatial.setEnabled(active)

    def apply_to(self, solver: SolverModel) -> None:
        solver.max_iterations = int(self.iters.value())
        solver.convergence_tolerance = float(self.tol.value())
        solver.basis_degree = int(self.degree.value())
        solver.weighting = self.weight.currentText()
        solver.load_step_weighting = self.load_step.currentData() or ""
        solver.experiment_weighting = self.experiment.currentData() or ""
        solver.spatial_correction = self.spatial.isChecked()
        solver.covariance_tolerance = float(self.cov_tol.value())
