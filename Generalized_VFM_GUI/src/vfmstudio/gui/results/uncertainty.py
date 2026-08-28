"""Three distinct UQ post-processing pages, plus a landing explainer."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QScrollArea, QVBoxLayout, QWidget

from vfmstudio.core.document import SessionDocument, parameter_unit_plain
from vfmstudio.core.uncertainty import UncertaintyResult
from vfmstudio.gui.results import uncertainty_plots as plots
from vfmstudio.gui.results.plot_style import empty_message
from vfmstudio.gui.results.table_style import (
    PlotResultsPage,
    number_item,
    style_column_widths,
    text_item,
)


def _verdict_label(raw: str) -> str:
    return str(raw or "").replace("_", " ")


class UncertaintyLanding(QWidget):
    """Parent page: when the sandwich is the quoted Cov(θ), and when it is not."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("resultsPage")
        title = QLabel("Uncertainty quantification")
        title.setObjectName("resultsTitle")
        self._verdict = QLabel()
        self._verdict.setObjectName("resultsBanner")
        self._verdict.setWordWrap(True)
        self._verdict.setTextFormat(Qt.TextFormat.RichText)
        self._verdict.setProperty("role", "hint")
        rule = QFrame()
        rule.setFrameShape(QFrame.Shape.HLine)
        rule.setObjectName("resultsRule")

        def card(heading: str, body: str) -> QWidget:
            box = QWidget()
            layout = QVBoxLayout(box)
            layout.setContentsMargins(0, 10, 0, 6)
            layout.setSpacing(4)
            head = QLabel(heading)
            head.setStyleSheet("font-weight: 600; font-size: 11pt; color: #1F1F1F;")
            text = QLabel(body)
            text.setWordWrap(True)
            text.setStyleSheet("color: #555555;")
            layout.addWidget(head)
            layout.addWidget(text)
            return box

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(0)
        layout.addWidget(title)
        layout.addSpacing(4)
        layout.addWidget(self._verdict)
        layout.addSpacing(8)
        layout.addWidget(rule)
        layout.addWidget(
            card(
                "When the sandwich is usable",
                "GLS gives θ̂ from any M ≥ 1. The cluster sandwich "
                "Cov(θ̂) = A⁻¹ B A⁻ᵀ, with A = Σᵢ Hᵢᵀ Wᵢ Hᵢ and B = Σᵢ ψᵢ ψᵢᵀ "
                "(CR2/CR3), is the covariance that does not assume Ωᵢ = Cᵢ. "
                "Because Σᵢ ψᵢ = 0, rank(B) ≤ min(M − 1, p), so you need "
                "M ≥ p + 1 independent coupons. Quote that sandwich σ — not HAC, "
                "not Birge, not φC + τ²I, and not a confidence interval. "
                "It remains variability around the GLS θ*, not a cure for a wrong "
                "mean model or errors-in-variables.",
            )
        )
        layout.addWidget(
            card(
                "When M < p + 1",
                "There is no well-posed cluster-robust standard deviation: "
                "replication only spans an (M − 1)-dimensional subspace of Rᵖ. "
                "A pseudo-inverse of B would invent tightness in the unseen "
                "directions. Working σ = √(A⁻¹)ₖₖ still exists, but it is not "
                "that missing robust SD — it is repeatability if C were Ω. "
                "Read Information geometry of A instead.",
            )
        )
        layout.addWidget(
            card(
                "Information geometry",
                "A and cond(A): which combinations of Dᵢⱼ the pooled design "
                "actually constrains. Aᵢ = Hᵢᵀ Wᵢ Hᵢ per test: which coupon "
                "supplies which direction. cond(A − Aᵢ) when a test is dropped: "
                "how the pooled geometry changes. Residual χ²/dof: C is "
                "optimistic on average, as a warning, not as a per-coefficient σ.",
            )
        )
        layout.addWidget(
            card(
                "Cov(θ)",
                "Covariance of the pooled estimator θ̂. Quote the sandwich "
                "only when M ≥ p + 1. Report σ, not a confidence interval. "
                "Working σ is shown either way; sandwich columns are empty "
                "when the rank bound is violated.",
            )
        )
        layout.addWidget(
            card(
                "Experiment uncertainty",
                "How hard each independent experiment pulls on the estimating "
                "equation: cluster scores ψᵢ and the meat B. This is the "
                "variability of the tests, not Cov(θ).",
            )
        )
        layout.addWidget(
            card(
                "Matrices A and B",
                "The two matrices of the sandwich. A (eq. 31) is sensitivity; "
                "B (eq. 33) is between-experiment score scatter. Cov(θ) is "
                "A⁻¹ B A⁻ᵀ; they are not themselves a covariance.",
            )
        )
        layout.addWidget(
            card(
                "Leave-one-out",
                "What happens to θ̂ if one whole experiment is dropped (never a "
                "load step). Precision contribution Uᵢₖ and influence Iᵢₖ are "
                "not the same thing, and neither is a covariance estimator.",
            )
        )
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        self.update_from(None)

    def update_from(self, uncertainty: Optional[UncertaintyResult]) -> None:
        if uncertainty is None:
            self._verdict.setText(
                "The cluster sandwich is the quoted Cov(θ̂) when C is wrong "
                "<b>and</b> there are enough independent coupons. "
                "Run Identify to apply the bound M ≥ p + 1 to this stacking."
            )
            self._verdict.setProperty("role", "hint")
        else:
            m = uncertainty.n_clusters
            p = uncertainty.n_parameters
            need = p + 1
            bound = uncertainty.rank_bound
            chi = uncertainty.chi2.get("reduced")
            chi_bit = (
                f" Residual χ²/dof = {chi:.1f} (C optimistic on average if this "
                "is large; not a per-coefficient σ)."
                if chi is not None and chi == chi
                else ""
            )
            if uncertainty.well_posed:
                self._verdict.setText(
                    f"<b>Sandwich is usable.</b> M = {m} experiments, "
                    f"p = {p} parameters (need M ≥ p + 1 = {need}). "
                    f"rank(B) ≤ min(M − 1, p) = {bound}. "
                    f"Quote σ from the cluster sandwich (CR2/CR3), not a "
                    f"confidence interval — B is still noisy at this M.{chi_bit}"
                )
                self._verdict.setProperty("role", "ok")
            else:
                self._verdict.setText(
                    f"<b>Sandwich is not well posed.</b> M = {m} experiments, "
                    f"p = {p} parameters (need M ≥ p + 1 = {need}). "
                    f"rank(B) ≤ min(M − 1, p) = {bound} &lt; p, so a full-rank "
                    "robust standard deviation cannot be estimated. "
                    "Do not manufacture an SD. Read Information geometry of A. "
                    "Working σ is repeatability if C were Ω, not the missing "
                    f"robust SD.{chi_bit}"
                )
                self._verdict.setProperty("role", "warn")
        style = self._verdict.style()
        if style is not None:
            style.unpolish(self._verdict)
            style.polish(self._verdict)


class _UncertaintyPage(PlotResultsPage):
    def __init__(self, title: str, modes: list[str], parent=None) -> None:
        super().__init__(title, modes, parent)
        self._result: Optional[UncertaintyResult] = None
        self._document: Optional[SessionDocument] = None

    def update_from(
        self,
        document: SessionDocument,
        uncertainty: Optional[UncertaintyResult],
    ) -> None:
        self._document = document
        self._result = uncertainty
        self._fill_table()
        self.refresh_plot()

    def _fill_table(self) -> None:
        raise NotImplementedError

    def _draw(self) -> None:
        raise NotImplementedError

    def refresh_plot(self) -> None:
        self._figure.clear()
        if self._result is None:
            empty_message(self._figure, "Run Identify to fill this analysis.")
        else:
            self._draw()
        self._canvas.draw_idle()


class CovariancePanel(_UncertaintyPage):
    def __init__(self, parent=None) -> None:
        super().__init__(
            "Estimation of Cov(θ)",
            [
                "Identified θ ± σ",
                "Uncertainty ladder",
                "Robust covariance",
                "Working covariance",
                "Correlation",
                "Misfit spectrum",
            ],
            parent,
        )

    def _fill_table(self) -> None:
        result = self._result
        if result is None:
            self.set_banner("No identification result. Run Identify first.")
            self._table.setRowCount(0)
            return
        headers = [
            "Parameter",
            "θ̂",
            "σ work",
            f"σ {result.kind}",
            "σ CR3",
            "robust / work",
            "σ / |θ|",
        ]
        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        if result.well_posed:
            chi = result.chi2.get("reduced")
            extra = f"  ·  χ²/dof = {chi:.1f}" if chi is not None and chi == chi else ""
            self.set_banner(
                "Quote the cluster sandwich (eq. 34, "
                f"{result.kind}), not working σ, not Birge, not a confidence "
                "interval. B is still noisy at this M; CR2 and CR3 sit next "
                "to each other for that reason. Working σ is repeatability "
                "if C were Ω."
                + extra
            )
        else:
            need = result.n_parameters + 1
            self.set_banner(
                f"Robust Cov(θ) refused: M = {result.n_clusters} experiments, "
                f"p = {result.n_parameters} parameters (need M ≥ p + 1 = {need}). "
                "Replication spans an (M − 1)-dimensional subspace; do not "
                "manufacture a robust SD. Working σ is shown — it is not that "
                "missing SD. Read Information geometry of A.",
                role="warn",
            )
        self._table.setRowCount(len(result.theta))
        for i, value in enumerate(result.theta):
            name = result.labels[i]
            unit = parameter_unit_plain(name)
            label = f"{name} ({unit})" if unit else name
            work = float(result.sigma_work[i])
            robust = float(result.sigma_robust[i]) if result.well_posed else float("nan")
            cr3 = float(result.sigma_cr3[i]) if result.well_posed else float("nan")
            ratio = robust / work if result.well_posed and work else float("nan")
            quoted = robust if result.well_posed else work
            rel = abs(quoted / value) if value else float("nan")
            self._table.setItem(i, 0, text_item(label, bold=True))
            self._table.setItem(i, 1, number_item(value))
            self._table.setItem(i, 2, number_item(work, digits=4))
            self._table.setItem(i, 3, number_item(robust, digits=4))
            self._table.setItem(i, 4, number_item(cr3, digits=4))
            self._table.setItem(i, 5, number_item(ratio, digits=3))
            self._table.setItem(i, 6, number_item(rel, digits=3))
        style_column_widths(self._table)

    def _draw(self) -> None:
        result = self._result
        mode = self._mode.currentText()
        if mode.startswith("Identified"):
            plots.plot_theta_bands(self._figure, result)
        elif mode.startswith("Uncertainty"):
            plots.plot_uncertainty_ladder(self._figure, result)
        elif mode.startswith("Robust"):
            plots.plot_covariance_matrix(self._figure, result, "robust")
        elif mode.startswith("Working"):
            plots.plot_covariance_matrix(self._figure, result, "work")
        elif mode.startswith("Correlation"):
            plots.plot_covariance_matrix(self._figure, result, "corr")
        else:
            plots.plot_misfit_spectrum(self._figure, result)


class ExperimentUncertaintyPanel(_UncertaintyPage):
    def __init__(self, parent=None) -> None:
        super().__init__(
            "Uncertainty of the experiments",
            [
                "Score norms",
                "Score components",
                "Sandwich anatomy",
            ],
            parent,
        )

    def _fill_table(self) -> None:
        result = self._result
        if result is None:
            self.set_banner("No identification result. Run Identify first.")
            self._table.setRowCount(0)
            return
        headers = ["Experiment", "Steps", "ψᵀ A⁻¹ ψ"] + [
            f"ψ {name}" for name in result.labels
        ]
        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        self.set_banner(
            "How hard each independent experiment pulls on the estimating "
            "equation (eq. 32). This is not Cov(θ)."
        )
        self._table.setRowCount(result.n_clusters)
        for i, name in enumerate(result.cluster_names):
            self._table.setItem(i, 0, text_item(name, bold=True))
            self._table.setItem(
                i,
                1,
                text_item(
                    str(result.cluster_n_steps[i]),
                    align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                ),
            )
            self._table.setItem(i, 2, number_item(result.score_norms[i], digits=4))
            for k in range(result.n_parameters):
                self._table.setItem(
                    i, 3 + k, number_item(result.scores[i, k], digits=4)
                )
        style_column_widths(self._table)

    def _draw(self) -> None:
        result = self._result
        mode = self._mode.currentText()
        if mode.startswith("Score n"):
            plots.plot_cluster_scores(self._figure, result)
        elif mode.startswith("Score c"):
            plots.plot_score_components(self._figure, result)
        else:
            plots.plot_sandwich_anatomy(self._figure, result)


class MatricesPanel(_UncertaintyPage):
    def __init__(self, parent=None) -> None:
        super().__init__(
            "Matrices A and B",
            ["A and B", "A (sensitivity)", "B (variability)"],
            parent,
        )

    def _fill_table(self) -> None:
        result = self._result
        if result is None:
            self.set_banner("No identification result. Run Identify first.")
            self._table.setRowCount(0)
            return
        labels = result.labels
        headers = ["Matrix", "Row"] + list(labels)
        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        self.set_banner(
            "A (eq. 31) is sensitivity of the identifying equation; "
            "B (eq. 33) is between-experiment score variability. "
            "These are not Cov(θ). cond(A) and the allocation of Aᵢ live "
            "on Information geometry."
        )
        blocks = (("A", result.A), ("B", result.B))
        self._table.setRowCount(len(blocks) * len(labels))
        row = 0
        for name, matrix in blocks:
            for i, label in enumerate(labels):
                self._table.setItem(row, 0, text_item(name, bold=True))
                self._table.setItem(row, 1, text_item(label, bold=True))
                for j in range(len(labels)):
                    self._table.setItem(row, 2 + j, number_item(matrix[i, j]))
                row += 1
        style_column_widths(self._table)

    def _draw(self) -> None:
        result = self._result
        mode = self._mode.currentText()
        if mode.startswith("A ("):
            plots.plot_ab_matrices(self._figure, result, "A")
        elif mode.startswith("B"):
            plots.plot_ab_matrices(self._figure, result, "B")
        else:
            plots.plot_ab_matrices(self._figure, result, "both")


class InformationGeometryPanel(_UncertaintyPage):
    def __init__(self, parent=None) -> None:
        super().__init__(
            "Information geometry of A",
            [
                "cond(A − Aᵢ)",
                "Share of A",
                "Eigenvalues of A",
                "Constrained combinations",
            ],
            parent,
        )

    def _fill_table(self) -> None:
        result = self._result
        if result is None:
            self.set_banner("No identification result. Run Identify first.")
            self._table.setRowCount(0)
            return
        info = result.information
        headers = [
            "Experiment",
            "Steps",
            "cond(A − Aᵢ)",
            "cond / cond(A)",
            "rank(A − Aᵢ)",
        ] + [f"share {name}" for name in result.labels]
        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        chi = info.chi2_reduced
        chi_text = (
            f" Residual χ²/dof = {chi:.1f} (C optimistic on average if this "
            "is large; not a per-coefficient σ)."
            if chi == chi
            else ""
        )
        cond = info.condition
        cond_text = f"{cond:.3g}" if np.isfinite(cond) else "∞"
        if info.well_posed:
            pose = (
                f"Sandwich is usable (M = {info.n_clusters} ≥ p + 1 = "
                f"{info.min_clusters_for_sandwich}). A still says how "
                "information is allocated across tests."
            )
            role = "hint"
        else:
            pose = (
                f"Sandwich refused (M = {info.n_clusters} < p + 1 = "
                f"{info.min_clusters_for_sandwich}). Do not manufacture an SD; "
                "this page is what to read."
            )
            role = "warn"
        self.set_banner(
            f"{pose}  cond(A) = {cond_text}.{chi_text}",
            role=role,
        )
        self._table.setRowCount(len(info.contributions))
        right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter

        def cond_item(value):
            if not np.isfinite(value):
                return text_item("∞", align=right, mono=True)
            return number_item(value, digits=3)

        for i, row in enumerate(info.contributions):
            self._table.setItem(i, 0, text_item(row.name, bold=True))
            self._table.setItem(i, 1, text_item(str(row.n_steps), align=right))
            self._table.setItem(i, 2, cond_item(row.remaining_condition))
            self._table.setItem(i, 3, cond_item(row.condition_ratio))
            self._table.setItem(i, 4, text_item(str(row.remaining_rank), align=right))
            for k, share in enumerate(row.share):
                self._table.setItem(i, 5 + k, number_item(share, digits=3))
        style_column_widths(self._table)

    def _draw(self) -> None:
        result = self._result
        mode = self._mode.currentText()
        if mode.startswith("cond"):
            plots.plot_deletion_condition(self._figure, result)
        elif mode.startswith("Share"):
            plots.plot_information_share(self._figure, result)
        elif mode.startswith("Eigen"):
            plots.plot_a_spectrum(self._figure, result)
        else:
            plots.plot_a_directions(self._figure, result)


class LeaveOneOutPanel(_UncertaintyPage):
    def __init__(self, parent=None) -> None:
        super().__init__(
            "Leave-one-out test",
            [
                "θ without each experiment",
                "Precision contribution U",
                "Influence I",
                "U versus I",
            ],
            parent,
        )

    def _fill_table(self) -> None:
        result = self._result
        if result is None:
            self.set_banner("No identification result. Run Identify first.")
            self._table.setRowCount(0)
            return
        headers = [
            "Dropped",
            "Parameter",
            "U_ik",
            "I_ik",
            "D_i",
            "Verdict",
            "θ_(-i) fixed",
            "θ_(-i) refit",
        ]
        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        has_refit = any(
            row.theta_deleted_refit is not None for row in result.deletions
        )
        extra = (
            " Dots in the plot are FGLS refits; crosses are the fixed-weight identity."
            if has_refit
            else " Use Properties → Refit FGLS on deletion to re-estimate Gamma."
        )
        if not result.deletions:
            extra = " Need at least two experiments to drop one."
        self.set_banner(
            "What happens to θ̂ if one whole experiment is dropped. "
            "U_ik (does it tighten this coefficient?) is not I_ik "
            "(does it move this coefficient?)."
            + extra
        )
        rows = []
        for row in result.deletions:
            for k, label in enumerate(result.labels):
                refit = (
                    float(row.theta_deleted_refit[k])
                    if row.theta_deleted_refit is not None
                    else float("nan")
                )
                rows.append(
                    (
                        row.dropped,
                        label,
                        float(row.precision_contribution[k]),
                        float(row.influence[k]),
                        float(row.cook_distance),
                        _verdict_label(row.verdicts[k]),
                        float(row.theta_deleted_fixed[k]),
                        refit,
                    )
                )
        self._table.setRowCount(len(rows))
        for i, record in enumerate(rows):
            self._table.setItem(i, 0, text_item(record[0], bold=True))
            self._table.setItem(i, 1, text_item(record[1]))
            self._table.setItem(i, 2, number_item(record[2], digits=3))
            self._table.setItem(i, 3, number_item(record[3], digits=3))
            self._table.setItem(i, 4, number_item(record[4], digits=3))
            self._table.setItem(i, 5, text_item(record[5]))
            self._table.setItem(i, 6, number_item(record[6]))
            self._table.setItem(i, 7, number_item(record[7]))
        style_column_widths(self._table)

    def _draw(self) -> None:
        result = self._result
        mode = self._mode.currentText()
        if mode.startswith("θ") or mode.startswith("theta"):
            plots.plot_deletion_shifts(self._figure, result)
        elif mode.startswith("Precision"):
            plots.plot_precision_heatmap(self._figure, result)
        elif mode.startswith("Influence"):
            plots.plot_influence_heatmap(self._figure, result)
        else:
            plots.plot_information_vs_influence(self._figure, result)
