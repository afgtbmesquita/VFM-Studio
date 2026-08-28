"""Identified coefficients — point estimates and working σ only."""

from __future__ import annotations

from typing import Any

import numpy as np

from vfmstudio.core import bridge
from vfmstudio.core.document import SessionDocument, parameter_unit_plain
from vfmstudio.core.jobs import reduced_chi_square
from vfmstudio.gui.results.table_style import (
    ResultsPage,
    number_item,
    style_column_widths,
    text_item,
)


class CoefficientPanel(ResultsPage):
    def __init__(self, parent=None) -> None:
        super().__init__("Identified coefficients", parent)
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(
            ["Parameter", "Value", "σ (working, eq. 20)", "σ / |θ|"]
        )
        style_column_widths(self._table)
        self.set_banner("Run Identify to fill this table.")

    def update_from(self, document: SessionDocument, result: Any) -> None:
        if result is None:
            self.set_banner("No identification result.")
            self._table.setRowCount(0)
            return
        labels = document.material.labels()
        theta = np.asarray(bridge.first_attr(result, "final_parameters"), dtype=float)
        std = bridge.first_attr(result, "parameter_std", default=None)
        if std is None:
            cov = getattr(result, "parameter_covariance", None)
            if cov is not None:
                std = np.sqrt(np.clip(np.diag(np.asarray(cov, dtype=float)), 0.0, None))
        chi = reduced_chi_square(result)
        if chi and chi["reduced"] > 2.0:
            self.set_banner(
                f"Noise model rejected: χ²/dof = {chi['reduced']:.1f}. "
                "Working σ is not the experimental covariance of θ̂ — "
                "see Post Processing → Uncertainty quantification → Cov(θ).",
                role="warn",
            )
        elif chi:
            self.set_banner(
                f"χ²/dof = {chi['reduced']:.2f}. Working σ (eq. 20) is a "
                "repeatability statement under the noise model. Robust Cov(θ) "
                "lives under Uncertainty quantification."
            )
        else:
            self.set_banner(
                "Point estimates. Robust Cov(θ) lives under Uncertainty quantification."
            )
        self._table.setRowCount(len(theta))
        for i, value in enumerate(theta):
            sigma = float(std[i]) if std is not None and i < len(std) else float("nan")
            rel = abs(sigma / value) if value else float("nan")
            name = labels[i] if i < len(labels) else f"P{i + 1}"
            unit = parameter_unit_plain(name)
            label = f"{name} ({unit})" if unit else name
            self._table.setItem(i, 0, text_item(label, bold=True))
            self._table.setItem(i, 1, number_item(value))
            self._table.setItem(i, 2, number_item(sigma, digits=4))
            self._table.setItem(i, 3, number_item(rel, digits=3))
        style_column_widths(self._table)
