"""Identified coefficients with eq. (20) σ and Birge-scaled σ."""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtWidgets import (
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from vfmstudio.core import bridge
from vfmstudio.core.document import SessionDocument
from vfmstudio.core.jobs import reduced_chi_square


def _item(text) -> QTableWidgetItem:
    return QTableWidgetItem(str(text))


class CoefficientPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._banner = QLabel()
        self._banner.setWordWrap(True)
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["Parameter", "Value", "σ (eq. 20)", "σ × Birge", "σ / |θ|"]
        )
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout = QVBoxLayout(self)
        layout.addWidget(self._banner)
        layout.addWidget(self._table)

    def update_from(self, document: SessionDocument, result: Any) -> None:
        if result is None:
            self._banner.setText("No identification result.")
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
        birge = chi["birge"] if chi else 1.0
        if chi and chi["reduced"] > 2.0:
            self._banner.setText(
                f"Noise model rejected: χ²/dof = {chi['reduced']:.1f} "
                f"(Birge {birge:.2f}). Eq. (20) is variance under the model, "
                "not a statement of accuracy. Quote the Birge-scaled column."
            )
            self._banner.setProperty("role", "warn")
        elif chi:
            self._banner.setText(f"χ²/dof = {chi['reduced']:.2f}")
        else:
            self._banner.setText("Identification complete.")
        self._table.setRowCount(len(theta))
        for i, value in enumerate(theta):
            sigma = float(std[i]) if std is not None and i < len(std) else float("nan")
            scaled = sigma * birge
            rel = abs(scaled / value) if value else float("nan")
            self._table.setItem(i, 0, _item(labels[i] if i < len(labels) else f"θ{i+1}"))
            self._table.setItem(i, 1, _item(f"{value:.6g}"))
            self._table.setItem(i, 2, _item(f"{sigma:.4g}"))
            self._table.setItem(i, 3, _item(f"{scaled:.4g}"))
            self._table.setItem(i, 4, _item(f"{rel:.3g}"))
