"""Fisher-like H / SVD / CID summary (not the speciality H ≈ I)."""

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


def _item(text) -> QTableWidgetItem:
    return QTableWidgetItem(str(text))


class IdentifiabilityPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._label = QLabel("Run Identify → Identifiability to fill this table.")
        self._label.setWordWrap(True)
        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Quantity", "Value"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout = QVBoxLayout(self)
        layout.addWidget(self._label)
        layout.addWidget(self._table)

    def update_from(self, ident: Any) -> None:
        if ident is None:
            self._label.setText("No identifiability diagnostics yet.")
            self._table.setRowCount(0)
            return
        aggregate = getattr(ident, "aggregate", None)
        if aggregate is None:
            return
        values = [
            ("cond(H)", getattr(aggregate, "condition_number", float("nan"))),
            ("det(H)", getattr(aggregate, "determinant", float("nan"))),
            ("pseudoinverse", getattr(aggregate, "used_pseudoinverse", False)),
        ]
        singular = np.asarray(getattr(aggregate, "singular_values", []), dtype=float)
        for i, sigma in enumerate(singular):
            values.append((f"σ{i+1}", sigma))
        self._table.setRowCount(len(values))
        for i, (name, value) in enumerate(values):
            self._table.setItem(i, 0, _item(name))
            self._table.setItem(i, 1, _item(f"{value}"))
        self._label.setText(
            "Fisher-like information H from constitutive sensitivities. "
            "This is the usable identifiability check — speciality H≈I is not."
        )
