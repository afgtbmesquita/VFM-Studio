"""GLS iteration table: δQ, δC, cond-related log-dets."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


def _item(text) -> QTableWidgetItem:
    return QTableWidgetItem(str(text))


class ConvergencePanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["Iter", "δQ", "δC", "log10|C|", "log10|R|"]
        )
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout = QVBoxLayout(self)
        layout.addWidget(self._table)

    def update_from(self, result: Any) -> None:
        diagnostics = getattr(result, "iteration_diagnostics", None) or []
        self._table.setRowCount(len(diagnostics))
        for i, row in enumerate(diagnostics):
            self._table.setItem(i, 0, _item(getattr(row, "iteration", i)))
            self._table.setItem(i, 1, _item(f"{getattr(row, 'delta_q', float('nan')):.3e}"))
            self._table.setItem(
                i, 2, _item(f"{getattr(row, 'covariance_delta', float('nan')):.3e}")
            )
            self._table.setItem(
                i, 3, _item(f"{getattr(row, 'covariance_logdet', float('nan')):.3g}")
            )
            self._table.setItem(
                i, 4, _item(f"{getattr(row, 'correlation_logdet', float('nan')):.3g}")
            )
