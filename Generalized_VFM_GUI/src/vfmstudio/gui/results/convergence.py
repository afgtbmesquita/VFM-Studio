"""GLS iteration table: δQ, δC, cond-related log-dets.

Kept for optional reuse; the Post Processing tree uses the Plots page for
convergence instead of a dedicated Convergence entry.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt

from vfmstudio.gui.results.table_style import (
    ResultsPage,
    number_item,
    style_column_widths,
    text_item,
)


class ConvergencePanel(ResultsPage):
    def __init__(self, parent=None) -> None:
        super().__init__("Convergence", parent)
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(
            ["Iter", "δQ", "δC", "log10|C|", "log10|R|"]
        )
        style_column_widths(self._table)
        self.set_banner("Iteration diagnostics from the FGLS loop.")

    def update_from(self, result: Any) -> None:
        diagnostics = getattr(result, "iteration_diagnostics", None) or []
        self._table.setRowCount(len(diagnostics))
        right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        for i, row in enumerate(diagnostics):
            self._table.setItem(
                i, 0, text_item(getattr(row, "iteration", i), align=right)
            )
            self._table.setItem(
                i, 1, number_item(getattr(row, "delta_q", float("nan")), digits=3)
            )
            self._table.setItem(
                i,
                2,
                number_item(getattr(row, "covariance_delta", float("nan")), digits=3),
            )
            self._table.setItem(
                i,
                3,
                number_item(getattr(row, "covariance_logdet", float("nan")), digits=3),
            )
            self._table.setItem(
                i,
                4,
                number_item(getattr(row, "correlation_logdet", float("nan")), digits=3),
            )
        style_column_widths(self._table)
        if not diagnostics:
            self.set_banner("No iteration diagnostics yet.")
        else:
            self.set_banner("")
