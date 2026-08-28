"""Fisher-like H / SVD / CID summary (not the speciality H ≈ I)."""

from __future__ import annotations

from typing import Any

import numpy as np

from vfmstudio.gui.results.table_style import (
    ResultsPage,
    number_item,
    style_column_widths,
    text_item,
)


class IdentifiabilityPanel(ResultsPage):
    def __init__(self, parent=None) -> None:
        super().__init__("Identifiability", parent)
        self._table.setColumnCount(2)
        self._table.setHorizontalHeaderLabels(["Quantity", "Value"])
        style_column_widths(self._table)
        self.set_banner("Run Identify → Identifiability to fill this table.")

    def update_from(self, ident: Any) -> None:
        if ident is None:
            self.set_banner("No identifiability diagnostics yet.")
            self._table.setRowCount(0)
            return
        aggregate = getattr(ident, "aggregate", None)
        if aggregate is None:
            self.set_banner("Identifiability result has no aggregate summary.")
            self._table.setRowCount(0)
            return
        values = [
            ("cond(H)", getattr(aggregate, "condition_number", float("nan"))),
            ("det(H)", getattr(aggregate, "determinant", float("nan"))),
            ("pseudoinverse", getattr(aggregate, "used_pseudoinverse", False)),
        ]
        singular = np.asarray(getattr(aggregate, "singular_values", []), dtype=float)
        for i, sigma in enumerate(singular):
            values.append((f"σ{i + 1}", sigma))
        self.set_banner("")
        self._table.setRowCount(len(values))
        for i, (name, value) in enumerate(values):
            self._table.setItem(i, 0, text_item(name, bold=True))
            if isinstance(value, bool):
                self._table.setItem(i, 1, text_item("yes" if value else "no"))
            else:
                self._table.setItem(i, 1, number_item(value))
        style_column_widths(self._table)
