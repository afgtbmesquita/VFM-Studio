"""Toolbar field and display-mode controls."""

from __future__ import annotations

from typing import Iterable, Optional

from PySide6.QtWidgets import QComboBox


DEFAULT_SCALARS = ("kappa_xx", "kappa_yy", "kappa_xy", "w", "u", "v", "CID")


def _label(name: str) -> str:
    from vfmstudio.core.fields import field_display_label

    return field_display_label(name)


class FieldCombo(QComboBox):
    def __init__(self, parent=None, items: Optional[Iterable[str]] = None) -> None:
        super().__init__(parent)
        self.set_fields(list(items or DEFAULT_SCALARS), current="kappa_yy")
        self.setMaximumHeight(26)
        self.setMinimumWidth(90)

    def set_fields(self, names: Iterable[str], current: Optional[str] = None) -> None:
        selected = current if current is not None else self.currentData()
        names = list(names)
        existing = [self.itemData(i) for i in range(self.count())]
        if existing == names and (
            selected is None or self.currentData() == selected
        ):
            return
        self.blockSignals(True)
        self.clear()
        for name in names:
            self.addItem(_label(name), name)
        if selected in names:
            self.setCurrentIndex(names.index(selected))
        elif names:
            self.setCurrentIndex(0)
        self.blockSignals(False)


class DisplayModeCombo(QComboBox):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.addItem("Contour", "contour")
        self.addItem("Points", "points")
        self.setCurrentIndex(0)
        self.setMaximumHeight(26)
        self.setMinimumWidth(96)
        self.setToolTip(
            "Contour: filled plot of the selected field on the DIC cloud only. "
            "Points: measurement locations in light gray."
        )