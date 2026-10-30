"""Compact colour swatch that opens a palette, then a full colour dialog."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QGridLayout,
    QMenu,
    QPushButton,
    QWidget,
    QWidgetAction,
)

# Okabe–Ito first (matches fig_deletion_shifts), then the Studio series colours.
LINE_PALETTE = (
    "#0072B2",
    "#E69F00",
    "#CC79A7",
    "#009E73",
    "#D55E00",
    "#000000",
    "#56B4E9",
    "#C59B2D",
    "#eb6834",
    "#1baf7a",
    "#4a3aa7",
    "#8f8e88",
    "#6B3FA0",
    "#3D5A40",
)


class ColorSwatchButton(QPushButton):
    """Small square that shows the current colour and opens a palette on click."""

    colorChanged = Signal(str)

    def __init__(self, color: str = "#0072B2", parent=None) -> None:
        super().__init__(parent)
        self._color = QColor(color).name()
        self.setFixedSize(22, 22)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Choose line colour")
        self.clicked.connect(self._open_palette)
        self._paint()

    def color(self) -> str:
        return self._color

    def set_color(self, color: str) -> None:
        name = QColor(color).name()
        if name == self._color:
            return
        self._color = name
        self._paint()

    def _paint(self) -> None:
        self.setStyleSheet(
            "QPushButton {"
            f" background: {self._color};"
            " border: 1px solid #666666;"
            " border-radius: 3px;"
            "}"
            "QPushButton:hover { border: 1px solid #222222; }"
        )

    def _open_palette(self) -> None:
        menu = QMenu(self)
        host = QWidget(menu)
        grid = QGridLayout(host)
        grid.setContentsMargins(6, 6, 6, 6)
        grid.setSpacing(4)
        columns = 8
        for index, hex_color in enumerate(LINE_PALETTE):
            cell = QPushButton(host)
            cell.setFixedSize(18, 18)
            cell.setCursor(Qt.CursorShape.PointingHandCursor)
            cell.setToolTip(hex_color)
            cell.setStyleSheet(
                "QPushButton {"
                f" background: {hex_color};"
                " border: 1px solid #888888;"
                "}"
                "QPushButton:hover { border: 1px solid #222222; }"
            )
            cell.clicked.connect(
                lambda _checked=False, chosen=hex_color: self._pick(chosen, menu)
            )
            grid.addWidget(cell, index // columns, index % columns)
        more = QPushButton("More…", host)
        more.clicked.connect(lambda: self._pick_custom(menu))
        grid.addWidget(more, (len(LINE_PALETTE) + columns - 1) // columns, 0, 1, columns)
        action = QWidgetAction(menu)
        action.setDefaultWidget(host)
        menu.addAction(action)
        menu.exec(self.mapToGlobal(self.rect().bottomLeft()))

    def _pick(self, color: str, menu: QMenu) -> None:
        menu.close()
        self._emit_color(color)

    def _pick_custom(self, menu: QMenu) -> None:
        menu.close()
        dialog = QColorDialog(QColor(self._color), self.window())
        dialog.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog, True)
        dialog.setWindowTitle("Line colour")
        for index, hex_color in enumerate(LINE_PALETTE):
            dialog.setCustomColor(index, QColor(hex_color))
        if dialog.exec():
            chosen = dialog.selectedColor()
            if chosen.isValid():
                self._emit_color(chosen.name())

    def _emit_color(self, color: str) -> None:
        name = QColor(color).name()
        if name == self._color:
            return
        self._color = name
        self._paint()
        self.colorChanged.emit(self._color)
