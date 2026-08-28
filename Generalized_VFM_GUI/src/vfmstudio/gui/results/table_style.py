"""Shared styling helpers for post-processing result tables."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QHeaderView,
    QLabel,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


def format_number(value, *, digits: int = 6) -> str:
    """Compact engineering number for table cells."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:  # NaN
        return "—"
    if abs(number) == 0.0:
        return "0"
    abs_value = abs(number)
    if abs_value >= 1.0e4 or abs_value < 1.0e-3:
        return f"{number:.{max(digits - 2, 3)}e}"
    return f"{number:.{digits}g}"


def text_item(
    text,
    *,
    align: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
    bold: bool = False,
    mono: bool = False,
) -> QTableWidgetItem:
    item = QTableWidgetItem("" if text is None else str(text))
    item.setTextAlignment(int(align))
    flags = item.flags()
    flags &= ~Qt.ItemFlag.ItemIsEditable
    item.setFlags(flags)
    font = QFont(item.font())
    if mono:
        font.setFamilies(["Consolas", "Courier New", "monospace"])
    if bold:
        font.setBold(True)
    item.setFont(font)
    return item


def number_item(
    value,
    *,
    digits: int = 6,
    align: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
) -> QTableWidgetItem:
    return text_item(format_number(value, digits=digits), align=align, mono=True)


def configure_results_table(table: QTableWidget) -> None:
    """Apply CAE-style read-only table chrome."""
    table.setObjectName("resultsTable")
    table.setAlternatingRowColors(True)
    table.setShowGrid(False)
    table.setWordWrap(False)
    table.setSortingEnabled(False)
    table.setCornerButtonEnabled(False)
    table.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)

    vertical = table.verticalHeader()
    vertical.setVisible(False)
    vertical.setDefaultSectionSize(30)
    vertical.setMinimumSectionSize(26)

    header = table.horizontalHeader()
    header.setObjectName("resultsHeader")
    header.setHighlightSections(False)
    header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    header.setStretchLastSection(True)
    header.setMinimumSectionSize(72)


def style_column_widths(table: QTableWidget, *, first_stretch: bool = False) -> None:
    header = table.horizontalHeader()
    columns = table.columnCount()
    if columns <= 0:
        return
    if first_stretch and columns == 1:
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        return
    header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
    for column in range(1, columns):
        mode = (
            QHeaderView.ResizeMode.Stretch
            if column == columns - 1
            else QHeaderView.ResizeMode.ResizeToContents
        )
        header.setSectionResizeMode(column, mode)


class ResultsPage(QWidget):
    """Title + status banner + styled table shell used by post-processing pages."""

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("resultsPage")

        self._title = QLabel(title)
        self._title.setObjectName("resultsTitle")

        self._banner = QLabel()
        self._banner.setObjectName("resultsBanner")
        self._banner.setWordWrap(True)
        self._banner.setProperty("role", "hint")

        self._rule = QFrame()
        self._rule.setFrameShape(QFrame.Shape.HLine)
        self._rule.setObjectName("resultsRule")

        self._table = QTableWidget(0, 0)
        configure_results_table(self._table)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(0)
        layout.addWidget(self._title)
        layout.addSpacing(4)
        layout.addWidget(self._banner)
        layout.addSpacing(8)
        layout.addWidget(self._rule)
        layout.addSpacing(10)
        self._content = QVBoxLayout()
        self._content.setContentsMargins(0, 0, 0, 0)
        self._content.setSpacing(8)
        layout.addLayout(self._content, 1)
        self._content.addWidget(self._table, 1)

    def set_banner(self, text: str, *, role: Optional[str] = "hint") -> None:
        text = str(text or "").strip()
        self._banner.setText(text)
        self._banner.setVisible(bool(text))
        self._banner.setProperty("role", role or "hint")
        self._rule.setVisible(bool(text))
        style = self._banner.style()
        if style is not None:
            style.unpolish(self._banner)
            style.polish(self._banner)


class PlotResultsPage(ResultsPage):
    """Results page with a table above a publication-style plot."""

    def __init__(self, title: str, modes: list[str], parent=None) -> None:
        super().__init__(title, parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        from vfmstudio.gui.results.plot_style import apply_matplotlib_style

        apply_matplotlib_style()
        self._mode = QComboBox()
        self._mode.addItems(modes)
        self._mode.currentTextChanged.connect(lambda _: self.refresh_plot())
        self._figure = Figure(facecolor="#FFFFFF")
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        plot_box = QWidget()
        plot_layout = QVBoxLayout(plot_box)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(6)
        plot_layout.addWidget(self._mode)
        plot_layout.addWidget(self._canvas, 1)

        self._content.removeWidget(self._table)
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._table)
        splitter.addWidget(plot_box)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setChildrenCollapsible(False)
        self._content.addWidget(splitter, 1)

    def refresh_plot(self) -> None:
        raise NotImplementedError
