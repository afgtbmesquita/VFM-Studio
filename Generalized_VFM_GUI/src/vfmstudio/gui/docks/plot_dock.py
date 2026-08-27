"""PyQtGraph plot dock: histories, SVD, information / covariance heatmaps."""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtWidgets import QComboBox, QStackedWidget, QVBoxLayout, QWidget

import pyqtgraph as pg

from vfmstudio.core.jobs import reduced_chi_square


class PlotDock(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        pg.setConfigOptions(antialias=True, background="#FFFFFF", foreground="#222222")
        self._mode = QComboBox()
        self._mode.addItems(
            [
                "θ evolution",
                "Singular values",
                "Information matrix H",
                "Parameter covariance",
                "Correlation",
            ]
        )
        self._mode.currentTextChanged.connect(lambda _: self.refresh())
        self._plot = pg.PlotWidget()
        self._image_plot = pg.PlotWidget()
        self._image_item = pg.ImageItem()
        self._image_plot.addItem(self._image_item)
        self._image_plot.getPlotItem().hideAxis("left")
        self._image_plot.getPlotItem().hideAxis("bottom")
        self._stack = QStackedWidget()
        self._stack.addWidget(self._plot)
        self._stack.addWidget(self._image_plot)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._mode)
        layout.addWidget(self._stack)
        self._result = None
        self._ident = None

    def set_results(self, result: Any, identifiability: Any = None) -> None:
        self._result = result
        self._ident = identifiability
        self.refresh()

    def refresh(self) -> None:
        mode = self._mode.currentText()
        if mode == "θ evolution":
            self._show_plot()
            self._plot_evolution()
        elif mode == "Singular values":
            self._show_plot()
            self._plot_svd()
        else:
            self._show_image()
            self._plot_matrix(mode)

    def _show_plot(self) -> None:
        self._stack.setCurrentWidget(self._plot)

    def _show_image(self) -> None:
        self._stack.setCurrentWidget(self._image_plot)

    def _plot_evolution(self) -> None:
        self._plot.clear()
        self._plot.setLogMode(x=False, y=False)
        result = self._result
        if result is None:
            return
        diagnostics = getattr(result, "iteration_diagnostics", None) or []
        if not diagnostics:
            history = getattr(result, "history", None) or []
            if not history:
                return
            xs = [getattr(row, "iteration", i) for i, row in enumerate(history)]
            params = np.vstack(
                [np.asarray(getattr(row, "parameters_after", getattr(row, "parameters"))) for row in history]
            )
        else:
            xs = [d.iteration for d in diagnostics]
            params = np.vstack([np.asarray(d.parameters) for d in diagnostics])
        colors = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#E8A444", "#E06C75"]
        for col in range(params.shape[1]):
            self._plot.plot(
                xs, params[:, col],
                pen=pg.mkPen(colors[col % len(colors)], width=2),
                symbol="o",
                symbolSize=6,
                name=f"θ{col+1}",
            )
        self._plot.setLabel("bottom", "iteration")
        self._plot.setLabel("left", "parameter")
        chi = reduced_chi_square(result)
        if chi:
            self._plot.setTitle(f"χ²/dof = {chi['reduced']:.2f}  (Birge {chi['birge']:.2f})")

    def _plot_svd(self) -> None:
        self._plot.clear()
        ident = self._ident
        if ident is None:
            return
        aggregate = getattr(ident, "aggregate", None)
        if aggregate is None:
            return
        values = np.asarray(getattr(aggregate, "singular_values"), dtype=float)
        self._plot.plot(
            np.arange(1, values.size + 1),
            values,
            pen=pg.mkPen("#4A90D9", width=2),
            symbol="s",
            symbolSize=8,
        )
        self._plot.setLabel("bottom", "mode")
        self._plot.setLabel("left", "σ")
        self._plot.setLogMode(y=True)

    def _plot_matrix(self, mode: str) -> None:
        ident = self._ident
        result = self._result
        matrix = None
        if mode.startswith("Information") and ident is not None:
            aggregate = getattr(ident, "aggregate", None)
            if aggregate is not None:
                matrix = np.asarray(getattr(aggregate, "H"), dtype=float)
        elif mode.startswith("Parameter") and result is not None:
            matrix = np.asarray(
                getattr(result, "parameter_covariance", None), dtype=float
            )
        elif mode.startswith("Correlation"):
            source = None
            if ident is not None:
                source = getattr(getattr(ident, "aggregate", None), "correlation", None)
            if source is None and result is not None:
                cov = getattr(result, "parameter_covariance", None)
                if cov is not None:
                    cov = np.asarray(cov, dtype=float)
                    std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
                    with np.errstate(divide="ignore", invalid="ignore"):
                        source = cov / np.outer(std, std)
            if source is not None:
                matrix = np.asarray(source, dtype=float)
        if matrix is None or matrix.size == 0:
            self._image_item.setImage(np.zeros((1, 1)))
            return
        self._image_item.setImage(np.asarray(matrix).T, autoLevels=True)
