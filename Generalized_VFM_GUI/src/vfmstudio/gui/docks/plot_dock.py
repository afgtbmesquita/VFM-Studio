"""Publication-style coefficient-evolution post-processing plot."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from vfmstudio.core.document import SessionDocument, parameter_math_label, parameter_unit_math
from vfmstudio.core.parameter_space import (
    is_full_combination,
    selected_experiment_ids,
    space_from_document,
    theta_for_experiment_ids,
)
from vfmstudio.gui.results.plot_style import (
    INK_MUTED,
    INK_PRIMARY,
    SERIES_COLOUR,
    SERIES_MARKER,
    apply_matplotlib_style,
    pad_limits as _pad_limits,
    style_axis as _style_axis,
)


def _parameter_history(result: Any, document: Optional[SessionDocument]):
    diagnostics = getattr(result, "iteration_diagnostics", None) or []
    if diagnostics:
        iterations = np.array([int(d.iteration) for d in diagnostics], dtype=int)
        values = np.vstack([np.asarray(d.parameters, dtype=float) for d in diagnostics])
    else:
        history = getattr(result, "history", None) or []
        if not history:
            theta = getattr(result, "final_parameters", None)
            if theta is None:
                return None, None
            iterations = np.array([0], dtype=int)
            values = np.asarray(theta, dtype=float).reshape(1, -1)
            return iterations, values
        iterations = np.array(
            [int(getattr(row, "iteration", i)) for i, row in enumerate(history)],
            dtype=int,
        )
        values = np.vstack(
            [
                np.asarray(
                    getattr(row, "parameters_after", getattr(row, "parameters")),
                    dtype=float,
                )
                for row in history
            ]
        )
    initial = getattr(result, "initial_parameters", None)
    if initial is None and document is not None:
        initial = document.material.initial_parameters
    if initial is not None and (iterations.size == 0 or int(iterations[0]) != 0):
        start = np.asarray(initial, dtype=float).reshape(1, -1)
        if start.shape[1] == values.shape[1]:
            iterations = np.concatenate(([0], iterations))
            values = np.vstack([start, values])
    return iterations, values


def _labels_for(document: Optional[SessionDocument], n: int) -> list[str]:
    labels: Sequence[str] = ()
    if document is not None:
        labels = document.material.labels()
    names = list(labels)
    while len(names) < n:
        names.append(f"P{len(names) + 1}")
    return names[:n]


class PlotDock(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        apply_matplotlib_style()
        self._title = QLabel("Coefficient evolution through the FGLS loop")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title.setWordWrap(True)
        self._title.setStyleSheet(
            f"color: {INK_PRIMARY}; font-size: 12pt; font-family: 'Times New Roman', serif;"
        )
        self._subtitle = QLabel("")
        self._subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._subtitle.setWordWrap(True)
        self._subtitle.setVisible(False)
        self._subtitle.setStyleSheet(
            f"color: {INK_MUTED}; font-size: 9pt; font-family: 'Times New Roman', serif;"
        )
        self._figure = Figure(facecolor="#FFFFFF")
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 10)
        layout.setSpacing(6)
        layout.addWidget(self._title)
        layout.addWidget(self._subtitle)
        layout.addWidget(self._canvas, 1)
        self._result = None
        self._ident = None
        self._document: Optional[SessionDocument] = None
        self._fit_names: list[str] = []

    def set_results(
        self,
        result: Any,
        identifiability: Any = None,
        document: Optional[SessionDocument] = None,
    ) -> None:
        self._result = result
        self._ident = identifiability
        if document is not None:
            self._document = document
        self._sync_fit_names()
        self.refresh()

    def apply_combination(self, document: Optional[SessionDocument] = None) -> None:
        if document is not None:
            self._document = document
        self._sync_fit_names()
        self.refresh()

    def _sync_fit_names(self) -> None:
        document = self._document
        if document is None:
            self._fit_names = []
            return
        self._fit_names = [
            experiment.name or f"Experiment {experiment.experiment_id}"
            for experiment in document.experiments_in_identification()
        ]

    def refresh(self) -> None:
        self._figure.clear()
        try:
            self._plot_results()
        except Exception as exc:
            self._figure.clear()
            self._message(f"Could not apply that combination.\n{exc}")
        self._canvas.draw()

    def _message(self, text: str) -> None:
        axis = self._figure.add_subplot(111)
        axis.set_facecolor("#FFFFFF")
        axis.axis("off")
        axis.text(
            0.5, 0.5, text, ha="center", va="center",
            color=INK_MUTED, fontsize=10, transform=axis.transAxes,
        )

    def _set_fit_subtitle(self, extra: str = "") -> None:
        names = [str(name).strip() for name in self._fit_names if str(name).strip()]
        if not names and not extra:
            self._subtitle.setText("")
            self._subtitle.setVisible(False)
            return
        if not names:
            text = extra
        elif len(names) == 1:
            text = f"Identification from {names[0]}"
        else:
            text = "Identification from " + ", ".join(names)
        if extra:
            text = f"{text}. {extra}" if names else extra
        self._subtitle.setText(text)
        self._subtitle.setVisible(True)

    def _plot_results(self) -> None:
        result = self._result
        if result is None:
            self._subtitle.setText("")
            self._subtitle.setVisible(False)
            self._message("No identification result.")
            return
        space = space_from_document(result, self._document)
        ids = selected_experiment_ids(self._document)
        if not ids:
            self._set_fit_subtitle("No tests selected.")
            self._message("Tick at least one test to show a combination.")
            return
        theta = theta_for_experiment_ids(space, ids) if space is not None else None
        if space is None:
            self._set_fit_subtitle()
            n_ticked = len(ids)
            n_all = len(self._document.experiments) if self._document else n_ticked
            if n_ticked < n_all:
                self._message(
                    "This identification has no per-test GLS blocks, so a "
                    "subset cannot be applied. Run Identify again."
                )
                return
            self._plot_evolution()
            return
        if theta is None:
            self._set_fit_subtitle(
                "Last Identify did not include every ticked test."
            )
            self._message(
                "Those tests were not in the last Identify."
            )
            return
        if not is_full_combination(space, ids):
            self._set_fit_subtitle(
                "Frozen GLS at the virtual fields of the last Identify."
            )
            self._plot_combination(
                theta, space.theta_full, _labels_for(self._document, int(theta.size))
            )
            return
        self._set_fit_subtitle()
        self._plot_evolution()

    def _plot_combination(self, theta: np.ndarray, theta_full: np.ndarray, labels: Sequence[str]) -> None:
        n = int(theta.size)
        n_rows = max(1, (n + 1) // 2)
        grid = self._figure.add_gridspec(
            n_rows, 3,
            width_ratios=[1.0, 1.0, 1.18],
            left=0.10,
            right=0.97,
            top=0.90,
            bottom=0.12,
            wspace=0.58,
            hspace=0.50,
        )
        panels = [
            self._figure.add_subplot(grid[index // 2, index % 2])
            for index in range(n)
        ]
        ax_shift = self._figure.add_subplot(grid[:, 2])
        x = np.array([0.0, 1.0])
        floor = 1.0e-12
        for index, axis in enumerate(panels):
            colour = SERIES_COLOUR[index % len(SERIES_COLOUR)]
            identify = float(theta_full[index])
            selected = float(theta[index])
            _style_axis(axis)
            axis.plot(
                x,
                [identify, selected],
                color=colour,
                linewidth=1.4,
                zorder=2,
            )
            axis.scatter(
                [0.0], [identify],
                marker="o",
                s=42,
                facecolors="white",
                edgecolors=colour,
                linewidths=1.2,
                zorder=3,
            )
            axis.scatter(
                [1.0], [selected],
                marker="*",
                s=90,
                facecolors=colour,
                edgecolors="white",
                linewidths=0.6,
                zorder=4,
            )
            axis.axhline(
                selected, color=INK_MUTED, linewidth=0.7,
                linestyle=(0, (4, 3)), zorder=1,
            )
            axis.set_xticks([0.0, 1.0])
            axis.set_xticklabels(["Identify", "Selected"])
            axis.set_xlim(-0.28, 1.28)
            _pad_limits(axis, np.array([identify, selected]))
            axis.set_title(parameter_math_label(labels[index]), pad=8)
            unit = parameter_unit_math(labels[index])
            if unit:
                axis.set_ylabel(unit, labelpad=8)
            axis.annotate(
                f"{selected:.2f}", xy=(1.0, selected),
                xytext=(-4, 6), textcoords="offset points",
                ha="right", va="bottom", fontsize=8, color=colour,
                annotation_clip=True,
            )

        for index, label in enumerate(labels):
            hat = max(abs(float(theta_full[index])), floor)
            relative = abs(float(theta[index]) - float(theta_full[index])) / hat
            ax_shift.bar(
                index,
                max(relative, floor),
                color=SERIES_COLOUR[index % len(SERIES_COLOUR)],
                width=0.62,
                zorder=3,
                label=parameter_math_label(label),
            )
        _style_axis(ax_shift)
        ax_shift.set_yscale("log")
        ax_shift.set_xticks(range(n))
        ax_shift.set_xticklabels([parameter_math_label(name) for name in labels])
        ax_shift.set_xlim(-0.7, n - 0.3)
        ax_shift.set_ylabel(r"$|\theta_S-\hat{\theta}|\,/\,|\hat{\theta}|$", labelpad=10)
        ax_shift.set_title("Shift from Identify", pad=8)

    def _plot_evolution(self) -> None:
        result = self._result
        iterations, values = _parameter_history(result, self._document)
        if iterations is None or values is None or values.size == 0:
            self._message("No iteration history on this result.")
            return
        n = int(values.shape[1])
        labels = _labels_for(self._document, n)
        n_rows = max(1, (n + 1) // 2)
        grid = self._figure.add_gridspec(
            n_rows, 3,
            width_ratios=[1.0, 1.0, 1.18],
            left=0.10,
            right=0.97,
            top=0.90,
            bottom=0.12,
            wspace=0.58,
            hspace=0.50,
        )
        panels = [
            self._figure.add_subplot(grid[index // 2, index % 2])
            for index in range(n)
        ]
        ax_rate = self._figure.add_subplot(grid[:, 2])
        last_iter = float(iterations[-1])

        for index, axis in enumerate(panels):
            colour = SERIES_COLOUR[index % len(SERIES_COLOUR)]
            marker = SERIES_MARKER[index % len(SERIES_MARKER)]
            series = values[:, index]
            final = float(series[-1])
            _style_axis(axis)
            axis.axhline(
                final, color=INK_MUTED, linewidth=0.7,
                linestyle=(0, (4, 3)), zorder=1,
            )
            axis.plot(
                iterations, series, color=colour, marker=marker,
                markeredgecolor="white", markeredgewidth=0.6,
                linewidth=1.4, markersize=5.0, zorder=3,
            )
            axis.set_xlim(-0.28, last_iter + 0.28)
            axis.set_xticks(iterations)
            _pad_limits(axis, series)
            axis.set_title(parameter_math_label(labels[index]), pad=8)
            unit = parameter_unit_math(labels[index])
            if unit:
                axis.set_ylabel(unit, labelpad=8)
            if index // 2 == n_rows - 1:
                axis.set_xlabel("FGLS iteration", labelpad=8)
            axis.annotate(
                f"{final:.2f}", xy=(iterations[-1], final),
                xytext=(-4, 6), textcoords="offset points",
                ha="right", va="bottom", fontsize=8, color=colour,
                annotation_clip=True,
            )

        floor = 1.0e-12
        for index, label in enumerate(labels):
            hat = max(abs(float(values[-1, index])), floor)
            relative = np.abs(values[:, index] - values[-1, index]) / hat
            x = iterations[:-1] if iterations.size > 1 else iterations
            y = np.maximum(relative[:-1] if iterations.size > 1 else relative, floor)
            ax_rate.plot(
                x, y,
                color=SERIES_COLOUR[index % len(SERIES_COLOUR)],
                marker=SERIES_MARKER[index % len(SERIES_MARKER)],
                markeredgecolor="white", markeredgewidth=0.6,
                linewidth=1.4, markersize=5.0,
                label=parameter_math_label(label), zorder=3,
            )
        _style_axis(ax_rate)
        ax_rate.set_yscale("log")
        if iterations.size > 1:
            ax_rate.set_xticks(iterations[:-1])
            ax_rate.set_xlim(-0.28, float(iterations[-2]) + 0.28)
        ax_rate.set_xlabel("FGLS iteration", labelpad=8)
        ax_rate.set_ylabel(r"$|\theta_k-\hat{\theta}|\,/\,|\hat{\theta}|$", labelpad=10)
        ax_rate.set_title("Convergence", pad=8)
        ax_rate.legend(
            ncol=2, loc="upper right", fontsize=8,
            borderaxespad=0.8, handlelength=1.5, columnspacing=1.0,
        )
