"""Per-experiment, per-load-step coefficient estimates.

These are the speciality-VFM step contributions that GLS weights together.
They are not independent identifications of θ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np
from PySide6.QtCore import Qt

from vfmstudio.core.document import (
    SessionDocument,
    parameter_math_label,
    parameter_unit_math,
)
from vfmstudio.gui.results.plot_style import (
    INK_MUTED,
    INK_PRIMARY,
    SERIES_MARKER,
    empty_message,
    experiment_colour,
    pad_limits,
    place_legend_below,
    style_axis,
)
from vfmstudio.gui.results.table_style import (
    PlotResultsPage,
    number_item,
    style_column_widths,
    text_item,
)


@dataclass
class ExperimentStepSeries:
    name: str
    experiment_id: int
    step_labels: List[str]
    values: np.ndarray  # (T, p)
    series_index: int = 0


def _as_step_matrix(raw: Any, n_par: int) -> Optional[np.ndarray]:
    if raw is None or n_par <= 0:
        return None
    array = np.asarray(raw, dtype=float)
    if array.size == 0:
        return None
    if array.ndim == 1:
        if array.size % n_par == 0:
            return array.reshape(-1, n_par)
        return None
    if array.ndim != 2:
        return None
    if array.shape[1] == n_par:
        return array
    if array.shape[0] == n_par:
        return array.T
    if array.shape[1] > n_par:
        return array[:, :n_par]
    return None


def _labels_for(document: Optional[SessionDocument], result: Any, n_par: int) -> List[str]:
    names: Sequence[str] = ()
    if document is not None:
        names = document.material.labels()
    if not names:
        model = getattr(result, "constitutive_model", None)
        names = list(getattr(model, "parameter_labels", ()) or ())
    labels = list(names)
    while len(labels) < n_par:
        labels.append(f"P{len(labels) + 1}")
    return labels[:n_par]


def _experiment_name(report: Any, document: Optional[SessionDocument], fallback: int) -> str:
    experiment_id = int(getattr(report, "experiment_id", fallback) or fallback)
    preferred = str(getattr(report, "label", "") or "").strip()
    if document is not None:
        experiment = document.experiment_by_id(experiment_id)
        if experiment is not None and (experiment.name or "").strip():
            preferred = experiment.name.strip()
    return preferred or f"Experiment {experiment_id}"


def _step_labels(report: Any, n_steps: int, document: Optional[SessionDocument]) -> List[str]:
    raw = list(getattr(report, "step_labels", None) or [])
    labels = [str(item).strip() for item in raw[:n_steps]]
    while len(labels) < n_steps:
        labels.append(f"Step {len(labels) + 1}")
    if document is None:
        return labels
    experiment = document.experiment_by_id(int(getattr(report, "experiment_id", 0) or 0))
    if experiment is None or not experiment.steps:
        return labels
    from vfmstudio.core.step_series import step_display_label

    for index, step in enumerate(experiment.steps[:n_steps]):
        text = step_display_label(step)
        if text:
            labels[index] = text
    return labels


def collect_experiment_steps(
    result: Any,
    document: Optional[SessionDocument] = None,
) -> tuple[List[ExperimentStepSeries], List[str], np.ndarray]:
    """Pull per-load-step coefficient matrices out of a GLS result."""
    theta = np.asarray(getattr(result, "final_parameters", None), dtype=float).reshape(-1)
    n_par = int(theta.size)
    labels = _labels_for(document, result, n_par)
    series: List[ExperimentStepSeries] = []
    reports = list(getattr(result, "experiments", None) or [])
    for index, report in enumerate(reports):
        values = _as_step_matrix(getattr(report, "step_estimates", None), n_par)
        if values is None:
            continue
        experiment_id = int(getattr(report, "experiment_id", index + 1) or index + 1)
        series.append(
            ExperimentStepSeries(
                name=_experiment_name(report, document, experiment_id),
                experiment_id=experiment_id,
                step_labels=_step_labels(report, int(values.shape[0]), document),
                values=values,
                series_index=len(series),
            )
        )
    return series, labels, theta


def _parameter_grid(figure, n: int):
    n_rows = max(1, (n + 1) // 2)
    n_cols = 2 if n > 1 else 1
    if n == 1:
        return [figure.add_subplot(111)], n_rows, n_cols
    axes = figure.subplots(n_rows, n_cols)
    figure.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.22, wspace=0.42, hspace=0.48)
    flat = list(np.ravel(axes))
    for extra in flat[n:]:
        extra.set_visible(False)
    return flat[:n], n_rows, n_cols


def _step_axis(values: np.ndarray) -> np.ndarray:
    return np.arange(1, int(values.shape[0]) + 1, dtype=float)


def _style_step_axis(axis, n_steps: int) -> None:
    style_axis(axis)
    if n_steps <= 12:
        axis.set_xticks(np.arange(1, n_steps + 1))
    axis.set_xlim(0.6, n_steps + 0.4)


def _colour_for(
    rec: ExperimentStepSeries,
    colors: Optional[dict[int, str]] = None,
) -> str:
    if colors:
        chosen = colors.get(rec.experiment_id)
        if chosen:
            return str(chosen)
    return experiment_colour(rec.series_index)


def _draw_series(
    axis,
    rec: ExperimentStepSeries,
    column: int,
    *,
    color: Optional[str] = None,
) -> None:
    colour = color or experiment_colour(rec.series_index)
    marker = SERIES_MARKER[rec.series_index % len(SERIES_MARKER)]
    x = _step_axis(rec.values)
    axis.plot(
        x,
        rec.values[:, column],
        color=colour,
        marker=marker,
        markeredgecolor="white",
        markeredgewidth=0.7,
        markersize=6.2,
        linewidth=1.55,
        zorder=3,
        label=rec.name,
    )


def plot_coefficients_vs_step(
    figure,
    series: Sequence[ExperimentStepSeries],
    labels: Sequence[str],
    theta: np.ndarray,
    *,
    colors: Optional[dict[int, str]] = None,
) -> None:
    n = len(labels)
    panels, n_rows, _ = _parameter_grid(figure, n)
    max_steps = max(int(item.values.shape[0]) for item in series)
    for column, axis in enumerate(panels):
        _style_step_axis(axis, max_steps)
        hat = float(theta[column]) if column < theta.size else float("nan")
        if np.isfinite(hat):
            axis.axhline(
                hat,
                color=INK_MUTED,
                linewidth=1.05,
                linestyle=(0, (4, 3)),
                zorder=2,
            )
        for rec in series:
            _draw_series(
                axis, rec, column, color=_colour_for(rec, colors)
            )
        samples = [rec.values[:, column] for rec in series]
        if np.isfinite(hat):
            samples.append(np.array([hat]))
        pad_limits(axis, np.concatenate(samples))
        axis.set_title(parameter_math_label(labels[column]), pad=8, color=INK_PRIMARY)
        unit = parameter_unit_math(labels[column])
        if unit:
            axis.set_ylabel(unit, labelpad=6)
        if np.isfinite(hat):
            axis.annotate(
                rf"$\hat\theta={hat:.3g}$",
                xy=(max_steps, hat),
                xytext=(-2, 5),
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontsize=7,
                color=INK_MUTED,
                annotation_clip=True,
            )
        if column // 2 == n_rows - 1 or n <= 2:
            axis.set_xlabel("Load step", labelpad=6)
    handles, handle_labels = panels[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D

    handles.append(
        Line2D(
            [],
            [],
            color=INK_MUTED,
            linestyle=(0, (4, 3)),
            linewidth=1.05,
            label=r"pooled FGLS  $\hat\theta$",
        )
    )
    handle_labels.append(r"pooled FGLS  $\hat\theta$")
    place_legend_below(figure, handles, handle_labels)


def plot_by_experiment(
    figure,
    series: Sequence[ExperimentStepSeries],
    labels: Sequence[str],
    theta: np.ndarray,
    *,
    colors: Optional[dict[int, str]] = None,
) -> None:
    n_exp = len(series)
    n_par = len(labels)
    if n_exp == 1:
        plot_coefficients_vs_step(figure, series, labels, theta, colors=colors)
        return
    axes = figure.subplots(n_exp, n_par, squeeze=False)
    figure.subplots_adjust(
        left=0.14,
        right=0.98,
        top=0.92,
        bottom=0.22,
        wspace=0.28,
        hspace=0.22,
    )
    for row, rec in enumerate(series):
        n_steps = int(rec.values.shape[0])
        colour = _colour_for(rec, colors)
        for column in range(n_par):
            axis = axes[row, column]
            _style_step_axis(axis, n_steps)
            hat = float(theta[column]) if column < theta.size else float("nan")
            if np.isfinite(hat):
                axis.axhline(
                    hat,
                    color=INK_MUTED,
                    linewidth=1.0,
                    linestyle=(0, (4, 3)),
                    zorder=2,
                )
            _draw_series(axis, rec, column, color=colour)
            samples = [rec.values[:, column]]
            if np.isfinite(hat):
                samples.append(np.array([hat]))
            pad_limits(axis, np.concatenate(samples))
            if row == 0:
                title = parameter_math_label(labels[column])
                unit = parameter_unit_math(labels[column])
                axis.set_title(
                    rf"{title}  ({unit})" if unit else title,
                    pad=7,
                )
            if column == 0:
                axis.set_ylabel(rec.name, labelpad=8, fontsize=9, color=INK_PRIMARY)
            unit = parameter_unit_math(labels[column])
            if unit and row == n_exp - 1:
                axis.set_xlabel("Load step", labelpad=4)
            if row < n_exp - 1:
                axis.set_xlabel("")
            if row != n_exp - 1:
                axis.tick_params(labelbottom=False)
    handles, handle_labels = axes[0, 0].get_legend_handles_labels()
    from matplotlib.lines import Line2D

    handles.append(
        Line2D(
            [],
            [],
            color=INK_MUTED,
            linestyle=(0, (4, 3)),
            linewidth=1.05,
            label=r"pooled FGLS  $\hat\theta$",
        )
    )
    handle_labels.append(r"pooled FGLS  $\hat\theta$")
    place_legend_below(figure, handles, handle_labels)


def plot_deviation(
    figure,
    series: Sequence[ExperimentStepSeries],
    labels: Sequence[str],
    theta: np.ndarray,
    *,
    colors: Optional[dict[int, str]] = None,
) -> None:
    n = len(labels)
    panels, n_rows, _ = _parameter_grid(figure, n)
    max_steps = max(int(item.values.shape[0]) for item in series)
    floor = np.finfo(float).tiny
    for column, axis in enumerate(panels):
        _style_step_axis(axis, max_steps)
        axis.axhline(0.0, color=INK_MUTED, linewidth=0.9, zorder=2)
        hat = float(theta[column]) if column < theta.size else float("nan")
        scale = max(abs(hat), floor)
        all_rel = []
        for rec in series:
            relative = 100.0 * (rec.values[:, column] - hat) / scale
            all_rel.append(relative)
            colour = _colour_for(rec, colors)
            marker = SERIES_MARKER[rec.series_index % len(SERIES_MARKER)]
            axis.plot(
                _step_axis(rec.values),
                relative,
                color=colour,
                marker=marker,
                markeredgecolor="white",
                markeredgewidth=0.7,
                markersize=6.2,
                linewidth=1.55,
                zorder=3,
                label=rec.name,
            )
        pad_limits(axis, np.concatenate(all_rel) if all_rel else np.array([0.0]))
        axis.set_title(parameter_math_label(labels[column]), pad=8)
        axis.set_ylabel(r"$(\theta_t-\hat\theta)\,/\,|\hat\theta|$   (%)", labelpad=6)
        if column // 2 == n_rows - 1 or n <= 2:
            axis.set_xlabel("Load step", labelpad=6)
    handles, handle_labels = panels[0].get_legend_handles_labels()
    place_legend_below(figure, handles, handle_labels)


class ExperimentCheckPanel(PlotResultsPage):
    def __init__(self, parent=None) -> None:
        super().__init__(
            "Individual Experiment check",
            [
                "Coefficients vs load step",
                "One panel per experiment",
                "Deviation from θ̂",
            ],
            parent,
        )
        self._result: Any = None
        self._document: Optional[SessionDocument] = None
        self._series: List[ExperimentStepSeries] = []
        self._labels: List[str] = []
        self._theta = np.zeros(0)
        self._visible: dict[int, bool] = {}
        self._colors: dict[int, str] = {}

    def update_from(self, document: SessionDocument, result: Any) -> None:
        self._document = document
        self._result = result
        if result is None:
            self._series, self._labels, self._theta = [], [], np.zeros(0)
        else:
            self._series, self._labels, self._theta = collect_experiment_steps(
                result, document
            )
        self._ensure_style_defaults()
        self._fill_table()
        self.refresh_plot()

    def apply_display(
        self,
        *,
        visible: Optional[dict[int, bool]] = None,
        colors: Optional[dict[int, str]] = None,
    ) -> None:
        if visible is not None:
            self._visible.update({int(k): bool(v) for k, v in visible.items()})
        if colors is not None:
            self._colors.update({int(k): str(v) for k, v in colors.items() if v})
        self._ensure_style_defaults()
        self.refresh_plot()

    def _ensure_style_defaults(self) -> None:
        for rec in self._series:
            self._visible.setdefault(rec.experiment_id, True)
            self._colors.setdefault(rec.experiment_id, experiment_colour(rec.series_index))

    def _drawn_series(self) -> List[ExperimentStepSeries]:
        return [
            rec
            for rec in self._series
            if self._visible.get(rec.experiment_id, True)
        ]

    def _fill_table(self) -> None:
        if self._result is None:
            self.set_banner("No identification result. Run Identify first.")
            self._table.setRowCount(0)
            return
        if not self._series:
            self.set_banner(
                "This result has no per-load-step coefficient estimates. "
                "Run FGLS identification first.",
                role="warn",
            )
            self._table.setRowCount(0)
            return
        headers = ["Experiment", "Step", "Label"] + list(self._labels)
        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        n_steps = sum(int(item.values.shape[0]) for item in self._series)
        self.set_banner(
            "Coefficient estimates at each load step of each experiment "
            "(the speciality-VFM contributions that GLS weights together). "
            "The dashed line is the pooled FGLS θ̂ — not an independent fit "
            f"per step.  {len(self._series)} experiment(s), {n_steps} step(s)."
        )
        rows: list[tuple] = []
        rows.append(("Pooled FGLS", "—", "θ̂", self._theta))
        for rec in self._series:
            for step, label in enumerate(rec.step_labels):
                rows.append((rec.name, str(step + 1), label, rec.values[step]))
        self._table.setRowCount(len(rows))
        for i, (name, step, label, values) in enumerate(rows):
            bold = i == 0
            self._table.setItem(i, 0, text_item(name, bold=True))
            self._table.setItem(
                i,
                1,
                text_item(
                    step,
                    align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                    bold=bold,
                ),
            )
            self._table.setItem(i, 2, text_item(label, bold=bold))
            for k, value in enumerate(np.asarray(values, dtype=float).ravel()):
                self._table.setItem(i, 3 + k, number_item(value))
        style_column_widths(self._table)

    def refresh_plot(self) -> None:
        self._figure.clear()
        if self._result is None:
            empty_message(self._figure, "Run Identify to fill this analysis.")
        elif not self._series:
            empty_message(self._figure, "No per-load-step estimates on this result.")
        elif not self._drawn_series():
            empty_message(self._figure, "No experiments selected. Turn a test on in Properties.")
        else:
            self._draw()
        self._canvas.draw_idle()

    def _draw(self) -> None:
        mode = self._mode.currentText()
        series = self._drawn_series()
        colors = dict(self._colors)
        if mode.startswith("One panel"):
            plot_by_experiment(
                self._figure, series, self._labels, self._theta, colors=colors
            )
        elif mode.startswith("Deviation"):
            plot_deviation(
                self._figure, series, self._labels, self._theta, colors=colors
            )
        else:
            plot_coefficients_vs_step(
                self._figure, series, self._labels, self._theta, colors=colors
            )
