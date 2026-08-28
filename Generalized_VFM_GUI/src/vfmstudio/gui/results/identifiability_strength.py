"""Information gain G_k and per-experiment deletion influence from eq. (63)."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
from matplotlib.patches import Patch

from vfmstudio.core.document import SessionDocument
from vfmstudio.core.uncertainty import (
    ExperimentShift,
    UncertaintyResult,
    assemble_sensitivity,
    default_baseline_flags,
    identifiability_strength as compute_strength,
    subset_deletion_shifts,
)
from vfmstudio.core.uncertainty.information import G_HIGH, IdentifiabilityStrength
from vfmstudio.gui.results.plot_style import (
    INK_MUTED,
    INK_PRIMARY,
    empty_message,
    experiment_colour,
    parameter_tick_labels,
    place_legend_below,
    style_axis,
)
from vfmstudio.gui.results.table_style import (
    PlotResultsPage,
    number_item,
    style_column_widths,
    text_item,
)

GAIN_POSITIVE = "#009E73"
GAIN_NEAR_ZERO = "#8f8e88"
GAIN_NEGATIVE = "#D55E00"
SHIFT_UNITS = "units"
SHIFT_RELATIVE = "relative"


def scaled_deletion_shifts(
    shifts: Sequence[ExperimentShift],
    theta: np.ndarray,
    *,
    relative: bool,
) -> list[np.ndarray]:
    """Eq. (63) in parameter units, or (θ̂_(-i) − θ̂) / |θ̂|."""
    theta = np.asarray(theta, dtype=float).reshape(-1)
    denom = np.maximum(np.abs(theta), np.finfo(float).tiny) if relative else None
    values = []
    for row in shifts:
        shift = np.asarray(row.shift, dtype=float)
        if denom is None:
            values.append(shift)
            continue
        n = min(shift.size, denom.size)
        scaled = np.full(shift.shape, np.nan)
        scaled[:n] = shift[:n] / denom[:n]
        values.append(scaled)
    return values


def _gain_colours(values: np.ndarray) -> list[str]:
    colours = []
    for value in values:
        if not np.isfinite(value) or abs(float(value)) < 0.05:
            colours.append(GAIN_NEAR_ZERO)
        elif float(value) > 0.0:
            colours.append(GAIN_POSITIVE)
        else:
            colours.append(GAIN_NEGATIVE)
    return colours


def plot_identifiability_strength(
    figure,
    labels: Sequence[str],
    metrics: IdentifiabilityStrength,
    shifts: Sequence[ExperimentShift] | None = None,
    *,
    theta: np.ndarray | None = None,
    relative: bool = False,
) -> None:
    axes = figure.subplots(1, 2)
    figure.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.22, wspace=0.38)
    ticks = np.arange(len(labels), dtype=float)
    names = parameter_tick_labels(labels)
    series = list(shifts or ())
    theta = np.zeros(len(labels)) if theta is None else np.asarray(theta, dtype=float)
    displayed = scaled_deletion_shifts(series, theta, relative=relative)
    shift_ylabel = "Δθ / |θ̂|" if relative else "Δθ"

    robust = axes[0]
    style_axis(robust, grid="y")
    robust.set_xticks(ticks)
    robust.set_xticklabels(names)
    robust.set_ylabel(shift_ylabel)
    robust.set_title("Deletion influence (eq. 63)", pad=8)
    if len(series) < 2:
        robust.text(
            0.5,
            0.5,
            "Need at least two experiments\nfor deletion influence (eq. 63).",
            transform=robust.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color=INK_MUTED,
        )
    else:
        n_exp = len(series)
        width = 0.8 / n_exp
        samples = [0.0]
        for j, (row, values) in enumerate(zip(series, displayed)):
            offset = (j - 0.5 * (n_exp - 1)) * width
            robust.bar(
                ticks + offset,
                np.nan_to_num(values),
                width * 0.92,
                color=experiment_colour(j),
                edgecolor="white",
                linewidth=0.35,
                zorder=3,
                label=row.name,
            )
            samples.extend(values[np.isfinite(values)].tolist())
        robust.axhline(0.0, color=INK_MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
        finite = np.asarray(samples, dtype=float)
        finite = finite[np.isfinite(finite)]
        lo = float(np.min(finite)) if finite.size else -1.0
        hi = float(np.max(finite)) if finite.size else 1.0
        span = max(hi - lo, 0.2)
        robust.set_ylim(lo - 0.12 * span, hi + 0.18 * span)

    gain_ax = axes[1]
    style_axis(gain_ax, grid="y")
    gain = np.asarray(metrics.information_gain, dtype=float)
    finite_gain = gain[np.isfinite(gain)]
    if finite_gain.size == 0:
        gain_ax.set_xticks(ticks)
        gain_ax.set_xticklabels(names)
        gain_ax.set_ylabel(r"$G_k$")
        gain_ax.set_title("Information gained by adding tests", pad=8)
        gain_ax.text(
            0.5,
            0.5,
            "Untick added tests in Properties\nto measure information gain.",
            transform=gain_ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color=INK_MUTED,
        )
    else:
        gain_ax.bar(
            ticks,
            np.nan_to_num(gain),
            color=_gain_colours(gain),
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        gain_ax.axhline(0.0, color=INK_MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
        gain_ax.axhline(
            G_HIGH, color=INK_MUTED, linewidth=0.7, linestyle=(0, (2, 3)), zorder=2
        )
        lo = min(0.0, float(np.min(finite_gain)))
        hi = max(G_HIGH, float(np.max(finite_gain)))
        span = max(hi - lo, 0.2)
        gain_ax.set_ylim(lo - 0.12 * span, hi + 0.18 * span)
        gain_ax.set_xticks(ticks)
        gain_ax.set_xticklabels(names)
        gain_ax.set_ylabel(r"$G_k$")
        gain_ax.set_title(
            r"Information gain  $\log[(A_{\mathrm{base}}^{-1})_{kk}/(A_{\mathrm{all}}^{-1})_{kk}]$",
            pad=8,
        )
        for x, value in zip(ticks, gain):
            if np.isfinite(value):
                va = "bottom" if value >= 0.0 else "top"
                gain_ax.text(
                    x,
                    value,
                    f"{value:.3g}",
                    ha="center",
                    va=va,
                    fontsize=7,
                    color=INK_PRIMARY,
                )

    if len(series) >= 2:
        handles = [
            Patch(
                facecolor=experiment_colour(j),
                edgecolor="white",
                label=row.name,
            )
            for j, row in enumerate(series)
        ]
        place_legend_below(
            figure,
            handles,
            [item.get_label() for item in handles],
            caption=(
                "Left: eq. (63) for every test"
                + (
                    ", relative to |θ̂|."
                    if relative
                    else ", in the original parameter units."
                )
                + " Right: information gained by the unticked tests."
            ),
            ncol=min(len(handles), 4),
        )


class IdentifiabilityStrengthPanel(PlotResultsPage):
    def __init__(self, parent=None) -> None:
        super().__init__(
            "Identifiability strength",
            ["Deletion influence and information gain"],
            parent,
        )
        self._mode.setVisible(False)
        self._result: Optional[UncertaintyResult] = None
        self._document: Optional[SessionDocument] = None
        self._metrics: Optional[IdentifiabilityStrength] = None
        self._shifts: list[ExperimentShift] = []
        self._baseline: dict[int, bool] = {}
        self._shift_scale: str = SHIFT_UNITS

    def update_from(
        self,
        document: SessionDocument,
        uncertainty: Optional[UncertaintyResult],
    ) -> None:
        self._document = document
        self._result = uncertainty
        self._recompute()
        self._fill_table()
        self.refresh_plot()

    def apply_baseline(
        self,
        baseline: Mapping[int, bool] | None,
        *,
        shift_scale: str = SHIFT_UNITS,
    ) -> None:
        if baseline:
            self._baseline = {int(key): bool(value) for key, value in baseline.items()}
        self._shift_scale = (
            SHIFT_RELATIVE if str(shift_scale) == SHIFT_RELATIVE else SHIFT_UNITS
        )
        self._recompute()
        self._fill_table()
        self.refresh_plot()

    def _resolved_baseline(self) -> dict[int, bool]:
        result = self._result
        if result is None:
            return {}
        resolved = default_baseline_flags(result.cluster_names, result.cluster_ids)
        resolved.update({int(key): bool(value) for key, value in self._baseline.items()})
        return resolved

    def _included_ids(self) -> list[int]:
        flags = self._resolved_baseline()
        result = self._result
        if result is None:
            return [eid for eid, included in flags.items() if included]
        return [
            int(eid)
            for eid in result.cluster_ids
            if flags.get(int(eid), True)
        ]

    def _recompute(self) -> None:
        result = self._result
        if result is None:
            self._metrics = None
            self._shifts = []
            return
        included_ids = self._included_ids()
        A_baseline = None
        if included_ids:
            A_baseline = assemble_sensitivity(result.information, included_ids)
            all_ids = {int(row.experiment_id) for row in result.information.contributions}
            if set(int(eid) for eid in included_ids) >= all_ids:
                A_baseline = result.A
        self._metrics = compute_strength(
            result.A,
            result.theta,
            result.labels,
            A_baseline=A_baseline,
        )
        self._shifts = subset_deletion_shifts(result, result.cluster_ids)

    def _fill_table(self) -> None:
        result = self._result
        metrics = self._metrics
        if result is None or metrics is None:
            self.set_banner(
                "Information gain and deletion influence need Identify. "
                "Run Identify first."
            )
            self._table.setRowCount(0)
            return
        flags = self._resolved_baseline()
        added_names = [
            name
            for name, eid in zip(result.cluster_names, result.cluster_ids)
            if not flags.get(int(eid), True)
        ]
        extra = (
            " A was rank-deficient; a small eigenvalue floor was applied so "
            "unidentified directions have large working variance."
            if metrics.used_eigenvalue_floor
            else ""
        )
        relative = self._shift_scale == SHIFT_RELATIVE
        if not added_names:
            extra += (
                " Every test is ticked, so G_k = 0. Untick tests to measure "
                "the information they add."
            )
        else:
            extra += f" Added for G_k: {', '.join(added_names)}."
        shift_text = (
            r"(θ̂₍₋ᵢ₎ − θ̂)/|θ̂|"
            if relative
            else r"θ̂₍₋ᵢ₎ − θ̂ in the original parameter units"
        )
        self.set_banner(
            "Left: eq. (63) for every test, "
            r"θ̂₍₋ᵢ₎ − θ̂ = −(A − Aᵢ)⁻¹ ψᵢ(θ̂), "
            f"shown as {shift_text}. "
            "Right: information gained by the unticked tests."
            + extra
        )
        headers = ["Parameter", "Gₖ"] + [row.name for row in self._shifts]
        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        self._table.setRowCount(len(result.labels))
        displayed = scaled_deletion_shifts(
            self._shifts, result.theta, relative=relative
        )
        for i, label in enumerate(result.labels):
            self._table.setItem(i, 0, text_item(label, bold=True))
            self._table.setItem(i, 1, number_item(metrics.information_gain[i]))
            for column, values in enumerate(displayed, start=2):
                value = values[i] if i < values.size else float("nan")
                self._table.setItem(i, column, number_item(value))
        style_column_widths(self._table)

    def refresh_plot(self) -> None:
        self._figure.clear()
        if self._result is None or self._metrics is None:
            empty_message(self._figure, "Run Identify to fill this analysis.")
        else:
            plot_identifiability_strength(
                self._figure,
                self._result.labels,
                self._metrics,
                self._shifts,
                theta=self._result.theta,
                relative=self._shift_scale == SHIFT_RELATIVE,
            )
        self._canvas.draw_idle()
