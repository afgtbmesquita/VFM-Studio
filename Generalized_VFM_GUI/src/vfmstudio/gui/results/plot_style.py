"""Shared matplotlib chrome for post-processing plots."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

SERIES_COLOUR = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#C59B2D", "#CC6677")
SERIES_MARKER = ("o", "s", "^", "D", "v", "P")

INK_PRIMARY = "#141413"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8f8e88"
GRID_COLOUR = "#dcdbd5"

MPL_RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "axes.linewidth": 0.6,
    "text.usetex": False,
}


def apply_matplotlib_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(MPL_RCPARAMS)


def pad_limits(axis, values, fraction: float = 0.14, *, which: str = "y") -> None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return
    low, high = float(np.min(finite)), float(np.max(finite))
    span = high - low
    if span <= 0:
        span = max(abs(high), 1.0) * 0.1
    limits = (low - fraction * span, high + fraction * span)
    if which == "x":
        axis.set_xlim(*limits)
    else:
        axis.set_ylim(*limits)


def style_axis(axis, *, grid: str = "y") -> None:
    axis.set_facecolor("#FFFFFF")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(INK_SECONDARY)
        axis.spines[side].set_linewidth(0.6)
    axis.tick_params(
        colors=INK_SECONDARY,
        width=0.6,
        size=3.0,
        labelsize=8,
        direction="out",
    )
    axis.xaxis.label.set_color(INK_PRIMARY)
    axis.yaxis.label.set_color(INK_PRIMARY)
    axis.title.set_color(INK_PRIMARY)
    if grid:
        axis.grid(
            True,
            which="major",
            axis=grid,
            color=GRID_COLOUR,
            linewidth=0.5,
            zorder=0,
        )
    axis.set_axisbelow(True)


def style_heatmap_axis(axis) -> None:
    axis.set_facecolor("#FFFFFF")
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(length=0, colors=INK_SECONDARY, labelsize=8)
    axis.xaxis.label.set_color(INK_PRIMARY)
    axis.yaxis.label.set_color(INK_PRIMARY)
    axis.title.set_color(INK_PRIMARY)


def experiment_colour(index: int) -> str:
    return SERIES_COLOUR[index % len(SERIES_COLOUR)]


def empty_message(figure, text: str) -> None:
    axis = figure.add_subplot(111)
    axis.set_facecolor("#FFFFFF")
    axis.axis("off")
    axis.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        color=INK_MUTED,
        fontsize=10,
        transform=axis.transAxes,
    )


def annotate_heatmap(axis, matrix, *, fmt: str = "{:.2f}", vmin=None, vmax=None) -> None:
    span_low = -1.0 if vmin is None else float(vmin)
    span_high = 1.0 if vmax is None else float(vmax)
    span = max(abs(span_low), abs(span_high), np.finfo(float).tiny)
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            cell = matrix[i, j]
            if not np.isfinite(cell):
                axis.text(
                    j, i, "—", ha="center", va="center", fontsize=7, color=INK_MUTED
                )
                continue
            axis.text(
                j,
                i,
                fmt.format(cell),
                ha="center",
                va="center",
                fontsize=7,
                color="#f7f6f2" if abs(cell) > 0.55 * span else INK_PRIMARY,
            )


def parameter_tick_labels(names: Sequence[str]) -> list[str]:
    from vfmstudio.core.document import parameter_math_label

    return [parameter_math_label(name) for name in names]


def draw_matrix(
    axis,
    matrix: np.ndarray,
    names: Sequence[str],
    *,
    title: str,
    cmap: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    labeled: bool = True,
    fmt: str = "{:.2f}",
):
    style_heatmap_axis(axis)
    matrix = np.asarray(matrix, dtype=float)
    n = int(matrix.shape[0])
    kwargs = {"origin": "upper", "cmap": cmap, "interpolation": "nearest"}
    if vmin is not None or vmax is not None:
        kwargs["vmin"] = vmin
        kwargs["vmax"] = vmax
    image = axis.imshow(matrix, **kwargs)
    ticks = np.arange(n)
    labels = parameter_tick_labels(names)
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    axis.set_xticklabels(labels)
    axis.set_yticklabels(labels)
    axis.set_title(title, color=INK_PRIMARY, pad=6)
    if labeled:
        annotate_heatmap(axis, matrix, fmt=fmt, vmin=vmin, vmax=vmax)
    return image
