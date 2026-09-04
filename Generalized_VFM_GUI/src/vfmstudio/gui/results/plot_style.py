"""Shared matplotlib chrome for post-processing plots."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

SERIES_COLOUR = (
    "#0072B2",
    "#E69F00",
    "#F0E442",
    "#009E73",
    "#D55E00",
    "#000000",
    "#CC79A7",
    "#56B4E9",
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#4a3aa7",
)
SERIES_MARKER = ("o", "s", "^", "D", "v", "P")

INK_PRIMARY = "#141413"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8f8e88"
GRID_COLOUR = "#dcdbd5"
UI_FONT_STACK = "Arial, Helvetica, sans-serif"

MPL_RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "axes.linewidth": 0.6,
    "text.usetex": False,
    # Type 42 = embedded TrueType. Type 3 (the default) rasterizes poorly
    # when a journal or Illustrator opens the PDF.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_matplotlib_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(MPL_RCPARAMS)


def heading_stylesheet(*, muted: bool = False) -> str:
    """QLabel chrome matching the Abaqus-style Arial UI, not journal serif."""
    color = INK_MUTED if muted else INK_PRIMARY
    size = "9pt" if muted else "12pt"
    return (
        f"color: {color}; font-size: {size}; font-family: {UI_FONT_STACK}; "
        "background: none;"
    )


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


def place_legend_below(
    figure,
    handles,
    labels,
    *,
    caption: str | None = None,
    ncol: int | None = None,
) -> None:
    """Put a shared legend (and optional caption) under every panel.

    Leaves enough bottom margin for wrapped legend rows so the handles
    do not collide with x-labels.
    """
    n = max(len(handles), 1)
    columns = ncol if ncol is not None else min(n, 4)
    n_rows = int(np.ceil(n / max(columns, 1)))
    caption_space = 0.11 if caption else 0.0
    legend_space = 0.038 * n_rows + 0.036
    bottom = min(0.46, 0.12 + legend_space + caption_space)
    figure.subplots_adjust(
        left=max(float(figure.subplotpars.left), 0.10),
        right=min(float(figure.subplotpars.right), 0.98),
        top=min(float(figure.subplotpars.top), 0.93),
        bottom=bottom,
    )
    if handles:
        figure.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.018 + caption_space),
            ncol=columns,
            fontsize=8,
            frameon=False,
            handlelength=1.6,
            columnspacing=1.15,
            handletextpad=0.45,
            borderaxespad=0.0,
        )
    if caption:
        figure.text(
            0.5,
            0.012,
            caption,
            ha="center",
            va="bottom",
            fontsize=7.2,
            color=INK_MUTED,
        )


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
