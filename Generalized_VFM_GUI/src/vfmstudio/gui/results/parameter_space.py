"""Two-panel stiffness-parameter design space for every test combination."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from vfmstudio.core.document import (
    SessionDocument,
    parameter_math_label,
    parameter_unit_math,
)
from vfmstudio.core.parameter_space import (
    ParameterSpaceResult,
    assign_highlight_groups,
    compute_parameter_space,
    mask_for_experiment_ids,
    resolve_plot_axes,
)
from vfmstudio.gui.results.plot_style import (
    INK_MUTED,
    INK_PRIMARY,
    apply_matplotlib_style,
    empty_message,
    pad_limits,
    place_legend_below,
    style_axis,
)

UNGROUPED_FACE = "#d0cfc9"

PS_CMAP = LinearSegmentedColormap.from_list(
    "parameter_space",
    ["#d6e6f2", "#56B4E9", "#0072B2"],
)
PIXEL_HIT = 14.0


def _axis_label(name: str) -> str:
    math = parameter_math_label(name)
    unit = parameter_unit_math(name)
    return f"{math} ({unit})" if unit else math


def _subset_text(
    space: ParameterSpaceResult,
    mask: int,
    group_name: str = "",
) -> str:
    names = space.subset_names(int(mask))
    if not names:
        return "Empty subset"
    if len(names) == 1:
        heading = names[0]
    elif len(names) == space.n_clusters:
        heading = "All tests"
    else:
        heading = f"{len(names)} tests"
    if group_name:
        heading = f"{heading} · {group_name}"
    body = ", ".join(names)
    return f"{heading}\n{body}"


class ParameterSpacePanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        apply_matplotlib_style()
        self._title = QLabel("Parameter space")
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

        self._space: Optional[ParameterSpaceResult] = None
        self._axis_names = ("", "", "", "")
        self._groups: list[dict] = []
        self._assigned: Optional[np.ndarray] = None
        self._selected_ids: list[int] = []
        self._xy: list[Optional[np.ndarray]] = [None, None]
        self._axes: list = []
        self._annots: list = []
        self._pinned: Optional[int] = None
        self._canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._canvas.mpl_connect("button_press_event", self._on_click)
        self._canvas.mpl_connect("axes_leave_event", self._on_leave)

    def update_from(
        self,
        document: Optional[SessionDocument],
        result: Any,
    ) -> None:
        names = None
        labels = None
        if document is not None:
            labels = document.material.labels()
            names = {
                int(experiment.experiment_id): (
                    experiment.name or f"Experiment {experiment.experiment_id}"
                )
                for experiment in document.experiments
            }
        self._space = compute_parameter_space(
            result, labels=labels, experiment_names=names
        )
        if document is not None:
            self._selected_ids = [
                int(experiment.experiment_id)
                for experiment in document.experiments_in_identification()
            ]
        self._pinned = None
        self.refresh()

    def set_selected_ids(self, experiment_ids) -> None:
        self._selected_ids = [int(experiment_id) for experiment_id in experiment_ids]

    def set_groups(self, groups) -> None:
        copied: list[dict] = []
        for group in groups or []:
            copied.append(
                {
                    "name": str(group.get("name") or ""),
                    "color": str(group.get("color") or "#D55E00"),
                    "experiment_ids": [
                        int(experiment_id)
                        for experiment_id in (group.get("experiment_ids") or [])
                    ],
                }
            )
        self._groups = copied

    def _active_groups(self) -> list[dict]:
        return [group for group in self._groups if group.get("experiment_ids")]

    def apply_axes(
        self,
        left_x: str = "",
        left_y: str = "",
        right_x: str = "",
        right_y: str = "",
    ) -> None:
        self._axis_names = (left_x, left_y, right_x, right_y)
        self.refresh()

    def refresh(self) -> None:
        self._figure.clear()
        self._xy = [None, None]
        self._axes = []
        self._annots = []
        self._assigned = None
        space = self._space
        if space is None or space.n_subsets == 0:
            self._subtitle.setText("")
            self._subtitle.setVisible(False)
            empty_message(
                self._figure,
                "Run Identify first. Combinations use the per-test GLS "
                "blocks from that solve.",
            )
            self._canvas.draw_idle()
            return
        n = space.n_subsets
        m = space.n_clusters
        extra = ""
        if self._active_groups():
            extra = " Highlight groups are set in Properties."
        self._subtitle.setText(
            f"{n} non-empty combination{'s' if n != 1 else ''} of {m} "
            "test"
            f"{'s' if m != 1 else ''}, at the virtual fields of the full "
            "identification. The star is all tests together."
            + extra
        )
        self._subtitle.setVisible(True)
        self._draw(space)
        self._canvas.draw_idle()

    def _draw(self, space: ParameterSpaceResult) -> None:
        from matplotlib.lines import Line2D

        ix, iy, jx, jy = resolve_plot_axes(space.labels, *self._axis_names)
        pairs = ((ix, iy), (jx, jy))
        groups = self._active_groups()
        assigned = assign_highlight_groups(space, self._groups) if groups else None
        self._assigned = assigned
        axes = self._figure.subplots(1, 2)
        if groups:
            self._figure.subplots_adjust(
                left=0.08, right=0.97, top=0.90, bottom=0.22, wspace=0.34
            )
        else:
            self._figure.subplots_adjust(
                left=0.08, right=0.90, top=0.90, bottom=0.16, wspace=0.34
            )
        n_max = int(np.max(space.n_tests)) if space.n_tests.size else 1
        norm = Normalize(vmin=1.0, vmax=float(max(n_max, 1)))
        mappable = None
        selected = mask_for_experiment_ids(space, self._selected_ids)
        show_selected = (
            selected is not None and int(selected) != int(space.full_mask)
        )
        for panel, axis, (x_index, y_index) in zip((0, 1), axes, pairs):
            style_axis(axis, grid="both")
            x_name = space.labels[x_index]
            y_name = space.labels[y_index]
            xy = np.column_stack(
                (space.theta[:, x_index], space.theta[:, y_index])
            )
            self._xy[panel] = xy
            well = np.asarray(space.well_posed, dtype=bool)
            ill = ~well
            if groups:
                ungrouped = assigned < 0
                if np.any(ungrouped & well):
                    pick = ungrouped & well
                    axis.scatter(
                        xy[pick, 0],
                        xy[pick, 1],
                        c=UNGROUPED_FACE,
                        s=22,
                        marker="o",
                        edgecolors="white",
                        linewidths=0.4,
                        alpha=0.7,
                        zorder=2,
                    )
                if np.any(ungrouped & ill):
                    pick = ungrouped & ill
                    axis.scatter(
                        xy[pick, 0],
                        xy[pick, 1],
                        c=UNGROUPED_FACE,
                        s=22,
                        marker="o",
                        edgecolors=INK_MUTED,
                        linewidths=0.4,
                        alpha=0.28,
                        zorder=1,
                    )
                for group_index, group in enumerate(self._groups):
                    pick = assigned == group_index
                    if not np.any(pick):
                        continue
                    color = group["color"]
                    if np.any(pick & well):
                        shown = pick & well
                        axis.scatter(
                            xy[shown, 0],
                            xy[shown, 1],
                            c=color,
                            s=34,
                            marker="o",
                            edgecolors="white",
                            linewidths=0.45,
                            alpha=0.95,
                            zorder=3,
                        )
                    if np.any(pick & ill):
                        shown = pick & ill
                        axis.scatter(
                            xy[shown, 0],
                            xy[shown, 1],
                            c=color,
                            s=34,
                            marker="o",
                            edgecolors=INK_MUTED,
                            linewidths=0.45,
                            alpha=0.35,
                            zorder=2,
                        )
            else:
                if np.any(well):
                    scatter = axis.scatter(
                        xy[well, 0],
                        xy[well, 1],
                        c=space.n_tests[well],
                        cmap=PS_CMAP,
                        norm=norm,
                        s=28,
                        marker="o",
                        edgecolors="white",
                        linewidths=0.45,
                        alpha=0.92,
                        zorder=3,
                    )
                    mappable = scatter
                if np.any(ill):
                    ill_scatter = axis.scatter(
                        xy[ill, 0],
                        xy[ill, 1],
                        c=space.n_tests[ill],
                        cmap=PS_CMAP,
                        norm=norm,
                        s=28,
                        marker="o",
                        edgecolors=INK_MUTED,
                        linewidths=0.45,
                        alpha=0.32,
                        zorder=2,
                    )
                    if mappable is None:
                        mappable = ill_scatter
            star_x = float(space.theta_full[x_index])
            star_y = float(space.theta_full[y_index])
            axis.scatter(
                [star_x],
                [star_y],
                marker="*",
                s=190,
                facecolor="#111111",
                edgecolors="white",
                linewidths=0.7,
                zorder=5,
                label="All tests",
            )
            if show_selected:
                row = space.index_of_mask(selected)
                axis.scatter(
                    [float(xy[row, 0])],
                    [float(xy[row, 1])],
                    s=96,
                    facecolors="none",
                    edgecolors="#111111",
                    linewidths=1.15,
                    zorder=4,
                    label="Selected combination",
                )
            pad_limits(axis, xy[:, 0], which="x")
            pad_limits(axis, xy[:, 1], which="y")
            axis.set_xlabel(_axis_label(x_name), labelpad=8)
            axis.set_ylabel(_axis_label(y_name), labelpad=8)
            annot = axis.annotate(
                "",
                xy=(0.0, 0.0),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=8,
                color=INK_PRIMARY,
                bbox={
                    "boxstyle": "round,pad=0.28",
                    "facecolor": "#FFFFFF",
                    "edgecolor": INK_MUTED,
                    "linewidth": 0.6,
                },
                zorder=6,
            )
            annot.set_visible(False)
            self._annots.append(annot)
            self._axes.append(axis)
        if groups:
            handles = []
            labels = []
            for group in self._groups:
                if not group.get("experiment_ids"):
                    continue
                handles.append(
                    Line2D(
                        [],
                        [],
                        linestyle="none",
                        marker="o",
                        markersize=7,
                        markerfacecolor=group["color"],
                        markeredgecolor="white",
                        markeredgewidth=0.5,
                    )
                )
                labels.append(group["name"] or "Group")
            handles.append(
                Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker="*",
                    markersize=10,
                    markerfacecolor="#111111",
                    markeredgecolor="white",
                    markeredgewidth=0.5,
                )
            )
            labels.append("All tests")
            if show_selected:
                handles.append(
                    Line2D(
                        [],
                        [],
                        linestyle="none",
                        marker="o",
                        markersize=9,
                        markerfacecolor="none",
                        markeredgecolor="#111111",
                        markeredgewidth=1.15,
                    )
                )
                labels.append("Selected combination")
            place_legend_below(self._figure, handles, labels, ncol=min(4, len(handles)))
            return
        if mappable is not None:
            colorbar = self._figure.colorbar(
                mappable, ax=axes, fraction=0.03, pad=0.03
            )
            colorbar.set_label("Number of tests", color=INK_PRIMARY)
            colorbar.ax.tick_params(labelsize=8, colors=INK_MUTED, width=0.6, length=3)
            ticks = np.arange(1, n_max + 1)
            if ticks.size <= 12:
                colorbar.set_ticks(ticks)
            colorbar.outline.set_linewidth(0.6)
            colorbar.outline.set_edgecolor(INK_MUTED)
        axes[1].legend(
            loc="upper right",
            fontsize=8,
            frameon=False,
            handletextpad=0.4,
            borderaxespad=0.4,
        )

    def _nearest(self, event) -> Optional[int]:
        if event is None or event.inaxes is None:
            return None
        if event.xdata is None or event.ydata is None:
            return None
        try:
            panel = self._axes.index(event.inaxes)
        except ValueError:
            return None
        xy = self._xy[panel]
        if xy is None or xy.size == 0:
            return None
        display = event.inaxes.transData.transform(xy)
        point = event.inaxes.transData.transform(
            np.array([[float(event.xdata), float(event.ydata)]])
        )[0]
        distance = np.hypot(display[:, 0] - point[0], display[:, 1] - point[1])
        index = int(np.argmin(distance))
        if float(distance[index]) > PIXEL_HIT:
            return None
        return index

    def _show(self, index: int) -> None:
        space = self._space
        if space is None or not self._annots:
            return
        mask = int(space.masks[index])
        group_name = ""
        assigned = self._assigned
        if assigned is not None and 0 <= index < assigned.size:
            group_index = int(assigned[index])
            if 0 <= group_index < len(self._groups):
                group_name = str(self._groups[group_index].get("name") or "")
        text = _subset_text(space, mask, group_name)
        for panel, annot in enumerate(self._annots):
            xy = self._xy[panel]
            if xy is None:
                continue
            annot.xy = (float(xy[index, 0]), float(xy[index, 1]))
            annot.set_text(text)
            annot.set_visible(True)

    def _hide(self) -> None:
        for annot in self._annots:
            annot.set_visible(False)

    def _on_motion(self, event) -> None:
        if self._pinned is not None:
            return
        index = self._nearest(event)
        if index is None:
            self._hide()
        else:
            self._show(index)
        self._canvas.draw_idle()

    def _on_click(self, event) -> None:
        if getattr(event, "button", 1) != 1:
            return
        index = self._nearest(event)
        if index is None:
            self._pinned = None
            self._hide()
        else:
            self._pinned = index
            self._show(index)
        self._canvas.draw_idle()

    def _on_leave(self, _event) -> None:
        if self._pinned is not None:
            return
        self._hide()
        self._canvas.draw_idle()
