"""Central-pane overlay: raw moment (N·mm) vs curvature with a specimen map."""

from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStyleFactory,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from vfmstudio.core.fields import field_axis_math, field_display_label
from vfmstudio.core.step_series import (
    SpatialSelection,
    StepSeries,
    collect_step_series,
    selection_mask,
)
from vfmstudio.gui.session import AppSession
from vfmstudio.gui.viewport.overlays import add_bc_symbols_2d

INK_PRIMARY = "#141413"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8f8e88"
GRID_COLOUR = "#dcdbd5"
INCLUDED = "#2a78d6"
REFERENCE = "#eb6834"
EXCLUDED = "#8f8e88"
SELECTED = "#C0392B"
APP_GRAY = "#E6E6E6"


def _style_axis(axis) -> None:
    axis.set_facecolor("#FFFFFF")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(INK_SECONDARY)
    axis.spines["bottom"].set_color(INK_SECONDARY)
    axis.spines["left"].set_linewidth(0.6)
    axis.spines["bottom"].set_linewidth(0.6)
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
    axis.set_axisbelow(True)


class LoadPlotOverlay(QWidget):
    closed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._session: Optional[AppSession] = None
        self._series: Optional[StepSeries] = None
        self._series_key = None
        self._cache: dict = {}
        self._selection = SpatialSelection()
        self._drag_origin = None
        self._rubber: Optional[Rectangle] = None
        self._spec_bg = None
        self._spec_xlim = None
        self._spec_ylim = None

        import matplotlib as mpl

        mpl.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                "mathtext.fontset": "stix",
                "font.size": 9,
                "axes.labelsize": 10,
                "axes.titlesize": 10,
                "legend.fontsize": 8,
                "legend.frameon": False,
                "axes.linewidth": 0.6,
                "text.usetex": False,
            }
        )

        self._force = QComboBox()
        self._force.currentIndexChanged.connect(self._redraw)
        self._field = QComboBox()
        self._field.currentIndexChanged.connect(self._redraw)
        for combo in (self._force, self._field):
            self._paint_combo_like_app(combo)

        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._btn_all = QPushButton("Whole field")
        self._btn_point = QPushButton("Point")
        self._btn_roi = QPushButton("ROI")
        for button, kind in (
            (self._btn_all, "all"),
            (self._btn_point, "point"),
            (self._btn_roi, "roi"),
        ):
            button.setCheckable(True)
            button.setProperty("kind", kind)
            self._mode_group.addButton(button)
        self._btn_all.setChecked(True)
        self._mode_group.buttonClicked.connect(self._on_mode)

        close = QToolButton()
        close.setText("✕")
        close.setToolTip("Back to 3D visualization")
        close.setAutoRaise(True)
        close.setCursor(Qt.CursorShape.PointingHandCursor)
        close.setFixedSize(28, 28)
        close.setStyleSheet(
            "QToolButton { font-size: 14pt; font-weight: 600; border: none; }"
            "QToolButton:hover { background: #E0E0E0; }"
        )
        close.clicked.connect(self.closed.emit)

        bar = QHBoxLayout()
        bar.setContentsMargins(8, 8, 8, 4)
        bar.addWidget(QLabel("Moment"))
        bar.addWidget(self._force)
        bar.addSpacing(12)
        bar.addWidget(QLabel("Curvature"))
        bar.addWidget(self._field)
        bar.addSpacing(16)
        bar.addWidget(self._btn_all)
        bar.addWidget(self._btn_point)
        bar.addWidget(self._btn_roi)
        bar.addStretch(1)
        bar.addWidget(close, 0, Qt.AlignmentFlag.AlignTop)

        self._status = QLabel("Whole-field mean")
        self._status.setStyleSheet(f"color: {INK_MUTED}; padding: 0 8px 4px 8px;")

        self._plot_fig = Figure(figsize=(5.2, 3.6), layout="constrained")
        self._plot_canvas = FigureCanvas(self._plot_fig)
        self._plot_ax = self._plot_fig.add_subplot(111)
        self._plot_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self._spec_fig = Figure(figsize=(3.6, 3.6), layout="constrained")
        self._spec_canvas = FigureCanvas(self._spec_fig)
        self._spec_ax = self._spec_fig.add_subplot(111)
        self._spec_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._spec_canvas.mpl_connect("button_press_event", self._on_spec_press)
        self._spec_canvas.mpl_connect("button_release_event", self._on_spec_release)
        self._spec_canvas.mpl_connect("motion_notify_event", self._on_spec_move)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._plot_canvas)
        splitter.addWidget(self._spec_canvas)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([640, 420])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(bar)
        layout.addWidget(self._status)
        layout.addWidget(splitter, 1)
        self.setObjectName("loadPlotOverlay")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            """
            #loadPlotOverlay {
                background-color: #E6E6E6;
            }
            #loadPlotOverlay QLabel {
                background: transparent;
                color: #222222;
            }
            #loadPlotOverlay QComboBox {
                background-color: #E6E6E6;
                color: #222222;
                border: 1px solid #A8A8A8;
                padding: 3px 8px;
                min-width: 96px;
            }
            #loadPlotOverlay QComboBox:editable,
            #loadPlotOverlay QComboBox:!editable,
            #loadPlotOverlay QComboBox:on,
            #loadPlotOverlay QComboBox QLineEdit {
                background-color: #E6E6E6;
                color: #222222;
            }
            #loadPlotOverlay QComboBox::drop-down {
                background-color: #E6E6E6;
                border: none;
                border-left: 1px solid #A8A8A8;
                width: 16px;
            }
            #loadPlotOverlay QComboBox QAbstractItemView {
                background-color: #E6E6E6;
                color: #222222;
                selection-background-color: #4A90D9;
                selection-color: #FFFFFF;
                border: 1px solid #A8A8A8;
            }
            """
        )

    @staticmethod
    def _paint_combo_like_app(combo: QComboBox) -> None:
        """Windows native combos ignore QSS fill; Fusion + palette match #E6E6E6."""
        fusion = QStyleFactory.create("Fusion")
        if fusion is not None:
            combo.setStyle(fusion)
        combo.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        combo.setAutoFillBackground(True)
        palette = combo.palette()
        fill = QColor(APP_GRAY)
        for role in (
            QPalette.ColorRole.Base,
            QPalette.ColorRole.Button,
            QPalette.ColorRole.Window,
            QPalette.ColorRole.AlternateBase,
        ):
            palette.setColor(role, fill)
        combo.setPalette(palette)

    def set_session(self, session: AppSession) -> None:
        self._session = session
        experiment = session.selected_experiment()
        if experiment is None:
            self._series = None
            self._series_key = None
            self._force.clear()
            self._field.clear()
            self._redraw()
            return
        key = (
            experiment.experiment_id,
            session.document.material.analysis_type,
            tuple(str(step.data_file) for step in experiment.steps),
            str(session.document.source_path),
            bool(experiment.fit_data_to_system_axes),
        )
        if key != self._series_key:
            self._series = collect_step_series(
                experiment,
                session.document.material.analysis_type,
                session.document.source_path,
                cache=self._cache,
            )
            self._series_key = key
            self._selection = SpatialSelection()
            self._btn_all.setChecked(True)
            self._fill_combos()
        self._redraw()

    def _fill_combos(self) -> None:
        series = self._series
        force_data = self._force.currentData()
        field_data = self._field.currentData()
        self._force.blockSignals(True)
        self._field.blockSignals(True)
        self._force.clear()
        self._field.clear()
        if series is not None:
            for channel in series.force_channels:
                self._force.addItem(channel.combo_label, channel.index)
            for name in series.field_names:
                self._field.addItem(field_display_label(name), name)
            if isinstance(force_data, int):
                match = self._force.findData(force_data)
                if match >= 0:
                    self._force.setCurrentIndex(match)
            if field_data is not None:
                match = self._field.findData(field_data)
                if match >= 0:
                    self._field.setCurrentIndex(match)
                else:
                    default = self._field.findData("kappa_yy")
                    if default >= 0:
                        self._field.setCurrentIndex(default)
        self._force.blockSignals(False)
        self._field.blockSignals(False)

    def _on_mode(self, button) -> None:
        kind = str(button.property("kind") or "all")
        if kind == "all":
            self._selection = SpatialSelection()
            self._redraw()
        elif kind == "point":
            self._status.setText("Click a DIC point on the specimen.")
        else:
            self._status.setText("Drag a rectangle on the specimen to set the ROI.")

    def _geometry(self):
        series = self._series
        if series is None:
            return None
        for point in series.points:
            if point.x is not None and point.y is not None and point.error is None:
                return point
        return None

    def _active_cloud(self):
        series = self._series
        session = self._session
        if series is None:
            return None
        index = 0
        if session is not None and session.selection.step_index is not None:
            index = int(session.selection.step_index)
        for point in series.points:
            if point.index == index and point.x is not None and point.error is None:
                return point
        return self._geometry()

    def _redraw(self) -> None:
        self._draw_plot()
        self._draw_specimen()

    def _draw_plot(self) -> None:
        axis = self._plot_ax
        axis.clear()
        series = self._series
        session = self._session
        if series is None or session is None:
            axis.set_title("No experiment")
            self._plot_fig.patch.set_facecolor("#FFFFFF")
            _style_axis(axis)
            self._plot_canvas.draw_idle()
            return
        experiment = session.selected_experiment()
        if experiment is None:
            axis.set_title("No experiment")
            self._plot_fig.patch.set_facecolor("#FFFFFF")
            _style_axis(axis)
            self._plot_canvas.draw_idle()
            return
        force_index = self._force.currentData()
        field_name = self._field.currentData()
        if force_index is None or field_name is None:
            axis.set_title("No moment / curvature components")
            self._plot_fig.patch.set_facecolor("#FFFFFF")
            _style_axis(axis)
            self._plot_canvas.draw_idle()
            return
        included = set(experiment.included_step_indices())
        reference = (
            int(experiment.reference_step) - 1
            if experiment.reference_step is not None
            else None
        )
        selected = session.selection.step_index
        if selected is None and session.selection.kind == "step":
            selected = 0

        included_x, included_y = [], []
        excluded_x, excluded_y = [], []
        reference_x, reference_y = [], []
        selected_xy = None
        all_x, all_y = [], []
        for point in series.points:
            x_value = point.field_value(str(field_name), self._selection)
            y_value = point.force_value(int(force_index))
            if not np.isfinite(x_value) or not np.isfinite(y_value):
                continue
            all_x.append(x_value)
            all_y.append(y_value)
            if reference is not None and point.index == reference:
                reference_x.append(x_value)
                reference_y.append(y_value)
            elif point.index in included:
                included_x.append(x_value)
                included_y.append(y_value)
            else:
                excluded_x.append(x_value)
                excluded_y.append(y_value)
            if selected is not None and point.index == selected:
                selected_xy = (x_value, y_value)

        if included_x:
            axis.plot(
                included_x,
                included_y,
                color=INCLUDED,
                marker="o",
                markersize=5.2,
                markeredgecolor="white",
                markeredgewidth=0.6,
                linewidth=1.4,
                zorder=3,
                clip_on=False,
                label="included",
            )
        if excluded_x:
            axis.plot(
                excluded_x,
                excluded_y,
                linestyle="None",
                marker="o",
                markersize=4.6,
                markerfacecolor="none",
                markeredgecolor=EXCLUDED,
                markeredgewidth=1.0,
                zorder=2,
                clip_on=False,
                label="excluded",
            )
        if reference_x:
            axis.plot(
                reference_x,
                reference_y,
                linestyle="None",
                marker="D",
                markersize=6.2,
                color=REFERENCE,
                markeredgecolor="white",
                markeredgewidth=0.5,
                zorder=4,
                clip_on=False,
                label="reference",
            )
        if selected_xy is not None:
            axis.plot(
                [selected_xy[0]],
                [selected_xy[1]],
                linestyle="None",
                marker="o",
                markersize=10,
                markerfacecolor="none",
                markeredgecolor=SELECTED,
                markeredgewidth=1.4,
                zorder=5,
                clip_on=False,
                label="selected",
            )

        channel = None
        if series.force_channels:
            for item in series.force_channels:
                if item.index == int(force_index):
                    channel = item
                    break
            if channel is None:
                channel = series.force_channels[self._force.currentIndex()]
        y_label = channel.axis_math if channel is not None else r"$M\ (\mathrm{N\cdot mm})$"
        axis.set_xlabel(field_axis_math(str(field_name)))
        axis.set_ylabel(y_label)
        axis.set_title(self._plot_title(experiment, series), pad=6)
        axis.grid(True, axis="y", color=GRID_COLOUR, linewidth=0.5)
        axis.xaxis.set_major_locator(MaxNLocator(6))
        axis.yaxis.set_major_locator(MaxNLocator(6))
        axis.xaxis.set_minor_locator(AutoMinorLocator(2))
        axis.yaxis.set_minor_locator(AutoMinorLocator(2))
        if all_x and all_y:
            self._pad_limits(axis, all_x, "x")
            self._pad_limits(axis, all_y, "y")
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(loc="best", handlelength=1.6)
        _style_axis(axis)
        self._plot_fig.patch.set_facecolor("#FFFFFF")
        self._plot_canvas.draw_idle()

    def _plot_title(self, experiment, series: StepSeries) -> str:
        name = experiment.name or "Experiment"
        if self._selection.kind == "point":
            extra = r"point"
        elif self._selection.kind == "roi":
            extra = r"ROI mean"
        else:
            extra = r"whole-field mean"
        n_ok = sum(1 for point in series.points if point.error is None)
        return f"{name}: {extra}  ({n_ok} steps)"

    def _pad_limits(self, axis, values, which: str) -> None:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return
        low, high = float(np.min(finite)), float(np.max(finite))
        span = high - low
        if span <= 0:
            span = max(abs(high), 1.0) * 0.1
        lower, upper = low - 0.14 * span, high + 0.14 * span
        if which == "y":
            axis.set_ylim(lower, upper)
        else:
            axis.set_xlim(lower, upper)

    def _draw_specimen(self) -> None:
        self._spec_fig.clear()
        axis = self._spec_fig.add_subplot(111)
        self._spec_ax = axis
        geometry = self._active_cloud()
        field_name = self._field.currentData()
        if geometry is None or geometry.x is None or geometry.y is None:
            axis.set_title("No specimen")
            self._spec_fig.patch.set_facecolor("#FFFFFF")
            _style_axis(axis)
            self._spec_canvas.draw_idle()
            return
        x = np.asarray(geometry.x, dtype=float)
        y = np.asarray(geometry.y, dtype=float)
        values = None
        if field_name and field_name in geometry.fields:
            values = np.asarray(geometry.fields[str(field_name)], dtype=float)
        if values is None:
            axis.scatter(x, y, s=8, c="#B8B8B8", linewidths=0, zorder=2)
        else:
            vmax = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1.0
            vmax = vmax if vmax > 0 else 1.0
            scatter = axis.scatter(
                x,
                y,
                s=10,
                c=values,
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                linewidths=0,
                zorder=2,
            )
            colorbar = self._spec_fig.colorbar(
                scatter, ax=axis, fraction=0.046, pad=0.04
            )
            colorbar.set_label(field_axis_math(str(field_name)), fontsize=8)
            colorbar.outline.set_visible(False)
            colorbar.ax.tick_params(labelsize=7, width=0.5, length=2.5)

        x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
        y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))

        if self._selection.kind == "point":
            axis.plot(
                self._selection.x,
                self._selection.y,
                marker="o",
                markersize=9,
                markerfacecolor="none",
                markeredgecolor=SELECTED,
                markeredgewidth=1.4,
                zorder=5,
            )
        elif self._selection.kind == "roi":
            x0, x1 = sorted((self._selection.x_min, self._selection.x_max))
            y0, y1 = sorted((self._selection.y_min, self._selection.y_max))
            axis.add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    edgecolor=SELECTED,
                    linewidth=1.2,
                    zorder=5,
                    in_layout=False,
                )
            )

        axis.set_xlabel(r"$x\ (\mathrm{mm})$")
        axis.set_ylabel(r"$y\ (\mathrm{mm})$")
        axis.set_title("Specimen", pad=6)
        axis.grid(True, axis="y", color=GRID_COLOUR, linewidth=0.5)
        _style_axis(axis)
        self._spec_fig.patch.set_facecolor("#FFFFFF")
        span_x = max(x_max - x_min, 1.0e-9)
        span_y = max(y_max - y_min, 1.0e-9)
        axis.set_xlim(x_min - 0.04 * span_x, x_max + 0.04 * span_x)
        axis.set_ylim(y_min - 0.06 * span_y, y_max + 0.10 * span_y)
        axis.set_aspect("equal", adjustable="box")
        axis.autoscale(enable=False)
        experiment = self._session.selected_experiment() if self._session else None
        if experiment is not None:
            add_bc_symbols_2d(axis, experiment, x_min, x_max, y_min, y_max)
        self._spec_xlim = axis.get_xlim()
        self._spec_ylim = axis.get_ylim()
        self._rubber = None
        self._spec_bg = None
        self._update_status()
        self._spec_canvas.draw_idle()

    def _update_status(self) -> None:
        geometry = self._active_cloud()
        if self._selection.kind == "point":
            self._status.setText(
                f"Point  ({self._selection.x:.3g}, {self._selection.y:.3g}) mm"
            )
            return
        if self._selection.kind == "roi":
            n_pts = 0
            if geometry is not None:
                n_pts = int(selection_mask(geometry.x, geometry.y, self._selection).sum())
            self._status.setText(f"ROI mean  ({n_pts} points)")
            return
        self._status.setText("Whole-field mean")

    def _lock_specimen_view(self) -> None:
        axis = self._spec_ax
        if self._spec_xlim is not None:
            axis.set_xlim(self._spec_xlim)
        if self._spec_ylim is not None:
            axis.set_ylim(self._spec_ylim)
        axis.autoscale(enable=False)

    def _ensure_rubber(self) -> Rectangle:
        axis = self._spec_ax
        if self._rubber is not None and self._rubber.axes is axis:
            return self._rubber
        rubber = Rectangle(
            (0.0, 0.0),
            0.0,
            0.0,
            fill=False,
            edgecolor=SELECTED,
            linestyle="--",
            linewidth=1.0,
            zorder=6,
            visible=False,
            clip_on=True,
            animated=True,
        )
        try:
            rubber.set_in_layout(False)
        except Exception:
            pass
        axis.add_patch(rubber)
        self._lock_specimen_view()
        self._rubber = rubber
        return rubber

    def _paint_rubber(self) -> None:
        rubber = self._rubber
        if rubber is None:
            return
        canvas = self._spec_canvas
        if self._spec_bg is not None:
            try:
                canvas.restore_region(self._spec_bg)
                self._spec_ax.draw_artist(rubber)
                canvas.blit(self._spec_ax.bbox)
                canvas.flush_events()
                return
            except Exception:
                self._spec_bg = None
        self._lock_specimen_view()
        canvas.draw_idle()

    def _on_spec_press(self, event) -> None:
        if event.inaxes != self._spec_ax or event.xdata is None or event.ydata is None:
            return
        if event.button != 1:
            return
        self._drag_origin = (float(event.xdata), float(event.ydata), event.x, event.y)
        rubber = self._ensure_rubber()
        rubber.set_visible(False)
        self._lock_specimen_view()
        self._spec_canvas.draw()
        try:
            self._spec_bg = self._spec_canvas.copy_from_bbox(self._spec_ax.bbox)
        except Exception:
            self._spec_bg = None

    def _on_spec_move(self, event) -> None:
        if self._drag_origin is None or event.inaxes != self._spec_ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        x0, y0, px0, py0 = self._drag_origin
        if abs(event.x - px0) < 4 and abs(event.y - py0) < 4:
            return
        x1, y1 = float(event.xdata), float(event.ydata)
        x_lo, x_hi = sorted((x0, x1))
        y_lo, y_hi = sorted((y0, y1))
        rubber = self._ensure_rubber()
        rubber.set_bounds(x_lo, y_lo, x_hi - x_lo, y_hi - y_lo)
        rubber.set_visible(True)
        self._paint_rubber()

    def _on_spec_release(self, event) -> None:
        origin = self._drag_origin
        self._drag_origin = None
        self._spec_bg = None
        if origin is None:
            return
        rubber = self._rubber
        if rubber is not None:
            rubber.set_visible(False)
        if event.inaxes != self._spec_ax or event.xdata is None or event.ydata is None:
            self._lock_specimen_view()
            self._spec_canvas.draw_idle()
            return
        x0, y0, px0, py0 = origin
        x1, y1 = float(event.xdata), float(event.ydata)
        dragged = abs(event.x - px0) >= 6 or abs(event.y - py0) >= 6
        if dragged:
            self._btn_roi.setChecked(True)
            self._selection = SpatialSelection(
                kind="roi",
                x_min=min(x0, x1),
                x_max=max(x0, x1),
                y_min=min(y0, y1),
                y_max=max(y0, y1),
            )
            self._redraw()
            return
        self._select_point(x1, y1)

    def _select_point(self, x_click: float, y_click: float) -> None:
        geometry = self._geometry()
        if geometry is None or geometry.x is None or geometry.y is None:
            return
        x = np.asarray(geometry.x, dtype=float)
        y = np.asarray(geometry.y, dtype=float)
        nearest = int(np.nanargmin((x - x_click) ** 2 + (y - y_click) ** 2))
        self._btn_point.setChecked(True)
        self._selection = SpatialSelection(
            kind="point",
            x=float(x[nearest]),
            y=float(y[nearest]),
        )
        self._redraw()
