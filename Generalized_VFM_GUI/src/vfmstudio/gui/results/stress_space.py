"""MatchID-style stress-resultant space: DIC cloud in 2D (left) and 3D (right).

The 2D scatter stays matplotlib. Interactive 3D uses VTK / PyVista so orbit,
pan, and zoom stay GPU-backed. matplotlib mplot3d is only the offscreen
fallback (tests, headless). Export snapshots the live VTK view.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QLabel, QSizePolicy, QSplitter, QVBoxLayout, QWidget

from vfmstudio.core.document import SessionDocument
from vfmstudio.core.stress_space import (
    ALL_STEPS,
    COLOR_NONE,
    COLOR_STEP,
    StressSpaceCloud,
    default_plot_axes,
    load_experiment_clouds,
    resolve_axis_key,
    resolve_color_key,
    resultant_spec,
)
from vfmstudio.gui.results.plot_style import (
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    apply_matplotlib_style,
    empty_message,
    heading_stylesheet,
    pad_limits,
    style_axis,
)

MAX_MPL_POINTS = 40_000
MAX_VTK_POINTS = 250_000
POINT_COLOR = "#0072B2"
CMAP = "coolwarm"


@dataclass
class _PreparedCloud:
    x2d: np.ndarray
    y2d: np.ndarray
    x3d: np.ndarray
    y3d: np.ndarray
    z3d: np.ndarray
    colour: Optional[np.ndarray]
    vmin: Optional[float]
    vmax: Optional[float]
    n_finite: int


def _subsample(n: int, limit: int) -> slice:
    if n <= limit:
        return slice(None)
    stride = int(np.ceil(n / max(limit, 1)))
    return slice(None, None, max(stride, 1))


def _finite_mask(*columns: Optional[np.ndarray]) -> Optional[np.ndarray]:
    mask = None
    for column in columns:
        if column is None:
            continue
        finite = np.isfinite(np.asarray(column, dtype=float).ravel())
        mask = finite if mask is None else mask & finite
    return mask


def _span(values) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = float(np.min(finite)), float(np.max(finite))
    if high <= low:
        pad = max(abs(high), 1.0) * 0.1
        return low - pad, high + pad
    span = high - low
    return low - 0.08 * span, high + 0.08 * span


def _plain_label(analysis: str, key: str) -> str:
    if key == COLOR_STEP:
        return "Load step"
    spec = resultant_spec(analysis, key)
    if spec is None:
        return key
    return f"{spec.key} ({spec.unit_math})"


def _math_label(analysis: str, key: str) -> str:
    spec = resultant_spec(analysis, key)
    if spec is None:
        return key
    return spec.axis_label()


def _color_label(analysis: str, key: str) -> str:
    if key == COLOR_STEP:
        return "Load step"
    spec = resultant_spec(analysis, key)
    if spec is None:
        return key
    return spec.axis_label()


class StressSpacePanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        apply_matplotlib_style()
        self._title = QLabel("Stress resultant space")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title.setWordWrap(True)
        self._title.setStyleSheet(heading_stylesheet())
        self._subtitle = QLabel("")
        self._subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._subtitle.setWordWrap(True)
        self._subtitle.setVisible(False)
        self._subtitle.setStyleSheet(heading_stylesheet(muted=True))
        self._figure = Figure(facecolor="#FFFFFF")
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._vtk_page = QWidget()
        self._vtk_layout = QVBoxLayout(self._vtk_page)
        self._vtk_layout.setContentsMargins(0, 0, 0, 0)
        self._vtk_page.setVisible(False)
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.addWidget(self._canvas)
        self._splitter.addWidget(self._vtk_page)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 10)
        layout.setSpacing(6)
        layout.addWidget(self._title)
        layout.addWidget(self._subtitle)
        layout.addWidget(self._splitter, 1)

        self._document: Optional[SessionDocument] = None
        self._result: Any = None
        self._cloud: Optional[StressSpaceCloud] = None
        self._cache_key: Optional[tuple] = None
        self._experiment_id: Optional[int] = None
        self._step_index: int = ALL_STEPS
        self._x2d = "Mx"
        self._y2d = "My"
        self._x3d = "Mx"
        self._y3d = "My"
        self._z3d = "Mxy"
        self._color = "Mxy"
        self._vtk = None
        self._pv = None
        self._vtk_failed = False
        self._vtk_view_key: Optional[tuple] = None
        self._vtk_camera = None

    def update_from(
        self,
        document: Optional[SessionDocument],
        result: Any,
    ) -> None:
        self._document = document
        self._result = result
        self._cache_key = None
        self._cloud = None
        self._sync_axis_keys()
        self.refresh()

    def apply_view(
        self,
        experiment_id: Optional[int] = None,
        step_index: Optional[int] = None,
        x2d: str = "",
        y2d: str = "",
        x3d: str = "",
        y3d: str = "",
        z3d: str = "",
        color: str = "",
    ) -> None:
        self._experiment_id = None if experiment_id is None else int(experiment_id)
        self._step_index = ALL_STEPS if step_index is None else int(step_index)
        self._x2d = x2d
        self._y2d = y2d
        self._x3d = x3d
        self._y3d = y3d
        self._z3d = z3d
        self._color = color
        self._sync_axis_keys()
        self.refresh()

    def figure_for_export(self):
        """Matplotlib figure for File → Export. VTK view is snapshotted in."""
        if self._vtk is None or not self._vtk_page.isVisible():
            return self._figure
        cloud = self._cloud
        prepared = self._prepare(cloud, MAX_MPL_POINTS) if cloud is not None else None
        if prepared is None:
            return self._figure
        try:
            image = self._vtk.screenshot(return_img=True)
        except Exception:
            return self._figure
        figure = Figure(facecolor="#FFFFFF", figsize=(11.0, 4.8))
        axis_2d = figure.add_subplot(1, 2, 1)
        axis_3d = figure.add_subplot(1, 2, 2)
        figure.subplots_adjust(left=0.08, right=0.96, top=0.90, bottom=0.16, wspace=0.22)
        mappable = self._paint_2d(axis_2d, cloud.analysis_type, prepared)
        axis_3d.imshow(np.asarray(image))
        axis_3d.set_title("3D", color=INK_PRIMARY, pad=6)
        axis_3d.axis("off")
        if mappable is not None:
            self._attach_colorbar(figure, mappable, [axis_2d, axis_3d], cloud.analysis_type)
        return figure

    def closeEvent(self, event) -> None:
        self._close_vtk()
        super().closeEvent(event)

    def _sync_axis_keys(self) -> None:
        analysis = self._analysis_type()
        defaults = default_plot_axes(analysis)
        self._x2d = resolve_axis_key(analysis, self._x2d, defaults["x2d"])
        self._y2d = resolve_axis_key(analysis, self._y2d, defaults["y2d"])
        self._x3d = resolve_axis_key(analysis, self._x3d, defaults["x3d"])
        self._y3d = resolve_axis_key(analysis, self._y3d, defaults["y3d"])
        self._z3d = resolve_axis_key(analysis, self._z3d, defaults["z3d"])
        self._color = resolve_color_key(analysis, self._color, defaults["color"])

    def _analysis_type(self) -> str:
        if self._document is not None:
            return str(self._document.material.analysis_type)
        if self._cloud is not None:
            return self._cloud.analysis_type
        return "out_of_plane"

    def _ensure_cloud(self) -> Optional[StressSpaceCloud]:
        key = (id(self._result), self._experiment_id, int(self._step_index))
        if self._cloud is not None and self._cache_key == key:
            return self._cloud
        cloud = load_experiment_clouds(
            self._document,
            self._result,
            self._experiment_id,
            self._step_index,
        )
        self._cloud = cloud
        self._cache_key = key
        return cloud

    def _use_vtk(self) -> bool:
        if self._vtk_failed:
            return False
        from vfmstudio.gui.viewport.interactor import viewport_enabled

        return viewport_enabled()

    def refresh(self) -> None:
        if self._result is None:
            self._show_empty(
                "Run Identify first. Stress resultants are N = A ε and "
                "M = D κ at every DIC point, using the identified stiffness."
            )
            return
        cloud = self._ensure_cloud()
        if cloud is None or cloud.n_points == 0:
            self._show_empty(
                (cloud.warning if cloud is not None else None)
                or "No DIC points to plot."
            )
            return
        prepared_vtk = self._prepare(cloud, MAX_VTK_POINTS)
        if prepared_vtk is None:
            self._show_empty("Resultants are empty or non-finite.")
            return
        self._set_subtitle(cloud, prepared_vtk)
        if self._use_vtk():
            try:
                self._draw_vtk_split(cloud, prepared_vtk)
                return
            except Exception:
                self._close_vtk()
                self._vtk_failed = True
        prepared_mpl = self._prepare(cloud, MAX_MPL_POINTS)
        if prepared_mpl is None:
            self._show_empty("Resultants are empty or non-finite.")
            return
        self._draw_matplotlib_combined(cloud, prepared_mpl)

    def _show_empty(self, text: str) -> None:
        self._set_vtk_visible(False)
        self._subtitle.setVisible(False)
        self._figure.clear()
        empty_message(self._figure, text)
        self._canvas.draw_idle()

    def _set_subtitle(self, cloud: StressSpaceCloud, prepared: _PreparedCloud) -> None:
        extra = ""
        if prepared.n_finite > prepared.x3d.size:
            extra = f" Showing {prepared.x3d.size:,} of {prepared.n_finite:,} points."
        step_text = (
            "all load steps"
            if self._step_index == ALL_STEPS
            else (cloud.step_labels[0] if cloud.step_labels else "one load step")
        )
        self._subtitle.setText(
            f"{cloud.experiment_name} · {step_text} · {prepared.n_finite:,} DIC "
            f"point{'s' if prepared.n_finite != 1 else ''}.{extra}"
        )
        self._subtitle.setVisible(True)

    def _prepare(self, cloud: StressSpaceCloud, limit: int) -> Optional[_PreparedCloud]:
        x2d = cloud.column(self._x2d)
        y2d = cloud.column(self._y2d)
        x3d = cloud.column(self._x3d)
        y3d = cloud.column(self._y3d)
        z3d = cloud.column(self._z3d)
        color_values = None
        if self._color not in {"", COLOR_NONE}:
            color_values = cloud.column(self._color)
        mask = _finite_mask(x2d, y2d, x3d, y3d, z3d, color_values)
        if mask is None or not np.any(mask):
            return None
        n_finite = int(np.count_nonzero(mask))
        pick = np.flatnonzero(mask)[_subsample(n_finite, limit)]
        colour = None if color_values is None else np.asarray(color_values, dtype=float)[pick]
        vmin = vmax = None
        if colour is not None and colour.size:
            finite_c = colour[np.isfinite(colour)]
            if finite_c.size:
                vmin, vmax = float(np.min(finite_c)), float(np.max(finite_c))
                if vmax <= vmin:
                    vmax = vmin + 1.0
        return _PreparedCloud(
            x2d=np.asarray(x2d, dtype=float)[pick],
            y2d=np.asarray(y2d, dtype=float)[pick],
            x3d=np.asarray(x3d, dtype=float)[pick],
            y3d=np.asarray(y3d, dtype=float)[pick],
            z3d=np.asarray(z3d, dtype=float)[pick],
            colour=colour,
            vmin=vmin,
            vmax=vmax,
            n_finite=n_finite,
        )

    def _paint_2d(self, axis, analysis: str, prepared: _PreparedCloud):
        style_axis(axis, grid="both")
        norm = None
        if prepared.vmin is not None and prepared.vmax is not None:
            norm = Normalize(vmin=prepared.vmin, vmax=prepared.vmax)
        scatter = axis.scatter(
            prepared.x2d,
            prepared.y2d,
            c=prepared.colour if prepared.colour is not None else POINT_COLOR,
            cmap=CMAP if prepared.colour is not None else None,
            norm=norm,
            s=8,
            marker="o",
            linewidths=0.0,
            alpha=0.55,
            rasterized=True,
            zorder=3,
        )
        pad_limits(axis, prepared.x2d, which="x")
        pad_limits(axis, prepared.y2d, which="y")
        axis.set_xlabel(_math_label(analysis, self._x2d), labelpad=8)
        axis.set_ylabel(_math_label(analysis, self._y2d), labelpad=8)
        axis.set_title("2D", color=INK_PRIMARY, pad=6)
        return scatter if prepared.colour is not None else None

    def _attach_colorbar(self, figure, mappable, axes, analysis: str) -> None:
        colorbar = figure.colorbar(mappable, ax=axes, fraction=0.03, pad=0.03)
        colorbar.set_label(_color_label(analysis, self._color), color=INK_PRIMARY)
        colorbar.ax.tick_params(labelsize=8, colors=INK_MUTED, width=0.6, length=3)
        colorbar.outline.set_linewidth(0.6)
        colorbar.outline.set_edgecolor(INK_MUTED)

    def _draw_matplotlib_combined(self, cloud: StressSpaceCloud, prepared: _PreparedCloud) -> None:
        self._set_vtk_visible(False)
        self._figure.clear()
        analysis = cloud.analysis_type
        axis_2d = self._figure.add_subplot(1, 2, 1)
        axis_3d = self._figure.add_subplot(1, 2, 2, projection="3d")
        self._figure.subplots_adjust(
            left=0.08, right=0.96, top=0.90, bottom=0.16, wspace=0.28
        )
        mappable = self._paint_2d(axis_2d, analysis, prepared)
        scatter_3d = axis_3d.scatter(
            prepared.x3d,
            prepared.y3d,
            prepared.z3d,
            c=prepared.colour if prepared.colour is not None else POINT_COLOR,
            cmap=CMAP if prepared.colour is not None else None,
            norm=mappable.norm if mappable is not None else None,
            s=6,
            marker="o",
            linewidths=0.0,
            alpha=0.55,
            rasterized=True,
            depthshade=False,
        )
        if mappable is None and prepared.colour is not None:
            mappable = scatter_3d
        _style_3d_axis(axis_3d, prepared.x3d, prepared.y3d, prepared.z3d)
        axis_3d.set_xlabel(_math_label(analysis, self._x3d))
        axis_3d.set_ylabel(_math_label(analysis, self._y3d))
        axis_3d.set_zlabel(_math_label(analysis, self._z3d))
        axis_3d.set_title("3D", color=INK_PRIMARY, pad=2)
        if mappable is not None:
            self._attach_colorbar(self._figure, mappable, [axis_2d, axis_3d], analysis)
        self._canvas.draw_idle()

    def _draw_vtk_split(self, cloud: StressSpaceCloud, prepared: _PreparedCloud) -> None:
        analysis = cloud.analysis_type
        self._figure.clear()
        axis_2d = self._figure.add_subplot(1, 1, 1)
        right = 0.86 if prepared.colour is not None else 0.96
        self._figure.subplots_adjust(left=0.16, right=right, top=0.90, bottom=0.16)
        mappable = self._paint_2d(axis_2d, analysis, prepared)
        if mappable is not None:
            self._attach_colorbar(self._figure, mappable, axis_2d, analysis)
        self._canvas.draw_idle()
        self._set_vtk_visible(True)
        self._draw_vtk_cloud(analysis, prepared)
        QTimer.singleShot(0, self._render_vtk)

    def _draw_vtk_cloud(self, analysis: str, prepared: _PreparedCloud) -> None:
        from vfmstudio.gui.viewport import overlays

        plotter = self._ensure_vtk()
        view_key = (
            self._x3d,
            self._y3d,
            self._z3d,
            self._experiment_id,
            int(self._step_index),
        )
        camera = None
        if view_key == self._vtk_view_key:
            try:
                camera = plotter.camera_position
            except Exception:
                camera = self._vtk_camera
        try:
            plotter.clear()
        except Exception:
            pass
        xyz = np.column_stack((prepared.x3d, prepared.y3d, prepared.z3d)).astype(
            np.float32, copy=False
        )
        cloud = self._pv.PolyData(xyz)
        try:
            plotter.set_scale(1.0, 1.0, 1.0)
        except Exception:
            pass
        mesh_kwargs = dict(
            style="points",
            render_points_as_spheres=False,
            point_size=5,
            opacity=0.65,
            lighting=False,
            show_scalar_bar=False,
            pickable=False,
            reset_camera=False,
        )
        if prepared.colour is not None:
            cloud["values"] = np.asarray(prepared.colour, dtype=np.float32)
            mesh_kwargs.update(
                scalars="values",
                cmap=CMAP,
                clim=(prepared.vmin, prepared.vmax),
            )
            color = None
        else:
            color = POINT_COLOR
        _add_point_mesh(plotter, cloud, color=color, **mesh_kwargs)
        plotter.set_background("white")
        try:
            plotter.enable_parallel_projection()
        except Exception:
            pass
        try:
            sx, sy, sz = _cubic_scales(prepared.x3d, prepared.y3d, prepared.z3d)
            plotter.set_scale(sx, sy, sz)
        except Exception:
            pass
        xlabel = _plain_label(analysis, self._x3d)
        ylabel = _plain_label(analysis, self._y3d)
        zlabel = _plain_label(analysis, self._z3d)
        try:
            plotter.show_bounds(
                grid="front",
                location="outer",
                ticks="outside",
                n_xlabels=4,
                n_ylabels=4,
                n_zlabels=4,
                xtitle=xlabel,
                ytitle=ylabel,
                ztitle=zlabel,
                font_size=9,
                color=INK_SECONDARY,
                fmt="%.3g",
            )
        except Exception:
            try:
                plotter.show_grid(
                    xtitle=xlabel,
                    ytitle=ylabel,
                    ztitle=zlabel,
                    color=INK_SECONDARY,
                )
            except Exception:
                pass
        overlays.add_plain_text(
            plotter,
            "3D",
            position="upper_edge",
            font_size=10,
            color=INK_PRIMARY,
            name="ss_title",
        )
        if camera is not None:
            try:
                plotter.camera_position = camera
            except Exception:
                camera = None
        if camera is None:
            try:
                plotter.reset_camera()
                plotter.camera.azimuth = -58.0
                plotter.camera.elevation = 22.0
            except Exception:
                try:
                    plotter.view_isometric()
                except Exception:
                    pass
        self._render_vtk()
        try:
            self._vtk_camera = plotter.camera_position
        except Exception:
            self._vtk_camera = camera
        self._vtk_view_key = view_key

    def _ensure_vtk(self):
        import pyvista as pv
        from pyvistaqt import QtInteractor

        if self._vtk is not None:
            return self._vtk
        try:
            plotter = QtInteractor(
                self._vtk_page,
                auto_update=False,
                line_smoothing=False,
                point_smoothing=False,
                polygon_smoothing=False,
            )
        except TypeError:
            plotter = QtInteractor(self._vtk_page)
        if hasattr(plotter, "set_multi_samples"):
            try:
                plotter.set_multi_samples(0)
            except Exception:
                pass
        try:
            window = getattr(plotter, "ren_win", None) or getattr(
                plotter, "render_window", None
            )
            if window is not None:
                window.SetDesiredUpdateRate(0.0001)
        except Exception:
            pass
        interactor = getattr(plotter, "interactor", plotter)
        interactor.setMinimumSize(0, 0)
        interactor.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored
        )
        self._vtk_layout.addWidget(interactor)
        self._vtk = plotter
        self._pv = pv
        return plotter

    def _close_vtk(self) -> None:
        if self._vtk is None:
            return
        try:
            self._vtk.close()
        except Exception:
            pass
        self._vtk = None
        self._pv = None
        self._vtk_view_key = None
        self._vtk_camera = None
        while self._vtk_layout.count():
            item = self._vtk_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

    def _render_vtk(self) -> None:
        if self._vtk is None:
            return
        try:
            self._vtk.render()
        except Exception:
            pass

    def _set_vtk_visible(self, visible: bool) -> None:
        self._vtk_page.setVisible(visible)
        if visible:
            width = max(int(self._splitter.width()), 2)
            self._splitter.setSizes([width // 2, width - width // 2])


_OPTIONAL_MESH_KEYS = (
    "reset_camera",
    "pickable",
    "lighting",
    "render_points_as_spheres",
    "show_scalar_bar",
    "style",
    "opacity",
    "point_size",
)


def _add_point_mesh(plotter, cloud, *, color=None, **kwargs) -> None:
    """Add a point cloud, dropping kwargs older PyVista builds reject."""
    if color is not None:
        kwargs["color"] = color
    while True:
        try:
            plotter.add_mesh(cloud, **kwargs)
            return
        except TypeError:
            removable = [key for key in _OPTIONAL_MESH_KEYS if key in kwargs]
            if not removable:
                raise
            kwargs.pop(removable[0])


def _cubic_scales(x, y, z) -> tuple[float, float, float]:
    """Scale each data axis so the VTK box is cubic, like matplotlib 3D."""

    def _extent(values) -> float:
        low, high = _span(values)
        extent = high - low
        return extent if extent > 0 else 1.0

    sx, sy, sz = _extent(x), _extent(y), _extent(z)
    target = max(sx, sy, sz, 1e-12)
    return target / sx, target / sy, target / sz


def _style_3d_axis(axis, x, y, z) -> None:
    axis.set_xlim(*_span(x))
    axis.set_ylim(*_span(y))
    axis.set_zlim(*_span(z))
    axis.view_init(elev=22, azim=-58)
    axis.tick_params(axis="x", pad=1, labelsize=7, colors=INK_SECONDARY)
    axis.tick_params(axis="y", pad=1, labelsize=7, colors=INK_SECONDARY)
    axis.tick_params(axis="z", pad=1, labelsize=7, colors=INK_SECONDARY)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
    axis.zaxis.set_major_locator(MaxNLocator(nbins=4))
    axis.xaxis.label.set_color(INK_PRIMARY)
    axis.yaxis.label.set_color(INK_PRIMARY)
    axis.zaxis.label.set_color(INK_PRIMARY)
    axis.xaxis.labelpad = 4
    axis.yaxis.labelpad = 4
    axis.zaxis.labelpad = 4
    try:
        axis.set_proj_type("ortho")
    except Exception:
        pass
    try:
        pane_colour = (1.0, 1.0, 1.0, 1.0)
        axis.xaxis.set_pane_color(pane_colour)
        axis.yaxis.set_pane_color(pane_colour)
        axis.zaxis.set_pane_color(pane_colour)
    except Exception:
        pass
