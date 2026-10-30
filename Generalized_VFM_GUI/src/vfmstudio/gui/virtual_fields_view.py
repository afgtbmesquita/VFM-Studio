"""Tiled optimal virtual fields: one panel per unknown, 3D or 2D contour."""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
from matplotlib import cm
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from vfmstudio.core import bridge, mesh_convert
from vfmstudio.core.document import SessionDocument, parameter_math_label
from vfmstudio.core.virtual_fields import virtual_fields_for_target
from vfmstudio.gui.results.plot_style import (
    apply_matplotlib_style,
    heading_stylesheet,
)
from vfmstudio.gui.results.scene3d import (
    fit_vtk_camera,
    padded_span,
    plate_z_scale,
    show_vtk_axes,
    size_vtk_interactor,
)

INK_PRIMARY = "#141413"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8f8e88"
CMAP = "coolwarm"


def _tile_shape(n: int) -> tuple[int, int]:
    if n <= 1:
        return 1, 1
    if n == 2:
        return 1, 2
    if n <= 4:
        return 2, 2
    if n <= 6:
        return 2, 3
    ncols = int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))
    return nrows, ncols


def _3d_gridspec_margins(n: int) -> dict[str, float]:
    """Leave enough figure-edge room for 3D tick labels and axis names.

    mplot3d draws x/y labels outside the axes rectangle. A 5% bottom margin
    clips the bottom-row ``x (mm)`` labels against the Jobs dock.
    """
    nrows, ncols = _tile_shape(n)
    return {
        "left": 0.11,
        "right": 0.90,
        "top": 0.90 if nrows > 1 else 0.88,
        "bottom": 0.16 if nrows > 1 else 0.14,
        "wspace": 0.36 if ncols > 1 else 0.12,
        "hspace": 0.46 if nrows > 1 else 0.18,
    }


def _triangles(mesh_pkg) -> np.ndarray:
    conn = np.asarray(mesh_convert.mesh_quads(mesh_pkg), dtype=np.int64)
    if conn.ndim != 2 or conn.shape[0] == 0:
        raise ValueError("Mesh has no element connectivity.")
    if conn.shape[1] == 3:
        return conn
    if conn.shape[1] == 4:
        return np.vstack([conn[:, [0, 1, 2]], conn[:, [0, 2, 3]]])
    raise ValueError("Expected triangular or quadrilateral elements.")


def _u_from_field(item: Any) -> np.ndarray:
    if hasattr(item, "nodal_variables_k"):
        return np.asarray(item.nodal_variables_k, dtype=float).ravel()
    return np.asarray(item, dtype=float).ravel()


def collect_virtual_fields(result: Any, document: Optional[SessionDocument]):
    fields = list(getattr(result, "optimal_field_results", None) or [])
    u_list = [_u_from_field(item) for item in fields]
    labels: list[str] = []
    model = getattr(result, "constitutive_model", None)
    if model is not None:
        labels = list(getattr(model, "parameter_labels", []) or [])
    if document is not None and len(labels) < len(u_list):
        labels = list(document.material.labels())
    while len(labels) < len(u_list):
        labels.append(f"P{len(labels) + 1}")
    return u_list, labels[: len(u_list)]


def _mesh_for_fields(document: Optional[SessionDocument], u: np.ndarray):
    if document is None:
        return None
    n_dof = int(np.asarray(u).size)
    for experiment in document.experiments:
        try:
            mesh_pkg = mesh_convert.build_experiment_mesh(document, experiment)
        except Exception:
            continue
        if int(getattr(mesh_pkg, "n_dofs", -1)) == n_dof:
            return mesh_pkg
    return None


VIEW_MODES = ("3D surface", "2D contour")

QUANTITY_ITEMS = (
    ("w*", r"$w^{*}$"),
    ("kx*", r"$\kappa_x^{*}$"),
    ("ky*", r"$\kappa_y^{*}$"),
    ("kxy*", r"$\kappa_{xy}^{*}$"),
)


def normalize_view_mode(text: str | None) -> str:
    return "2D contour" if str(text or "").startswith("2D") else "3D surface"


def normalize_quantity(text: str | None) -> str:
    key = str(text or "w*").strip()
    allowed = {item[0] for item in QUANTITY_ITEMS}
    return key if key in allowed else "w*"


QUANTITY_CBAR = {
    "w*": r"$w^{*}\ (\mathrm{mm})$",
    "kx*": r"$\kappa_x^{*}\ (\mathrm{mm}^{-1})$",
    "ky*": r"$\kappa_y^{*}\ (\mathrm{mm}^{-1})$",
    "kxy*": r"$\kappa_{xy}^{*}\ (\mathrm{mm}^{-1})$",
}

QUANTITY_AXIS = {
    "w*": "w* (mm)",
    "kx*": "kx* (1/mm)",
    "ky*": "ky* (1/mm)",
    "kxy*": "kxy* (1/mm)",
}


def _scalar_and_label(post, mesh_pkg, u: np.ndarray, quantity: str):
    """Nodal scalar for the selected virtual-field quantity."""
    analysis = str(getattr(mesh_pkg, "analysis_type", "out_of_plane")).lower()
    key = str(quantity or "w*").strip().lower()
    if key in {"w*", "w"}:
        if analysis in {"in_plane", "inplane"}:
            uu = np.asarray(post.nodal_field(mesh_pkg, u, "U"), dtype=float)
            vv = np.asarray(post.nodal_field(mesh_pkg, u, "V"), dtype=float)
            return np.hypot(uu, vv), r"$|\mathbf{u}^*|\ (\mathrm{mm})$"
        ww = np.asarray(post.nodal_field(mesh_pkg, u, "W"), dtype=float)
        return ww, QUANTITY_CBAR["w*"]
    strains = np.asarray(
        post.nodal_virtual_generalized_strains(mesh_pkg, u), dtype=float
    )
    offset = 3 if analysis == "both" else 0
    column = {"kx*": 0, "ky*": 1, "kxy*": 2}.get(key)
    if column is None:
        raise ValueError(f"Unknown virtual-field quantity {quantity!r}.")
    return strains[:, offset + column].copy(), QUANTITY_CBAR[key]


def _style_2d(axis) -> None:
    axis.set_facecolor("#FFFFFF")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(INK_SECONDARY)
        axis.spines[side].set_linewidth(0.6)
    axis.tick_params(colors=INK_SECONDARY, width=0.6, size=3.0, labelsize=7.5)
    axis.xaxis.label.set_color(INK_PRIMARY)
    axis.yaxis.label.set_color(INK_PRIMARY)
    axis.title.set_color(INK_PRIMARY)


def _style_3d(axis) -> None:
    axis.set_facecolor("#FFFFFF")
    axis.tick_params(colors=INK_SECONDARY, labelsize=7)
    axis.xaxis.label.set_color(INK_PRIMARY)
    axis.yaxis.label.set_color(INK_PRIMARY)
    axis.zaxis.label.set_color(INK_PRIMARY)
    axis.title.set_color(INK_PRIMARY)
    try:
        for axis3 in (axis.xaxis, axis.yaxis, axis.zaxis):
            axis3.pane.fill = False
            axis3.pane.set_edgecolor("#dcdbd5")
    except Exception:
        pass


def _colormap():
    try:
        from matplotlib import colormaps
        return colormaps[CMAP]
    except Exception:
        return cm.get_cmap(CMAP)


def _shared_clim(values_list) -> tuple[float, float]:
    chunks = []
    for values in values_list:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            chunks.append(finite)
    if not chunks:
        return 0.0, 1.0
    finite = np.concatenate(chunks)
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if vmax - vmin <= 0.0:
        pad = max(abs(vmax), 1.0) * 1.0e-6
        return vmin - pad, vmax + pad
    return vmin, vmax


def _scalar_mappable(
    values: np.ndarray, vmin: float | None = None, vmax: float | None = None
) -> ScalarMappable:
    cmap = _colormap()
    if vmin is None or vmax is None:
        vmin, vmax = _shared_clim([values])
    norm = Normalize(vmin=float(vmin), vmax=float(vmax))
    mappable = ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    return mappable


def _quad_facecolors(post, mesh_pkg, scalar: np.ndarray, mappable: ScalarMappable):
    grids_fn = getattr(post, "structured_surface_grids", None)
    if grids_fn is None:
        return None
    _, _, mag_grid = grids_fn(mesh_pkg, np.asarray(scalar, dtype=float))
    corners = np.stack(
        [
            mag_grid[:-1, :-1],
            mag_grid[1:, :-1],
            mag_grid[1:, 1:],
            mag_grid[:-1, 1:],
        ],
        axis=0,
    )
    quad_mean = np.nanmean(corners, axis=0)
    cmap = mappable.cmap
    norm = mappable.norm
    filled = np.where(np.isfinite(quad_mean), quad_mean, 0.0)
    facecolors = cmap(norm(filled))
    facecolors = np.asarray(facecolors, dtype=float)
    facecolors[~np.isfinite(quad_mean)] = (1.0, 1.0, 1.0, 0.0)
    return facecolors


def _add_colorbar(mappable, cax, label: str):
    colorbar = cax.figure.colorbar(mappable, cax=cax)
    colorbar.set_label(label, color=INK_PRIMARY, labelpad=4, fontsize=8)
    colorbar.ax.tick_params(colors=INK_SECONDARY, labelsize=6.5, pad=1)
    try:
        colorbar.locator = MaxNLocator(nbins=5)
        colorbar.update_ticks()
    except Exception:
        pass
    colorbar.outline.set_edgecolor(INK_SECONDARY)
    colorbar.outline.set_linewidth(0.6)
    return colorbar


def _attach_2d_colorbar(axis, mappable, label: str):
    """Place a colorbar flush with the equal-aspect axes, same height."""
    try:
        axis.apply_aspect()
    except Exception:
        pass
    pos = axis.get_position()
    gap = min(0.008, max(0.004, 0.06 * pos.width))
    width = min(0.014, max(0.010, 0.08 * pos.width))
    x0 = pos.x1 + gap
    if x0 + width > 0.985:
        width = max(0.008, 0.985 - x0)
    cax = axis.figure.add_axes([x0, pos.y0, width, pos.height])
    _add_colorbar(mappable, cax, label)
    return cax


def _center_plot_and_cbar(axis, cax, cell) -> None:
    """Shift plot + colorbar as one group to the centre of the original tile."""
    plot = axis.get_position().frozen()
    bar = cax.get_position().frozen()
    cell = cell.frozen() if hasattr(cell, "frozen") else cell
    group_w = bar.x1 - plot.x0
    if group_w >= cell.width or group_w <= 0:
        return
    target_left = cell.x0 + 0.5 * (cell.width - group_w)
    dx = target_left - plot.x0
    axis.set_position([plot.x0 + dx, plot.y0, plot.width, plot.height])
    cax.set_position([bar.x0 + dx, bar.y0, bar.width, bar.height])


def _make_2d_axes(figure: Figure, n: int):
    """Plot axes only; colorbars are attached to the equal-aspect box afterwards."""
    nrows, ncols = _tile_shape(n)
    try:
        figure.set_layout_engine("none")
    except Exception:
        figure.set_tight_layout(False)
    outer = figure.add_gridspec(
        nrows,
        ncols,
        left=0.08,
        right=0.93,
        top=0.90,
        bottom=0.12,
        wspace=0.22 if ncols > 1 else 0.08,
        hspace=0.30 if nrows > 1 else 0.14,
    )
    axes = []
    for index in range(n):
        row, col = divmod(index, ncols)
        axes.append(figure.add_subplot(outer[row, col]))
    return axes


def _make_3d_tile_axes(figure: Figure, n: int):
    """Reserved (3D plot, colorbar) pairs with room for titles and cbar labels."""
    nrows, ncols = _tile_shape(n)
    try:
        figure.set_layout_engine("none")
    except Exception:
        figure.set_tight_layout(False)
    # 3D tick labels and ``x (mm)`` sit outside the axes box; keep them on-canvas.
    outer = figure.add_gridspec(nrows, ncols, **_3d_gridspec_margins(n))
    pairs = []
    for index in range(n):
        row, col = divmod(index, ncols)
        # Title band on top; plot + colorbar share the body of the tile.
        tile = outer[row, col].subgridspec(
            2,
            2,
            height_ratios=(0.10, 0.90),
            width_ratios=(1.0, 0.055),
            wspace=0.10,
            hspace=0.0,
        )
        title_ax = figure.add_subplot(tile[0, :])
        title_ax.set_axis_off()
        axis = figure.add_subplot(tile[1, 0], projection="3d")
        cax = figure.add_subplot(tile[1, 1])
        pairs.append((axis, cax, title_ax))
    return pairs


def _finish_3d_colorbars(pairs) -> None:
    """Shrink and centre each colorbar beside its 3D axes; keep labels on-canvas."""
    for axis, cax, _title_ax in pairs:
        plot = axis.get_position().frozen()
        bar = cax.get_position().frozen()
        height = 0.72 * plot.height
        y0 = plot.y0 + 0.5 * (plot.height - height)
        width = min(max(bar.width, 0.012), 0.018)
        x0 = plot.x1 + min(0.012, max(0.006, 0.04 * plot.width))
        if x0 + width > 0.975:
            width = max(0.010, 0.975 - x0)
        cax.set_position([x0, y0, width, height])


def _set_3d_title(title_ax, label: str) -> None:
    title_ax.clear()
    title_ax.set_axis_off()
    title_ax.text(
        0.5,
        0.15,
        label,
        transform=title_ax.transAxes,
        ha="center",
        va="bottom",
        color=INK_PRIMARY,
        fontsize=10,
    )


def _set_3d_limits_and_box(axis, x, y, z) -> None:
    """Fill the tile with a plate-like 3D box. Do not use data-unit equal aspect."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    zmin, zmax = float(np.nanmin(z)), float(np.nanmax(z))
    if not np.isfinite(xmin) or xmax <= xmin:
        xmin, xmax = 0.0, 1.0
    if not np.isfinite(ymin) or ymax <= ymin:
        ymin, ymax = 0.0, 1.0
    if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax <= zmin:
        zmin, zmax = -1.0, 1.0
    else:
        pad = 0.06 * (zmax - zmin)
        zmin, zmax = zmin - pad, zmax + pad
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)
    axis.set_zlim(zmin, zmax)
    xr = max(xmax - xmin, 1.0e-12)
    yr = max(ymax - ymin, 1.0e-12)
    axis.view_init(elev=28, azim=-58)
    axis.tick_params(axis="x", pad=1, labelsize=6.5)
    axis.tick_params(axis="y", pad=1, labelsize=6.5)
    axis.tick_params(axis="z", pad=1, labelsize=6.5)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
    axis.zaxis.set_major_locator(MaxNLocator(nbins=4))
    axis.xaxis.labelpad = 4
    axis.yaxis.labelpad = 4
    axis.zaxis.labelpad = 8
    try:
        axis.zaxis.set_rotate_label(False)
    except Exception:
        pass
    try:
        axis.zaxis.set_visible(True)
    except Exception:
        pass
    # Visual plate: XY in data units, Z tall enough that the out-of-plane axis reads.
    try:
        axis.set_box_aspect((xr, yr, 0.62 * max(xr, yr)), zoom=0.70)
    except TypeError:
        try:
            axis.set_box_aspect((xr, yr, 0.62 * max(xr, yr)))
        except Exception:
            pass
    try:
        axis.set_proj_type("ortho")
    except Exception:
        pass


def _make_tile_axes(figure: Figure, n: int, *, projection: str | None = None):
    """Backward-compatible tile factory (3D uses reserved colorbar columns)."""
    if projection:
        return [
            (axis, cax) for axis, cax, _title in _make_3d_tile_axes(figure, n)
        ]
    nrows, ncols = _tile_shape(n)
    try:
        figure.set_layout_engine("none")
    except Exception:
        figure.set_tight_layout(False)
    outer = figure.add_gridspec(
        nrows,
        ncols,
        left=0.07,
        right=0.97,
        top=0.92,
        bottom=0.08,
        wspace=0.28 if ncols > 1 else 0.08,
        hspace=0.24 if nrows > 1 else 0.10,
    )
    pairs = []
    for index in range(n):
        row, col = divmod(index, ncols)
        inner = outer[row, col].subgridspec(
            1,
            2,
            width_ratios=(1.0, 0.06),
            wspace=0.08,
        )
        axis = figure.add_subplot(inner[0, 0])
        cax = figure.add_subplot(inner[0, 1])
        pairs.append((axis, cax))
    return pairs


class VirtualFieldTiledView(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        apply_matplotlib_style()
        import matplotlib as mpl

        mpl.rcParams.update(
            {
                "axes.labelsize": 9,
                "xtick.labelsize": 7.5,
                "ytick.labelsize": 7.5,
            }
        )
        self.setObjectName("plotPage")
        self._title = QLabel("Optimal virtual fields")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title.setStyleSheet(heading_stylesheet())
        self._figure = Figure(facecolor="#FFFFFF")
        try:
            self._figure.set_layout_engine("none")
        except Exception:
            self._figure.set_tight_layout(False)
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setStyleSheet("background: #FFFFFF; border: none;")
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setWidget(self._canvas)
        self._stack = QStackedWidget()
        self._stack.addWidget(self._scroll)
        self._vtk_page = QWidget()
        self._vtk_page.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._vtk_layout = QVBoxLayout(self._vtk_page)
        self._vtk_layout.setContentsMargins(0, 0, 0, 0)
        self._stack.addWidget(self._vtk_page)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(self._title)
        layout.addWidget(self._stack, 1)
        self._vtk = None
        self._vtk_shape = None
        self._pv = None
        self._document: Optional[SessionDocument] = None
        self._result = None
        self._warp_scale = 1.0
        self._experiment_id: Optional[int] = None
        self._step_index: Optional[int] = None
        self._view_mode = "3D surface"
        self._quantity = "w*"

    def _fit_canvas_height(self, n: int) -> None:
        """Keep tiles tall enough that 3D tick labels are not clipped."""
        if n <= 0:
            self._canvas.setMinimumHeight(0)
            return
        nrows, _ = _tile_shape(n)
        self._canvas.setMinimumHeight(max(320, 310 * nrows))

    def _use_vtk_tiles(self) -> bool:
        from vfmstudio.gui.viewport.interactor import viewport_enabled

        return viewport_enabled()

    def _show_matplotlib(self) -> None:
        self._stack.setCurrentWidget(self._scroll)

    def _show_vtk(self) -> None:
        self._stack.setCurrentWidget(self._vtk_page)

    def _close_vtk(self) -> None:
        if self._vtk is None:
            return
        try:
            self._vtk.close()
        except Exception:
            pass
        self._vtk = None
        self._vtk_shape = None
        self._pv = None
        while self._vtk_layout.count():
            item = self._vtk_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

    def _ensure_vtk(self, nrows: int, ncols: int):
        import pyvista as pv
        from pyvistaqt import QtInteractor

        shape = (int(nrows), int(ncols))
        if self._vtk is not None and self._vtk_shape == shape:
            try:
                self._vtk.clear()
            except Exception:
                pass
            return self._vtk
        self._close_vtk()
        try:
            plotter = QtInteractor(
                self._vtk_page, shape=shape, auto_update=False
            )
        except TypeError:
            plotter = QtInteractor(self._vtk_page, shape=shape)
        interactor = getattr(plotter, "interactor", plotter)
        size_vtk_interactor(interactor)
        self._vtk_layout.addWidget(interactor)
        self._vtk = plotter
        self._vtk_shape = shape
        self._pv = pv
        return plotter

    def _current_quantity(self) -> str:
        return normalize_quantity(self._quantity)

    def update_from(
        self,
        document: SessionDocument,
        result: Any,
        warp_scale: float = 1.0,
        experiment_id: Optional[int] = None,
        step_index: Optional[int] = None,
        view_mode: Optional[str] = None,
        quantity: Optional[str] = None,
    ) -> None:
        self._document = document
        self._result = result
        self._warp_scale = float(warp_scale)
        self._experiment_id = experiment_id
        self._step_index = step_index
        if view_mode is not None:
            self._view_mode = normalize_view_mode(view_mode)
        if quantity is not None:
            self._quantity = normalize_quantity(quantity)
        self.refresh()

    def refresh(self) -> None:
        result = self._result
        if result is None:
            self._show_matplotlib()
            self._figure.clear()
            self._title.setText("Optimal virtual fields")
            self._message("Run Identify. Virtual fields are computed with FGLS.")
            self._fit_canvas_height(0)
            self._canvas.draw_idle()
            return
        payload = virtual_fields_for_target(
            self._document,
            result,
            experiment_id=self._experiment_id,
            step_index=self._step_index,
        )
        if payload is None or not payload.u_list:
            self._show_matplotlib()
            self._figure.clear()
            self._title.setText("Optimal virtual fields")
            self._message("No virtual fields on this result. Re-run Identify.")
            self._fit_canvas_height(0)
            self._canvas.draw_idle()
            return
        mesh_pkg = payload.mesh_pkg
        if mesh_pkg is None:
            self._show_matplotlib()
            self._figure.clear()
            self._title.setText("Optimal virtual fields")
            self._message(
                f"No experiment mesh matches the stored virtual fields "
                f"({payload.u_list[0].size} DOFs)."
            )
            self._fit_canvas_height(0)
            self._canvas.draw_idle()
            return
        u_list = payload.u_list
        labels = payload.labels
        n = len(u_list)
        quantity = self._current_quantity()
        where = ""
        if payload.experiment_name or payload.step_label:
            where = f"  ·  {payload.experiment_name}, {payload.step_label}"
        self._title.setText(
            f"Optimal virtual fields{where}  "
            f"({n} unknown{'s' if n != 1 else ''}, {quantity})"
        )
        if payload.warning:
            self._title.setToolTip(payload.warning)
        else:
            self._title.setToolTip(
                "One tile per constitutive unknown. FGLS recomputes these "
                "fields at every load step of every experiment."
            )
        self._fit_canvas_height(n)
        try:
            if self._view_mode.startswith("2D"):
                self._show_matplotlib()
                self._figure.clear()
                self._draw_contours(mesh_pkg, u_list, labels, quantity)
            else:
                self._draw_surfaces(mesh_pkg, u_list, labels, quantity)
        except Exception as exc:
            self._show_matplotlib()
            self._figure.clear()
            self._message(str(exc))
            self._fit_canvas_height(0)
        if self._stack.currentWidget() is self._scroll:
            self._canvas.draw_idle()

    def _message(self, text: str) -> None:
        axis = self._figure.add_subplot(111)
        axis.set_facecolor("#FFFFFF")
        axis.axis("off")
        axis.text(
            0.5, 0.5, text, ha="center", va="center",
            color=INK_MUTED, fontsize=10, transform=axis.transAxes,
        )

    def _draw_contours(self, mesh_pkg, u_list, labels, quantity: str) -> None:
        post = bridge.vfm_virtual_field_post()
        xy = np.asarray(mesh_pkg.node_coords, dtype=float)
        x = xy[:, 0]
        y = xy[:, 1]
        tris = _triangles(mesh_pkg)
        pending = []
        for axis, u, name in zip(_make_2d_axes(self._figure, len(u_list)), u_list, labels):
            scalar, cbar_label = _scalar_and_label(post, mesh_pkg, u, quantity)
            mappable = _scalar_mappable(scalar)
            finite = np.isfinite(scalar)
            span = float(np.ptp(scalar[finite])) if np.any(finite) else 0.0
            if span > 0.0:
                artist = axis.tricontourf(
                    x, y, tris, scalar,
                    levels=21,
                    cmap=CMAP,
                    vmin=mappable.norm.vmin,
                    vmax=mappable.norm.vmax,
                )
                mappable = artist
            else:
                mappable = axis.tripcolor(
                    x, y, tris, scalar, cmap=CMAP, shading="gouraud",
                )
            axis.set_xlabel(r"$x$ (mm)")
            axis.set_ylabel(r"$y$ (mm)")
            axis.set_title(parameter_math_label(name), pad=8)
            _style_2d(axis)
            cell = axis.get_position().frozen()
            axis.set_aspect("equal", adjustable="box")
            axis.set_anchor("C")
            pending.append((axis, mappable, cbar_label, cell))
        self._canvas.draw()
        for axis, mappable, cbar_label, cell in pending:
            cax = _attach_2d_colorbar(axis, mappable, cbar_label)
            _center_plot_and_cbar(axis, cax, cell)

    def _draw_surfaces(self, mesh_pkg, u_list, labels, quantity: str) -> None:
        if self._use_vtk_tiles():
            try:
                self._draw_vtk_surfaces(mesh_pkg, u_list, labels, quantity)
                self._show_vtk()
                return
            except Exception:
                self._close_vtk()
        self._show_matplotlib()
        self._figure.clear()
        self._draw_mpl_surfaces(mesh_pkg, u_list, labels, quantity)

    def _draw_vtk_surfaces(self, mesh_pkg, u_list, labels, quantity: str) -> None:
        from vfmstudio.gui.viewport import overlays

        n = len(u_list)
        nrows, ncols = _tile_shape(n)
        plotter = self._ensure_vtk(nrows, ncols)
        post = bridge.vfm_virtual_field_post()
        warp = self._warp_scale
        xy = np.asarray(mesh_pkg.node_coords, dtype=float)
        scalars = []
        heights = []
        cbar_label = ""
        for u in u_list:
            _cx, _cy, cz = post.nodal_deformed_xyz(mesh_pkg, u)
            scalar, cbar_label = _scalar_and_label(post, mesh_pkg, u, quantity)
            scalars.append(np.asarray(scalar, dtype=float))
            heights.append(warp * np.asarray(cz, dtype=float))
        clim = _shared_clim(scalars)
        all_z = np.concatenate([np.ravel(item) for item in heights]) if heights else np.zeros(1)
        z_scale = plate_z_scale(xy[:, 0], xy[:, 1], all_z)
        xlo, xhi = padded_span(xy[:, 0])
        ylo, yhi = padded_span(xy[:, 1])
        zlo, zhi = padded_span(all_z)
        ztitle = QUANTITY_AXIS.get(normalize_quantity(quantity), "z")
        for index, (name, scalar, cz) in enumerate(zip(labels, scalars, heights)):
            row, col = divmod(index, ncols)
            plotter.subplot(row, col)
            grid = overlays.unstructured_quad_grid(self._pv, mesh_pkg, z=cz)
            grid["vf"] = np.asarray(scalar, dtype=float).ravel()[: grid.n_points]
            kwargs = dict(
                scalars="vf",
                cmap=CMAP,
                clim=clim,
                show_edges=True,
                show_scalar_bar=(index == n - 1),
            )
            if index == n - 1:
                kwargs["scalar_bar_args"] = {"title": ztitle}
            plotter.add_mesh(grid, **kwargs)
            try:
                plotter.set_scale(1.0, 1.0, z_scale)
            except Exception:
                pass
            show_vtk_axes(
                plotter,
                "x (mm)",
                "y (mm)",
                ztitle,
                axes_ranges=[xlo, xhi, ylo, yhi, zlo, zhi],
            )
            overlays.add_plain_text(plotter, str(name), font_size=9)
            fit_vtk_camera(plotter)
        try:
            plotter.link_views()
        except Exception:
            pass
        try:
            plotter.render()
        except Exception:
            pass

    def _draw_mpl_surfaces(self, mesh_pkg, u_list, labels, quantity: str) -> None:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        post = bridge.vfm_virtual_field_post()
        warp = self._warp_scale
        grids = getattr(post, "structured_surface_grids_from_nodal_xyz", None)
        xy = np.asarray(mesh_pkg.node_coords, dtype=float)
        prepared = []
        for u in u_list:
            cx, cy, cz = post.nodal_deformed_xyz(mesh_pkg, u)
            cx = xy[:, 0] + warp * (np.asarray(cx, dtype=float) - xy[:, 0])
            cy = xy[:, 1] + warp * (np.asarray(cy, dtype=float) - xy[:, 1])
            cz = warp * np.asarray(cz, dtype=float)
            scalar, cbar_label = _scalar_and_label(post, mesh_pkg, u, quantity)
            prepared.append((cx, cy, cz, scalar, cbar_label))
        vmin, vmax = _shared_clim([item[3] for item in prepared])
        tiles = _make_3d_tile_axes(self._figure, len(u_list))
        for (axis, cax, title_ax), name, (cx, cy, cz, scalar, cbar_label) in zip(
            tiles, labels, prepared
        ):
            mappable = _scalar_mappable(scalar, vmin=vmin, vmax=vmax)
            drawn = False
            surface_x, surface_y, surface_z = cx, cy, cz
            if grids is not None:
                try:
                    X, Y, Z = grids(mesh_pkg, cx, cy, cz)
                    facecolors = _quad_facecolors(post, mesh_pkg, scalar, mappable)
                    if facecolors is not None:
                        axis.plot_surface(
                            X, Y, Z,
                            facecolors=facecolors,
                            linewidth=0,
                            antialiased=True,
                            rstride=1,
                            cstride=1,
                            shade=False,
                            alpha=0.95,
                        )
                        drawn = True
                        surface_x, surface_y, surface_z = X, Y, Z
                except Exception:
                    drawn = False
            if not drawn:
                tris = _triangles(mesh_pkg)
                try:
                    axis.plot_trisurf(
                        cx, cy, cz, triangles=tris, cmap=CMAP,
                        linewidth=0, antialiased=True,
                        vmin=vmin, vmax=vmax,
                    )
                except TypeError:
                    axis.plot_trisurf(
                        cx, cy, cz, triangles=tris, cmap=CMAP,
                        linewidth=0, antialiased=True,
                    )
            axis.set_xlabel("x (mm)")
            axis.set_ylabel("y (mm)")
            axis.set_zlabel(cbar_label)
            _set_3d_title(title_ax, parameter_math_label(name))
            _style_3d(axis)
            _set_3d_limits_and_box(axis, surface_x, surface_y, surface_z)
            _add_colorbar(mappable, cax, cbar_label)
        self._canvas.draw()
        _finish_3d_colorbars(tiles)
