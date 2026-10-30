"""Shared 3-D scene helpers for VTK and matplotlib result views."""

from __future__ import annotations

import numpy as np

from vfmstudio.gui.results.plot_style import INK_SECONDARY


def padded_span(values, fraction: float = 0.08) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = float(np.min(finite)), float(np.max(finite))
    if high <= low:
        pad = max(abs(high), 1.0) * 0.1
        return low - pad, high + pad
    span = high - low
    return low - fraction * span, high + fraction * span


def axis_extent(values) -> float:
    low, high = padded_span(values)
    extent = high - low
    return extent if extent > 0 else 1.0


def cubic_scales(x, y, z) -> tuple[float, float, float]:
    """Scale each data axis so the 3-D box is a cube (matplotlib-style)."""
    sx, sy, sz = axis_extent(x), axis_extent(y), axis_extent(z)
    target = max(sx, sy, sz, 1e-12)
    return target / sx, target / sy, target / sz


def cubic_display_xyz(x, y, z):
    """Map (x, y, z) into a visual cube; ticks keep the original ranges."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    xlo, xhi = padded_span(x)
    ylo, yhi = padded_span(y)
    zlo, zhi = padded_span(z)
    sx, sy, sz = cubic_scales(x, y, z)
    xmid = 0.5 * (xlo + xhi)
    ymid = 0.5 * (ylo + yhi)
    zmid = 0.5 * (zlo + zhi)
    vis = np.column_stack(
        ((x - xmid) * sx, (y - ymid) * sy, (z - zmid) * sz)
    ).astype(np.float32, copy=False)
    hx = 0.5 * (xhi - xlo) * sx
    hy = 0.5 * (yhi - ylo) * sy
    hz = 0.5 * (zhi - zlo) * sz
    bounds = (-hx, hx, -hy, hy, -hz, hz)
    ranges = (xlo, xhi, ylo, yhi, zlo, zhi)
    return vis, ranges, bounds


def plate_z_scale(x, y, z, *, fraction: float = 0.55) -> float:
    """Inflate out-of-plane height so a plate is not a pancake on screen."""
    xy = max(axis_extent(x), axis_extent(y), 1e-12)
    zr = axis_extent(z)
    if zr <= 1e-15:
        return 1.0
    return (fraction * xy) / zr


def size_vtk_interactor(interactor) -> None:
    from PySide6.QtWidgets import QSizePolicy

    interactor.setMinimumSize(160, 160)
    interactor.setSizePolicy(
        QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
    )


def show_vtk_axes(
    plotter,
    xlabel: str,
    ylabel: str,
    zlabel: str,
    *,
    font_size: int = 9,
    axes_ranges=None,
    bounds=None,
    location: str = "outer",
) -> bool:
    """Draw X/Y/Z cube axes, including the vertical out-of-plane axis."""
    attempts = []
    base = dict(
        grid="back",
        location=location,
        ticks="outside",
        n_xlabels=4,
        n_ylabels=4,
        n_zlabels=4,
        xtitle=xlabel,
        ytitle=ylabel,
        ztitle=zlabel,
        font_size=font_size,
        color=INK_SECONDARY,
        fmt="%.3g",
        show_xaxis=True,
        show_yaxis=True,
        show_zaxis=True,
        show_xlabels=True,
        show_ylabels=True,
        show_zlabels=True,
        padding=0.08,
        use_3d_text=False,
        all_edges=True,
        bold=False,
    )
    if bounds is not None:
        base["bounds"] = list(bounds)
    if axes_ranges is not None:
        attempts.append({**base, "axes_ranges": list(axes_ranges)})
    attempts.append(base)
    attempts.append(
        dict(
            xtitle=xlabel,
            ytitle=ylabel,
            ztitle=zlabel,
            font_size=font_size,
            color=INK_SECONDARY,
            show_zaxis=True,
            show_zlabels=True,
            use_3d_text=False,
            location=location,
        )
    )
    for kwargs in attempts:
        try:
            plotter.show_bounds(**kwargs)
            return True
        except TypeError:
            continue
        except Exception:
            continue
    try:
        plotter.show_grid(
            xtitle=xlabel,
            ytitle=ylabel,
            ztitle=zlabel,
            color=INK_SECONDARY,
        )
        return True
    except Exception:
        pass
    try:
        plotter.add_axes(line_width=2, labels_off=False)
        return True
    except Exception:
        return False


def fit_vtk_camera(
    plotter,
    *,
    azimuth: float = -58.0,
    elevation: float = 28.0,
    zoom: float = 0.78,
) -> None:
    """Orbit so all three axes read, then zoom out so the box fits the pane."""
    try:
        plotter.enable_parallel_projection()
    except Exception:
        pass
    try:
        plotter.reset_camera()
    except Exception:
        pass
    camera = getattr(plotter, "camera", None)
    if camera is None:
        try:
            plotter.view_isometric()
        except Exception:
            pass
        return
    for attr, value in (("azimuth", azimuth), ("elevation", elevation)):
        try:
            setattr(camera, attr, value)
        except Exception:
            pass
    try:
        plotter.reset_camera_clipping_range()
    except Exception:
        pass
    try:
        camera.zoom(zoom)
    except Exception:
        pass


def set_cubic_box_aspect(axis, *, zoom: float = 0.78) -> None:
    try:
        axis.set_box_aspect((1.0, 1.0, 1.0), zoom=zoom)
    except TypeError:
        try:
            axis.set_box_aspect((1.0, 1.0, 1.0))
        except Exception:
            pass
