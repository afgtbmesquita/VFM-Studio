"""Tessellate a DIC cloud without painting across physical voids.

A raw Delaunay triangulation of a tabbed / cruciform / L-shaped point cloud
fills the convex hull, so contour plots show long streaks through regions
where the specimen does not exist.  Identification never uses that
triangulation -- it sums only at measurement points and drops empty virtual
elements -- but the viewport does, and unconstrained ``delaunay_2d`` is what
makes a plus-shaped plate look like a filled square.

The two filters below are the same length-scale tests the plate BC GUI uses
(``bc_gui.mesh_canvas._filtered_triangulation``).  Both scale with the
median nearest-neighbour pitch ``h``, so they do not need per-dataset tuning.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

EDGE_FACTOR = 2.0
NN_RADIUS_FACTOR = 1.6
MIN_COMMON = 2


def finite_xy(x, y) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return finite ``(x, y)`` and the integer index map into the originals."""
    xs = np.asarray(x, dtype=float).ravel()
    ys = np.asarray(y, dtype=float).ravel()
    n = min(xs.size, ys.size)
    xs = xs[:n]
    ys = ys[:n]
    ok = np.isfinite(xs) & np.isfinite(ys)
    index = np.flatnonzero(ok)
    return xs[ok], ys[ok], index


def occupancy_sample_area(
    x,
    y,
    *,
    fill_threshold: float = 0.9,
) -> Optional[float]:
    """Return ``n · h²`` when the cloud does not fill its bounding box.

    Identification weights every DIC point by ``sample_area_total / n``.  The
    loader's default is the axis-aligned bounding box, which is correct for a
    filled rectangle and ~1.3× too large for a plus-shaped plate (the empty
    corners between the tabs are not material).  When the raster occupancy is
    below ``fill_threshold`` of that box, this returns the facet area
    ``n · h · h`` instead.  A filled rectangle yields ``None`` so existing
    coupon projects keep the bbox default.

    Prefer :func:`vfmstudio.core.sample_area.estimate_sample_area` for new
    GUI prefill: it uses a filtered Delaunay cell area and is exact when
    ``dx ≠ dy``.  This helper remains as a cheap occupancy fallback.
    """
    xs, ys, _ = finite_xy(x, y)
    n = int(xs.size)
    if n < 2:
        return None
    span_x = float(np.max(xs) - np.min(xs))
    span_y = float(np.max(ys) - np.min(ys))
    bbox = span_x * span_y
    if not np.isfinite(bbox) or bbox <= 0.0:
        return None
    h = median_nearest_neighbour(np.column_stack([xs, ys]))
    if h <= 0.0:
        return None
    occupied = float(n) * h * h
    if not np.isfinite(occupied) or occupied <= 0.0:
        return None
    if occupied >= float(fill_threshold) * bbox:
        return None
    return occupied


def median_nearest_neighbour(xy: np.ndarray) -> float:
    """Median nearest-neighbour distance of a 2-D cloud, or 0 if undefined."""
    points = np.asarray(xy, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
        return 0.0
    from scipy.spatial import cKDTree

    tree = cKDTree(points[:, :2])
    distances, _ = tree.query(points[:, :2], k=2)
    finite = np.asarray(distances[:, 1], dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    h = float(np.median(finite))
    return h if np.isfinite(h) and h > 0.0 else 0.0


def _delaunay_simplices(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    if xs.size < 3:
        return np.zeros((0, 3), dtype=np.int64)
    try:
        from matplotlib.tri import Triangulation

        triang = Triangulation(xs, ys)
        return np.asarray(triang.triangles, dtype=np.int64)
    except Exception:
        try:
            from scipy.spatial import Delaunay, QhullError
        except ImportError:
            return np.zeros((0, 3), dtype=np.int64)
        try:
            return np.asarray(Delaunay(np.column_stack([xs, ys])).simplices, dtype=np.int64)
        except QhullError:
            return np.zeros((0, 3), dtype=np.int64)


def _is_scattered_cloud(xs: np.ndarray, ys: np.ndarray) -> bool:
    """True when unique x and y look like a rotated / unstructured cloud.

    The common-neighbour test is tuned for an axis-aligned raster.  After a
    small in-plane rotation every coordinate is unique and that test would
    fragment a valid interior.  The max-edge filter still applies.
    """
    n = int(xs.size)
    if n < 4:
        return False
    grid_axis_scale = max(4.0 * np.sqrt(n), 16.0)
    return int(np.unique(xs).size) > grid_axis_scale and int(np.unique(ys).size) > grid_axis_scale


def void_triangle_mask(
    xs: np.ndarray,
    ys: np.ndarray,
    triangles: np.ndarray,
    *,
    edge_factor: float = EDGE_FACTOR,
    radius_factor: float = NN_RADIUS_FACTOR,
    min_common: int = MIN_COMMON,
) -> np.ndarray:
    """``True`` for Delaunay triangles that span a hole in the point cloud."""
    tris = np.asarray(triangles, dtype=np.int64)
    n_tri = int(tris.shape[0]) if tris.ndim == 2 and tris.shape[1] >= 3 else 0
    if n_tri == 0:
        return np.zeros(0, dtype=bool)

    xs = np.asarray(xs, dtype=float).ravel()
    ys = np.asarray(ys, dtype=float).ravel()
    points = np.column_stack([xs, ys])
    if points.shape[0] < 4:
        return np.zeros(n_tri, dtype=bool)

    h = median_nearest_neighbour(points)
    if h <= 0.0:
        return np.zeros(n_tri, dtype=bool)

    p0 = tris[:, 0]
    p1 = tris[:, 1]
    p2 = tris[:, 2]
    e01 = np.hypot(xs[p1] - xs[p0], ys[p1] - ys[p0])
    e12 = np.hypot(xs[p2] - xs[p1], ys[p2] - ys[p1])
    e20 = np.hypot(xs[p0] - xs[p2], ys[p0] - ys[p2])
    max_edge = np.maximum(np.maximum(e01, e12), e20)
    edge_mask = max_edge > float(edge_factor) * h

    if _is_scattered_cloud(xs, ys):
        return edge_mask

    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    radius = float(radius_factor) * h
    neighbour_lists = tree.query_ball_point(points, r=radius)
    neighbour_sets = [set(nb) - {i} for i, nb in enumerate(neighbour_lists)]

    corner_mask = np.zeros(n_tri, dtype=bool)
    need = int(min_common)
    for i in range(n_tri):
        if edge_mask[i]:
            continue
        a = int(p0[i])
        b = int(p1[i])
        c = int(p2[i])
        na = neighbour_sets[a]
        nb = neighbour_sets[b]
        nc = neighbour_sets[c]
        if (
            len(na & nb) < need
            or len(nb & nc) < need
            or len(nc & na) < need
        ):
            corner_mask[i] = True
    return edge_mask | corner_mask


def filtered_triangles(
    x,
    y,
    *,
    edge_factor: float = EDGE_FACTOR,
    radius_factor: float = NN_RADIUS_FACTOR,
    min_common: int = MIN_COMMON,
) -> np.ndarray:
    """Vertex indices ``(n_tri, 3)`` of Delaunay triangles that stay on the specimen.

    Indices refer to the original ``x`` / ``y`` arrays (NaN coordinates are
    simply never used as vertices).  If the filters would hide every triangle
    the unmasked Delaunay is returned -- better a filled hull on a sparse
    cloud than a blank viewport.
    """
    xs, ys, index = finite_xy(x, y)
    local = _delaunay_simplices(xs, ys)
    if local.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.int64)
    mask = void_triangle_mask(
        xs,
        ys,
        local,
        edge_factor=edge_factor,
        radius_factor=radius_factor,
        min_common=min_common,
    )
    if mask.size == 0:
        kept = local
    elif bool(np.all(mask)):
        kept = local
    else:
        kept = local[~mask]
    return index[kept].astype(np.int64, copy=False)


def triangle_centroids(x, y, triangles: np.ndarray) -> np.ndarray:
    """``(n_tri, 2)`` centroids of ``triangles`` in the ``x`` / ``y`` cloud."""
    xs = np.asarray(x, dtype=float).ravel()
    ys = np.asarray(y, dtype=float).ravel()
    tris = np.asarray(triangles, dtype=np.int64)
    if tris.ndim != 2 or tris.shape[0] == 0:
        return np.zeros((0, 2), dtype=float)
    px = xs[tris]
    py = ys[tris]
    return np.column_stack([px.mean(axis=1), py.mean(axis=1)])


def vtk_triangle_cells(triangles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Connectivity + VTK triangle cell-type array."""
    tris = np.asarray(triangles, dtype=np.int64)
    n = int(tris.shape[0]) if tris.ndim == 2 else 0
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.uint8)
    cells = np.hstack([np.full((n, 1), 3, dtype=np.int64), tris]).ravel()
    celltypes = np.full(n, 5, dtype=np.uint8)  # VTK_TRIANGLE
    return cells, celltypes


def field_surface(pv, xyz: np.ndarray, scalars: np.ndarray, scalar_name: str):
    """PyVista surface whose faces follow the specimen, not its convex hull.

    Unused vertices (NaN coordinates, or points only connected by void
    triangles) stay in the point array so a later warp can replace
    coordinates by the original index.
    """
    xyz = np.asarray(xyz, dtype=float)
    scalars = np.asarray(scalars, dtype=float).ravel()
    n = min(int(xyz.shape[0]), int(scalars.size))
    if n == 0:
        cloud = pv.PolyData(np.zeros((0, 3), dtype=float))
        cloud[scalar_name] = np.zeros(0, dtype=float)
        return cloud
    if xyz.ndim != 2 or xyz.shape[1] < 2:
        raise ValueError("field_surface expects an (n, 2) or (n, 3) point array.")
    if xyz.shape[1] == 2:
        xyz = np.column_stack([xyz[:n, 0], xyz[:n, 1], np.zeros(n, dtype=float)])
    else:
        xyz = np.asarray(xyz[:n, :3], dtype=float).copy()
    scalars = scalars[:n]
    faces = filtered_triangles(xyz[:, 0], xyz[:, 1])
    if faces.shape[0] == 0:
        cloud = pv.PolyData(xyz)
        cloud[scalar_name] = scalars
        return cloud
    cells, celltypes = vtk_triangle_cells(faces)
    try:
        triangle = getattr(getattr(pv, "CellType", None), "TRIANGLE", 5)
        celltypes = np.full(faces.shape[0], triangle, dtype=np.uint8)
        surf = pv.UnstructuredGrid(cells, celltypes, xyz)
    except Exception:
        try:
            surf = pv.PolyData(xyz, cells)
        except TypeError:
            surf = pv.PolyData(xyz, faces=cells)
    surf[scalar_name] = scalars
    return surf
