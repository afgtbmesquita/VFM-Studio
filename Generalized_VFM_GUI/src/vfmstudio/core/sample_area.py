"""Estimate VFM ``sample_area_total`` from a DIC point cloud.

Identification weights every measurement by ``A / n``.  For a structured
facet lattice that means ``A = n · dx · dy`` -- including tabbed plates,
L / C outlines, and interior holes -- because missing points are simply
absent from ``n``.  A bounding-box default is wrong as soon as the cloud
does not fill its hull.

Libraries considered (see tests/test_sample_area.py):

* ``shapely.concave_hull(..., allow_holes=True)`` -- fast, but ``ratio``
  is unstable: small values Swiss-cheese a filled rectangle, and the
  returned area is the hull of the *coordinates*, which omits the half-
  facet strip around the boundary that VFM still integrates.
* ``alpha_shapes.Alpha_Shaper.optimize()`` -- reconstructs holes more
  faithfully at a pitch-based alpha, but the optimizer (largest alpha that
  still covers every vertex) can punch spurious holes, and the polygon
  area is again the interior of the triangulation, not ``n · dx · dy``.
* Occupancy ``n · h²`` -- exact on a square lattice, 2× wrong when
  ``dx ≠ dy`` because nearest-neighbour ``h`` is ``min(dx, dy)``.

The estimator used here is Delaunay-based and matches ``n · dx · dy`` on
those cases: keep the same void-triangle filter the viewport uses, take
twice the median kept-triangle area as the facet cell, and multiply by
``n``.  Two Delaunay triangles fill one rectangular cell, so

    2 · median(triangle area) = dx · dy

even after in-plane rotation and with holes / concavities present (those
only change ``n``).  Shapely is not required at runtime.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from vfmstudio.core.contour_mesh import (
    filtered_triangles,
    finite_xy,
    median_nearest_neighbour,
    occupancy_sample_area,
)


def triangle_areas(x, y, triangles: np.ndarray) -> np.ndarray:
    """Signed-area magnitude of each triangle, ``(n_tri,)``."""
    xs = np.asarray(x, dtype=float).ravel()
    ys = np.asarray(y, dtype=float).ravel()
    tris = np.asarray(triangles, dtype=np.int64)
    if tris.ndim != 2 or tris.shape[0] == 0 or tris.shape[1] < 3:
        return np.zeros(0, dtype=float)
    x0 = xs[tris[:, 0]]
    y0 = ys[tris[:, 0]]
    x1 = xs[tris[:, 1]]
    y1 = ys[tris[:, 1]]
    x2 = xs[tris[:, 2]]
    y2 = ys[tris[:, 2]]
    return 0.5 * np.abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))


def characteristic_cell_area(x, y) -> float:
    """Typical DIC facet area ``dx · dy`` from the filtered Delaunay mesh."""
    xs, ys, _ = finite_xy(x, y)
    if xs.size < 3:
        return 0.0
    faces = filtered_triangles(xs, ys)
    areas = triangle_areas(xs, ys, faces)
    areas = areas[np.isfinite(areas) & (areas > 0.0)]
    if areas.size == 0:
        return 0.0
    cell = 2.0 * float(np.median(areas))
    return cell if np.isfinite(cell) and cell > 0.0 else 0.0


def estimate_sample_area(x, y) -> Optional[float]:
    """Return ``n · dx · dy`` for a DIC cloud, or ``None`` if undefined.

    Always attempts a Delaunay estimate, including filled rectangles.  Falls
    back to occupancy ``n · h²`` and then to ``None`` (the loader's bbox)
    only when the triangulation does not yield a positive cell area.
    """
    xs, ys, _ = finite_xy(x, y)
    n = int(xs.size)
    if n < 1:
        return None
    cell = characteristic_cell_area(xs, ys)
    if cell > 0.0:
        area = float(n) * cell
        if np.isfinite(area) and area > 0.0:
            return area
    occupied = occupancy_sample_area(xs, ys, fill_threshold=1.1)
    if occupied is not None:
        return float(occupied)
    if n < 2:
        return None
    h = median_nearest_neighbour(np.column_stack([xs, ys]))
    if h <= 0.0:
        return None
    area = float(n) * h * h
    return area if np.isfinite(area) and area > 0.0 else None
