"""Axis-aligned free-quad construction from user-placed nodes.

Clicked points are clustered onto shared vertical and horizontal lines.
A unit cell becomes a quadrilateral when all four corners exist.  L-shaped
and notched domains are allowed; hanging nodes and leftover points are not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

CONFORM_MESSAGE = (
    "The mesh is not conform and it is not possible to construct the mesh."
)

RELATIVE_CLUSTER_TOLERANCE = 1.0e-3
LABEL_OFFSET_PX = 12.0
LABEL_MARGIN_PX = 16.0


class ConformError(ValueError):
    """Raised when placed nodes cannot form a conformal quad mesh."""

    def __init__(self, message: str = CONFORM_MESSAGE) -> None:
        super().__init__(message)


@dataclass
class FreeMeshStatus:
    ok: bool
    message: str
    n_points: int = 0
    n_quads: int = 0
    x_nodes: List[float] = field(default_factory=list)
    y_nodes: List[float] = field(default_factory=list)
    snapped_points: List[Tuple[float, float]] = field(default_factory=list)
    keep_cells: List[Tuple[int, int]] = field(default_factory=list)
    quad_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 4, 2)))

    @property
    def n_elements_x(self) -> int:
        return max(len(self.x_nodes) - 1, 0)

    @property
    def n_elements_y(self) -> int:
        return max(len(self.y_nodes) - 1, 0)

    @property
    def hint(self) -> str:
        if self.n_points == 0:
            return "Click in the viewport to add nodes."
        if self.ok:
            return f"{self.n_points} nodes · {self.n_quads} quads"
        return (
            f"{self.n_points} nodes · {self.n_quads} quads — "
            "missing corners / not a conformal quad mesh"
        )


def _as_points(points: Iterable) -> np.ndarray:
    arr = np.asarray(list(points), dtype=float)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("Free-mesh points must be an (N, 2) array of x/y pairs.")
    return arr[:, :2].copy()


def cluster_axis(values: Sequence[float], tolerance: float) -> np.ndarray:
    """Merge 1-D coordinates whose consecutive gap is within ``tolerance``."""
    arr = np.asarray(list(values), dtype=float).ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(0, dtype=float)
    order = np.sort(finite)
    clusters: List[List[float]] = [[float(order[0])]]
    for value in order[1:]:
        current = clusters[-1]
        if float(value) - current[-1] <= tolerance:
            current.append(float(value))
        else:
            clusters.append([float(value)])
    return np.asarray([float(np.mean(group)) for group in clusters], dtype=float)


def clustering_tolerance(points: np.ndarray) -> float:
    if points.shape[0] == 0:
        return 1.0e-12
    span_x = float(np.ptp(points[:, 0]))
    span_y = float(np.ptp(points[:, 1]))
    diag = float(np.hypot(span_x, span_y))
    return max(1.0e-12, RELATIVE_CLUSTER_TOLERANCE * max(diag, 1.0e-9))


def _nearest_index(axis: np.ndarray, value: float) -> int:
    if axis.size == 0:
        return 0
    pos = int(np.searchsorted(axis, value, side="left"))
    if pos <= 0:
        return 0
    if pos >= axis.size:
        return int(axis.size - 1)
    left = pos - 1
    if abs(value - axis[left]) <= abs(value - axis[pos]):
        return left
    return pos


def inspect_free_points(points: Iterable) -> FreeMeshStatus:
    """Cluster points and report whether they form a conformal quad mesh."""
    try:
        arr = _as_points(points)
    except ValueError:
        return FreeMeshStatus(ok=False, message=CONFORM_MESSAGE)
    n_points = int(arr.shape[0])
    if n_points == 0:
        return FreeMeshStatus(
            ok=False,
            message="Click in the viewport to add nodes.",
            n_points=0,
        )
    if not np.all(np.isfinite(arr)):
        return FreeMeshStatus(
            ok=False, message=CONFORM_MESSAGE, n_points=n_points
        )

    tol = clustering_tolerance(arr)
    xs = cluster_axis(arr[:, 0], tol)
    ys = cluster_axis(arr[:, 1], tol)
    if xs.size < 2 or ys.size < 2:
        snapped = [
            (float(xs[_nearest_index(xs, x)]), float(ys[_nearest_index(ys, y)]))
            for x, y in arr
        ]
        return FreeMeshStatus(
            ok=False,
            message=CONFORM_MESSAGE,
            n_points=n_points,
            x_nodes=[float(v) for v in xs],
            y_nodes=[float(v) for v in ys],
            snapped_points=snapped,
        )

    occupancy = -np.ones((xs.size, ys.size), dtype=np.int64)
    snapped: List[Tuple[float, float]] = []
    for index, (x, y) in enumerate(arr):
        ix = _nearest_index(xs, float(x))
        iy = _nearest_index(ys, float(y))
        if occupancy[ix, iy] >= 0:
            return FreeMeshStatus(
                ok=False,
                message=CONFORM_MESSAGE,
                n_points=n_points,
                x_nodes=[float(v) for v in xs],
                y_nodes=[float(v) for v in ys],
            )
        occupancy[ix, iy] = index
        snapped.append((float(xs[ix]), float(ys[iy])))

    keep_cells: List[Tuple[int, int]] = []
    used = np.zeros_like(occupancy, dtype=bool)
    quads: List[np.ndarray] = []
    nx = int(xs.size - 1)
    ny = int(ys.size - 1)
    for iy in range(ny):
        for ix in range(nx):
            corners = (
                occupancy[ix, iy],
                occupancy[ix + 1, iy],
                occupancy[ix + 1, iy + 1],
                occupancy[ix, iy + 1],
            )
            if any(slot < 0 for slot in corners):
                continue
            keep_cells.append((ix, iy))
            used[ix, iy] = True
            used[ix + 1, iy] = True
            used[ix + 1, iy + 1] = True
            used[ix, iy + 1] = True
            quads.append(
                np.array(
                    [
                        [xs[ix], ys[iy]],
                        [xs[ix + 1], ys[iy]],
                        [xs[ix + 1], ys[iy + 1]],
                        [xs[ix], ys[iy + 1]],
                    ],
                    dtype=float,
                )
            )

    occupied = occupancy >= 0
    leftover = bool(np.any(occupied & ~used))
    ok = bool(keep_cells) and not leftover
    quad_xy = (
        np.stack(quads, axis=0)
        if quads
        else np.zeros((0, 4, 2), dtype=float)
    )
    return FreeMeshStatus(
        ok=ok,
        message="" if ok else CONFORM_MESSAGE,
        n_points=n_points,
        n_quads=len(keep_cells),
        x_nodes=[float(v) for v in xs],
        y_nodes=[float(v) for v in ys],
        snapped_points=snapped,
        keep_cells=keep_cells,
        quad_xy=quad_xy,
    )


def construct_free_quad_mesh(points: Iterable) -> FreeMeshStatus:
    """Return a conformal free mesh or raise ``ConformError``."""
    status = inspect_free_points(points)
    if not status.ok:
        raise ConformError(CONFORM_MESSAGE)
    return status


def offset_label_display(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    offset: float = LABEL_OFFSET_PX,
    margin: float = LABEL_MARGIN_PX,
    prefer_right: bool = True,
    prefer_above: bool = True,
) -> Tuple[float, float]:
    """Place a 2-D label beside ``(x, y)`` in display coordinates.

    ``(0, 0)`` is the lower-left corner (VTK display space).  The label is
    shifted off the point, then flipped inward when it would hit a margin.
    """
    width = max(float(width), 1.0)
    height = max(float(height), 1.0)
    margin = min(float(margin), 0.45 * width, 0.45 * height)
    offset = float(offset)
    dx = offset if prefer_right else -offset
    dy = offset if prefer_above else -offset
    lx = float(x) + dx
    ly = float(y) + dy
    if lx > width - margin:
        lx = float(x) - offset
    if lx < margin:
        lx = float(x) + offset
    if ly > height - margin:
        ly = float(y) - offset
    if ly < margin:
        ly = float(y) + offset
    lx = min(max(lx, margin), width - margin)
    ly = min(max(ly, margin), height - margin)
    if abs(lx - x) < 1.0 and abs(ly - y) < 1.0:
        lx = min(max(float(x) + offset, margin), width - margin)
        ly = min(max(float(y) + offset, margin), height - margin)
    return lx, ly


def snap_coordinate(
    value: float,
    axis: Sequence[float],
    *,
    display_value: float,
    axis_display: Sequence[float],
    pixel_tol: float = 10.0,
) -> float:
    """Snap ``value`` to the nearest axis line within ``pixel_tol`` pixels."""
    if not axis:
        return float(value)
    best = float(value)
    best_dist = float(pixel_tol)
    for world, disp in zip(axis, axis_display):
        dist = abs(float(disp) - float(display_value))
        if dist < best_dist:
            best_dist = dist
            best = float(world)
    return best


def nearest_point_index(
    display_xy: Sequence[float],
    point_display: Sequence[Sequence[float]],
    *,
    pixel_tol: float = 10.0,
) -> Optional[int]:
    """Return the index of the node under the cursor, if any."""
    if not point_display:
        return None
    click = np.asarray(display_xy, dtype=float)
    pts = np.asarray(point_display, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return None
    delta = pts[:, :2] - click[:2]
    dist = np.sqrt(np.einsum("ij,ij->i", delta, delta))
    index = int(np.argmin(dist))
    if dist[index] <= float(pixel_tol):
        return index
    return None
