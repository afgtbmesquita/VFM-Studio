"""Drag structured-mesh grid lines without breaking axis-aligned quads.

The solver mesh is a Cartesian product of ``x_nodes`` and ``y_nodes``.
Moving one node independently is not allowed.  Clicking a node or an
element edge selects the whole grid line of connecting element edges;
dragging it updates that one coordinate in ``x_nodes`` or ``y_nodes``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

PICK_PIXEL_TOL = 12.0
AXIS_LOCK_PIXELS = 8.0
SNAP_PIXEL_TOL = 10.0
MIN_RELATIVE_GAP = 1.0e-4
MIN_ABSOLUTE_GAP = 1.0e-9

DRAG_HINT = (
    "Click a mesh node or element edge, then drag. "
    "Connected nodes on that grid line move together. "
    "Outer edges can be pushed to the specimen."
)


@dataclass(frozen=True)
class DragTarget:
    """What a click selected in the mesh."""

    node_id: Optional[int]
    ix: int
    iy: int
    axis: Optional[str]
    """``'x'`` / ``'y'`` if an edge was clicked; ``None`` until drag direction."""


def axis_nodes_from_mesh(mesh) -> Tuple[List[float], List[float]]:
    """Return ``(x_nodes, y_nodes)``, materializing a uniform grid if needed."""
    nx = max(int(getattr(mesh, "n_elements_x", 1) or 1), 1)
    ny = max(int(getattr(mesh, "n_elements_y", 1) or 1), 1)
    bounds = getattr(mesh, "bounds", None)
    x_min = float(getattr(bounds, "x_min", 0.0) if bounds is not None else 0.0)
    x_max = float(getattr(bounds, "x_max", 1.0) if bounds is not None else 1.0)
    y_min = float(getattr(bounds, "y_min", 0.0) if bounds is not None else 0.0)
    y_max = float(getattr(bounds, "y_max", 1.0) if bounds is not None else 1.0)
    xs = _finite_axis(getattr(mesh, "x_nodes", None), x_min, x_max, nx)
    ys = _finite_axis(getattr(mesh, "y_nodes", None), y_min, y_max, ny)
    return xs, ys


def materialize_axis_nodes(mesh) -> bool:
    """Write explicit ``x_nodes`` / ``y_nodes`` and matching bounds.

    Returns True if the model changed.
    """
    xs, ys = axis_nodes_from_mesh(mesh)
    changed = _axis_changed(mesh.x_nodes, xs) or _axis_changed(mesh.y_nodes, ys)
    mesh.x_nodes = list(xs)
    mesh.y_nodes = list(ys)
    mesh.n_elements_x = max(len(xs) - 1, 1)
    mesh.n_elements_y = max(len(ys) - 1, 1)
    _sync_bounds(mesh, xs, ys)
    return changed


def clamp_axis_value(nodes: Sequence[float], index: int, value: float) -> float:
    """Keep grid coordinates strictly increasing with a small interior gap."""
    arr = [float(v) for v in nodes]
    n = len(arr)
    if n < 2 or index < 0 or index >= n:
        return float(value)
    span = abs(arr[-1] - arr[0]) if n >= 2 else 1.0
    gap = max(MIN_ABSOLUTE_GAP, MIN_RELATIVE_GAP * max(span, 1.0e-12))
    lo = float("-inf")
    hi = float("inf")
    if index > 0:
        lo = arr[index - 1] + gap
    if index < n - 1:
        hi = arr[index + 1] - gap
    if lo > hi:
        return arr[index]
    return float(min(max(float(value), lo), hi))


def apply_grid_line_drag(mesh, axis: str, index: int, value: float) -> float:
    """Move one grid line and keep bounds / free-mesh nodes in sync."""
    materialize_axis_nodes(mesh)
    key = str(axis).strip().lower()
    if key not in {"x", "y"}:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    nodes = list(mesh.x_nodes if key == "x" else mesh.y_nodes)
    if index < 0 or index >= len(nodes):
        raise IndexError(f"grid-line index {index} is out of range for {key}_nodes")
    old = float(nodes[index])
    new = clamp_axis_value(nodes, index, value)
    nodes[index] = new
    if key == "x":
        mesh.x_nodes = nodes
        _shift_free_nodes(mesh, 0, old, new)
    else:
        mesh.y_nodes = nodes
        _shift_free_nodes(mesh, 1, old, new)
    _sync_bounds(mesh, mesh.x_nodes, mesh.y_nodes)
    return new


def restore_mesh_axes(
    mesh,
    x_nodes: Optional[Sequence[float]],
    y_nodes: Optional[Sequence[float]],
    bounds: Optional[dict] = None,
    free_nodes: Optional[Sequence] = None,
    *,
    n_elements_x: Optional[int] = None,
    n_elements_y: Optional[int] = None,
) -> None:
    """Put mesh axes back to a snapshot (cancel an in-progress drag)."""
    mesh.x_nodes = None if x_nodes is None else [float(v) for v in x_nodes]
    mesh.y_nodes = None if y_nodes is None else [float(v) for v in y_nodes]
    if n_elements_x is not None:
        mesh.n_elements_x = max(int(n_elements_x), 1)
    if n_elements_y is not None:
        mesh.n_elements_y = max(int(n_elements_y), 1)
    if bounds is not None:
        mesh.bounds.x_min = float(bounds["x_min"])
        mesh.bounds.x_max = float(bounds["x_max"])
        mesh.bounds.y_min = float(bounds["y_min"])
        mesh.bounds.y_max = float(bounds["y_max"])
    elif mesh.x_nodes and mesh.y_nodes:
        _sync_bounds(mesh, mesh.x_nodes, mesh.y_nodes)
    if free_nodes is None:
        mesh.free_nodes = None
    else:
        mesh.free_nodes = [[float(p[0]), float(p[1])] for p in free_nodes]


def snapshot_mesh_axes(mesh) -> dict:
    return {
        "x_nodes": None if mesh.x_nodes is None else [float(v) for v in mesh.x_nodes],
        "y_nodes": None if mesh.y_nodes is None else [float(v) for v in mesh.y_nodes],
        "n_elements_x": int(mesh.n_elements_x),
        "n_elements_y": int(mesh.n_elements_y),
        "bounds": {
            "x_min": float(mesh.bounds.x_min),
            "x_max": float(mesh.bounds.x_max),
            "y_min": float(mesh.bounds.y_min),
            "y_max": float(mesh.bounds.y_max),
        },
        "free_nodes": None
        if mesh.free_nodes is None
        else [[float(p[0]), float(p[1])] for p in mesh.free_nodes],
    }


def node_grid_slot(x: float, y: float, xs: Sequence[float], ys: Sequence[float]) -> Tuple[int, int]:
    return _nearest_axis_index(xs, x), _nearest_axis_index(ys, y)


def nodes_on_grid_line(
    xyz: np.ndarray,
    xs: Sequence[float],
    ys: Sequence[float],
    axis: str,
    index: int,
) -> np.ndarray:
    """Node ids whose coordinates lie on the given grid line."""
    coords = np.asarray(xyz, dtype=float)
    if coords.ndim != 2 or coords.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    xs_arr = np.asarray(list(xs), dtype=float)
    ys_arr = np.asarray(list(ys), dtype=float)
    key = str(axis).strip().lower()
    if key == "x":
        if index < 0 or index >= xs_arr.size:
            return np.zeros(0, dtype=np.int64)
        target = float(xs_arr[index])
        tol = _line_tolerance(xs_arr, ys_arr)
        return np.flatnonzero(np.abs(coords[:, 0] - target) <= tol)
    if index < 0 or index >= ys_arr.size:
        return np.zeros(0, dtype=np.int64)
    target = float(ys_arr[index])
    tol = _line_tolerance(xs_arr, ys_arr)
    return np.flatnonzero(np.abs(coords[:, 1] - target) <= tol)


def grid_line_segments(
    xyz: np.ndarray,
    quads: np.ndarray,
    xs: Sequence[float],
    ys: Sequence[float],
    axis: str,
    index: int,
) -> np.ndarray:
    """Element-edge segments that lie on a grid line. Shape ``(n, 2, 2)``."""
    coords = np.asarray(xyz, dtype=float)
    cells = np.asarray(quads, dtype=np.int64)
    if coords.ndim != 2 or cells.ndim != 2 or cells.shape[0] == 0:
        return np.zeros((0, 2, 2), dtype=float)
    xs_arr = np.asarray(list(xs), dtype=float)
    ys_arr = np.asarray(list(ys), dtype=float)
    key = str(axis).strip().lower()
    coord = 0 if key == "x" else 1
    axis_vals = xs_arr if key == "x" else ys_arr
    if index < 0 or index >= axis_vals.size:
        return np.zeros((0, 2, 2), dtype=float)
    target = float(axis_vals[index])
    tol = _line_tolerance(xs_arr, ys_arr)
    segs: List[np.ndarray] = []
    n_corner = int(cells.shape[1])
    for cell in cells:
        for i in range(n_corner):
            a = coords[int(cell[i])]
            b = coords[int(cell[(i + 1) % n_corner])]
            if abs(float(a[coord]) - target) <= tol and abs(float(b[coord]) - target) <= tol:
                segs.append(np.array([[a[0], a[1]], [b[0], b[1]]], dtype=float))
    if not segs:
        return np.zeros((0, 2, 2), dtype=float)
    return np.stack(segs, axis=0)


def choose_drag_axis(
    ix: int,
    iy: int,
    n_x_nodes: int,
    n_y_nodes: int,
    display_dx: float,
    display_dy: float,
    *,
    preferred: Optional[str] = None,
    lock_pixels: float = AXIS_LOCK_PIXELS,
) -> Optional[str]:
    """Lock to x or y from pointer motion. ``None`` until the dead-zone is left."""
    if preferred in {"x", "y"}:
        return preferred
    dx = float(display_dx)
    dy = float(display_dy)
    if hypot(dx, dy) < float(lock_pixels):
        return None
    on_vert = ix <= 0 or ix >= max(int(n_x_nodes) - 1, 0)
    on_horz = iy <= 0 or iy >= max(int(n_y_nodes) - 1, 0)
    adx, ady = abs(dx), abs(dy)
    if on_vert and not on_horz:
        return "x" if adx >= 0.35 * ady else "y"
    if on_horz and not on_vert:
        return "y" if ady >= 0.35 * adx else "x"
    return "x" if adx >= ady else "y"


def snap_axis_value(
    value: float,
    display_value: float,
    targets: Sequence[float],
    target_display: Sequence[float],
    *,
    pixel_tol: float = SNAP_PIXEL_TOL,
) -> float:
    """Snap a dragged coordinate to nearby specimen (or other) lines."""
    best = float(value)
    best_dist = float(pixel_tol)
    for world, disp in zip(targets, target_display):
        dist = abs(float(disp) - float(display_value))
        if dist < best_dist:
            best_dist = dist
            best = float(world)
    return best


def pick_drag_target(
    click_display: Sequence[float],
    node_display: np.ndarray,
    node_xyz: np.ndarray,
    edge_display: np.ndarray,
    edge_axes: Sequence[Tuple[str, int]],
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    pixel_tol: float = PICK_PIXEL_TOL,
) -> Optional[DragTarget]:
    """Pick a node (preferred) or an element edge in display pixels."""
    click = np.asarray(click_display, dtype=float).ravel()
    if click.size < 2:
        return None
    node_hit = _nearest_node(click, node_display, pixel_tol)
    if node_hit is not None:
        xyz = np.asarray(node_xyz, dtype=float)
        x, y = float(xyz[node_hit, 0]), float(xyz[node_hit, 1])
        ix, iy = node_grid_slot(x, y, xs, ys)
        return DragTarget(node_id=int(node_hit), ix=ix, iy=iy, axis=None)
    edge_hit = _nearest_edge(click, edge_display, pixel_tol)
    if edge_hit is None or edge_hit >= len(edge_axes):
        return None
    axis, index = edge_axes[edge_hit]
    ix, iy = _slot_from_edge(axis, index, xs, ys, click, node_display, node_xyz)
    return DragTarget(node_id=None, ix=ix, iy=iy, axis=str(axis))


def element_edge_records(
    xyz: np.ndarray,
    quads: np.ndarray,
    xs: Sequence[float],
    ys: Sequence[float],
) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
    """World-space element edges and the grid line each belongs to."""
    coords = np.asarray(xyz, dtype=float)
    cells = np.asarray(quads, dtype=np.int64)
    if coords.ndim != 2 or cells.ndim != 2 or cells.shape[0] == 0:
        return np.zeros((0, 2, 2), dtype=float), []
    xs_arr = np.asarray(list(xs), dtype=float)
    ys_arr = np.asarray(list(ys), dtype=float)
    tol = _line_tolerance(xs_arr, ys_arr)
    segs: List[np.ndarray] = []
    axes: List[Tuple[str, int]] = []
    n_corner = int(cells.shape[1])
    for cell in cells:
        for i in range(n_corner):
            a = coords[int(cell[i])]
            b = coords[int(cell[(i + 1) % n_corner])]
            rec = _classify_edge(a, b, xs_arr, ys_arr, tol)
            if rec is None:
                continue
            segs.append(np.array([[a[0], a[1]], [b[0], b[1]]], dtype=float))
            axes.append(rec)
    if not segs:
        return np.zeros((0, 2, 2), dtype=float), []
    return np.stack(segs, axis=0), axes


def dic_axis_targets(dic_xyz: Optional[np.ndarray], axis: str) -> List[float]:
    if dic_xyz is None:
        return []
    cloud = np.asarray(dic_xyz, dtype=float)
    if cloud.ndim != 2 or cloud.shape[0] == 0 or cloud.shape[1] < 2:
        return []
    finite = np.isfinite(cloud[:, 0]) & np.isfinite(cloud[:, 1])
    if not np.any(finite):
        return []
    col = 0 if str(axis).strip().lower() == "x" else 1
    lo = float(np.min(cloud[finite, col]))
    hi = float(np.max(cloud[finite, col]))
    if hi <= lo:
        return [lo]
    return [lo, hi]


def hypot(dx: float, dy: float) -> float:
    return float(np.hypot(float(dx), float(dy)))


def _finite_axis(values, lo: float, hi: float, n_el: int) -> List[float]:
    if values:
        arr = [float(v) for v in values if np.isfinite(v)]
        if len(arr) >= 2:
            return arr
    if hi <= lo:
        hi = lo + 1.0
    return [float(v) for v in np.linspace(lo, hi, int(n_el) + 1)]


def _axis_changed(current, new_vals: Sequence[float]) -> bool:
    if current is None:
        return True
    if len(current) != len(new_vals):
        return True
    return any(abs(float(a) - float(b)) > 1.0e-12 for a, b in zip(current, new_vals))


def _sync_bounds(mesh, xs: Sequence[float], ys: Sequence[float]) -> None:
    mesh.bounds.x_min = float(xs[0])
    mesh.bounds.x_max = float(xs[-1])
    mesh.bounds.y_min = float(ys[0])
    mesh.bounds.y_max = float(ys[-1])


def _shift_free_nodes(mesh, coord: int, old: float, new: float) -> None:
    if abs(new - old) < 1.0e-15:
        return
    pairs = getattr(mesh, "free_nodes", None)
    if not pairs:
        return
    span = abs(float(old)) + abs(float(new)) + 1.0
    tol = max(1.0e-9, 1.0e-6 * span)
    updated = []
    for pair in pairs:
        if pair is None or len(pair) < 2:
            continue
        item = [float(pair[0]), float(pair[1])]
        if abs(item[coord] - old) <= tol:
            item[coord] = float(new)
        updated.append(item)
    mesh.free_nodes = updated


def _nearest_axis_index(axis: Sequence[float], value: float) -> int:
    arr = np.asarray(list(axis), dtype=float).ravel()
    if arr.size == 0:
        return 0
    return int(np.argmin(np.abs(arr - float(value))))


def _line_tolerance(xs: np.ndarray, ys: np.ndarray) -> float:
    gaps = []
    if xs.size >= 2:
        gaps.append(float(np.min(np.abs(np.diff(xs)))))
    if ys.size >= 2:
        gaps.append(float(np.min(np.abs(np.diff(ys)))))
    span = 1.0
    if xs.size:
        span = max(span, float(np.ptp(xs)))
    if ys.size:
        span = max(span, float(np.ptp(ys)))
    min_gap = min(gaps) if gaps else span
    return max(1.0e-9, 1.0e-6 * span, 1.0e-4 * min_gap)


def _classify_edge(
    a: np.ndarray,
    b: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    tol: float,
) -> Optional[Tuple[str, int]]:
    if abs(float(a[0]) - float(b[0])) <= tol:
        ix = _nearest_axis_index(xs, 0.5 * (float(a[0]) + float(b[0])))
        if abs(float(xs[ix]) - 0.5 * (float(a[0]) + float(b[0]))) <= tol:
            return ("x", ix)
        return None
    if abs(float(a[1]) - float(b[1])) <= tol:
        iy = _nearest_axis_index(ys, 0.5 * (float(a[1]) + float(b[1])))
        if abs(float(ys[iy]) - 0.5 * (float(a[1]) + float(b[1]))) <= tol:
            return ("y", iy)
        return None
    return None


def _nearest_node(click: np.ndarray, node_display: np.ndarray, pixel_tol: float) -> Optional[int]:
    pts = np.asarray(node_display, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return None
    delta = pts[:, :2] - click[:2]
    dist = np.sqrt(np.einsum("ij,ij->i", delta, delta))
    index = int(np.argmin(dist))
    if dist[index] <= float(pixel_tol):
        return index
    return None


def _nearest_edge(click: np.ndarray, edge_display: np.ndarray, pixel_tol: float) -> Optional[int]:
    segs = np.asarray(edge_display, dtype=float)
    if segs.ndim != 3 or segs.shape[0] == 0:
        return None
    best_i = -1
    best_d = float(pixel_tol)
    px, py = float(click[0]), float(click[1])
    for i, seg in enumerate(segs):
        d = _point_segment_distance(px, py, seg[0, 0], seg[0, 1], seg[1, 0], seg[1, 1])
        if d < best_d:
            best_d = d
            best_i = i
    if best_i < 0:
        return None
    return best_i


def _point_segment_distance(
    px: float, py: float, x0: float, y0: float, x1: float, y1: float
) -> float:
    dx = float(x1) - float(x0)
    dy = float(y1) - float(y0)
    length2 = dx * dx + dy * dy
    if length2 <= 1.0e-18:
        return hypot(px - x0, py - y0)
    t = ((px - x0) * dx + (py - y0) * dy) / length2
    t = min(max(t, 0.0), 1.0)
    return hypot(px - (x0 + t * dx), py - (y0 + t * dy))


def _slot_from_edge(
    axis: str,
    index: int,
    xs: Sequence[float],
    ys: Sequence[float],
    click: np.ndarray,
    node_display: np.ndarray,
    node_xyz: np.ndarray,
) -> Tuple[int, int]:
    """Pick a representative grid slot on the selected line (nearest node)."""
    ids = nodes_on_grid_line(node_xyz, xs, ys, axis, index)
    if ids.size and node_display is not None and np.asarray(node_display).shape[0]:
        disp = np.asarray(node_display, dtype=float)[ids]
        local = int(np.argmin(np.sum((disp[:, :2] - click[:2]) ** 2, axis=1)))
        nid = int(ids[local])
        x, y = float(node_xyz[nid, 0]), float(node_xyz[nid, 1])
        return node_grid_slot(x, y, xs, ys)
    if str(axis) == "x":
        return int(index), 0
    return 0, int(index)
