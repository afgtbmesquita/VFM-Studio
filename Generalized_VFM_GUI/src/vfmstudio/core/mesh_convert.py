"""Convert solver mesh packages and DIC clouds into NumPy arrays for PyVista."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from vfmstudio.core import bridge

MAX_ELEMENT_COUNT_LABELS = 500


def mesh_xyz(mesh_pkg: Any) -> np.ndarray:
    coords = np.asarray(bridge.first_attr(mesh_pkg, "node_coords"), dtype=float)
    if coords.ndim != 2:
        raise ValueError("Mesh node coordinates must be a 2-D array.")
    if coords.shape[1] == 2:
        return np.column_stack([coords, np.zeros(coords.shape[0])])
    return coords[:, :3].copy()


def mesh_quads(mesh_pkg: Any) -> np.ndarray:
    return np.asarray(bridge.first_attr(mesh_pkg, "element_nodes"), dtype=np.int64)


def vtk_quad_cells(quads: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Connectivity + VTK cell-type array for linear quads."""
    quads = np.asarray(quads, dtype=np.int64)
    n = int(quads.shape[0])
    cells = np.hstack([np.full((n, 1), 4, dtype=np.int64), quads]).ravel()
    celltypes = np.full(n, 9, dtype=np.uint8)  # VTK_QUAD
    return cells, celltypes


def edge_node_xyz(mesh_pkg: Any, edge: str) -> np.ndarray:
    xyz = mesh_xyz(mesh_pkg)
    xs = xyz[:, 0]
    ys = xyz[:, 1]
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    span = max(x_max - x_min, y_max - y_min, 1.0)
    tol = 1.0e-8 * span
    key = str(edge).strip().lower()
    if key == "left":
        mask = np.abs(xs - x_min) <= tol
    elif key == "right":
        mask = np.abs(xs - x_max) <= tol
    elif key == "bottom":
        mask = np.abs(ys - y_min) <= tol
    elif key == "top":
        mask = np.abs(ys - y_max) <= tol
    else:
        mask = np.zeros(xs.shape, dtype=bool)
    return xyz[mask]


def material_axes(
    mesh_pkg: Any,
    angle_deg: float,
    scale: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    xyz = mesh_xyz(mesh_pkg)
    origin = np.array(
        [np.mean(xyz[:, 0]), np.mean(xyz[:, 1]), 0.0], dtype=float
    )
    width = float(np.ptp(xyz[:, 0]) or 1.0)
    height = float(np.ptp(xyz[:, 1]) or 1.0)
    length = scale if scale is not None else 0.25 * max(width, height)
    phi = np.deg2rad(float(angle_deg))
    axis_1 = length * np.array([np.cos(phi), np.sin(phi), 0.0])
    axis_2 = length * np.array([-np.sin(phi), np.cos(phi), 0.0])
    return {"origin": origin, "axis_1": axis_1, "axis_2": axis_2}


def points_xyz(x: Iterable[float], y: Iterable[float], z=None) -> np.ndarray:
    x_arr = np.asarray(list(x), dtype=float).ravel()
    y_arr = np.asarray(list(y), dtype=float).ravel()
    if z is None:
        z_arr = np.zeros_like(x_arr)
    else:
        z_arr = np.asarray(list(z), dtype=float).ravel()
    return np.column_stack([x_arr, y_arr, z_arr])


def _keep_element_ids(mesh_pkg: Any, keep_cells) -> list:
    if not keep_cells:
        return []
    grid = np.asarray(getattr(mesh_pkg, "element_grid_index", []), dtype=np.int64)
    if grid.ndim != 2 or grid.size == 0:
        return []
    ids: list = []
    nx, ny = int(grid.shape[0]), int(grid.shape[1])
    for cell in keep_cells:
        if cell is None or len(cell) < 2:
            continue
        ix, iy = int(cell[0]), int(cell[1])
        if 0 <= ix < nx and 0 <= iy < ny:
            eid = int(grid[ix, iy])
            if eid >= 0:
                ids.append(eid)
    return ids


def node_xyz(mesh_pkg: Any, node_ids) -> np.ndarray:
    """Coordinates of the given node ids (invalid ids are skipped)."""
    xyz = mesh_xyz(mesh_pkg)
    if xyz.size == 0:
        return np.zeros((0, 3), dtype=float)
    ids = np.asarray(list(node_ids), dtype=np.int64).ravel()
    ids = ids[(ids >= 0) & (ids < xyz.shape[0])]
    if ids.size == 0:
        return np.zeros((0, 3), dtype=float)
    return xyz[ids]


def bc_anchor_xyz(mesh_pkg: Any, entry: Any) -> np.ndarray:
    """Points where a BC glyph is drawn: named edge, else explicit nodes."""
    edge = getattr(entry, "edge", None)
    if edge:
        return edge_node_xyz(mesh_pkg, edge)
    nodes = getattr(entry, "nodes", None) or []
    if nodes:
        return node_xyz(mesh_pkg, nodes)
    return np.zeros((0, 3), dtype=float)


def _prune_by_dic(mesh_pkg: Any, document: Any, experiment: Any, analysis_type: str):
    """Drop empty cells so GUI node ids match Identify (tabbed plates)."""
    source_path = getattr(document, "source_path", None)
    xy = dic_xy_for_counts(experiment, analysis_type, source_path)
    if xy is None or xy.shape[0] == 0:
        return mesh_pkg
    prune = getattr(bridge.vfm_mesh(), "prune_mesh_by_measurement_points", None)
    if not callable(prune):
        return mesh_pkg
    try:
        pruned, _ = prune(mesh_pkg, np.asarray(xy, dtype=float)[:, :2])
    except Exception:
        return mesh_pkg
    return pruned


def build_experiment_mesh(document: Any, experiment: Any):
    """Build the solver mesh package for an experiment (GUI wrapper only)."""
    MeshSettings = bridge.mesh_settings_class()
    Bounds = bridge.bounds_class()
    payload = experiment.mesh.to_payload()
    bounds = payload["bounds"]
    settings = MeshSettings(
        number_of_elements_x=int(payload["number_of_elements_x"]),
        number_of_elements_y=int(payload["number_of_elements_y"]),
        bounds=Bounds(
            x_min=float(bounds["x_min"]),
            x_max=float(bounds["x_max"]),
            y_min=float(bounds["y_min"]),
            y_max=float(bounds["y_max"]),
        ),
        x_nodes=payload.get("x_nodes"),
        y_nodes=payload.get("y_nodes"),
    )
    analysis_type = "out_of_plane"
    material = getattr(document, "material", None)
    if material is not None:
        analysis_type = getattr(material, "analysis_type", analysis_type)
    mesh_pkg = bridge.build_mesh(settings, analysis_type)
    keep_ids = _keep_element_ids(mesh_pkg, payload.get("keep_cells"))
    n_el = int(getattr(mesh_pkg, "n_elements", 0) or 0)
    if keep_ids and len(set(keep_ids)) < n_el:
        mesh_pkg, _ = bridge.subset_mesh_elements(mesh_pkg, keep_ids)
    return _prune_by_dic(mesh_pkg, document, experiment, analysis_type)


def element_centroids(mesh_pkg: Any) -> np.ndarray:
    """Return (n_elements, 3) centroids of the linear quads."""
    xyz = mesh_xyz(mesh_pkg)
    quads = mesh_quads(mesh_pkg)
    if quads.ndim != 2 or quads.shape[0] == 0:
        return np.zeros((0, 3), dtype=float)
    return np.mean(xyz[quads], axis=1)


def dic_xy_for_counts(
    experiment: Any,
    analysis_type: str,
    source_path: Optional[Path] = None,
    step: Any = None,
) -> Optional[np.ndarray]:
    """Finite DIC (x, y) for the given step, with axes alignment if enabled."""
    from vfmstudio.core.axes_align import rotate_xy
    from vfmstudio.core.fields import experiment_alignment_angle, read_xy_from_file
    from vfmstudio.core.step_series import resolve_step_data_file

    if step is None:
        steps = list(getattr(experiment, "steps", []) or [])
        if not steps:
            return None
        step = steps[0]
    data_file = getattr(step, "data_file", None)
    if data_file is None:
        return None
    resolved = resolve_step_data_file(Path(data_file), source_path)
    if resolved is None:
        return None
    xy = read_xy_from_file(resolved, analysis_type)
    if xy is None:
        return None
    x_arr, y_arr = xy
    if getattr(experiment, "fit_data_to_system_axes", False):
        angle = experiment_alignment_angle(experiment, analysis_type, source_path)
        x_arr, y_arr = rotate_xy(x_arr, y_arr, angle)
    return np.column_stack([np.asarray(x_arr, dtype=float), np.asarray(y_arr, dtype=float)])


def count_dic_points_per_element(mesh_pkg: Any, xy: np.ndarray) -> np.ndarray:
    """Per-element DIC counts using the same mapping as VFM assembly."""
    points = np.asarray(xy, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2:
        n_el = int(bridge.first_attr(mesh_pkg, "n_elements", default=0) or 0)
        return np.zeros(n_el, dtype=np.int64)
    fn = getattr(bridge.vfm_mesh(), "count_measurements_per_element")
    return np.asarray(fn(points[:, :2], mesh_pkg), dtype=np.int64)


def summarize_element_point_counts(counts: np.ndarray) -> str:
    arr = np.asarray(counts, dtype=np.int64).ravel()
    n = int(arr.size)
    if n == 0:
        return "No elements."
    n_empty = int(np.sum(arr == 0))
    n_mapped = int(np.sum(arr))
    bits = [
        f"{n} element{'s' if n != 1 else ''}",
        f"{n_mapped} DIC point{'s' if n_mapped != 1 else ''} mapped",
        f"{int(arr.min())}–{int(arr.max())} per element",
        f"median {int(np.median(arr))}",
    ]
    if n_empty:
        bits.append(f"{n_empty} empty")
    return " · ".join(bits)
