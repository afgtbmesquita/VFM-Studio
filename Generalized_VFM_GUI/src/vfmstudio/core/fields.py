"""Load DIC / kinematic fields as xarray datasets (Qt-free)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import xarray as xr

from vfmstudio.core import bridge
from vfmstudio.core.axes_align import (
    alignment_rotation_deg,
    apply_alignment_to_measurements,
    rotate_displacement_extras,
    rotate_xy,
)


SCALAR_LABELS_OUT_OF_PLANE = (
    ("kappa_xx", "κₓ"),
    ("kappa_yy", "κᵧ"),
    ("kappa_xy", "κₓᵧ"),
)
SCALAR_LABELS_IN_PLANE = (
    ("eps_xx", "εₓ"),
    ("eps_yy", "εᵧ"),
    ("eps_xy", "γₓᵧ"),
)

_FIELD_UNICODE = {
    "kappa_xx": "κₓ",
    "kappa_yy": "κᵧ",
    "kappa_xy": "κₓᵧ",
    "eps_xx": "εₓ",
    "eps_yy": "εᵧ",
    "eps_xy": "γₓᵧ",
    "w": "w",
    "u": "u",
    "v": "v",
    "CID": "CID",
}

_FIELD_MATH = {
    "kappa_xx": r"$\kappa_{xx}$",
    "kappa_yy": r"$\kappa_{yy}$",
    "kappa_xy": r"$\kappa_{xy}$",
    "eps_xx": r"$\varepsilon_{xx}$",
    "eps_yy": r"$\varepsilon_{yy}$",
    "eps_xy": r"$\gamma_{xy}$",
    "w": r"$w$",
    "u": r"$u$",
    "v": r"$v$",
    "CID": "CID",
}

_FIELD_AXIS_MATH = {
    "kappa_xx": r"$\kappa_{xx}\ (\mathrm{mm}^{-1})$",
    "kappa_yy": r"$\kappa_{yy}\ (\mathrm{mm}^{-1})$",
    "kappa_xy": r"$\kappa_{xy}\ (\mathrm{mm}^{-1})$",
    "eps_xx": r"$\varepsilon_{xx}$",
    "eps_yy": r"$\varepsilon_{yy}$",
    "eps_xy": r"$\gamma_{xy}$",
}

MOMENT_AXIS_MATH = r"$M\ (\mathrm{N\cdot mm})$"

# Kirchhoff traction conjugate to a virtual-field DOF on a named edge.
# W_X / W_Y are slopes, not moment names: the same DOF is a normal moment on
# one pair of cuts and a twisting moment on the other.
#   x = const (left/right):  W_X → M_xx,  W_Y → M_xy
#   y = const (top/bottom):  W_Y → M_yy,  W_X → M_xy
_MOMENT_MATH_BY_CURVATURE = {
    "kappa_xx": r"M_{xx}",
    "kappa_yy": r"M_{yy}",
    "kappa_xy": r"M_{xy}",
}


def conjugate_curvature_name(dof: str, edge: Optional[str] = None) -> Optional[str]:
    """Measured curvature conjugate to ``dof`` acting on ``edge``."""
    key = str(dof or "").strip().upper().replace(" ", "_")
    side = str(edge or "").strip().lower()
    horizontal = side in {"top", "bottom"}
    vertical = side in {"left", "right"}
    if key == "W_XY":
        return "kappa_xy"
    if key == "W_X":
        return "kappa_xy" if horizontal else "kappa_xx"
    if key == "W_Y":
        return "kappa_xy" if vertical else "kappa_yy"
    return None


def conjugate_moment_math(dof: str, edge: Optional[str] = None) -> str:
    """MathText moment / force symbol for an EXTERNAL (or loaded) DOF."""
    key = str(dof or "").strip().upper().replace(" ", "_")
    if key == "W":
        return r"F_{z}"
    if key == "U":
        return r"F_{x}"
    if key == "V":
        return r"F_{y}"
    curvature = conjugate_curvature_name(key, edge)
    if curvature is not None:
        return _MOMENT_MATH_BY_CURVATURE[curvature]
    return r"M"


def conjugate_moment_plain(dof: str, edge: Optional[str] = None) -> str:
    """Viewport / combo label, e.g. ``Mxy`` from ``W_X`` on the top cut."""
    math = conjugate_moment_math(dof, edge)
    return math.replace("{", "").replace("}", "").replace("_", "")


def field_display_label(name: str) -> str:
    """Plain-text / Qt label (LaTeX-like Unicode)."""
    return _FIELD_UNICODE.get(str(name), str(name))


def field_math_title(name: str) -> str:
    """MathText title for the VTK scalar bar."""
    return _FIELD_MATH.get(str(name), str(name))


def field_axis_math(name: str) -> str:
    """Axis label with units for publication-style matplotlib plots."""
    return _FIELD_AXIS_MATH.get(str(name), field_math_title(name))



def _xy(measurements: Any) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(getattr(measurements, "points"), dtype=float)
    return points[:, 0], points[:, 1]


def measurements_to_dataset(
    measurements: Any,
    analysis_type: str = "out_of_plane",
    extras: Optional[Dict[str, np.ndarray]] = None,
) -> xr.Dataset:
    x, y = _xy(measurements)
    strains = np.asarray(getattr(measurements, "strains"), dtype=float)
    names = (
        SCALAR_LABELS_IN_PLANE
        if str(analysis_type).lower() in {"in_plane", "inplane"}
        else SCALAR_LABELS_OUT_OF_PLANE
    )
    data_vars = {}
    for row, (key, _label) in enumerate(names):
        if row < strains.shape[0]:
            data_vars[key] = ("point", strains[row])
    extras = dict(extras or {})
    z_values = extras.pop("z", None)
    n_points = int(np.asarray(x).size)

    def _matching(values: Any) -> Optional[np.ndarray]:
        arr = np.asarray(values, dtype=float).ravel()
        if arr.size != n_points:
            return None
        return arr

    for key, values in extras.items():
        fitted = _matching(values)
        if fitted is not None:
            data_vars[key] = ("point", fitted)
    force = getattr(measurements, "applied_force", None)
    attrs = {
        "analysis_type": analysis_type,
        "area_per_point": float(
            bridge.first_attr(measurements, "area_per_point", default=np.nan)
        ),
        "applied_force": None if force is None else np.asarray(force).tolist(),
    }
    coords = {"x": ("point", x), "y": ("point", y)}
    z_arr = _matching(z_values) if z_values is not None else None
    if z_arr is not None:
        coords["z"] = ("point", z_arr)
    return xr.Dataset(
        data_vars,
        coords=coords,
        attrs=attrs,
    )


def try_load_displacements(mat_path: Path) -> Dict[str, np.ndarray]:
    """Best-effort extra channels (W, U, V, Z) from a Vic-3D style MAT file."""
    try:
        from scipy.io import loadmat
    except ImportError:
        return {}
    if not Path(mat_path).is_file():
        return {}
    try:
        raw = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    except Exception:
        return {}
    extras: Dict[str, np.ndarray] = {}
    aliases = {
        "w": ("W", "Wgrid", "Uz", "UZ"),
        "u": ("U", "Ugrid", "Ux", "UX"),
        "v": ("V", "Vgrid", "Uy", "UY"),
        "z": ("Z", "Zgrid"),
    }
    for key, names in aliases.items():
        for name in names:
            if name in raw:
                extras[key] = np.asarray(raw[name], dtype=float).ravel()
                break
    return extras


def mat_has_deformed_configuration(mat_path: Optional[Path]) -> bool:
    """True if a MAT file contains matching-length U, V, and W arrays."""
    if mat_path is None:
        return False
    extras = try_load_displacements(Path(mat_path))
    return _uvw_same_length(extras)


def has_deformed_configuration(dataset: xr.Dataset) -> bool:
    """True if the dataset has U, V, W of the same length as the point cloud."""
    n_points = int(dataset.sizes.get("point", 0)) if dataset is not None else 0
    if n_points <= 0:
        return False
    extras = {}
    for name in ("u", "v", "w"):
        if name not in dataset:
            return False
        extras[name] = np.asarray(dataset[name].values, dtype=float).ravel()
    return _uvw_same_length(extras, n_points=n_points)


def specimen_points_xyz(
    dataset: xr.Dataset,
    *,
    deformed: bool = False,
    scale: float = 1.0,
) -> np.ndarray:
    """Reference ``(X, Y, Z)`` or current ``(X+sU, Y+sV, Z+sW)`` specimen points."""
    x = np.asarray(dataset["x"].values, dtype=float).ravel()
    y = np.asarray(dataset["y"].values, dtype=float).ravel()
    n = min(x.size, y.size)
    if "z" in dataset:
        z = np.asarray(dataset["z"].values, dtype=float).ravel()
        n = min(n, z.size)
    else:
        z = np.zeros(n, dtype=float)
    pts = np.column_stack([x[:n], y[:n], z[:n]])
    if not deformed or not has_deformed_configuration(dataset):
        return pts
    factor = float(scale) if np.isfinite(scale) else 1.0
    u = np.asarray(dataset["u"].values, dtype=float).ravel()[:n]
    v = np.asarray(dataset["v"].values, dtype=float).ravel()[:n]
    w = np.asarray(dataset["w"].values, dtype=float).ravel()[:n]
    pts = pts.copy()
    pts[:, 0] = pts[:, 0] + factor * u
    pts[:, 1] = pts[:, 1] + factor * v
    pts[:, 2] = pts[:, 2] + factor * w
    return pts


def _uvw_same_length(
    extras: Dict[str, np.ndarray],
    n_points: Optional[int] = None,
) -> bool:
    arrays = []
    for name in ("u", "v", "w"):
        if name not in extras:
            return False
        arrays.append(np.asarray(extras[name], dtype=float).ravel())
    sizes = {arr.size for arr in arrays}
    if len(sizes) != 1:
        return False
    size = next(iter(sizes))
    if size <= 0:
        return False
    if n_points is not None and size != int(n_points):
        return False
    return True


def load_step_fields(
    data_file: Path,
    analysis_type: str,
    total_sample_area: Optional[float] = None,
    *,
    align_to_axes: bool = False,
    alignment_angle_deg: Optional[float] = None,
) -> xr.Dataset:
    measurements = bridge.load_point_measurements(
        Path(data_file),
        analysis_type=analysis_type,
        total_sample_area=total_sample_area,
    )
    extras = try_load_displacements(Path(data_file))
    if align_to_axes:
        if alignment_angle_deg is None:
            x, y = _xy(measurements)
            alignment_angle_deg = alignment_rotation_deg(x, y)
        apply_alignment_to_measurements(measurements, float(alignment_angle_deg))
        extras = rotate_displacement_extras(extras, float(alignment_angle_deg))
        attrs_angle = float(alignment_angle_deg)
    else:
        attrs_angle = 0.0
    dataset = measurements_to_dataset(measurements, analysis_type, extras=extras)
    dataset.attrs["alignment_angle_deg"] = attrs_angle
    return dataset


def _finite_xy(x, y) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    n = min(x_arr.size, y_arr.size)
    if n == 0:
        return None
    x_arr = x_arr[:n]
    y_arr = y_arr[:n]
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if not np.any(mask):
        return None
    return x_arr[mask], y_arr[mask]


def _xy_from_raw(raw: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    nodes = raw.get("nodes")
    if nodes is not None:
        arr = np.asarray(nodes, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return _finite_xy(arr[:, 0], arr[:, 1])
        if arr.ndim == 2 and arr.shape[0] == 2 and arr.shape[1] != 2:
            return _finite_xy(arr[0], arr[1])
    for x_name, y_name in (("x", "y"), ("X", "Y"), ("coord_x", "coord_y")):
        if x_name in raw and y_name in raw:
            found = _finite_xy(raw[x_name], raw[y_name])
            if found is not None:
                return found
    return None


def read_xy_from_file(
    data_file: Path,
    analysis_type: str = "both",
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return finite (x, y) from a MAT file, or ``None``."""
    path = Path(data_file)
    if not path.is_file():
        return None
    try:
        measurements = bridge.load_point_measurements(path, analysis_type=analysis_type)
        points = np.asarray(getattr(measurements, "points"), dtype=float)
        if points.ndim == 2 and points.shape[1] >= 2:
            return _finite_xy(points[:, 0], points[:, 1])
    except Exception:
        pass
    try:
        from scipy.io import loadmat

        raw = loadmat(str(path), squeeze_me=True, struct_as_record=False)
        return _xy_from_raw(raw)
    except Exception:
        return None


def experiment_alignment_angle(
    experiment: Any,
    analysis_type: str = "both",
    source_path: Optional[Path] = None,
) -> float:
    """Shared CCW correction for every step of an experiment (first readable MAT)."""
    if not getattr(experiment, "fit_data_to_system_axes", False):
        return 0.0
    from vfmstudio.core.step_series import resolve_step_data_file

    for step in getattr(experiment, "steps", []) or []:
        resolved = resolve_step_data_file(Path(step.data_file), source_path)
        if resolved is None:
            continue
        xy = read_xy_from_file(resolved, analysis_type)
        if xy is None:
            continue
        return float(alignment_rotation_deg(*xy))
    return 0.0


def detect_xy_bounds(
    data_file: Path,
    analysis_type: str = "both",
    *,
    align_to_axes: bool = False,
    alignment_angle_deg: Optional[float] = None,
) -> Optional[Any]:
    """Return ``BoundsModel`` covering the measured point cloud, or ``None``."""
    from vfmstudio.core.document import BoundsModel

    xy = read_xy_from_file(Path(data_file), analysis_type)
    if xy is None:
        return None
    xs, ys = xy
    if align_to_axes:
        if alignment_angle_deg is None:
            alignment_angle_deg = alignment_rotation_deg(xs, ys)
        xs, ys = rotate_xy(xs, ys, float(alignment_angle_deg))
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    if x_max <= x_min or y_max <= y_min:
        return None
    return BoundsModel(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)


def sync_mesh_bounds_from_file(
    experiment: Any,
    data_file: Path,
    *,
    source_path: Optional[Path] = None,
    analysis_type: str = "both",
    replace: bool = False,
) -> bool:
    """Detect specimen limits from a MAT file and write them onto the mesh."""
    from vfmstudio.core.document import apply_detected_mesh_bounds
    from vfmstudio.core.step_series import resolve_step_data_file

    resolved = resolve_step_data_file(Path(data_file), source_path) or Path(data_file)
    align = bool(getattr(experiment, "fit_data_to_system_axes", False))
    angle = (
        experiment_alignment_angle(experiment, analysis_type, source_path)
        if align
        else 0.0
    )
    detected = detect_xy_bounds(
        resolved,
        analysis_type,
        align_to_axes=align,
        alignment_angle_deg=angle if align else None,
    )
    if detected is None:
        return False
    return apply_detected_mesh_bounds(experiment, detected, replace=replace)
