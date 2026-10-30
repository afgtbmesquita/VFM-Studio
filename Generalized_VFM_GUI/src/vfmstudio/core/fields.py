"""Load DIC / kinematic fields as xarray datasets (Qt-free)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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


_DISPLACEMENT_ALIASES = {
    "w": ("W", "Wgrid", "Uz", "UZ"),
    "u": ("U", "Ugrid", "Ux", "UX"),
    "v": ("V", "Vgrid", "Uy", "UY"),
    "z": ("Z", "Zgrid"),
}


def extras_from_mat(raw: Optional[Mapping[str, Any]]) -> Dict[str, np.ndarray]:
    """U/V/W/Z channels from an already-loaded MAT dict."""
    if not raw:
        return {}
    extras: Dict[str, np.ndarray] = {}
    for key, names in _DISPLACEMENT_ALIASES.items():
        for name in names:
            if name in raw:
                extras[key] = np.asarray(raw[name], dtype=float).ravel()
                break
    return extras


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
    return extras_from_mat(raw)


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
    """Specimen points as a flat plate, or the warped current configuration.

    Undeformed (``deformed=False``) is always ``(X, Y, 0)``. Measured ``Z`` is
    a current-configuration coordinate, so it is used only when drawing the
    deformed shape ``(X+sU, Y+sV, Z+sW)``.
    """
    x = np.asarray(dataset["x"].values, dtype=float).ravel()
    y = np.asarray(dataset["y"].values, dtype=float).ravel()
    n = min(x.size, y.size)
    if deformed and "z" in dataset:
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


def _loadmat_once(path: Path) -> Optional[dict]:
    try:
        from scipy.io import loadmat
    except ImportError:
        return None
    if not path.is_file():
        return None
    try:
        return loadmat(str(path), squeeze_me=True, mat_dtype=False)
    except Exception:
        return None


def load_step_fields(
    data_file: Path,
    analysis_type: str,
    total_sample_area: Optional[float] = None,
    *,
    align_to_axes: bool = False,
    alignment_angle_deg: Optional[float] = None,
) -> xr.Dataset:
    path = Path(data_file)
    raw = _loadmat_once(path)
    measurements = bridge.load_point_measurements(
        path,
        analysis_type=analysis_type,
        total_sample_area=total_sample_area,
        raw=raw,
    )
    extras = extras_from_mat(raw) if raw is not None else try_load_displacements(path)
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


def subtract_reference_dataset(target: xr.Dataset, reference: xr.Dataset) -> xr.Dataset:
    """Return ``target`` minus ``reference`` for kinematic fields, U/V/W, and F.

    Coordinates (X, Y, Z) are left unchanged.  Used by the viewport and load
    table so a selected reference step matches Identify.
    """
    out = target.copy(deep=True)
    skip = {"x", "y", "z"}
    for name in list(out.data_vars):
        if str(name).lower() in skip:
            continue
        if name not in reference.data_vars:
            continue
        t_vals = np.asarray(out[name].values, dtype=float)
        r_vals = np.asarray(reference[name].values, dtype=float)
        if t_vals.shape != r_vals.shape:
            raise ValueError(
                f"Cannot reference field {name!r}: shapes {t_vals.shape} vs "
                f"{r_vals.shape}."
            )
        out[name] = (out[name].dims, t_vals - r_vals)
    t_force = out.attrs.get("applied_force")
    r_force = reference.attrs.get("applied_force")
    if t_force is None and r_force is None:
        return out
    if t_force is None or r_force is None:
        raise ValueError(
            "Cannot reference load steps: applied force is missing on the "
            "reference or the current step."
        )
    t_arr = np.atleast_1d(np.asarray(t_force, dtype=float)).ravel()
    r_arr = np.atleast_1d(np.asarray(r_force, dtype=float)).ravel()
    if t_arr.shape != r_arr.shape:
        raise ValueError(
            "Cannot reference load steps: applied-force vectors have different "
            f"lengths {t_arr.size} vs {r_arr.size}."
        )
    out.attrs["applied_force"] = (t_arr - r_arr).tolist()
    return out


def referenced_step_dataset(
    dataset: xr.Dataset,
    experiment: Any,
    *,
    analysis_type: str,
    source_path: Optional[Path] = None,
    cache: Optional[Dict[tuple, xr.Dataset]] = None,
) -> xr.Dataset:
    """Subtract the experiment's reference-step fields from ``dataset``."""
    if experiment is None or getattr(experiment, "reference_step", None) is None:
        return dataset
    steps = list(getattr(experiment, "steps", []) or [])
    ref_index = int(experiment.reference_step) - 1
    if ref_index < 0 or ref_index >= len(steps):
        return dataset
    from vfmstudio.core.step_series import resolve_step_data_file

    ref_path = resolve_step_data_file(Path(steps[ref_index].data_file), source_path)
    if ref_path is None:
        raise FileNotFoundError(
            f"Reference step {experiment.reference_step} file not found: "
            f"{steps[ref_index].data_file}"
        )
    cache_key = (
        str(ref_path),
        str(analysis_type),
        getattr(experiment, "sample_area_total", None),
        bool(getattr(experiment, "fit_data_to_system_axes", False)),
    )
    reference = None if cache is None else cache.get(cache_key)
    if reference is None:
        angle = experiment_alignment_angle(experiment, analysis_type, source_path)
        align = bool(getattr(experiment, "fit_data_to_system_axes", False))
        reference = load_step_fields(
            ref_path,
            analysis_type,
            getattr(experiment, "sample_area_total", None),
            align_to_axes=align,
            alignment_angle_deg=angle if align else None,
        )
        if cache is not None:
            cache[cache_key] = reference
    return subtract_reference_dataset(dataset, reference)


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
    estimate_experiment_sample_area(
        experiment,
        resolved,
        analysis_type=analysis_type,
        align=align,
        angle=angle,
        replace=False,
    )
    return apply_detected_mesh_bounds(experiment, detected, replace=replace)


def estimate_experiment_sample_area(
    experiment: Any,
    data_file: Path,
    *,
    analysis_type: str = "both",
    source_path: Optional[Path] = None,
    align: Optional[bool] = None,
    angle: Optional[float] = None,
    replace: bool = False,
) -> Optional[float]:
    """Fill ``sample_area_total`` from the DIC cloud (Delaunay cell area × n).

    Does not overwrite an area the user or a project file already set unless
    ``replace`` is true.  Returns the area written, or ``None`` if the cloud
    could not be read.
    """
    if not replace and getattr(experiment, "sample_area_total", None) not in (
        None,
        0,
        0.0,
    ):
        return float(experiment.sample_area_total)
    from vfmstudio.core.step_series import resolve_step_data_file

    resolved = resolve_step_data_file(Path(data_file), source_path) or Path(data_file)
    xy = read_xy_from_file(resolved, analysis_type)
    if xy is None:
        return None
    xs, ys = xy
    if align is None:
        align = bool(getattr(experiment, "fit_data_to_system_axes", False))
    if angle is None and align:
        angle = experiment_alignment_angle(experiment, analysis_type, source_path)
    if align and angle is not None and abs(float(angle)) > 0.0:
        xs, ys = rotate_xy(xs, ys, float(angle))
    from vfmstudio.core.sample_area import estimate_sample_area

    area = estimate_sample_area(xs, ys)
    if area is None:
        return None
    experiment.sample_area_total = float(area)
    return float(area)


_FORCE_CACHE: Dict[tuple, Optional[tuple]] = {}


def _as_label_list(raw) -> Optional[List[str]]:
    if raw is None:
        return None
    if isinstance(raw, (str, bytes, np.str_)):
        text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
        text = text.strip()
        return [text] if text else None
    try:
        arr = np.atleast_1d(np.asarray(raw, dtype=object)).ravel()
    except (TypeError, ValueError):
        return None
    labels: List[str] = []
    for item in arr:
        if item is None:
            continue
        if isinstance(item, bytes):
            text = item.decode("utf-8", "replace").strip()
        else:
            text = str(item).strip()
        if text:
            labels.append(text)
    return labels or None


def read_force_labels_from_file(data_file: Path) -> Optional[List[str]]:
    """Return ``F_labels`` from a measurement MAT, or ``None``."""
    path = Path(data_file)
    if not path.is_file():
        return None
    try:
        from scipy.io import loadmat
    except ImportError:
        return None
    try:
        raw = loadmat(
            str(path),
            squeeze_me=True,
            struct_as_record=False,
            variable_names=["F_labels"],
        )
    except Exception:
        try:
            raw = loadmat(str(path), squeeze_me=True, struct_as_record=False)
        except Exception:
            return None
    if not raw or "F_labels" not in raw:
        return None
    return _as_label_list(raw.get("F_labels"))


def force_component_text(index: int, labels: Optional[Sequence[str]] = None) -> str:
    """Combo label for ``F[index]``, using a stored name when present."""
    if labels is not None and 0 <= int(index) < len(labels):
        name = str(labels[int(index)]).strip()
        if name:
            return name
    return f"F[{int(index)}]"


def read_applied_force_from_file(data_file: Path) -> Optional[np.ndarray]:
    """Return the MAT-file ``F`` vector, or ``None`` if missing or unreadable.

    Loads only ``F`` when SciPy allows it, so the load table can preview every
    step without reading the DIC cloud.
    """
    path = Path(data_file)
    try:
        stat = path.stat()
    except OSError:
        return None
    key = (str(path.resolve()) if path.exists() else str(path), int(stat.st_mtime_ns), int(stat.st_size))
    cached = _FORCE_CACHE.get(key)
    if key in _FORCE_CACHE:
        if cached is None:
            return None
        return np.asarray(cached, dtype=float)

    force = _load_applied_force_raw(path)
    _FORCE_CACHE[key] = None if force is None else tuple(float(v) for v in force)
    return force


def _load_applied_force_raw(path: Path) -> Optional[np.ndarray]:
    try:
        from scipy.io import loadmat
    except ImportError:
        return None
    raw = None
    try:
        raw = loadmat(
            str(path),
            squeeze_me=True,
            struct_as_record=False,
            variable_names=["F"],
        )
    except Exception:
        try:
            raw = loadmat(str(path), squeeze_me=True, struct_as_record=False)
        except Exception:
            return None
    if not raw or "F" not in raw:
        return None
    try:
        applied = np.atleast_1d(np.asarray(raw["F"], dtype=float)).ravel()
    except (TypeError, ValueError):
        return None
    if applied.size == 0:
        return None
    return applied


def mat_force_component(force: Optional[np.ndarray], index: int, n_groups: int) -> Optional[float]:
    """Pick the ``F`` entry that pairs with EXTERNAL group ``index``.

    Legacy identity map: ``F[i]`` feeds group ``i``.  A scalar ``F`` is
    copied to every group.  Prefer :func:`mapped_force_component` when the
    group stores an explicit ``forceIndex``.
    """
    if force is None or n_groups <= 0 or index < 0:
        return None
    arr = np.atleast_1d(np.asarray(force, dtype=float)).ravel()
    if arr.size == 0:
        return None
    if arr.size == 1 and n_groups > 1:
        value = float(arr[0])
        return value if np.isfinite(value) else None
    if index >= arr.size:
        return None
    value = float(arr[index])
    return value if np.isfinite(value) else None


def force_index_for_group(entry, group_index: int) -> int:
    """``F`` component used by this EXTERNAL group.  Default is list order."""
    raw = None
    if isinstance(entry, dict):
        raw = entry.get("forceIndex", entry.get("force_index"))
    else:
        raw = getattr(entry, "force_index", None)
    if raw is None or raw == "":
        return int(group_index)
    try:
        index = int(raw)
    except (TypeError, ValueError):
        return int(group_index)
    return index


def mapped_force_component(force, entry, group_index: int) -> Optional[float]:
    """``F[forceIndex]`` for one group.  A scalar ``F`` still broadcasts."""
    if force is None:
        return None
    arr = np.atleast_1d(np.asarray(force, dtype=float)).ravel()
    if arr.size == 0:
        return None
    f_index = force_index_for_group(entry, group_index)
    if arr.size == 1:
        value = float(arr[0])
        return value if np.isfinite(value) else None
    if f_index < 0 or f_index >= arr.size:
        return None
    value = float(arr[f_index])
    return value if np.isfinite(value) else None
