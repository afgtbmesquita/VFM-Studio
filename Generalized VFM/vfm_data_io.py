# =============================================================================
#  plate / vfm_data_io.py
# -----------------------------------------------------------------------------
#  Plate-specific data ingestion.  Analysis-type aware: the loader only
#  requires the strain / curvature fields that the active analysis type
#  actually consumes.
#
#  Two MAT-file layouts are accepted; the loader auto-detects which one a
#  given input file uses.
#
#  Layout A -- "nodes" (preferred -- matches Loading_Data/FromAbaqustoVFM.m)
#  -------------------------------------------------------------------------
#      nodes         : (N, 2) array of measurement coordinates ``[x, y]``.
#      SE1grid       : 1-D array, len N -- eps_xx          (Abaqus SE-SE1)
#      SE2grid       : 1-D array, len N -- eps_yy          (Abaqus SE-SE2)
#      SE3grid       : 1-D array, len N -- gamma_xy        (Abaqus SE-SE3)
#      SK1grid       : 1-D array, len N -- kappa_xx        (Abaqus SK-SK1)
#      SK2grid       : 1-D array, len N -- kappa_yy        (Abaqus SK-SK2)
#      SK3grid       : 1-D array, len N -- kappa_xy        (Abaqus SK-SK3)
#      F             : optional -- applied force / moment vector
#
#  The ``grid`` suffix is accepted but not required.
#
#  Layout B -- "X1/X2" (legacy)
#  -------------------------------------------------------------------------
#      X1, X2                 : coordinates, either 1-D length N or
#                               same-sized 2-D arrays.
#      Eps1, Eps2, Eps6       : membrane Voigt strains, 1-D or same-sized 2-D.
#      K1, K2, K6             : engineering curvatures, 1-D or same-sized 2-D.
#      F                      : optional applied force / moment vector.
#
#  Engineering / Voigt sign conventions:
#      Eps6   = gamma_xy = 2 eps_xy
#      K1     = kappa_xx = -d^2 w / dx^2
#      K2     = kappa_yy = -d^2 w / dy^2
#      K6     = kappa_xy = -2 d^2 w / (dx dy)
#
#  Analysis-type contract on ``PointMeasurements.strains``:
#      'in_plane'      -> shape (3, n_points) rows = [Eps1, Eps2, Eps6]
#      'out_of_plane'  -> shape (3, n_points) rows = [K1,   K2,   K6  ]
#                         (use this for curvature-only bending files)
#      'both'          -> shape (6, n_points) rows = [Eps1, Eps2, Eps6,
#                                                      K1,   K2,   K6  ]
# =============================================================================

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.io import loadmat


# =============================================================================
#  Data containers
# =============================================================================

@dataclass
class PointMeasurements:
    """Measured generalised-strain field sampled on a point cloud.

    Fields
    ------
    points
        (n_points, 2) array of measurement coordinates ``[x, y]``.
    strains
        ``(n_gen, n_points)`` array of measured generalised strains; row
        layout depends on the analysis type the loader was called with
        (see module docstring).
    applied_force
        Optional applied-force / moment vector (one entry per external-work
        group declared in the BC file's ``EXTERNAL`` list).  ``None`` if
        the MAT file did not provide an ``F`` field.
    area_per_point
        Discrete weight that multiplies each pointwise contribution.
    data_bounds
        Bounding-box extents of the measurement cloud.
    analysis_type
        Analysis type the loader was called with (informational; used by
        downstream callers to cross-check shapes).
    """
    points: np.ndarray
    strains: np.ndarray
    applied_force: Optional[np.ndarray]
    area_per_point: float
    data_bounds: Dict[str, float]
    analysis_type: str = "both"


@dataclass
class BoundaryConditions:
    """Parsed boundary-condition Python file.

    ``analysis_type`` is the value of the optional top-level
    ``ANALYSIS_TYPE`` constant if the BC file declares one (``None``
    otherwise).  Callers compare it against the active run's analysis
    type and surface a warning on mismatch.
    """
    prescribed: List[Dict[str, Any]]
    linking: List[Dict[str, Any]]
    external: List[Dict[str, Any]]
    analysis_type: Optional[str] = None


# =============================================================================
#  Field-name tables (per layout)
# =============================================================================

# New "nodes" layout.  Each tuple lists the canonical name first, then the
# short form.
_NEW_STRAIN_NAMES = (
    ("SE1grid", "SE1"),
    ("SE2grid", "SE2"),
    ("SE3grid", "SE3"),
)
_NEW_CURV_NAMES = (
    ("SK1grid", "SK1"),
    ("SK2grid", "SK2"),
    ("SK3grid", "SK3"),
)

_LEGACY_REQUIRED_COORD_FIELDS = ("X1", "X2")
_LEGACY_STRAIN_FIELDS = ("Eps1", "Eps2", "Eps6")
_LEGACY_CURVATURE_FIELDS = ("K1", "K2", "K6")


def _pick_field(raw: Dict[str, Any], candidates) -> Optional[str]:
    for name in candidates:
        if name in raw:
            return name
    return None


def _is_new_layout(raw: Dict[str, Any]) -> bool:
    return "nodes" in raw


# =============================================================================
#  Loaders
# =============================================================================

def load_point_measurements(
    data_file: Path,
    *,
    analysis_type: str = "both",
    total_sample_area: Optional[float] = None,
) -> PointMeasurements:
    """Load the plate experimental MAT file and return a ``PointMeasurements``.

    Parameters
    ----------
    data_file
        Path to the ``.mat`` measurement file.
    analysis_type
        One of ``'in_plane'``, ``'out_of_plane'``, ``'both'``.  Controls
        which strain / curvature fields are required vs. optional and the
        shape of the returned ``strains`` array.
    total_sample_area
        Optional override for ``area_per_point`` (see the docstring of
        :class:`PointMeasurements`).
    """
    data_file = Path(data_file)
    if not data_file.is_file():
        raise FileNotFoundError(f"Measurement file not found: {data_file}")

    analysis_key = str(analysis_type).strip().lower()
    if analysis_key not in ("in_plane", "out_of_plane", "both"):
        raise ValueError(
            f"Unsupported analysis_type {analysis_type!r}; expected one of "
            "'in_plane', 'out_of_plane', 'both'."
        )

    raw = loadmat(str(data_file), squeeze_me=True, mat_dtype=False)

    if _is_new_layout(raw):
        points, strains, n_points = _read_new_layout(raw, data_file,
                                                     analysis_key)
    else:
        points, strains, n_points = _read_legacy_layout(raw, data_file,
                                                        analysis_key)

    x_min = float(points[:, 0].min())
    x_max = float(points[:, 0].max())
    y_min = float(points[:, 1].min())
    y_max = float(points[:, 1].max())
    bbox_area = (x_max - x_min) * (y_max - y_min)
    if bbox_area <= 0.0:
        raise ValueError(
            "Measured data footprint has non-positive area; check the "
            "node coordinates in the input file."
        )
    if total_sample_area is None:
        integration_area = bbox_area
    else:
        integration_area = float(total_sample_area)
        if not np.isfinite(integration_area) or integration_area <= 0.0:
            raise ValueError(
                "total_sample_area must be a finite strictly positive "
                f"scalar; got {total_sample_area!r}."
            )
    area_per_point = integration_area / n_points

    applied_force: Optional[np.ndarray] = None
    if "F" in raw:
        applied_force = np.atleast_1d(
            np.asarray(raw["F"], dtype=float)
        ).ravel()
        if applied_force.size == 0:
            applied_force = None

    return PointMeasurements(
        points=points,
        strains=strains,
        applied_force=applied_force,
        area_per_point=area_per_point,
        data_bounds={"x_min": x_min, "x_max": x_max,
                     "y_min": y_min, "y_max": y_max},
        analysis_type=analysis_key,
    )


# -----------------------------------------------------------------------------
#  New layout: nodes + SE*/SK* 1-D arrays
# -----------------------------------------------------------------------------

def _read_new_layout(raw: Dict[str, Any],
                     data_file: Path,
                     analysis_type: str):
    nodes_raw = np.asarray(raw["nodes"], dtype=float)
    if nodes_raw.ndim != 2 or nodes_raw.shape[1] != 2:
        raise ValueError(
            f"In {data_file.name}: 'nodes' must be a (N, 2) array of "
            f"[x, y] coordinates; got shape {nodes_raw.shape}."
        )
    n_points = int(nodes_raw.shape[0])
    if n_points == 0:
        raise ValueError(
            f"In {data_file.name}: 'nodes' is empty; nothing to identify."
        )

    need_strains = analysis_type in ("in_plane", "both")
    need_curvatures = analysis_type in ("out_of_plane", "both")

    strain_picks = (
        [_pick_field(raw, names) for names in _NEW_STRAIN_NAMES]
        if need_strains else [None, None, None]
    )
    curv_picks = (
        [_pick_field(raw, names) for names in _NEW_CURV_NAMES]
        if need_curvatures else [None, None, None]
    )

    if need_strains:
        missing_strains = [list(names) for names, pick in zip(_NEW_STRAIN_NAMES,
                                                              strain_picks)
                           if pick is None]
        if missing_strains:
            raise KeyError(
                f"In {data_file.name}: analysis_type='{analysis_type}' "
                "requires in-plane strain fields (any of "
                + ", ".join(repr(n) for pair in _NEW_STRAIN_NAMES for n in pair)
                + "); the following slots could not be matched: "
                f"{missing_strains}.  Add the fields to the MAT file or "
                "switch analysis_type to 'out_of_plane'."
            )

    if need_curvatures:
        missing_curvatures = [list(names) for names, pick in zip(_NEW_CURV_NAMES,
                                                                 curv_picks)
                              if pick is None]
        if missing_curvatures:
            raise KeyError(
                f"In {data_file.name}: analysis_type='{analysis_type}' "
                "requires curvature fields (any of "
                + ", ".join(repr(n) for pair in _NEW_CURV_NAMES for n in pair)
                + "); the following slots could not be matched: "
                f"{missing_curvatures}.  Add the fields to the MAT file or "
                "switch analysis_type to 'in_plane'."
            )

    rows: List[np.ndarray] = []
    sizes: Dict[str, int] = {"nodes": n_points}

    if need_strains:
        eps_xx = np.asarray(raw[strain_picks[0]], dtype=float).ravel()
        eps_yy = np.asarray(raw[strain_picks[1]], dtype=float).ravel()
        gamma_xy = np.asarray(raw[strain_picks[2]], dtype=float).ravel()
        sizes[strain_picks[0]] = eps_xx.size
        sizes[strain_picks[1]] = eps_yy.size
        sizes[strain_picks[2]] = gamma_xy.size
        rows.extend([eps_xx, eps_yy, gamma_xy])

    if need_curvatures:
        kappa_xx = np.asarray(raw[curv_picks[0]], dtype=float).ravel()
        kappa_yy = np.asarray(raw[curv_picks[1]], dtype=float).ravel()
        kappa_xy = np.asarray(raw[curv_picks[2]], dtype=float).ravel()
        sizes[curv_picks[0]] = kappa_xx.size
        sizes[curv_picks[1]] = kappa_yy.size
        sizes[curv_picks[2]] = kappa_xy.size
        rows.extend([kappa_xx, kappa_yy, kappa_xy])

    if len(set(sizes.values())) != 1:
        raise ValueError(
            f"In {data_file.name}: inconsistent field sizes -- "
            + ", ".join(f"{k}={v}" for k, v in sizes.items())
            + ".  All strain / curvature arrays must have the same "
            "length as the number of rows in 'nodes'."
        )

    strains = np.vstack(rows)
    return nodes_raw.astype(float, copy=False), strains, n_points


# -----------------------------------------------------------------------------
#  Legacy layout: X1/X2 + Eps*/K* arrays
# -----------------------------------------------------------------------------

def _read_legacy_layout(raw: Dict[str, Any],
                        data_file: Path,
                        analysis_type: str):
    need_strains = analysis_type in ("in_plane", "both")
    need_curvatures = analysis_type in ("out_of_plane", "both")

    missing_coords = [name for name in _LEGACY_REQUIRED_COORD_FIELDS
                      if name not in raw]
    if missing_coords:
        raise KeyError(
            f"In {data_file.name}: missing coordinate field(s) "
            f"{missing_coords}; expected legacy layout with X1, X2 or new "
            "'nodes' layout."
        )

    if need_strains:
        missing_strains = [name for name in _LEGACY_STRAIN_FIELDS
                           if name not in raw]
        if missing_strains:
            raise KeyError(
                f"In {data_file.name}: analysis_type='{analysis_type}' "
                "requires legacy strain fields "
                f"{list(_LEGACY_STRAIN_FIELDS)}; missing: "
                f"{missing_strains}.  Add them, or switch analysis_type "
                "to 'out_of_plane'."
            )

    if need_curvatures:
        missing_curvatures = [name for name in _LEGACY_CURVATURE_FIELDS
                              if name not in raw]
        if missing_curvatures:
            raise KeyError(
                f"In {data_file.name}: analysis_type='{analysis_type}' "
                "requires legacy curvature fields "
                f"{list(_LEGACY_CURVATURE_FIELDS)}; missing: "
                f"{missing_curvatures}.  Add them, or switch analysis_type "
                "to 'in_plane'."
            )

    x_coordinates = np.asarray(raw["X1"], dtype=float).ravel()
    y_coordinates = np.asarray(raw["X2"], dtype=float).ravel()
    n_points = x_coordinates.size
    if n_points == 0:
        raise ValueError(f"In {data_file.name}: coordinate arrays are empty.")

    sizes: Dict[str, int] = {"X1": x_coordinates.size,
                              "X2": y_coordinates.size}
    rows: List[np.ndarray] = []

    if need_strains:
        eps_xx = np.asarray(raw["Eps1"], dtype=float).ravel()
        eps_yy = np.asarray(raw["Eps2"], dtype=float).ravel()
        gamma_xy = np.asarray(raw["Eps6"], dtype=float).ravel()
        sizes["Eps1"] = eps_xx.size
        sizes["Eps2"] = eps_yy.size
        sizes["Eps6"] = gamma_xy.size
        rows.extend([eps_xx, eps_yy, gamma_xy])

    if need_curvatures:
        kappa_xx = np.asarray(raw["K1"], dtype=float).ravel()
        kappa_yy = np.asarray(raw["K2"], dtype=float).ravel()
        kappa_xy = np.asarray(raw["K6"], dtype=float).ravel()
        sizes["K1"] = kappa_xx.size
        sizes["K2"] = kappa_yy.size
        sizes["K6"] = kappa_xy.size
        rows.extend([kappa_xx, kappa_yy, kappa_xy])

    if len(set(sizes.values())) != 1:
        raise ValueError(
            f"In {data_file.name}: inconsistent field sizes -- "
            + ", ".join(f"{k}={v}" for k, v in sizes.items())
        )

    points = np.column_stack([x_coordinates, y_coordinates])
    strains = np.vstack(rows)
    return points, strains, n_points


# =============================================================================
#  BC file loader
# =============================================================================

def load_boundary_conditions(bc_file: Path) -> BoundaryConditions:
    """Parse the plate boundary-conditions Python file.

    Top-level variables read from the module:
        PRESCRIBED, LINKING, EXTERNAL : list of dicts.
        ANALYSIS_TYPE                 : optional string -- when present it
                                        records the analysis type the file
                                        was authored for.
    """
    bc_file = Path(bc_file)
    if not bc_file.is_file():
        raise FileNotFoundError(f"Boundary-conditions file not found: {bc_file}")

    if bc_file.suffix.lower() != ".py":
        raise ValueError(
            f"Plate boundary conditions must be provided as a Python .py file; "
            f"got {bc_file.name}."
        )

    module_name = f"_plate_boundary_conditions_{bc_file.stem}"
    spec = importlib.util.spec_from_file_location(module_name, bc_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import boundary-condition file: {bc_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    missing = [
        name for name in ("PRESCRIBED", "LINKING", "EXTERNAL")
        if not hasattr(module, name)
    ]
    if missing:
        raise AttributeError(
            f"Boundary-condition file {bc_file.name} is missing required "
            f"variable(s): {missing}.  Expected PRESCRIBED, LINKING, EXTERNAL."
        )

    prescribed = list(getattr(module, "PRESCRIBED") or [])
    linking = list(getattr(module, "LINKING") or [])
    external = list(getattr(module, "EXTERNAL") or [])

    raw_analysis = getattr(module, "ANALYSIS_TYPE", None)
    analysis_type: Optional[str] = None
    if raw_analysis is not None:
        analysis_type = str(raw_analysis).strip().lower()

    return BoundaryConditions(
        prescribed=prescribed,
        linking=linking,
        external=external,
        analysis_type=analysis_type,
    )
