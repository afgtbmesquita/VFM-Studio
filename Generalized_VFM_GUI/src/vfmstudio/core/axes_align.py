"""Rotate a DIC cloud onto the nearest lab X/Y axis.

Specimens are often a few degrees off the camera axes (A1_PB is typically
2–3 deg). The correction is the smallest rotation that takes the cloud's
principal axis onto ±X or ±Y. Coordinates, in-plane displacements, and
Voigt engineering strain/curvature all transform with that same angle.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


def _finite_xy(x, y) -> Tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    n = min(x_arr.size, y_arr.size)
    x_arr = x_arr[:n]
    y_arr = y_arr[:n]
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    return x_arr[mask], y_arr[mask]


def principal_axis_angle_deg(x, y) -> Optional[float]:
    """Angle of the major principal axis from +X, in degrees."""
    xs, ys = _finite_xy(x, y)
    if xs.size < 3:
        return None
    centered = np.column_stack([xs - np.mean(xs), ys - np.mean(ys)])
    if np.allclose(centered, 0.0):
        return None
    covariance = np.cov(centered, rowvar=False)
    if covariance.shape != (2, 2):
        return None
    evals, evecs = np.linalg.eigh(covariance)
    axis = evecs[:, int(np.argmax(evals))]
    return float(np.degrees(np.arctan2(axis[1], axis[0])))


def nearest_axis_correction_deg(principal_deg: float) -> float:
    """Smallest rotation that maps ``principal_deg`` onto ±X or ±Y."""
    folded = ((float(principal_deg) + 90.0) % 180.0) - 90.0
    if abs(folded) <= 45.0:
        return -folded
    target = 90.0 if folded > 0.0 else -90.0
    return target - folded


def alignment_rotation_deg(x, y) -> float:
    """CCW rotation (deg) that lines the cloud up with the nearest lab axis."""
    principal = principal_axis_angle_deg(x, y)
    if principal is None:
        return 0.0
    return float(nearest_axis_correction_deg(principal))


def rotation_matrix(angle_deg: float) -> np.ndarray:
    phi = np.deg2rad(float(angle_deg))
    cosine, sine = np.cos(phi), np.sin(phi)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=float)


def engineering_rotation_matrix(angle_deg: float) -> np.ndarray:
    """Voigt T(φ) for engineering shear / curvature (K6 = 2 κ_xy)."""
    phi = np.deg2rad(float(angle_deg))
    cosine, sine = np.cos(phi), np.sin(phi)
    return np.array(
        [
            [cosine * cosine, sine * sine, cosine * sine],
            [sine * sine, cosine * cosine, -cosine * sine],
            [-2.0 * cosine * sine, 2.0 * cosine * sine, cosine * cosine - sine * sine],
        ],
        dtype=float,
    )


def rotate_xy(x, y, angle_deg: float, origin: Optional[Tuple[float, float]] = None):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if abs(float(angle_deg)) <= 1.0e-12:
        return x_arr, y_arr
    if origin is None:
        xs, ys = _finite_xy(x_arr, y_arr)
        origin = (
            float(np.mean(xs)) if xs.size else 0.0,
            float(np.mean(ys)) if ys.size else 0.0,
        )
    matrix = rotation_matrix(angle_deg)
    shifted = np.column_stack([x_arr - origin[0], y_arr - origin[1]])
    rotated = shifted @ matrix.T
    return rotated[:, 0] + origin[0], rotated[:, 1] + origin[1]


def rotate_vector_xy(u, v, angle_deg: float):
    u_arr = np.asarray(u, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    if abs(float(angle_deg)) <= 1.0e-12:
        return u_arr, v_arr
    matrix = rotation_matrix(angle_deg)
    stacked = np.column_stack([u_arr.ravel(), v_arr.ravel()])
    rotated = stacked @ matrix.T
    return rotated[:, 0].reshape(u_arr.shape), rotated[:, 1].reshape(v_arr.shape)


def rotate_engineering_fields(strains: np.ndarray, points_angle_deg: float) -> np.ndarray:
    """Rotate Voigt rows by the same CCW point-cloud rotation."""
    fields = np.asarray(strains, dtype=float)
    if fields.ndim != 2 or abs(float(points_angle_deg)) <= 1.0e-12:
        return fields
    transform = engineering_rotation_matrix(-float(points_angle_deg))
    rotated = np.array(fields, copy=True)
    n_gen = int(rotated.shape[0])
    if n_gen >= 3:
        rotated[:3] = transform @ rotated[:3]
    if n_gen >= 6:
        rotated[3:6] = transform @ rotated[3:6]
    return rotated


def rotate_displacement_extras(extras: Dict[str, np.ndarray], angle_deg: float) -> Dict[str, np.ndarray]:
    if not extras or abs(float(angle_deg)) <= 1.0e-12:
        return extras
    if "u" not in extras and "v" not in extras:
        return extras
    u = extras.get("u")
    v = extras.get("v")
    if u is None:
        u = np.zeros_like(np.asarray(v, dtype=float))
    if v is None:
        v = np.zeros_like(np.asarray(u, dtype=float))
    u2, v2 = rotate_vector_xy(u, v, angle_deg)
    out = dict(extras)
    if "u" in extras:
        out["u"] = u2
    if "v" in extras:
        out["v"] = v2
    return out


def apply_alignment_to_measurements(measurements: Any, angle_deg: float) -> float:
    """Rotate ``points`` and Voigt ``strains`` in place. Returns the angle used."""
    angle = float(angle_deg)
    if abs(angle) <= 1.0e-12:
        return 0.0
    points = np.asarray(getattr(measurements, "points"), dtype=float)
    if points.ndim != 2 or points.shape[1] < 2 or points.shape[0] == 0:
        return 0.0
    x, y = rotate_xy(points[:, 0], points[:, 1], angle)
    measurements.points = np.column_stack([x, y])
    strains = getattr(measurements, "strains", None)
    if strains is not None:
        measurements.strains = rotate_engineering_fields(strains, angle)
    bounds = {
        "x_min": float(np.nanmin(x)),
        "x_max": float(np.nanmax(x)),
        "y_min": float(np.nanmin(y)),
        "y_max": float(np.nanmax(y)),
    }
    if hasattr(measurements, "data_bounds"):
        measurements.data_bounds = bounds
    return angle
