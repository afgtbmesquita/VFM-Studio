"""Calibrated DIC curvature-noise model of noise_audit_report_v2.

    sigma_c(x) = sigma_int_c * s_c(x)
    s_c(x)     = 1 + (g_c(x) - 1) / c
    g_c(x)     = ||w_c(x)|| / ||w_c(full window)||

``w_c`` is the row of Vic-3D's 15 x 15 local quadratic fit that yields curvature
component ``c``; near any boundary of the facet mask the window is truncated and
``g_c`` rises. Constants are those of
``Digital Twins/laminate_45deg_twins_measured_noise/config.py``. The damping
constant was calibrated on raw, untrimmed exports, so the observed facet mask of
a raw export is the right mask to evaluate it on.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

COMPONENTS = ("kxx", "kyy", "kxy")
FILTER_HALF_WIDTH_STEPS = 7
DAMPING_CONSTANT = 1.282
DAMPING_CONFIDENCE_INTERVAL = (1.240, 1.330)
# Median over facets at least 9 facets from any boundary (where s_c = 1).
INTERIOR_FLOOR_INV_MM = (4.801e-4, 5.082e-4, 5.152e-4)
# Median over the whole measured region, boundary amplification included.
REGION_MEDIAN_FLOOR_INV_MM = (5.278e-4, 5.388e-4, 5.590e-4)
LEGACY_ENVELOPE_INV_MM = (1.436e-3, 1.046e-3, 1.606e-3)
INTERIOR_DISTANCE_FACETS = 9.0
EDGE_DISTANCE_FACETS = 2.0
CALIBRATION_SPECIMENS = (
    "45_deg Laminate A1_PB, A1_Twist, A2_PB, A2_Twist, B1_PB, B1_Twist, B2_PB, B2_Twist",
    "45_deg_optimized Test_2, Test_3 (cutout coupons)",
)
MODEL_REFERENCE = (
    "Test_Design_Optimization_OLD/noise_audit_report_v2 (dic_curvature_noise.tex); "
    "implementation Digital Twins/laminate_45deg_twins_measured_noise "
    "(MEASURED_SEPARABLE)"
)

# Basis order of the local quadratic fit: 1, y, x, y^2, xy, x^2.
_COEFFICIENT_INDEX = {"kyy": 3, "kxy": 4, "kxx": 5}


def _window_basis(pitch_x_mm: float, pitch_y_mm: float, half: int) -> np.ndarray:
    steps = np.arange(-half, half + 1)
    grid_x, grid_y = np.meshgrid(steps * pitch_x_mm, steps * pitch_y_mm, indexing="xy")
    x = grid_x.ravel()
    y = grid_y.ravel()
    return np.stack([np.ones_like(x), y, x, y**2, x * y, x**2], axis=1)


def full_window_norm(component: str, pitch_x_mm: float, pitch_y_mm: float) -> float:
    basis = _window_basis(pitch_x_mm, pitch_y_mm, FILTER_HALF_WIDTH_STEPS)
    index = _COEFFICIENT_INDEX[component]
    return float(np.sqrt(np.linalg.inv(basis.T @ basis)[index, index]))


def local_stencil_norms(
    mask: np.ndarray, pitch_x_mm: float, pitch_y_mm: float
) -> dict[str, np.ndarray]:
    """Per-facet ``||w_c||`` for every component; NaN off the mask or when unfittable."""
    mask = np.asarray(mask, dtype=bool)
    half = FILTER_HALF_WIDTH_STEPS
    side = 2 * half + 1
    basis = _window_basis(pitch_x_mm, pitch_y_mm, half)
    n_terms = basis.shape[1]
    weights = mask.astype(float)
    rows, cols = np.nonzero(mask)
    normal = np.empty((rows.size, n_terms, n_terms))
    for p in range(n_terms):
        for q in range(p, n_terms):
            kernel = (basis[:, p] * basis[:, q]).reshape(side, side)
            value = ndimage.correlate(weights, kernel, mode="constant", cval=0.0)[rows, cols]
            normal[:, p, q] = value
            normal[:, q, p] = value
    counts = normal[:, 0, 0]
    usable = counts >= 2 * n_terms
    usable &= np.linalg.cond(normal) < 1e12
    inverse = np.full_like(normal, np.nan)
    if usable.any():
        inverse[usable] = np.linalg.inv(normal[usable])
    out: dict[str, np.ndarray] = {}
    for component in COMPONENTS:
        index = _COEFFICIENT_INDEX[component]
        variance = inverse[:, index, index]
        norms = np.full(mask.shape, np.nan)
        good = np.isfinite(variance) & (variance > 0.0)
        norms[rows[good], cols[good]] = np.sqrt(variance[good])
        out[component] = norms
    return out


def amplitude_maps(
    mask: np.ndarray, pitch_x_mm: float, pitch_y_mm: float, damping: float = DAMPING_CONSTANT
) -> dict[str, np.ndarray]:
    """``s_c(x)`` per component on ``mask``. Unfittable facets are held at 1."""
    norms = local_stencil_norms(mask, pitch_x_mm, pitch_y_mm)
    maps: dict[str, np.ndarray] = {}
    for component in COMPONENTS:
        ratio = norms[component] / full_window_norm(component, pitch_x_mm, pitch_y_mm)
        ratio = np.where(np.isfinite(ratio), ratio, 1.0)
        amplitude = 1.0 + (ratio - 1.0) / float(damping)
        maps[component] = np.where(mask, amplitude, np.nan)
    return maps


def model_sigma_maps(
    mask: np.ndarray, pitch_x_mm: float, pitch_y_mm: float
) -> dict[str, np.ndarray]:
    amplitudes = amplitude_maps(mask, pitch_x_mm, pitch_y_mm)
    return {
        component: INTERIOR_FLOOR_INV_MM[index] * amplitudes[component]
        for index, component in enumerate(COMPONENTS)
    }


def edge_distance(mask: np.ndarray) -> np.ndarray:
    """Facets to the nearest point outside the mask (0 on the boundary row)."""
    padded = np.pad(np.asarray(mask, dtype=bool), 1, constant_values=False)
    return ndimage.distance_transform_edt(padded)[1:-1, 1:-1] - 1.0
