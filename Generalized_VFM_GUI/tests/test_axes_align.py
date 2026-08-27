"""Align a tilted DIC cloud with the nearest lab X/Y axis."""

from __future__ import annotations

import numpy as np

from vfmstudio.core.axes_align import (
    alignment_rotation_deg,
    apply_alignment_to_measurements,
    engineering_rotation_matrix,
    nearest_axis_correction_deg,
    rotate_engineering_fields,
    rotate_xy,
)


def _tilted_strip(angle_deg: float, n: int = 400):
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, n)
    y = rng.uniform(-12.0, 12.0, n)
    return rotate_xy(x, y, angle_deg, origin=(0.0, 0.0))


def test_nearest_axis_picks_smallest_rotation():
    assert abs(nearest_axis_correction_deg(87.0) - 3.0) < 1.0e-9
    assert abs(nearest_axis_correction_deg(93.0) + 3.0) < 1.0e-9
    assert abs(nearest_axis_correction_deg(3.0) + 3.0) < 1.0e-9
    assert abs(nearest_axis_correction_deg(-3.0) - 3.0) < 1.0e-9


def test_alignment_undoes_small_y_tilt():
    x, y = _tilted_strip(3.0)
    correction = alignment_rotation_deg(x, y)
    assert abs(correction + 3.0) < 0.4
    xr, yr = rotate_xy(x, y, correction, origin=(0.0, 0.0))
    leftover = alignment_rotation_deg(xr, yr)
    assert abs(leftover) < 0.2


def test_aligned_aabb_is_tighter():
    x, y = _tilted_strip(3.0)
    tilted_area = (np.ptp(x) * np.ptp(y))
    xr, yr = rotate_xy(x, y, alignment_rotation_deg(x, y), origin=(0.0, 0.0))
    aligned_area = (np.ptp(xr) * np.ptp(yr))
    assert aligned_area < tilted_area


def test_engineering_T_inverts():
    transform = engineering_rotation_matrix(3.0)
    inverse = engineering_rotation_matrix(-3.0)
    assert np.allclose(inverse @ transform, np.eye(3), atol=1.0e-12)


def test_curvature_rotates_with_the_cloud():
    kappa = np.vstack(
        [np.zeros(8), np.ones(8), np.zeros(8)]
    )
    rotated = rotate_engineering_fields(kappa, 3.0)
    restored = rotate_engineering_fields(rotated, -3.0)
    assert np.allclose(restored, kappa, atol=1.0e-12)
    assert not np.allclose(rotated, kappa)


class _Measurements:
    def __init__(self, x, y, strains):
        self.points = np.column_stack([x, y])
        self.strains = strains
        self.data_bounds = {}


def test_apply_alignment_updates_points_and_bounds():
    x, y = _tilted_strip(3.0)
    strains = np.vstack([np.zeros(x.size), np.ones(x.size), np.zeros(x.size)])
    measurements = _Measurements(x, y, strains)
    angle = alignment_rotation_deg(x, y)
    apply_alignment_to_measurements(measurements, angle)
    leftover = alignment_rotation_deg(
        measurements.points[:, 0], measurements.points[:, 1]
    )
    assert abs(leftover) < 0.2
    assert measurements.data_bounds["x_min"] == float(np.min(measurements.points[:, 0]))
