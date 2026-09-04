"""DIC sample-area estimate: Delaunay cell area times point count.

VFM integrates with weight A/n.  On a facet lattice that is n·dx·dy, including
non-convex outlines and holes (missing points simply lower n).  Shapely
concave_hull and alpha-shape polygon area omit the boundary half-cell and
are sensitive to a global ratio/alpha; see the module docstring of
``vfmstudio.core.sample_area``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from vfmstudio.core.sample_area import estimate_sample_area


def _grid(x0, x1, y0, y1, dx: float, dy: Optional[float] = None) -> np.ndarray:
    dy = dx if dy is None else dy
    xs = np.arange(x0, x1 + 0.5 * dx, dx)
    ys = np.arange(y0, y1 + 0.5 * dy, dy)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()])


def _cruciform(pitch: float = 0.5) -> np.ndarray:
    square = _grid(2.0, 8.0, 2.0, 8.0, pitch)
    top = _grid(4.0, 6.0, 8.0 + pitch, 10.0, pitch)
    left = _grid(0.0, 2.0 - pitch, 4.0, 6.0, pitch)
    return np.vstack([square, top, left])


def _assert_n_cell(xy: np.ndarray, dx: float, dy: Optional[float] = None) -> None:
    dy = dx if dy is None else dy
    area = estimate_sample_area(xy[:, 0], xy[:, 1])
    expected = float(xy.shape[0]) * dx * dy
    assert area is not None
    assert abs(float(area) - expected) < 1.0e-9 * max(1.0, expected)


def test_filled_rectangle_is_n_dx_dy():
    xy = _grid(0.0, 10.0, 0.0, 6.0, 0.5)
    _assert_n_cell(xy, 0.5)
    bbox = (xy[:, 0].max() - xy[:, 0].min()) * (xy[:, 1].max() - xy[:, 1].min())
    area = estimate_sample_area(xy[:, 0], xy[:, 1])
    assert area is not None
    assert float(area) > bbox


def test_anisotropic_pitch_uses_dx_times_dy_not_h_squared():
    dx, dy = 0.4, 0.8
    xy = _grid(0.0, 10.0, 0.0, 6.4, dx, dy)
    _assert_n_cell(xy, dx, dy)
    h_sq = float(xy.shape[0]) * dx * dx
    area = estimate_sample_area(xy[:, 0], xy[:, 1])
    assert area is not None
    assert abs(float(area) - h_sq) > 0.25 * h_sq


def test_cruciform_matches_facet_count():
    xy = _cruciform(0.5)
    _assert_n_cell(xy, 0.5)
    bbox = (xy[:, 0].max() - xy[:, 0].min()) * (xy[:, 1].max() - xy[:, 1].min())
    area = estimate_sample_area(xy[:, 0], xy[:, 1])
    assert area is not None
    assert float(area) < 0.85 * bbox


def test_l_shape_is_not_the_bbox():
    pitch = 0.4
    bar_h = _grid(0.0, 8.0, 0.0, 3.0, pitch)
    bar_v = _grid(0.0, 3.0, 3.0 + pitch, 8.0, pitch)
    xy = np.vstack([bar_h, bar_v])
    _assert_n_cell(xy, pitch)
    bbox = (xy[:, 0].max() - xy[:, 0].min()) * (xy[:, 1].max() - xy[:, 1].min())
    area = estimate_sample_area(xy[:, 0], xy[:, 1])
    assert area is not None
    assert float(area) < 0.9 * bbox


def test_c_shape_narrow_concavity():
    pitch = 0.4
    xy = _grid(0.0, 10.0, 0.0, 8.0, pitch)
    notch = (
        (xy[:, 0] >= 3.0)
        & (xy[:, 1] >= 2.0)
        & (xy[:, 1] <= 6.0)
    )
    xy = xy[~notch]
    _assert_n_cell(xy, pitch)


def test_rectangle_with_square_hole():
    pitch = 0.4
    xy = _grid(0.0, 12.0, 0.0, 8.0, pitch)
    hole = (
        (xy[:, 0] > 4.0)
        & (xy[:, 0] < 8.0)
        & (xy[:, 1] > 2.5)
        & (xy[:, 1] < 5.5)
    )
    filled = xy.shape[0]
    xy = xy[~hole]
    assert xy.shape[0] < filled
    _assert_n_cell(xy, pitch)
    filled_area = float(filled) * pitch * pitch
    area = estimate_sample_area(xy[:, 0], xy[:, 1])
    assert area is not None
    assert float(area) < filled_area


def test_rectangle_with_two_holes():
    pitch = 0.35
    xy = _grid(0.0, 14.0, 0.0, 8.0, pitch)
    hole_a = (xy[:, 0] > 2.0) & (xy[:, 0] < 4.5) & (xy[:, 1] > 2.0) & (xy[:, 1] < 6.0)
    hole_b = (xy[:, 0] > 8.5) & (xy[:, 0] < 11.5) & (xy[:, 1] > 3.0) & (xy[:, 1] < 5.5)
    xy = xy[~(hole_a | hole_b)]
    _assert_n_cell(xy, pitch)


def test_rotated_cruciform_is_rotation_invariant():
    xy = _cruciform(0.4)
    expected = float(xy.shape[0]) * 0.4 * 0.4
    phi = np.deg2rad(17.0)
    cosine, sine = np.cos(phi), np.sin(phi)
    origin = xy.mean(axis=0)
    centered = xy - origin
    rotated = np.column_stack(
        [
            cosine * centered[:, 0] - sine * centered[:, 1],
            sine * centered[:, 0] + cosine * centered[:, 1],
        ]
    ) + origin
    area = estimate_sample_area(rotated[:, 0], rotated[:, 1])
    assert area is not None
    assert abs(float(area) - expected) < 1.0e-6 * expected


def test_cruciform_with_centre_hole():
    pitch = 0.4
    xy = _cruciform(pitch)
    hole = (
        (xy[:, 0] > 4.5)
        & (xy[:, 0] < 5.5)
        & (xy[:, 1] > 4.5)
        & (xy[:, 1] < 5.5)
    )
    xy = xy[~hole]
    _assert_n_cell(xy, pitch)


def test_importing_rectangle_mat_sets_sample_area(tmp_path):
    from scipy.io import savemat

    from vfmstudio.core.document import ExperimentModel
    from vfmstudio.core.fields import estimate_experiment_sample_area, sync_mesh_bounds_from_file

    xy = _grid(0.0, 4.0, 0.0, 3.0, 0.5)
    path = tmp_path / "rect.mat"
    savemat(
        path,
        {
            "nodes": xy,
            "SK1grid": np.zeros(xy.shape[0]),
            "SK2grid": np.zeros(xy.shape[0]),
            "SK3grid": np.zeros(xy.shape[0]),
        },
    )
    experiment = ExperimentModel()
    assert sync_mesh_bounds_from_file(experiment, path, analysis_type="out_of_plane")
    expected = float(xy.shape[0]) * 0.5 * 0.5
    assert experiment.sample_area_total is not None
    assert abs(float(experiment.sample_area_total) - expected) < 1.0e-9
    experiment.sample_area_total = 99.0
    estimate_experiment_sample_area(
        experiment, path, analysis_type="out_of_plane", replace=False
    )
    assert experiment.sample_area_total == 99.0
    estimate_experiment_sample_area(
        experiment, path, analysis_type="out_of_plane", replace=True
    )
    assert abs(float(experiment.sample_area_total) - expected) < 1.0e-9
