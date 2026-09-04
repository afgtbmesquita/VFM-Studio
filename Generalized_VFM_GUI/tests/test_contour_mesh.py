"""Contour tessellation must follow tabbed / cruciform DIC clouds.

Unconstrained Delaunay fills the convex hull, which is exactly the streaked
κyy map the viewport showed for the biaxial plate.  Identification does not
use that triangulation: it maps each measurement to the element that contains
it and drops empty cells.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay

from vfmstudio.core.contour_mesh import (
    filtered_triangles,
    occupancy_sample_area,
    triangle_centroids,
    void_triangle_mask,
)


def _grid(x0, x1, y0, y1, pitch: float) -> np.ndarray:
    xs = np.arange(x0, x1 + 0.5 * pitch, pitch)
    ys = np.arange(y0, y1 + 0.5 * pitch, pitch)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()])


def _cruciform(pitch: float = 0.5) -> np.ndarray:
    """Plus-shaped cloud: inner square plus top and left tabs, like the plate."""
    square = _grid(2.0, 8.0, 2.0, 8.0, pitch)
    top = _grid(4.0, 6.0, 8.0 + pitch, 10.0, pitch)
    left = _grid(0.0, 2.0 - pitch, 4.0, 6.0, pitch)
    return np.vstack([square, top, left])


def _void_centroids(xy: np.ndarray, triangles: np.ndarray, pitch: float) -> np.ndarray:
    """Centroids that sit well inside the two concave corners of ``_cruciform``."""
    centers = triangle_centroids(xy[:, 0], xy[:, 1], triangles)
    if centers.size == 0:
        return centers
    inset = 0.75 * pitch
    top_left = (
        (centers[:, 0] > 0.0 + inset)
        & (centers[:, 0] < 2.0 - inset)
        & (centers[:, 1] > 8.0 + inset)
        & (centers[:, 1] < 10.0 - inset)
    )
    bottom_left = (
        (centers[:, 0] > 0.0 + inset)
        & (centers[:, 0] < 2.0 - inset)
        & (centers[:, 1] > 0.0 + inset)
        & (centers[:, 1] < 4.0 - inset)
    )
    return centers[top_left | bottom_left]


def test_raw_delaunay_fills_cruciform_voids():
    """Guard: the bug we are fixing is real on this geometry."""
    xy = _cruciform(0.5)
    raw = np.asarray(Delaunay(xy).simplices, dtype=np.int64)
    leaked = _void_centroids(xy, raw, 0.5)
    assert leaked.shape[0] > 0


def test_filtered_triangles_keep_cruciform_tabs():
    xy = _cruciform(0.5)
    faces = filtered_triangles(xy[:, 0], xy[:, 1])
    assert faces.shape[0] > 50
    leaked = _void_centroids(xy, faces, 0.5)
    assert leaked.shape[0] == 0

    used = np.unique(faces.ravel())
    top_tab = (xy[:, 1] >= 8.5) & (xy[:, 0] >= 4.0) & (xy[:, 0] <= 6.0)
    left_tab = (xy[:, 0] <= 1.5) & (xy[:, 1] >= 4.0) & (xy[:, 1] <= 6.0)
    assert np.any(np.isin(np.flatnonzero(top_tab), used))
    assert np.any(np.isin(np.flatnonzero(left_tab), used))


def test_filtered_triangles_survive_plate_rotation():
    """Exporter de-rotates by ~1.28°, which makes every x/y unique."""
    xy = _cruciform(0.5)
    phi = np.deg2rad(1.28)
    cosine, sine = np.cos(phi), np.sin(phi)
    origin = xy.mean(axis=0)
    centered = xy - origin
    rotated = np.column_stack(
        [
            cosine * centered[:, 0] - sine * centered[:, 1],
            sine * centered[:, 0] + cosine * centered[:, 1],
        ]
    ) + origin
    faces = filtered_triangles(rotated[:, 0], rotated[:, 1])
    assert faces.shape[0] > 50
    inverse = np.column_stack(
        [
            cosine * (rotated[:, 0] - origin[0]) + sine * (rotated[:, 1] - origin[1]),
            -sine * (rotated[:, 0] - origin[0]) + cosine * (rotated[:, 1] - origin[1]),
        ]
    ) + origin
    leaked = _void_centroids(inverse, faces, 0.5)
    assert leaked.shape[0] == 0


def test_l_notch_is_not_bridged():
    """The inside corner of an L is a hull triangle with grid-scale edges."""
    points = [
        (0.0, 0.0), (1.0, 0.0), (2.0, 0.0),
        (0.0, 1.0), (1.0, 1.0), (2.0, 1.0),
        (0.0, 2.0), (1.0, 2.0),
    ]
    xy = np.asarray(points, dtype=float)
    faces = filtered_triangles(xy[:, 0], xy[:, 1])
    centers = triangle_centroids(xy[:, 0], xy[:, 1], faces)
    in_missing_cell = (
        (centers[:, 0] > 1.05)
        & (centers[:, 0] < 1.95)
        & (centers[:, 1] > 1.05)
        & (centers[:, 1] < 1.95)
    )
    assert not np.any(in_missing_cell)


def test_filled_rectangle_keeps_grid_triangles():
    xy = _grid(0.0, 4.0, 0.0, 3.0, 0.5)
    faces = filtered_triangles(xy[:, 0], xy[:, 1])
    raw = np.asarray(Delaunay(xy).simplices, dtype=np.int64)
    mask = void_triangle_mask(xy[:, 0], xy[:, 1], raw)
    assert int(mask.sum()) == 0
    assert faces.shape[0] == raw.shape[0]


def test_identification_prunes_empty_tab_cells():
    """FGLS never integrates over the concave corners: empty cells are dropped."""
    from vfmstudio.core import bridge

    xy = _cruciform(0.5)
    MeshSettings = bridge.mesh_settings_class()
    Bounds = bridge.bounds_class()
    settings = MeshSettings(
        number_of_elements_x=5,
        number_of_elements_y=5,
        bounds=Bounds(x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0),
    )
    mesh = bridge.build_mesh(settings, "out_of_plane")
    n_before = int(mesh.n_elements)
    pruned, report = bridge.vfm_mesh().prune_mesh_by_measurement_points(mesh, xy)
    assert n_before == 25
    assert int(report.n_elements_removed) > 0
    assert int(pruned.n_elements) == n_before - int(report.n_elements_removed)

    counts = bridge.vfm_mesh().count_measurements_per_element(xy, pruned)
    assert int(counts.min()) >= 1
    mapping = bridge.vfm_mesh().map_points_to_mesh(xy, pruned)
    assert int(mapping.n_unmapped) == 0


def test_area_weight_uses_sample_area_not_bbox(tmp_path):
    """A bbox default would inflate every D_ij of the plus by ~1/fill-fraction."""
    from scipy.io import savemat

    from vfmstudio.core import bridge

    xy = _cruciform(0.5)
    bbox = (xy[:, 0].max() - xy[:, 0].min()) * (xy[:, 1].max() - xy[:, 1].min())
    true_area = float(xy.shape[0]) * 0.5 * 0.5
    assert true_area < 0.85 * bbox

    path = tmp_path / "plus.mat"
    savemat(
        path,
        {
            "nodes": xy,
            "SK1grid": np.zeros(xy.shape[0]),
            "SK2grid": np.zeros(xy.shape[0]),
            "SK3grid": np.zeros(xy.shape[0]),
            "F": np.array([1.0, 1.0]),
        },
    )
    with_bbox = bridge.load_point_measurements(path, "out_of_plane")
    with_area = bridge.load_point_measurements(
        path, "out_of_plane", total_sample_area=true_area
    )
    assert abs(with_bbox.area_per_point * xy.shape[0] - bbox) < 1.0e-9
    assert abs(with_area.area_per_point * xy.shape[0] - true_area) < 1.0e-9
    assert with_area.area_per_point < with_bbox.area_per_point


def test_occupancy_area_none_for_filled_rectangle():
    xy = _grid(0.0, 4.0, 0.0, 3.0, 0.5)
    assert occupancy_sample_area(xy[:, 0], xy[:, 1]) is None


def test_occupancy_area_for_cruciform():
    xy = _cruciform(0.5)
    area = occupancy_sample_area(xy[:, 0], xy[:, 1])
    bbox = (xy[:, 0].max() - xy[:, 0].min()) * (xy[:, 1].max() - xy[:, 1].min())
    assert area is not None
    assert area < 0.85 * bbox
    assert abs(area - float(xy.shape[0]) * 0.5 * 0.5) < 1.0e-9


def test_importing_plus_mat_sets_sample_area(tmp_path):
    from scipy.io import savemat

    from vfmstudio.core.document import ExperimentModel
    from vfmstudio.core.fields import sync_mesh_bounds_from_file

    xy = _cruciform(0.5)
    path = tmp_path / "plus.mat"
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
    assert experiment.sample_area_total is None
    assert sync_mesh_bounds_from_file(experiment, path, analysis_type="out_of_plane")
    expected = float(xy.shape[0]) * 0.5 * 0.5
    assert experiment.sample_area_total is not None
    assert abs(float(experiment.sample_area_total) - expected) < 1.0e-9
    preset = 1234.5
    experiment.sample_area_total = preset
    sync_mesh_bounds_from_file(
        experiment, path, analysis_type="out_of_plane", replace=True
    )
    assert experiment.sample_area_total == preset
