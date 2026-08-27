"""Per-element DIC counts use the same mapping as VFM assembly."""

from __future__ import annotations

import numpy as np

from vfmstudio.core import mesh_convert
from vfmstudio.core.document import BoundsModel, ExperimentModel, MeshModel, empty_document


def _unit_quad_experiment():
    document = empty_document()
    document.material.analysis_type = "out_of_plane"
    experiment = ExperimentModel(
        experiment_id=1,
        mesh=MeshModel(
            n_elements_x=2,
            n_elements_y=2,
            bounds=BoundsModel(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0),
        ),
    )
    return document, experiment


def test_element_centroids_on_unit_quads():
    document, experiment = _unit_quad_experiment()
    mesh_pkg = mesh_convert.build_experiment_mesh(document, experiment)
    centroids = mesh_convert.element_centroids(mesh_pkg)
    assert centroids.shape == (4, 3)
    xs = np.sort(np.unique(np.round(centroids[:, 0], 6)))
    ys = np.sort(np.unique(np.round(centroids[:, 1], 6)))
    assert np.allclose(xs, [0.25, 0.75])
    assert np.allclose(ys, [0.25, 0.75])


def test_count_dic_points_per_element():
    document, experiment = _unit_quad_experiment()
    mesh_pkg = mesh_convert.build_experiment_mesh(document, experiment)
    points = np.array(
        [
            [0.25, 0.25],
            [0.25, 0.25],
            [0.75, 0.25],
            [0.25, 0.75],
            [0.25, 0.75],
            [0.25, 0.75],
            [2.0, 2.0],
        ]
    )
    counts = mesh_convert.count_dic_points_per_element(mesh_pkg, points)
    assert counts.shape == (4,)
    assert int(counts.sum()) == 6
    assert int((counts == 0).sum()) == 1
    assert int(counts.max()) == 3
    summary = mesh_convert.summarize_element_point_counts(counts)
    assert "4 elements" in summary
    assert "1 empty" in summary
    assert "6 DIC points mapped" in summary
