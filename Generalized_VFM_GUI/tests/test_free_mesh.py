"""Free-mesh clustering, conformity, and label placement."""

from __future__ import annotations

import pytest

from vfmstudio.core.free_mesh import (
    CONFORM_MESSAGE,
    ConformError,
    construct_free_quad_mesh,
    inspect_free_points,
    offset_label_display,
)


def test_complete_2x2_grid():
    points = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
    ]
    status = construct_free_quad_mesh(points)
    assert status.ok
    assert status.n_quads == 1
    assert status.n_elements_x == 1
    assert status.n_elements_y == 1
    assert status.keep_cells == [(0, 0)]


def test_l_shape_eight_of_nine_nodes():
    points = [
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (2.0, 1.0),
        (0.0, 2.0),
        (1.0, 2.0),
    ]
    status = construct_free_quad_mesh(points)
    assert status.ok
    assert status.n_quads == 3
    assert status.n_points == 8
    assert (1, 1) not in status.keep_cells
    assert set(status.keep_cells) == {(0, 0), (1, 0), (0, 1)}


def test_hanging_mid_edge_point_fails():
    points = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (0.5, 1.0),
    ]
    status = inspect_free_points(points)
    assert not status.ok
    assert status.n_quads == 0
    with pytest.raises(ConformError, match="not conform"):
        construct_free_quad_mesh(points)


def test_three_points_fail():
    with pytest.raises(ConformError) as exc:
        construct_free_quad_mesh([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])
    assert str(exc.value) == CONFORM_MESSAGE


def test_near_equal_x_values_cluster():
    points = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0000008, 1.0),
    ]
    status = construct_free_quad_mesh(points)
    assert status.ok
    assert status.n_quads == 1
    assert len(status.x_nodes) == 2


def test_label_offset_stays_inside_viewport():
    width, height = 200.0, 100.0
    margin = 16.0
    samples = [
        (10.0, 10.0),
        (190.0, 90.0),
        (100.0, 50.0),
        (5.0, 95.0),
        (195.0, 5.0),
    ]
    for x, y in samples:
        lx, ly = offset_label_display(x, y, width, height, offset=12.0, margin=margin)
        assert margin <= lx <= width - margin
        assert margin <= ly <= height - margin
        assert abs(lx - x) > 0.5 or abs(ly - y) > 0.5


def test_l_shape_builds_subset_mesh():
    from vfmstudio.core.document import (
        BoundsModel,
        ExperimentModel,
        MeshModel,
        empty_document,
    )
    from vfmstudio.core import mesh_convert

    points = [
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (2.0, 1.0),
        (0.0, 2.0),
        (1.0, 2.0),
    ]
    status = construct_free_quad_mesh(points)
    document = empty_document()
    document.experiments.append(
        ExperimentModel(
            experiment_id=1,
            mesh=MeshModel(
                n_elements_x=status.n_elements_x,
                n_elements_y=status.n_elements_y,
                bounds=BoundsModel(
                    x_min=status.x_nodes[0],
                    x_max=status.x_nodes[-1],
                    y_min=status.y_nodes[0],
                    y_max=status.y_nodes[-1],
                ),
                x_nodes=list(status.x_nodes),
                y_nodes=list(status.y_nodes),
                free_nodes=[[x, y] for x, y in status.snapped_points],
                keep_cells=[[ix, iy] for ix, iy in status.keep_cells],
            ),
        )
    )
    mesh_pkg = mesh_convert.build_experiment_mesh(document, document.experiments[0])
    assert int(mesh_pkg.n_elements) == 3
    assert int(mesh_pkg.n_nodes) == 8
