"""Drag-mesh grid-line picking, clamping, and MeshModel updates."""

from __future__ import annotations

import numpy as np

from vfmstudio.core.document import BoundsModel, MeshModel
from vfmstudio.core.drag_mesh import (
    apply_grid_line_drag,
    axis_nodes_from_mesh,
    choose_drag_axis,
    clamp_axis_value,
    dic_axis_targets,
    element_edge_records,
    grid_line_segments,
    materialize_axis_nodes,
    nodes_on_grid_line,
    pick_drag_target,
    restore_mesh_axes,
    snapshot_mesh_axes,
)


def _square_mesh():
    xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
        ],
        dtype=float,
    )
    quads = np.array(
        [
            [0, 1, 4, 3],
            [1, 2, 5, 4],
            [3, 4, 7, 6],
            [4, 5, 8, 7],
        ],
        dtype=np.int64,
    )
    xs = [0.0, 1.0, 2.0]
    ys = [0.0, 1.0, 2.0]
    return xyz, quads, xs, ys


def test_materialize_uniform_grid_from_bounds():
    mesh = MeshModel(n_elements_x=2, n_elements_y=3, bounds=BoundsModel(x_max=4.0, y_max=6.0))
    assert mesh.x_nodes is None
    changed = materialize_axis_nodes(mesh)
    assert changed
    assert mesh.x_nodes == [0.0, 2.0, 4.0]
    assert mesh.y_nodes == [0.0, 2.0, 4.0, 6.0]


def test_clamp_interior_cannot_cross_neighbors():
    nodes = [0.0, 1.0, 2.0]
    assert clamp_axis_value(nodes, 1, 1.4) == 1.4
    moved = clamp_axis_value(nodes, 1, 2.0)
    assert moved < 2.0
    assert moved > 1.0
    outer = clamp_axis_value(nodes, 0, -3.5)
    assert outer == -3.5


def test_drag_right_edge_updates_bounds_and_keeps_cells():
    mesh = MeshModel(
        n_elements_x=2,
        n_elements_y=2,
        bounds=BoundsModel(x_max=2.0, y_max=2.0),
        x_nodes=[0.0, 1.0, 2.0],
        y_nodes=[0.0, 1.0, 2.0],
        keep_cells=[[0, 0], [1, 0], [0, 1]],
        free_nodes=[[2.0, 0.0], [2.0, 1.0], [0.0, 0.0]],
    )
    new = apply_grid_line_drag(mesh, "x", 2, 2.7)
    assert new == 2.7
    assert mesh.x_nodes[-1] == 2.7
    assert mesh.bounds.x_max == 2.7
    assert mesh.y_nodes == [0.0, 1.0, 2.0]
    assert mesh.keep_cells == [[0, 0], [1, 0], [0, 1]]
    assert mesh.free_nodes[0][0] == 2.7
    assert mesh.free_nodes[1][0] == 2.7
    assert mesh.free_nodes[2][0] == 0.0


def test_l_shape_vertical_line_uses_connecting_element_edges():
    xyz, _quads, xs, ys = _square_mesh()
    keep = np.array([[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6]], dtype=np.int64)
    segs = grid_line_segments(xyz, keep, xs, ys, "x", 2)
    assert segs.shape[0] == 1
    endpoints = {tuple(np.round(p, 6)) for seg in segs for p in seg}
    assert endpoints == {(2.0, 0.0), (2.0, 1.0)}
    used = xyz[[0, 1, 2, 3, 4, 5, 6, 7]]
    ids = nodes_on_grid_line(used, xs, ys, "x", 2)
    assert set(ids.tolist()) == {2, 5}


def test_pick_prefers_node_then_edge():
    xyz, quads, xs, ys = _square_mesh()
    node_display = xyz[:, :2]
    edges, axes = element_edge_records(xyz, quads, xs, ys)
    edge_display = edges.copy()
    corner = pick_drag_target([0.0, 0.0], node_display, xyz, edge_display, axes, xs, ys)
    assert corner is not None
    assert corner.node_id == 0
    assert corner.axis is None
    mid_right = pick_drag_target(
        [2.0, 0.5],
        node_display,
        xyz,
        edge_display,
        axes,
        xs,
        ys,
        pixel_tol=0.2,
    )
    assert mid_right is not None
    assert mid_right.axis == "x"
    assert mid_right.ix == 2
    miss = pick_drag_target([50.0, 50.0], node_display, xyz, edge_display, axes, xs, ys)
    assert miss is None


def test_choose_drag_axis_prefers_outward_edge():
    assert choose_drag_axis(2, 1, 3, 3, 12.0, 1.0) == "x"
    assert choose_drag_axis(1, 2, 3, 3, 1.0, 12.0) == "y"
    assert choose_drag_axis(1, 1, 3, 3, 1.0, 1.0) is None
    assert choose_drag_axis(1, 1, 3, 3, 12.0, 3.0) == "x"
    assert choose_drag_axis(1, 1, 3, 3, 3.0, 12.0, preferred="y") == "y"


def test_snapshot_restore_round_trip():
    mesh = MeshModel(
        n_elements_x=2,
        n_elements_y=1,
        bounds=BoundsModel(x_max=2.0, y_max=1.0),
        x_nodes=[0.0, 1.0, 2.0],
        y_nodes=[0.0, 1.0],
    )
    snap = snapshot_mesh_axes(mesh)
    apply_grid_line_drag(mesh, "x", 2, 3.0)
    assert mesh.bounds.x_max == 3.0
    restore_mesh_axes(
        mesh,
        snap["x_nodes"],
        snap["y_nodes"],
        snap["bounds"],
        snap["free_nodes"],
        n_elements_x=snap["n_elements_x"],
        n_elements_y=snap["n_elements_y"],
    )
    assert mesh.x_nodes == [0.0, 1.0, 2.0]
    assert mesh.bounds.x_max == 2.0


def test_dic_axis_targets():
    cloud = np.array([[0.2, 1.0], [8.1, 31.4], [0.2, 31.4]], dtype=float)
    assert dic_axis_targets(cloud, "x") == [0.2, 8.1]
    assert dic_axis_targets(cloud, "y") == [1.0, 31.4]


def test_axis_nodes_from_existing_lists():
    mesh = MeshModel(
        n_elements_x=4,
        n_elements_y=4,
        x_nodes=[0.0, 0.4, 1.0],
        y_nodes=[0.0, 2.0],
    )
    xs, ys = axis_nodes_from_mesh(mesh)
    assert xs == [0.0, 0.4, 1.0]
    assert ys == [0.0, 2.0]
