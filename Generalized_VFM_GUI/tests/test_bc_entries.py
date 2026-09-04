"""Add-BC dialog: edge vs nodes, one tree entry per DOF."""

from __future__ import annotations

import os

import numpy as np
import pytest
from scipy.io import savemat

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from vfmstudio.core.bc_entries import (
    bc_label,
    make_boundary_entries,
    ordered_dofs,
)
from vfmstudio.core.document import (
    BoundsModel,
    BoundaryEntry,
    ExperimentModel,
    LoadStepModel,
    MeshModel,
    empty_document,
)
from vfmstudio.core import mesh_convert


def test_ordered_dofs_kirchhoff_sequence():
    assert ordered_dofs(["W_XY", "W", "W_Y", "W", "W_X"]) == ["W", "W_X", "W_Y", "W_XY"]


def test_make_entries_default_is_one_edge_bc():
    entries = make_boundary_entries("prescribed", ["W"], edge="top")
    assert len(entries) == 1
    assert entries[0].dof == "W"
    assert entries[0].edge == "top"
    assert entries[0].nodes is None
    assert entries[0].value == 0.0
    assert entries[0].label == "W @top"


def test_make_entries_expands_multiple_dofs():
    entries = make_boundary_entries(
        "prescribed",
        ["W_XY", "W", "W_Y", "W_X"],
        edge="bottom",
        label="bottom junction",
    )
    assert [item.dof for item in entries] == ["W", "W_X", "W_Y", "W_XY"]
    assert all(item.edge == "bottom" for item in entries)
    assert all(item.nodes is None for item in entries)
    assert entries[0].label == "bottom junction  W"
    assert entries[3].label == "bottom junction  W_XY"


def test_make_entries_nodes_have_no_edge():
    entries = make_boundary_entries(
        "prescribed", ["W", "W_X"], nodes=[16, 24, 16, -1]
    )
    assert len(entries) == 2
    assert all(item.edge is None for item in entries)
    assert entries[0].nodes == [16, 24]
    assert entries[1].nodes == [16, 24]
    assert "16" in entries[0].label


def test_make_entries_external_keeps_load_type():
    entries = make_boundary_entries(
        "external", ["W_Y"], edge="top", load_type="concentrated"
    )
    assert entries[0].load_type == "concentrated"
    assert entries[0].sort_direction == "x"
    assert entries[0].value is None


def test_make_entries_rejects_empty_choice():
    with pytest.raises(ValueError):
        make_boundary_entries("prescribed", [], edge="top")
    with pytest.raises(ValueError):
        make_boundary_entries("prescribed", ["W"])


def test_bc_label_prefix_and_edge():
    assert bc_label("W", edge="left") == "W @left"
    assert bc_label("W_X", prefix="encastre") == "encastre  W_X"


def test_bc_anchor_prefers_named_edge():
    document = empty_document()
    experiment = ExperimentModel(
        mesh=MeshModel(
            n_elements_x=2,
            n_elements_y=2,
            bounds=BoundsModel(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0),
        )
    )
    mesh = mesh_convert.build_experiment_mesh(document, experiment)
    edge_pts = mesh_convert.bc_anchor_xyz(
        mesh, BoundaryEntry(dof="W", edge="left", nodes=[0])
    )
    left = mesh_convert.edge_node_xyz(mesh, "left")
    assert edge_pts.shape[0] == left.shape[0]
    node_pts = mesh_convert.bc_anchor_xyz(
        mesh, BoundaryEntry(dof="W", nodes=[0])
    )
    assert node_pts.shape[0] == 1
    assert np.allclose(node_pts[0, :2], mesh_convert.mesh_xyz(mesh)[0, :2])


def test_build_mesh_without_dic_is_unpruned():
    document = empty_document()
    experiment = ExperimentModel(
        mesh=MeshModel(
            n_elements_x=2,
            n_elements_y=2,
            bounds=BoundsModel(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0),
        )
    )
    mesh = mesh_convert.build_experiment_mesh(document, experiment)
    assert int(mesh.n_elements) == 4


def test_build_mesh_prunes_plus_dic(tmp_path):
    xs, ys = [], []
    for x in np.arange(0.0, 10.1, 0.5):
        for y in np.arange(0.0, 10.1, 0.5):
            if (2 <= x <= 8 and 2 <= y <= 8) or (4 <= x <= 6 and y >= 8) or (
                x <= 2 and 4 <= y <= 6
            ):
                xs.append(x)
                ys.append(y)
    xy = np.column_stack([xs, ys])
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
    document = empty_document()
    experiment = ExperimentModel(
        mesh=MeshModel(
            n_elements_x=5,
            n_elements_y=5,
            bounds=BoundsModel(x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0),
        ),
        steps=[LoadStepModel(data_file=path)],
    )
    mesh = mesh_convert.build_experiment_mesh(document, experiment)
    assert int(mesh.n_elements) < 25
    assert int(mesh.n_elements) >= 8
