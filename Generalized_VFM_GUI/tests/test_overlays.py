"""Kirchhoff BC glyph orientation and 2-D specimen symbols."""

from __future__ import annotations

import numpy as np

from vfmstudio.core.document import BoundaryEntry, ExperimentModel
from vfmstudio.gui.viewport.overlays import (
    PRESCRIBED_CONE_SCALE,
    add_bc_symbols_2d,
    display_axis,
    dof_direction,
    is_angular_dof,
    moment_arc_points,
    moment_label_anchor,
    moment_plane_normal,
    plane_glyph_direction,
    set_mesh_points,
)


def test_kirchhoff_dof_axes():
    assert np.allclose(dof_direction("W"), [0.0, 0.0, 1.0])
    assert np.allclose(dof_direction("U"), [1.0, 0.0, 0.0])
    assert np.allclose(dof_direction("V"), [0.0, 1.0, 0.0])
    assert np.allclose(dof_direction("W_X"), [1.0, 0.0, 0.0])
    assert np.allclose(dof_direction("W_Y"), [0.0, 1.0, 0.0])
    assert np.allclose(dof_direction("W_XY"), [0.0, 0.0, 1.0])


def test_angular_dofs():
    assert is_angular_dof("W_X")
    assert is_angular_dof("W_Y")
    assert is_angular_dof("w_xy")
    assert not is_angular_dof("W")
    assert not is_angular_dof("U")


def test_prescribed_cone_scale_is_half():
    assert PRESCRIBED_CONE_SCALE == 0.5


def test_moment_plane_wx_is_zx():
    n = moment_plane_normal("W_X")
    assert np.allclose(n, [1.0, 0.0, 0.0])


def test_moment_plane_wy_is_zy():
    n = moment_plane_normal("W_Y")
    assert np.allclose(n, [0.0, 1.0, 0.0])


def test_moment_plane_wxy_follows_edge_tangent():
    assert np.allclose(moment_plane_normal("W_XY", edge="top"), [1.0, 0.0, 0.0])
    assert np.allclose(moment_plane_normal("W_XY", edge="bottom"), [1.0, 0.0, 0.0])
    assert np.allclose(moment_plane_normal("W_XY", edge="left"), [0.0, 1.0, 0.0])
    assert np.allclose(moment_plane_normal("W_XY", edge="right"), [0.0, 1.0, 0.0])


def test_wx_on_top_is_still_zx_twist_about_y():
    """Top-edge W_X is M_xy, but the rotation is still ∂w/∂x (Z–X arc)."""
    n = moment_plane_normal("W_X", edge="top")
    assert np.allclose(n, [1.0, 0.0, 0.0])


def test_moment_arc_wx_lies_in_zx_above_plate():
    origin = np.array([2.0, 3.0, 0.0])
    pts = moment_arc_points(origin, moment_plane_normal("W_X"), radius=1.0)
    assert pts.shape[0] >= 3
    assert np.all(pts[:, 2] >= -1.0e-12)
    assert np.max(np.abs(pts[:, 1] - origin[1])) < 1.0e-12
    # Arrowhead at +n (positive x); crest takes +Z toward +n.
    assert pts[-1, 0] > pts[0, 0]
    assert pts[-1, 0] > origin[0]
    crest = pts[int(np.argmax(pts[:, 2]))]
    assert crest[2] > 0.5


def test_moment_arc_wy_lies_in_zy_above_plate():
    origin = np.array([2.0, 3.0, 0.0])
    pts = moment_arc_points(origin, moment_plane_normal("W_Y"), radius=1.0)
    assert np.all(pts[:, 2] >= -1.0e-12)
    assert np.max(np.abs(pts[:, 0] - origin[0])) < 1.0e-12
    assert pts[-1, 1] > pts[0, 1]
    assert pts[-1, 1] > origin[1]


def test_moment_label_clears_myy_arc_plane():
    """Top-edge Myy: inward is −Y, which is the arc plane. Label must leave it."""
    origin = np.array([4.0, 32.0, 0.0])
    center = np.array([4.0, 16.0, 0.0])
    radius = 1.0
    n = moment_plane_normal("W_Y", "top")
    arc = moment_arc_points(origin, n, radius)
    anchor = moment_label_anchor(origin, n, radius, specimen_center=center)
    assert abs(anchor[0] - origin[0]) > 0.4
    assert anchor[2] > float(np.max(arc[:, 2]))


def test_moment_label_clears_mxy_arc_plane():
    """Top-edge Mxy lives in Z–X; offset toward the specimen, above the crest."""
    origin = np.array([4.0, 32.0, 0.0])
    center = np.array([4.0, 16.0, 0.0])
    radius = 1.0
    n = moment_plane_normal("W_X", "top")
    arc = moment_arc_points(origin, n, radius)
    anchor = moment_label_anchor(origin, n, radius, specimen_center=center)
    assert abs(anchor[1] - origin[1]) > 0.4
    assert anchor[1] < origin[1]
    assert anchor[2] > float(np.max(arc[:, 2]))


def test_mesh_wireframe_is_dark():
    from vfmstudio.gui.viewport.overlays import MESH_EDGE_COLOR, MESH_EDGE_WIDTH

    hexcolor = MESH_EDGE_COLOR.lstrip("#")
    red, green, blue = (int(hexcolor[i : i + 2], 16) for i in (0, 2, 4))
    assert (red + green + blue) / 3.0 < 80.0
    assert MESH_EDGE_WIDTH >= 3.0


def test_in_plane_axis_points_inward():
    top_inward = np.array([0.0, -1.0, 0.0])
    axis = display_axis("W_Y", top_inward)
    assert axis[1] < 0.0
    assert abs(axis[0]) < 1.0e-12
    assert abs(axis[2]) < 1.0e-12


def test_z_constraint_stays_along_z():
    axis = display_axis("W", np.array([0.0, -1.0, 0.0]))
    assert np.allclose(axis, [0.0, 0.0, 1.0])
    plane = plane_glyph_direction(axis)
    assert abs(plane[1] / plane[0]) != 0.0
    # Must not collapse onto specimen X or Y.
    assert abs(plane[0]) > 1.0e-9 and abs(plane[1]) > 1.0e-9


def test_bc_symbols_2d_draw_pins_and_moments():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrow, PathPatch, Polygon

    figure, axis = plt.subplots()
    axis.set_xlim(0.0, 10.0)
    axis.set_ylim(0.0, 10.0)
    experiment = ExperimentModel(
        prescribed=[
            BoundaryEntry(label="U", dof="U", edge="left"),
            BoundaryEntry(label="W", dof="W", edge="bottom"),
            BoundaryEntry(label="RX", dof="W_X", edge="right"),
        ],
        external=[
            BoundaryEntry.model_validate(
                {
                    "label": "M_yy",
                    "dof": "W_Y",
                    "edge": "top",
                    "loadType": "distributed",
                }
            )
        ],
    )
    add_bc_symbols_2d(axis, experiment, 0.0, 10.0, 0.0, 10.0)
    polygons = [patch for patch in axis.patches if isinstance(patch, Polygon)]
    arrows = [patch for patch in axis.patches if isinstance(patch, FancyArrow)]
    arcs = [patch for patch in axis.patches if isinstance(patch, PathPatch)]
    assert len(polygons) >= 5  # X pin + Z pyramid faces + double cone + moment heads
    assert len(arrows) == 0  # moments are n–Z arcs, not Abaqus double heads
    assert len(arcs) == 5  # 5 edge samples × 1 isometric moment arc
    labels = [text.get_text() for text in axis.texts]
    assert "Myy" in labels
    plt.close(figure)


def test_bc_symbols_2d_draws_node_only_prescribed():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    figure, axis = plt.subplots()
    axis.set_xlim(0.0, 10.0)
    axis.set_ylim(0.0, 10.0)
    experiment = ExperimentModel(
        prescribed=[BoundaryEntry(label="W @0, 2", dof="W", nodes=[0, 2])]
    )
    cloud = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [10.0, 10.0, 0.0]]
    )
    add_bc_symbols_2d(axis, experiment, 0.0, 10.0, 0.0, 10.0, node_xyz=cloud)
    polygons = [patch for patch in axis.patches if isinstance(patch, Polygon)]
    assert len(polygons) == 4  # two nodes × two-face Z pyramid
    plt.close(figure)


def test_bc_symbols_2d_skips_node_only_without_cloud():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    experiment = ExperimentModel(
        prescribed=[BoundaryEntry(label="W", dof="W", nodes=[0, 2])]
    )
    add_bc_symbols_2d(axis, experiment, 0.0, 10.0, 0.0, 10.0)
    assert len(axis.patches) == 0
    plt.close(figure)


def test_bc_symbols_2d_label_top_wx_as_mxy():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    axis.set_xlim(0.0, 10.0)
    axis.set_ylim(0.0, 10.0)
    experiment = ExperimentModel(
        external=[
            BoundaryEntry.model_validate(
                {
                    "label": "twisting couple on the loaded (top) cut",
                    "dof": "W_X",
                    "edge": "top",
                    "loadType": "concentrated",
                }
            )
        ],
    )
    add_bc_symbols_2d(axis, experiment, 0.0, 10.0, 0.0, 10.0)
    labels = [text.get_text() for text in axis.texts]
    assert labels == ["Mxy"]
    plt.close(figure)


def test_set_mesh_points_replaces_coordinates():
    class _Mesh:
        n_points = 2
        points = np.zeros((2, 3))

    mesh = _Mesh()
    assert set_mesh_points(mesh, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    assert np.allclose(mesh.points[0], [1.0, 2.0, 3.0])
    assert set_mesh_points(mesh, np.zeros((1, 3))) is False


def test_nice_ticks_cover_specimen_ends():
    from vfmstudio.gui.viewport.overlays import _nice_ticks, specimen_xy_extents

    ticks = _nice_ticks(2.1, 8.4, n=5)
    assert ticks[0] <= 2.1 + 1.0e-9
    assert ticks[-1] >= 8.4 - 1.0e-9

    class _Mesh:
        node_coords = np.array([[0.0, 1.0, 0.0], [4.0, 5.0, 0.0]])

    dic = np.array([[-1.0, 0.5, 0.0], [6.0, 7.0, 0.0]])
    x_min, x_max, y_min, y_max = specimen_xy_extents(_Mesh(), dic)
    assert x_min == -1.0
    assert x_max == 6.0
    assert y_min == 0.5
    assert y_max == 7.0


def test_node_highlight_geometry_selected_and_lattice():
    from vfmstudio.gui.viewport.overlays import node_highlight_geometry

    class _Mesh:
        node_coords = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
        )

    selected, rest, labels, ids = node_highlight_geometry(_Mesh(), [0, 2, 2, -1, 99])
    assert ids == [0, 2]
    assert labels == ["0", "2"]
    assert selected.shape == (2, 3)
    assert rest.shape == (0, 3)
    assert selected[0, 2] > 0.0

    selected, rest, labels, ids = node_highlight_geometry(
        _Mesh(), [2], show_all=True
    )
    assert ids == [2]
    assert rest.shape == (3, 3)


def test_node_highlight_spec_preview_overrides_bc():
    from vfmstudio.core.document import BoundaryEntry, ExperimentModel
    from vfmstudio.gui.session import AppSession, Selection
    from vfmstudio.gui.viewport.interactor import _node_highlight_spec

    session = AppSession()
    session.preview_node_ids = (3, 5)
    session.preview_all_nodes = True
    ids, show_all = _node_highlight_spec(session)
    assert ids == (3, 5)
    assert show_all is True

    session.preview_node_ids = ()
    session.preview_all_nodes = False
    session.document.experiments = [
        ExperimentModel(
            experiment_id=1,
            prescribed=[BoundaryEntry(dof="W", nodes=[16, 24])],
        )
    ]
    session.selection = Selection(
        kind="bc", experiment_id=1, bc_kind="prescribed", bc_index=0
    )
    ids, show_all = _node_highlight_spec(session)
    assert ids == (16, 24)
    assert show_all is False

