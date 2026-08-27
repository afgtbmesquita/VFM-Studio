"""Abaqus BC glyph orientation and 2-D specimen symbols."""

from __future__ import annotations

import numpy as np

from vfmstudio.core.document import BoundaryEntry, ExperimentModel
from vfmstudio.gui.viewport.overlays import (
    add_bc_symbols_2d,
    display_axis,
    dof_direction,
    is_angular_dof,
    plane_glyph_direction,
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
    from matplotlib.patches import FancyArrow, Polygon

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
    assert len(polygons) >= 5  # X pin + Z pyramid faces + double cone on right
    assert len(arrows) == 10  # 5 edge samples × 2 moment heads
    plt.close(figure)


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

