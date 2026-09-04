"""Qt tests for the Add BC dialog."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("VFMSTUDIO_NO_VISTA", "1")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from vfmstudio.gui.dialogs.bc_editor import AddBcDialog


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_dialog_default_matches_original_edge_w(qapp):
    dialog = AddBcDialog(kind="prescribed")
    entries = dialog.entries()
    assert len(entries) == 1
    assert entries[0].dof == "W"
    assert entries[0].edge == "top"
    assert entries[0].nodes is None
    assert entries[0].label == "W @top"
    dialog.close()


def test_dialog_external_defaults_to_wy_on_top(qapp):
    dialog = AddBcDialog(kind="external")
    entries = dialog.entries()
    assert len(entries) == 1
    assert entries[0].dof == "W_Y"
    assert entries[0].edge == "top"
    assert entries[0].load_type == "distributed"
    dialog.close()


def test_dialog_multiple_dofs_one_entry_each(qapp):
    dialog = AddBcDialog(kind="prescribed")
    for name in ("W", "W_X", "W_Y", "W_XY"):
        dialog._dof_boxes[name].setChecked(True)
    dialog.edge.setCurrentText("bottom")
    dialog.label_edit.setText("bottom junction")
    entries = dialog.entries()
    assert [item.dof for item in entries] == ["W", "W_X", "W_Y", "W_XY"]
    assert all(item.edge == "bottom" for item in entries)
    assert [item.label for item in entries][0] == "bottom junction  W"
    dialog.close()


def test_dialog_individual_nodes(qapp):
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    dialog = AddBcDialog(kind="prescribed", node_coords=coords)
    dialog._nodes_radio.setChecked(True)
    dialog.nodes.item(0).setSelected(True)
    dialog.nodes.item(2).setSelected(True)
    dialog._dof_boxes["W"].setChecked(True)
    dialog._dof_boxes["W_X"].setChecked(True)
    entries = dialog.entries()
    assert len(entries) == 2
    assert all(item.edge is None for item in entries)
    assert entries[0].nodes == [0, 2]
    assert entries[1].dof == "W_X"
    dialog.close()


def test_dialog_bc_entries_keeps_every_dof(qapp):
    from vfmstudio.core.bc_entries import make_boundary_entries
    from vfmstudio.gui.main_window import _dialog_bc_entries

    class Multi:
        def entries(self):
            return make_boundary_entries(
                "prescribed", ["W", "W_X", "W_Y", "W_XY"], nodes=[16, 24]
            )

        def entry(self):
            raise AssertionError("must use entries(), not entry()")

    found = _dialog_bc_entries(Multi())
    assert [item.dof for item in found] == ["W", "W_X", "W_Y", "W_XY"]
    assert all(item.nodes == [16, 24] for item in found)
    assert all(item.edge is None for item in found)


def test_dialog_nodes_without_selection_raises(qapp):
    coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    dialog = AddBcDialog(kind="prescribed", node_coords=coords)
    dialog._nodes_radio.setChecked(True)
    with pytest.raises(ValueError):
        dialog.entries()
    dialog.close()


def test_dialog_emits_preview_and_toggle_node(qapp):
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    dialog = AddBcDialog(kind="prescribed", node_coords=coords)
    seen = []
    dialog.preview_nodes_changed.connect(
        lambda ids, show_all: seen.append((list(ids), bool(show_all)))
    )
    dialog._nodes_radio.setChecked(True)
    assert seen[-1] == ([], True)
    dialog.nodes.item(0).setSelected(True)
    dialog.nodes.item(2).setSelected(True)
    assert seen[-1][1] is True
    assert seen[-1][0] == [0, 2]
    dialog.toggle_node(0)
    assert seen[-1][0] == [2]
    dialog.toggle_node(1)
    assert set(seen[-1][0]) == {1, 2}
    dialog._edge_radio.setChecked(True)
    assert seen[-1] == ([], False)
    dialog.close()

