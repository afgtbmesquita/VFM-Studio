"""Add-load-steps dialog: MAT radio and CSV conversion smoke."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("VFMSTUDIO_NO_VISTA", "1")

pytest.importorskip("PySide6")
pytest.importorskip("pandas")

from PySide6.QtWidgets import QApplication

from vfmstudio.gui.dialogs.import_steps import AddLoadStepsDialog


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _write_dic(path: Path, *, cxx: float, number: int) -> Path:
    pd.DataFrame(
        {
            "X": [0.0, 1.0, 0.0],
            "Y": [0.0, 0.0, 1.0],
            "Z": [0.0, 0.0, 0.0],
            "U": [0.0, 0.0, 0.0],
            "V": [0.0, 0.0, 0.0],
            "W": [0.1, 0.1, 0.1],
            "exx": [0.0, 0.0, 0.0],
            "eyy": [0.0, 0.0, 0.0],
            "exy": [0.0, 0.0, 0.0],
            "cxx": [cxx, cxx, cxx],
            "cyy": [0.01, 0.01, 0.01],
            "cxy": [-0.02, -0.02, -0.02],
            "x": [1, 2, 1],
            "y": [1, 1, 2],
        }
    ).to_csv(path, index=False)
    return path


def test_mat_radio_returns_sorted_steps(qapp, tmp_path: Path):
    a = tmp_path / "step10.mat"
    b = tmp_path / "step2.mat"
    for path in (a, b):
        path.write_bytes(b"x")
    dialog = AddLoadStepsDialog()
    dialog.set_source_mode("mat")
    dialog._mat_list.set_paths([a, b])
    dialog._on_finish()
    steps = dialog.steps()
    assert [step.label for step in steps] == ["step2", "step10"]


def test_csv_dialog_converts_two_steps(qapp, tmp_path: Path):
    dic_a = _write_dic(tmp_path / "exp-00000044_0.csv", cxx=0.05, number=44)
    dic_b = _write_dic(tmp_path / "exp-00000045_0.csv", cxx=0.08, number=45)
    sync = tmp_path / "sync.csv"
    pd.DataFrame(
        {
            "Time_s": [0.25, 1.0],
            "DIC_file_number": [44, 45],
            "Machine_sys1_Ty_Nm_": [-0.001, -0.003],
            "Machine_sys2_Ty_Nm_": [0.002, 0.004],
        }
    ).to_csv(sync, index=False)
    out = tmp_path / "converted"
    dialog = AddLoadStepsDialog()
    dialog.set_source_mode("csv")
    dialog._dic_list.set_paths([dic_a, dic_b])
    dialog._load_list.set_paths([sync])
    dialog._go_mapping()
    assert dialog._vic3d.isChecked()
    assert "κₓₓ = −2 cxx" in dialog._vic3d_note.text()
    assert "γₓᵧ = 2 exy" in dialog._vic3d_note.text()
    used = [item for item in dialog._columns if item.used]
    assert len(used) == 2
    used[0].import_as = "Ty_sys1"
    used[1].import_as = "Ty_sys2"
    dialog._output.setText(str(out))
    dialog._rebuild_column_table()
    dialog._rematch()
    assert dialog._mapping_ready()
    dialog._on_finish()
    steps = dialog.steps()
    assert len(steps) == 2
    assert all(Path(step.data_file).is_file() for step in steps)
    assert [step.label for step in steps] == ["exp-00000044_0", "exp-00000045_0"]
    from vfmstudio.core import bridge
    from vfmstudio.core.fields import read_force_labels_from_file

    force = bridge.load_point_measurements(
        steps[0].data_file, "out_of_plane", total_sample_area=1.0)
    assert force.applied_force is not None
    assert force.applied_force.size == 2
    assert read_force_labels_from_file(steps[0].data_file) == ["Ty_sys1", "Ty_sys2"]
