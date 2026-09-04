"""CSV → Layout A MAT conversion for load-step import."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import loadmat

from vfmstudio.core.fields import force_component_text, read_force_labels_from_file
from vfmstudio.io.csv_import import (
    CURVATURE_FACTORS,
    STRAIN_FACTORS,
    LoadColumn,
    UNIT_NM,
    VIC3D_CONVENTION_NOTE,
    collect_load_columns,
    convert_csv_steps,
    default_import_name,
    dic_file_number,
    infer_unit_scale,
    is_vic3d_csv,
    load_frames_from_paths,
    match_dic_steps,
    path_identity,
    suggest_load_column,
    write_step_mat,
)


def _write_dic(path: Path, *, cxx: float, facets: list[tuple[int, int]]) -> Path:
    rows = []
    for px, py in facets:
        rows.append(
            {
                "X": float(px),
                "Y": float(py),
                "Z": 0.1,
                "U": 0.0,
                "V": 0.0,
                "W": 0.2,
                "exx": 0.001,
                "eyy": 0.002,
                "exy": 0.0005,
                "cxx": cxx,
                "cyy": 0.01,
                "cxy": -0.02,
                "x": px,
                "y": py,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_sync(path: Path) -> Path:
    pd.DataFrame(
        {
            "Time_s": [0.25, 1.0],
            "DIC_file_number": [44, 45],
            "Machine_sys1_Ty_Nm_": [-0.001, -0.003],
            "Machine_sys2_Ty_Nm_": [0.002, 0.004],
            "Machine_Fx_N_": [1.0, 2.0],
        }
    ).to_csv(path, index=False)
    return path


def test_dic_file_number_from_exp_name():
    assert dic_file_number("exp-00000044_0.csv") == 44
    assert dic_file_number(Path("exp-00000292_0.csv")) == 292


def test_infer_nm_scale_and_names():
    assert infer_unit_scale("Machine_sys1_Ty_Nm_") == 1000.0
    assert infer_unit_scale("couple_Nmm") == 1.0
    assert default_import_name("Machine_sys1_Ty_Nm_") == "sys1_Ty"
    assert suggest_load_column("Machine_sys1_Ty_Nm_")
    assert suggest_load_column("dTy_Nm")
    assert not suggest_load_column("Time_s")
    assert not suggest_load_column("Machine_Fx_N_")


def test_match_by_dic_file_number(tmp_path: Path):
    sync = _write_sync(tmp_path / "sync.csv")
    frames = load_frames_from_paths([sync])
    dic = [
        tmp_path / "exp-00000045_0.csv",
        tmp_path / "exp-00000044_0.csv",
    ]
    matches = match_dic_steps(dic, frames)
    assert [item.dic_path.name for item in matches] == [
        "exp-00000044_0.csv",
        "exp-00000045_0.csv",
    ]
    identity = path_identity(sync)
    assert matches[0].row_by_source[identity] == 0
    assert matches[1].row_by_source[identity] == 1


def test_convert_vic3d_and_nm_to_nmm(tmp_path: Path):
    dic_a = _write_dic(
        tmp_path / "exp-00000044_0.csv",
        cxx=0.05,
        facets=[(1, 1), (2, 1), (1, 2)],
    )
    dic_b = _write_dic(
        tmp_path / "exp-00000045_0.csv",
        cxx=0.08,
        facets=[(1, 1), (2, 1), (1, 2), (3, 1)],
    )
    sync = _write_sync(tmp_path / "sync.csv")
    frames = load_frames_from_paths([sync])
    columns = collect_load_columns([sync], frames)
    used = [item for item in columns if item.used]
    assert [item.source_column for item in used] == [
        "Machine_sys1_Ty_Nm_",
        "Machine_sys2_Ty_Nm_",
    ]
    used[0].import_as = "Ty_sys1"
    used[0].sign = -1.0
    used[1].import_as = "Ty_sys2"
    matches = match_dic_steps([dic_a, dic_b], frames)
    out = tmp_path / "mats"
    steps = convert_csv_steps(
        matches, frames, columns, out, vic3d_conventions=True, intersect_facets=True
    )
    assert [step.label for step in steps] == ["exp-00000044_0", "exp-00000045_0"]

    from vfmstudio.core import bridge

    loaded = bridge.load_point_measurements(steps[0].data_file, "out_of_plane")
    assert loaded.points.shape == (3, 2)
    assert np.allclose(loaded.points[0], [1.0, 1.0])
    expected_kxx = CURVATURE_FACTORS[0] * 0.05
    assert np.allclose(loaded.strains[0], expected_kxx)
    # sign -1 * 1000 * -0.001 = +1.0 ; +1 * 1000 * 0.002 = 2.0
    assert np.allclose(loaded.applied_force, [1.0, 2.0])
    labels = read_force_labels_from_file(steps[0].data_file)
    assert labels == ["Ty_sys1", "Ty_sys2"]
    assert force_component_text(0, labels) == "Ty_sys1"

    loaded_b = bridge.load_point_measurements(steps[1].data_file, "both")
    assert loaded_b.strains.shape[0] == 6
    expected_gxy = STRAIN_FACTORS[2] * 0.0005
    assert np.allclose(loaded_b.strains[2], expected_gxy)
    assert np.allclose(loaded_b.applied_force, [3.0, 4.0])


def test_write_step_mat_omits_f_when_unused(tmp_path: Path):
    payload = {
        "nodes": np.array([[0.0, 0.0], [1.0, 0.0]]),
        "X": np.array([0.0, 1.0]),
        "Y": np.array([0.0, 0.0]),
        "SK1grid": np.zeros(2),
        "SK2grid": np.zeros(2),
        "SK3grid": np.zeros(2),
    }
    path = write_step_mat(tmp_path / "plain.mat", payload)
    raw = loadmat(str(path), squeeze_me=True)
    assert "F" not in raw
    assert force_component_text(0, None) == "F[0]"


def test_is_vic3d_csv(tmp_path: Path):
    path = _write_dic(tmp_path / "cloud.csv", cxx=0.01, facets=[(1, 1)])
    assert is_vic3d_csv(path) is True
    other = tmp_path / "machine.csv"
    pd.DataFrame({"Time_s": [0.0], "Machine_Ty_Nm_": [1.0]}).to_csv(other, index=False)
    assert is_vic3d_csv(other) is False


def test_vic3d_convention_note_states_plate_theory_factors():
    assert "κₓₓ = −2 cxx" in VIC3D_CONVENTION_NOTE
    assert "γₓᵧ = 2 exy" in VIC3D_CONVENTION_NOTE


def test_custom_nm_column_scale():
    column = LoadColumn(
        source_file=Path("sync.csv"),
        source_column="Ty",
        import_as="Ty",
        unit=UNIT_NM,
        sign=1.0,
        used=True,
    )
    assert column.resolved_scale() == 1000.0


def test_missing_x_raises(tmp_path: Path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"cxx": [0.1], "cyy": [0.1], "cxy": [0.1]}).to_csv(path, index=False)
    frames = {}
    matches = match_dic_steps([path], frames)
    with pytest.raises(Exception, match="missing column"):
        convert_csv_steps(matches, frames, [], tmp_path / "out")
