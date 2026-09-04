"""Deformed-configuration extras (X,Y,Z + U,V,W) for the 3D viewport."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from scipy.io import savemat

from vfmstudio.core.fields import (
    has_deformed_configuration,
    load_step_fields,
    mat_has_deformed_configuration,
    specimen_points_xyz,
    try_load_displacements,
)


def _write_step_mat(path: Path, *, with_displacements: bool = True) -> Path:
    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 2.0, 2.0])
    payload = {
        "nodes": np.column_stack([x, y]),
        "SK1grid": np.zeros(4),
        "SK2grid": np.zeros(4),
        "SK3grid": np.zeros(4),
        "X": x,
        "Y": y,
        "Z": np.array([0.1, 0.1, 0.2, 0.2]),
    }
    if with_displacements:
        payload.update(
            {
                "U": np.array([0.5, 0.5, 0.5, 0.5]),
                "V": np.array([0.0, 0.0, 1.0, 1.0]),
                "W": np.array([3.0, 3.0, 3.0, 3.0]),
            }
        )
    savemat(path, payload)
    return path


def test_try_load_displacements_reads_z_uvw(tmp_path):
    path = _write_step_mat(tmp_path / "full.mat")
    extras = try_load_displacements(path)
    assert set(extras) >= {"u", "v", "w", "z"}
    assert np.allclose(extras["u"], 0.5)
    assert np.allclose(extras["w"], 3.0)
    assert np.allclose(extras["z"], [0.1, 0.1, 0.2, 0.2])
    assert mat_has_deformed_configuration(path) is True


def test_mat_without_uvw_is_unavailable(tmp_path):
    path = _write_step_mat(tmp_path / "plain.mat", with_displacements=False)
    extras = try_load_displacements(path)
    assert "u" not in extras and "v" not in extras and "w" not in extras
    assert mat_has_deformed_configuration(path) is False
    assert mat_has_deformed_configuration(None) is False


def test_has_deformed_configuration_on_dataset():
    n = 4
    dataset = xr.Dataset(
        {
            "kappa_xx": ("point", np.zeros(n)),
            "u": ("point", np.ones(n)),
            "v": ("point", np.full(n, 2.0)),
            "w": ("point", np.full(n, 3.0)),
        },
        coords={
            "x": ("point", np.arange(n, dtype=float)),
            "y": ("point", np.zeros(n)),
            "z": ("point", np.full(n, 0.25)),
        },
    )
    assert has_deformed_configuration(dataset) is True
    reference = specimen_points_xyz(dataset, deformed=False)
    deformed = specimen_points_xyz(dataset, deformed=True)
    assert np.allclose(reference[:, 2], 0.0)
    assert np.allclose(deformed[:, 0], np.arange(n) + 1.0)
    assert np.allclose(deformed[:, 1], 2.0)
    assert np.allclose(deformed[:, 2], 3.25)


def test_missing_w_is_not_deformed():
    n = 3
    dataset = xr.Dataset(
        {
            "u": ("point", np.ones(n)),
            "v": ("point", np.ones(n)),
        },
        coords={"x": ("point", np.zeros(n)), "y": ("point", np.zeros(n))},
    )
    assert has_deformed_configuration(dataset) is False
    pts = specimen_points_xyz(dataset, deformed=True)
    assert np.allclose(pts[:, 2], 0.0)


def test_load_step_fields_warps_xyz(tmp_path):
    path = _write_step_mat(tmp_path / "step.mat")
    dataset = load_step_fields(path, "out_of_plane")
    assert has_deformed_configuration(dataset) is True
    assert "z" in dataset
    undeformed = specimen_points_xyz(dataset, deformed=False)
    deformed = specimen_points_xyz(dataset, deformed=True)
    assert np.allclose(undeformed[0], [0.0, 0.0, 0.0])
    assert np.allclose(np.unique(undeformed[:, 2]), [0.0])
    assert np.allclose(deformed[0], [0.5, 0.0, 3.1])
    assert np.allclose(deformed[2], [0.5, 3.0, 3.2])
    origin = specimen_points_xyz(dataset, deformed=True, scale=0.0)
    assert np.allclose(origin[0], [0.0, 0.0, 0.1])


def test_specimen_points_xyz_respects_scale(tmp_path):
    path = _write_step_mat(tmp_path / "scaled.mat")
    dataset = load_step_fields(path, "out_of_plane")
    twice = specimen_points_xyz(dataset, deformed=True, scale=2.0)
    assert np.allclose(twice[0], [1.0, 0.0, 6.1])


def test_subtract_reference_dataset_fields_displacements_and_force():
    from vfmstudio.core.fields import subtract_reference_dataset

    n = 3
    target = xr.Dataset(
        {
            "kappa_yy": ("point", np.array([3.0, 4.0, 5.0])),
            "u": ("point", np.array([1.0, 1.0, 1.0])),
            "w": ("point", np.array([6.0, 6.0, 6.0])),
        },
        coords={"x": ("point", np.arange(n, dtype=float)), "y": ("point", np.zeros(n))},
        attrs={"applied_force": [30.0, 40.0]},
    )
    reference = xr.Dataset(
        {
            "kappa_yy": ("point", np.array([1.0, 1.0, 1.0])),
            "u": ("point", np.array([0.5, 0.5, 0.5])),
            "w": ("point", np.array([2.0, 2.0, 2.0])),
        },
        coords={"x": ("point", np.arange(n, dtype=float)), "y": ("point", np.zeros(n))},
        attrs={"applied_force": [10.0, 15.0]},
    )
    out = subtract_reference_dataset(target, reference)
    assert np.allclose(out["kappa_yy"].values, [2.0, 3.0, 4.0])
    assert np.allclose(out["u"].values, [0.5, 0.5, 0.5])
    assert np.allclose(out["w"].values, [4.0, 4.0, 4.0])
    assert np.allclose(out.attrs["applied_force"], [20.0, 25.0])
    assert np.allclose(target["kappa_yy"].values, [3.0, 4.0, 5.0])


def test_referenced_step_dataset_uses_experiment_reference(tmp_path):
    from vfmstudio.core.document import ExperimentModel, LoadStepModel
    from vfmstudio.core.fields import load_step_fields, referenced_step_dataset

    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 2.0, 2.0])
    ref = tmp_path / "ref.mat"
    cur = tmp_path / "cur.mat"
    savemat(
        ref,
        {
            "nodes": np.column_stack([x, y]),
            "SK1grid": np.ones(4),
            "SK2grid": np.full(4, 2.0),
            "SK3grid": np.zeros(4),
            "U": np.ones(4),
            "V": np.zeros(4),
            "W": np.full(4, 3.0),
            "F": np.array([10.0]),
        },
    )
    savemat(
        cur,
        {
            "nodes": np.column_stack([x, y]),
            "SK1grid": np.full(4, 4.0),
            "SK2grid": np.full(4, 5.0),
            "SK3grid": np.ones(4),
            "U": np.full(4, 2.0),
            "V": np.ones(4),
            "W": np.full(4, 7.0),
            "F": np.array([40.0]),
        },
    )
    experiment = ExperimentModel(
        experiment_id=1,
        reference_step=1,
        steps=[
            LoadStepModel(data_file=ref, label="ref"),
            LoadStepModel(data_file=cur, label="cur"),
        ],
    )
    current = load_step_fields(cur, "out_of_plane")
    referenced = referenced_step_dataset(current, experiment, analysis_type="out_of_plane")
    assert np.allclose(referenced["kappa_xx"].values, 3.0)
    assert np.allclose(referenced["kappa_yy"].values, 3.0)
    assert np.allclose(referenced["u"].values, 1.0)
    assert np.allclose(referenced["w"].values, 4.0)
    assert np.allclose(referenced.attrs["applied_force"], [30.0])
    self_ref = referenced_step_dataset(
        load_step_fields(ref, "out_of_plane"), experiment, analysis_type="out_of_plane"
    )
    assert np.allclose(self_ref["kappa_yy"].values, 0.0)
    assert np.allclose(self_ref.attrs["applied_force"], [0.0])


def test_collect_step_series_subtracts_reference_step(tmp_path):
    from vfmstudio.core.document import ExperimentModel, LoadStepModel
    from vfmstudio.core.step_series import collect_step_series

    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 2.0, 2.0])

    def write(path, kappa, force):
        savemat(
            path,
            {
                "nodes": np.column_stack([x, y]),
                "SK1grid": np.zeros(4),
                "SK2grid": np.full(4, kappa),
                "SK3grid": np.zeros(4),
                "F": np.array([force]),
            },
        )
        return path

    experiment = ExperimentModel(
        experiment_id=1,
        reference_step=2,
        steps=[
            LoadStepModel(data_file=write(tmp_path / "s1.mat", 1.0, 10.0), label="s1"),
            LoadStepModel(data_file=write(tmp_path / "s2.mat", 2.0, 20.0), label="s2"),
            LoadStepModel(data_file=write(tmp_path / "s3.mat", 5.0, 50.0), label="s3"),
        ],
    )
    series = collect_step_series(experiment, "out_of_plane")
    assert np.allclose(series.points[0].field_means["kappa_yy"], -1.0)
    assert np.allclose(series.points[1].field_means["kappa_yy"], 0.0)
    assert np.allclose(series.points[2].field_means["kappa_yy"], 3.0)
    assert np.allclose(series.points[0].force, [-10.0])
    assert np.allclose(series.points[1].force, [0.0])
    assert np.allclose(series.points[2].force, [30.0])


def test_scene_key_includes_reference_step():
    from vfmstudio.core.document import ExperimentModel, LoadStepModel
    from vfmstudio.gui.session import AppSession, Selection
    from vfmstudio.gui.viewport.interactor import _scene_key

    experiment = ExperimentModel(
        experiment_id=1,
        steps=[LoadStepModel(data_file="a.mat"), LoadStepModel(data_file="b.mat")],
    )
    session = AppSession()
    session.document.experiments = [experiment]
    session.selection = Selection(kind="step", experiment_id=1, step_index=1)
    before = _scene_key(session)
    experiment.reference_step = 2
    after = _scene_key(session)
    assert before != after
    assert after[-3] == 2
