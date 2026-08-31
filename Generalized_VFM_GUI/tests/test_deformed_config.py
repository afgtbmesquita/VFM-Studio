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
    assert np.allclose(reference[:, 2], 0.25)
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
    assert np.allclose(undeformed[0], [0.0, 0.0, 0.1])
    assert np.allclose(deformed[0], [0.5, 0.0, 3.1])
    assert np.allclose(deformed[2], [0.5, 3.0, 3.2])


def test_specimen_points_xyz_respects_scale(tmp_path):
    path = _write_step_mat(tmp_path / "scaled.mat")
    dataset = load_step_fields(path, "out_of_plane")
    twice = specimen_points_xyz(dataset, deformed=True, scale=2.0)
    assert np.allclose(twice[0], [1.0, 0.0, 6.1])
