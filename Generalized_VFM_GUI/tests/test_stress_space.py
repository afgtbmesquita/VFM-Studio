"""Stress-resultant reconstruction at DIC points from identified stiffness."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("VFMSTUDIO_NO_VISTA", "1")

from vfmstudio.core import bridge
from vfmstudio.core.document import empty_document
from vfmstudio.core.stress_space import (
    ALL_STEPS,
    available_resultants,
    constitutive_model_from_result,
    default_plot_axes,
    reconstruct_resultants,
)


def _constitutive():
    return bridge.vfm_constitutive()


def test_cubic_scales_make_a_visual_cube():
    from vfmstudio.gui.results.scene3d import (
        axis_extent,
        cubic_display_xyz,
        cubic_scales,
        plate_z_scale,
    )

    x = np.array([0.0, 10.0])
    y = np.array([0.0, 2.0])
    z = np.array([0.0, 0.5])
    sx, sy, sz = cubic_scales(x, y, z)
    visual = [axis_extent(x) * sx, axis_extent(y) * sy, axis_extent(z) * sz]
    np.testing.assert_allclose(visual, visual[0])
    vis, ranges, bounds = cubic_display_xyz(x, y, z)
    assert abs((bounds[1] - bounds[0]) - (bounds[5] - bounds[4])) < 1e-6
    assert abs((bounds[3] - bounds[2]) - (bounds[5] - bounds[4])) < 1e-6
    assert ranges[0] < ranges[1]
    lift = plate_z_scale(x, y, z, fraction=0.55)
    expected = 0.55 * max(axis_extent(x), axis_extent(y))
    assert abs(axis_extent(z) * lift - expected) < 1e-12


def test_available_resultants_follow_analysis_type():
    assert [spec.key for spec in available_resultants("in_plane")] == ["Nx", "Ny", "Nxy"]
    assert [spec.key for spec in available_resultants("out_of_plane")] == [
        "Mx",
        "My",
        "Mxy",
    ]
    both = available_resultants("both")
    assert [spec.key for spec in both] == ["Nx", "Ny", "Nxy", "Mx", "My", "Mxy"]
    assert both[0].row == 0
    assert both[3].row == 3
    assert default_plot_axes("out_of_plane")["x2d"] == "Mx"
    assert default_plot_axes("in_plane")["z3d"] == "Nxy"
    assert ALL_STEPS == -1
    membrane = available_resultants("in_plane")
    bending = available_resultants("out_of_plane")
    assert all(spec.unit_math == r"N/mm" for spec in membrane)
    assert all(spec.unit_math == "N" for spec in bending)
    assert all(spec.axis_label().endswith("(N)") for spec in bending)


def test_isotropic_bending_reconstruction():
    constitutive = _constitutive()
    d11, d12 = 100.0, 20.0
    d66 = 0.5 * (d11 - d12)
    model = constitutive.define_constitutive_model("iso", "out_of_plane", [d11, d12])
    kappa = np.array(
        [
            [1.0e-3, 2.0e-3, 0.0],
            [0.0, 0.5e-3, 1.0e-3],
            [0.0, 0.0, 4.0e-3],
        ],
        dtype=float,
    )
    values = reconstruct_resultants(kappa, model, 0.0)
    expected = model.material_matrix @ kappa
    np.testing.assert_allclose(values["Mx"], expected[0])
    np.testing.assert_allclose(values["My"], expected[1])
    np.testing.assert_allclose(values["Mxy"], expected[2])
    np.testing.assert_allclose(values["Mx"][0], d11 * 1.0e-3)
    np.testing.assert_allclose(values["My"][0], d12 * 1.0e-3)
    np.testing.assert_allclose(values["Mxy"][2], d66 * 4.0e-3)
    assert set(values) == {"Mx", "My", "Mxy"}


def test_orthotropic_90deg_rotation_swaps_mx_my_stiffness():
    constitutive = _constitutive()
    d11, d12, d22, d66 = 200.0, 20.0, 80.0, 30.0
    model = constitutive.define_constitutive_model(
        "orthotropic", "out_of_plane", [d11, d12, d22, d66]
    )
    kappa = np.array([[1.0], [0.0], [0.5]], dtype=float)
    at_zero = reconstruct_resultants(kappa, model, 0.0)
    at_ninety = reconstruct_resultants(kappa, model, 90.0)
    np.testing.assert_allclose(at_zero["Mx"], [d11])
    np.testing.assert_allclose(at_zero["My"], [d12])
    np.testing.assert_allclose(at_zero["Mxy"], [d66 * 0.5])
    np.testing.assert_allclose(at_ninety["Mx"], [d22])
    np.testing.assert_allclose(at_ninety["My"], [d12])
    np.testing.assert_allclose(at_ninety["Mxy"], at_zero["Mxy"])
    rotated = constitutive.rotate_constitutive_model(model, 90.0)
    np.testing.assert_allclose(rotated.material_matrix[0, 0], d22)
    np.testing.assert_allclose(rotated.material_matrix[1, 1], d11)
    np.testing.assert_allclose(rotated.material_matrix[2, 2], d66)


def test_in_plane_reconstruction_uses_a():
    constitutive = _constitutive()
    a11, a12, a22, a66 = 10.0, 2.0, 8.0, 3.0
    model = constitutive.define_constitutive_model(
        "orthotropic", "in_plane", [a11, a12, a22, a66]
    )
    strain = np.array([[0.1], [0.2], [0.05]], dtype=float)
    values = reconstruct_resultants(strain, model, 0.0)
    expected = model.material_matrix @ strain
    np.testing.assert_allclose(values["Nx"], expected[0])
    np.testing.assert_allclose(values["Ny"], expected[1])
    np.testing.assert_allclose(values["Nxy"], expected[2])
    assert set(values) == {"Nx", "Ny", "Nxy"}


def test_hdf5_thin_model_rebuilds_from_theta():
    constitutive = _constitutive()
    theta = np.array([100.0, 20.0])
    document = empty_document()
    document.material.material_type = "iso"
    document.material.analysis_type = "out_of_plane"
    document.material.initial_parameters = [1.0, 1.0]
    result = SimpleNamespace(
        final_parameters=theta,
        constitutive_model=SimpleNamespace(parameter_labels=["D11", "D12"]),
    )
    model = constitutive_model_from_result(document, result)
    assert model is not None
    assert getattr(model, "material_matrix", None) is not None
    reference = constitutive.define_constitutive_model("iso", "out_of_plane", theta)
    np.testing.assert_allclose(model.material_matrix, reference.material_matrix)
    kappa = np.array([[1.0e-3], [0.0], [0.0]], dtype=float)
    values = reconstruct_resultants(kappa, model, 0.0)
    np.testing.assert_allclose(values["Mx"], [100.0 * 1.0e-3])


def test_both_mode_exposes_membrane_and_bending():
    constitutive = _constitutive()
    model = constitutive.define_constitutive_model(
        "iso", "both", [10.0, 2.0, 100.0, 20.0]
    )
    fields = np.zeros((6, 1), dtype=float)
    fields[0, 0] = 0.1
    fields[3, 0] = 1.0e-3
    values = reconstruct_resultants(fields, model, 0.0)
    assert set(values) == {"Nx", "Ny", "Nxy", "Mx", "My", "Mxy"}
    np.testing.assert_allclose(values["Nx"], [10.0 * 0.1])
    np.testing.assert_allclose(values["Mx"], [100.0 * 1.0e-3])


def test_panel_draws_2d_and_3d_cloud():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MPLBACKEND", "Agg")
    import pytest

    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from vfmstudio.core.stress_space import StressSpaceCloud
    from vfmstudio.gui.results.stress_space import StressSpacePanel

    QApplication.instance() or QApplication([])
    panel = StressSpacePanel()
    n = 40
    cloud = StressSpaceCloud(
        values={
            "Mx": np.linspace(-1.0, 1.0, n),
            "My": np.linspace(0.0, 2.0, n),
            "Mxy": np.linspace(-0.5, 0.5, n),
        },
        step_index=np.zeros(n, dtype=int),
        step_labels=["step-01"],
        experiment_name="Coupon",
        analysis_type="out_of_plane",
        n_points=n,
    )
    panel._result = object()
    panel._cloud = cloud
    panel._cache_key = (id(panel._result), None, ALL_STEPS)
    panel.refresh()
    names = [getattr(axis, "name", "") for axis in panel._figure.axes]
    assert "3d" in names
    assert any(name != "3d" for name in names if name)
    axes_3d = [axis for axis in panel._figure.axes if getattr(axis, "name", "") == "3d"]
    assert axes_3d[0].get_zlabel()
    aspect = axes_3d[0].get_box_aspect()
    if aspect is not None:
        ratios = [float(value) for value in aspect]
        assert max(ratios) / min(ratios) < 1.05
    assert panel._vtk is None
    assert not panel._pane_headers.isVisible()
    assert panel._title_2d.text() == "2D"
    assert panel._title_3d.text() == "3D"
    assert panel.figure_for_export() is panel._figure
    panel.close()
