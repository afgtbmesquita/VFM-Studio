# =============================================================================
#  plate / _smoke_test.py
# -----------------------------------------------------------------------------
#  End-to-end smoke test for the plate VFM inverse solver.  Verifies that
#
#    (1) the plate data loader raises a focused error when the curvature
#        fields are missing from the input file;
#    (2) the BFS Kronecker-delta self-test passes (also exercised by
#        ``python -m vfm_mesh``);
#    (3) the full assembly + identification pipeline runs without raising
#        on a synthetic measurement file generated in-memory, and produces
#        correctly-shaped operators (G : (n_dofs, n_dofs), A : (n_dofs, 4),
#        K : (4, 4)).
#
#  Run from the plate folder with:
#      python -m tests.smoke_test
# =============================================================================

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from scipy.io import savemat

from vfm_constitutive import define_constitutive_model
from vfm_data_io import (
    BoundaryConditions,
    PointMeasurements,
    load_boundary_conditions,
    load_point_measurements,
)
from vfm_identification import (
    IdentificationCase,
    run_identification,
    run_identification_cases,
)
from vfm_identifiability import (
    compute_case_identifiability,
    compute_identifiability_diagnostics,
)
from vfm_mesh import (
    _bfs_kronecker_check,
    build_mesh,
    map_points_to_mesh,
    mesh_settings_from_node_points,
    prune_mesh_by_measurement_points,
    strain_displacement_batch,
    subset_measurements_to_mapped_mesh,
)
from vfm_optimal_field import (
    OptimalFieldSolverOptions,
    solve_optimal_virtual_field,
)
from vfm_user_inputs import (
    Bounds,
    IdentificationSettings,
    LinearAlgebraSettings,
    MeshSettings,
)


def _synthesise_measurement_file_legacy(target: Path,
                                        include_curvatures: bool,
                                        bounds: Bounds,
                                        grid_nx: int = 16,
                                        grid_ny: int = 10) -> None:
    """Write a small synthetic LEGACY-layout .mat file for the smoke test.

    Uses the legacy field names (X1/X2 + Eps*/K*) so the smoke test
    exercises the back-compatible loader path.  ``include_curvatures``
    controls whether K1/K2/K6 are written, so the loader's missing-
    curvature error path is also exercised.
    """
    xs = np.linspace(bounds.x_min, bounds.x_max, grid_nx)
    ys = np.linspace(bounds.y_min, bounds.y_max, grid_ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    payload = {
        "X1": X.ravel(),
        "X2": Y.ravel(),
        "Eps1": (1.0e-3 * (X / max(abs(bounds.x_min), abs(bounds.x_max)))).ravel(),
        "Eps2": (1.0e-3 * (Y / max(abs(bounds.y_min), abs(bounds.y_max)))).ravel(),
        "Eps6": (1.0e-3 * (X * Y) / (max(abs(bounds.x_max), 1.0)
                                     * max(abs(bounds.y_max), 1.0))).ravel(),
        "F": np.asarray([1.0, 1.0]),
    }
    if include_curvatures:
        payload["K1"] = (1.0e-5 * X).ravel()
        payload["K2"] = (1.0e-5 * Y).ravel()
        payload["K6"] = (1.0e-5 * (X + Y)).ravel()

    savemat(str(target), payload)


def _synthesise_measurement_file_pierron_bending(target: Path,
                                                 bounds: Bounds,
                                                 force: float = 9.82,
                                                 grid_nx: int = 7,
                                                 grid_ny: int = 5) -> None:
    """Write a Pierron-style bending-only LEGACY-layout .mat file.

    This mirrors ``Pierron_Plate_bending_sim.mat``: coordinates and
    curvatures are same-sized 2-D arrays, the force is scalar ``F``, and no
    membrane strain fields are present.
    """
    xs = np.linspace(bounds.x_min, bounds.x_max, grid_nx)
    ys = np.linspace(bounds.y_min, bounds.y_max, grid_ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    payload = {
        "X1": X,
        "X2": Y,
        "K1": 1.0e-5 * X,
        "K2": 2.0e-5 * Y,
        "K6": 3.0e-5 * (X + Y),
        "F": np.asarray([[force]], dtype=float),
    }

    savemat(str(target), payload)


def _synthesise_measurement_file_new(target: Path,
                                     bounds: Bounds,
                                     include_F: bool = False,
                                     grid_nx: int = 16,
                                     grid_ny: int = 10) -> None:
    """Write a small synthetic NEW-layout .mat file for the smoke test.

    Matches the names produced by ``Loading_Data/FromAbaqustoVFM.m``:
    one ``nodes`` (N, 2) array plus six 1-D strain / curvature arrays.
    ``F`` is intentionally OMITTED unless ``include_F`` is true so the
    BC-fallback path is exercised by default.
    """
    xs = np.linspace(bounds.x_min, bounds.x_max, grid_nx)
    ys = np.linspace(bounds.y_min, bounds.y_max, grid_ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    payload = {
        "nodes": np.column_stack([X.ravel(), Y.ravel()]),
        "SE1grid": (1.0e-3 * (X / max(abs(bounds.x_min),
                                       abs(bounds.x_max)))).ravel(),
        "SE2grid": (1.0e-3 * (Y / max(abs(bounds.y_min),
                                       abs(bounds.y_max)))).ravel(),
        "SE3grid": (1.0e-3 * (X * Y) / (max(abs(bounds.x_max), 1.0)
                                        * max(abs(bounds.y_max), 1.0))).ravel(),
        "SK1grid": (1.0e-5 * X).ravel(),
        "SK2grid": (1.0e-5 * Y).ravel(),
        "SK3grid": (1.0e-5 * (X + Y)).ravel(),
    }
    if include_F:
        payload["F"] = np.asarray([1.0, 1.0])

    savemat(str(target), payload)


# Back-compatible alias so older calls (if any) still work.
_synthesise_measurement_file = _synthesise_measurement_file_legacy


def main() -> int:
    print("[smoke] plate VFM smoke test starting")

    # -------------------------------------------------------------------
    # 1.  BFS Kronecker-delta sanity check
    # -------------------------------------------------------------------
    assert _bfs_kronecker_check(dx=2.5, dy=4.0), \
        "BFS Kronecker-delta self-test failed!"
    print("[smoke]   BFS Kronecker-delta self-test PASSED.")

    # -------------------------------------------------------------------
    # 2.  Per-point plate B-matrix shape check at a few sample points
    # -------------------------------------------------------------------
    coords = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 2.0], [0.0, 2.0]])
    B_all, _det_J = strain_displacement_batch(
        analysis_type="both",
        coords_per_point=np.broadcast_to(coords[None, :, :], (5, 4, 2)).copy(),
        xi=np.array([-0.7, -0.3, 0.0, 0.3, 0.7]),
        eta=np.array([-0.5, 0.5, 0.0, -0.5, 0.5]),
        dx_per_point=np.full(5, 3.0),
        dy_per_point=np.full(5, 2.0),
    )
    assert B_all.shape == (5, 6, 24), B_all.shape
    print(f"[smoke]   strain-displacement (both) shape OK: {B_all.shape}")

    # -------------------------------------------------------------------
    # 2b.  Nonuniform structured mesh construction and mapping
    # -------------------------------------------------------------------
    manual_points = np.asarray([
        [0.0, 0.0], [1.0, 0.0], [3.0, 0.0],
        [0.0, 0.5], [1.0, 0.5], [3.0, 0.5],
        [0.0, 2.0], [1.0, 2.0], [3.0, 2.0],
    ])
    manual_settings = mesh_settings_from_node_points(manual_points)
    manual_mesh = build_mesh(manual_settings, analysis_type="both")
    assert manual_mesh.n_nodes == 9, manual_mesh.n_nodes
    assert manual_mesh.n_elements == 4, manual_mesh.n_elements
    assert np.allclose(manual_mesh.x_nodes, [0.0, 1.0, 3.0])
    assert np.allclose(manual_mesh.y_nodes, [0.0, 0.5, 2.0])
    assert np.allclose(manual_mesh.element_dx, [1.0, 2.0, 1.0, 2.0])
    assert np.allclose(manual_mesh.element_dy, [0.5, 0.5, 1.5, 1.5])
    first = manual_mesh.node_coords[manual_mesh.element_nodes[0]]
    assert np.allclose(
        first,
        [[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [0.0, 0.5]],
    )
    mapped = map_points_to_mesh(
        np.asarray([
            [0.5, 0.25],
            [2.0, 0.25],
            [0.5, 1.25],
            [2.0, 1.25],
            [3.0, 2.0],
            [-0.1, 0.0],
        ]),
        manual_mesh,
    )
    assert mapped.element_ids.tolist() == [0, 1, 2, 3, 3, -1], \
        mapped.element_ids
    assert np.allclose(mapped.local_xy[:4], 0.0)
    outside_meas = PointMeasurements(
        points=np.asarray([
            [0.5, 0.25],
            [2.0, 0.25],
            [-0.1, 0.0],
        ]),
        strains=np.ones((3, 3), dtype=float),
        applied_force=None,
        area_per_point=0.25,
        data_bounds={"x_min": -0.1, "x_max": 2.0,
                     "y_min": 0.0, "y_max": 0.25},
        analysis_type="out_of_plane",
    )
    kept, kept_mapping = subset_measurements_to_mapped_mesh(
        outside_meas, manual_mesh)
    assert kept.points.shape[0] == 2, kept.points.shape
    assert kept_mapping.n_unmapped == 0
    assert np.allclose(kept.area_per_point, 0.25)
    print("[smoke]   unmapped measurement points are dropped, not rejected.")
    try:
        mesh_settings_from_node_points(manual_points[:-1])
    except ValueError as exc:
        assert "complete structured" in str(exc), str(exc)
    else:
        raise AssertionError("Expected missing manual node to be rejected.")
    try:
        mesh_settings_from_node_points(np.vstack([manual_points, manual_points[0]]))
    except ValueError as exc:
        assert "Duplicate" in str(exc), str(exc)
    else:
        raise AssertionError("Expected duplicate manual node to be rejected.")
    print("[smoke]   nonuniform manual mesh checks OK.")

    # -------------------------------------------------------------------
    # 3.  Constitutive model dimension check
    # -------------------------------------------------------------------
    model = define_constitutive_model("iso", "both", [10.0, 3.0, 1.0, 0.3])
    assert model.material_matrix.shape == (6, 6), model.material_matrix.shape
    assert model.basis_matrices.shape == (4, 6, 6), \
        model.basis_matrices.shape
    print(f"[smoke]   constitutive model OK: ABD shape "
          f"{model.material_matrix.shape}, basis shape "
          f"{model.basis_matrices.shape}")

    ortho_in = define_constitutive_model(
        "orthotropic", "in_plane", [10.0, 3.0, 8.0, 2.0])
    assert ortho_in.parameter_labels == ["A11", "A12", "A22", "A66"], \
        ortho_in.parameter_labels
    assert ortho_in.material_matrix.shape == (3, 3), \
        ortho_in.material_matrix.shape
    assert ortho_in.basis_matrices.shape == (4, 3, 3), \
        ortho_in.basis_matrices.shape
    assert np.allclose(ortho_in.material_matrix, np.array([
        [10.0, 3.0, 0.0],
        [3.0, 8.0, 0.0],
        [0.0, 0.0, 2.0],
    ]))

    ortho_out = define_constitutive_model(
        "orthotropic", "out_of_plane", [1.0, 0.3, 0.8, 0.2])
    assert ortho_out.parameter_labels == ["D11", "D12", "D22", "D66"], \
        ortho_out.parameter_labels
    assert ortho_out.material_matrix.shape == (3, 3), \
        ortho_out.material_matrix.shape
    assert ortho_out.basis_matrices.shape == (4, 3, 3), \
        ortho_out.basis_matrices.shape
    assert np.allclose(ortho_out.material_matrix, np.array([
        [1.0, 0.3, 0.0],
        [0.3, 0.8, 0.0],
        [0.0, 0.0, 0.2],
    ]))

    try:
        define_constitutive_model(
            "orthotropic", "both", [10.0, 3.0, 8.0, 2.0])
    except ValueError as exc:
        assert "both" in str(exc) and "Orthotropic" in str(exc), str(exc)
    else:
        raise AssertionError("Expected orthotropic + both to be unsupported.")

    print("[smoke]   orthotropic constitutive checks OK.")

    aniso_in = define_constitutive_model(
        "anisotropic", "in_plane", [10.0, 3.0, 8.0, 2.0, 0.7, 0.4])
    assert aniso_in.parameter_labels == \
        ["A11", "A12", "A22", "A66", "A16", "A26"], \
        aniso_in.parameter_labels
    assert aniso_in.material_matrix.shape == (3, 3), \
        aniso_in.material_matrix.shape
    assert aniso_in.basis_matrices.shape == (6, 3, 3), \
        aniso_in.basis_matrices.shape
    assert np.allclose(aniso_in.material_matrix, np.array([
        [10.0, 3.0, 0.7],
        [3.0, 8.0, 0.4],
        [0.7, 0.4, 2.0],
    ]))

    aniso_out = define_constitutive_model(
        "anisotropic", "out_of_plane", [1.0, 0.3, 0.8, 0.2, 0.07, 0.04])
    assert aniso_out.parameter_labels == \
        ["D11", "D12", "D22", "D66", "D16", "D26"], \
        aniso_out.parameter_labels
    assert aniso_out.material_matrix.shape == (3, 3), \
        aniso_out.material_matrix.shape
    assert aniso_out.basis_matrices.shape == (6, 3, 3), \
        aniso_out.basis_matrices.shape
    assert np.allclose(aniso_out.material_matrix, np.array([
        [1.0, 0.3, 0.07],
        [0.3, 0.8, 0.04],
        [0.07, 0.04, 0.2],
    ]))

    try:
        define_constitutive_model(
            "anisotropic", "both", [10.0, 3.0, 8.0, 2.0, 0.7, 0.4])
    except ValueError as exc:
        assert "both" in str(exc) and "Anisotropic" in str(exc), str(exc)
    else:
        raise AssertionError("Expected anisotropic + both to be unsupported.")

    try:
        define_constitutive_model(
            "anisotropic", "in_plane", [10.0, 3.0, 8.0, 2.0])
    except ValueError as exc:
        assert "expects 6 parameters" in str(exc), str(exc)
    else:
        raise AssertionError("Expected anisotropic parameter-count rejection.")

    print("[smoke]   anisotropic constitutive checks OK.")

    from vfm_user_inputs import PARAMETER_LABELS_BY_MODEL, parameter_labels_for

    for (material, analysis), expected_labels in PARAMETER_LABELS_BY_MODEL.items():
        labels = parameter_labels_for(material, analysis)
        assert labels == expected_labels, (material, analysis, labels)
        dummy = [1.0] * len(labels)
        #  NOT `model`: that name holds the iso/both model built above, which
        #  section 5 still needs for its optimal-field solve.  Rebinding it
        #  here left it as the last (material, analysis) pair of this loop --
        #  anisotropic/out_of_plane -- and section 5 then failed with
        #  "Mesh analysis_type 'both' does not match constitutive
        #  analysis_type 'out_of_plane'".
        labelled_model = define_constitutive_model(material, analysis, dummy)
        assert list(labelled_model.parameter_labels) == list(labels), (
            material, analysis, labelled_model.parameter_labels, labels
        )
    print("[smoke]   parameter_labels_for matches constitutive labels.")

    #  Frame rotation, across EVERY supported (material, analysis) pair.  The
    #  campaign only ever exercises orthotropic/out_of_plane, so the 6x6 'both'
    #  block-diagonal path and the in-plane path are only covered here.
    from vfm_constitutive import (
        engineering_rotation_matrix,
        rotate_constitutive_model,
    )

    for (material, analysis), expected_labels in PARAMETER_LABELS_BY_MODEL.items():
        dummy = [1.0 + 0.5 * i for i in range(len(expected_labels))]
        base = define_constitutive_model(material, analysis, dummy)
        n_gen = base.n_generalised_strain

        assert rotate_constitutive_model(base, 0.0) is base, (
            "a zero angle must be a no-op", material, analysis)

        rotated = rotate_constitutive_model(base, 33.0)
        #  Labels and theta stay in the MATERIAL frame -- that is what lets
        #  differently-oriented experiments share one unknown vector.
        assert list(rotated.parameter_labels) == list(expected_labels)
        assert np.allclose(rotated.parameters, base.parameters)
        assert rotated.orientation_deg == 33.0
        assert rotated.material_matrix.shape == (n_gen, n_gen)
        assert rotated.basis_matrices.shape == base.basis_matrices.shape

        #  The rotated basis must still be a constant decomposition.
        recon = np.einsum("k,kij->ij",
                          rotated.parameters, rotated.basis_matrices)
        assert np.allclose(recon, rotated.material_matrix, atol=1e-11), (
            material, analysis)
        assert np.allclose(rotated.material_matrix,
                           rotated.material_matrix.T, atol=1e-13)

        back = rotate_constitutive_model(rotated, -33.0)
        assert np.allclose(back.material_matrix, base.material_matrix,
                           atol=1e-11), (material, analysis)

        #  Internal virtual work is frame-invariant -- the identity that makes
        #  rotating D equivalent to rotating the fields.
        T = engineering_rotation_matrix(33.0, n_gen)
        rng_rot = np.random.default_rng(7)
        a, b = rng_rot.normal(size=n_gen), rng_rot.normal(size=n_gen)
        assert abs((T @ a) @ base.material_matrix @ (T @ b)
                   - a @ rotated.material_matrix @ b) < 1e-11, (
            material, analysis)

        if analysis == "both":
            #  blockdiag(T, T) must not couple the membrane and bending blocks.
            assert np.abs(rotated.material_matrix[:3, 3:]).max() < 1e-13
            assert np.abs(rotated.material_matrix[3:, :3]).max() < 1e-13

    #  90 deg on the orthotropic plate model is the transverse coupon's case:
    #  D11 <-> D22 swap, D12 and D66 untouched, no D16/D26 introduced.
    ortho = define_constitutive_model(
        "orthotropic", "out_of_plane", [61.43, 2.07, 174.35, 6.0])
    swapped = rotate_constitutive_model(ortho, 90.0).material_matrix
    assert np.allclose(swapped, np.array([[174.35, 2.07, 0.0],
                                          [2.07, 61.43, 0.0],
                                          [0.0, 0.0, 6.0]]), atol=1e-12)
    print("[smoke]   constitutive frame rotation OK (all models, 3x3 and 6x6).")

    diag_points = np.asarray([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ], dtype=float)
    diag_strains = np.asarray([
        [1.0e-3, 2.0e-3, 1.5e-3, 2.5e-3],
        [0.5e-3, 1.2e-3, 1.0e-3, 1.8e-3],
        [0.2e-3, 0.4e-3, 0.3e-3, 0.6e-3],
    ], dtype=float)
    diag_measurements = PointMeasurements(
        points=diag_points,
        strains=diag_strains,
        applied_force=None,
        area_per_point=0.25,
        data_bounds={"x_min": 0.0, "x_max": 1.0,
                     "y_min": 0.0, "y_max": 1.0},
        analysis_type="in_plane",
    )
    diag_models = [
        define_constitutive_model("iso", "in_plane", [10.0, 3.0]),
        define_constitutive_model(
            "orthotropic", "in_plane", [10.0, 3.0, 8.0, 2.0]),
        define_constitutive_model(
            "anisotropic", "in_plane", [10.0, 3.0, 8.0, 2.0, 0.7, 0.4]),
    ]
    for diag_model in diag_models:
        case_diag = compute_case_identifiability(
            diag_measurements,
            diag_model,
            label=f"{diag_model.material_type} in-plane",
        )
        n_param = diag_model.number_of_parameters
        assert case_diag.matrix.H.shape == (n_param, n_param), \
            case_diag.matrix.H.shape
        assert np.allclose(case_diag.matrix.H, case_diag.matrix.H.T), \
            case_diag.matrix.H
        assert case_diag.CID.shape == (n_param, diag_points.shape[0]), \
            case_diag.CID.shape
        assert np.all(case_diag.CID >= -1.0e-18), case_diag.CID
        assert case_diag.matrix.singular_values.shape == (n_param,), \
            case_diag.matrix.singular_values.shape
        diag_corr = case_diag.matrix.correlation
        finite_diag = np.isfinite(np.diag(diag_corr))
        if np.any(finite_diag):
            assert np.allclose(np.diag(diag_corr)[finite_diag], 1.0), \
                diag_corr

    singular_measurements = PointMeasurements(
        points=diag_points,
        strains=np.zeros_like(diag_strains),
        applied_force=None,
        area_per_point=0.25,
        data_bounds={"x_min": 0.0, "x_max": 1.0,
                     "y_min": 0.0, "y_max": 1.0},
        analysis_type="in_plane",
    )
    singular_diag = compute_case_identifiability(
        singular_measurements,
        diag_models[1],
        label="singular H",
    )
    assert singular_diag.matrix.used_pseudoinverse, \
        "Expected zero-information H to use pseudoinverse."

    stacked_diag = compute_identifiability_diagnostics(
        cases=[
            IdentificationCase(
                measurements=diag_measurements,
                mesh_pkg=None,
                boundary_conditions=None,
                label="diag A",
            ),
            IdentificationCase(
                measurements=diag_measurements,
                mesh_pkg=None,
                boundary_conditions=None,
                label="diag B",
            ),
        ],
        constitutive_model=diag_models[1],
    )
    expected_H = 2.0 * compute_case_identifiability(
        diag_measurements, diag_models[1]).matrix.H
    assert np.allclose(stacked_diag.aggregate.H, expected_H), \
        stacked_diag.aggregate.H
    assert len(stacked_diag.cases) == 2
    hole_meas = PointMeasurements(
        points=np.vstack([diag_points, np.array([[2.5, 1.5]])]),
        strains=np.hstack([diag_strains, diag_strains[:, :1]]),
        applied_force=None,
        area_per_point=0.25,
        data_bounds={"x_min": 0.0, "x_max": 3.0,
                     "y_min": 0.0, "y_max": 2.0},
        analysis_type="in_plane",
    )
    pruned_mesh, _ = prune_mesh_by_measurement_points(
        build_mesh(
            MeshSettings(number_of_elements_x=3,
                         number_of_elements_y=2,
                         bounds=Bounds(x_min=0.0, x_max=3.0,
                                       y_min=0.0, y_max=2.0)),
            analysis_type="in_plane",
        ),
        diag_points,
    )
    pruned_diag = compute_case_identifiability(
        hole_meas,
        diag_models[0],
        mesh_pkg=pruned_mesh,
    )
    assert pruned_diag.n_points_used < pruned_diag.n_points_total, \
        (pruned_diag.n_points_used, pruned_diag.n_points_total)
    assert pruned_diag.CID.shape[1] == pruned_diag.n_points_used
    print(
        "[smoke]   identifiability mesh filter OK: "
        f"{pruned_diag.n_points_used}/{pruned_diag.n_points_total} points"
    )
    print("[smoke]   identifiability diagnostics checks OK.")

    # -------------------------------------------------------------------
    # 4.  Missing-curvature error path (legacy layout)
    # -------------------------------------------------------------------
    bounds = Bounds(x_min=0.0, x_max=30.0, y_min=-9.5, y_max=9.5)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        membrane_only = tmpdir / "membrane_only.mat"
        _synthesise_measurement_file_legacy(membrane_only,
                                            include_curvatures=False,
                                            bounds=bounds)
        try:
            load_point_measurements(membrane_only, analysis_type="both")
        except KeyError as exc:
            msg = str(exc)
            assert "K1" in msg or "K2" in msg or "K6" in msg, msg
            print(f"[smoke]   missing-curvature error OK: {msg[:120]} ...")
        else:
            raise AssertionError(
                "Expected KeyError when curvature fields are missing."
            )

        # ---------------------------------------------------------------
        # 4b. Pierron-style bending-only file with 2-D X1/X2/K1/K2/K6.
        # ---------------------------------------------------------------
        pierron_file = tmpdir / "Pierron_Plate_bending_sim.mat"
        _synthesise_measurement_file_pierron_bending(
            pierron_file, bounds=bounds, force=9.82)
        pierron = load_point_measurements(pierron_file,
                                          analysis_type="out_of_plane")
        expected_n = 7 * 5
        assert pierron.points.shape == (expected_n, 2), pierron.points.shape
        assert pierron.strains.shape == (3, expected_n), pierron.strains.shape
        assert np.allclose(pierron.applied_force, np.asarray([9.82])), \
            pierron.applied_force
        assert pierron.analysis_type == "out_of_plane", pierron.analysis_type
        try:
            load_point_measurements(pierron_file, analysis_type="both")
        except KeyError as exc:
            msg = str(exc)
            assert "Eps1" in msg and "Eps2" in msg and "Eps6" in msg, msg
            print(f"[smoke]   Pierron bending-only load OK; "
                  f"both-mode rejection OK: {msg[:120]} ...")
        else:
            raise AssertionError(
                "Expected KeyError for Pierron bending-only file in both mode."
            )

        # ---------------------------------------------------------------
        # 5.  Full pipeline on a synthetic LEGACY file WITH curvatures
        # ---------------------------------------------------------------
        plate_file = tmpdir / "plate.mat"
        _synthesise_measurement_file_legacy(plate_file,
                                            include_curvatures=True,
                                            bounds=bounds)
        measurements = load_point_measurements(plate_file, analysis_type="both")
        assert measurements.strains.shape[0] == 6, measurements.strains.shape
        assert measurements.applied_force is not None, \
            "legacy MAT file declared F but loader returned None"
        print(f"[smoke]   legacy plate file loaded OK: "
              f"strains.shape = {measurements.strains.shape}, "
              f"applied_force = {measurements.applied_force}")

        # ---------------------------------------------------------------
        # 5b.  New 'nodes' layout WITHOUT F (BC-fallback workflow)
        # ---------------------------------------------------------------
        nodes_file = tmpdir / "plate_nodes.mat"
        _synthesise_measurement_file_new(nodes_file, bounds=bounds,
                                         include_F=False)
        measurements_nodes = load_point_measurements(nodes_file,
                                                     analysis_type="both")
        assert measurements_nodes.strains.shape == measurements.strains.shape, \
            (measurements_nodes.strains.shape, measurements.strains.shape)
        assert measurements_nodes.applied_force is None, \
            "new-layout MAT file omits F but loader returned a force"
        # The coordinates must match exactly (same grid factory).
        assert np.allclose(measurements_nodes.points, measurements.points), \
            "new and legacy layouts produced different point clouds"
        print(f"[smoke]   new 'nodes' layout loaded OK: "
              f"points={measurements_nodes.points.shape}, "
              f"applied_force=None (using BC fallback)")

        bc = load_boundary_conditions(
            Path(__file__).resolve().parents[1]
            / "examples" / "boundary_conditions_plate_example.py"
        )

        mesh_pkg = build_mesh(
            MeshSettings(number_of_elements_x=3,
                         number_of_elements_y=2,
                         bounds=bounds),
            analysis_type="both",
        )
        n_dofs = mesh_pkg.n_dofs
        print(f"[smoke]   mesh built: {mesh_pkg.n_elements} elements, "
              f"{mesh_pkg.n_nodes} nodes, {n_dofs} DOFs")

        sparse_center = np.array([[
            0.5 * (mesh_pkg.x_nodes[0] + mesh_pkg.x_nodes[1]),
            0.5 * (mesh_pkg.y_nodes[0] + mesh_pkg.y_nodes[1]),
        ]])
        pruned_pkg, prune_report = prune_mesh_by_measurement_points(
            mesh_pkg, sparse_center)
        assert pruned_pkg.n_elements == 1, pruned_pkg.n_elements
        assert pruned_pkg.n_nodes == 4, pruned_pkg.n_nodes
        assert pruned_pkg.n_dofs == 24, pruned_pkg.n_dofs
        assert prune_report.n_elements_removed == 5, prune_report.n_elements_removed
        mapping_sparse = map_points_to_mesh(sparse_center, pruned_pkg)
        assert mapping_sparse.n_mapped == 1, mapping_sparse.n_mapped
        assert mapping_sparse.element_ids[0] == 0, mapping_sparse.element_ids
        print("[smoke]   mesh prune OK: kept 1 element with data, removed 5 empty")

        from vfm_virtual_field_post import (
            nodal_deformed_xyz,
            structured_surface_grids_from_nodal_xyz,
        )
        u_pruned = np.zeros(pruned_pkg.n_dofs, dtype=float)
        cx, cy, cz = nodal_deformed_xyz(pruned_pkg, u_pruned)
        Xg, Yg, Zg = structured_surface_grids_from_nodal_xyz(
            pruned_pkg, cx, cy, cz)
        assert np.isfinite(Xg).sum() == 4, np.isfinite(Xg).sum()
        assert np.isnan(Xg).sum() == Xg.size - 4, np.isnan(Xg).sum()
        print("[smoke]   pruned-mesh surface grids OK for 3D viewer")

        # Test ONE inner solve (block-1 / block-2 / saddle point) to
        # confirm dimensions; we don't care about convergence here.
        result = solve_optimal_virtual_field(
            parameter_index_k=0,
            measurements=measurements,
            mesh_pkg=mesh_pkg,
            boundary_conditions=bc,
            constitutive_model=model,
            options=OptimalFieldSolverOptions(),
        )
        assert result.G.shape == (n_dofs, n_dofs), result.G.shape
        assert result.A.shape == (n_dofs, 4), result.A.shape
        assert result.nodal_variables_k.shape == (n_dofs,), \
            result.nodal_variables_k.shape
        print(f"[smoke]   optimal-field solve OK: "
              f"G {result.G.shape}, A {result.A.shape}, "
              f"u_k {result.nodal_variables_k.shape}")

        # Full outer loop with the LEGACY measurements (MAT-file F).
        ident = run_identification(
            measurements=measurements,
            mesh_pkg=mesh_pkg,
            boundary_conditions=bc,
            material_type="iso",
            analysis_type="both",
            initial_parameters=[10.0, 3.0, 1.0, 0.3],
            identification_settings=IdentificationSettings(
                max_iterations=2,
                convergence_tolerance=1.0e-4,
                solution_scale=1.0,
            ),
            linear_algebra_settings=LinearAlgebraSettings(),
            verbose=False,
        )
        assert ident.K_matrix.shape == (4, 4), ident.K_matrix.shape
        assert ident.U_vector.shape == (4,), ident.U_vector.shape
        assert ident.final_parameters.shape == (4,), \
            ident.final_parameters.shape
        assert len(ident.diagnostic_cases) == 1, ident.diagnostic_cases
        assert len(ident.optimal_nodal_history) == ident.iterations_completed
        assert all(len(row) == 4 for row in ident.optimal_nodal_history)
        from vfm_virtual_field_post import nodal_virtual_generalized_strains
        s0 = nodal_virtual_generalized_strains(
            mesh_pkg, ident.optimal_nodal_history[-1][0])
        assert s0.shape == (mesh_pkg.n_nodes, 6), s0.shape
        print(f"[smoke]   identification (legacy F) OK: "
              f"K {ident.K_matrix.shape}, U {ident.U_vector.shape}, "
              f"theta_final = {np.array2string(ident.final_parameters, precision=4)}")

        # Partial-mode orthotropic path: in-plane A has four independent
        # coefficients [A11, A12, A22, A66].  Use an in-plane-only BC set
        # because the example BC file also contains bending DOFs.
        measurements_in = load_point_measurements(plate_file,
                                                  analysis_type="in_plane")
        mesh_in = build_mesh(
            MeshSettings(number_of_elements_x=3,
                         number_of_elements_y=2,
                         bounds=bounds),
            analysis_type="in_plane",
        )
        bc_in = BoundaryConditions(
            prescribed=[
                {"label": "Clamp left U", "dof": "U",
                 "edge": "left", "value": 0.0},
                {"label": "Clamp left V", "dof": "V",
                 "edge": "left", "value": 0.0},
                {"label": "Pin right U", "dof": "U",
                 "edge": "right", "value": 0.0},
            ],
            linking=[
                {"label": "Rigid-link right V", "dof": "V",
                 "edge": "right", "sortDirection": "y", "value": 0.0},
            ],
            external=[
                {"label": "Measure right V", "dof": "V",
                 "edge": "right", "loadType": "concentrated"},
                {"label": "Measure top V", "dof": "V",
                 "edge": "top", "loadType": "concentrated"},
            ],
            analysis_type="in_plane",
        )
        ident_ortho = run_identification(
            measurements=measurements_in,
            mesh_pkg=mesh_in,
            boundary_conditions=bc_in,
            material_type="orthotropic",
            analysis_type="in_plane",
            initial_parameters=[10.0, 3.0, 8.0, 2.0],
            identification_settings=IdentificationSettings(
                max_iterations=2,
                convergence_tolerance=1.0e-4,
                solution_scale=1.0,
            ),
            linear_algebra_settings=LinearAlgebraSettings(),
            verbose=False,
        )
        assert ident_ortho.K_matrix.shape == (4, 4), \
            ident_ortho.K_matrix.shape
        assert ident_ortho.U_vector.shape == (4,), ident_ortho.U_vector.shape
        assert ident_ortho.final_parameters.shape == (4,), \
            ident_ortho.final_parameters.shape
        assert len(ident_ortho.optimal_field_results) == 4
        assert ident_ortho.optimal_field_results[0].A.shape == \
            (mesh_in.n_dofs, 4), ident_ortho.optimal_field_results[0].A.shape
        print(f"[smoke]   orthotropic in-plane identification OK: "
              f"K {ident_ortho.K_matrix.shape}, U {ident_ortho.U_vector.shape}, "
              f"theta_final = "
              f"{np.array2string(ident_ortho.final_parameters, precision=4)}")

        ident_ortho_stacked = run_identification_cases(
            cases=[
                IdentificationCase(
                    measurements=measurements_in,
                    mesh_pkg=mesh_in,
                    boundary_conditions=bc_in,
                    label="duplicate A",
                    data_file=plate_file,
                ),
                IdentificationCase(
                    measurements=measurements_in,
                    mesh_pkg=mesh_in,
                    boundary_conditions=bc_in,
                    label="duplicate B",
                    data_file=plate_file,
                ),
            ],
            material_type="orthotropic",
            analysis_type="in_plane",
            initial_parameters=[10.0, 3.0, 8.0, 2.0],
            identification_settings=IdentificationSettings(
                max_iterations=2,
                convergence_tolerance=1.0e-4,
                solution_scale=1.0,
            ),
            linear_algebra_settings=LinearAlgebraSettings(),
            verbose=False,
        )
        assert ident_ortho_stacked.K_matrix.shape == (8, 4), \
            ident_ortho_stacked.K_matrix.shape
        assert ident_ortho_stacked.U_vector.shape == (8,), \
            ident_ortho_stacked.U_vector.shape
        assert len(ident_ortho_stacked.diagnostic_cases) == 2, \
            ident_ortho_stacked.diagnostic_cases
        assert np.allclose(
            ident_ortho_stacked.final_parameters,
            ident_ortho.final_parameters,
            rtol=1.0e-8,
            atol=1.0e-8,
        ), (ident_ortho_stacked.final_parameters, ident_ortho.final_parameters)
        print("[smoke]   duplicate-case least-squares stack OK")

        # ----------------------------------------------------------------
        # 6.  New-layout identification with the BC-fallback path
        # ----------------------------------------------------------------
        # The example BC file does NOT declare a 'value' for any EXTERNAL
        # entry, so the resolver must raise a clear error first.
        try:
            run_identification(
                measurements=measurements_nodes,
                mesh_pkg=mesh_pkg,
                boundary_conditions=bc,
                material_type="iso",
                analysis_type="both",
                initial_parameters=[10.0, 3.0, 1.0, 0.3],
                identification_settings=IdentificationSettings(
                    max_iterations=1, convergence_tolerance=1.0e-4,
                    solution_scale=1.0,
                ),
                linear_algebra_settings=LinearAlgebraSettings(),
                verbose=False,
            )
        except ValueError as exc:
            msg = str(exc)
            assert "value" in msg.lower() and "F" in msg, msg
            print(f"[smoke]   missing-F error OK: {msg[:120]} ...")
        else:
            raise AssertionError(
                "Expected ValueError when neither MAT.F nor BC.value is "
                "available."
            )

        # Now build a BC with explicit per-entry 'value' fields and rerun.
        bc_with_values = type(bc)(
            prescribed=list(bc.prescribed),
            linking=list(bc.linking),
            external=[
                {**(entry if isinstance(entry, dict) else dict(entry)),
                 "value": 1.0}
                for entry in bc.external
            ],
        )
        ident_new = run_identification(
            measurements=measurements_nodes,
            mesh_pkg=mesh_pkg,
            boundary_conditions=bc_with_values,
            material_type="iso",
            analysis_type="both",
            initial_parameters=[10.0, 3.0, 1.0, 0.3],
            identification_settings=IdentificationSettings(
                max_iterations=2, convergence_tolerance=1.0e-4,
                solution_scale=1.0,
            ),
            linear_algebra_settings=LinearAlgebraSettings(),
            verbose=False,
        )
        assert ident_new.K_matrix.shape == (4, 4), ident_new.K_matrix.shape
        assert ident_new.U_vector.shape == (4,), ident_new.U_vector.shape
        print(f"[smoke]   identification (BC 'value' fallback) OK: "
              f"theta_final = "
              f"{np.array2string(ident_new.final_parameters, precision=4)}")

        # ---- applied-load source: 'mat' vs 'manual' ---------------------
        #  The whole point of the 'manual' source is that the declared values
        #  must WIN over a MAT file that also carries F.  Before it existed
        #  they were read, stored, and then silently ignored.
        from vfm_identification import (
            APPLIED_LOAD_FROM_MAT,
            APPLIED_LOAD_MANUAL,
            _resolve_applied_force,
            normalise_applied_load_source,
        )

        assert normalise_applied_load_source(None) == APPLIED_LOAD_FROM_MAT
        assert normalise_applied_load_source("auto") == APPLIED_LOAD_FROM_MAT
        assert normalise_applied_load_source("MANUAL") == APPLIED_LOAD_MANUAL
        try:
            normalise_applied_load_source("nonsense")
        except ValueError as exc:
            assert "nonsense" in str(exc)
        else:
            raise AssertionError("Expected an unknown load source to raise.")

        n_groups = len(bc_with_values.external)
        assert measurements.applied_force is not None, (
            "this fixture is supposed to carry an F field")
        mat_force = np.asarray(measurements.applied_force,
                               dtype=float).ravel()

        #  Deliberately unlike the MAT file's F (which is all ones here), or
        #  the two sources would be indistinguishable and the test vacuous.
        table_force = np.asarray([7.0, -3.0][:n_groups], dtype=float)
        bc_table = type(bc)(
            prescribed=list(bc.prescribed),
            linking=list(bc.linking),
            external=[{**dict(entry), "value": float(table_force[i])}
                      for i, entry in enumerate(bc.external)],
        )
        assert not np.allclose(mat_force, table_force), (
            "fixture setup error: the two load sources must differ")

        from_mat = _resolve_applied_force(
            measurements=measurements,
            boundary_conditions=bc_table,
            n_external_groups=n_groups,
            applied_load_source=APPLIED_LOAD_FROM_MAT)
        assert np.allclose(from_mat, mat_force), from_mat

        from_table = _resolve_applied_force(
            measurements=measurements,
            boundary_conditions=bc_table,
            n_external_groups=n_groups,
            applied_load_source=APPLIED_LOAD_MANUAL)
        assert np.allclose(from_table, table_force), from_table

        #  Manual mode falls back to F rather than failing, when no value is
        #  declared at all...
        fallback = _resolve_applied_force(
            measurements=measurements,
            boundary_conditions=bc,
            n_external_groups=n_groups,
            applied_load_source=APPLIED_LOAD_MANUAL)
        assert np.allclose(fallback, mat_force), fallback

        #  ...but a PARTIALLY filled set must not silently mix the two
        #  sources: it falls back wholesale, never group-by-group.
        partial_external = [dict(e) for e in bc_table.external]
        partial_external[0].pop("value")
        bc_partial = type(bc)(prescribed=list(bc.prescribed),
                              linking=list(bc.linking),
                              external=partial_external)
        mixed = _resolve_applied_force(
            measurements=measurements,
            boundary_conditions=bc_partial,
            n_external_groups=n_groups,
            applied_load_source=APPLIED_LOAD_MANUAL)
        assert np.allclose(mixed, mat_force), (
            "a half-filled load table must fall back whole, not blend", mixed)

        #  With no F anywhere, manual mode names the right knob.
        try:
            _resolve_applied_force(
                measurements=measurements_nodes,
                boundary_conditions=bc_partial,
                n_external_groups=n_groups,
                applied_load_source=APPLIED_LOAD_MANUAL)
        except ValueError as exc:
            assert "per-time-step" in str(exc), str(exc)
        else:
            raise AssertionError(
                "Expected manual mode with no values and no F to raise.")

        print("[smoke]   applied-load source ('mat' vs 'manual') OK.")

        bc_values_1 = type(bc)(
            prescribed=list(bc.prescribed),
            linking=list(bc.linking),
            external=[
                {**(entry if isinstance(entry, dict) else dict(entry)),
                 "value": 1.0}
                for entry in bc.external
            ],
            analysis_type="both",
        )
        bc_values_2 = type(bc)(
            prescribed=list(bc.prescribed),
            linking=list(bc.linking),
            external=[
                {**(entry if isinstance(entry, dict) else dict(entry)),
                 "value": 2.0}
                for entry in bc.external
            ],
            analysis_type="both",
        )
        ident_ordered = run_identification_cases(
            cases=[
                IdentificationCase(
                    measurements=measurements_nodes,
                    mesh_pkg=mesh_pkg,
                    boundary_conditions=bc_values_1,
                    label="unit load",
                    data_file=nodes_file,
                ),
                IdentificationCase(
                    measurements=measurements_nodes,
                    mesh_pkg=mesh_pkg,
                    boundary_conditions=bc_values_2,
                    label="double load",
                    data_file=nodes_file,
                ),
            ],
            material_type="iso",
            analysis_type="both",
            initial_parameters=[10.0, 3.0, 1.0, 0.3],
            identification_settings=IdentificationSettings(
                max_iterations=1, convergence_tolerance=1.0e-4,
                solution_scale=1.0,
            ),
            linear_algebra_settings=LinearAlgebraSettings(),
            verbose=False,
        )
        assert ident_ordered.K_matrix.shape == (8, 4), \
            ident_ordered.K_matrix.shape
        assert ident_ordered.U_vector.shape == (8,), ident_ordered.U_vector.shape
        assert ident_ordered.case_labels == ["unit load", "double load"], \
            ident_ordered.case_labels
        assert np.allclose(
            ident_ordered.case_U_vectors[1],
            2.0 * ident_ordered.case_U_vectors[0],
        ), (ident_ordered.case_U_vectors[0], ident_ordered.case_U_vectors[1])
        print("[smoke]   ordered multi-test case diagnostics OK")

        # ==============================================================
        #  Project-level guards on the GUI route
        # --------------------------------------------------------------
        #  Both of these failures used to be SILENT on the project/GUI
        #  path: run_campaign_gls.py guards them, vfm_project.py did not.
        # ==============================================================
        from scipy.io import savemat

        from vfm_project import (
            ProjectTest,
            VfmProject,
            check_experiment_consistency,
            check_load_orientation,
            project_to_cases,
        )

        guard_dir = Path(tmpdir) / "guards"
        guard_dir.mkdir(exist_ok=True)

        def _guard_mat(name, k_xx, k_yy):
            """A tiny coupon whose length runs along y, bending in kappa_yy."""
            gx, gy = np.meshgrid(np.linspace(0.0, 4.0, 6),
                                 np.linspace(0.0, 9.0, 8), indexing="ij")
            pts = np.column_stack([gx.ravel(), gy.ravel()])
            n = pts.shape[0]
            path = guard_dir / name
            savemat(path, {"nodes": pts,
                           "SK1grid": np.full(n, k_xx),
                           "SK2grid": np.full(n, k_yy),
                           "SK3grid": np.zeros(n),
                           "F": np.array([-1.0, 1.0])})
            return path

        def _guard_project(dof, k_xx, k_yy, with_external=True):
            external = ([{"label": "M top", "dof": dof, "edge": "top",
                          "loadType": "distributed", "sortDirection": "x"},
                         {"label": "M bottom", "dof": dof, "edge": "bottom",
                          "loadType": "distributed", "sortDirection": "x"}]
                        if with_external else [])
            tests = [
                ProjectTest(
                    data_file=_guard_mat(f"g_{dof}_{i}_{k_xx:g}.mat",
                                         k_xx, k_yy),
                    experiment_id=1,
                    prescribed=[{"label": "top w=0", "dof": "W",
                                 "edge": "top", "value": 0.0}],
                    external=[dict(e) for e in external],
                    mesh_settings={"number_of_elements_x": 2,
                                   "number_of_elements_y": 2,
                                   "bounds": {"x_min": 0.0, "x_max": 4.0,
                                              "y_min": 0.0, "y_max": 9.0}},
                    sample_area_total=36.0,
                )
                for i in range(2)
            ]
            return VfmProject(tests=tests)

        #  (1) prescribed BCs but NO external group.  This used to sail past
        #  the consistency check and then fail inside the solver as a
        #  confusing "F has 2 entries but the BC file declares 1
        #  external-work group" -- because the operator falls back to one
        #  placeholder column when the list is empty.
        no_external = _guard_project("W_Y", 5.0e-4, -9.7e-3,
                                     with_external=False)
        problems = [p for p in check_experiment_consistency(no_external)
                    if "EXTERNAL" in p]
        assert problems, (
            "an experiment with no EXTERNAL group must be reported",
            check_experiment_consistency(no_external))

        case = _guard_project("W_Y", 5.0e-4, -9.7e-3).tests[0]
        empty_bc = BoundaryConditions(prescribed=list(case.prescribed),
                                      linking=[], external=[],
                                      analysis_type="out_of_plane")
        try:
            _resolve_applied_force(
                measurements=measurements,
                boundary_conditions=empty_bc,
                n_external_groups=1)
        except ValueError as exc:
            assert "NO EXTERNAL work group" in str(exc), str(exc)
            assert "not one" in str(exc), (
                "the message must say the count is zero, not one", str(exc))
        else:
            raise AssertionError("Expected an empty EXTERNAL list to raise.")

        #  (2) load declared on the wrong axis.  kappa_yy dominates 18x, so a
        #  W_X-on-top group (twisting M_xy, not M_xx) is inconsistent with a
        #  bending field -- the run would complete and report the OTHER
        #  coefficient.
        assert not check_load_orientation(
            _guard_project("W_Y", 5.0e-4, -9.7e-3)), (
            "a correctly-oriented experiment must not warn")

        swapped = check_load_orientation(_guard_project("W_X", 5.0e-4, -9.7e-3))
        assert swapped and "kappa_yy" in swapped[0], swapped
        assert "kappa_xy" in swapped[0] or "M_xy" in swapped[0], swapped

        #  ...but a genuinely biaxial BENDING coupon (an off-axis test) has
        #  no dominant normal curvature, so the check must stay SILENT rather
        #  than cry wolf on the 45 deg experiment this solver now supports.
        #  W_Y on top is M_yy; W_X on top is twist and is checked separately.
        assert not check_load_orientation(
            _guard_project("W_Y", -7.5e-3, -9.7e-3)), (
            "the orientation check must not fire on a biaxial coupon")

        print("[smoke]   project guards (missing EXTERNAL, swapped load) OK.")

        #  Points past the virtual-mesh rectangle are dropped, not a hard
        #  failure.  The DIC cloud can overhang the mesh; those locations
        #  simply do not enter the virtual-work sums.
        overhang = _guard_project("W_Y", 5.0e-4, -9.7e-3)
        for test in overhang.tests:
            test.mesh_settings["bounds"]["y_max"] = 8.0
        overhang_cases, _ = project_to_cases(overhang)
        assert overhang_cases, "overhanging points must not abort the project"
        for case in overhang_cases:
            assert case.measurements.points.shape[0] > 0
            assert float(case.measurements.points[:, 1].max()) <= 8.0 + 1.0e-9
        print("[smoke]   project drops out-of-mesh measurement points OK.")

    print("[smoke] all checks PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
