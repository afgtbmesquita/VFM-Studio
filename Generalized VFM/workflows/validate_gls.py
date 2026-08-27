# =============================================================================
#  Generalized VFM / validate_gls.py   --   READ THIS BEFORE THE REAL RESULT
# -----------------------------------------------------------------------------
#  On real data the truth is unknown, so "D11 and D22 came out plausible" would
#  prove nothing.  This script establishes, in order:
#
#    0. KERNEL.  The covariance propagation reproduces the solver's own G
#       matrix when Gamma = I.  This is the single test that pins down the
#       dA factor, the B-operator scatter and the index ordering.
#
#    1. REDUCTION LADDER.  On the real cases, the FGLS estimator collapses to
#       the estimators it generalises:
#          weighting='ols'                       -> run_identification_cases
#          weighting='diagonal', Gamma = dt I    -> inverse-variance pooling
#       If these do not hold, the assembly is wrong and nothing below matters.
#
#    2. GAMMA RECOVERY.  Spatially white noise with a KNOWN cross-time Gamma is
#       injected, and the residual-projector estimator must recover it -- both
#       the component covariance and the temporal correlation.  kappa must come
#       out ~1, which is the self-consistency check on the spatial correction.
#
#    3. FULL PIPELINE.  One known D generates synthetic fields on BOTH real
#       measurement clouds, with the imposed correlated noise on top, and the
#       whole chain -- residual estimation, virtual fields, covariance
#       propagation, whitening, FGLS -- must recover it.
#
#  HOW THE SYNTHETIC FIELDS ARE BUILT.  Each test gets an equilibrated moment
#  field that is traction-free on ITS OWN free edges, because that is what its
#  shipped BC file assumes.  The two families are mirror images:
#
#    LONGITUDINAL   free edges at x = const, phi(x) = (x-x0)^2 (x-x1)^2
#        M_xx = a phi(x)     M_xy = -(a/2) phi'(x) y     M_yy = m2 + p y
#        equilibrium: a phi'' - a phi'' + 0 = 0
#        free edges:  M_xx = 0 and V_x = M_xx,x + 2 M_xy,y = 0 identically
#
#    TRANSVERSE     free edges at y = const, phi(y) = (y-y0)^2 (y-y1)^2
#        M_yy = a phi(y)     M_xy = -(a/2) phi'(y) x     M_xx = m2 + p x
#
#  Both come from THE SAME D_TRUE in THE SAME frame, and F is written into each
#  .mat so the SHIPPED BC files are used unmodified: this validates the real
#  boundary conditions, not a copy of them.  The moment fields are polynomials
#  of degree <= 4, so a degree-6 smoothing basis represents the true kinematics
#  exactly and M Y_true = 0 to round-off -- which is what isolates the noise
#  estimation from lack-of-fit in test 2.
#
#  Run:  python -m workflows.validate_gls
# =============================================================================

from __future__ import annotations

import shutil
import sys

import numpy as np
from scipy.io import loadmat, savemat

from workflows.campaign_common import (
    ANALYSIS_TYPE,
    HERE,
    INITIAL_PARAMETERS,
    LONGITUDINAL,
    MATERIAL_TYPE,
    RESULTS_DIR,
    TRANSVERSE,
    build_campaign,
    build_cases,
    mat_files,
    report_parameters,
)
from vfm_constitutive import (
    define_constitutive_model,
    rotate_constitutive_model,
)
from vfm_gls_identification import run_gls_identification_cases
from vfm_identification import run_identification_cases
from vfm_mesh import map_points_to_mesh
from vfm_noise_covariance import (
    build_gram_tensor,
    estimate_kinematic_covariance,
    known_kinematic_covariance,
    virtual_work_gram,
)
from vfm_optimal_field import (
    OptimalFieldSolverOptions,
    collect_assembly_payload,
    solve_optimal_virtual_field,
    solve_optimal_virtual_fields,
)
from vfm_user_inputs import (
    GlsSettings,
    IdentificationSettings,
    LinearAlgebraSettings,
)

SCRATCH = RESULTS_DIR / "_scratch" / "validate_gls"

#  The known answer, chosen near the two measured values so the benchmark
#  exercises the solver in the regime the real data occupies.
#  D11 is the LONGITUDINAL coupon's stiffness (the material 1-axis runs along
#  its length) and D22 the transverse one, so the stiff value belongs to D11.
#  Getting this the wrong way round makes the synthetic transverse coupon three
#  times stiffer than the real one and the benchmark stops being representative.
D_TRUE = np.array([175.0, 15.0, 65.0, 40.0])          # D11, D12, D22, D66

#  Nine synthetic load cases per test, matching the real campaign's nine steps,
#  with a load ramp so the cross-time covariance has something to bite on.
FIELD_SETS = [
    dict(m2=0.40 + 0.22 * i, p=0.045 - 0.010 * i)
    for i in range(9)
]

#  THE BENCHMARK MUST BE AS HARD AS THE EXPERIMENT.
#  The amplitude `a` of the secondary moment field controls how far each test
#  departs from pure cylindrical bending, and therefore how much information it
#  carries about D12.  Left free, a synthetic can be made far richer than the
#  real coupons and would then "prove" that D12 is recoverable when it is not.
#  These coefficients set a = BIAXIALITY_ALPHA * m2 so that the synthetic
#  reproduces each test's MEASURED biaxiality,
#
#      |mean secondary curvature| / |mean primary curvature|
#      longitudinal  0.090        transverse  0.015
#
#  which is what makes "FGLS recovers D11 and D22 here" transfer to the real
#  data.  assert_representative() below verifies the match rather than trusting
#  these numbers, so a change to D_TRUE cannot silently make the test easy.
#  Re-solved after the material frame was corrected (D11 is now the
#  LONGITUDINAL coupon).  Each coupon's synthetic is generated with the
#  compliance IT actually sees, so these depend on the orientations in
#  campaign_common and must be re-solved if those change.  Note the mapping
#  alpha -> biaxiality is not monotonic: the alpha-driven secondary moment
#  partly cancels the Poisson term, so there is a minimum (0.013 longitudinal,
#  0.017 transverse) and two roots either side of it.  These are the ones on
#  the descending branch.
BIAXIALITY_ALPHA = {"longitudinal": 0.094738, "transverse": 0.440}
MEASURED_BIAXIALITY = {"longitudinal": 0.090, "transverse": 0.015}

#  Imposed noise model for tests 2 and 3.
NOISE_SIGMA = np.array([1.5e-3, 1.2e-3, 1.8e-3])      # per curvature component
NOISE_TIME_RHO = 0.80                                  # AR(1) across load steps
NOISE_COMPONENT_CORR = np.array([[1.00, -0.30, 0.10],
                                 [-0.30, 1.00, 0.05],
                                 [0.10, 0.05, 1.00]])

TOLERANCES = {
    "kernel": 1.0e-11,
    "ols_reduction": 1.0e-6,
    "diagonal_reduction": 1.0e-6,
    "gamma_recovery": 0.08,
    "identifiable_parameter": 5.0,      # per cent
}

_FAILURES = []


def _check(name: str, ok: bool, detail: str) -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: {detail}")
    if not ok:
        _FAILURES.append(name)
    return ok


def imposed_gamma(n_steps: int) -> np.ndarray:
    """Gamma = kron(AR(1) in time, component covariance).  Step-major."""
    lags = np.abs(np.subtract.outer(np.arange(n_steps), np.arange(n_steps)))
    time_correlation = NOISE_TIME_RHO ** lags
    component = np.outer(NOISE_SIGMA, NOISE_SIGMA) * NOISE_COMPONENT_CORR
    return np.kron(time_correlation, component)


# =============================================================================
#  Test 0 -- the covariance kernel against the solver's own G
# =============================================================================

def test_kernel() -> None:
    print()
    print("-" * 78)
    print("  0. KERNEL: does the covariance propagation reproduce the solver?")
    print("-" * 78)
    print("  G_solver = sum_p w B^T K^T K B  uses w to the FIRST power.")
    print("  The covariance needs w^2 (eq. 4's dA), so Phi with Gamma = I must")
    print("  equal w * G_solver exactly.  Any error in the B scatter, the")
    print("  component ordering or the dA power shows up here.")

    case = build_cases(LONGITUDINAL, verbose=False)[0]
    model = define_constitutive_model(
        MATERIAL_TYPE, ANALYSIS_TYPE, INITIAL_PARAMETERS)
    mapping = map_points_to_mesh(case.measurements.points, case.mesh_pkg)
    payload = collect_assembly_payload(
        case.measurements, case.mesh_pkg, mapping,
        model.n_generalised_strain)

    weight = case.measurements.area_per_point
    gram_tensor = build_gram_tensor(payload, weight, case.mesh_pkg.n_dofs)
    mine = virtual_work_gram(
        gram_tensor, model.material_matrix,
        np.eye(model.n_generalised_strain)) / weight

    reference = solve_optimal_virtual_field(
        parameter_index_k=0, measurements=case.measurements,
        mesh_pkg=case.mesh_pkg, boundary_conditions=case.boundary_conditions,
        constitutive_model=model, mapping=mapping).G

    scale = float(np.abs(reference).max())
    relative = float(np.abs(mine - reference).max()) / scale
    _check("kernel vs solver G", relative < TOLERANCES["kernel"],
           f"max relative difference = {relative:.3e}  (scale {scale:.3e})")


# =============================================================================
#  Test 1 -- the reduction ladder, on the real cases
# =============================================================================

def _solver_settings(tolerance=1.0e-10, iterations=40):
    return (IdentificationSettings(max_iterations=iterations,
                                   convergence_tolerance=tolerance),
            LinearAlgebraSettings())


def _run_gls(cases, ids, weighting, noise_overrides=None,
             spatial_correction=True, basis_degree=6, initial=None):
    identification, linear_algebra = _solver_settings()
    return run_gls_identification_cases(
        cases=cases, material_type=MATERIAL_TYPE, analysis_type=ANALYSIS_TYPE,
        initial_parameters=INITIAL_PARAMETERS if initial is None else initial,
        identification_settings=identification,
        linear_algebra_settings=linear_algebra,
        experiment_ids=ids,
        gls_settings=GlsSettings(basis_degree=basis_degree,
                                 weighting=weighting,
                                 spatial_correction=spatial_correction),
        noise_overrides=noise_overrides,
        verbose=False,
    )


def _inverse_variance_reference(cases, ids):
    """theta_k = sum_i U_ik/(w_i eta_ik^2) / sum_i 1/(w_i eta_ik^2).

    The stacked-VFM estimator, with the dA factor it omits put back in.  Solved
    as its own fixed point so the comparison is like for like.
    """
    from vfm_mesh import build_external_work_matrix
    from vfm_identification import _resolve_applied_force

    prepared = []
    for case in cases:
        mapping = map_points_to_mesh(case.measurements.points, case.mesh_pkg)
        matrix, _ = build_external_work_matrix(
            mesh_pkg=case.mesh_pkg, bc=case.boundary_conditions)
        force = _resolve_applied_force(
            measurements=case.measurements,
            boundary_conditions=case.boundary_conditions,
            n_external_groups=matrix.shape[1])
        prepared.append((case, mapping, matrix, force))

    theta = np.asarray(INITIAL_PARAMETERS, dtype=float)
    for _ in range(60):
        model = define_constitutive_model(MATERIAL_TYPE, ANALYSIS_TYPE, theta)
        n_parameters = model.number_of_parameters
        estimates = np.zeros((len(prepared), n_parameters))
        weights = np.zeros((len(prepared), n_parameters))
        for index, (case, mapping, matrix, force) in enumerate(prepared):
            #  Same per-case rotation the solver applies, or this reference is
            #  not comparing like with like once a coupon sits off-axis.
            results = solve_optimal_virtual_fields(
                measurements=case.measurements, mesh_pkg=case.mesh_pkg,
                boundary_conditions=case.boundary_conditions,
                constitutive_model=rotate_constitutive_model(
                    model, case.material_orientation_deg),
                mapping=mapping)
            weight = case.measurements.area_per_point
            for k, result in enumerate(results):
                u = result.nodal_variables_k
                estimates[index, k] = float(
                    np.dot(matrix.T @ u, force))
                eta_squared = max(float(u @ result.G @ u), 0.0)
                weights[index, k] = 1.0 / (weight * eta_squared)
        new_theta = ((estimates * weights).sum(axis=0) / weights.sum(axis=0))
        if np.max(np.abs(new_theta - theta)
                  / np.maximum(np.abs(new_theta), 1e-30)) < 1e-12:
            theta = new_theta
            break
        theta = new_theta
    return theta


def test_reduction_ladder(cases, ids) -> None:
    print()
    print("-" * 78)
    print("  1. REDUCTION LADDER: does FGLS collapse to what it generalises?")
    print("-" * 78)

    identification, linear_algebra = _solver_settings()
    reference_ols = run_identification_cases(
        cases=cases, material_type=MATERIAL_TYPE, analysis_type=ANALYSIS_TYPE,
        initial_parameters=INITIAL_PARAMETERS,
        identification_settings=identification,
        linear_algebra_settings=linear_algebra, verbose=False
    ).final_parameters
    mine_ols = _run_gls(cases, ids, "ols").final_parameters
    gap = float(np.max(np.abs(reference_ols - mine_ols)
                       / np.maximum(np.abs(reference_ols), 1e-30)))
    print(f"    run_identification_cases : "
          f"{np.array2string(reference_ols, precision=6)}")
    print(f"    FGLS weighting='ols'     : "
          f"{np.array2string(mine_ols, precision=6)}")
    _check("weighting='ols' reproduces the OLS solver",
           gap < TOLERANCES["ols_reduction"], f"max relative gap = {gap:.3e}")

    # ---- diagonal + white Gamma  ==  inverse-variance pooling -------------
    n_generalised = define_constitutive_model(
        MATERIAL_TYPE, ANALYSIS_TYPE, INITIAL_PARAMETERS).n_generalised_strain
    steps_per_experiment = {}
    for exp_id in ids:
        steps_per_experiment[exp_id] = steps_per_experiment.get(exp_id, 0) + 1
    white = {
        exp_id: known_kinematic_covariance(
            np.eye(n_steps * n_generalised), n_generalised)
        for exp_id, n_steps in steps_per_experiment.items()
    }

    reference_iv = _inverse_variance_reference(cases, ids)
    mine_iv = _run_gls(cases, ids, "diagonal",
                       noise_overrides=white).final_parameters
    gap = float(np.max(np.abs(reference_iv - mine_iv)
                       / np.maximum(np.abs(reference_iv), 1e-30)))
    print(f"    inverse-variance (dA-corrected) : "
          f"{np.array2string(reference_iv, precision=6)}")
    print(f"    FGLS diagonal, Gamma = dt I     : "
          f"{np.array2string(mine_iv, precision=6)}")
    _check("weighting='diagonal' reproduces inverse-variance pooling",
           gap < TOLERANCES["diagonal_reduction"],
           f"max relative gap = {gap:.3e}")


# =============================================================================
#  Test 2 -- can the estimator recover a Gamma we imposed?
# =============================================================================

def test_gamma_recovery() -> None:
    print()
    print("-" * 78)
    print("  2. GAMMA RECOVERY: inject a known cross-time covariance")
    print("-" * 78)
    print("  Spatially white noise with a known AR(1) temporal correlation and")
    print("  known cross-component covariance is added to an exactly-")
    print("  representable smooth field.  The residual-projector estimator must")
    print("  return it, and kappa must come out ~1 because the noise IS white.")

    rng = np.random.default_rng(20260816)
    points = np.asarray(
        loadmat(mat_files(LONGITUDINAL)[-1], squeeze_me=True)["nodes"],
        dtype=float)
    n_points = points.shape[0]
    n_steps, n_generalised = 9, 3

    gamma_true = imposed_gamma(n_steps)
    factor = np.linalg.cholesky(gamma_true + 1e-18 * np.eye(gamma_true.shape[0]))
    noise = (rng.standard_normal((n_points, n_steps * n_generalised))
             @ factor.T)

    #  A smooth field the degree-6 basis represents exactly, so the residual is
    #  pure injected noise and the test isolates the estimator.
    x = (points[:, 0] - points[:, 0].mean()) / np.ptp(points[:, 0])
    y = (points[:, 1] - points[:, 1].mean()) / np.ptp(points[:, 1])
    smooth = np.stack([
        np.column_stack([
            (1.0 + 0.3 * t) * (0.02 + 0.01 * x + 0.004 * y * y),
            (1.0 + 0.3 * t) * (-0.03 + 0.006 * y + 0.002 * x * y),
            (1.0 + 0.3 * t) * (0.005 * x * y),
        ])
        for t in range(n_steps)
    ])
    observed = smooth + noise.reshape(n_points, n_steps, n_generalised
                                      ).transpose(1, 0, 2)

    estimate = estimate_kinematic_covariance(
        points=points, strain_stack=observed,
        basis_degree=6, spatial_correction=False)

    scale = np.sqrt(np.outer(np.diag(gamma_true), np.diag(gamma_true)))
    relative = float(np.abs(estimate.gamma - gamma_true).max()
                     / np.sqrt(np.diag(gamma_true)).max() ** 2)
    _check("Gamma recovered from injected noise",
           relative < TOLERANCES["gamma_recovery"],
           f"max |Gamma_hat - Gamma_true| / max(diag) = {relative:.4f}")

    measured_rho = estimate.temporal_correlation(1)[0, 1]
    _check("temporal correlation recovered",
           abs(measured_rho - NOISE_TIME_RHO) < 0.05,
           f"rho(step0, step1) = {measured_rho:.4f}, "
           f"imposed {NOISE_TIME_RHO:.4f}")

    sigma_ratio = estimate.residual_scale[0] / NOISE_SIGMA
    _check("per-component sigma recovered",
           float(np.abs(sigma_ratio - 1.0).max()) < 0.05,
           f"sigma_hat/sigma_true = "
           f"{np.array2string(sigma_ratio, precision=3)}")

    with_correction = estimate_kinematic_covariance(
        points=points, strain_stack=observed,
        basis_degree=6, spatial_correction=True)
    kappa = with_correction.correlation.kappa
    _check("kappa ~ 1 on genuinely white noise", kappa < 1.6,
           f"kappa = {kappa:.3f}  (1.0 is the white-noise value)")


# =============================================================================
#  Test 3 -- the full pipeline against a known D
# =============================================================================

def _phi(t, t0, t1):
    """phi = (t-t0)^2 (t-t1)^2 normalised to max 1, and its derivative."""
    scale = ((t1 - t0) / 2.0) ** 4
    phi = ((t - t0) ** 2 * (t - t1) ** 2) / scale
    dphi = (2 * (t - t0) * (t - t1) ** 2 + 2 * (t - t0) ** 2 * (t - t1)) / scale
    return phi, dphi


def synthetic_for(test, compliance, rng, noisy: bool):
    """Write synthetic .mat for one test; return their paths."""
    nodes = np.asarray(loadmat(mat_files(test)[-1], squeeze_me=True)["nodes"],
                       dtype=float)
    x, y = nodes[:, 0], nodes[:, 1]
    x0, x1 = float(x.min()), float(x.max())
    y0, y1 = float(y.min()), float(y.max())
    n_points = nodes.shape[0]
    n_steps = len(FIELD_SETS)

    if noisy:
        gamma_true = imposed_gamma(n_steps)
        factor = np.linalg.cholesky(
            gamma_true + 1e-18 * np.eye(gamma_true.shape[0]))
        noise = (rng.standard_normal((n_points, n_steps * 3)) @ factor.T)
        noise = noise.reshape(n_points, n_steps, 3).transpose(1, 0, 2)
    else:
        noise = np.zeros((n_steps, n_points, 3))

    alpha = BIAXIALITY_ALPHA[test.name]

    SCRATCH.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, setting in enumerate(FIELD_SETS):
        amplitude = alpha * setting["m2"]
        if test.length_axis == "y":
            phi, dphi = _phi(x, x0, x1)
            moments = np.vstack([amplitude * phi,
                                 setting["m2"] + setting["p"] * y,
                                 -0.5 * amplitude * dphi * y])
            force = np.array([-(setting["m2"] + setting["p"] * y1),
                              +(setting["m2"] + setting["p"] * y0)])
        else:
            phi, dphi = _phi(y, y0, y1)
            moments = np.vstack([setting["m2"] + setting["p"] * x,
                                 amplitude * phi,
                                 -0.5 * amplitude * dphi * x])
            force = np.array([-(setting["m2"] + setting["p"] * x1),
                              +(setting["m2"] + setting["p"] * x0)])

        curvature = compliance @ moments + noise[index].T
        path = SCRATCH / f"{test.name}_syn_{index}.mat"
        savemat(path, {"nodes": nodes,
                       "SK1grid": curvature[0], "SK2grid": curvature[1],
                       "SK3grid": curvature[2], "F": force})
        paths.append(path)
    return paths


def _biaxiality(paths, length_axis: str) -> float:
    """|mean secondary curvature| / |mean primary curvature|, averaged."""
    primary, secondary = (1, 0) if length_axis == "y" else (0, 1)
    ratios = []
    for path in paths:
        data = loadmat(path, squeeze_me=True)
        means = [float(np.mean(data["SK1grid"])),
                 float(np.mean(data["SK2grid"]))]
        ratios.append(abs(means[secondary]) / abs(means[primary]))
    return float(np.mean(ratios))


def assert_representative(paths_by_test) -> None:
    """The synthetic must be no easier than the real experiment.

    A benchmark with more biaxial content than the coupons would recover D12
    and then imply, wrongly, that the real run can too.
    """
    print("    Is the benchmark as hard as the experiment?")
    for test in (LONGITUDINAL, TRANSVERSE):
        synthetic = _biaxiality(paths_by_test[test.name], test.length_axis)
        measured = MEASURED_BIAXIALITY[test.name]
        ratio = synthetic / measured
        _check(f"{test.name} biaxiality matches the real coupon",
               0.5 < ratio < 2.0,
               f"synthetic {synthetic:.4f} against measured {measured:.4f} "
               f"({ratio:.2f}x)")


def test_full_pipeline() -> None:
    print()
    print("-" * 78)
    print("  3. FULL PIPELINE: one known D, both real clouds, imposed noise")
    print("-" * 78)
    print(f"  D_true = "
          f"{dict(zip(['D11', 'D12', 'D22', 'D66'], D_TRUE))}")

    #  D_TRUE is in the MATERIAL frame, but each coupon's synthetic field is
    #  generated in ITS OWN measurement frame -- so the compliance used to turn
    #  moments into curvatures must be the one that coupon actually sees, i.e.
    #  D_TRUE rotated by its material_orientation_deg.  Skipping this rotation
    #  silently builds the benchmark in the wrong frame, and the solver then
    #  "recovers" the transposed answer.
    material_stiffness = define_constitutive_model(
        MATERIAL_TYPE, ANALYSIS_TYPE, D_TRUE)

    def compliance_for(test):
        rotated = rotate_constitutive_model(
            material_stiffness, test.material_orientation_deg)
        return np.linalg.inv(rotated.material_matrix)

    rng = np.random.default_rng(20260817)

    for noisy in (False, True):
        tag = "WITH imposed correlated noise" if noisy else "noise-free"
        paths = {test.name: synthetic_for(test, compliance_for(test), rng,
                                          noisy)
                 for test in (LONGITUDINAL, TRANSVERSE)}
        if not noisy:
            assert_representative(paths)
        cases, ids, _ = build_campaign(paths_by_test=paths, verbose=False)

        identification, linear_algebra = _solver_settings()
        ols = run_identification_cases(
            cases=cases, material_type=MATERIAL_TYPE,
            analysis_type=ANALYSIS_TYPE,
            initial_parameters=INITIAL_PARAMETERS,
            identification_settings=identification,
            linear_algebra_settings=linear_algebra, verbose=False
        ).final_parameters
        gls = _run_gls(cases, ids, "gls").final_parameters

        labels = ["D11", "D12", "D22", "D66"]
        print()
        print(f"    ---- {tag} ----")
        print(f"    {'run':<26}" + "".join(f"{n:>11}" for n in labels))
        for name, values in (("naive stacked (OLS)", ols), ("FGLS", gls)):
            errors = 100.0 * np.abs(values - D_TRUE) / np.abs(D_TRUE)
            print(f"    {name:<26}"
                  + "".join(f"{v:11.2f}" for v in values))
            print(f"    {'  error %':<26}"
                  + "".join(f"{e:10.1f}%" for e in errors))

        for name, index in (("D11", 0), ("D22", 2)):
            error = 100.0 * abs(gls[index] - D_TRUE[index]) / D_TRUE[index]
            _check(f"FGLS recovers {name} ({tag})",
                   error < TOLERANCES["identifiable_parameter"],
                   f"{gls[index]:.2f} against {D_TRUE[index]:.2f}, "
                   f"error {error:.2f} %")

    print()
    print("    Only D11 and D22 are checked.  The synthetic is built at the")
    print("    coupons' MEASURED biaxiality (0.090 / 0.015), so it is as")
    print("    near-cylindrical as the real tests: the secondary curvature is")
    print("    at noise level, and D12 -- which multiplies exactly that")
    print("    curvature -- has almost nothing to act on.  Whatever D12 and")
    print("    D66 do above, they do it on the same thin information the real")
    print("    run has, so they must not be quoted from either.  That is a")
    print("    property of the experiments, not of the estimator: it needs an")
    print("    off-axis coupon, a torsion test, or genuinely biaxial bending.")


# =============================================================================

def main() -> int:
    print("=" * 78)
    print("  GENERALIZED VFM -- KNOWN-ANSWER VALIDATION")
    print("=" * 78)

    try:
        test_kernel()
        cases, ids, _ = build_campaign(verbose=False)
        test_reduction_ladder(cases, ids)
        test_gamma_recovery()
        test_full_pipeline()
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)

    print()
    print("=" * 78)
    if _FAILURES:
        print(f"  {len(_FAILURES)} CHECK(S) FAILED: " + ", ".join(_FAILURES))
        print("=" * 78)
        return 1
    print("  ALL CHECKS PASSED")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
