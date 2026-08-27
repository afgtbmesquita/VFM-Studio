# =============================================================================
#  Generalized VFM / validate_rotation.py
# -----------------------------------------------------------------------------
#  Does rotating D give the same answer as rotating the fields?
#
#  THE QUESTION
#  ---------------------------------------------------------------------------
#  Internal virtual work is a frame-invariant scalar,
#
#      kappa*_lab^T D_lab kappa_lab  =  kappa*_mat^T D_mat kappa_mat
#
#  so with kappa_mat = T(phi) kappa_lab there are exactly TWO self-consistent
#  ways to reconcile a coupon cut off-axis with a material-frame theta:
#
#      (A)  rotate the fields, the mesh, the BCs and the wrench into the
#           material frame, and leave D alone;
#      (B)  leave every measurement in its own frame and use
#           D_lab = T(phi)^T D_mat T(phi).
#
#  Everything must live in ONE frame, so doing both would rotate twice.  This
#  file proves the claim on real data instead of asserting it: the transverse
#  coupon is identified each way and the two thetas must agree.
#
#      python -m workflows.make_raw_frame_transverse     # build the route (B) input
#      python -m workflows.validate_rotation
#
#  Route (A) is the shipped export, whose fields MATLAB already rotated +90 deg.
#  Route (B) is that rotation undone, plus material_orientation_deg = 90.
# =============================================================================

from __future__ import annotations

import numpy as np

from workflows.campaign_common import (
    ANALYSIS_TYPE,
    INITIAL_PARAMETERS,
    MATERIAL_TYPE,
    TRANSVERSE,
    TRANSVERSE_RAW,
    LONGITUDINAL,
    assert_orientation,
    build_cases,
)
from vfm_constitutive import (
    define_constitutive_model,
    engineering_rotation_matrix,
    rotate_constitutive_model,
)
from vfm_gls_identification import run_gls_identification_cases
from vfm_identification import run_identification_cases
from vfm_user_inputs import (
    GlsSettings,
    IdentificationSettings,
    LinearAlgebraSettings,
)

#  theta agreement between the two routes.  Bit-identity is NOT expected: the
#  two runs use different node orderings and different lstsq pivoting, so the
#  floating-point path differs even though the mathematics does not.
THETA_TOLERANCE = 1.0e-9

LABELS = ("D11", "D12", "D22", "D66")


def _fail(message: str) -> None:
    raise SystemExit(f"\nFAILED: {message}")


def _settings():
    return (IdentificationSettings(max_iterations=40,
                                   convergence_tolerance=1.0e-8),
            LinearAlgebraSettings())


# =============================================================================
#  Test 1 -- the algebra, before any data is involved
# =============================================================================

def test_algebra() -> None:
    print("-" * 78)
    print("  1. rotate_constitutive_model: algebra")
    print("-" * 78)

    D11, D12, D22, D66 = 61.43, 2.07, 174.35, 6.0
    model = define_constitutive_model(
        MATERIAL_TYPE, ANALYSIS_TYPE, [D11, D12, D22, D66])

    #  A zero angle must be a genuine no-op, so every pre-existing run is
    #  provably untouched by this feature.
    if rotate_constitutive_model(model, 0.0) is not model:
        _fail("a zero angle must return the model unchanged.")
    print("  zero angle is a no-op                                     OK")

    #  90 deg: the relabeling the MATLAB export performs.
    r90 = rotate_constitutive_model(model, 90.0)
    expected90 = np.array([[D22, D12, 0.0],
                           [D12, D11, 0.0],
                           [0.0, 0.0, D66]])
    err90 = float(np.abs(r90.material_matrix - expected90).max())
    print(f"  90 deg gives the D11<->D22 swap          {err90:.2e}         "
          f"{'OK' if err90 < 1e-12 else 'FAIL'}")
    if err90 >= 1.0e-12:
        _fail("the 90 deg rotation is not the expected swap.")

    #  45 deg: Jones's closed forms.  This is the case that would expose a
    #  plain-tensor T instead of the Reuter-adjusted one, because the two
    #  differ by factors of two in exactly the 16/26/66 entries.
    r45 = rotate_constitutive_model(model, 45.0)
    expected45 = np.array([
        [(D11 + D22 + 2 * D12) / 4 + D66,
         (D11 + D22 - 4 * D66) / 4 + D12 / 2,
         (D11 - D22) / 4],
        [(D11 + D22 - 4 * D66) / 4 + D12 / 2,
         (D11 + D22 + 2 * D12) / 4 + D66,
         (D11 - D22) / 4],
        [(D11 - D22) / 4,
         (D11 - D22) / 4,
         (D11 + D22 - 2 * D12) / 4],
    ])
    err45 = float(np.abs(r45.material_matrix - expected45).max())
    print(f"  45 deg matches Jones's closed forms      {err45:.2e}         "
          f"{'OK' if err45 < 1e-11 else 'FAIL'}")
    if err45 >= 1.0e-11:
        _fail("the 45 deg rotation does not match the closed forms -- the "
              "engineering-shear (Reuter) factor of two is the usual cause.")

    #  Round trip.
    back = rotate_constitutive_model(
        rotate_constitutive_model(model, 37.5), -37.5)
    err_rt = float(np.abs(back.material_matrix - model.material_matrix).max())
    print(f"  rotate(+37.5) then rotate(-37.5)         {err_rt:.2e}         "
          f"{'OK' if err_rt < 1e-11 else 'FAIL'}")
    if err_rt >= 1.0e-11:
        _fail("the rotation does not round-trip.")

    #  The property the whole design rests on: the rotated basis is still a
    #  constant decomposition, so D(theta) = sum_k theta_k basis_k survives.
    worst_lin = 0.0
    for angle in (13.0, 45.0, 90.0, -60.0, 122.5):
        rotated = rotate_constitutive_model(model, angle)
        recon = np.einsum("k,kij->ij",
                          rotated.parameters, rotated.basis_matrices)
        worst_lin = max(worst_lin,
                        float(np.abs(recon - rotated.material_matrix).max()))
    print(f"  D(theta) = sum theta_k basis_k stays true {worst_lin:.2e}         "
          f"{'OK' if worst_lin < 1e-11 else 'FAIL'}")
    if worst_lin >= 1.0e-11:
        _fail("the rotated basis is no longer a constant decomposition.")

    #  Positive definiteness survives (Sylvester's law of inertia -- this is a
    #  congruence, not a similarity, so the EIGENVALUES do move).
    worst_min_eig = min(
        float(np.linalg.eigvalsh(
            rotate_constitutive_model(model, a).material_matrix).min())
        for a in (0.0, 13.0, 45.0, 90.0, 122.5))
    print(f"  rotated D stays positive definite         min eig "
          f"{worst_min_eig:7.3f}   {'OK' if worst_min_eig > 0 else 'FAIL'}")
    if worst_min_eig <= 0.0:
        _fail("a rotated D lost positive definiteness.")

    #  And the invariance the routes are equivalent BY.
    rng = np.random.default_rng(20260817)
    worst_work = 0.0
    for _ in range(200):
        k_lab, ks_lab = rng.normal(size=3), rng.normal(size=3)
        for angle in (13.0, 45.0, 90.0, -60.0):
            T = engineering_rotation_matrix(angle)
            w_mat = (T @ ks_lab) @ model.material_matrix @ (T @ k_lab)
            w_lab = (ks_lab
                     @ rotate_constitutive_model(model, angle).material_matrix
                     @ k_lab)
            worst_work = max(worst_work, abs(w_mat - w_lab))
    print(f"  virtual work is frame-invariant           {worst_work:.2e}         "
          f"{'OK' if worst_work < 1e-11 else 'FAIL'}")
    if worst_work >= 1.0e-11:
        _fail("internal virtual work is not invariant under the rotation.")


# =============================================================================
#  Test 2 -- the two routes on the real transverse coupon
# =============================================================================

def _run_single_experiment(test_spec, weighting="gls"):
    """Identify from one test's nine load steps, as one experiment."""
    cases = build_cases(test_spec, verbose=True)
    identification, linear_algebra = _settings()
    result = run_gls_identification_cases(
        cases=cases,
        material_type=MATERIAL_TYPE,
        analysis_type=ANALYSIS_TYPE,
        initial_parameters=INITIAL_PARAMETERS,
        identification_settings=identification,
        linear_algebra_settings=linear_algebra,
        experiment_ids=[test_spec.experiment_id] * len(cases),
        gls_settings=GlsSettings(basis_degree=6, weighting=weighting),
        verbose=False,
    )
    return cases, result


def _compare(name, theta_a, theta_b, tolerance=THETA_TOLERANCE) -> float:
    scale = max(float(np.abs(theta_a).max()), np.finfo(float).tiny)
    relative = float(np.abs(theta_a - theta_b).max()) / scale
    print(f"\n  {name}")
    print("        " + "".join(f"{label:>14s}" for label in LABELS))
    print("    A:  " + "".join(f"{v:14.8f}" for v in theta_a))
    print("    B:  " + "".join(f"{v:14.8f}" for v in theta_b))
    print(f"    max relative difference = {relative:.3e}   "
          f"(tolerance {tolerance:.0e})   "
          f"{'OK' if relative <= tolerance else 'FAIL'}")
    return relative


def test_transverse_equivalence() -> None:
    print()
    print("-" * 78)
    print("  2. the transverse coupon, identified both ways")
    print("-" * 78)
    print("  route A: fields pre-rotated +90 deg by the export, D unrotated")
    print("  route B: fields raw, D rotated by material_orientation_deg = 90")
    print()

    #  Both datasets must be in the frame their TestSpec claims, checked from
    #  the .mat rather than trusted.
    assert_orientation(TRANSVERSE)
    assert_orientation(TRANSVERSE_RAW)
    print()

    cases_a, result_a = _run_single_experiment(TRANSVERSE)
    cases_b, result_b = _run_single_experiment(TRANSVERSE_RAW)

    #  The two routes describe the SAME coupon in frames 90 deg apart, so their
    #  declared orientations must differ by exactly that -- whatever absolute
    #  convention the campaign currently uses.
    angle_a = cases_a[0].material_orientation_deg
    angle_b = cases_b[0].material_orientation_deg
    if angle_a != TRANSVERSE.material_orientation_deg:
        _fail(f"route A carries {angle_a} deg, not the TestSpec's "
              f"{TRANSVERSE.material_orientation_deg}.")
    if angle_b != TRANSVERSE_RAW.material_orientation_deg:
        _fail(f"route B carries {angle_b} deg, not the TestSpec's "
              f"{TRANSVERSE_RAW.material_orientation_deg}.")
    if abs((angle_a - angle_b) % 180.0) != 90.0:
        _fail(f"the pre-rotated and raw frames must sit 90 deg apart; got "
              f"{angle_a} and {angle_b}.")

    #  A 90 deg rotated D is still sparse, so the inactive-DOF pruning that
    #  drives the constraint nullity should land identically.  If it does not,
    #  the two runs are not solving the same discrete problem and the theta
    #  agreement below would be luck rather than proof.
    #  R is the nullspace basis of the kinematic constraints, so its column
    #  count IS the nullity, and it sets the size of the saddle-point system.
    nullity_a = int(result_a.optimal_field_results[0].R.shape[1])
    nullity_b = int(result_b.optimal_field_results[0].R.shape[1])
    print(f"\n  constraint nullity   A = {nullity_a}, B = {nullity_b}   "
          f"{'OK' if nullity_a == nullity_b else 'DIFFERENT'}")
    if nullity_a != nullity_b:
        _fail(
            f"the two routes build different virtual-field spaces "
            f"(nullity {nullity_a} vs {nullity_b}). The theta agreement below "
            "would not mean what it claims; investigate before trusting it.")

    relative = _compare("FGLS, transverse alone",
                        result_a.final_parameters, result_b.final_parameters)
    if relative > THETA_TOLERANCE:
        _fail(
            "the two routes disagree. Either the fields still need rotating "
            "(they should not), or the rotation convention/sign is wrong.")

    #  The same check through the OLS path, which is a different assembly and
    #  would not catch the same mistakes.
    identification, linear_algebra = _settings()
    ols_a = run_identification_cases(
        cases=cases_a, material_type=MATERIAL_TYPE,
        analysis_type=ANALYSIS_TYPE, initial_parameters=INITIAL_PARAMETERS,
        identification_settings=identification,
        linear_algebra_settings=linear_algebra, verbose=False)
    ols_b = run_identification_cases(
        cases=cases_b, material_type=MATERIAL_TYPE,
        analysis_type=ANALYSIS_TYPE, initial_parameters=INITIAL_PARAMETERS,
        identification_settings=identification,
        linear_algebra_settings=linear_algebra, verbose=False)
    relative_ols = _compare("OLS, transverse alone",
                            ols_a.final_parameters, ols_b.final_parameters)
    if relative_ols > THETA_TOLERANCE:
        _fail("the OLS path disagrees between the two routes.")


# =============================================================================
#  Test 3 -- the whole campaign, both ways
# =============================================================================

def test_campaign_equivalence() -> None:
    print()
    print("-" * 78)
    print("  3. the full two-coupon campaign, both ways")
    print("-" * 78)

    identification, linear_algebra = _settings()

    def run(transverse_spec):
        cases = (build_cases(LONGITUDINAL, verbose=False)
                 + build_cases(transverse_spec, verbose=False))
        ids = ([LONGITUDINAL.experiment_id] * 9
               + [transverse_spec.experiment_id] * 9)
        return run_gls_identification_cases(
            cases=cases, material_type=MATERIAL_TYPE,
            analysis_type=ANALYSIS_TYPE,
            initial_parameters=INITIAL_PARAMETERS,
            identification_settings=identification,
            linear_algebra_settings=linear_algebra,
            experiment_ids=ids,
            gls_settings=GlsSettings(basis_degree=6, weighting="gls"),
            verbose=False)

    result_a = run(TRANSVERSE)
    result_b = run(TRANSVERSE_RAW)
    relative = _compare("FGLS, longitudinal + transverse",
                        result_a.final_parameters, result_b.final_parameters)
    if relative > THETA_TOLERANCE:
        _fail("the full campaign disagrees between the two routes.")

    #  The published numbers, reproduced through the new code path.
    #  Both-cuts BC form, material 1-axis along the LONGITUDINAL coupon, so
    #  D11 is the longitudinal stiffness and D22 the transverse one.
    published = {"D11": 174.35, "D22": 61.43}
    print("\n  against the published campaign result:")
    for label, expected in published.items():
        index = LABELS.index(label)
        got = float(result_b.final_parameters[index])
        drift = abs(got - expected)
        print(f"    {label} = {got:8.4f}   published {expected:8.2f}   "
              f"|drift| = {drift:.4f}   "
              f"{'OK' if drift < 0.01 else 'CHECK'}")
        if drift >= 0.01:
            _fail(f"{label} moved by {drift:.4f} from the published value.")


def main() -> int:
    print("=" * 78)
    print("  ROTATING D vs ROTATING THE FIELDS -- equivalence on real data")
    print("=" * 78)
    test_algebra()
    test_transverse_equivalence()
    test_campaign_equivalence()
    print()
    print("=" * 78)
    print("  ALL CHECKS PASSED")
    print("  The fields do NOT need rotating when D is rotated -- the two are")
    print("  the same identification, and doing both would rotate twice.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
