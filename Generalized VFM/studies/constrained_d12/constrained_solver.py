# =============================================================================
#  Generalized VFM / studies / constrained_d12 / constrained_solver.py
# -----------------------------------------------------------------------------
#  FGLS with D12 >= 0 imposed.  Closed form, no optimiser.
#
#  THE FORMULA
#  ---------------------------------------------------------------------------
#  The GLS objective is quadratic,
#
#      Q(theta) = (H theta - b)^T C^-1 (H theta - b),
#
#  so with ONE inequality constraint theta_k >= 0 the KKT conditions have only
#  two cases:
#
#      unconstrained theta_k >= 0   ->  constraint inactive, answer unchanged
#      unconstrained theta_k <  0   ->  constraint ACTIVE, theta_k = 0
#
#  and with the active set known the solution is the orthogonal projection of
#  the unconstrained estimate onto {theta_k = 0} IN THE INFORMATION METRIC:
#
#      Sigma      = (H^T C^-1 H)^-1                  (the eq. (20) covariance)
#      theta_c    = theta_hat - (theta_hat_k / Sigma_kk) * Sigma[:, k]
#
#  That is exact for a quadratic objective -- it is the same answer as deleting
#  column k and re-solving, and it is the standard conditional-mean formula for
#  a Gaussian, which is what the GLS objective makes theta.  The Lagrange
#  multiplier is  lambda = theta_hat_k / Sigma_kk,  and the cost of the
#  constraint is
#
#      Delta Q = theta_hat_k^2 / Sigma_kk
#
#  which is distributed as chi^2 with 1 dof under H0: D12 = 0.  So the SAME
#  formula that imposes the constraint also tests whether the data ever
#  objected to it.
#
#  C depends on theta, so the projection is applied INSIDE the FGLS loop:
#  re-form C at the constrained theta, re-project, repeat.  Convergence is in
#  2-3 passes.
#
#      python -m studies.constrained_d12.constrained_solver
# =============================================================================

from __future__ import annotations

import numpy as np

from workflows.campaign_common import (
    ANALYSIS_TYPE, INITIAL_PARAMETERS, MATERIAL_TYPE, TESTS, TESTS_WITH_45,
    build_campaign,
)
from workflows.make_figures import chi_square_diagnostics
from vfm_gls_identification import run_gls_identification_cases
from vfm_user_inputs import (GlsSettings, IdentificationSettings,
                             LinearAlgebraSettings)

LABELS = ["D11", "D12", "D22", "D66"]
K_D12 = 1  # index of D12 in theta


def fgls(cases, ids, initial):
    return run_gls_identification_cases(
        cases=cases, material_type=MATERIAL_TYPE, analysis_type=ANALYSIS_TYPE,
        initial_parameters=list(initial),
        identification_settings=IdentificationSettings(
            max_iterations=40, convergence_tolerance=1.0e-8),
        linear_algebra_settings=LinearAlgebraSettings(),
        experiment_ids=ids, gls_settings=GlsSettings(), verbose=False)


def project(theta, covariance, k=K_D12, bound=0.0):
    """Closed-form KKT step: theta with theta_k pinned to `bound`."""
    theta = np.asarray(theta, dtype=float)
    gap = theta[k] - bound
    multiplier = gap / covariance[k, k]
    return theta - multiplier * covariance[:, k], multiplier, gap ** 2 / covariance[k, k]


def explicit_check(result, k=K_D12):
    """Same answer by deleting column k from the whitened system."""
    from scipy.linalg import cholesky, solve_triangular
    design, rhs, cov = result.K_matrix, result.U_vector, result.global_covariance
    factor = cholesky(cov, lower=True)
    a = solve_triangular(factor, design, lower=True)
    b = solve_triangular(factor, rhs, lower=True)
    keep = [j for j in range(design.shape[1]) if j != k]
    reduced, *_ = np.linalg.lstsq(a[:, keep], b, rcond=None)
    out = np.zeros(design.shape[1])
    out[keep] = reduced
    return out


def constrained_fgls(cases, ids, tag):
    theta = np.asarray(INITIAL_PARAMETERS, dtype=float)
    free = fgls(cases, ids, theta)
    theta_free = np.asarray(free.final_parameters, dtype=float)
    chi2_free = chi_square_diagnostics(free)["reduced"]

    active = theta_free[K_D12] < 0.0
    if not active:
        print(f"  {tag}: D12 = {theta_free[K_D12]:+.3f} >= 0 already -- "
              f"constraint inactive, answer unchanged.")
        return theta_free, theta_free, chi2_free, chi2_free, 0.0

    #  Re-form C at the constrained theta and re-project until it settles.
    theta_c = theta_free
    for _ in range(6):
        result = fgls(cases, ids, theta_c)
        theta_c_new, multiplier, cost = project(result.final_parameters,
                                                result.parameter_covariance)
        if np.max(np.abs(theta_c_new - theta_c)) < 1e-9:
            theta_c = theta_c_new
            break
        theta_c = theta_c_new

    check = explicit_check(result)
    agreement = np.max(np.abs(check - theta_c))

    #  chi2 of the constrained fit, using the same C.
    design, rhs, cov = (result.K_matrix, result.U_vector,
                        result.global_covariance)
    residual = design @ theta_c - rhs
    dof = design.shape[0] - (design.shape[1] - 1)   # one parameter removed
    chi2_c = float(residual @ np.linalg.solve(cov, residual)) / dof

    print()
    print(f"  {tag}")
    print("  " + "-" * 72)
    print(f"    {'':<22}" + "".join(f"{n:>11}" for n in LABELS)
          + f"{'chi2/dof':>12}")
    print(f"    {'unconstrained':<22}"
          + "".join(f"{v:11.3f}" for v in theta_free) + f"{chi2_free:12.0f}")
    print(f"    {'D12 >= 0 imposed':<22}"
          + "".join(f"{v:11.3f}" for v in theta_c) + f"{chi2_c:12.0f}")
    print(f"    {'shift':<22}"
          + "".join(f"{v:11.3f}" for v in theta_c - theta_free))
    print()
    print(f"    Lagrange multiplier      = {multiplier:.4g}")
    print(f"    cost of the constraint   = {cost:.1f}  "
          f"(chi2 with 1 dof under H0: D12 = 0)")
    print(f"    -> p-value               = "
          f"{np.exp(-0.5 * cost) if cost < 1400 else 0.0:.3g}   "
          f"(exp(-DeltaQ/2), one-sided)")
    #  DeltaQ above is computed with the eq. (20) covariance, which the data
    #  reject at chi2/dof = {chi2_free:.0f}.  The Birge-scaled statistic divides
    #  it by that misfit -- the same correction the README applies to every
    #  error bar -- and is the one to quote.
    scaled = cost / chi2_free
    from math import exp
    print(f"    Birge-scaled statistic   = {scaled:.2f}  = DeltaQ / (chi2/dof)")
    print(f"    -> p-value               = {exp(-0.5 * scaled):.3f}  "
          f"<- THE ONE TO QUOTE")
    print(f"    projection vs explicit column-deletion: {agreement:.2e}")
    return theta_free, theta_c, chi2_free, chi2_c, cost


def main() -> None:
    print("=" * 78)
    print("  FGLS WITH D12 >= 0  --  closed-form active-set solution")
    print("=" * 78)
    print("  theta_c = theta_hat - (theta_hat_k / Sigma_kk) * Sigma[:, k]")
    print("  exact for a quadratic objective; identical to deleting column k.")

    for tag, tests in (("2 coupons (longitudinal + transverse)", TESTS),
                       ("3 coupons (with 45 deg)", TESTS_WITH_45)):
        cases, ids, _ = build_campaign(tests, verbose=False)
        constrained_fgls(cases, ids, tag)


if __name__ == "__main__":
    main()
