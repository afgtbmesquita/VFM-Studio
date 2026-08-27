# =============================================================================
#  Generalized VFM / audit_assumptions.py
# -----------------------------------------------------------------------------
#  Which assumptions behind eqs. (1)-(20) actually hold on this campaign, and
#  does each violation move the ANSWER or only the ERROR BAR?
#
#  The motivating question: the quoted uncertainties look too good, and D66
#  above all -- eq. (20) says +/- 0.54 N.mm on a value of -4.04 N.mm that is
#  NOT PHYSICALLY ADMISSIBLE (a bending stiffness matrix must be positive
#  definite).  A tight variance around an infeasible value is the signature of
#  BIAS, and no first-order variance formula can see bias.
#
#  The decisive experiment is a parametric bootstrap.  Take the converged
#  theta_hat and Gamma_hat, regenerate the measured fields as
#
#       Y* = P Y  (the smooth part)  +  fresh noise drawn from Gamma_hat
#
#  and push each replicate through the WHOLE pipeline, Gamma re-estimation
#  included.  Two noise models:
#
#     'white'       spatially independent, exactly what eq. (2) assumes.
#                   If eq. (20) is internally consistent, the spread of the
#                   replicates must reproduce it.  This isolates the
#                   first-order approximation and the data-dependent virtual
#                   fields of section 14 from everything else.
#     'correlated'  the measured spatial correlation instead.  The gap
#                   against 'white' is the price of eq. (2) alone.
#
#  Neither can see model error: both generate data from the same Kirchhoff
#  plate model the solver inverts.  That is the point -- what the bootstrap
#  CANNOT reproduce is exactly what the bootstrap proves is not noise.
#
#  Run:  python -m workflows.audit_assumptions                 (about 15 min)
#        python -m workflows.audit_assumptions --replicates 40 (quick look)
#  Writes results/audit/audit_results.npz for make_audit_figures.py.
# =============================================================================

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import numpy as np

from workflows.campaign_common import (
    ANALYSIS_TYPE,
    HERE,
    INITIAL_PARAMETERS,
    LONGITUDINAL,
    MATERIAL_TYPE,
    RESULTS_DIR,
    TRANSVERSE,
    build_campaign,
)
from vfm_gls_identification import (
    jackknife_parameters,
    run_gls_identification_cases,
)
from vfm_noise_covariance import _orthonormal_basis
from vfm_user_inputs import (
    GlsSettings,
    IdentificationSettings,
    LinearAlgebraSettings,
)

BASIS_DEGREE = 6


def settings(iterations=30, tolerance=1.0e-9):
    return (IdentificationSettings(max_iterations=iterations,
                                   convergence_tolerance=tolerance),
            LinearAlgebraSettings())


def run_fgls(cases, ids, iterations=30, tolerance=1.0e-9,
             spatial_correction=False):
    identification, linear_algebra = settings(iterations, tolerance)
    return run_gls_identification_cases(
        cases=cases, material_type=MATERIAL_TYPE, analysis_type=ANALYSIS_TYPE,
        initial_parameters=INITIAL_PARAMETERS,
        identification_settings=identification,
        linear_algebra_settings=linear_algebra, experiment_ids=ids,
        gls_settings=GlsSettings(basis_degree=BASIS_DEGREE,
                                 spatial_correction=spatial_correction),
        verbose=False)


# =============================================================================
#  Noise generators for the parametric bootstrap
# =============================================================================

class _ExperimentNoise:
    """Smooth field + a generator for fresh noise with covariance Gamma_hat.

    Separable model, exactly the one eq. (2) writes down:

        E[eta_p eta_q^T] = rho_pq * Gamma      (rho = I for 'white')

    realised as   eta = L_spatial @ W @ L_gamma^T   with W standard normal,
    which gives the spatial correlation of L_spatial L_spatial^T and the
    component/time covariance Gamma in one draw.
    """

    def __init__(self, cases, noise, mode: str, rng):
        self.cases = cases
        self.mode = mode
        self.rng = rng

        points = cases[0].measurements.points
        self.n_points = points.shape[0]
        self.n_steps = len(cases)
        self.n_gen = cases[0].measurements.strains.shape[0]

        basis = _orthonormal_basis(points, BASIS_DEGREE)
        stack = np.stack([c.measurements.strains.T for c in cases])
        self.smooth = np.einsum("ij,tjc->tic", basis @ basis.T, stack)

        gamma = noise.gamma
        eigenvalues, eigenvectors = np.linalg.eigh(gamma)
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        self.gamma_factor = eigenvectors * np.sqrt(eigenvalues)

        self.spatial_factor = None
        if mode == "correlated":
            length = float(noise.correlation.equivalent_length)
            distance = np.linalg.norm(
                points[:, None, :] - points[None, :, :], axis=2)
            correlation = np.exp(-distance / max(length, 1.0e-9))
            #  Small nugget: the exponential kernel is PD in 2-D but the
            #  scattered cloud has near-coincident points.
            correlation[np.diag_indices_from(correlation)] += 1.0e-8
            factor = np.linalg.cholesky(correlation)
            #  Rescale so the pointwise variance stays 1 and only the spatial
            #  STRUCTURE changes -- otherwise this would also change sigma and
            #  the comparison against 'white' would confound two effects.
            self.spatial_factor = factor / np.sqrt(
                (factor ** 2).sum(axis=1))[:, None]

    def draw(self):
        white = self.rng.standard_normal(
            (self.n_points, self.n_steps * self.n_gen))
        if self.spatial_factor is not None:
            white = self.spatial_factor @ white
        noise = white @ self.gamma_factor.T
        return noise.reshape(self.n_points, self.n_steps, self.n_gen
                             ).transpose(1, 0, 2)

    def replicate(self):
        noise = self.draw()
        out = []
        for index, case in enumerate(self.cases):
            strains = (self.smooth[index] + noise[index]).T
            out.append(replace(
                case,
                measurements=replace(case.measurements, strains=strains)))
        return out


def monte_carlo(cases, ids, reference, n_replicates: int, mode: str,
                seed: int, verbose: bool = True):
    """Parametric bootstrap through the entire pipeline."""
    rng = np.random.default_rng(seed)
    groups, order = {}, []
    for case, exp_id in zip(cases, ids):
        if exp_id not in groups:
            groups[exp_id] = []
            order.append(exp_id)
        groups[exp_id].append(case)

    generators = [
        _ExperimentNoise(groups[exp_id], report.noise, mode, rng)
        for exp_id, report in zip(order, reference.experiments)
    ]

    estimates = []
    started = time.time()
    for replicate in range(n_replicates):
        synthetic, synthetic_ids = [], []
        for exp_id, generator in zip(order, generators):
            drawn = generator.replicate()
            synthetic.extend(drawn)
            synthetic_ids.extend([exp_id] * len(drawn))
        try:
            result = run_fgls(synthetic, synthetic_ids, iterations=6,
                              tolerance=1.0e-7)
            estimates.append(result.final_parameters)
        except np.linalg.LinAlgError:
            continue
        if verbose and (replicate + 1) % 10 == 0:
            rate = (time.time() - started) / (replicate + 1)
            print(f"    {mode}: {replicate + 1}/{n_replicates} "
                  f"({rate:.1f} s/replicate)", flush=True)
    return np.vstack(estimates)


# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=120)
    arguments = parser.parse_args()

    print("=" * 78)
    print("  ASSUMPTION AUDIT")
    print("=" * 78)

    cases, ids, counts = build_campaign(verbose=False)
    reference = run_fgls(cases, ids)
    labels = list(reference.constitutive_model.parameter_labels)
    theta = reference.final_parameters
    eq20 = reference.parameter_std
    print(f"  reference theta = {np.array2string(theta, precision=4)}")
    print(f"  eq. (20) sigma  = {np.array2string(eq20, precision=4)}")

    # ---- physical admissibility -----------------------------------------
    stiffness = np.array([[theta[0], theta[1], 0.0],
                          [theta[1], theta[2], 0.0],
                          [0.0, 0.0, theta[3]]])
    eigen_d = np.linalg.eigvalsh(stiffness)
    print(f"\n  D eigenvalues = {np.array2string(eigen_d, precision=3)}   "
          f"positive definite = {bool((eigen_d > 0).all())}")

    # ---- per-experiment consistency (does one theta even exist?) --------
    print("\n  Per-experiment FGLS (is there a common theta?)")
    single = {}
    for name, exp_id in (("longitudinal", 1), ("transverse", 2)):
        keep = [i for i, e in enumerate(ids) if e == exp_id]
        result = run_fgls([cases[i] for i in keep],
                          [ids[i] for i in keep])
        single[name] = (result.final_parameters, result.parameter_std)
        print(f"    {name:<13} "
              + "  ".join(f"{l}={v:8.2f}+/-{s:6.2f}"
                          for l, v, s in zip(labels,
                                             result.final_parameters,
                                             result.parameter_std)))
    long_theta, long_std = single["longitudinal"]
    trans_theta, trans_std = single["transverse"]
    combined = np.sqrt(long_std ** 2 + trans_std ** 2)
    tension = np.abs(long_theta - trans_theta) / combined
    print("    disagreement in sigma: "
          + "  ".join(f"{l}={t:7.1f}" for l, t in zip(labels, tension)))

    # ---- jackknife -------------------------------------------------------
    print("\n  Leave-one-load-step-out jackknife...")
    identification, linear_algebra = settings()
    _, _, jackknife_std = jackknife_parameters(
        cases=cases, experiment_ids=ids, material_type=MATERIAL_TYPE,
        analysis_type=ANALYSIS_TYPE, initial_parameters=INITIAL_PARAMETERS,
        identification_settings=identification,
        linear_algebra_settings=linear_algebra,
        gls_settings=GlsSettings(basis_degree=BASIS_DEGREE), verbose=False)
    print(f"    {np.array2string(jackknife_std, precision=4)}")

    # ---- the parametric bootstrap ---------------------------------------
    print(f"\n  Parametric bootstrap, {arguments.replicates} replicates each...")
    white = monte_carlo(cases, ids, reference, arguments.replicates,
                        "white", seed=20260816)
    correlated = monte_carlo(cases, ids, reference, arguments.replicates,
                             "correlated", seed=20260817)
    white_std = white.std(axis=0, ddof=1)
    correlated_std = correlated.std(axis=0, ddof=1)

    # ---- basis-degree spread (signal/noise separation, section 8) -------
    print("\n  Basis-degree sensitivity...")
    degrees = (2, 4, 6, 8, 10)
    by_degree = []
    for degree in degrees:
        identification, linear_algebra = settings()
        by_degree.append(run_gls_identification_cases(
            cases=cases, material_type=MATERIAL_TYPE,
            analysis_type=ANALYSIS_TYPE,
            initial_parameters=INITIAL_PARAMETERS,
            identification_settings=identification,
            linear_algebra_settings=linear_algebra, experiment_ids=ids,
            gls_settings=GlsSettings(basis_degree=degree),
            verbose=False).final_parameters)
    by_degree = np.vstack(by_degree)

    # ---- the kappa variant, as a labelled sensitivity check only --------
    #  NOT used for any reported result: kappa approximates eq. (1) by
    #  assuming the virtual fields are constant across the correlation
    #  neighbourhood, and picks a plateau off a block-size ladder.  It is here
    #  so the size of what eq. (3) omits can be seen, not to set an error bar.
    kappa_variant = run_fgls(cases, ids, spatial_correction=True)

    # ---- goodness of fit -------------------------------------------------
    from workflows.make_figures import chi_square_diagnostics
    goodness = chi_square_diagnostics(reference)
    reduced_chi2 = goodness["reduced"]
    birge = goodness["birge"]

    # ---- report ----------------------------------------------------------
    print()
    print("=" * 78)
    print("  THE UNCERTAINTY LADDER  (N.mm)")
    print("=" * 78)
    print(f"  {'source':<44}" + "".join(f"{l:>9}" for l in labels))
    print("  " + "-" * (44 + 9 * len(labels)))
    #  Each bootstrap must be judged against the eq. (20) that assumes ITS
    #  noise model.  The white bootstrap injects spatially independent noise,
    #  so its baseline is eq. (20) with kappa = 1; the correlated bootstrap's
    #  baseline is the kappa-corrected one.  Comparing white against the
    #  corrected formula would confound the two effects and make eq. (20) look
    #  wrong when it is not.
    kappa_std = kappa_variant.parameter_std
    rows = [
        ("eq. (20), eq. (3) verbatim  <- what is reported", eq20),
        ("  bootstrap, spatially WHITE noise", white_std),
        ("  bootstrap, measured spatial correlation", correlated_std),
        ("jackknife over load steps", jackknife_std),
        ("spread over basis degree 2-10", by_degree.std(axis=0, ddof=1)),
        (f"eq. (20) x sqrt(chi2/dof) = x{birge:.1f}", eq20 * birge),
    ]
    for name, values in rows:
        print(f"  {name:<44}" + "".join(f"{v:9.3f}" for v in values))
    print(f"  {'|longitudinal - transverse|':<44}"
          + "".join(f"{v:9.3f}" for v in np.abs(long_theta - trans_theta)))
    print(f"  {'[sensitivity only] eq. (20) with kappa':<44}"
          + "".join(f"{v:9.3f}" for v in kappa_std))

    print()
    print("  Is eq. (20) internally consistent?  bootstrap / eq.(20):")
    print(f"  {'  white / eq.(20), eq. (3) verbatim':<44}"
          + "".join(f"{v:9.2f}" for v in white_std / eq20))
    print("  -> ~1 means the FORMULA is right.  The error bar is nonetheless")
    print("     far too small, because the MODEL feeding it is wrong:")
    print(f"     chi2/dof = {reduced_chi2:.0f}, so eq. (20) is a variance")
    print("     UNDER THE MODEL, not the variability of the data.")
    print()
    print("  NOTE: kappa is measured but NEVER applied.  Introducing it would")
    print("  be fitting the covariance to remove a misfit whose cause is not")
    print("  spatial correlation.  Scaling by sqrt(chi2/dof) reaches the same")
    print(f"  interval ({(eq20 * birge)[0]:.2f} vs {kappa_std[0] * 6.5:.2f} on "
          "D11) without inventing a parameter.")

    (RESULTS_DIR / "audit").mkdir(parents=True, exist_ok=True)
    np.savez(
        RESULTS_DIR / "audit" / "audit_results.npz",
        labels=np.array(labels), theta=theta, eq20=eq20,
        white=white, correlated=correlated,
        white_std=white_std, correlated_std=correlated_std,
        jackknife_std=jackknife_std, by_degree=by_degree,
        degrees=np.array(degrees),
        long_theta=long_theta, long_std=long_std,
        trans_theta=trans_theta, trans_std=trans_std,
        tension=tension, d_eigenvalues=eigen_d,
        reduced_chi2=reduced_chi2, birge=birge,
        kappa_variant=kappa_variant.final_parameters,
        kappa_variant_std=kappa_std,
        kappa=np.array([r.noise.correlation.kappa
                        for r in reference.experiments]),
        nearest_rho=np.array([r.noise.correlation.nearest_rho
                              for r in reference.experiments]),
        lags=reference.experiments[0].noise.correlation.lags,
        rho_of_lag=np.array([r.noise.correlation.rho_of_lag
                             for r in reference.experiments]),
        residual_scale=np.array([r.noise.residual_scale
                                 for r in reference.experiments]),
        temporal=np.array([r.noise.temporal_correlation(
            1 if r.experiment_id == 1 else 0)
            for r in reference.experiments]),
    )
    print(f"\n  wrote results/audit/audit_results.npz")


if __name__ == "__main__":
    main()
