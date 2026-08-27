# =============================================================================
#  Generalized VFM / run_campaign_gls.py
# -----------------------------------------------------------------------------
#  The real identification: longitudinal + transverse bending, one D, by FGLS.
#
#  READ validate_gls.py FIRST.  On a known-answer synthetic built at the
#  coupons' own measured biaxiality it shows that this estimator recovers D11
#  and D22 to ~0.1 % under correlated noise, where the naive stacked solve is
#  36 % and 54 % wrong.  Without that, the numbers below would be unfalsifiable.
#
#  WHAT IS DIFFERENT FROM ../stacked VFM solution
#  ---------------------------------------------------------------------------
#  That folder pooled the per-load-step estimates with per-parameter scalar
#  weights 1/(sigma_i eta_ik)^2, assuming sigma_longitudinal = sigma_transverse
#  and treating the nine load steps of each test as nine independent estimates.
#  Here the noise covariance is MEASURED from the fields, including its
#  correlation between load steps, and propagated through the actual virtual
#  fields into a dense equation-error covariance that is re-formed at every
#  iteration alongside the parameters.
#
#  Usage
#  -----
#      python -m workflows.run_campaign_gls                 # the identification
#      python -m workflows.run_campaign_gls --basis-sweep   # sensitivity to the noise basis
#      python -m workflows.run_campaign_gls --jackknife     # empirical error bars (slow)
#      python -m workflows.run_campaign_gls --all           # everything
# =============================================================================

from __future__ import annotations

import argparse
import csv

import numpy as np

from workflows.campaign_common import (
    ANALYSIS_TYPE,
    HERE,
    INITIAL_PARAMETERS,
    MATERIAL_TYPE,
    REFERENCE_RESULTS,
    RESULTS_DIR,
    TESTS,
    assert_orientation,
    build_campaign,
    derived_constants,
    report_parameters,
)
from vfm_constitutive import define_constitutive_model
from vfm_gls_identification import (
    jackknife_parameters,
    run_gls_identification_cases,
)
from vfm_identifiability import compute_identifiability_diagnostics
from vfm_identification import run_identification_cases
from vfm_user_inputs import (
    GlsSettings,
    IdentificationSettings,
    LinearAlgebraSettings,
)

BASIS_DEGREES = (2, 4, 6, 8, 10)
COMPONENT_NAMES = ("kappa_xx", "kappa_yy", "kappa_xy")


def _settings(iterations=40, tolerance=1.0e-8):
    return (IdentificationSettings(max_iterations=iterations,
                                   convergence_tolerance=tolerance),
            LinearAlgebraSettings())


def run_gls(cases, ids, weighting="gls", basis_degree=6,
            spatial_correction=False, verbose=False):
    identification, linear_algebra = _settings()
    return run_gls_identification_cases(
        cases=cases, material_type=MATERIAL_TYPE, analysis_type=ANALYSIS_TYPE,
        initial_parameters=INITIAL_PARAMETERS,
        identification_settings=identification,
        linear_algebra_settings=linear_algebra,
        experiment_ids=ids,
        gls_settings=GlsSettings(basis_degree=basis_degree,
                                 weighting=weighting,
                                 spatial_correction=spatial_correction),
        verbose=verbose,
    )


def run_ols(cases):
    identification, linear_algebra = _settings()
    return run_identification_cases(
        cases=cases, material_type=MATERIAL_TYPE, analysis_type=ANALYSIS_TYPE,
        initial_parameters=INITIAL_PARAMETERS,
        identification_settings=identification,
        linear_algebra_settings=linear_algebra, verbose=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis-sweep", action="store_true",
                        help="vary the smoothing-basis degree")
    parser.add_argument("--jackknife", action="store_true",
                        help="leave-one-load-step-out error bars (slow)")
    parser.add_argument("--all", action="store_true", help="every section")
    arguments = parser.parse_args()
    do_sweep = arguments.basis_sweep or arguments.all
    do_jackknife = arguments.jackknife or arguments.all

    print("=" * 78)
    print("  GENERALIZED VFM -- longitudinal + transverse, FGLS")
    print("=" * 78)
    print("  Orientation checks (read off the shipped .mat):")
    for test in TESTS:
        assert_orientation(test)

    print()
    print("  Cases:")
    cases, ids, counts = build_campaign(verbose=True)
    n_long = counts["longitudinal"]
    labels = list(define_constitutive_model(
        MATERIAL_TYPE, ANALYSIS_TYPE, INITIAL_PARAMETERS).parameter_labels)

    print()
    print("-" * 78)
    print("  THE IDENTIFICATION")
    print("-" * 78)
    result = run_gls(cases, ids, verbose=True)

    # ---- noise diagnostics ------------------------------------------------
    print()
    print("=" * 78)
    print("  THE MEASURED NOISE -- what the weighting is actually built on")
    print("=" * 78)
    for report in result.experiments:
        noise = report.noise
        correlation = noise.correlation
        print()
        print(f"  {report.label}  ({report.n_steps} load steps, "
              f"{report.n_points} points)")
        print(f"    sigma per load step and component (x1e3):")
        print(f"      {'step':>6}" + "".join(f"{n:>11}" for n in COMPONENT_NAMES))
        for step in range(noise.n_steps):
            print(f"      {step:>6}"
                  + "".join(f"{1e3 * v:11.4f}"
                            for v in noise.residual_scale[step]))
        print(f"    temporal correlation of the PRIMARY curvature "
              f"across load steps:")
        primary = 1 if "longitudinal" in report.label else 0
        temporal = noise.temporal_correlation(primary)
        for row in range(noise.n_steps):
            print("      " + "".join(f"{v:7.3f}" for v in temporal[row]))
        off_diagonal = temporal[np.triu_indices(noise.n_steps, k=1)]
        print(f"    mean off-diagonal = {off_diagonal.mean():.3f} "
              f"-- 0.0 would mean the load steps are independent estimates,")
        print(f"      which is what the earlier inverse-variance pooling "
              f"assumed.")
        print(f"    spatial: rho at the nearest neighbour "
              f"({correlation.point_spacing:.3f} mm) = "
              f"{correlation.nearest_rho:.3f}, kappa = {correlation.kappa:.2f}")
        print(f"    C conditioning: PD = "
              f"{report.covariance_diagnostics['positive_definite']}, "
              f"cond = {report.covariance_diagnostics['condition_number']:.2e}")

    # ---- side by side -----------------------------------------------------
    print()
    print("=" * 78)
    print("  SIDE BY SIDE")
    print("=" * 78)
    table = dict(REFERENCE_RESULTS)
    table["stacked naive (OLS, rerun)"] = run_ols(cases).final_parameters
    table["FGLS block-diagonal"] = run_gls(
        cases, ids, weighting="block_diagonal").final_parameters
    table["FGLS (this method, eq. 3 verbatim)"] = result.final_parameters
    #  Reported for transparency only: the solver never uses kappa.  See the
    #  README section "No kappa, and why".
    table["[sensitivity only] FGLS with kappa"] = run_gls(
        cases, ids, spatial_correction=True).final_parameters

    print(f"  {'run':<30}" + "".join(f"{n:>11}" for n in labels))
    print("  " + "-" * (30 + 11 * len(labels)))
    for title, values in table.items():
        print(f"  {title:<30}" + "".join(f"{v:11.2f}" for v in values))

    print()
    print("  The first four rows are the earlier folder's results, quoted for")
    print("  comparison.  THE REPORTED ROW is 'eq. (3) verbatim'.  The last")
    print("  row shows what applying kappa would do; it is never used, because")
    print("  kappa approximates eq. (1) with two judgement calls and a")
    print("  covariance the derivation does not contain must not set an error")
    print("  bar.  The two differ by ~2 % because a factor common to both")
    print("  experiments cancels exactly when every H_i is the identity.")

    # ---- the cross-check that matters -------------------------------------
    print()
    print("=" * 78)
    print("  THE CONSISTENCY CHECK")
    print("=" * 78)
    single = {"D11": REFERENCE_RESULTS["transverse alone (OLS)"][0],
              "D22": REFERENCE_RESULTS["longitudinal alone (OLS)"][2]}
    source = {"D11": "transverse coupon alone",
              "D22": "longitudinal coupon alone"}
    print("  Each stiffness has an independent single-test value.  Agreement")
    print("  between the two routes is itself a result.")
    print()
    print(f"  {'':<6}{'FGLS':>10}{'single test':>14}{'gap':>9}"
          f"{'  inverse-variance gap':>24}")
    for name in ("D11", "D22"):
        index = labels.index(name)
        gls_value = result.final_parameters[index]
        reference = single[name]
        gap = 100.0 * abs(gls_value - reference) / abs(reference)
        old = REFERENCE_RESULTS["stacked inverse-variance"][index]
        old_gap = 100.0 * abs(old - reference) / abs(reference)
        print(f"  {name:<6}{gls_value:10.2f}{reference:14.2f}{gap:8.1f}%"
              f"{old_gap:23.1f}%")
        print(f"        ({source[name]})")

    # ---- uncertainty ------------------------------------------------------
    print()
    print("=" * 78)
    print("  UNCERTAINTY")
    print("=" * 78)
    model_std = result.parameter_std
    print("  eq. (20), the model-based first-order GLS covariance:")
    for name, value, std in zip(labels, result.final_parameters, model_std):
        print(f"    {name:>6} = {value:10.3f} +/- {std:8.3f} N.mm "
              f"({100.0 * std / abs(value):.2f} %)")
    print()
    print()
    print("  READ THIS AS: 'if the ONLY error were spatially white DIC noise")
    print("  with covariance Gamma_hat, how far would theta move?'  It is NOT")
    print("  'how well is D_ij known?'.  It excludes the uncertainty in Gamma")
    print("  itself, the higher-order effect of data-dependent virtual fields,")
    print("  and ALL model error -- and model error dominates here.")

    jackknife_std = None
    if do_jackknife:
        print()
        print("  Leave-one-load-step-out jackknife (re-estimating Gamma inside")
        print("  every replicate, so it does propagate the noise model's own")
        print("  uncertainty):")
        identification, linear_algebra = _settings()
        estimates, mean, jackknife_std = jackknife_parameters(
            cases=cases, experiment_ids=ids, material_type=MATERIAL_TYPE,
            analysis_type=ANALYSIS_TYPE,
            initial_parameters=INITIAL_PARAMETERS,
            identification_settings=identification,
            linear_algebra_settings=linear_algebra,
            gls_settings=GlsSettings(), verbose=False)
        print(f"    {len(estimates)} replicates")
        for index, name in enumerate(labels):
            print(f"    {name:>6} = {result.final_parameters[index]:10.3f} "
                  f"+/- {jackknife_std[index]:8.3f} N.mm "
                  f"({100.0 * jackknife_std[index] / abs(result.final_parameters[index]):.2f} %)"
                  f"   [eq.20 said {model_std[index]:.3f}]")
        ratio = jackknife_std / np.maximum(model_std, 1e-30)
        print(f"    jackknife / eq.(20) = "
              + np.array2string(ratio, precision=1))
        print("    QUOTE THE JACKKNIFE.  The ratio is how much the noise model")
        print("    is carrying that eq. (20) cannot see.")
    else:
        print()
        print("  Run with --jackknife for the empirical interval to quote.")

    # ---- basis sweep ------------------------------------------------------
    if do_sweep:
        print()
        print("=" * 78)
        print("  SENSITIVITY TO THE SMOOTHING BASIS")
        print("=" * 78)
        print("  The basis separates smooth field from fluctuation when")
        print("  estimating Gamma.  Too coarse leaks real mechanics into the")
        print("  noise; too fine fits the noise.  If D11 and D22 barely move,")
        print("  the noise model is not carrying the result.")
        print()
        print(f"  {'degree':>8}{'d':>6}" + "".join(f"{n:>11}" for n in labels))
        for degree in BASIS_DEGREES:
            swept = run_gls(cases, ids, basis_degree=degree)
            n_basis = (degree + 1) * (degree + 2) // 2
            print(f"  {degree:>8}{n_basis:>6}"
                  + "".join(f"{v:11.2f}" for v in swept.final_parameters))

    # ---- identifiability --------------------------------------------------
    diagnostics = compute_identifiability_diagnostics(
        cases=cases,
        constitutive_model=define_constitutive_model(
            MATERIAL_TYPE, ANALYSIS_TYPE, result.final_parameters))
    aggregate = diagnostics.aggregate
    print()
    print("=" * 78)
    print("  IDENTIFIABILITY (Pierron H / CID), both tests pooled")
    print("=" * 78)
    print(f"  aggregate cond(H) = {aggregate.condition_number:.3e}")
    print("  singular values   = "
          + np.array2string(aggregate.singular_values, precision=3))
    print("  parameter correlation matrix:")
    print("           " + "".join(f"{n:>9}" for n in diagnostics.parameter_labels))
    for row, name in enumerate(diagnostics.parameter_labels):
        print(f"    {name:>6} " + "".join(
            f"{aggregate.correlation[row, column]:9.3f}"
            for column in range(len(diagnostics.parameter_labels))))

    # ---- headline ---------------------------------------------------------
    report_parameters(labels, result.final_parameters,
                      "FGLS -- THE RESULT", std=jackknife_std)
    print()
    derived_constants(labels, result.final_parameters)
    print()
    print("  QUOTE D11 and D22 ONLY.  D12 and D66 are not recovered by two")
    print("  near-cylindrical bending tests -- validate_gls.py shows the same")
    print("  failure on a synthetic built at these coupons' own biaxiality.")

    # ---- CSV --------------------------------------------------------------
    output_dir = RESULTS_DIR / "two_coupon"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "gls_results_orthotropic.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run"] + labels)
        for title, values in table.items():
            writer.writerow([title] + [f"{v:.6f}" for v in values])
        writer.writerow(["FGLS std eq.(20)"] + [f"{v:.6f}" for v in model_std])
        if jackknife_std is not None:
            writer.writerow(["FGLS std jackknife"]
                            + [f"{v:.6f}" for v in jackknife_std])
    print(f"\n  wrote {out_path.name}")

    step_path = output_dir / "gls_step_estimates.csv"
    with step_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["experiment", "step"] + [f"U_{n}" for n in labels])
        for report in result.experiments:
            for step, label in enumerate(report.step_labels):
                writer.writerow([report.label, label]
                                + [f"{v:.6f}"
                                   for v in report.step_estimates[step]])
    print(f"  wrote {step_path.name}")


if __name__ == "__main__":
    main()
