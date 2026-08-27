# =============================================================================
#  Generalized VFM / run_campaign_45.py
# -----------------------------------------------------------------------------
#  The THREE-COUPON identification: longitudinal + transverse + 45 deg, one D,
#  by FGLS.  A separate driver from run_campaign_gls.py on purpose -- that one
#  and every figure, project and CSV it feeds are left bit-identical, and the
#  two-coupon baseline is re-run here so the comparison is like for like rather
#  than quoted from a file.
#
#  WHY THIS COUPON WAS ADDED
#  ---------------------------------------------------------------------------
#  The README's own conclusion on the two-coupon campaign was:
#
#      "D12 and D66 are still not recovered, and this is not a shortcoming of
#       the estimator.  Both coupons bend into a near-cylindrical shape, so the
#       secondary curvature is at noise level and D12 -- which multiplies
#       exactly that curvature -- has almost nothing to act on.  Recovering
#       them needs an off-axis (+-45 deg) coupon, a torsion test, or genuinely
#       biaxial bending."
#
#  This is that off-axis coupon, and it runs through the code path the README
#  built for it: route (B), fields in their own measurement frame, D rotated by
#  material_orientation_deg = 45.
#
#  WHERE THE NEW INFORMATION COMES FROM -- READ THIS BEFORE THE NUMBERS
#  ---------------------------------------------------------------------------
#  At exactly 45 deg the D66 basis matrix rotates onto the NORMAL curvatures:
#
#      T(45)^T [[0,0,0],[0,0,0],[0,0,1]] T(45)  =  [[ 1,-1, 0],
#                                                   [-1, 1, 0],
#                                                   [ 0, 0, 0]]
#
#  so D_lab[0,0] = D_lab[1,1] = S + D66 and D_lab[0,1] = S - D66 with
#  S = (D11 + D22 + 2 D12)/4.  With the lateral edges traction-free (M_xx = 0),
#
#      kappa_xx / kappa_yy  =  -(S - D66) / (S + D66)
#
#  THE ANTICLASTIC RATIO IS THE D66 MEASUREMENT.  Measured on this coupon:
#  -0.3814 +/- 0.0173 over nine load steps, stable to 2 % across the whole trim
#  sweep.  On the two axis-aligned coupons the same ratio is -0.03 and -0.01,
#  where it is noise -- which is precisely why they carry no D66 information.
#
#  This matters because the coupon's TWIST is restrained by the fixture
#  (kappa_xy/kappa_yy = -0.022 against the ~-0.33 a free coupon would show), so
#  a D66 route that needed a measurable twist would not work here.  This one
#  does not need it.
#
#  Usage
#  -----
#      python -m workflows.run_campaign_45                    # the identification
#      python -m workflows.run_campaign_45 --orientation-check # 45 deg vs 135 deg cut sense
#      python -m workflows.run_campaign_45 --leave-one-out    # each coupon's contribution
#      python -m workflows.run_campaign_45 --basis-sweep      # noise-basis sensitivity
#      python -m workflows.run_campaign_45 --jackknife        # empirical error bars (slow)
#      python -m workflows.run_campaign_45 --all              # everything
# =============================================================================

from __future__ import annotations

import argparse
import csv
import dataclasses

import numpy as np

from workflows.campaign_common import (
    ANALYSIS_TYPE,
    FORTYFIVE,
    HERE,
    INITIAL_PARAMETERS,
    LONGITUDINAL,
    MATERIAL_TYPE,
    MATERIAL_TYPE as _MT,  # noqa: F401  (kept for symmetry with the sibling driver)
    RESULTS_DIR,
    TESTS,
    TESTS_WITH_45,
    TRANSVERSE,
    assert_orientation,
    build_campaign,
    derived_constants,
    report_parameters,
)
from workflows.make_figures import chi_square_diagnostics
from vfm_constitutive import define_constitutive_model, engineering_rotation_matrix
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
LABELS = ["D11", "D12", "D22", "D66"]
#  Must match SPECIMEN_WIDTH_MM in
#  ../45deg_bending/VFM_export/export_vfm_dataset.py -- the width the shipped
#  .mat were written with, and the point the sweep in open_inputs() pivots on.
WIDTH_PLACEHOLDER_MM = 9.28


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


def reduced_chi_square(result) -> float:
    try:
        return float(chi_square_diagnostics(result)["reduced"])
    except Exception:
        return float("nan")


def row(title, values, chi2=None) -> str:
    line = f"  {title:<34}" + "".join(f"{v:10.2f}" for v in values)
    if chi2 is not None:
        line += f"{chi2:12.0f}"
    return line


# =============================================================================
#  The cut-sense check: is the material 1-axis at +45 or +135 deg?
# =============================================================================

def orientation_check(cases, ids, reference) -> None:
    """Re-run the whole campaign with the 45 deg coupon's frame flipped.

    Both cut senses are geometrically possible.  Everything EVEN in the angle
    is identical between them -- T(135) differs from T(45) only in the sign of
    its third row and column, so the effective bending stiffness, the
    anticlastic ratio and hence D66 do not distinguish them.  Only the sign of
    the bend-twist coupling does, and on this coupon the twist is restrained by
    the fixture to ~1/15 of its free value.

    So rather than argue the sign, run it both ways and report the difference.
    """
    print()
    print("=" * 78)
    print("  CUT SENSE: IS THE MATERIAL 1-AXIS AT +45 OR +135 DEG?")
    print("=" * 78)
    print("  The coupon length runs along its own +y (90 deg from +x) and the")
    print("  material 1-axis is 45 deg from the length, so the 1-axis is at")
    print("  either 90-45 = 45 deg or 90+45 = 135 deg.  Both are geometrically")
    print("  possible; only the data can say which.")
    print()

    t45 = engineering_rotation_matrix(45.0)
    t135 = engineering_rotation_matrix(135.0)
    basis = np.zeros((3, 3))
    basis[2, 2] = 1.0
    print("  T(45)^T B66 T(45)  diag/offdiag = "
          f"{(t45.T @ basis @ t45)[0, 0]:+.2f} / {(t45.T @ basis @ t45)[0, 1]:+.2f}")
    print("  T(135)^T B66 T(135) diag/offdiag = "
          f"{(t135.T @ basis @ t135)[0, 0]:+.2f} / "
          f"{(t135.T @ basis @ t135)[0, 1]:+.2f}")
    print("    -> identical.  D66 cannot distinguish the two senses, by")
    print("       construction, so the ambiguity is harmless for it.")
    print()

    flipped = []
    for case, experiment in zip(cases, ids):
        if experiment == FORTYFIVE.experiment_id:
            case = dataclasses.replace(case, material_orientation_deg=135.0)
        flipped.append(case)
    other = run_gls(flipped, ids)

    print(row("phi = 45 deg  (shipped)", reference.final_parameters,
              reduced_chi_square(reference)))
    print(row("phi = 135 deg (other sense)", other.final_parameters,
              reduced_chi_square(other)))
    delta = np.abs(np.asarray(other.final_parameters)
                   - np.asarray(reference.final_parameters))
    scale = np.maximum(np.abs(reference.final_parameters), 1e-12)
    print(row("|difference|", delta))
    print(row("|difference| / |value|  [%]", 100.0 * delta / scale))
    print()
    worst = float(np.max(100.0 * delta / scale))
    if worst < 1.0:
        print(f"  The two senses agree to {worst:.2f} % on every coefficient, so")
        print("  the cut-sense ambiguity DOES NOT AFFECT THE REPORTED THETA.")
        print("  The shipped choice (+45) is the one the measured twist sign")
        print("  supports -- kappa_xy/kappa_yy < 0 at all nine load steps -- but")
        print("  nothing rests on it.")
    else:
        print(f"  The two senses differ by up to {worst:.1f} %.  The cut sense")
        print("  MUST then be established independently (fixture photographs,")
        print("  or the sign of the machine Tx channel) before quoting theta.")


# =============================================================================
#  The two open inputs of this coupon, swept together
# =============================================================================

def open_inputs(cases, ids) -> None:
    """The 45 deg coupon carries two inputs that were not measured.

    (1) THE SPECIMEN WIDTH.  9.28 mm is a placeholder; the DIC field spans
        9.00 mm and the two measured coupons ran 1.007x and 1.038x their DIC
        span, so the true value is bracketed by 9.06-9.34 mm.  Every moment
        resultant scales as 1/width.

    (2) THE CUT SENSE, +45 or +135 deg.  See orientation_check().

    Neither is a statistical uncertainty -- no covariance can see either -- so
    the honest treatment is an envelope over both, reported alongside theta.
    """
    print()
    print("=" * 78)
    print("  THE TWO OPEN INPUTS OF THE 45 DEG COUPON")
    print("=" * 78)

    def variant(width_scale=1.0, angle=45.0):
        adjusted = []
        for case, experiment in zip(cases, ids):
            if experiment == FORTYFIVE.experiment_id:
                measurements = case.measurements
                force = (np.asarray(measurements.applied_force, dtype=float)
                         * width_scale)
                case = dataclasses.replace(
                    case,
                    measurements=dataclasses.replace(measurements,
                                                     applied_force=force),
                    material_orientation_deg=angle)
            adjusted.append(case)
        return run_gls(adjusted, ids)

    print("  (1) SPECIMEN WIDTH -- placeholder 9.28 mm, M scales as 1/width")
    print(f"  {'assumed width':<34}" + "".join(f"{n:>10}" for n in LABELS))
    print("  " + "-" * (34 + 10 * len(LABELS)))
    for width in (9.00, 9.14, 9.28, 9.45, 9.60):
        result = variant(width_scale=WIDTH_PLACEHOLDER_MM / width)
        tag = f"{width:.2f} mm" + ("   [SHIPPED]"
                                   if abs(width - WIDTH_PLACEHOLDER_MM) < 1e-9
                                   else "")
        print(row(tag, result.final_parameters))

    print()
    print("  (2) BOTH TOGETHER -- the four corners of the plausible box")
    print(f"  {'variant':<34}" + "".join(f"{n:>10}" for n in LABELS))
    print("  " + "-" * (34 + 10 * len(LABELS)))
    corners = []
    for width in (9.06, 9.34):
        for angle in (45.0, 135.0):
            result = variant(width_scale=WIDTH_PLACEHOLDER_MM / width,
                             angle=angle)
            corners.append(np.asarray(result.final_parameters))
            print(row(f"w = {width:.2f} mm,  phi = {angle:.0f} deg",
                      result.final_parameters))
    corners = np.array(corners)
    low, high = corners.min(axis=0), corners.max(axis=0)
    middle = 0.5 * (low + high)
    print()
    print(row("envelope min", low))
    print(row("envelope max", high))
    print(row("span [% of midpoint]",
              100.0 * (high - low) / np.maximum(np.abs(middle), 1e-12)))
    print()
    if float(low[LABELS.index("D66")]) > 0.0:
        print("  D66 STAYS POSITIVE over the whole envelope.  That is the")
        print("  robust result: whatever the width and whichever the cut")
        print("  sense, adding this coupon moves D66 out of the physically")
        print("  impossible negative region the two-coupon campaign put it in.")


# =============================================================================
#  Leave-one-coupon-out: what does each test actually contribute?
# =============================================================================

def leave_one_out(reference) -> None:
    print()
    print("=" * 78)
    print("  WHAT EACH COUPON CONTRIBUTES")
    print("=" * 78)
    print("  Each row drops one coupon and re-identifies.  A coefficient that")
    print("  barely moves when a coupon is removed was not being constrained")
    print("  by it.")
    print()
    print(f"  {'campaign':<34}" + "".join(f"{n:>10}" for n in LABELS)
          + f"{'chi2/dof':>12}")
    print("  " + "-" * (34 + 10 * len(LABELS) + 12))

    combinations = [
        ("all three", TESTS_WITH_45),
        ("drop 45deg  (the old campaign)", TESTS),
        ("drop transverse", (LONGITUDINAL, FORTYFIVE)),
        ("drop longitudinal", (TRANSVERSE, FORTYFIVE)),
        ("45deg alone", (FORTYFIVE,)),
        ("longitudinal alone", (LONGITUDINAL,)),
        ("transverse alone", (TRANSVERSE,)),
    ]
    for title, tests in combinations:
        cases, ids, _ = build_campaign(tests, verbose=False)
        try:
            result = run_gls(cases, ids)
            print(row(title, result.final_parameters,
                      reduced_chi_square(result)))
        except Exception as error:  # single coupon can be rank deficient
            print(f"  {title:<34}  failed: {type(error).__name__}: {error}")


# =============================================================================
#  Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orientation-check", action="store_true",
                        help="identify with the 45 deg frame flipped to 135")
    parser.add_argument("--leave-one-out", action="store_true",
                        help="each coupon's contribution")
    parser.add_argument("--open-inputs", action="store_true",
                        help="envelope over the unmeasured width and cut sense")
    parser.add_argument("--basis-sweep", action="store_true",
                        help="vary the smoothing-basis degree")
    parser.add_argument("--jackknife", action="store_true",
                        help="leave-one-load-step-out error bars (slow)")
    parser.add_argument("--all", action="store_true", help="every section")
    arguments = parser.parse_args()
    do_orientation = arguments.orientation_check or arguments.all
    do_loo = arguments.leave_one_out or arguments.all
    do_open = arguments.open_inputs or arguments.all
    do_sweep = arguments.basis_sweep or arguments.all
    do_jackknife = arguments.jackknife or arguments.all

    print("=" * 78)
    print("  GENERALIZED VFM -- longitudinal + transverse + 45 deg, FGLS")
    print("=" * 78)
    print("  Orientation checks (read off the shipped .mat):")
    for test in TESTS_WITH_45:
        assert_orientation(test)

    print()
    print("  Cases:")
    cases, ids, counts = build_campaign(TESTS_WITH_45, verbose=True)
    base_cases, base_ids, _ = build_campaign(TESTS, verbose=False)

    print()
    print("-" * 78)
    print("  THE IDENTIFICATION")
    print("-" * 78)
    result = run_gls(cases, ids, verbose=True)
    baseline = run_gls(base_cases, base_ids)

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
        #  The PRIMARY curvature is kappa_xx (index 0) for a coupon whose length
        #  runs along x, kappa_yy (index 1) otherwise.  The 45 deg coupon's
        #  length runs along y, so it takes index 1 like the longitudinal one.
        primary = 0 if "transverse" in report.label else 1
        temporal = noise.temporal_correlation(primary)
        off_diagonal = temporal[np.triu_indices(noise.n_steps, k=1)]
        print(f"    mean off-diagonal temporal correlation of the PRIMARY "
              f"curvature = {off_diagonal.mean():.3f}")
        print(f"    spatial: rho at the nearest neighbour "
              f"({correlation.point_spacing:.3f} mm) = "
              f"{correlation.nearest_rho:.3f}, kappa = {correlation.kappa:.2f}")
        print(f"    C conditioning: PD = "
              f"{report.covariance_diagnostics['positive_definite']}, "
              f"cond = {report.covariance_diagnostics['condition_number']:.2e}")

    # ---- the headline comparison ------------------------------------------
    print()
    print("=" * 78)
    print("  TWO COUPONS vs THREE")
    print("=" * 78)
    print(f"  {'run':<34}" + "".join(f"{n:>10}" for n in LABELS)
          + f"{'chi2/dof':>12}")
    print("  " + "-" * (34 + 10 * len(LABELS) + 12))
    print(row("2 coupons (the old campaign)", baseline.final_parameters,
              reduced_chi_square(baseline)))
    print(row("3 coupons (with 45 deg)", result.final_parameters,
              reduced_chi_square(result)))
    print()
    print(row("2 coupons, eq.(20) sigma", baseline.parameter_std))
    print(row("3 coupons, eq.(20) sigma", result.parameter_std))
    ratio = (np.asarray(result.parameter_std)
             / np.maximum(np.asarray(baseline.parameter_std), 1e-30))
    print(row("sigma ratio (3 / 2)", ratio))

    # ---- side by side -----------------------------------------------------
    print()
    print("=" * 78)
    print("  SIDE BY SIDE, three coupons")
    print("=" * 78)
    table = {}
    table["stacked naive (OLS)"] = run_ols(cases).final_parameters
    table["FGLS block-diagonal"] = run_gls(
        cases, ids, weighting="block_diagonal").final_parameters
    table["FGLS (this method, eq. 3 verbatim)"] = result.final_parameters
    table["[sensitivity only] FGLS with kappa"] = run_gls(
        cases, ids, spatial_correction=True).final_parameters

    print(f"  {'run':<34}" + "".join(f"{n:>10}" for n in LABELS))
    print("  " + "-" * (34 + 10 * len(LABELS)))
    for title, values in table.items():
        print(row(title, values))

    # ---- does the anticlastic ratio agree with the fitted D66? ------------
    print()
    print("=" * 78)
    print("  THE D66 CROSS-CHECK -- closed form vs the full identification")
    print("=" * 78)
    print("  At 45 deg with traction-free lateral edges the measured")
    print("  anticlastic ratio r = kappa_xx/kappa_yy fixes D66 in closed form:")
    print("      r = -(S - D66)/(S + D66)   with   S = (D11 + D22 + 2 D12)/4")
    print("      =>  D66 = S (1 + r) / (1 - r)")
    print("  This uses ONLY the two normal curvature channels of the 45 deg")
    print("  coupon and the D11, D22, D12 the other two coupons supply.  It is")
    print("  an independent route to the same number, and it is immune to the")
    print("  specimen width and the moment gradient, both of which cancel from")
    print("  a ratio of curvatures.")
    print()
    manifest = FORTYFIVE.directory / "vfm_export_manifest.csv"
    ratios = []
    if manifest.exists():
        with manifest.open(encoding="utf-8") as handle:
            ratios = [float(line["anticlastic_ratio"])
                      for line in csv.DictReader(handle)]
    if ratios:
        r_mean = float(np.mean(ratios))
        d11, d12, d22, d66 = result.final_parameters
        s_value = (d11 + d22 + 2.0 * d12) / 4.0
        closed_form = s_value * (1.0 + r_mean) / (1.0 - r_mean)
        print(f"  measured r                      = {r_mean:+.4f} "
              f"+/- {np.std(ratios):.4f}  (9 load steps)")
        print(f"  S from the fitted D11, D12, D22 = {s_value:9.2f} N.mm")
        print(f"  D66 from the closed form        = {closed_form:9.2f} N.mm")
        print(f"  D66 from the full FGLS          = {d66:9.2f} N.mm")
        gap = abs(closed_form - d66) / max(abs(closed_form), 1e-12) * 100.0
        print(f"  gap                             = {gap:9.1f} %")

    # ---- identifiability --------------------------------------------------
    for tag, subset_cases in (("2 coupons", base_cases),
                              ("3 coupons", cases)):
        diagnostics = compute_identifiability_diagnostics(
            cases=subset_cases,
            constitutive_model=define_constitutive_model(
                MATERIAL_TYPE, ANALYSIS_TYPE, result.final_parameters))
        aggregate = diagnostics.aggregate
        print()
        print("=" * 78)
        print(f"  IDENTIFIABILITY (Pierron H / CID) -- {tag}")
        print("=" * 78)
        print(f"  aggregate cond(H) = {aggregate.condition_number:.3e}")
        print("  singular values   = "
              + np.array2string(aggregate.singular_values, precision=3))
        print("  parameter correlation matrix:")
        print("           "
              + "".join(f"{n:>9}" for n in diagnostics.parameter_labels))
        for index, name in enumerate(diagnostics.parameter_labels):
            print(f"    {name:>6} " + "".join(
                f"{aggregate.correlation[index, column]:9.3f}"
                for column in range(len(diagnostics.parameter_labels))))

    # ---- optional sections ------------------------------------------------
    if do_orientation:
        orientation_check(cases, ids, result)
    if do_open:
        open_inputs(cases, ids)
    if do_loo:
        leave_one_out(result)

    jackknife_std = None
    if do_jackknife:
        print()
        print("=" * 78)
        print("  JACKKNIFE (leave one load step out, Gamma re-estimated inside)")
        print("=" * 78)
        identification, linear_algebra = _settings()
        estimates, mean, jackknife_std = jackknife_parameters(
            cases=cases, experiment_ids=ids, material_type=MATERIAL_TYPE,
            analysis_type=ANALYSIS_TYPE,
            initial_parameters=INITIAL_PARAMETERS,
            identification_settings=identification,
            linear_algebra_settings=linear_algebra,
            gls_settings=GlsSettings(), verbose=False)
        print(f"  {len(estimates)} replicates")
        for index, name in enumerate(LABELS):
            print(f"    {name:>6} = {result.final_parameters[index]:10.3f} "
                  f"+/- {jackknife_std[index]:8.3f} N.mm"
                  f"   [eq.20 said {result.parameter_std[index]:.3f}]")

    if do_sweep:
        print()
        print("=" * 78)
        print("  SENSITIVITY TO THE SMOOTHING BASIS")
        print("=" * 78)
        print(f"  {'degree':>8}{'d':>6}" + "".join(f"{n:>11}" for n in LABELS))
        for degree in BASIS_DEGREES:
            swept = run_gls(cases, ids, basis_degree=degree)
            n_basis = (degree + 1) * (degree + 2) // 2
            print(f"  {degree:>8}{n_basis:>6}"
                  + "".join(f"{v:11.2f}" for v in swept.final_parameters))

    # ---- headline ---------------------------------------------------------
    report_parameters(LABELS, result.final_parameters,
                      "FGLS, THREE COUPONS -- THE RESULT", std=jackknife_std)
    print()
    derived_constants(LABELS, result.final_parameters)

    birge = np.sqrt(max(reduced_chi_square(result), 1.0))
    print()
    print(f"  Birge factor sqrt(chi2/dof) = {birge:.1f}.  Eq. (20) is a variance")
    print("  UNDER THE NOISE MODEL, and the data reject that model here as they")
    print("  do on the two-coupon campaign.  A defensible interval is at least")
    print("  the eq. (20) sigma times that factor:")
    for index, name in enumerate(LABELS):
        value = result.final_parameters[index]
        scaled = result.parameter_std[index] * birge
        print(f"    {name:>6} = {value:9.2f} +/- {scaled:7.2f} N.mm")

    # ---- CSV --------------------------------------------------------------
    output_dir = RESULTS_DIR / "three_coupon"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "gls_results_orthotropic_with45.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run"] + LABELS)
        writer.writerow(["2 coupons FGLS (baseline)"]
                        + [f"{v:.6f}" for v in baseline.final_parameters])
        for title, values in table.items():
            writer.writerow([title] + [f"{v:.6f}" for v in values])
        writer.writerow(["FGLS std eq.(20)"]
                        + [f"{v:.6f}" for v in result.parameter_std])
        writer.writerow([f"FGLS std x Birge ({birge:.2f})"]
                        + [f"{v * birge:.6f}" for v in result.parameter_std])
        if jackknife_std is not None:
            writer.writerow(["FGLS std jackknife"]
                            + [f"{v:.6f}" for v in jackknife_std])
    print(f"\n  wrote {out_path.name}")

    step_path = output_dir / "gls_step_estimates_with45.csv"
    with step_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["experiment", "step"] + [f"U_{n}" for n in LABELS])
        for report in result.experiments:
            for step, label in enumerate(report.step_labels):
                writer.writerow([report.label, label]
                                + [f"{v:.6f}"
                                   for v in report.step_estimates[step]])
    print(f"  wrote {step_path.name}")


if __name__ == "__main__":
    main()
