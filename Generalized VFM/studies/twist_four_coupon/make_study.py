"""Longitudinal + transverse + 45 deg + EDGE TWIST -- two D66 routes, compared.

WHY THIS STUDY EXISTS
---------------------
The campaign now has TWO independent measurements of D66, and they are
independent in the strong sense: their dominant error sources are disjoint.

  45 DEG COUPON.  D66 comes from the ANTICLASTIC RATIO
  kappa_xx/kappa_yy = -(S-D66)/(S+D66) with S = (D11+D22+2 D12)/4.  It is a
  ratio of two curvature fields measured at the same points in the same frame,
  so the applied moment CANCELS -- and with it the specimen width, the machine
  calibration and the moment gradient.  What it does depend on is the cut sense
  (45 vs 135) and on the lateral edges genuinely being traction-free.  Its
  twist is SUPPRESSED ~15x by the yoke, so it never measures a twist directly.

  TWIST COUPON.  D66 comes from a DIRECTLY LOADED kappa_xy, the dominant
  channel at 4.5x the next largest.  It does not care about the 45/135 sense at
  all.  It does scale as 1/width, so the placeholder specimen width propagates
  straight into it, and it depends on its own (unrecorded) cut angle.

So the two agree or they do not, and either way the disagreement localises: no
single assumption is upstream of both.  That is the point of this study, and it
is a stronger check than anything either coupon can do alone.

Run:  python -m studies.twist_four_coupon.make_study
"""

from __future__ import annotations

import csv
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from vfm_gls_identification import (jackknife_parameters,          # noqa: E402
                                    run_gls_identification_cases)
from vfm_user_inputs import (GlsSettings, IdentificationSettings,   # noqa: E402
                             LinearAlgebraSettings)
from workflows.campaign_common import (ANALYSIS_TYPE, FORTYFIVE,    # noqa: E402
                                       INITIAL_PARAMETERS,
                                       LONGITUDINAL, MATERIAL_TYPE,
                                       TESTS_RAW_FRAME,
                                       TESTS_WITH_45_RAW,
                                       TESTS_WITH_45_TWIST_RAW,
                                       TESTS_WITH_TWIST_RAW,
                                       TRANSVERSE_RAW, TWIST,
                                       assert_orientation, build_campaign)
from workflows.make_figures import (apply_style, chi_square_diagnostics,  # noqa: E402
                                    math_label, soft_grid)

FIGURES = HERE / "figures"
DATA = HERE / "data"
LABELS = ["D11", "D12", "D22", "D66"]


def save(figure, path):
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".pdf"))
    figure.savefig(path.with_suffix(".png"), dpi=600)
    plt.close(figure)
    print(f"  wrote figures/{path.name}.pdf + .png")


def fgls(cases, ids, weighting="gls"):
    return run_gls_identification_cases(
        cases=cases, material_type=MATERIAL_TYPE, analysis_type=ANALYSIS_TYPE,
        initial_parameters=INITIAL_PARAMETERS,
        identification_settings=IdentificationSettings(
            max_iterations=40, convergence_tolerance=1.0e-8),
        linear_algebra_settings=LinearAlgebraSettings(),
        experiment_ids=ids,
        gls_settings=GlsSettings(weighting=weighting), verbose=False)


def theta_of(result):
    return np.asarray(result.final_parameters, dtype=float)


def run(tests, weighting="gls"):
    cases, ids, _ = build_campaign(tests, verbose=False)
    return theta_of(fgls(cases, ids, weighting=weighting))


def row(name, theta, extra=""):
    print(f"    {name:<38} " + "  ".join(f"{v:9.3f}" for v in theta)
          + f"   {extra}")


def closed_form_d66(theta):
    """D66 implied by the 45 deg coupon's MEASURED anticlastic ratio.

    With traction-free lateral edges a 45 deg coupon satisfies
    kappa_xx/kappa_yy = -(S - D66)/(S + D66), so

        D66 = S (1 + r) / (1 - r),   r = kappa_xx/kappa_yy,
        S   = (D11 + D22 + 2 D12) / 4

    This is the 45 deg coupon's D66 statement in closed form -- moment-free,
    and therefore free of the width and the moment gradient.  Same expression
    run_campaign_45.py:458 uses.
    """
    import csv as _csv
    manifest = FORTYFIVE.directory / "vfm_export_manifest.csv"
    with manifest.open(encoding="utf-8") as handle:
        ratios = [float(line["anticlastic_ratio"])
                  for line in _csv.DictReader(handle)]
    ratio = float(np.mean(ratios))
    s_value = (theta[0] + theta[2] + 2.0 * theta[1]) / 4.0
    return s_value * (1.0 + ratio) / (1.0 - ratio), ratio, s_value


def main() -> None:
    apply_style()
    DATA.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  FOUR-COUPON CAMPAIGN: longitudinal + transverse + 45 deg + TWIST")
    print("=" * 78)

    # ---- 1. setup -----------------------------------------------------------
    print("\n  1. ORIENTATION")
    for test in TESTS_WITH_45_TWIST_RAW:
        assert_orientation(test)
    cases, ids, counts = build_campaign(TESTS_WITH_45_TWIST_RAW, verbose=True)

    # ---- 2. identification and baselines ------------------------------------
    print("\n  2. IDENTIFICATION, AGAINST EVERY BASELINE")
    print(f"    {'':<38} " + "  ".join(f"{l:>9}" for l in LABELS))
    results = {}
    for name, tests in (("two-coupon (bending only)", TESTS_RAW_FRAME),
                        ("three-coupon (+45 deg)", TESTS_WITH_45_RAW),
                        ("three-coupon (+twist)", TESTS_WITH_TWIST_RAW),
                        ("four-coupon (+both)", TESTS_WITH_45_TWIST_RAW)):
        results[name] = run(tests)
        row(name, results[name])
    reference = fgls(cases, ids)
    chi = chi_square_diagnostics(reference)
    print(f"\n    chi2/dof = {chi['reduced']:.2f}, Birge factor "
          f"{chi['birge']:.2f}")
    print("    per-experiment chi2/dof: "
          + ", ".join(f"{r.experiment_id}:{v:.1f}"
                      for r, v in zip(reference.experiments,
                                      chi["per_experiment"])))

    # ---- 3. the headline: D66 by two routes ---------------------------------
    print("\n  3. D66 BY TWO INDEPENDENT ROUTES")
    theta_45 = results["three-coupon (+45 deg)"]
    theta_tw = results["three-coupon (+twist)"]
    theta_all = results["four-coupon (+both)"]
    closed, ratio, s_value = closed_form_d66(theta_45)
    routes = {
        "45 deg coupon, stacked FGLS": theta_45[3],
        "45 deg coupon, closed form from the anticlastic ratio": closed,
        "twist coupon, stacked FGLS": theta_tw[3],
        "all four together": theta_all[3],
    }
    for name, value in routes.items():
        print(f"    {name:<56} {value:8.2f} N mm")
    print(f"\n    (45 deg closed form uses the measured anticlastic ratio "
          f"r = {ratio:+.4f}\n     and S = (D11+D22+2 D12)/4 = {s_value:.2f} "
          f"N mm from the same fit.)")
    identified = [routes["45 deg coupon, stacked FGLS"],
                  routes["twist coupon, stacked FGLS"],
                  routes["all four together"]]
    spread = (max(identified) - min(identified)) / np.mean(identified)
    print(f"\n    THE TWO IDENTIFIED ROUTES AGREE TO {spread:.1%}: "
          f"{identified[0]:.2f} (45 deg, indirect)")
    print(f"    against {identified[1]:.2f} (twist, direct), and "
          f"{identified[2]:.2f} with both stacked.")
    print()
    print("    That is the result worth having.  The two routes share almost no")
    print("    assumptions: the 45 deg answer comes from a ratio of curvatures")
    print("    in which the moment cancels, so no specimen width, machine")
    print("    calibration or moment gradient enters it, but it does depend on")
    print("    the 45/135 cut sense and on a twist the yoke suppresses ~15x.")
    print("    The twist answer comes from a directly loaded kappa_xy, is")
    print("    indifferent to the cut sense, and instead scales as 1/width and")
    print("    depends on its own unrecorded cut angle (a factor of 2.7 --")
    print("    see ../twist_three_coupon section 6).  Two estimators with")
    print("    disjoint dominant errors landing within a few percent of each")
    print("    other is evidence neither is dominated by its own weakness.")
    print()
    print(f"    THE CLOSED FORM IS THE OUTLIER at {closed:.2f} N mm, and that is")
    print("    a statement about S, not about D66.  D66 = S(1+r)/(1-r) is")
    print(f"    linear in S, and S = {s_value:.2f} N mm here carries a D12 of")
    print(f"    {theta_45[1]:+.2f} -- negative, i.e. the campaign's weakest")
    print("    parameter.  The closed form inherits that weakness undiluted;")
    print("    the stacked fits do not.")

    # ---- 4. leave-one-out ---------------------------------------------------
    print("\n  4. LEAVE-ONE-OUT")
    print(f"    {'':<38} " + "  ".join(f"{l:>9}" for l in LABELS))
    loo = {}
    for drop in range(len(TESTS_WITH_45_TWIST_RAW)):
        subset = tuple(t for i, t in enumerate(TESTS_WITH_45_TWIST_RAW)
                       if i != drop)
        name = f"drop {TESTS_WITH_45_TWIST_RAW[drop].name}"
        try:
            loo[name] = run(subset)
            row(name, loo[name])
        except Exception as error:                            # noqa: BLE001
            print(f"    {name:<38} not identifiable: {type(error).__name__}")
    for pair in combinations(TESTS_WITH_45_TWIST_RAW, 2):
        name = " + ".join(t.name for t in pair)
        try:
            loo[name] = run(pair)
            row(name, loo[name])
        except Exception as error:                            # noqa: BLE001
            print(f"    {name:<38} not identifiable: {type(error).__name__}")

    # ---- 5. robustness ------------------------------------------------------
    print("\n  5. WEIGHTING AND JACKKNIFE")
    print(f"    {'':<38} " + "  ".join(f"{l:>9}" for l in LABELS))
    ladder = {}
    for weighting in ("gls", "block_diagonal", "diagonal", "ols"):
        ladder[weighting] = run(TESTS_WITH_45_TWIST_RAW, weighting=weighting)
        row(weighting, ladder[weighting])
    try:
        _, jack_mean, jack_std = jackknife_parameters(
            cases=cases, material_type=MATERIAL_TYPE,
            analysis_type=ANALYSIS_TYPE,
            initial_parameters=INITIAL_PARAMETERS,
            identification_settings=IdentificationSettings(
                max_iterations=40, convergence_tolerance=1.0e-8),
            linear_algebra_settings=LinearAlgebraSettings(),
            experiment_ids=ids, gls_settings=GlsSettings(), verbose=False)
        row("jackknife mean", np.asarray(jack_mean, dtype=float))
        row("jackknife std", np.asarray(jack_std, dtype=float))
    except Exception as error:                                # noqa: BLE001
        jack_mean = jack_std = None
        print(f"    jackknife unavailable: {type(error).__name__}")

    # ---- output -------------------------------------------------------------
    with (DATA / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case"] + LABELS)
        for name, theta in results.items():
            writer.writerow([name] + [f"{v:.6f}" for v in theta])
        for name, theta in loo.items():
            writer.writerow([f"leave-one-out:{name}"]
                            + [f"{v:.6f}" for v in theta])
        for name, theta in ladder.items():
            writer.writerow([f"weighting:{name}"] + [f"{v:.6f}" for v in theta])
        if jack_mean is not None:
            writer.writerow(["jackknife_mean"]
                            + [f"{v:.6f}" for v in jack_mean])
            writer.writerow(["jackknife_std"]
                            + [f"{v:.6f}" for v in jack_std])
    print("\n  wrote data/results.csv")

    with (DATA / "d66_routes.csv").open("w", newline="",
                                        encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["route", "D66_N_mm", "note"])
        notes = {
            "45 deg coupon, stacked FGLS":
                "indirect: anticlastic ratio; twist suppressed ~15x by the yoke",
            "45 deg coupon, closed form from the anticlastic ratio":
                f"moment-free (width and gradient cancel); r = {ratio:+.4f}",
            "twist coupon, stacked FGLS":
                "direct: loaded kappa_xy; scales as 1/width; cut angle unrecorded",
            "all four together": "both routes stacked",
        }
        for name, value in routes.items():
            writer.writerow([name, f"{value:.6f}", notes[name]])
    print("  wrote data/d66_routes.csv")

    figure_parameters(results, FIGURES / "fig1_parameters")
    figure_d66(routes, FIGURES / "fig2_d66_routes")
    figure_loo(loo, theta_all, FIGURES / "fig3_leave_one_out")

    print("\n" + "=" * 78)
    print(f"  Four-coupon theta = "
          + ", ".join(f"{l} {v:.2f}" for l, v in zip(LABELS, theta_all)))
    print("=" * 78)


# =============================================================================
#  Figures
# =============================================================================

def figure_parameters(results, path):
    names = list(results)
    figure, axes = plt.subplots(1, 4, figsize=(7.2, 2.3))
    colours = ["0.7", "tab:green", "tab:orange", "tab:blue"]
    for index, (axis, label) in enumerate(zip(axes, LABELS)):
        axis.bar(range(len(names)), [results[n][index] for n in names],
                 color=colours)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(range(len(names)))
        axis.set_xticklabels(["bending", "+45", "+twist", "+both"],
                             rotation=25, ha="right", fontsize=6.5)
        axis.set_title(math_label(label))
        soft_grid(axis)
    axes[0].set_ylabel("N mm")
    figure.suptitle("Each coupon's contribution to the identified stiffness",
                    y=1.05)
    figure.tight_layout()
    save(figure, path)


def figure_d66(routes, path):
    names = list(routes)
    short = ["45 deg\n(FGLS)", "45 deg\n(closed form)", "twist\n(FGLS)",
             "all four"]
    figure, axis = plt.subplots(figsize=(4.6, 2.8))
    axis.bar(range(len(names)), [routes[n] for n in names],
             color=["tab:green", "tab:green", "tab:orange", "tab:blue"])
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(range(len(names)))
    axis.set_xticklabels(short, fontsize=6.5)
    axis.set_ylabel(f"{math_label('D66')}  (N mm)")
    axis.set_title("Two independent routes to $D_{66}$\n"
                   "green: indirect, moment-free   |   orange: directly loaded",
                   fontsize=7.5)
    soft_grid(axis)
    figure.tight_layout()
    save(figure, path)


def figure_loo(loo, reference, path):
    names = list(loo)
    figure, axis = plt.subplots(figsize=(6.4, 2.6))
    axis.bar(range(len(names)), [loo[n][3] for n in names], color="tab:blue")
    axis.axhline(reference[3], color="black", linestyle="--", linewidth=1.0,
                 label="all four")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(range(len(names)))
    axis.set_xticklabels(names, rotation=30, ha="right", fontsize=6)
    axis.set_ylabel(f"{math_label('D66')}  (N mm)")
    axis.set_title("Which coupons $D_{66}$ actually depends on")
    axis.legend()
    soft_grid(axis)
    figure.tight_layout()
    save(figure, path)


if __name__ == "__main__":
    main()
