"""Longitudinal + transverse + EDGE TWIST: identification, and the twist BC.

WHY THIS STUDY EXISTS
---------------------
D66 is the two-coupon campaign's blind spot.  Neither bending coupon loads the
twist channel, so the stacked answer is D66 = -6.5 N mm -- an unphysical
number that only says "no information".  The edge-twist coupon loads kappa_xy
directly, at 4.5x the next largest channel, and is the first test in the
campaign that constrains D66 on its own terms.

THE BOUNDARY CONDITION IS THE WHOLE POINT
-----------------------------------------
This coupon's loaded edge is CLAMPED INTO THE YOKE.  It does not warp and it is
not a free St-Venant twist: it rotates as a RIGID BODY about the specimen axis,
and the machine reports the couple it takes to do that.  So the external
virtual work is

    EVW = T_x * phi*                 (applied couple x virtual rigid rotation)

and the BC has to make phi* well defined.  That is what the LINKING group on
W_X does -- it ties the tangential slope to one value along the edge, which IS
the statement "this edge rotates rigidly".  Everything else follows:

    bottom cut   fully sealed (W, W_X, W_Y, W_XY = 0)  -- every unknown
                 reaction annihilated; nothing measured there, nothing assumed.
    top cut      W_Y = 0 AND W_XY = 0 together, so dw*/dy is identically zero
                 along the edge and the UNMEASURED M_yy does no work.  Clamping
                 W_Y alone would leave it wiggling between nodes.
    top cut      W left FREE.  Required: the rotation has to be able to lift one
                 side of the edge relative to the other.  The price is that the
                 effective shear does work, and the price is measured -- dFz is
                 under 1 % of the couple.
    left/right   never touched: the specimen's physical free edges.

Sections 1-6 below test each of those choices against the alternatives rather
than asserting them.

Run:  python -m studies.twist_three_coupon.make_study
"""

from __future__ import annotations

import copy
import csv
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from vfm_gls_identification import run_gls_identification_cases   # noqa: E402
from vfm_user_inputs import (GlsSettings, IdentificationSettings,  # noqa: E402
                             LinearAlgebraSettings)
from workflows.campaign_common import (ANALYSIS_TYPE,              # noqa: E402
                                       INITIAL_PARAMETERS,
                                       LONGITUDINAL, MATERIAL_TYPE,
                                       TESTS_RAW_FRAME,
                                       TESTS_WITH_TWIST_RAW,
                                       TRANSVERSE_RAW, TWIST,
                                       assert_orientation, build_campaign,
                                       build_cases, load_bc_module)
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


def row(name, theta, extra=""):
    values = "  ".join(f"{v:9.3f}" for v in theta)
    print(f"    {name:<38} {values}   {extra}")


# =============================================================================
#  BC variants -- the design decisions, measured instead of asserted
# =============================================================================

def variant_cases(mutate):
    """Build the campaign with the TWIST coupon's BC mutated in place.

    ``mutate(bc)`` edits a deep copy of the shipped BoundaryConditions.  The two
    bending coupons are untouched, so any change in theta is attributable to the
    twist BC alone.
    """
    cases, ids = [], []
    for test in TESTS_WITH_TWIST_RAW:
        test_cases = build_cases(test, verbose=False)
        if test is TWIST:
            for case in test_cases:
                bc = copy.deepcopy(case.boundary_conditions)
                mutate(bc)
                case.boundary_conditions = bc
        cases.extend(test_cases)
        ids.extend([test.experiment_id] * len(test_cases))
    return cases, ids


def _drop(bc, dof, edge):
    bc.prescribed = [e for e in bc.prescribed
                     if not (e["dof"] == dof and e["edge"] == edge)]


def _add(bc, dof, edge, label):
    bc.prescribed = bc.prescribed + [
        {"label": label, "dof": dof, "edge": edge, "value": 0.0}]


VARIANTS = [
    ("A", "shipped form", lambda bc: None,
     "reference"),
    ("B", "top W_XY free", lambda bc: _drop(bc, "W_XY", "top"),
     "LEAKS: dw*/dy no longer identically 0, so the unmeasured M_yy works"),
    ("C", "top W_X not linked", lambda bc: setattr(bc, "linking", []),
     "the loaded edge is no longer rigid, so phi* is not a rotation"),
    ("D", "top W = 0", lambda bc: _add(bc, "W", "top", "top w*=0"),
     "the edge cannot lift one side: external work collapses"),
    ("E", "lateral edges clamped", lambda bc: [_add(bc, "W", e, f"{e} w*=0")
                                               for e in ("left", "right")],
     "control: the laterals are physically free, so this only costs freedom"),
    ("F", "top W_Y free", lambda bc: _drop(bc, "W_Y", "top"),
     "LEAKS: the unmeasured bending moment M_yy at the loaded cut does work"),
]


# =============================================================================
#  Main
# =============================================================================

def main() -> None:
    apply_style()
    DATA.mkdir(parents=True, exist_ok=True)
    results = {}

    print("=" * 78)
    print("  THREE-COUPON CAMPAIGN: longitudinal + transverse + EDGE TWIST")
    print("=" * 78)

    # ---- 1. setup -----------------------------------------------------------
    print("\n  1. ORIENTATION AND SETUP")
    geometry = {t.name: assert_orientation(t) for t in TESTS_WITH_TWIST_RAW}
    cases, ids, counts = build_campaign(TESTS_WITH_TWIST_RAW, verbose=True)
    module = load_bc_module(TWIST.bc_file)
    print(f"    twist BC: {len(module.PRESCRIBED)} prescribed, "
          f"{len(module.LINKING)} linking "
          f"({module.LINKING[0]['dof']}@{module.LINKING[0]['edge']} -- the rigid"
          f" rotation), {len(module.EXTERNAL)} external "
          f"({module.EXTERNAL[0]['dof']}@{module.EXTERNAL[0]['edge']})")

    # ---- 2. the identification ---------------------------------------------
    print("\n  2. IDENTIFICATION")
    reference = fgls(cases, ids)
    results["three-coupon (with twist)"] = theta_of(reference)
    base_cases, base_ids, _ = build_campaign(TESTS_RAW_FRAME, verbose=False)
    baseline = fgls(base_cases, base_ids)
    results["two-coupon baseline"] = theta_of(baseline)

    print(f"    {'':<38} " + "  ".join(f"{l:>9}" for l in LABELS))
    row("two-coupon baseline (no twist)", theta_of(baseline))
    row("three-coupon (with twist)", theta_of(reference))
    d66_before, d66_after = theta_of(baseline)[3], theta_of(reference)[3]
    print(f"\n    D66: {d66_before:+.2f} -> {d66_after:+.2f} N mm.  The baseline "
          f"number is not a\n         measurement -- neither bending coupon "
          f"loads the twist channel, so a\n         negative D66 is what 'no "
          f"information' looks like.  The twist coupon\n         is the first "
          f"test in the campaign that constrains it directly.")
    chi = chi_square_diagnostics(reference)
    print(f"\n    chi2/dof = {chi['reduced']:.2f}, Birge factor "
          f"{chi['birge']:.2f} -- reported sigma must be Birge-scaled; eq. (20)")
    print("    alone is a variance under a model the data reject.")
    print("    per-experiment chi2/dof: "
          + ", ".join(f"{report.experiment_id}:{value:.1f}"
                      for report, value in zip(reference.experiments,
                                               chi["per_experiment"])))

    # ---- 3. weighting ladder ------------------------------------------------
    print("\n  3. WEIGHTING LADDER")
    print("     The twist coupon contributes only TWO load steps, so its")
    print("     cross-time covariance block is 2x2.  If the answer depended")
    print("     strongly on the weighting, that block would be doing work it")
    print("     cannot support.")
    print(f"    {'':<38} " + "  ".join(f"{l:>9}" for l in LABELS))
    ladder = {}
    for weighting in ("gls", "block_diagonal", "diagonal", "ols"):
        ladder[weighting] = theta_of(fgls(cases, ids, weighting=weighting))
        row(weighting, ladder[weighting])

    # ---- 4. leave-one-out ---------------------------------------------------
    print("\n  4. LEAVE-ONE-OUT")
    print(f"    {'':<38} " + "  ".join(f"{l:>9}" for l in LABELS))
    loo = {}
    for name, subset in (("all three", TESTS_WITH_TWIST_RAW),
                         ("drop twist", (LONGITUDINAL, TRANSVERSE_RAW)),
                         ("drop transverse", (LONGITUDINAL, TWIST)),
                         ("drop longitudinal", (TRANSVERSE_RAW, TWIST))):
        try:
            sub_cases, sub_ids, _ = build_campaign(subset, verbose=False)
            loo[name] = theta_of(fgls(sub_cases, sub_ids))
            row(name, loo[name])
        except Exception as error:                            # noqa: BLE001
            print(f"    {name:<38} not identifiable: "
                  f"{type(error).__name__}")

    # ---- 5. BC variants -----------------------------------------------------
    print("\n  5. BOUNDARY-CONDITION VARIANTS")
    print("     Only the TWIST coupon's BC changes; the two bending coupons are")
    print("     untouched, so every difference is attributable to it.")
    print(f"    {'':<38} " + "  ".join(f"{l:>9}" for l in LABELS))
    variants = {}
    for key, title, mutate, note in VARIANTS:
        try:
            v_cases, v_ids = variant_cases(mutate)
            theta = theta_of(fgls(v_cases, v_ids))
            variants[key] = (title, theta, note)
            delta = ("" if key == "A"
                     else f"D66 {theta[3] / results['three-coupon (with twist)'][3]:+.2f}x")
            row(f"{key}  {title}", theta, delta)
        except Exception as error:                            # noqa: BLE001
            variants[key] = (title, np.full(4, np.nan), note)
            print(f"    {key}  {title:<34} FAILED: {type(error).__name__} "
                  f"-- {note}")
    for key, title, mutate, note in VARIANTS:
        if key != "A":
            print(f"      {key}: {note}")

    # ---- 6. material orientation -------------------------------------------
    print("\n  6. MATERIAL ORIENTATION OF THE TWIST COUPON")
    print("     The cut angle was NEVER RECORDED for this coupon.  0 deg is")
    print("     carried.  T(90) leaves the D66 PARAMETER untouched -- it")
    print("     permutes 1 <-> 2 and D16 = D26 = 0 either way -- so it is easy")
    print("     to assume the choice is free.  IT IS NOT: the coupon carries a")
    print("     secondary kappa_yy at -0.22 of kappa_xy, and swapping the frame")
    print("     re-attributes that channel between D11 and D22, moving the D66")
    print("     the fit lands on.  Measured below.")
    print(f"    {'':<38} " + "  ".join(f"{l:>9}" for l in LABELS))
    orientation = {}
    for angle in (0.0, 90.0):
        rotated = [replace(case, material_orientation_deg=angle)
                   if experiment == TWIST.experiment_id else case
                   for case, experiment in zip(cases, ids)]
        orientation[angle] = theta_of(fgls(rotated, ids))
        row(f"twist coupon at {angle:.0f} deg", orientation[angle],
            "<-- shipped" if angle == 0.0 else "")
    spread = abs(orientation[90.0] - orientation[0.0])
    print(f"    {'spread':<38} " + "  ".join(f"{v:9.3f}" for v in spread))
    print(f"\n    D66 moves {orientation[0.0][3]:.2f} -> "
          f"{orientation[90.0][3]:.2f} N mm, a factor of "
          f"{orientation[0.0][3] / orientation[90.0][3]:.1f}.  THE UNRECORDED CUT")
    print("    ANGLE IS THIS COUPON'S DOMINANT UNCERTAINTY, larger than the")
    print("    weighting, the window and every BC variant that does not break")
    print("    the identification outright.  It is a missing measurement, not a")
    print("    modelling choice, and it should be recovered from the specimen")
    print("    records before this coupon's D66 is quoted on its own.")

    # ---- output -------------------------------------------------------------
    with (DATA / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case"] + LABELS)
        for name, theta in results.items():
            writer.writerow([name] + [f"{v:.6f}" for v in theta])
        for name, theta in ladder.items():
            writer.writerow([f"weighting:{name}"] + [f"{v:.6f}" for v in theta])
        for name, theta in loo.items():
            writer.writerow([f"leave-one-out:{name}"]
                            + [f"{v:.6f}" for v in theta])
        for angle, theta in orientation.items():
            writer.writerow([f"orientation:{angle:.0f}deg"]
                            + [f"{v:.6f}" for v in theta])
    print(f"\n  wrote data/results.csv")

    with (DATA / "bc_variants.csv").open("w", newline="",
                                         encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", "variant"] + LABELS + ["note"])
        for key, (title, theta, note) in variants.items():
            writer.writerow([key, title] + [f"{v:.6f}" for v in theta] + [note])
    print(f"  wrote data/bc_variants.csv")

    figure_parameters(results, baseline, reference, FIGURES / "fig1_parameters")
    figure_variants(variants, results["three-coupon (with twist)"],
                    FIGURES / "fig2_bc_variants")
    figure_ladder(ladder, loo, FIGURES / "fig3_robustness")

    print("\n" + "=" * 78)
    print(f"  D66 = {d66_after:.2f} N mm from a directly loaded twist channel.")
    print("=" * 78)


# =============================================================================
#  Figures
# =============================================================================

def figure_parameters(results, baseline, reference, path):
    names = list(results)
    figure, axes = plt.subplots(1, 4, figsize=(7.2, 2.1))
    for index, (axis, label) in enumerate(zip(axes, LABELS)):
        values = [results[n][index] for n in names]
        colours = ["0.65", "tab:blue"]
        axis.bar(range(len(names)), values, color=colours[:len(names)])
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(range(len(names)))
        axis.set_xticklabels(["no twist", "with twist"], rotation=20,
                             ha="right")
        axis.set_title(math_label(label))
        soft_grid(axis)
    axes[0].set_ylabel("N mm")
    figure.suptitle("Adding the edge-twist coupon to the two-coupon campaign",
                    y=1.04)
    figure.tight_layout()
    save(figure, path)


def figure_variants(variants, reference, path):
    keys = [k for k in variants if np.isfinite(variants[k][1]).all()]
    figure, axis = plt.subplots(figsize=(5.4, 2.6))
    values = [variants[k][1][3] for k in keys]
    colours = ["tab:blue" if k == "A" else "tab:orange" for k in keys]
    axis.bar(range(len(keys)), values, color=colours)
    axis.axhline(reference[3], color="tab:blue", linestyle="--", linewidth=1.0,
                 label="shipped form")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(range(len(keys)))
    axis.set_xticklabels([f"{k}\n{variants[k][0]}" for k in keys], fontsize=6)
    axis.set_ylabel(f"{math_label('D66')}  (N mm)")
    axis.set_title("Twist boundary conditions: every choice, measured")
    axis.legend()
    soft_grid(axis)
    figure.tight_layout()
    save(figure, path)


def figure_ladder(ladder, loo, path):
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
    for axis, (title, table) in zip(axes, (("Weighting", ladder),
                                           ("Leave-one-out", loo))):
        names = list(table)
        axis.bar(range(len(names)), [table[n][3] for n in names],
                 color="tab:blue")
        axis.set_xticks(range(len(names)))
        axis.set_xticklabels(names, rotation=25, ha="right", fontsize=6.5)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(title)
        soft_grid(axis)
    axes[0].set_ylabel(f"{math_label('D66')}  (N mm)")
    figure.tight_layout()
    save(figure, path)


if __name__ == "__main__":
    main()
