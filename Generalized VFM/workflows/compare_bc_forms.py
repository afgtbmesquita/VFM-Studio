# =============================================================================
#  Generalized VFM / compare_bc_forms.py
# -----------------------------------------------------------------------------
#  Moment at both cuts, or clamp the cut whose moment is unknown?
#
#  THE TWO FORMULATIONS
#  ---------------------------------------------------------------------------
#  The VFM requires that every UNKNOWN boundary traction do zero virtual work.
#  On a loaded cut the tractions are the shear Q, the twisting moment, and the
#  bending moment M.  The shipped BCs prescribe w* = 0, the tangential slope
#  and the twist on both cuts, which annihilates Q and the twist -- but leave
#  the NORMAL ROTATION free, so M does work at both ends:
#
#      (a) BOTH ENDS      W, W_X, W_XY = 0 on both cuts; two EXTERNAL groups
#                         carrying F = [-M, +M].
#
#  If the moment at one cut is genuinely UNKNOWN, the rigorous treatment is to
#  annihilate it too, by prescribing the normal rotation there as well.  That
#  cut then contributes nothing, and only the measured moment remains:
#
#      (b) CLAMPED FAR END    ... plus W_Y = 0 on the unknown cut; ONE
#                             EXTERNAL group, carrying the measured moment.
#
#  Form (b) is the stricter formulation and is what you want WHENEVER one end's
#  moment is a reaction you do not measure.  This script runs both on the real
#  coupons so the choice is made on evidence rather than principle alone.
#
#  WHICH END IS "UNKNOWN" IS NOT OBVIOUS HERE
#  ---------------------------------------------------------------------------
#  The machine reports ONE couple (the dTy channel).  The export turns it into
#  F = [-M, +M], i.e. it ASSUMES moment equilibrium and puts the same magnitude
#  on both cuts.  So strictly neither individual cut moment is measured -- only
#  their common value under an equilibrium assumption.  Both clamp choices are
#  therefore run, and they are not interchangeable: see the longitudinal
#  README's Finding 3.
#
#      python -m workflows.compare_bc_forms
# =============================================================================

from __future__ import annotations

import copy
import dataclasses

import numpy as np
from scipy.io import loadmat

from workflows.campaign_common import (
    ANALYSIS_TYPE,
    INITIAL_PARAMETERS,
    LONGITUDINAL,
    MATERIAL_TYPE,
    PROJECTS_DIR,
    TRANSVERSE_RAW,
    build_cases,
    mat_files,
)
from vfm_gls_identification import run_gls_identification_cases
from vfm_identification import run_identification_cases
from vfm_user_inputs import (
    GlsSettings,
    IdentificationSettings,
    LinearAlgebraSettings,
)

LABELS = ("D11", "D12", "D22", "D66")

#  Both coupons are loaded on their top/bottom cuts once they are in their own
#  measurement frame, so the normal rotation there is W_Y for both.
NORMAL_ROTATION_DOF = "W_Y"


# =============================================================================
#  Building the variants
# =============================================================================

def clamp_variant(spec, clamped_edge: str):
    """Cases with ``clamped_edge``'s unknown moment annihilated.

    Adds ``W_Y = 0`` on ``clamped_edge`` so no traction there does virtual
    work, and keeps a single EXTERNAL group on the opposite cut carrying that
    cut's measured moment (taken from this step's own ``F``).
    """
    loaded_edge = "bottom" if clamped_edge == "top" else "top"
    cases = []
    for case in build_cases(spec, verbose=False):
        bc = copy.deepcopy(case.boundary_conditions)

        bc.prescribed = list(bc.prescribed) + [{
            "label": f"{clamped_edge}  dw*/dn = 0  (unknown moment)",
            "dof": NORMAL_ROTATION_DOF,
            "edge": clamped_edge,
            "value": 0.0,
        }]

        force = np.asarray(loadmat(case.data_file, squeeze_me=True)["F"],
                           dtype=float).ravel()
        kept = None
        for index, entry in enumerate(bc.external):
            if str(entry.get("edge")) == loaded_edge:
                kept = {**dict(entry), "value": float(force[index])}
                break
        if kept is None:
            raise SystemExit(
                f"{spec.name}: no EXTERNAL group on the {loaded_edge} edge to "
                "keep; this script assumes the shipped two-group layout.")
        bc.external = [kept]

        variant = dataclasses.replace(case)
        variant.boundary_conditions = bc
        #  The single remaining moment is now declared per step, so the table
        #  must win over the .mat F (which still has two entries).
        variant.applied_load_source = "manual"
        cases.append(variant)
    return cases


def both_ends(spec):
    """The shipped form: moment doing work at both cuts."""
    return build_cases(spec, verbose=False)


# =============================================================================
#  Running
# =============================================================================

def _settings():
    return (IdentificationSettings(max_iterations=40,
                                   convergence_tolerance=1.0e-8),
            LinearAlgebraSettings())


def run_ols(cases):
    identification, linear_algebra = _settings()
    return run_identification_cases(
        cases=cases, material_type=MATERIAL_TYPE, analysis_type=ANALYSIS_TYPE,
        initial_parameters=INITIAL_PARAMETERS,
        identification_settings=identification,
        linear_algebra_settings=linear_algebra, verbose=False)


def run_fgls(cases, experiment_id):
    identification, linear_algebra = _settings()
    return run_gls_identification_cases(
        cases=cases, material_type=MATERIAL_TYPE, analysis_type=ANALYSIS_TYPE,
        initial_parameters=INITIAL_PARAMETERS,
        identification_settings=identification,
        linear_algebra_settings=linear_algebra,
        experiment_ids=[experiment_id] * len(cases),
        gls_settings=GlsSettings(basis_degree=6, weighting="gls"),
        verbose=False)


def cut_moment_spread(spec) -> float:
    """How much the primary curvature varies end to end, as a fraction.

    Under a pure end couple the internal moment is uniform, so this is a direct
    read-out of how badly the "same M at both cuts" assumption is violated --
    and therefore of how much the two clamp choices should be expected to
    disagree.
    """
    ratios = []
    for path in mat_files(spec):
        data = loadmat(path, squeeze_me=True)
        nodes = np.asarray(data["nodes"], dtype=float)
        axis = 1 if spec.length_axis == "y" else 0
        key = "SK2grid" if spec.length_axis == "y" else "SK1grid"
        kappa = np.asarray(data[key], dtype=float)
        gradient, _ = np.polyfit(nodes[:, axis], kappa, 1)
        span = float(nodes[:, axis].max() - nodes[:, axis].min())
        ratios.append(abs(gradient * span / float(kappa.mean())))
    return float(np.mean(ratios))


def report(spec) -> None:
    print()
    print("=" * 78)
    print(f"  {spec.name.upper()}  "
          f"(length along {spec.length_axis}, identifies {spec.identifies}, "
          f"material frame {spec.material_orientation_deg:+.0f} deg)")
    print("=" * 78)

    spread = cut_moment_spread(spec)
    print(f"  Primary curvature varies {spread * 100:.0f} % end to end.")
    print("  Under a pure end couple the internal moment is UNIFORM, so this")
    print("  is how badly 'the same M at both cuts' is violated here.")
    print()

    variants = [
        ("(a) moment at BOTH cuts   [shipped]", both_ends(spec)),
        ("(b) clamp BOTTOM, load top          ", clamp_variant(spec, "bottom")),
        ("(b) clamp TOP,    load bottom       ", clamp_variant(spec, "top")),
    ]

    for estimator, runner in (("OLS", lambda c: run_ols(c)),
                              ("FGLS", lambda c: run_fgls(
                                  c, spec.experiment_id))):
        print(f"  --- {estimator} " + "-" * (72 - len(estimator)))
        print(f"  {'':36s}" + "".join(f"{n:>11}" for n in LABELS))
        target_values = []
        for name, cases in variants:
            theta = runner(cases).final_parameters
            target_values.append(float(theta[LABELS.index(spec.identifies)]))
            print(f"  {name:36s}"
                  + "".join(f"{v:11.3f}" for v in theta))
        both, clamp_bottom, clamp_top = target_values
        print(f"  {'':36s}" + f"  -> {spec.identifies}: both-cuts "
              f"{both:.2f}, one-cut {clamp_bottom:.2f} and {clamp_top:.2f}")
        low, high = min(clamp_bottom, clamp_top), max(clamp_bottom, clamp_top)
        brackets = low <= both <= high
        print(f"  {'':36s}     the two one-cut answers "
              f"{'BRACKET' if brackets else 'do NOT bracket'} the both-cuts "
              f"one; spread {high - low:.1f} N.mm "
              f"({(high - low) / abs(both) * 100:.0f} % of it)")
        print()


def write_clamped_project(clamped_edge: str, path):
    """Emit a loadable .vfmproj.json using the clamped-far-end form.

    The single surviving moment is written as a per-time-step value with
    ``applied_load_source = 'manual'``, so the GUI's load table is what feeds
    the solver -- the .mat ``F`` still has two entries and must not win.
    """
    from vfm_project import ProjectTest, VfmProject, save_project, describe_project

    tests = []
    for spec in (LONGITUDINAL, TRANSVERSE_RAW):
        for case in clamp_variant(spec, clamped_edge):
            bc = case.boundary_conditions
            tests.append(ProjectTest(
                data_file=case.data_file,
                experiment_id=spec.experiment_id,
                label=case.label,
                prescribed=[dict(e) for e in bc.prescribed],
                linking=[dict(e) for e in bc.linking],
                external=[dict(e) for e in bc.external],
                mesh_settings={
                    "number_of_elements_x": int(case.mesh_pkg.nx),
                    "number_of_elements_y": int(case.mesh_pkg.ny),
                    "bounds": {"x_min": float(case.mesh_pkg.x_min),
                               "x_max": float(case.mesh_pkg.x_max),
                               "y_min": float(case.mesh_pkg.y_min),
                               "y_max": float(case.mesh_pkg.y_max)},
                },
                sample_area_total=float(
                    case.measurements.area_per_point
                    * case.measurements.points.shape[0]),
                material_orientation_deg=spec.material_orientation_deg,
                applied_load_source="manual",
                description=(
                    f"{spec.name}: clamped-far-end form, {clamped_edge} cut's "
                    "unknown moment annihilated (W_Y = 0 there)."),
            ))

    project = VfmProject(
        tests=tests,
        experiment_names={LONGITUDINAL.experiment_id: "longitudinal",
                          TRANSVERSE_RAW.experiment_id: "transverse-raw"},
        material_type=MATERIAL_TYPE,
        analysis_type=ANALYSIS_TYPE,
        initial_parameters=[1.0, 1.0, 1.0, 1.0],
        max_iterations=40, convergence_tolerance=1.0e-8,
        basis_degree=6, weighting="gls", spatial_correction=False,
        description=(
            f"Clamped-far-end BC form: the {clamped_edge} cut's unknown "
            "moment does no virtual work; the opposite cut carries the "
            "measured moment as a per-step manual load."),
    )
    save_project(project, path)
    print(f"  wrote {path.name}")
    print(f"    {describe_project(project)}")
    return project


def main() -> int:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-projects", action="store_true",
        help="also emit loadable .vfmproj.json files for the clamped form")
    arguments = parser.parse_args()

    print("=" * 78)
    print("  BOUNDARY-CONDITION FORM: both cuts loaded vs clamped far end")
    print("  REAL DATA ONLY")
    print("=" * 78)
    print()
    print("  (a) both cuts loaded   -- w*, tangential slope and twist are zero")
    print("      on both cuts, so Q and the twisting moment do no work; the")
    print("      normal rotation is free at BOTH ends, so M works at both.")
    print()
    print("  (b) clamped far end    -- as (a), plus the normal rotation is")
    print("      prescribed to zero on one cut, so that cut's UNKNOWN moment")
    print("      does no virtual work either. One EXTERNAL group remains,")
    print("      carrying the measured moment on the opposite cut.")
    print()
    print("  Form (b) is the stricter formulation. Whether it is the better")
    print("  ESTIMATOR here is what the numbers below decide.")

    report(LONGITUDINAL)
    report(TRANSVERSE_RAW)

    if arguments.write_projects:
        print()
        print("=" * 78)
        print("  LOADABLE PROJECTS FOR THE CLAMPED FORM")
        print("=" * 78)
        PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
        for clamped_edge in ("bottom", "top"):
            write_clamped_project(
                clamped_edge,
                PROJECTS_DIR / f"campaign_clamped_{clamped_edge}.vfmproj.json")
        print()
        print("  Open either with:  python run_vfm.py  ->  Load project...")
        print("  Each experiment comes set to 'Applied load from: the values")
        print("  typed below', with the measured moment already filled in per")
        print("  step -- so the load table is what feeds the solver.")

    print()
    print("=" * 78)
    print("  HOW TO READ THIS")
    print("=" * 78)
    print("""
  Form (b) is right whenever one cut's moment is a reaction you do not
  measure -- an unknown traction must not do virtual work, and (a) lets it.

  But (b) then depends on ONE cut's moment being known, and on this rig it is
  not.  The machine reports a single couple (dTy); the export writes it to both
  cuts as F = [-M, +M], which ASSUMES equilibrium.  The longitudinal README's
  Finding 3 measures the violation: an unmeasured dead weight (~170 g of grip
  and fixture outboard of the load cell) makes the internal moment differ
  between the two cuts by about 100 % of the applied couple, and every force
  channel reads <= 0.1 N, so the load cell cannot see it.

  That is why the two one-cut answers land either side of the both-cuts one.
  With a virtual field antisymmetric along the length, (a)'s external work
  depends on the MEAN of the two cut moments -- which the machine couple does
  estimate well -- while each (b) depends on a single cut moment, which it does
  not.

  So on THIS data:
    * prefer (a) for a reported number, and
    * read the (b) spread as a measurement of the moment error, not as noise.

  On a rig where one end is genuinely instrumented and the other is a
  reaction, (b) is the formulation to use, and clamp_variant() above builds
  it.  The fix that would make (b) trustworthy here is the one Finding 3 names:
  weigh the fixture outboard of the load cell, or move the DIC window to where
  the curvature is flat.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
