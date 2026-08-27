# =============================================================================
#  Generalized VFM / make_gui_project.py
# -----------------------------------------------------------------------------
#  Build projects/campaign.vfmproj.json -- the loadable run description for the GUI --
#  from exactly the same TestSpecs, BC files and mesh settings that
#  run_campaign_gls.py uses, then PROVE it reproduces the campaign numbers.
#
#  The proof matters more than the file.  A project that merely looks right is
#  worth nothing; this script runs the identification twice, once through
#  campaign_common (the published path) and once through the project, and
#  refuses to leave the file on disk if the two disagree beyond round-off.
#
#  Run:  python -m workflows.make_gui_project
#  Then: python run_vfm.py  ->  "Load project..."  ->  projects/campaign.vfmproj.json
# =============================================================================

from __future__ import annotations

import sys

import numpy as np

from workflows.campaign_common import (
    ANALYSIS_TYPE,
    HERE,
    INITIAL_PARAMETERS,
    MATERIAL_TYPE,
    PROJECTS_DIR,
    TESTS,
    TESTS_RAW_FRAME,
    build_campaign,
    mat_files,
)
from vfm_gls_identification import run_gls_identification_cases
from vfm_project import (
    VfmProject,
    describe_project,
    load_project,
    project_test_from_bc_file,
    run_project,
    save_project,
)
from vfm_user_inputs import (
    GlsSettings,
    IdentificationSettings,
    LinearAlgebraSettings,
)

PROJECT_PATH = PROJECTS_DIR / "campaign.vfmproj.json"

#  The tolerance the two routes must agree to.  They should be bit-identical
#  in exact arithmetic; 1e-9 relative allows only for a different summation
#  order inside numpy.
AGREEMENT_TOLERANCE = 1.0e-9


def build(specs=TESTS) -> VfmProject:
    """One ProjectTest per load step, carrying its own mesh and BCs."""
    tests = []
    for spec in specs:
        paths = mat_files(spec)
        print(f"  {spec.name:<13} {len(paths)} load steps, "
              f"BC = {spec.bc_file.name}, experiment {spec.experiment_id}, "
              f"material frame {spec.material_orientation_deg:+.1f} deg")
        for path in paths:
            tests.append(project_test_from_bc_file(
                data_file=path,
                bc_file=spec.bc_file,
                experiment_id=spec.experiment_id,
                label=f"{spec.name}:{path.stem}",
                material_orientation_deg=spec.material_orientation_deg,
            ))

    return VfmProject(
        tests=tests,
        experiment_names={spec.experiment_id: spec.name for spec in specs},
        material_type=MATERIAL_TYPE,
        analysis_type=ANALYSIS_TYPE,
        #  All ones: the GUI opens on the least informative admissible start,
        #  and the FGLS fixed point is independent of it.
        initial_parameters=[1.0, 1.0, 1.0, 1.0],
        max_iterations=40,
        convergence_tolerance=1.0e-8,
        basis_degree=6,
        weighting="gls",
        #  eq. (3) verbatim.  kappa is measured and reported, never applied.
        spatial_correction=False,
        covariance_tolerance=1.0e-4,
        description=(
            "Conference campaign: longitudinal (experiment 1) and transverse "
            "(experiment 2) bending, 9 load steps each. Reproduces "
            "run_campaign_gls.py."),
    )


def reference_run():
    """The published path: campaign_common -> run_gls_identification_cases."""
    cases, ids, _ = build_campaign(verbose=False)
    return run_gls_identification_cases(
        cases=cases, material_type=MATERIAL_TYPE, analysis_type=ANALYSIS_TYPE,
        initial_parameters=[1.0, 1.0, 1.0, 1.0],
        identification_settings=IdentificationSettings(
            max_iterations=40, convergence_tolerance=1.0e-8),
        linear_algebra_settings=LinearAlgebraSettings(),
        experiment_ids=ids,
        gls_settings=GlsSettings(basis_degree=6, weighting="gls",
                                 spatial_correction=False),
        verbose=False)


def main() -> int:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("  BUILDING THE GUI PROJECT FILE")
    print("=" * 78)
    #  The project uses the RAW-FRAME transverse data with a 90 deg material
    #  orientation, while reference_run() below still uses the pre-rotated
    #  export with no rotation at all.  So check 2 is no longer only "does the
    #  project file reproduce the campaign" -- it is simultaneously "does
    #  rotating D reproduce rotating the fields", on the full two-coupon run.
    project = build(TESTS_RAW_FRAME)
    save_project(project, PROJECT_PATH)
    print(f"\n  wrote {PROJECT_PATH.name}")
    print(f"  {describe_project(project)}")

    # ---- 1. does it survive a round trip? --------------------------------
    print()
    print("-" * 78)
    print("  1. ROUND TRIP: does the file read back as what was written?")
    print("-" * 78)
    reloaded = load_project(PROJECT_PATH)
    problems = []
    if len(reloaded.tests) != len(project.tests):
        problems.append(f"{len(reloaded.tests)} tests vs {len(project.tests)}")
    for original, back in zip(project.tests, reloaded.tests):
        if back.data_file.resolve() != original.data_file.resolve():
            problems.append(f"path drift on {original.label}")
        if back.experiment_id != original.experiment_id:
            problems.append(f"experiment id drift on {original.label}")
        if len(back.prescribed) != len(original.prescribed) or \
                len(back.external) != len(original.external):
            problems.append(f"BC count drift on {original.label}")
        if not back.data_file.is_file():
            problems.append(f"unresolvable data file for {original.label}")
    if reloaded.initial_parameters != list(project.initial_parameters):
        problems.append("initial parameters drift")
    if reloaded.spatial_correction != project.spatial_correction:
        problems.append("spatial_correction drift")
    if problems:
        print("  [FAIL] " + "; ".join(problems))
        return 1
    print(f"  [PASS] {len(reloaded.tests)} tests, all data files resolve, "
          "BCs and settings identical")

    # ---- 2. does running it reproduce the campaign? ----------------------
    print()
    print("-" * 78)
    print("  2. EQUIVALENCE: does the project reproduce run_campaign_gls.py?")
    print("-" * 78)
    print("  reference: pre-rotated transverse fields, D unrotated")
    print("  project  : raw transverse fields, D rotated by 90 deg")
    print("  running the published path (campaign_common)...")
    reference = reference_run()
    print("  running the project path (vfm_project)...")
    from_project = run_project(reloaded)

    labels = list(reference.constitutive_model.parameter_labels)
    gap = np.max(np.abs(reference.final_parameters
                        - from_project.final_parameters)
                 / np.maximum(np.abs(reference.final_parameters), 1e-30))
    print(f"    {'':<22}" + "".join(f"{n:>12}" for n in labels))
    print(f"    {'campaign_common':<22}"
          + "".join(f"{v:12.6f}" for v in reference.final_parameters))
    print(f"    {'project file':<22}"
          + "".join(f"{v:12.6f}" for v in from_project.final_parameters))
    print(f"    max relative difference: {gap:.3e}")

    sigma_gap = np.max(np.abs(reference.parameter_std
                              - from_project.parameter_std)
                       / np.maximum(np.abs(reference.parameter_std), 1e-30))
    print(f"    eq. (20) sigma agreement: {sigma_gap:.3e}")

    if gap > AGREEMENT_TOLERANCE or sigma_gap > AGREEMENT_TOLERANCE:
        print(f"  [FAIL] the project does NOT reproduce the campaign "
              f"(tolerance {AGREEMENT_TOLERANCE:.0e}).")
        PROJECT_PATH.unlink(missing_ok=True)
        print(f"  removed {PROJECT_PATH.name} so it cannot be used.")
        return 1
    print("  [PASS] identical to round-off")

    # ---- 3. the numbers the GUI will show --------------------------------
    from workflows.make_figures import chi_square_diagnostics
    goodness = chi_square_diagnostics(from_project)
    print()
    print("=" * 78)
    print("  WHAT THE GUI WILL SHOW")
    print("=" * 78)
    for label, value, sigma in zip(labels, from_project.final_parameters,
                                   from_project.parameter_std):
        print(f"  {label:>5} = {value:11.4f}   eq.(20) sigma {sigma:8.4f}   "
              f"x sqrt(chi2/dof) -> {sigma * goodness['birge']:8.4f}")
    print(f"  converged in {from_project.iterations_completed} iterations; "
          f"chi2/dof = {goodness['reduced']:.0f}")
    print()
    print("  Load it with:  python run_vfm.py  ->  \"Load project...\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
