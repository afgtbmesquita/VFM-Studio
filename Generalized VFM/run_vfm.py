# =============================================================================
#  plate / run_vfm.py
# -----------------------------------------------------------------------------
#  Entry point of the plate-VFM inverse solution.  Edit
#  ``plate/vfm_user_inputs.py`` to change the run (the file documents the
#  ``material_type`` and ``analysis_type`` knobs).  Then execute (from
#  inside this folder):
#
#      python run_vfm.py
#
#  The script does only a few things, in order:
#
#    1. Read ``vfm_user_inputs.build_user_inputs()`` to get the settings.
#    2. Open the run-setup GUI so you can confirm / change data files,
#       constitutive model, and initial parameters (seeded from step 1).
#    3. Open the GUI BC editor with no pre-drawn virtual mesh (measurement
#       point cloud on by default when a MAT path exists).  ``UserInputs.mesh``
#       seeds the Virtual mesh tab until you click Apply.
#    4. After you choose **Use saved file && run solver**, load the MAT file,
#       check BCs against ``analysis_type``, and run identification.
#    5. Print a short convergence report and the final ``K`` matrix.
#    6. Optionally open the 3D viewer of optimal virtual-field deformed shapes
#       (when Qt deps are present); the script asks [y/N] before launching.
# =============================================================================

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import List

import numpy as np

from vfm_data_io import (
    BoundaryConditions,
    load_boundary_conditions,
    load_point_measurements,
)
from vfm_gls_identification import run_gls_identification_cases
from vfm_identification import IdentificationCase, run_identification_cases
from vfm_mesh import build_mesh, prune_mesh_by_measurement_points
from vfm_user_inputs import build_user_inputs


def _check_bc_against_analysis(bc: BoundaryConditions,
                               analysis_type: str) -> List[str]:
    """Return human-readable warnings for BC entries that won't fit the run.

    Two kinds of mismatch are reported:
      * the BC file's recorded ``ANALYSIS_TYPE`` differs from the active one;
      * any entry references a DOF label that is not legal in this mode.
    Returns an empty list when the BC file is fully compatible.
    """
    from bc_gui.bc_model import allowed_dofs_for

    warnings: List[str] = []
    if bc.analysis_type is not None and bc.analysis_type != analysis_type:
        warnings.append(
            f"BC file declares ANALYSIS_TYPE='{bc.analysis_type}' but the "
            f"current run uses '{analysis_type}'."
        )

    allowed = set(allowed_dofs_for(analysis_type))
    bad: List[str] = []
    for category, entries in (("PRESCRIBED", bc.prescribed),
                              ("LINKING",    bc.linking),
                              ("EXTERNAL",   bc.external)):
        for idx, entry in enumerate(entries):
            dof = str(entry.get("dof", "")).strip().upper()
            if dof and dof not in allowed:
                label = entry.get("label", "<unlabeled>")
                bad.append(f"{category}[{idx}] '{label}': DOF '{dof}'")
    if bad:
        warnings.append(
            "Entries with DOF labels not allowed in this analysis type:\n   "
            + "\n   ".join(bad)
        )
    return warnings


def main() -> None:
    """Set up a run, solve it, and show the results.

    The flow is deliberately two-level, because the method is:

        EXPERIMENT  one physical test -- owns the boundary conditions, the
                    virtual mesh and the sample area
          TIME STEP one load step of that test -- owns a .mat file and the
                    applied force/moment

    Everything the setup dialog produces is a VfmProject, the same object a
    .vfmproj.json holds, so a GUI run and a headless run are the same run.
    """
    # ---- Headless project run --------------------------------------------
    #  `python run_vfm.py --project run.vfmproj.json` runs the file the GUI
    #  saves, without any dialog.  Useful on a machine with no display.
    argv = sys.argv[1:]
    if argv and argv[0] == "--project":
        if len(argv) < 2:
            print("[VFM-plate] --project needs a path to a .vfmproj.json")
            sys.exit(2)
        _run_project_headless(Path(argv[1]),
                              show_window="--no-gui" not in argv)
        return

    inputs = build_user_inputs()

    try:
        from bc_gui.experiment_setup_dialog import launch_experiment_setup
    except ImportError as exc:
        print(
            "[VFM-plate] could not import the run-setup GUI "
            f"({exc}).  Install the plate GUI extras (see requirements.txt), "
            "or run headlessly with --project <file>."
        )
        sys.exit(1)

    output_dir = Path(__file__).resolve().parent / "boundary_conditions"
    project, project_path = launch_experiment_setup(inputs, output_dir)
    if project is None:
        print("[VFM-plate] run setup cancelled; exiting.")
        sys.exit(1)

    from vfm_project import describe_project, project_to_cases, run_project

    print(f"[VFM-plate] {describe_project(project)}")
    for experiment_id, name, tests in _grouped(project):
        print(f"[VFM-plate]   {name}: {len(tests)} time step(s), "
              f"{len(tests[0].prescribed)} prescribed / "
              f"{len(tests[0].external)} external BC entries")

    try:
        ident_result = run_project(project, verbose=inputs.verbose)
    except (ValueError, FileNotFoundError) as error:
        print(f"[VFM-plate] the run could not be completed: {error}")
        sys.exit(1)

    _report(ident_result)
    cases, _ = project_to_cases(project)
    _show_results(ident_result, cases[0].mesh_pkg,
                  project_name=(project_path.name if project_path else ""))


def _grouped(project):
    from vfm_project import group_by_experiment
    return group_by_experiment(project)


def _show_results(ident_result, mesh_pkg, project_name: str = "") -> None:
    """Hand the run to the GUI results window.

    The terminal report above stays -- it is what a headless or piped run
    needs -- but the window is where the numbers, the iteration history and
    the route into the 3D virtual fields actually live.
    """
    try:
        from bc_gui.results_window import launch_results_window
    except ImportError as exc:
        print(f"[VFM-plate] could not open the results window ({exc}); "
              "the report above is the full output.")
        return
    launch_results_window(ident_result, mesh_pkg=mesh_pkg,
                          project_name=project_name)


def _run_project_headless(project_path: Path, show_window: bool = True) -> None:
    """Run a .vfmproj.json without the setup/BC dialogs."""
    from vfm_project import (
        describe_project,
        load_project,
        project_to_cases,
        run_project,
    )

    try:
        project = load_project(project_path)
    except (ValueError, FileNotFoundError, OSError) as error:
        print(f"[VFM-plate] {error}")
        sys.exit(1)

    print(f"[VFM-plate] project: {project_path.name}")
    print(f"[VFM-plate]   {describe_project(project)}")
    if project.description:
        print(f"[VFM-plate]   {project.description}")

    try:
        ident_result = run_project(project, verbose=True)
    except (ValueError, FileNotFoundError) as error:
        print(f"[VFM-plate] the project could not be run: {error}")
        sys.exit(1)

    _report(ident_result)
    if show_window:
        cases, _ = project_to_cases(project)
        _show_results(ident_result, cases[0].mesh_pkg,
                      project_name=project_path.name)


def _run_from_project(setup, inputs) -> None:
    """Solve straight from a loaded project and show the results window."""
    from vfm_project import describe_project, run_project

    project = setup.project
    name = (setup.project_path.name if setup.project_path else "project")
    print(f"[VFM-plate] running project {name}")
    print(f"[VFM-plate]   {describe_project(project)}")
    print("[VFM-plate]   boundary conditions and virtual meshes come from "
          "the project; the BC editor was skipped.")

    #  The dialog may have been edited after loading (different initial guess,
    #  different experiment grouping).  Honour the dialog, not the file.
    project.initial_parameters = [float(v) for v in setup.initial_parameters]
    if setup.experiment_ids and len(setup.experiment_ids) == len(project.tests):
        for test, experiment_id in zip(project.tests, setup.experiment_ids):
            test.experiment_id = int(experiment_id)

    try:
        ident_result = run_project(project, verbose=inputs.verbose)
    except (ValueError, FileNotFoundError) as error:
        print(f"[VFM-plate] the project could not be run: {error}")
        sys.exit(1)

    _report(ident_result)

    from vfm_project import project_to_cases
    cases, _ = project_to_cases(project)
    _show_results(ident_result, cases[0].mesh_pkg, project_name=name)




def _report(ident_result) -> None:
    """Print the full run report to stdout.

    Kept for headless and piped runs; the GUI results window shows the
    same content interactively.
    """
    # ---- 5. Final report -------------------------------------------------
    print()
    print("==============================================================")
    print("  Plate VFM converged stiffness coefficients (GLS)")
    print("==============================================================")
    labels = ident_result.constitutive_model.parameter_labels
    model_std = ident_result.parameter_std
    for index, (label, value) in enumerate(
            zip(labels, ident_result.final_parameters)):
        line = f"  {label:>5s} = {value:.6f}"
        if model_std is not None:
            line += f"  +/- {model_std[index]:.6f}"
        print(line)
    if model_std is not None:
        print()
        print("  The +/- is the model-based first-order GLS covariance,")
        print("  eq. (20).  It is a LOWER BOUND: it excludes the uncertainty")
        print("  in the estimated noise covariance and the higher-order effect")
        print("  of virtual fields built from noisy data.  Use")
        print("  vfm_gls_identification.jackknife_parameters for an empirical")
        print("  interval.")
    print()
    print(f"  Weighting          : {ident_result.weighting}")
    print(f"  Iterations executed: {ident_result.iterations_completed}")
    print(f"  Converged          : {ident_result.converged}")
    if ident_result.history:
        last = ident_result.history[-1]
        print(f"  Last deltaQ        : {last.delta_q:.3e}")
        print(f"  Last deltaC        : {ident_result.covariance_delta:.3e}")
        print(f"  Last cond(whitened): {last.condition_number:.3e}")
        print(f"  Last ||H - I||_F   : {last.identity_deviation:.3e}")

    print()
    print("  Per-experiment noise model:")
    for report in ident_result.experiments:
        correlation = report.noise.correlation
        diagnostics = report.covariance_diagnostics
        primary = int(np.argmax(report.noise.residual_scale[0]))
        temporal = report.noise.temporal_correlation(primary)
        off_diagonal = temporal[np.triu_indices(report.noise.n_steps, k=1)]
        print(f"    experiment {report.experiment_id} ({report.label}): "
              f"{report.n_steps} load steps, {report.n_points} points")
        print(f"      Gamma {report.noise.gamma.shape[0]}x"
              f"{report.noise.gamma.shape[0]}, "
              f"mean cross-step correlation = "
              f"{(off_diagonal.mean() if off_diagonal.size else 0.0):.3f}")
        print(f"      spatial rho(nearest) = {correlation.nearest_rho:.3f}, "
              f"kappa = {correlation.kappa:.2f}")
        print(f"      C: PD = {diagnostics['positive_definite']}, "
              f"cond = {diagnostics['condition_number']:.3e}")

    if ident_result.K_matrix is not None:
        with np.printoptions(precision=4, suppress=True):
            print()
            print("  Stacked H matrix:")
            print(ident_result.K_matrix)



if __name__ == "__main__":
    main()
