"""Identification and identifiability jobs. No Qt."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from vfmstudio.core import bridge
from vfmstudio.core.axes_align import apply_alignment_to_measurements
from vfmstudio.core.document import ExperimentModel, SessionDocument, document_to_vfm_project
from vfmstudio.core.fields import experiment_alignment_angle


@dataclass
class JobReport:
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def _experiment_label(experiment: ExperimentModel) -> str:
    return experiment.name or f"Experiment {experiment.experiment_id}"


def validate_analysis_window(experiment: ExperimentModel) -> List[str]:
    errors: List[str] = []
    label = _experiment_label(experiment)
    n_steps = len(experiment.steps)
    if n_steps == 0:
        return errors
    if experiment.analysis_start is not None:
        start = int(experiment.analysis_start)
        if start < 1 or start > n_steps:
            errors.append(
                f"{label}: analysis start {start} is outside 1…{n_steps}."
            )
    if experiment.analysis_end is not None:
        end = int(experiment.analysis_end)
        if end < 1 or end > n_steps:
            errors.append(
                f"{label}: analysis end {end} is outside 1…{n_steps}."
            )
    start = experiment.resolved_analysis_start()
    end = experiment.resolved_analysis_end()
    if experiment.analysis_start is not None and experiment.analysis_end is not None:
        if int(experiment.analysis_start) > int(experiment.analysis_end):
            errors.append(
                f"{label}: analysis start {experiment.analysis_start} is after "
                f"end {experiment.analysis_end}."
            )
    elif start > end:
        errors.append(f"{label}: analysis start {start} is after end {end}.")
    if experiment.reference_step is not None:
        reference = int(experiment.reference_step)
        if reference < 1 or reference > n_steps:
            errors.append(
                f"{label}: reference step {reference} is outside 1…{n_steps}."
            )
    if not experiment.included_step_indices():
        errors.append(
            f"{label}: no load steps remain after the analysis window "
            "and reference step."
        )
    return errors


def validate_document(document: SessionDocument) -> JobReport:
    from vfmstudio.core.step_series import relocate_document_data_files

    relocate_document_data_files(document)
    report = JobReport()
    if not document.experiments:
        report.errors.append("Project has no experiments.")
        return report
    for experiment in document.experiments:
        label = _experiment_label(experiment)
        if not experiment.steps:
            report.errors.append(f"{label}: no load steps.")
        report.errors.extend(validate_analysis_window(experiment))
        for step in experiment.steps:
            if not step.data_file:
                report.errors.append(f"{label}: a load step has no data file.")
            elif not Path(step.data_file).is_file():
                report.errors.append(
                    f"{label}: measurement file not found: {step.data_file}"
                )
    try:
        project = project_for_identification(document)
    except Exception as exc:
        report.errors.append(f"Cannot build solver project: {exc}")
        return report
    report.errors.extend(bridge.check_experiment_consistency(project))
    report.warnings.extend(bridge.check_load_orientation(project))
    return report


def tests_by_experiment(project) -> dict:
    grouped: dict = {}
    for test in list(getattr(project, "tests", []) or []):
        grouped.setdefault(int(test.experiment_id), []).append(test)
    return grouped


def analysis_load_indices(experiment: ExperimentModel) -> List[int]:
    """0-based step indices to load: analysis window plus the reference."""
    indices = list(experiment.analysis_window_indices())
    if experiment.reference_step is None:
        return indices
    reference = int(experiment.reference_step) - 1
    if 0 <= reference < len(experiment.steps) and reference not in indices:
        indices.append(reference)
    return indices


def project_for_identification(document: SessionDocument):
    """VfmProject containing only the steps needed for this identification."""
    from vfmstudio.core.step_series import relocate_document_data_files

    relocate_document_data_files(document)
    project = document_to_vfm_project(document)
    grouped = tests_by_experiment(project)
    kept = []
    for experiment in document.experiments:
        tests = grouped.get(int(experiment.experiment_id), [])
        for index in analysis_load_indices(experiment):
            if 0 <= index < len(tests):
                kept.append(tests[index])
    project.tests = kept
    return project


def subtract_reference_measurements(target: Any, reference: Any) -> None:
    """Subtract reference strains and applied force from ``target`` in place."""
    target_strains = np.asarray(getattr(target, "strains"), dtype=float)
    reference_strains = np.asarray(getattr(reference, "strains"), dtype=float)
    if target_strains.shape != reference_strains.shape:
        raise ValueError(
            "Cannot reference load steps: strain fields have different shapes "
            f"{target_strains.shape} vs {reference_strains.shape}."
        )
    target.strains = target_strains - reference_strains
    target_force = getattr(target, "applied_force", None)
    reference_force = getattr(reference, "applied_force", None)
    if target_force is None and reference_force is None:
        return
    if target_force is None or reference_force is None:
        raise ValueError(
            "Cannot reference load steps: applied force is missing on the "
            "reference or a later step."
        )
    target_force = np.atleast_1d(np.asarray(target_force, dtype=float)).ravel()
    reference_force = np.atleast_1d(
        np.asarray(reference_force, dtype=float)
    ).ravel()
    if target_force.shape != reference_force.shape:
        raise ValueError(
            "Cannot reference load steps: applied-force vectors have different "
            f"lengths {target_force.size} vs {reference_force.size}."
        )
    target.applied_force = target_force - reference_force


def apply_reference_subtraction(
    document: SessionDocument,
    cases: Sequence[Any],
    experiment_ids: Sequence[int],
    step_indices: Sequence[int],
) -> Tuple[List[Any], List[int], List[int]]:
    """Subtract each experiment's reference case and drop it from the list."""
    kept_cases: List[Any] = []
    kept_ids: List[int] = []
    kept_steps: List[int] = []
    by_experiment: dict = {}
    for case, exp_id, step_index in zip(cases, experiment_ids, step_indices):
        by_experiment.setdefault(int(exp_id), []).append((case, step_index))

    reference_by_exp: dict = {}
    for experiment in document.experiments:
        if experiment.reference_step is None:
            continue
        reference = int(experiment.reference_step) - 1
        entries = by_experiment.get(int(experiment.experiment_id), [])
        match = [case for case, index in entries if index == reference]
        if not match:
            raise ValueError(
                f"{_experiment_label(experiment)}: reference step "
                f"{experiment.reference_step} was not loaded."
            )
        reference_by_exp[int(experiment.experiment_id)] = match[0]

    for case, exp_id, step_index in zip(cases, experiment_ids, step_indices):
        reference_case = reference_by_exp.get(int(exp_id))
        if reference_case is not None:
            experiment = document.experiment_by_id(int(exp_id))
            reference_index = (
                None
                if experiment is None or experiment.reference_step is None
                else int(experiment.reference_step) - 1
            )
            if step_index == reference_index:
                continue
            subtract_reference_measurements(
                case.measurements, reference_case.measurements
            )
        kept_cases.append(case)
        kept_ids.append(int(exp_id))
        kept_steps.append(int(step_index))
    return kept_cases, kept_ids, kept_steps


def apply_axes_alignment(
    document: SessionDocument,
    cases: Sequence[Any],
    experiment_ids: Sequence[int],
) -> None:
    """Rotate each case's measurements onto the lab axes when the experiment asks."""
    angles: dict[int, float] = {}
    for experiment in document.experiments:
        if not experiment.fit_data_to_system_axes:
            continue
        angle = experiment_alignment_angle(
            experiment,
            document.material.analysis_type,
            document.source_path,
        )
        if abs(angle) <= 1.0e-12:
            continue
        angles[int(experiment.experiment_id)] = angle
    if not angles:
        return
    for case, exp_id in zip(cases, experiment_ids):
        angle = angles.get(int(exp_id))
        if angle is None:
            continue
        measurements = getattr(case, "measurements", None)
        if measurements is None:
            continue
        apply_alignment_to_measurements(measurements, angle)


def _step_indices_for_project(document: SessionDocument) -> List[int]:
    indices: List[int] = []
    for experiment in document.experiments:
        indices.extend(analysis_load_indices(experiment))
    return indices


def run_identification(document: SessionDocument, verbose: bool = True) -> Any:
    report = validate_document(document)
    if report.errors:
        raise ValueError("\n".join(report.errors))
    project = project_for_identification(document)
    needs_reference = any(
        experiment.reference_step is not None for experiment in document.experiments
    )
    needs_alignment = any(
        experiment.fit_data_to_system_axes for experiment in document.experiments
    )
    if not needs_reference and not needs_alignment:
        return bridge.run_project(project, verbose=verbose)
    cases, experiment_ids = bridge.project_to_cases(project, verbose=verbose)
    step_indices = _step_indices_for_project(document)
    if len(step_indices) != len(cases):
        raise ValueError(
            "Internal error: loaded cases do not match the analysis window."
        )
    apply_axes_alignment(document, cases, experiment_ids)
    if needs_reference:
        cases, experiment_ids, _ = apply_reference_subtraction(
            document, cases, experiment_ids, step_indices
        )
    if not cases:
        raise ValueError("No load steps remain after referencing.")
    return bridge.run_gls_identification_cases(
        project, cases, experiment_ids, verbose=verbose
    )


def run_identifiability(document: SessionDocument, result: Any) -> Any:
    cases = bridge.first_attr(result, "diagnostic_cases", default=None)
    model = bridge.first_attr(result, "constitutive_model", default=None)
    if cases is None or model is None:
        project = project_for_identification(document)
        built = bridge.project_to_cases(project)
        cases = built[0] if isinstance(built, tuple) else built
        if model is None:
            constitutive = bridge.vfm_constitutive()
            define = constitutive.define_constitutive_model
            theta = bridge.first_attr(
                result,
                "final_parameters",
                default=document.material.initial_parameters,
            )
            model = define(
                document.material.material_type,
                document.material.analysis_type,
                theta,
            )
        needs_reference = any(
            experiment.reference_step is not None
            for experiment in document.experiments
        )
        experiment_ids = built[1] if isinstance(built, tuple) else [
            experiment.experiment_id
            for experiment in document.experiments
            for _ in analysis_load_indices(experiment)
        ]
        apply_axes_alignment(document, cases, experiment_ids)
        if needs_reference:
            step_indices = _step_indices_for_project(document)
            cases, _, _ = apply_reference_subtraction(
                document, cases, experiment_ids, step_indices
            )
    return bridge.compute_identifiability(cases, model)


def reduced_chi_square(result: Any) -> Optional[dict]:
    try:
        design = bridge.first_attr(result, "K_matrix")
        rhs = bridge.first_attr(result, "U_vector")
        covariance = bridge.first_attr(result, "global_covariance")
        theta = bridge.first_attr(result, "final_parameters")
        if design is None or rhs is None or covariance is None or theta is None:
            return None
        residual = np.asarray(design) @ np.asarray(theta) - np.asarray(rhs)
        dof = int(np.asarray(design).shape[0] - np.asarray(design).shape[1])
        if dof <= 0:
            return None
        total = float(residual @ np.linalg.solve(np.asarray(covariance), residual))
        reduced = total / dof
        return {
            "chi2": total,
            "dof": dof,
            "reduced": reduced,
            "birge": float(np.sqrt(max(reduced, 0.0))),
        }
    except Exception:
        return None


def _experiment_names(document: SessionDocument) -> dict[int, str]:
    return {
        int(experiment.experiment_id): (
            experiment.name or f"Experiment {experiment.experiment_id}"
        )
        for experiment in document.experiments
    }


def run_uncertainty(document: SessionDocument, result: Any, kind: str = "CR2"):
    """Sandwich covariance, experiment scores, and fixed-weight leave-one-out."""
    from vfmstudio.core.uncertainty import analyse

    return analyse(
        result,
        labels=document.material.labels(),
        experiment_names=_experiment_names(document),
        kind=kind,
    )


def run_loo_refits(
    document: SessionDocument,
    uncertainty,
    *,
    kind: Optional[str] = None,
    log: Optional[Callable[[str], None]] = None,
):
    """Re-run FGLS without each experiment and attach those θ_(-i)."""
    from vfmstudio.core.uncertainty import attach_refits

    def refit_fn(experiment_id: int, name: str):
        if log is not None:
            log(f"Refitting FGLS without {name}…")
        clone = document.model_copy(deep=True)
        clone.experiments = [
            experiment
            for experiment in clone.experiments
            if int(experiment.experiment_id) != int(experiment_id)
        ]
        if not clone.experiments:
            raise ValueError("Need at least one remaining experiment to refit.")
        return run_identification(clone, verbose=False)

    return attach_refits(uncertainty, refit_fn, kind=kind)
