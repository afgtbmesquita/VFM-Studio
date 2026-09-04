"""Select and reconstruct optimal virtual fields for one experiment load step.

FGLS solves one virtual field per constitutive unknown at every load step of
every experiment. The identification result only stores the fields from the
first experiment's first step (``optimal_field_results``). This module lists
the identified (experiment, step) pairs and re-solves the same optimal-field
problem for the pair the user picked, using the final constitutive model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from vfmstudio.core import bridge, mesh_convert
from vfmstudio.core.document import SessionDocument
from vfmstudio.core.step_series import step_display_label


@dataclass(frozen=True)
class VirtualFieldTarget:
    experiment_id: int
    experiment_name: str
    step_index: int
    document_step_index: Optional[int]
    step_label: str


@dataclass
class VirtualFieldViewData:
    u_list: list[np.ndarray]
    labels: list[str]
    mesh_pkg: Any
    experiment_name: str
    step_label: str
    source: str
    warning: Optional[str] = None


_CASE_CACHE: dict[int, dict[tuple[int, int], Any]] = {}
_FIELD_CACHE: dict[tuple[int, int, int], VirtualFieldViewData] = {}


def parameter_labels_for(result: Any, document: Optional[SessionDocument]) -> list[str]:
    labels: list[str] = []
    model = getattr(result, "constitutive_model", None)
    if model is not None:
        labels = list(getattr(model, "parameter_labels", []) or [])
    fields = list(getattr(result, "optimal_field_results", None) or [])
    n = len(fields)
    if document is not None and len(labels) < max(n, 1):
        labels = list(document.material.labels())
    if n:
        while len(labels) < n:
            labels.append(f"P{len(labels) + 1}")
        return labels[:n]
    return labels


def list_virtual_field_targets(
    document: Optional[SessionDocument],
    result: Any,
) -> list[VirtualFieldTarget]:
    """Identified (experiment, load-step) pairs that have virtual fields."""
    if result is None:
        return []
    reports = list(getattr(result, "experiments", None) or [])
    if reports:
        return _targets_from_reports(document, reports)
    return _targets_from_document(document)


def resolve_virtual_field_target(
    document: Optional[SessionDocument],
    result: Any,
    experiment_id: Optional[int],
    step_index: Optional[int],
) -> Optional[VirtualFieldTarget]:
    targets = list_virtual_field_targets(document, result)
    if not targets:
        return None
    if experiment_id is None:
        return targets[0]
    wanted = int(experiment_id)
    for item in targets:
        if item.experiment_id != wanted:
            continue
        if step_index is None or int(item.step_index) == int(step_index):
            return item
    matching = [item for item in targets if item.experiment_id == wanted]
    return matching[0] if matching else targets[0]


def virtual_fields_for_target(
    document: Optional[SessionDocument],
    result: Any,
    experiment_id: Optional[int] = None,
    step_index: Optional[int] = None,
) -> Optional[VirtualFieldViewData]:
    """Nodal virtual fields for one identified experiment and load step."""
    if result is None:
        return None
    target = resolve_virtual_field_target(
        document, result, experiment_id, step_index
    )
    labels = parameter_labels_for(result, document)
    cache_key = (
        id(result),
        0 if target is None else int(target.experiment_id),
        0 if target is None else int(target.step_index),
    )
    cached = _FIELD_CACHE.get(cache_key)
    if cached is not None:
        return cached

    solved = None
    warning = None
    if target is not None:
        try:
            solved = _solve_target(document, result, target, labels)
        except Exception as exc:
            warning = str(exc)
            solved = None

    if solved is not None:
        _FIELD_CACHE[cache_key] = solved
        return solved

    stored = _stored_fields(document, result, labels, target, warning)
    if stored is not None:
        _FIELD_CACHE[cache_key] = stored
    return stored


def _targets_from_reports(document: Optional[SessionDocument], reports) -> list[VirtualFieldTarget]:
    targets: list[VirtualFieldTarget] = []
    for report in reports:
        experiment_id = int(getattr(report, "experiment_id", 0) or 0)
        experiment = (
            None if document is None else document.experiment_by_id(experiment_id)
        )
        name = _experiment_name(experiment, report, experiment_id)
        n_steps = int(getattr(report, "n_steps", 0) or 0)
        included = (
            list(experiment.included_step_indices()) if experiment is not None else []
        )
        report_labels = [str(item) for item in (getattr(report, "step_labels", None) or [])]
        for local in range(n_steps):
            doc_index = included[local] if local < len(included) else None
            label = _step_label(experiment, doc_index, report_labels, local)
            targets.append(
                VirtualFieldTarget(
                    experiment_id=experiment_id,
                    experiment_name=name,
                    step_index=local,
                    document_step_index=doc_index,
                    step_label=label,
                )
            )
    return targets


def _targets_from_document(document: Optional[SessionDocument]) -> list[VirtualFieldTarget]:
    if document is None:
        return []
    targets: list[VirtualFieldTarget] = []
    for experiment in document.experiments_in_identification():
        name = experiment.name or f"Experiment {experiment.experiment_id}"
        included = list(experiment.included_step_indices())
        for local, doc_index in enumerate(included):
            targets.append(
                VirtualFieldTarget(
                    experiment_id=int(experiment.experiment_id),
                    experiment_name=name,
                    step_index=local,
                    document_step_index=int(doc_index),
                    step_label=_step_label(experiment, doc_index, (), local),
                )
            )
    return targets


def _experiment_name(experiment, report, experiment_id: int) -> str:
    if experiment is not None and (experiment.name or "").strip():
        return experiment.name.strip()
    label = str(getattr(report, "label", "") or "").strip()
    return label or f"Experiment {experiment_id}"


def _step_label(experiment, document_step_index, report_labels, local: int) -> str:
    if (
        experiment is not None
        and document_step_index is not None
        and 0 <= int(document_step_index) < len(experiment.steps)
    ):
        step = experiment.steps[int(document_step_index)]
        text = step_display_label(step) or f"step {int(document_step_index) + 1}"
        return f"Step {int(document_step_index) + 1:02d} — {text}"
    if local < len(report_labels) and str(report_labels[local]).strip():
        return str(report_labels[local]).strip()
    return f"Step {local + 1}"


def _stored_fields(
    document: Optional[SessionDocument],
    result: Any,
    labels: list[str],
    target: Optional[VirtualFieldTarget],
    warning: Optional[str],
) -> Optional[VirtualFieldViewData]:
    fields = list(getattr(result, "optimal_field_results", None) or [])
    u_list = [_u_from_field(item) for item in fields if _u_from_field(item).size]
    if not u_list:
        return None
    mesh_pkg = _mesh_for_fields(document, u_list[0], target)
    names = labels[: len(u_list)]
    while len(names) < len(u_list):
        names.append(f"P{len(names) + 1}")
    first = target is None or (
        target.step_index == 0
        and _is_first_identified_experiment(document, result, target.experiment_id)
    )
    note = warning
    if not first:
        extra = (
            "Identify only stores the first experiment's first load step. "
            "Could not rebuild the other fields"
            + (f": {warning}" if warning else ".")
        )
        note = extra
    return VirtualFieldViewData(
        u_list=u_list,
        labels=names,
        mesh_pkg=mesh_pkg,
        experiment_name="" if target is None else target.experiment_name,
        step_label="" if target is None else target.step_label,
        source="stored",
        warning=note,
    )


def _is_first_identified_experiment(
    document: Optional[SessionDocument],
    result: Any,
    experiment_id: int,
) -> bool:
    targets = list_virtual_field_targets(document, result)
    return bool(targets) and int(targets[0].experiment_id) == int(experiment_id)


def _u_from_field(item: Any) -> np.ndarray:
    if hasattr(item, "nodal_variables_k"):
        return np.asarray(item.nodal_variables_k, dtype=float).ravel()
    return np.asarray(item, dtype=float).ravel()


def _mesh_for_fields(
    document: Optional[SessionDocument],
    u: np.ndarray,
    target: Optional[VirtualFieldTarget],
):
    if document is None:
        return None
    ordered = []
    seen: set[int] = set()
    if target is not None:
        experiment = document.experiment_by_id(target.experiment_id)
        if experiment is not None:
            ordered.append(experiment)
            seen.add(int(experiment.experiment_id))
    for experiment in document.experiments:
        if int(experiment.experiment_id) in seen:
            continue
        ordered.append(experiment)
    n_dof = int(np.asarray(u).size)
    for experiment in ordered:
        try:
            mesh_pkg = mesh_convert.build_experiment_mesh(document, experiment)
        except Exception:
            continue
        if int(getattr(mesh_pkg, "n_dofs", -1)) == n_dof:
            return mesh_pkg
    return None


def _solve_target(
    document: Optional[SessionDocument],
    result: Any,
    target: VirtualFieldTarget,
    labels: list[str],
) -> Optional[VirtualFieldViewData]:
    case = _case_for_target(document, result, target)
    if case is None:
        raise ValueError(
            "No identification case is available for this experiment and step."
        )
    model = _constitutive_model(document, result)
    if model is None:
        raise ValueError("Identification result has no constitutive model.")
    opt = bridge._load("vfm_optimal_field")
    constitutive = bridge.vfm_constitutive()
    mesh_mod = bridge._load("vfm_mesh")
    angle = float(getattr(case, "material_orientation_deg", 0.0) or 0.0)
    rotated = constitutive.rotate_constitutive_model(model, angle)
    mapping = mesh_mod.map_points_to_mesh(case.measurements.points, case.mesh_pkg)
    payload = opt.collect_assembly_payload(
        case.measurements,
        case.mesh_pkg,
        mapping,
        int(rotated.n_generalised_strain),
    )
    fields = opt.solve_optimal_virtual_fields(
        measurements=case.measurements,
        mesh_pkg=case.mesh_pkg,
        boundary_conditions=case.boundary_conditions,
        constitutive_model=rotated,
        mapping=mapping,
        payload=payload,
    )
    u_list = [_u_from_field(item) for item in fields]
    names = list(getattr(rotated, "parameter_labels", None) or labels)[: len(u_list)]
    while len(names) < len(u_list):
        names.append(f"P{len(names) + 1}")
    return VirtualFieldViewData(
        u_list=u_list,
        labels=names,
        mesh_pkg=case.mesh_pkg,
        experiment_name=target.experiment_name,
        step_label=target.step_label,
        source="solved",
    )


def _constitutive_model(document: Optional[SessionDocument], result: Any):
    model = getattr(result, "constitutive_model", None)
    if model is not None and getattr(model, "material_matrix", None) is not None:
        return model
    if document is None:
        return None
    theta = getattr(result, "final_parameters", None)
    if theta is None:
        return None
    constitutive = bridge.vfm_constitutive()
    return constitutive.define_constitutive_model(
        document.material.material_type,
        document.material.analysis_type,
        theta,
    )


def _case_for_target(document, result, target: VirtualFieldTarget):
    indexed = _indexed_cases(document, result)
    return indexed.get((int(target.experiment_id), int(target.step_index)))


def _indexed_cases(document, result) -> dict[tuple[int, int], Any]:
    result_id = id(result)
    cached = _CASE_CACHE.get(result_id)
    if cached is not None:
        return cached
    indexed = _cases_from_result(result)
    if not indexed:
        try:
            indexed = _cases_from_document(document)
        except Exception:
            indexed = {}
    _CASE_CACHE[result_id] = indexed
    if len(_CASE_CACHE) > 4:
        oldest = next(iter(_CASE_CACHE))
        if oldest != result_id:
            _CASE_CACHE.pop(oldest, None)
    return indexed


def _cases_from_result(result) -> dict[tuple[int, int], Any]:
    cases = list(getattr(result, "diagnostic_cases", None) or [])
    reports = list(getattr(result, "experiments", None) or [])
    if not cases or not reports:
        return {}
    indexed: dict[tuple[int, int], Any] = {}
    offset = 0
    for report in reports:
        experiment_id = int(getattr(report, "experiment_id", 0) or 0)
        n_steps = int(getattr(report, "n_steps", 0) or 0)
        for local in range(n_steps):
            position = offset + local
            if 0 <= position < len(cases):
                indexed[(experiment_id, local)] = cases[position]
        offset += n_steps
    return indexed


def _cases_from_document(document: Optional[SessionDocument]) -> dict[tuple[int, int], Any]:
    if document is None:
        return {}
    from vfmstudio.core.jobs import (
        analysis_load_indices,
        apply_axes_alignment,
        apply_reference_subtraction,
        project_for_identification,
    )

    included = document.experiments_in_identification()
    if not included:
        return {}
    project = project_for_identification(document)
    built = bridge.project_to_cases(project, verbose=False)
    if isinstance(built, tuple):
        cases, experiment_ids = built
    else:
        cases, experiment_ids = built, [
            experiment.experiment_id
            for experiment in included
            for _ in experiment.included_step_indices()
        ]
    needs_reference = any(
        experiment.reference_step is not None for experiment in included
    )
    needs_alignment = any(
        experiment.fit_data_to_system_axes for experiment in included
    )
    if needs_alignment:
        apply_axes_alignment(document, cases, experiment_ids)
    if needs_reference:
        step_indices: list[int] = []
        for experiment in included:
            step_indices.extend(analysis_load_indices(experiment))
        if len(step_indices) == len(cases):
            cases, experiment_ids, _ = apply_reference_subtraction(
                document, cases, experiment_ids, step_indices
            )
    indexed: dict[tuple[int, int], Any] = {}
    counts: dict[int, int] = {}
    for case, experiment_id in zip(cases, experiment_ids):
        key = int(experiment_id)
        local = counts.get(key, 0)
        indexed[(key, local)] = case
        counts[key] = local + 1
    return indexed
