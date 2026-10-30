"""Turn measurements into a virtual-mesh proposal and the open questions.

Mesh bounds come from the measured cloud. Element counts and boundary
conditions stay as they already are on the experiment unless the user adopts
one of the campaign precedents offered for it; this module never makes up a
mesh density or a boundary-condition scheme of its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from vfmstudio.agents.model_build.measure import (
    ExperimentMeasurements,
    StepMeasurement,
)
from vfmstudio.core.document import (
    BoundsModel,
    ExperimentModel,
    MeshModel,
    SessionDocument,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _step_role(number: int, start: Optional[int], end: Optional[int], reference: Optional[int]) -> str:
    if start is None or end is None or not (start <= number <= end):
        return "outside"
    if reference is not None and number == reference:
        return "reference"
    return "identification"


def _analysis_bounds(measured: ExperimentMeasurements) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """1-based analysis start, end, and reference.

    The reference is inside the analysis window so the solver loads it, and
    outside the identification set.
    """
    start, end, reference = measured.linear_start, measured.linear_end, measured.reference_step
    if start is None or end is None:
        return None, None, None
    if reference is not None:
        start = min(start, reference)
    return start, end, reference


def _included(start: Optional[int], end: Optional[int], reference: Optional[int]) -> list[int]:
    if start is None or end is None or start > end:
        return []
    return [number for number in range(start, end + 1) if number != reference]


def _question(
    experiment_id: int,
    name: str,
    kind: str,
    prompt: str,
    *,
    blocking: bool,
    value: Any,
    options: Optional[list[dict[str, Any]]] = None,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    suffix: str = "",
) -> dict[str, Any]:
    return {
        "id": f"{experiment_id}.{kind}",
        "experiment_id": int(experiment_id),
        "experiment": name,
        "kind": kind,
        "prompt": prompt,
        "blocking": blocking,
        "value": _jsonable(value),
        "options": options,
        "minimum": minimum,
        "maximum": maximum,
        "suffix": suffix,
    }


def _has_hand_mesh(experiment: ExperimentModel) -> bool:
    mesh = experiment.mesh
    return bool(mesh.x_nodes or mesh.y_nodes or mesh.free_nodes or mesh.keep_cells)


def _has_boundary_conditions(experiment: ExperimentModel) -> bool:
    return bool(experiment.prescribed or experiment.linking or experiment.external)


def _step_table(
    steps: list[StepMeasurement],
    load_index: Optional[int],
    start: Optional[int],
    end: Optional[int],
    reference: Optional[int],
) -> list[dict[str, Any]]:
    rows = []
    for step in steps:
        load = None
        if step.force is not None and load_index is not None and load_index < len(step.force):
            load = step.force[load_index]
        elif step.force is not None and len(step.force) == 1:
            load = step.force[0]
        means = step.curvature_mean or [None, None, None]
        rows.append(
            {
                "step": step.number,
                "label": step.label,
                "file": step.path,
                "readable": step.readable,
                "problem": step.problem,
                "load": load,
                "forces": list(step.force) if step.force else [],
                "kxx": means[0],
                "kyy": means[1],
                "kxy": means[2],
                "role": _step_role(step.number, start, end, reference) if step.readable else "unreadable",
            }
        )
    return rows


def _bounds(steps: list[StepMeasurement]) -> Optional[BoundsModel]:
    readable = [step for step in steps if step.readable and step.x_min is not None]
    if not readable:
        return None
    return BoundsModel(
        x_min=min(step.x_min for step in readable),
        x_max=max(step.x_max for step in readable),
        y_min=min(step.y_min for step in readable),
        y_max=max(step.y_max for step in readable),
    )


@dataclass
class ExperimentBuild:
    experiment_id: int
    name: str
    skipped: bool = False
    skip_reason: str = ""
    measurements: dict[str, Any] = field(default_factory=dict)
    mesh: dict[str, Any] = field(default_factory=dict)
    prescribed: list[dict[str, Any]] = field(default_factory=list)
    linking: list[dict[str, Any]] = field(default_factory=list)
    external: list[dict[str, Any]] = field(default_factory=list)
    analysis_start: Optional[int] = None
    analysis_end: Optional[int] = None
    reference_step: Optional[int] = None
    included_steps: list[int] = field(default_factory=list)
    sample_area: Optional[float] = None
    fit_data_to_system_axes: bool = False
    alignment_deg: float = 0.0
    applied_load_source: str = "mat"
    include_in_identification: bool = True
    material_orientation_deg: float = 0.0
    questions: list[dict[str, Any]] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    mesh_check: dict[str, Any] = field(default_factory=dict)
    data_agent: dict[str, Any] = field(default_factory=dict)
    signature: dict[str, Any] = field(default_factory=dict)
    bc_reference: list[dict[str, Any]] = field(default_factory=list)
    bc_candidates: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(
            {
                "experiment_id": self.experiment_id,
                "name": self.name,
                "skipped": self.skipped,
                "skip_reason": self.skip_reason,
                "measurements": self.measurements,
                "mesh": self.mesh,
                "prescribed": self.prescribed,
                "linking": self.linking,
                "external": self.external,
                "analysis_start": self.analysis_start,
                "analysis_end": self.analysis_end,
                "reference_step": self.reference_step,
                "included_steps": self.included_steps,
                "sample_area": self.sample_area,
                "fit_data_to_system_axes": self.fit_data_to_system_axes,
                "alignment_deg": self.alignment_deg,
                "applied_load_source": self.applied_load_source,
                "include_in_identification": self.include_in_identification,
                "material_orientation_deg": self.material_orientation_deg,
                "questions": self.questions,
                "checks": self.checks,
                "warnings": self.warnings,
                "mesh_check": self.mesh_check,
                "data_agent": self.data_agent,
                "signature": self.signature,
                "bc_reference": self.bc_reference,
                "bc_candidates": [
                    {
                        key: item[key]
                        for key in (
                            "id",
                            "transferable",
                            "load_wiring",
                            "check",
                            "used_by",
                            "element_counts",
                            "placement",
                            "lesson",
                        )
                        if key in item
                    }
                    for item in self.bc_candidates
                ],
            }
        )


def build_experiment(
    experiment: ExperimentModel,
    measured: ExperimentMeasurements,
    *,
    material_type: str,
) -> ExperimentBuild:
    built = ExperimentBuild(
        experiment_id=measured.experiment_id,
        name=measured.name,
        material_orientation_deg=float(experiment.material_orientation_deg),
        include_in_identification=bool(experiment.include_in_identification),
        fit_data_to_system_axes=measured.fit_data_to_system_axes,
        alignment_deg=measured.alignment_deg,
        sample_area=measured.sample_area,
        applied_load_source=str(experiment.applied_load_source or "mat"),
    )
    built.prescribed = [entry.to_solver_dict() for entry in experiment.prescribed]
    built.linking = [entry.to_solver_dict() for entry in experiment.linking]
    built.external = [entry.to_solver_dict() for entry in experiment.external]
    if not any(step.readable for step in measured.steps):
        built.skipped = True
        built.skip_reason = "no load step could be read"
        built.checks.append(built.skip_reason)
        return built

    start, end, reference = _analysis_bounds(measured)
    built.analysis_start = start
    built.analysis_end = end
    built.reference_step = reference if measured.linear_resolved else None
    if not measured.linear_resolved:
        built.reference_step = None
    built.included_steps = _included(built.analysis_start, built.analysis_end, built.reference_step)

    bounds = _bounds(measured.steps)
    if bounds is None or bounds.x_max <= bounds.x_min or bounds.y_max <= bounds.y_min:
        built.checks.append("the DIC cloud has no area, so the virtual mesh has no bounds")
    else:
        built.mesh = {
            "number_of_elements_x": int(experiment.mesh.n_elements_x),
            "number_of_elements_y": int(experiment.mesh.n_elements_y),
            "bounds": {
                "x_min": bounds.x_min,
                "x_max": bounds.x_max,
                "y_min": bounds.y_min,
                "y_max": bounds.y_max,
            },
        }

    built.measurements = {
        "load_index": measured.load_index,
        "load_name": measured.load_name,
        "load_resolved": measured.load_resolved,
        "load_reason": measured.load_reason,
        "probe_load_index": measured.probe_load_index,
        "linear_r2": measured.linear_r2,
        "curvature_r2": dict(measured.curvature_r2),
        "full_series_r2": measured.full_series_r2,
        "linear_resolved": measured.linear_resolved,
        "reference_reason": measured.reference_reason,
        "sample_area_source": measured.sample_area_source,
        "alignment_reason": (
            f"the DIC cloud is {measured.alignment_deg:.2f} deg off the lab axes, so it is rotated onto them"
            if measured.fit_data_to_system_axes and abs(measured.alignment_deg) > 1.0e-9
            else "the DIC cloud already lies on the lab axes"
        ),
        "steps": _step_table(
            measured.steps,
            measured.probe_load_index,
            built.analysis_start,
            built.analysis_end,
            built.reference_step,
        ),
    }
    built.questions = _questions(experiment, measured, built, material_type=material_type)
    if _force_width(measured) == 0:
        built.checks.append(measured.load_reason)
    return built


def _force_width(measured: ExperimentMeasurements) -> int:
    for step in measured.steps:
        if step.force:
            return len(step.force)
    return 0


def _step_options(measured: ExperimentMeasurements) -> list[dict[str, Any]]:
    return [{"value": step.number, "label": f"step {step.number}  {step.label}"} for step in measured.steps]


def _questions(
    experiment: ExperimentModel,
    measured: ExperimentMeasurements,
    built: ExperimentBuild,
    *,
    material_type: str,
) -> list[dict[str, Any]]:
    questions = []
    name = built.name
    if str(material_type).strip().lower() not in {"iso", "isotropic"}:
        questions.append(
            _question(
                built.experiment_id,
                name,
                "material_orientation",
                "Fiber angle of this coupon, in degrees. The measurement files do not contain it; "
                f"the value already on the experiment is {experiment.material_orientation_deg:g}.",
                blocking=False,
                value=float(experiment.material_orientation_deg),
                minimum=-180.0,
                maximum=180.0,
                suffix=" deg",
            )
        )
    if not _has_boundary_conditions(experiment):
        questions.append(
            _question(
                built.experiment_id,
                name,
                "boundary_conditions",
                "Boundary conditions are not stored on this experiment. Assess them from the "
                "machine data and the DIC. Keep a condition only when both agree. If they do "
                "not determine the edges and the degrees of freedom, say what is missing.",
                blocking=False,
                value=None,
            )
        )
    if not measured.linear_resolved:
        questions.append(
            _question(
                built.experiment_id,
                name,
                "linear_window",
                measured.reference_reason or "Which load steps are the linear candidates for identification?",
                blocking=False,
                value={"start": built.analysis_start or 1, "end": built.analysis_end or len(measured.steps)},
                options=_step_options(measured),
            )
        )
    elif measured.reference_step is None:
        questions.append(
            _question(
                built.experiment_id,
                name,
                "reference",
                measured.reference_reason,
                blocking=False,
                value=None,
                options=[{"value": None, "label": "no reference (use the absolute fields)"}] + _step_options(measured),
            )
        )
    if measured.sample_area is None:
        questions.append(
            _question(
                built.experiment_id,
                name,
                "sample_area",
                "Specimen area in mm². The DIC cloud is not a lattice, so the area was not measured.",
                blocking=True,
                value=None,
                minimum=0.0,
                suffix=" mm²",
            )
        )
    if _has_hand_mesh(experiment):
        questions.append(
            _question(
                built.experiment_id,
                name,
                "replace_existing",
                "This experiment has a hand-built mesh. Replace it with a structured mesh on the DIC bounds, "
                "keeping the element counts already on the experiment?",
                blocking=False,
                value=False,
            )
        )
    if not experiment.include_in_identification and measured.linear_resolved:
        questions.append(
            _question(
                built.experiment_id,
                name,
                "include",
                "This experiment is excluded from the identification. Include the linear load steps?",
                blocking=False,
                value=True,
            )
        )
    return questions


def _mesh_model(payload: dict[str, Any]) -> MeshModel:
    bounds = payload.get("bounds") or {}
    return MeshModel(
        n_elements_x=int(payload["number_of_elements_x"]),
        n_elements_y=int(payload["number_of_elements_y"]),
        bounds=BoundsModel(
            x_min=float(bounds.get("x_min", 0.0)),
            x_max=float(bounds.get("x_max", 1.0)),
            y_min=float(bounds.get("y_min", 0.0)),
            y_max=float(bounds.get("y_max", 1.0)),
        ),
    )


def check_virtual_model(document: SessionDocument, experiment: ExperimentModel, built: ExperimentBuild) -> None:
    """Mesh, constraints, external work, and the project pre-flight. Appends to ``built``."""
    if built.skipped or not built.mesh:
        return
    scratch = experiment.model_copy(deep=True)
    write_build(scratch, built, {})
    analysis = document.material.analysis_type
    try:
        from vfmstudio.core.mesh_convert import build_experiment_mesh

        mesh_pkg = build_experiment_mesh(document, scratch)
    except Exception as exc:
        built.checks.append(f"the virtual mesh did not build: {exc}")
        return
    n_nodes = int(getattr(mesh_pkg, "n_nodes", 0) or 0)
    n_elements = int(getattr(mesh_pkg, "n_elements", 0) or 0)
    built.mesh_check = {"n_nodes": n_nodes, "n_elements": n_elements}
    if n_nodes < 4 or n_elements < 1:
        built.checks.append(f"the virtual mesh has {n_nodes} nodes and {n_elements} elements")
        return
    if not (scratch.prescribed or scratch.linking or scratch.external):
        built.warnings.append(
            "boundary conditions are not on this experiment, so the constraints were not checked"
        )
        return
    problems, info = _constraint_check(
        mesh_pkg,
        [entry.to_solver_dict() for entry in scratch.prescribed],
        [entry.to_solver_dict() for entry in scratch.linking],
        [entry.to_solver_dict() for entry in scratch.external],
        str(analysis),
    )
    built.checks.extend(problems)
    built.mesh_check.update(info)

    if not _loads_present(built):
        built.checks.append(
            "no applied load was found in F for every identification step"
        )


def _constraint_check(
    mesh_pkg: Any,
    prescribed: list[dict[str, Any]],
    linking: list[dict[str, Any]],
    external: list[dict[str, Any]],
    analysis: str,
) -> tuple[list[str], dict[str, Any]]:
    """(problems, rank and nullity) of one boundary-condition set on one mesh."""
    problems: list[str] = []
    info: dict[str, Any] = {}
    try:
        from vfmstudio.core import bridge

        BoundaryConditions = bridge.vfm_data_io().BoundaryConditions
        boundary = BoundaryConditions(
            prescribed=prescribed,
            linking=linking,
            external=external,
            analysis_type=analysis,
        )
        mesh_module = bridge.vfm_mesh()
        if external:
            work, _meta = mesh_module.build_external_work_matrix(mesh_pkg, boundary)
            if int(work.shape[1]) != len(external) or work.nnz == 0:
                problems.append("the external-work matrix does not cover the loaded edges")
        constraints = mesh_module.build_kinematic_constraints(
            mesh_pkg, boundary, np.zeros(0, dtype=np.int64)
        )
        _basis, meta = mesh_module.build_admissible_basis(
            constraints.C,
            int(mesh_pkg.n_dofs),
            svd_tolerance_factor=100.0,
            rank_tolerance_floor=1.0e-12,
        )
        info["admissible_nullity"] = int(meta["nullity"])
        info["constraint_rank"] = int(meta["rank"])
        if int(meta["nullity"]) <= 0:
            problems.append("the constraints eliminate every virtual degree of freedom")
    except ValueError as exc:
        problems.append(str(exc))
    except Exception as exc:
        problems.append(f"the boundary-condition check failed: {exc}")
    return problems, info


def attach_precedents(
    document: SessionDocument,
    experiment: ExperimentModel,
    measured: ExperimentMeasurements,
    built: ExperimentBuild,
    library: dict[str, Any],
) -> None:
    """Retrieve BC training cards for ``built`` and offer the sets that carry over."""
    from vfmstudio.agents.model_build import precedents as lib
    from vfmstudio.agents.model_build.library import as_training_library, rank_schemes

    built.signature = lib.signature(measured)
    trained = as_training_library(library)
    if built.skipped or not trained.get("schemes"):
        return
    built.bc_reference = rank_schemes(trained, built.signature, document.material.material_type)
    schemes = {scheme["id"]: scheme for scheme in trained.get("schemes") or []}
    for card in built.bc_reference:
        scheme = schemes.get(card["id"])
        if scheme is None:
            continue
        moved = lib.transfer(scheme, (), measured, built.signature)
        mesh = scheme.get("typical_mesh")
        candidate = {
            "id": scheme["id"],
            "used_by": list(scheme.get("used_on") or []),
            "lesson": scheme.get("lesson"),
            "placement": scheme.get("placement"),
            "transferable": moved["transferable"],
            "load_wiring": moved["wiring"],
            "prescribed": moved["prescribed"],
            "linking": moved["linking"],
            "external": moved["external"],
            "element_counts": (
                {"x": int(mesh["x"]), "y": int(mesh["y"])} if mesh else None
            ),
            "check": {},
        }
        if candidate["transferable"] and built.mesh:
            candidate["check"] = _check_candidate(document, experiment, built, candidate)
        card["transferable"] = candidate["transferable"]
        card["load_wiring"] = candidate["load_wiring"]
        if candidate["check"]:
            card["check"] = {
                key: candidate["check"][key]
                for key in ("admissible_nullity", "constraint_rank", "problems")
                if key in candidate["check"]
            }
        built.bc_candidates.append(candidate)
    _offer_candidates(experiment, built)


def _check_candidate(
    document: SessionDocument,
    experiment: ExperimentModel,
    built: ExperimentBuild,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    scratch = experiment.model_copy(deep=True)
    write_build(scratch, built, {})
    counts = candidate.get("element_counts")
    if counts:
        scratch.mesh.n_elements_x = int(counts["x"])
        scratch.mesh.n_elements_y = int(counts["y"])
    try:
        from vfmstudio.core.mesh_convert import build_experiment_mesh

        mesh_pkg = build_experiment_mesh(document, scratch)
    except Exception as exc:
        return {"problems": [f"the virtual mesh did not build: {exc}"]}
    problems, info = _constraint_check(
        mesh_pkg,
        candidate["prescribed"],
        candidate["linking"],
        candidate["external"],
        str(document.material.analysis_type),
    )
    return {
        "n_nodes": int(getattr(mesh_pkg, "n_nodes", 0) or 0),
        "n_elements": int(getattr(mesh_pkg, "n_elements", 0) or 0),
        **info,
        "problems": problems,
    }


def _offer_candidates(experiment: ExperimentModel, built: ExperimentBuild) -> None:
    usable = [
        item for item in built.bc_candidates
        if item["transferable"] and not (item.get("check") or {}).get("problems")
    ]
    if not usable:
        return
    keep = (
        "keep the boundary conditions already on the experiment"
        if _has_boundary_conditions(experiment)
        else "leave the boundary conditions unset"
    )
    options = [{"value": None, "label": keep}]
    for item in usable:
        mesh = item.get("element_counts")
        size = f", {mesh['x']}×{mesh['y']} elements" if mesh else ""
        options.append({
            "value": item["id"],
            "label": f"{item['id']}: as on {', '.join(item['used_by'][:4])}{size}",
        })
    prompt = (
        "Boundary conditions for this experiment. The training library offers these sets "
        "for this mesh. Adopt one only if the machine data and the DIC of this test agree with it."
    )
    for question in built.questions:
        if question["kind"] == "boundary_conditions":
            question["prompt"] = prompt
            question["options"] = options
            return
    built.questions.append(
        _question(
            built.experiment_id,
            built.name,
            "boundary_conditions",
            prompt,
            blocking=False,
            value=None,
            options=options,
        )
    )


def _loads_present(built: ExperimentBuild) -> bool:
    if not built.external:
        return True
    needed = set(built.included_steps)
    if built.reference_step is not None:
        needed.add(int(built.reference_step))
    if not needed:
        needed = set(range(1, len(built.measurements.get("steps") or []) + 1))
    for row in built.measurements.get("steps") or []:
        if int(row["step"]) not in needed:
            continue
        if row.get("load") is None:
            return False
    return True


def write_build(experiment: ExperimentModel, built: ExperimentBuild, answers: dict[str, Any]) -> None:
    """Copy one proposal onto an experiment. Blocking answers must already be filled."""
    if built.mesh:
        experiment.mesh = _mesh_model(built.mesh)
        experiment.mesh.clear_free_and_manual()
    candidate = adopted_candidate(built, answers)
    if candidate is not None:
        from vfmstudio.agents.model_build.precedents import boundary_entries

        entries = boundary_entries(candidate)
        experiment.prescribed = entries["prescribed"]
        experiment.linking = entries["linking"]
        experiment.external = entries["external"]
        counts = candidate.get("element_counts")
        if counts and built.mesh:
            experiment.mesh.n_elements_x = int(counts["x"])
            experiment.mesh.n_elements_y = int(counts["y"])
    experiment.fit_data_to_system_axes = bool(built.fit_data_to_system_axes)
    area = _answered(built, answers, "sample_area", built.sample_area)
    if area not in (None, 0, 0.0):
        experiment.sample_area_total = float(area)
    orientation = _answered(built, answers, "material_orientation", built.material_orientation_deg)
    if orientation is not None:
        experiment.material_orientation_deg = float(orientation)
    include = _answered(built, answers, "include", built.include_in_identification)
    experiment.include_in_identification = bool(include)

    start, end = built.analysis_start, built.analysis_end
    window = _answered(built, answers, "linear_window", None)
    if isinstance(window, dict):
        start = int(window.get("start") or start or 1)
        end = int(window.get("end") or end or start)
    reference = built.reference_step
    if _has_question(built, "reference"):
        reference = _answered(built, answers, "reference", None)
    if start is not None and end is not None:
        if start > end:
            start, end = end, start
        n_steps = len(experiment.steps)
        start = max(1, min(int(start), n_steps))
        end = max(1, min(int(end), n_steps))
        experiment.analysis_start = start
        experiment.analysis_end = end
        if reference is not None:
            reference = int(reference)
            if reference < 1 or reference > n_steps:
                reference = None
        experiment.reference_step = None if reference is None else int(reference)
        built.analysis_start = start
        built.analysis_end = end
        built.reference_step = experiment.reference_step
        built.included_steps = _included(start, end, experiment.reference_step)


def adopted_candidate(built: ExperimentBuild, answers: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The precedent set the user chose for ``built``, or None."""
    chosen = _answered(built, answers, "boundary_conditions", None)
    if not chosen:
        return None
    for candidate in built.bc_candidates:
        if candidate["id"] == chosen and candidate["transferable"]:
            return candidate
    return None


def _has_question(built: ExperimentBuild, kind: str) -> bool:
    return any(question["kind"] == kind for question in built.questions)


def _answered(built: ExperimentBuild, answers: dict[str, Any], kind: str, default: Any) -> Any:
    for question in built.questions:
        if question["kind"] != kind:
            continue
        if question["id"] in answers:
            return answers[question["id"]]
        if question["blocking"]:
            return None
        return question["value"]
    return default


def blocking_unanswered(built: ExperimentBuild, answers: dict[str, Any]) -> list[str]:
    missing = []
    for question in built.questions:
        if not question["blocking"]:
            continue
        value = answers.get(question["id"], None)
        if value is None or (question["kind"] == "sample_area" and not value):
            missing.append(question["prompt"])
    return missing
