"""One model-build run: measure, assemble, check, and write the facts the agent may read."""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from vfmstudio.agents.model_build.apply import _declines_replace
from vfmstudio.agents.model_build.data_record import stage_record
from vfmstudio.agents.model_build.gate import refusal_reasons
from vfmstudio.agents.model_build.measure import ModelRefused, measure_experiment
from vfmstudio.agents.model_build.model import (
    ExperimentBuild,
    attach_precedents,
    build_experiment,
    check_virtual_model,
)
from vfmstudio.agents.model_build.library import (
    RANKING_RULE,
    REFERENCE_NAME,
    as_training_library,
    load_library,
    write_reference,
)
from vfmstudio.agents.model_build.report import render_report, unresolved_questions
from vfmstudio.core.document import SessionDocument

FACTS_NAME = "facts.json"
REPORT_NAME = "report.md"


def scratch_root() -> Path:
    return Path(tempfile.gettempdir()) / "vfmstudio_model_agent"


@dataclass
class ModelBuildRun:
    workdir: Path
    builds: list[ExperimentBuild] = field(default_factory=list)
    facts_path: Optional[Path] = None
    report_path: Optional[Path] = None
    report_markdown: str = ""
    solver_errors: list[str] = field(default_factory=list)
    solver_warnings: list[str] = field(default_factory=list)
    # Data agent figures copied into ``workdir``: experiment name -> figure key -> relative path.
    figures: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def questions(self) -> list[dict]:
        return [question for built in self.builds for question in built.questions]


def _solver_preflight(document: SessionDocument, builds: list[ExperimentBuild]) -> tuple[list[str], list[str]]:
    """Pre-flight the proposal on a copy. The open project is not modified."""
    from vfmstudio.agents.model_build.model import blocking_unanswered, write_build
    from vfmstudio.core.jobs import validate_document

    copy = document.model_copy(deep=True)
    ready = False
    for built in builds:
        experiment = copy.experiment_by_id(built.experiment_id)
        if experiment is None or built.skipped or not built.mesh:
            if experiment is not None:
                experiment.include_in_identification = False
            continue
        if blocking_unanswered(built, {}):
            experiment.include_in_identification = False
            continue
        if not (experiment.prescribed or experiment.linking or experiment.external):
            experiment.include_in_identification = False
            continue
        if _declines_replace(built, {}):
            experiment.include_in_identification = False
            continue
        write_build(experiment, built, {})
        ready = True
    if not ready or not copy.experiments_in_identification():
        return [], []
    try:
        report = validate_document(copy)
    except Exception as exc:
        return [f"solver pre-flight failed: {exc}"], []
    rotated = {
        built.name
        for built in builds
        if built.fit_data_to_system_axes and abs(built.alignment_deg) > 1.0e-6
    }
    warnings = []
    for item in report.warnings:
        if any(item.startswith(f"{name}:") and "applied load is declared" in item for name in rotated):
            continue
        warnings.append(item)
    for built in builds:
        if built.name in rotated and not built.skipped:
            warnings.append(
                f"{built.name}: the cloud is rotated {built.alignment_deg:.2f} deg onto the lab axes "
                "before identification, so the load direction was judged in that frame."
            )
    return list(report.errors), warnings


def build_model(
    document: SessionDocument,
    *,
    workdir: Optional[Path] = None,
    library: Optional[dict] = None,
) -> ModelBuildRun:
    """Assemble a virtual model for every experiment that has imported data.

    ``library`` is the distilled BC training library; None reads the one in the
    agents folder. A harvest payload is distilled first. Does not modify ``document``.
    """
    reasons = refusal_reasons(document)
    if reasons:
        raise ModelRefused(reasons)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    workdir = Path(workdir) if workdir else scratch_root() / stamp
    workdir.mkdir(parents=True, exist_ok=True)
    library = as_training_library(load_library() if library is None else library)
    material = document.material.material_type
    builds: list[ExperimentBuild] = []
    staged: set[str] = set()
    figures: dict[str, dict[str, str]] = {}
    for experiment in document.experiments:
        if not experiment.steps:
            name = experiment.name or f"Experiment {experiment.experiment_id}"
            builds.append(
                ExperimentBuild(
                    experiment_id=int(experiment.experiment_id),
                    name=name,
                    skipped=True,
                    skip_reason="no load steps imported",
                )
            )
            continue
        measured = measure_experiment(experiment)
        built = build_experiment(experiment, measured, material_type=material)
        built.data_agent = stage_record(experiment, workdir, staged)
        if built.data_agent.get("figures"):
            figures[built.name] = dict(built.data_agent["figures"])
        if not built.skipped:
            check_virtual_model(document, experiment, built)
        attach_precedents(document, experiment, measured, built, library)
        if not built.data_agent.get("available"):
            built.warnings.append(f"no DIC data agent record: {built.data_agent.get('reason')}")
        builds.append(built)
    run = ModelBuildRun(workdir=workdir, builds=builds, figures=figures)
    run.solver_errors, run.solver_warnings = _solver_preflight(document, builds)
    experiments = [built.to_dict() for built in builds]
    cards = {built.name: built.bc_reference for built in builds if built.bc_reference}
    has_library = bool(library.get("schemes"))
    if cards:
        write_reference(workdir / REFERENCE_NAME, RANKING_RULE, cards)
    facts = {
        "generated_by": (
            "VFM Studio model build agent (deterministic measurements). Mesh bounds and the "
            "linear window come from the imported steps. The mesh and the boundary conditions "
            "recorded here are the ones already on the experiment; the assessment chooses among them "
            "and the training-library cards retrieved for each experiment."
        ),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gate": "load steps already imported in the project",
        "material_type": material,
        "analysis_type": document.material.analysis_type,
        "virtual_mesh": (
            "bounds taken from the DIC cloud; the element counts recorded here are the ones "
            "already on the experiment. The assessment compares that mesh with one on the "
            "physical boundary and with the typical meshes on the retrieved scheme cards."
        ),
        "data_agent_records": (
            "per experiment under data_agent: the DIC data agent's facts, report, and figures for "
            "the same load steps, copied into data_agent/<experiment>/"
        ),
        "bc_training_library": {
            "available": has_library,
            "file": REFERENCE_NAME if cards else None,
            "schemes_in_library": len(library.get("schemes") or []),
            "cards_retrieved": sum(len(items) for items in cards.values()),
            "ranking_rule": RANKING_RULE,
            "meaning": (
                "A compact reference of boundary-condition schemes, stored in the agents folder. "
                "Only the cards retrieved for this build are written here. The agent does not "
                "read .vfm projects. Reward is updated when a scheme is adopted or declined."
            ),
            "problem": library.get("problem"),
        },
        "experiments": experiments,
        "unresolved": unresolved_questions(experiments),
        "solver_errors": run.solver_errors,
        "solver_warnings": run.solver_warnings,
    }
    run.report_markdown = render_report(facts)
    run.facts_path = workdir / FACTS_NAME
    run.report_path = workdir / REPORT_NAME
    run.facts_path.write_text(json.dumps(facts, indent=1, ensure_ascii=False), encoding="utf-8")
    run.report_path.write_text(run.report_markdown, encoding="utf-8")
    return run


def agent_request(run: ModelBuildRun) -> tuple[str, str]:
    """(prompt, short line shown in the chat) for a finished build."""
    names = ", ".join(built.name for built in run.builds if not built.skipped)
    n_open = len(run.questions)
    if n_open:
        left = f"{n_open} input(s) are still open; ask only those, using the wording in facts.json."
    else:
        left = "unresolved is empty; ask the user for nothing."
    with_record = [built.name for built in run.builds if built.data_agent.get("available")]
    without = [built.name for built in run.builds if not built.skipped and not built.data_agent.get("available")]
    records = (
        f"The DIC data agent's record is in data_agent/ for: {', '.join(with_record)}. "
        if with_record
        else ""
    )
    if without:
        records += f"There is no data agent record for: {', '.join(without)}. "
    reference = (
        f"The BC training cards retrieved for this build are in {REFERENCE_NAME}; "
        "the same cards are under bc_reference in facts.json. Do not open .vfm files. "
        if (run.workdir / REFERENCE_NAME).is_file()
        else "There is no BC training library. "
    )
    prompt = (
        f"The application measured the imported load steps for: {names}. "
        f"Read {REPORT_NAME} and {FACTS_NAME}. {records}{reference}"
        "Assess the boundary conditions from the machine data and the DIC, compare them with the "
        "training cards, and adopt the virtual mesh that follows the physical boundary, the loads, and "
        f"the kinematics. Use only those files, in the required structure. {left} "
        "Do not invent a number that is not in them."
    )
    noun = "experiment" if sum(not built.skipped for built in run.builds) == 1 else "experiments"
    display = f"Build the virtual model for {sum(not built.skipped for built in run.builds)} {noun}: {names}"
    return prompt, display
