"""Write a finished proposal onto the session document."""

from __future__ import annotations

from typing import Any, Sequence

from vfmstudio.agents.model_build.library import record_feedback
from vfmstudio.agents.model_build.model import (
    ExperimentBuild,
    adopted_candidate,
    blocking_unanswered,
    write_build,
)
from vfmstudio.core.document import SessionDocument


def _declines_replace(built: ExperimentBuild, answers: dict[str, Any]) -> bool:
    for question in built.questions:
        if question["kind"] != "replace_existing":
            continue
        value = answers[question["id"]] if question["id"] in answers else question["value"]
        return not bool(value)
    return False


def apply_to_document(
    document: SessionDocument,
    builds: Sequence[ExperimentBuild],
    answers: dict[str, Any],
) -> list[str]:
    """Write every proposal the answers allow. Returns one note per experiment."""
    notes: list[str] = []
    for built in builds:
        experiment = document.experiment_by_id(built.experiment_id)
        if built.skipped or experiment is None:
            notes.append(f"{built.name}: not written ({built.skip_reason or 'missing experiment'}).")
            continue
        if _declines_replace(built, answers):
            notes.append(
            f"{built.name}: left unchanged. It has a hand-built mesh; tick Replace to overwrite it."
            )
            continue
        missing = blocking_unanswered(built, answers)
        if missing:
            notes.append(f"{built.name}: not written. Still needed: {missing[0]}")
            continue
        write_build(experiment, built, answers)
        _record_choice(experiment, built, answers)
        window = ""
        if built.analysis_start is not None and built.analysis_end is not None:
            window = f" Load steps {built.analysis_start}–{built.analysis_end}"
            if built.reference_step is not None:
                window += f", reference {built.reference_step}"
            window += "."
        nx = experiment.mesh.n_elements_x
        ny = experiment.mesh.n_elements_y
        adopted = adopted_candidate(built, answers)
        if adopted is not None:
            source = f"took the boundary conditions of {adopted['id']} (as on {', '.join(adopted['used_by'][:4])})"
        else:
            source = "left the boundary conditions unchanged"
        notes.append(
            f"{built.name}: wrote a {nx}×{ny} virtual mesh on the DIC bounds and {source} "
            f"({len(experiment.prescribed)} prescribed, {len(experiment.linking)} linking, "
            f"{len(experiment.external)} external).{window}"
        )
    return notes


def _record_choice(experiment, built: ExperimentBuild, answers: dict[str, Any]) -> None:
    """Reward the training library: adoption if a card was applied, decline if none was."""
    adopted = adopted_candidate(built, answers)
    if adopted is not None:
        record_feedback(str(adopted["id"]), "adoption")
        return
    usable = [
        item for item in built.bc_candidates
        if item.get("transferable") and not (item.get("check") or {}).get("problems")
    ]
    if not usable:
        return
    if experiment.prescribed or experiment.linking or experiment.external:
        return
    record_feedback(str(usable[0]["id"]), "decline")
