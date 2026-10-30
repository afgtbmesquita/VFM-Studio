"""The model build agent runs only after load steps are imported."""

from __future__ import annotations

from pathlib import Path

from vfmstudio.core.document import ExperimentModel, SessionDocument


def _name(experiment: ExperimentModel) -> str:
    return experiment.name or f"Experiment {experiment.experiment_id}"


def experiments_with_data(document: SessionDocument) -> list[ExperimentModel]:
    """Experiments that have at least one measurement file on disk."""
    ready: list[ExperimentModel] = []
    for experiment in document.experiments:
        if any(Path(step.data_file).is_file() for step in experiment.steps):
            ready.append(experiment)
    return ready


def refusal_reasons(document: SessionDocument) -> list[str]:
    """Why the agent must refuse. Empty means imported data is available."""
    if not document.experiments:
        return ["The project has no experiments."]
    listed = [experiment for experiment in document.experiments if experiment.steps]
    if not listed:
        return ["No load steps have been imported. Add load steps or import MAT files first."]
    ready = experiments_with_data(document)
    if ready:
        return []
    missing = ", ".join(_name(experiment) for experiment in listed)
    return [
        (
            f"Load steps are listed ({missing}) but none of the measurement files were found on disk. "
            "Re-import them, or point the steps at the MAT files."
        )
    ]
