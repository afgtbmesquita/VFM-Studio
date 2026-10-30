"""Find the DIC data agent's record of an experiment and copy it into the build folder.

The data agent writes the record next to the .mat files it exports. The model
build reads it from there, so it follows the experiment after the project is
saved and reopened. The record is copied, never edited.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Optional

from vfmstudio.agents.dic_data.session import FACTS_NAME, FIGURES_DIR, RECORD_DIR, REPORT_NAME
from vfmstudio.core.document import ExperimentModel

STAGE_DIR = "data_agent"

_SPECIMEN_KEYS = (
    "measured_region_x_mm",
    "measured_region_y_mm",
    "measured_span_mm",
    "measured_area_mm2",
    "internal_holes",
    "width_profile_along_y_mm",
    "height_profile_along_x_mm",
    "machine_channels",
)
_DEFORMATION_KEYS = (
    "dominant_machine_moment",
    "dominant_machine_rotation",
    "interior_mean_dk_at_final_step_inv_mm",
    "secondary_to_dominant_curvature_ratios",
    "curvature_linearity_vs_load",
    "w_quadratic_fit_final_step",
    "curvature_frame_check",
    "bands_along_y_at_final_step",
)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_") or "experiment"


def find_record(experiment: ExperimentModel) -> tuple[Optional[Path], str]:
    """(record folder, reason). The folder is None when no record covers every step."""
    stems = [Path(step.data_file).stem for step in experiment.steps]
    folders = []
    for step in experiment.steps:
        folder = Path(step.data_file).parent / RECORD_DIR
        if folder not in folders:
            folders.append(folder)
    for folder in folders:
        facts = folder / FACTS_NAME
        if not facts.is_file():
            continue
        try:
            record = json.loads(facts.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None, f"the data agent record in {folder} could not be read"
        exported = set(record.get("exported_with") or [])
        missing = [stem for stem in stems if stem not in exported]
        if missing:
            return None, (
                f"the data agent record in {folder} does not cover {len(missing)} of the "
                f"{len(stems)} load steps (e.g. {missing[0]})"
            )
        return folder, ""
    return None, (
        "no data agent record sits next to the load-step files; the steps were not exported "
        "by the DIC data agent, so the machine data comes only from F in those files"
    )


def stage_record(experiment: ExperimentModel, workdir: Path, taken: set[str]) -> dict[str, Any]:
    """Copy the record into ``workdir``; return what facts.json says about it."""
    folder, reason = find_record(experiment)
    if folder is None:
        return {"available": False, "reason": reason}
    slug = _slug(experiment.name or f"experiment_{experiment.experiment_id}")
    while slug in taken:
        slug += "_"
    taken.add(slug)
    relative = f"{STAGE_DIR}/{slug}"
    target = Path(workdir) / STAGE_DIR / slug
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(folder / FACTS_NAME, target / FACTS_NAME)
    if (folder / REPORT_NAME).is_file():
        shutil.copy2(folder / REPORT_NAME, target / REPORT_NAME)
    if (folder / FIGURES_DIR).is_dir():
        shutil.copytree(folder / FIGURES_DIR, target / FIGURES_DIR, dirs_exist_ok=True)
    record = json.loads((target / FACTS_NAME).read_text(encoding="utf-8"))
    test = (record.get("tests") or [{}])[0]
    specimen = test.get("specimen") or {}
    deformation = test.get("deformation") or {}
    figures = {
        key: f"{relative}/{path}" for key, path in (test.get("figures") or {}).items()
        if (target / path).is_file()
    }
    return {
        "available": True,
        "source": str(folder),
        "test_name": test.get("test_name"),
        "facts": f"{relative}/{FACTS_NAME}",
        "report": f"{relative}/{REPORT_NAME}" if (target / REPORT_NAME).is_file() else None,
        "figures": figures,
        "specimen": {key: specimen[key] for key in _SPECIMEN_KEYS if key in specimen},
        "machine_load_vector": (test.get("solver_data") or {}).get("force_vector"),
        "machine_per_step": [
            {"step": row.get("step"), "machine": row.get("machine")} for row in deformation.get("steps") or []
        ],
        "deformation": {key: deformation[key] for key in _DEFORMATION_KEYS if key in deformation},
        "automatic_checks": test.get("automatic_checks"),
    }
