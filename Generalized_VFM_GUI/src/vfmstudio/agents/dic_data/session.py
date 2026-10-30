"""One data-agent run: analyse every test, write facts and figures to a scratch folder.

The scratch folder is the only place anything is written, and it is the only
folder the Claude agent can read. Raw inputs are never written to.
"""

from __future__ import annotations

import json
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from vfmstudio.agents.dic_data.analysis import (
    DataRefused,
    TestAnalysis,
    analyse_group,
    noise_model_facts,
)
from vfmstudio.agents.dic_data.figures import render_all
from vfmstudio.agents.dic_data.intake import TestGroup, frame_match
from vfmstudio.agents.dic_data.raw_format import describe_expected_format
from vfmstudio.agents.dic_data.report import CHECK_THRESHOLDS, enrich, render_report
from vfmstudio.io.csv_import import write_step_mat

FACTS_NAME = "facts.json"
REPORT_NAME = "report.md"
FIGURES_DIR = "figures"
# Written next to the exported .mat files of each test; the model build agent reads it from there.
RECORD_DIR = "data_agent"


def scratch_root() -> Path:
    return Path(tempfile.gettempdir()) / "vfmstudio_data_agent"


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_") or "test"


def _is_within(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
        return True
    except ValueError:
        return False


@dataclass
class DataAgentRun:
    groups: list[TestGroup]
    workdir: Path
    analyses: list[TestAnalysis] = field(default_factory=list)
    figures: dict[str, dict[str, str]] = field(default_factory=dict)
    facts_path: Optional[Path] = None
    report_path: Optional[Path] = None
    report_markdown: str = ""

    @property
    def input_folders(self) -> list[Path]:
        return sorted({info.path.parent for group in self.groups for info in group.files}, key=str)

    @property
    def n_files(self) -> int:
        return sum(len(group.files) for group in self.groups)


def validate_groups(groups: Sequence[TestGroup]) -> list[str]:
    """Reasons to refuse, before anything is read beyond the headers and the submitted machine file."""
    reasons: list[str] = []
    if not any(group.files for group in groups):
        reasons.append("no files were given")
    for group in groups:
        if not group.files:
            continue
        for info in group.files:
            if not info.ok:
                reasons.append(f"{info.path.name}: {'; '.join(info.problems)}")
        if any(not info.ok for info in group.files):
            continue
        for problem in frame_match(group).problems:
            reasons.append(f"{group.name}: {problem}")
    return reasons


def run_analysis(
    groups: Sequence[TestGroup],
    progress: Callable[[str], None] = lambda _text: None,
    workdir: Optional[Path] = None,
) -> DataAgentRun:
    groups = [group for group in groups if group.files]
    reasons = validate_groups(groups)
    if reasons:
        raise DataRefused(reasons)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    workdir = Path(workdir) if workdir else scratch_root() / stamp
    run = DataAgentRun(groups=list(groups), workdir=workdir)
    for folder in run.input_folders:
        if _is_within(workdir, folder):
            raise DataRefused([f"the scratch folder {workdir} lies inside the raw-data folder {folder}"])
    workdir.mkdir(parents=True, exist_ok=True)
    figures_dir = workdir / FIGURES_DIR
    used: set[str] = set()
    for group in groups:
        analysis = analyse_group(group, progress)
        run.analyses.append(analysis)
        slug = _slug(group.name)
        while slug in used:
            slug += "_"
        used.add(slug)
        progress(f"{group.name}: drawing figures")
        paths = render_all(analysis, figures_dir, slug)
        run.figures[group.name] = {
            key: str(Path(value).relative_to(workdir)).replace("\\", "/") for key, value in paths.items()
        }
    facts = {
        "generated_by": "VFM Studio DIC data agent (deterministic analysis; numbers are measured, not estimated by the language model)",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "format_check": {
            "expected": describe_expected_format(),
            "files_checked": run.n_files,
            "all_files_conform": True,
            "cross_file_checks": "same facet step in every file of a test; every file shares at least half its facets with the first",
        },
        "conventions": {
            "increments": "Δ quantities are relative to step 1 of each test, in the order the user confirmed",
            "axes": "figures and zone names use +X to the right and +Y up",
        },
        "noise_model": noise_model_facts(),
        "reporting_thresholds": dict(CHECK_THRESHOLDS),
        "tests": [
            enrich({**analysis.facts, "figures": run.figures.get(analysis.name, {})})
            for analysis in run.analyses
        ],
    }
    for analysis, test in zip(run.analyses, facts["tests"]):
        analysis.facts["summary"] = test["summary"]
        analysis.facts["automatic_checks"] = test["automatic_checks"]
    run.facts_path = workdir / FACTS_NAME
    run.facts_path.write_text(json.dumps(facts, indent=1, ensure_ascii=False), encoding="utf-8")
    run.report_markdown = render_report(facts)
    run.report_path = workdir / REPORT_NAME
    run.report_path.write_text(run.report_markdown, encoding="utf-8")
    return run


def agent_request(run: DataAgentRun) -> tuple[str, str]:
    """(prompt for the agent, short line shown in the chat) for a finished run."""
    lines = []
    for analysis in run.analyses:
        facts = analysis.facts
        lines.append(
            f"{analysis.name} ({facts['specimen']['n_steps']} steps, "
            f"{facts['read_quality']['facets_union']} facets)"
        )
    tests = "; ".join(lines)
    noun = "test" if len(run.analyses) == 1 else "tests"
    prompt = (
        f"The application converted and measured {len(run.analyses)} {noun}: {tests}. "
        f"The user already sees {REPORT_NAME} (all measured numbers, tables and automatic checks). "
        f"Read {REPORT_NAME}, {FACTS_NAME} and the figures listed for each test, then write only "
        "the interpretation sections in the required structure. Do not repeat the tables. "
        "Remember: present facts, do not accept or reject any test."
    )
    display = f"Interpret {len(run.analyses)} {noun}: {tests}"
    return prompt, display


def export_solver_mats(run: DataAgentRun, folder: Path) -> list[Path]:
    """Write one Layout A .mat per step into new subfolders of ``folder``. Never overwrites.

    Each test folder also receives the data agent's record of that test (facts, report,
    figures) under :data:`RECORD_DIR`. Only the .mat paths are returned.
    """
    folder = Path(folder)
    for source in run.input_folders:
        if _is_within(folder, source) or _is_within(source, folder):
            raise DataRefused(
                [f"export folder {folder} overlaps the raw-data folder {source}; choose another"]
            )
    planned: list[tuple[Path, dict]] = []
    targets: list[tuple[Path, TestAnalysis]] = []
    for analysis in run.analyses:
        target = folder / _slug(analysis.name)
        targets.append((target, analysis))
        for info, payload in zip(analysis.files, analysis.payloads):
            planned.append((target / f"{info.path.stem}.mat", payload))
    existing = [path for path, _ in planned if path.exists()]
    existing += [target / RECORD_DIR for target, _ in targets if (target / RECORD_DIR).exists()]
    if existing:
        raise DataRefused([f"{len(existing)} file(s) already exist, e.g. {existing[0]}; nothing written"])
    written = []
    for path, payload in planned:
        arrays = {key: value for key, value in payload.items() if key not in {"F", "F_labels"}}
        force = payload.get("F")
        labels = list(payload["F_labels"]) if "F_labels" in payload else None
        written.append(write_step_mat(path, arrays, force=force, force_labels=labels))
    for target, analysis in targets:
        write_data_record(run, analysis, target / RECORD_DIR)
    return written


def write_data_record(run: DataAgentRun, analysis: TestAnalysis, folder: Path) -> Path:
    """Copy one test's facts, report, and figures into ``folder``. Returns the facts path."""
    import shutil

    folder = Path(folder)
    shared = json.loads(Path(run.facts_path).read_text(encoding="utf-8")) if run.facts_path else {}
    tests = [test for test in shared.get("tests") or [] if test.get("test_name") == analysis.name]
    if not tests:
        raise ValueError(f"the data agent run has no test named {analysis.name!r}")
    test = dict(tests[0])
    figures_dir = folder / FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)
    copied = {}
    for key, relative in (test.get("figures") or {}).items():
        source = Path(run.workdir) / relative
        if not source.is_file():
            continue
        shutil.copy2(source, figures_dir / source.name)
        copied[key] = f"{FIGURES_DIR}/{source.name}"
    test["figures"] = copied
    record = {key: value for key, value in shared.items() if key != "tests"}
    record["tests"] = [test]
    record["exported_with"] = [info.path.stem for info in analysis.files]
    facts_path = folder / FACTS_NAME
    facts_path.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    (folder / REPORT_NAME).write_text(render_report(record), encoding="utf-8")
    return facts_path


def _unique_name(name: str, taken: set[str]) -> str:
    candidate, n = name, 2
    while candidate in taken:
        candidate, n = f"{name} ({n})", n + 1
    return candidate


def add_exported_experiments(
    document,
    run: DataAgentRun,
    written: Sequence[Path],
    *,
    analysis_type: str = "both",
) -> list:
    """Append one experiment per exported test to ``document``; return the new experiments.

    ``written`` is what :func:`export_solver_mats` returned: every step of every test, in the
    order the user confirmed. Steps keep that order; mesh bounds and sample area are detected
    from the files exactly as the Import MAT dialog does.
    """
    from vfmstudio.core.document import ExperimentModel, LoadStepModel, MeshModel
    from vfmstudio.core.fields import sync_mesh_bounds_from_file

    written = [Path(path) for path in written]
    expected = sum(len(analysis.files) for analysis in run.analyses)
    if len(written) != expected:
        raise ValueError(f"export wrote {len(written)} file(s) but the run has {expected} step(s)")
    taken = {item.name for item in document.experiments}
    created = []
    offset = 0
    for analysis in run.analyses:
        paths = written[offset: offset + len(analysis.files)]
        offset += len(analysis.files)
        rows = analysis.facts["deformation"]["steps"]
        steps = []
        for path, row in zip(paths, rows):
            where = [f"step {row['step']} of {len(rows)}"]
            if row.get("campaign_step") is not None:
                where.append(f"campaign step {row['campaign_step']}")
            if row.get("dic_frame") is not None:
                where.append(f"DIC frame {row['dic_frame']}")
            steps.append(LoadStepModel(
                data_file=path,
                label=path.stem,
                description=f"DIC data agent export of {row['file']} ({', '.join(where)})",
            ))
        name = _unique_name(analysis.name, taken)
        taken.add(name)
        experiment = ExperimentModel(
            experiment_id=document.next_experiment_id(),
            name=name,
            mesh=MeshModel(),
            steps=steps,
        )
        document.experiments.append(experiment)
        for step in steps:
            sync_mesh_bounds_from_file(
                experiment, step.data_file, source_path=document.source_path, analysis_type=analysis_type
            )
        created.append(experiment)
    return created


def describe_payload(payload: dict[str, np.ndarray]) -> list[tuple[str, str, str, str, str]]:
    """(name, shape, dtype, range, meaning) rows for the data-structure view."""
    from vfmstudio.agents.dic_data.analysis import FIELD_DOC

    rows = []
    order = ["nodes", "X", "Y", "Z", "U", "V", "W", "SK1grid", "SK2grid", "SK3grid",
             "SE1grid", "SE2grid", "SE3grid", "F", "F_labels"]
    for name in [key for key in order if key in payload] + sorted(set(payload) - set(order)):
        value = np.asarray(payload[name])
        unit, meaning = FIELD_DOC.get(name, ("", ""))
        if value.dtype == object:
            span = ", ".join(str(item) for item in value.ravel())
        else:
            finite = value[np.isfinite(value)] if value.size else value
            span = f"{finite.min():.4g} … {finite.max():.4g}" if finite.size else "—"
        rows.append((name, "×".join(str(n) for n in value.shape), str(value.dtype), span,
                     f"{meaning} [{unit}]" if unit else meaning))
    return rows
