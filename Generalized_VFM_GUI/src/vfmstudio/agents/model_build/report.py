"""Scripted report of a model build. Every number is a measurement or a count."""

from __future__ import annotations

import math
from typing import Any


def _num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "—"
    return f"{number:.{digits}g}"


def _role(role: str, settled: bool) -> str:
    if role == "identification" and not settled:
        return "suggested"
    if role == "reference" and not settled:
        return "suggested reference"
    return role


def render_report(facts: dict[str, Any]) -> str:
    lines = [
        "# Virtual model",
        "",
        (
            "Built from the load steps already imported in the project. "
            "The mesh and the boundary conditions listed here are the ones already on the experiment, "
            "together with the bounds and the linear window measured from those steps. "
            "The BC training library is a compact reference of schemes already identified, "
            "ranked by how closely this test matches and by the running reward of those schemes. "
            "The cards listed here are the only slice retrieved for this build."
        ),
        "",
    ]
    for experiment in facts.get("experiments") or []:
        lines.extend(_experiment(experiment))
    errors = list(facts.get("solver_errors") or [])
    warnings = list(facts.get("solver_warnings") or [])
    lines.append("## Project checks")
    lines.append("")
    if not errors and not warnings:
        lines.append("The solver pre-flight reported nothing.")
    else:
        for item in errors:
            lines.append(f"- Error: {item}")
        for item in warnings:
            lines.append(f"- Warning: {item}")
    lines.append("")
    questions = list(facts.get("unresolved") or [])
    lines.append("## Left for you")
    lines.append("")
    if not questions:
        lines.append(
            "Nothing further is unresolved in the measurements."
        )
    else:
        for question in questions:
            mark = "required" if question.get("blocking") else "confirm"
            lines.append(f"- ({mark}) {question.get('experiment')}: {question.get('prompt')}")
    lines.append("")
    return "\n".join(lines)


def _experiment(experiment: dict[str, Any]) -> list[str]:
    name = experiment.get("name") or f"Experiment {experiment.get('experiment_id')}"
    lines = [f"## {name}", ""]
    if experiment.get("skipped"):
        lines.append(experiment.get("skip_reason") or "Skipped.")
        lines.append("")
        return lines
    measured = experiment.get("measurements") or {}
    settled = bool(measured.get("linear_resolved"))
    lines.append(measured.get("load_reason") or "")
    scores = measured.get("curvature_r2") or {}
    if scores:
        parts = ", ".join(f"{name} R² {_num(scores.get(name), 5)}" for name in ("kxx", "kyy", "kxy"))
        lines.append(f"Each curvature mean against that load: {parts}.")
    lines.append(
        "The linear window uses the magnitude of the three curvature means. "
        "No curvature component is chosen as the bending axis."
    )
    lines.append("")
    lines.append(measured.get("alignment_reason") or "")
    area = experiment.get("sample_area")
    source = measured.get("sample_area_source") or "not measured"
    lines.append(f"Sample area: {_num(area, 6)} mm² ({source}).")
    lines.append("")
    lines.append("### Load steps")
    lines.append("")
    lines.append("| Step | Load | κxx | κyy | κxy | Role |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | --- |")
    for row in measured.get("steps") or []:
        if not row.get("readable"):
            lines.append(f"| {row.get('step')} | — | — | — | — | {row.get('problem') or 'unreadable'} |")
            continue
        lines.append(
            f"| {row.get('step')} | {_num(row.get('load'), 6)} | {_num(row.get('kxx'))} | "
            f"{_num(row.get('kyy'))} | {_num(row.get('kxy'))} | {_role(str(row.get('role')), settled)} |"
        )
    lines.append("")
    r2 = measured.get("linear_r2")
    full = measured.get("full_series_r2")
    if settled:
        lines.append(
            f"Linear run R² = {_num(r2, 5)} (whole series {_num(full, 5)}). "
            f"Identification steps: {', '.join(str(n) for n in experiment.get('included_steps') or []) or '—'}. "
            f"Reference: {experiment.get('reference_step') or 'none'}. {measured.get('reference_reason') or ''}"
        )
    else:
        lines.append(
            f"No load-step window was settled (best R² {_num(r2, 5)}, whole series {_num(full, 5)}). "
            f"{measured.get('reference_reason') or ''}"
        )
    lines.append("")
    lines.append("### Virtual mesh")
    lines.append("")
    mesh = experiment.get("mesh") or {}
    bounds = mesh.get("bounds") or {}
    if not mesh:
        lines.append("No mesh.")
    else:
        lines.append(
            f"{mesh.get('number_of_elements_x')}×{mesh.get('number_of_elements_y')} elements, "
            f"x {_num(bounds.get('x_min'), 6)}…{_num(bounds.get('x_max'), 6)} mm, "
            f"y {_num(bounds.get('y_min'), 6)}…{_num(bounds.get('y_max'), 6)} mm."
        )
    check = experiment.get("mesh_check") or {}
    if check:
        extra = ""
        if "admissible_nullity" in check:
            extra = (
                f" Constraint rank {check.get('constraint_rank')}, "
                f"admissible virtual DOFs {check.get('admissible_nullity')}."
            )
        lines.append(f"Built mesh: {check.get('n_nodes')} nodes, {check.get('n_elements')} elements.{extra}")
    lines.append("")
    lines.append("### Boundary conditions")
    lines.append("")
    prescribed = list(experiment.get("prescribed") or [])
    linking = list(experiment.get("linking") or [])
    external = list(experiment.get("external") or [])
    if not prescribed and not linking and not external:
        lines.append("None on the experiment. The measurements do not determine the edges or the degrees of freedom.")
    else:
        lines.append("Already on the experiment. This build does not change them.")
        for entry in prescribed:
            lines.append(f"- Prescribed {entry.get('dof')} on {entry.get('edge')} ({entry.get('label')})")
        for entry in linking:
            lines.append(f"- Linking {entry.get('dof')} ({entry.get('label')})")
        for entry in external:
            channel = entry.get("forceIndex")
            wired = f", F[{channel}]" if channel is not None else ""
            lines.append(
                f"- External {entry.get('dof')} on {entry.get('edge')} "
                f"({entry.get('loadType')}, sort {entry.get('sortDirection')}{wired})"
            )
    lines.append("")
    lines.extend(_data_agent(experiment.get("data_agent") or {}))
    lines.extend(_bc_reference(experiment))
    for item in experiment.get("checks") or []:
        lines.append(f"Check: {item}")
    for item in experiment.get("warnings") or []:
        lines.append(f"Warning: {item}")
    if experiment.get("checks") or experiment.get("warnings"):
        lines.append("")
    return lines


def _data_agent(record: dict[str, Any]) -> list[str]:
    lines = ["### DIC data agent record", ""]
    if not record.get("available"):
        lines.append(f"Not available: {record.get('reason') or 'no record'}.")
        lines.append("")
        return lines
    specimen = record.get("specimen") or {}
    span = specimen.get("measured_span_mm") or {}
    holes = specimen.get("internal_holes") or {}
    lines.append(
        f"Test {record.get('test_name')}, copied to {record.get('facts')}. "
        f"Measured span {_num(span.get('x'))} × {_num(span.get('y'))} mm, "
        f"measured area {_num(specimen.get('measured_area_mm2'))} mm², "
        f"{holes.get('count', 0)} internal hole(s)."
    )
    channels = specimen.get("machine_channels") or []
    if channels:
        lines.append(f"Machine channels: {', '.join(str(item) for item in channels)}.")
    vector = record.get("machine_load_vector")
    if vector:
        lines.append(f"F in the exported steps: {', '.join(str(item) for item in vector)}.")
    lines.append("")
    return lines


def _bc_reference(experiment: dict[str, Any]) -> list[str]:
    cards = list(experiment.get("bc_reference") or [])
    lines = ["### Boundary-condition training cards", ""]
    if not cards:
        lines.append("No training-library card was retrieved for this experiment.")
        lines.append("")
        return lines
    for card in cards:
        comparison = card.get("comparison") or {}
        scores = card.get("scores") or {}
        mesh = card.get("typical_mesh") or {}
        size = f"{mesh.get('x')}×{mesh.get('y')} elements" if mesh else "no typical mesh"
        compatible = comparison.get("load_channels_compatible")
        channels = "—" if compatible is None else ("compatible" if compatible else "different")
        lines.append(
            f"- {card.get('id')} ({', '.join(card.get('used_on') or []) or 'unlabelled'}): "
            f"{card.get('lesson')} Typical mesh {size}. Load channels {channels}, "
            f"direction match {_num(comparison.get('curvature_direction_match'), 3)}, "
            f"aspect factor {_num(comparison.get('aspect_ratio_factor'), 3)}. "
            f"Reward {scores.get('reward', 0)} "
            f"({scores.get('identifications', 0)} identifications, "
            f"{scores.get('adoptions', 0)} adoptions, {scores.get('declines', 0)} declines)."
        )
    lines.append("")
    for candidate in experiment.get("bc_candidates") or []:
        check = candidate.get("check") or {}
        if not candidate.get("transferable"):
            state = f"does not carry over: {candidate.get('load_wiring')}"
        elif check.get("problems"):
            state = "check: " + "; ".join(check["problems"])
        else:
            state = (
                f"{candidate.get('load_wiring')}; constraint rank {check.get('constraint_rank')}, "
                f"admissible virtual DOFs {check.get('admissible_nullity')}"
            )
        lines.append(f"- Set {candidate.get('id')} (as on {', '.join(candidate.get('used_by') or [])}): {state}.")
    if experiment.get("bc_candidates"):
        lines.append("")
    return lines


def unresolved_questions(experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    questions = []
    for experiment in experiments:
        questions.extend(experiment.get("questions") or [])
    return questions
