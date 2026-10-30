"""Boundary-condition training library the model build agent reads.

The language model never opens a .vfm file. An offline harvest may scan saved
projects once; this module distills them into scheme cards in the agents folder.
At build time the application retrieves a few cards, ranks them by how well the
new test matches and by a reward updated from later feedback (adopt / decline /
identification), and writes only that slice into the build folder.

That is a contextual bandit: the state is the test signature, the action is a
BC scheme, the reward is whether the scheme was identified or the user adopted
it. The library is the value table. The agent sees the top cards, not the table.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from vfmstudio.agents.model_build.precedents import (
    BC_KINDS,
    _channels_compatible,
    _cosine,
    _sig,
)

FORMAT_NAME = "vfmstudio-bc-library"
FORMAT_VERSION = 1
LIBRARY_ENV = "VFMSTUDIO_PRECEDENTS"
LIBRARY_PATH = Path(__file__).with_name("bc_library.json")
FEEDBACK_PATH = Path(__file__).with_name("bc_feedback.json")
REFERENCE_NAME = "bc_reference.json"
MAX_CARDS = 3
MAX_USED_ON = 8

RANKING_RULE = (
    "Scheme cards are ordered by: load channels compatible first; then a listed material "
    "type matches; then the curvature-direction match, highest first; then the aspect-ratio "
    "factor, lowest first; then reward (identifications + 2·adoptions − 2·declines) plus a "
    "small bonus for schemes with little feedback. Reward is the running score of the "
    "training loop, not a measurement of this test."
)


def library_path() -> Optional[Path]:
    raw = os.environ.get(LIBRARY_ENV)
    if raw is None:
        return LIBRARY_PATH
    if raw.strip().lower() in {"", "off", "none"}:
        return None
    return Path(raw)


def feedback_path(path: Optional[Path] = None) -> Optional[Path]:
    if path is not None:
        return Path(path)
    library = library_path()
    if library is None:
        return None
    if library == LIBRARY_PATH:
        return FEEDBACK_PATH
    return library.with_name("bc_feedback.json")


def _empty() -> dict[str, Any]:
    return {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "schemes": [],
        "source": None,
        "ranking_rule": RANKING_RULE,
    }


def _strip_nodes(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned = []
    for item in entries:
        row = {key: value for key, value in item.items() if key != "node_positions_on_mesh"}
        cleaned.append(row)
    return cleaned


def _lesson(bc: dict[str, Any]) -> str:
    verbs = {"prescribed": "Hold", "linking": "Link", "external": "Apply"}
    parts = []
    for kind in BC_KINDS:
        groups: dict[str, list[str]] = {}
        for entry in bc.get(kind) or []:
            where = str(entry.get("edge") or "selected nodes")
            dof = str(entry.get("dof") or "")
            if dof:
                groups.setdefault(where, []).append(dof)
        for where, dofs in groups.items():
            parts.append(f"{verbs[kind]} {', '.join(dofs)} on {where}")
    return ". ".join(parts) + ("." if parts else "No boundary conditions.")


def _median(values: list[float]) -> Optional[float]:
    ordered = sorted(values)
    if not ordered:
        return None
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _use_when(members: list[dict[str, Any]]) -> dict[str, Any]:
    moving = Counter()
    channels = Counter()
    labelled = 0
    directions: list[list[float]] = []
    aspects: list[float] = []
    materials: list[str] = []
    for item in members:
        signature = item.get("signature") or {}
        moving[tuple(signature.get("moving_channels") or [])] += 1
        if signature.get("channels_labelled") and signature.get("load_channels"):
            channels[tuple(signature["load_channels"])] += 1
            labelled += 1
        direction = signature.get("curvature_direction") or {}
        vector = [float(direction.get(key) or 0.0) for key in ("kxx", "kyy", "kxy")]
        if any(vector):
            directions.append(vector)
        aspect = signature.get("aspect_ratio_y_to_x")
        if aspect:
            aspects.append(float(aspect))
        material = item.get("material_type")
        if material and material not in materials:
            materials.append(str(material))
    load_channels = list(channels.most_common(1)[0][0]) if channels else []
    moving_channels = list(moving.most_common(1)[0][0]) if moving else []
    curvature = None
    if directions:
        mean = [sum(row[i] for row in directions) / len(directions) for i in range(3)]
        norm = math.sqrt(sum(value * value for value in mean))
        if norm:
            curvature = {
                "kxx": _sig(mean[0] / norm, 3),
                "kyy": _sig(mean[1] / norm, 3),
                "kxy": _sig(mean[2] / norm, 3),
            }
    return {
        "moving_channels": moving_channels,
        "load_channels": load_channels,
        "channels_labelled": bool(labelled),
        "curvature_direction": curvature,
        "aspect_ratio_y_to_x": _sig(_median(aspects), 3),
        "material_types": materials,
    }


def _typical_mesh(members: list[dict[str, Any]]) -> Optional[dict[str, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for item in members:
        mesh = item.get("mesh") or {}
        if not mesh.get("structured"):
            continue
        weight = int(item.get("n_projects_with_results") or 1)
        counts[(int(mesh["number_of_elements_x"]), int(mesh["number_of_elements_y"]))] += weight
    if not counts:
        return None
    (x_count, y_count), _weight = counts.most_common(1)[0]
    return {"x": x_count, "y": y_count}


def _reward(scores: dict[str, Any]) -> float:
    identifications = int(scores.get("identifications") or 0)
    adoptions = int(scores.get("adoptions") or 0)
    declines = int(scores.get("declines") or 0)
    return float(identifications + 2 * adoptions - 2 * declines)


def _explore(scores: dict[str, Any]) -> float:
    n = (
        int(scores.get("identifications") or 0)
        + int(scores.get("adoptions") or 0)
        + int(scores.get("declines") or 0)
    )
    return 1.0 / math.sqrt(1.0 + n)


def distill(harvest: dict[str, Any]) -> dict[str, Any]:
    """Turn a project harvest into the agent-facing training library.

    Project paths, step files, and per-experiment dumps are dropped. What remains
    is one card per boundary-condition scheme.
    """
    by_id = {item["id"]: item for item in harvest.get("precedents") or []}
    schemes = []
    for raw in harvest.get("schemes") or []:
        members = [by_id[item] for item in raw.get("precedents") or [] if item in by_id]
        if not members and any(raw.get(kind) for kind in BC_KINDS):
            members = []
        bc = {
            kind: _strip_nodes((raw.get("boundary_conditions") or {}).get(kind) or [])
            for kind in BC_KINDS
        }
        if not any(bc[kind] for kind in BC_KINDS):
            continue
        used = list(raw.get("experiments") or [])
        if not used:
            used = sorted({item.get("name") or "" for item in members if item.get("name")})
        scores = {
            "identifications": int(raw.get("n_projects_with_results") or 0),
            "adoptions": 0,
            "declines": 0,
        }
        scores["reward"] = _reward(scores)
        schemes.append(
            {
                "id": raw["id"],
                "lesson": _lesson(bc),
                "use_when": _use_when(members),
                "boundary_conditions": bc,
                "placement": raw.get("placement") or "edges",
                "typical_mesh": _typical_mesh(members),
                "used_on": used[:12],
                "scores": scores,
            }
        )
    schemes.sort(key=lambda item: (-float(item["scores"]["reward"]), item["id"]))
    return {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ranking_rule": RANKING_RULE,
        "schemes": schemes,
        "source": None,
    }


def _read_feedback(path: Optional[Path]) -> dict[str, dict[str, int]]:
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    rows = payload.get("schemes") if isinstance(payload, dict) else None
    if not isinstance(rows, dict):
        return {}
    out: dict[str, dict[str, int]] = {}
    for key, value in rows.items():
        if not isinstance(value, dict):
            continue
        out[str(key)] = {
            "adoptions": int(value.get("adoptions") or 0),
            "declines": int(value.get("declines") or 0),
        }
    return out


def apply_feedback(library: dict[str, Any], feedback: Optional[dict[str, dict[str, int]]] = None) -> dict[str, Any]:
    """Copy running adopt/decline counts onto each scheme and refresh reward."""
    if feedback is None:
        feedback = _read_feedback(feedback_path())
    for scheme in library.get("schemes") or []:
        scores = dict(scheme.get("scores") or {})
        extra = feedback.get(scheme["id"])
        if extra:
            scores["adoptions"] = int(extra.get("adoptions") or 0)
            scores["declines"] = int(extra.get("declines") or 0)
        scores["reward"] = _reward(scores)
        scheme["scores"] = scores
    return library


def as_training_library(data: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Accept either a harvest or a distilled library; always return the latter."""
    if not data:
        return _empty()
    if data.get("format") == FORMAT_NAME:
        library = json.loads(json.dumps(data))
    elif data.get("schemes") or data.get("precedents"):
        library = distill(data)
    else:
        return {**_empty(), "problem": data.get("problem"), "source": data.get("source")}
    apply_feedback(library)
    if data.get("source"):
        library["source"] = data.get("source")
    return library


def load_library(path: Optional[Path] = None) -> dict[str, Any]:
    """The distilled training library. Never opens a project file."""
    path = Path(path) if path is not None else library_path()
    empty = _empty()
    if path is None or not path.is_file():
        return empty
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {**empty, "source": str(path), "problem": "the training library could not be read"}
    payload["source"] = str(path)
    library = as_training_library(payload)
    library["source"] = str(path)
    return library


def record_feedback(scheme_id: str, event: str, path: Optional[Path] = None) -> None:
    """Update the running reward table. No-op when the library is switched off."""
    if event not in {"adoption", "decline"} or not scheme_id:
        return
    target = feedback_path(path)
    if target is None:
        return
    rows = _read_feedback(target)
    current = rows.setdefault(scheme_id, {"adoptions": 0, "declines": 0})
    if event == "adoption":
        current["adoptions"] += 1
    else:
        current["declines"] += 1
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "format": "vfmstudio-bc-feedback",
                "version": 1,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "schemes": rows,
            },
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def compare_scheme(current: dict[str, Any], scheme: dict[str, Any], material_type: str) -> dict[str, Any]:
    other = scheme.get("use_when") or {}
    cosine = _cosine(current.get("curvature_direction"), other.get("curvature_direction"))
    aspect = None
    this_aspect, other_aspect = current.get("aspect_ratio_y_to_x"), other.get("aspect_ratio_y_to_x")
    if this_aspect and other_aspect:
        aspect = max(this_aspect / other_aspect, other_aspect / this_aspect)
    materials = [str(item).lower() for item in other.get("material_types") or []]
    return {
        "load_channels_compatible": _channels_compatible(current, other),
        "moving_channels": {
            "this_test": current.get("moving_channels"),
            "scheme": other.get("moving_channels"),
        },
        "same_material_type": (not materials) or str(material_type).lower() in materials,
        "curvature_direction_match": _sig(cosine, 3),
        "aspect_ratio_factor": _sig(aspect, 3),
    }


def _slim_setup(bc: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    setup: dict[str, list[dict[str, Any]]] = {kind: [] for kind in BC_KINDS}
    for kind in BC_KINDS:
        for entry in bc.get(kind) or []:
            row = {"dof": entry.get("dof"), "edge": entry.get("edge") or "selected nodes"}
            if entry.get("forceIndex") is not None:
                row["forceIndex"] = entry["forceIndex"]
            setup[kind].append(row)
    return setup


def scheme_card(scheme: dict[str, Any], comparison: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """The slice of a scheme the language model is allowed to read."""
    scores = dict(scheme.get("scores") or {})
    scores["reward"] = _reward(scores)
    card = {
        "id": scheme["id"],
        "lesson": scheme.get("lesson"),
        "use_when": scheme.get("use_when") or {},
        "setup": _slim_setup(scheme.get("boundary_conditions") or {}),
        "typical_mesh": scheme.get("typical_mesh"),
        "placement": scheme.get("placement"),
        "used_on": list(scheme.get("used_on") or [])[:MAX_USED_ON],
        "scores": scores,
    }
    if comparison is not None:
        card["comparison"] = comparison
    return card


def rank_schemes(
    library: dict[str, Any],
    current: dict[str, Any],
    material_type: str,
    limit: int = MAX_CARDS,
) -> list[dict[str, Any]]:
    """The closest scheme cards, already slim."""
    rows = []
    for scheme in library.get("schemes") or []:
        comparison = compare_scheme(current, scheme, material_type)
        cosine = comparison["curvature_direction_match"]
        aspect = comparison["aspect_ratio_factor"]
        scores = scheme.get("scores") or {}
        key = (
            comparison["load_channels_compatible"] is False,
            not comparison["same_material_type"],
            -(round(cosine, 2) if cosine is not None else -1.0),
            round(math.log(aspect), 1) if aspect else 99.0,
            -(_reward(scores) + 0.5 * _explore(scores)),
            scheme.get("id") or "",
        )
        rows.append((key, scheme, comparison))
    rows.sort(key=lambda row: row[0])
    return [scheme_card(scheme, comparison) for _key, scheme, comparison in rows[:limit]]


def write_reference(path: Path, ranking_rule: str, cards_by_experiment: dict[str, list[dict[str, Any]]]) -> Path:
    """Write the retrieved cards only. This is the file the agent reads."""
    payload = {
        "meaning": (
            "Boundary-condition training cards retrieved for this build. "
            "The full library stays in the agents folder. Do not open .vfm files."
        ),
        "ranking_rule": ranking_rule,
        "experiments": cards_by_experiment,
    }
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    return path
