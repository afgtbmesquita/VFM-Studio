"""Offline harvest of saved projects into the BC training library.

This module is the only place that opens .vfm files, and only when someone
runs the harvest command. The model build agent never imports it for reading
projects. ``library.py`` distills the harvest into scheme cards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from vfmstudio.agents.model_build.measure import ExperimentMeasurements, measure_experiment
from vfmstudio.core.document import BoundaryEntry, ExperimentModel, SessionDocument

FORMAT_NAME = "vfmstudio-model-precedents"
FORMAT_VERSION = 1
LIBRARY_ENV = "VFMSTUDIO_PRECEDENTS"
LIBRARY_PATH = Path(__file__).with_name("bc_library.json")
MAX_MATCHES = 3
BC_KINDS = ("prescribed", "linking", "external")
_SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules"}

RANKING_RULE = (
    "Precedents are ordered by: load channels compatible first (same moving channel names when "
    "both are labelled, otherwise the same number of moving channels); then the same material "
    "type; then the curvature direction match |cos| between the curvature increments, to two "
    "decimals, highest first; then the aspect-ratio factor (larger span ratio over smaller, 1 = "
    "same shape), to one decimal, lowest first; then the number of saved projects with "
    "identification results that used the precedent, highest first."
)


# -- reading saved projects ----------------------------------------------------


def _relocate(path: Path, project: Path) -> Path:
    """Rebase a missing step file onto the folder tree the project now lives in."""
    if path.is_file():
        return path
    parts = path.parts
    for ancestor in Path(project).resolve().parents:
        if ancestor.name not in parts:
            continue
        index = len(parts) - 1 - parts[::-1].index(ancestor.name)
        candidate = ancestor.joinpath(*parts[index + 1 :])
        if candidate.is_file():
            return candidate
    return path


def _project_files(roots: Iterable[Path]) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        root = Path(root)
        if root.is_file():
            found.append(root)
            continue
        for folder, dirs, files in os.walk(root):
            dirs[:] = [
                name for name in dirs
                if name not in _SKIP_DIRS and not name.startswith("vfmstudio_job_")
            ]
            for name in files:
                lowered = name.lower()
                if lowered.endswith(".vfm") or lowered.endswith(".vfmproj.json"):
                    found.append(Path(folder) / name)
    return sorted(set(found), key=lambda item: str(item).lower())


def _read_project(path: Path) -> tuple[SessionDocument, bool]:
    """(document, identification results stored in the file)."""
    if path.suffix.lower() == ".vfm":
        import h5py

        with h5py.File(path, "r") as handle:
            raw = handle["document_json"][()]
            text = raw.decode("utf-8") if isinstance(raw, bytes) else bytes(raw).decode("utf-8")
            solved = "identification" in handle and "final_parameters" in handle["identification"]
        document = SessionDocument.model_validate(json.loads(text))
        document.source_path = path.resolve()
        from vfmstudio.core.step_series import relocate_document_data_files

        relocate_document_data_files(document)
    else:
        from vfmstudio.io.vfmproj_json import load_vfmproj

        document = load_vfmproj(path)
        solved = False
    for experiment in document.experiments:
        for step in experiment.steps:
            step.data_file = _relocate(Path(step.data_file), path)
    return document, solved


# -- what an experiment looks like ---------------------------------------------


def _sig(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return float(f"{number:.{digits}g}")


def signature(measured: ExperimentMeasurements) -> dict[str, Any]:
    """The measurements a precedent and a new test are compared on."""
    readable = [step for step in measured.steps if step.readable and step.curvature_mean is not None]
    out: dict[str, Any] = {"readable_steps": len(readable), "steps": len(measured.steps)}
    if not readable:
        return out
    labels: list[str] = next((list(step.force_labels) for step in readable if step.force_labels), [])
    width = max((len(step.force) for step in readable if step.force), default=0)
    names = [labels[index] if index < len(labels) else f"F[{index}]" for index in range(width)]
    moving: list[str] = []
    for index in range(width):
        values = [step.force[index] for step in readable if step.force and len(step.force) == width]
        if values and max(values) - min(values) > 0.0:
            moving.append(names[index])
    out["load_channels"] = names
    out["moving_channels"] = moving
    out["channels_labelled"] = bool(labels)

    probe = measured.probe_load_index
    loaded = [
        step for step in readable
        if step.force and probe is not None and probe < len(step.force) and math.isfinite(step.force[probe])
    ]
    peak = max(loaded, key=lambda step: abs(step.force[probe])) if loaded else readable[-1]
    base = readable[0]
    increment = np.asarray(peak.curvature_mean, dtype=float)
    if peak is not base:
        increment = increment - np.asarray(base.curvature_mean, dtype=float)
    size = float(np.linalg.norm(increment))
    out["curvature_increment_inv_mm"] = {
        "from_step": base.number,
        "to_step": peak.number,
        "kxx": _sig(increment[0]),
        "kyy": _sig(increment[1]),
        "kxy": _sig(increment[2]),
    }
    if size > 0.0:
        unit = increment / size
        out["curvature_direction"] = {"kxx": _sig(unit[0], 3), "kyy": _sig(unit[1], 3), "kxy": _sig(unit[2], 3)}

    xs = [value for step in readable for value in (step.x_min, step.x_max) if value is not None]
    ys = [value for step in readable for value in (step.y_min, step.y_max) if value is not None]
    if xs and ys:
        span_x, span_y = max(xs) - min(xs), max(ys) - min(ys)
        out["span_mm"] = {"x": _sig(span_x), "y": _sig(span_y)}
        if span_x > 0 and span_y > 0:
            out["aspect_ratio_y_to_x"] = _sig(span_y / span_x, 3)
            if measured.sample_area:
                out["area_over_bounding_box"] = _sig(measured.sample_area / (span_x * span_y), 3)
    out["sample_area_mm2"] = _sig(measured.sample_area)
    out["alignment_deg"] = _sig(measured.alignment_deg, 3)
    return out


def _mesh_record(experiment: ExperimentModel) -> dict[str, Any]:
    mesh = experiment.mesh
    custom = {
        "x_nodes": len(mesh.x_nodes or []),
        "y_nodes": len(mesh.y_nodes or []),
        "free_nodes": len(mesh.free_nodes or []),
        "keep_cells": len(mesh.keep_cells or []),
    }
    return {
        "number_of_elements_x": int(mesh.n_elements_x),
        "number_of_elements_y": int(mesh.n_elements_y),
        "structured": not any(custom.values()),
        "custom_layout": {key: value for key, value in custom.items() if value},
        "bounds": {key: _sig(value, 6) for key, value in mesh.bounds.model_dump().items()},
    }


def _node_positions(document: SessionDocument, experiment: ExperimentModel) -> Optional[np.ndarray]:
    """Mesh node coordinates scaled to 0-1 on the mesh, or None when the mesh does not build."""
    try:
        from vfmstudio.core.mesh_convert import build_experiment_mesh, mesh_xyz

        xyz = mesh_xyz(build_experiment_mesh(document, experiment))
    except Exception:
        return None
    lo = xyz[:, :2].min(axis=0)
    span = np.where(np.ptp(xyz[:, :2], axis=0) > 0, np.ptp(xyz[:, :2], axis=0), 1.0)
    return (xyz[:, :2] - lo) / span


def _bc_record(document: SessionDocument, experiment: ExperimentModel) -> dict[str, Any]:
    record: dict[str, Any] = {kind: [] for kind in BC_KINDS}
    positions = None
    uses_nodes = any(entry.nodes for kind in BC_KINDS for entry in getattr(experiment, kind))
    if uses_nodes:
        positions = _node_positions(document, experiment)
    for kind in BC_KINDS:
        for entry in getattr(experiment, kind):
            data = entry.to_solver_dict()
            if entry.nodes and positions is not None:
                ids = [index for index in entry.nodes if 0 <= index < positions.shape[0]]
                data["node_positions_on_mesh"] = [[round(float(u), 3), round(float(v), 3)] for u, v in positions[ids]]
            record[kind].append(data)
    record["placement"] = "nodes" if uses_nodes else "edges"
    return record


def _scheme_key(bc: dict[str, Any]) -> str:
    rows = []
    for kind in BC_KINDS:
        for entry in bc.get(kind) or []:
            rows.append([
                kind,
                entry.get("dof"),
                entry.get("edge"),
                entry.get("node_positions_on_mesh") if not entry.get("edge") else None,
                entry.get("relation"),
                entry.get("slopeDof"),
                entry.get("loadType"),
                entry.get("sortDirection"),
                entry.get("forceIndex"),
            ])
    rows.sort(key=lambda row: json.dumps(row, sort_keys=True))
    text = json.dumps(rows, sort_keys=True)
    return "S-" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


# -- building the library ------------------------------------------------------


def harvest(roots: Sequence[Path], progress=lambda _text: None) -> dict[str, Any]:
    """Scan saved projects under ``roots`` and return the precedent library."""
    roots = [Path(root).resolve() for root in roots]
    projects: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    precedents: dict[str, dict[str, Any]] = {}
    schemes: dict[str, dict[str, Any]] = {}
    cache: dict[tuple, dict[str, Any]] = {}
    for path in _project_files(roots):
        progress(f"reading {path}")
        try:
            document, solved = _read_project(path)
        except Exception as exc:
            skipped.append({"project": str(path), "reason": str(exc)})
            continue
        projects.append({
            "project": str(path),
            "results_stored": solved,
            "material_type": document.material.material_type,
            "analysis_type": document.material.analysis_type,
            "experiments": [experiment.name for experiment in document.experiments],
        })
        for experiment in document.experiments:
            if not experiment.steps:
                continue
            files = tuple(str(Path(step.data_file)) for step in experiment.steps)
            key_data = (files, bool(experiment.fit_data_to_system_axes), experiment.sample_area_total)
            if key_data not in cache:
                cache[key_data] = signature(measure_experiment(experiment))
            measured = cache[key_data]
            bc = _bc_record(document, experiment)
            if not any(bc[kind] for kind in BC_KINDS):
                continue
            scheme = _scheme_key(bc)
            mesh = _mesh_record(experiment)
            identity = json.dumps(
                [experiment.name, [Path(item).name for item in files], scheme,
                 mesh["number_of_elements_x"], mesh["number_of_elements_y"], mesh["custom_layout"],
                 float(experiment.material_orientation_deg), document.material.material_type],
                sort_keys=True,
            )
            precedent_id = "P-" + hashlib.sha1(identity.encode("utf-8")).hexdigest()[:8]
            entry = precedents.get(precedent_id)
            if entry is None:
                entry = precedents[precedent_id] = {
                    "id": precedent_id,
                    "name": experiment.name,
                    "material_type": document.material.material_type,
                    "analysis_type": document.material.analysis_type,
                    "material_orientation_deg": float(experiment.material_orientation_deg),
                    "fit_data_to_system_axes": bool(experiment.fit_data_to_system_axes),
                    "sample_area_mm2": _sig(experiment.sample_area_total),
                    "mesh": mesh,
                    "scheme": scheme,
                    "boundary_conditions": bc,
                    "signature": measured,
                    "step_files": [Path(item).name for item in files],
                    "occurrences": [],
                }
            entry["occurrences"].append({
                "project": str(path),
                "results_stored": solved,
                "included_in_identification": bool(experiment.include_in_identification),
                "analysis_start": experiment.analysis_start,
                "analysis_end": experiment.analysis_end,
                "reference_step": experiment.reference_step,
            })
            group = schemes.setdefault(scheme, {
                "id": scheme,
                "placement": bc["placement"],
                "boundary_conditions": {kind: bc[kind] for kind in BC_KINDS},
                "precedents": [],
            })
            if precedent_id not in group["precedents"]:
                group["precedents"].append(precedent_id)
    for entry in precedents.values():
        entry["n_projects"] = len(entry["occurrences"])
        entry["n_projects_with_results"] = sum(1 for item in entry["occurrences"] if item["results_stored"])
    for group in schemes.values():
        group["experiments"] = sorted({precedents[item]["name"] for item in group["precedents"]})
        group["n_projects_with_results"] = sum(precedents[item]["n_projects_with_results"] for item in group["precedents"])
    return {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "roots": [str(root) for root in roots],
        "ranking_rule": RANKING_RULE,
        "projects": projects,
        "skipped": skipped,
        "schemes": sorted(schemes.values(), key=lambda item: (-item["n_projects_with_results"], item["id"])),
        "precedents": sorted(precedents.values(), key=lambda item: (item["name"], item["id"])),
    }


def library_path() -> Optional[Path]:
    raw = os.environ.get(LIBRARY_ENV)
    if raw is None:
        return LIBRARY_PATH
    if raw.strip().lower() in {"", "off", "none"}:
        return None
    return Path(raw)


def load_library(path: Optional[Path] = None) -> dict[str, Any]:
    """The distilled training library. Never opens a project file."""
    from vfmstudio.agents.model_build.library import load_library as load_training

    return load_training(path)


# -- matching a new test ---------------------------------------------------------


def _cosine(a: Optional[dict], b: Optional[dict]) -> Optional[float]:
    if not a or not b:
        return None
    u = np.array([a.get(key) or 0.0 for key in ("kxx", "kyy", "kxy")], dtype=float)
    v = np.array([b.get(key) or 0.0 for key in ("kxx", "kyy", "kxy")], dtype=float)
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu == 0.0 or nv == 0.0:
        return None
    return abs(float(u @ v) / (nu * nv))


def _channels_compatible(current: dict, other: dict) -> Optional[bool]:
    if "moving_channels" not in current or "moving_channels" not in other:
        return None
    if current.get("channels_labelled") and other.get("channels_labelled"):
        return sorted(current["moving_channels"]) == sorted(other["moving_channels"])
    return len(current["moving_channels"]) == len(other["moving_channels"])


def compare(current: dict[str, Any], precedent: dict[str, Any], material_type: str) -> dict[str, Any]:
    other = precedent.get("signature") or {}
    cosine = _cosine(current.get("curvature_direction"), other.get("curvature_direction"))
    aspect = None
    a, b = current.get("aspect_ratio_y_to_x"), other.get("aspect_ratio_y_to_x")
    if a and b:
        aspect = max(a / b, b / a)
    area_ratio = None
    if current.get("area_over_bounding_box") is not None and other.get("area_over_bounding_box") is not None:
        area_ratio = abs(current["area_over_bounding_box"] - other["area_over_bounding_box"])
    return {
        "load_channels_compatible": _channels_compatible(current, other),
        "moving_channels": {"this_test": current.get("moving_channels"), "precedent": other.get("moving_channels")},
        "same_material_type": str(precedent.get("material_type")).lower() == str(material_type).lower(),
        "curvature_direction_match": _sig(cosine, 3),
        "aspect_ratio_factor": _sig(aspect, 3),
        "area_over_bounding_box_difference": _sig(area_ratio, 3),
    }


def rank(library: dict[str, Any], current: dict[str, Any], material_type: str, limit: int = MAX_MATCHES) -> list[dict[str, Any]]:
    """The closest precedents to ``current``, with the comparison that placed them."""
    rows = []
    for precedent in library.get("precedents") or []:
        comparison = compare(current, precedent, material_type)
        cosine = comparison["curvature_direction_match"]
        aspect = comparison["aspect_ratio_factor"]
        key = (
            comparison["load_channels_compatible"] is False,
            not comparison["same_material_type"],
            -(round(cosine, 2) if cosine is not None else -1.0),
            round(math.log(aspect), 1) if aspect else 99.0,
            -int(precedent.get("n_projects_with_results") or 0),
            precedent.get("name") or "",
        )
        rows.append((key, precedent, comparison))
    rows.sort(key=lambda row: row[0])
    matches = []
    for _key, precedent, comparison in rows[:limit]:
        matches.append({
            "id": precedent["id"],
            "name": precedent["name"],
            "comparison": comparison,
            "material_type": precedent.get("material_type"),
            "material_orientation_deg": precedent.get("material_orientation_deg"),
            "mesh": precedent.get("mesh"),
            "scheme": precedent.get("scheme"),
            "boundary_conditions": precedent.get("boundary_conditions"),
            "signature": precedent.get("signature"),
            "n_projects": precedent.get("n_projects"),
            "n_projects_with_results": precedent.get("n_projects_with_results"),
            "excluded_from_identification_in": sum(
                1 for item in precedent.get("occurrences") or [] if not item.get("included_in_identification")
            ),
        })
    return matches


def transfer(
    scheme: dict[str, Any],
    precedents: Sequence[dict[str, Any]],
    measured: ExperimentMeasurements,
    current: dict[str, Any],
) -> dict[str, Any]:
    """Boundary entries of ``scheme`` written for this test, and how the loads are wired to F."""
    bc = scheme.get("boundary_conditions") or {}
    entries = {kind: [dict(item) for item in bc.get(kind) or []] for kind in BC_KINDS}
    for kind in BC_KINDS:
        for item in entries[kind]:
            item.pop("node_positions_on_mesh", None)
    if scheme.get("placement") != "edges":
        return {
            **entries,
            "transferable": False,
            "wiring": "placed on mesh nodes of the precedent's own mesh, so it does not carry over to another mesh",
        }
    external = entries["external"]
    labels = list(current.get("load_channels") or [])
    moving = list(current.get("moving_channels") or [])
    if not external:
        return {**entries, "transferable": True, "wiring": "no external load"}
    width = len(labels)
    if width == 0:
        return {**entries, "transferable": False, "wiring": "this test has no F, so the external load has no value"}
    if width == 1:
        for item in external:
            item.pop("forceIndex", None)
        return {**entries, "transferable": True, "wiring": "this test has one F channel; every external load reads it"}
    explicit = [item for item in external if item.get("forceIndex") is not None]
    if explicit:
        source_labels = next(
            (p["signature"].get("load_channels") for p in precedents if (p.get("signature") or {}).get("channels_labelled")),
            None,
        )
        if not source_labels:
            source_labels = (scheme.get("use_when") or {}).get("load_channels")
        if not source_labels:
            return {**entries, "transferable": False, "wiring": "the precedent wires F by index without channel names"}
        for item in explicit:
            index = int(item["forceIndex"])
            name = source_labels[index] if index < len(source_labels) else None
            if name not in labels:
                return {**entries, "transferable": False, "wiring": f"this test has no load channel named {name}"}
            item["forceIndex"] = labels.index(name)
        return {**entries, "transferable": True, "wiring": "F channels matched by name"}
    if len(external) == 1 and measured.load_resolved and measured.load_index is not None:
        external[0]["forceIndex"] = int(measured.load_index)
        return {
            **entries,
            "transferable": True,
            "wiring": f"the external load reads {labels[measured.load_index]}, the only moving F channel",
        }
    if len(external) == len(moving) == width:
        return {**entries, "transferable": True, "wiring": "external loads read F in list order, as in the precedent"}
    return {
        **entries,
        "transferable": False,
        "wiring": (
            f"this test has {width} F channels ({', '.join(labels)}) and the precedent has "
            f"{len(external)} external load(s) without a channel; which channel loads which edge is open"
        ),
    }


def boundary_entries(candidate: dict[str, Any]) -> dict[str, list[BoundaryEntry]]:
    return {kind: [BoundaryEntry.model_validate(item) for item in candidate.get(kind) or []] for kind in BC_KINDS}


# -- command line -------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    from vfmstudio.agents.model_build.library import distill

    parser = argparse.ArgumentParser(
        prog="python -m vfmstudio.agents.model_build.precedents",
        description=(
            "Offline: scan saved VFM projects and write the distilled BC training library. "
            "The model build agent never reads the project files."
        ),
    )
    parser.add_argument("roots", nargs="+", type=Path, help="folders (or project files) to scan")
    parser.add_argument("--out", type=Path, default=LIBRARY_PATH, help=f"library file (default {LIBRARY_PATH})")
    args = parser.parse_args(argv)
    harvested = harvest(args.roots, progress=lambda text: print(f"  {text}", file=sys.stderr))
    library = distill(harvested)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(library, indent=1, ensure_ascii=False), encoding="utf-8")
    print(
        f"{len(library['schemes'])} BC scheme(s) distilled from "
        f"{len(harvested['precedents'])} experiment(s) in {len(harvested['projects'])} project(s); "
        f"{len(harvested['skipped'])} skipped. Wrote {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
