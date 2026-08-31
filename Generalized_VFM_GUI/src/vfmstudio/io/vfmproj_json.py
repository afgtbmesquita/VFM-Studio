"""Round-trip the existing Generalized VFM ``*.vfmproj.json`` format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from vfmstudio.core import bridge
from vfmstudio.core.document import (
    SessionDocument,
    document_from_vfm_project,
    document_to_vfm_project,
)


_SIDECAR_INT_KEYS = ("analysis_start", "analysis_end", "reference_step")
_SIDECAR_BOOL_KEYS = (
    "fit_data_to_system_axes",
    "include_in_identification",
    "show_deformed_configuration",
)
_SIDECAR_FLOAT_KEYS = ("deformation_scale",)


def _vfmstudio_sidecar(document: SessionDocument) -> Dict[str, Any]:
    experiments: Dict[str, Dict[str, Any]] = {}
    for experiment in document.experiments:
        payload = {
            key: getattr(experiment, key) for key in _SIDECAR_INT_KEYS
        }
        payload.update(
            {key: bool(getattr(experiment, key)) for key in _SIDECAR_BOOL_KEYS}
        )
        payload.update(
            {key: float(getattr(experiment, key)) for key in _SIDECAR_FLOAT_KEYS}
        )
        experiments[str(experiment.experiment_id)] = payload
    return {"experiments": experiments}


def _apply_vfmstudio_sidecar(document: SessionDocument, path: Path) -> None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    extra = payload.get("vfmstudio") or {}
    experiments = extra.get("experiments") or {}
    if not isinstance(experiments, dict):
        return
    for key, values in experiments.items():
        if not isinstance(values, dict):
            continue
        try:
            experiment_id = int(key)
        except (TypeError, ValueError):
            continue
        experiment = document.experiment_by_id(experiment_id)
        if experiment is None:
            continue
        for field in _SIDECAR_INT_KEYS:
            if field not in values:
                continue
            raw = values[field]
            setattr(experiment, field, None if raw is None else int(raw))
        for field in _SIDECAR_BOOL_KEYS:
            if field not in values:
                continue
            setattr(experiment, field, bool(values[field]))
        for field in _SIDECAR_FLOAT_KEYS:
            if field not in values:
                continue
            try:
                setattr(experiment, field, float(values[field]))
            except (TypeError, ValueError):
                continue


def _write_vfmstudio_sidecar(document: SessionDocument, path: Path) -> None:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["vfmstudio"] = _vfmstudio_sidecar(document)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_vfmproj(path: Path) -> SessionDocument:
    project = bridge.load_project(Path(path))
    document = document_from_vfm_project(project)
    document.source_path = Path(path).resolve()
    from vfmstudio.core.step_series import relocate_document_data_files

    relocate_document_data_files(document)
    _apply_vfmstudio_sidecar(document, path)
    return document


def save_vfmproj(document: SessionDocument, path: Path) -> Path:
    project = document_to_vfm_project(document)
    out = Path(path)
    bridge.save_project(project, out)
    _write_vfmstudio_sidecar(document, out)
    document.source_path = out.resolve()
    return out
