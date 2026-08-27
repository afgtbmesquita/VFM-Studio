"""Native HDF5 ``.vfm`` container. JSON remains the solver interchange format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np

from vfmstudio.core.document import SessionDocument
from vfmstudio.io.vfmproj_json import load_vfmproj, save_vfmproj


FORMAT_NAME = "vfmstudio-hdf5"
FORMAT_VERSION = 1
GROUP_NAMES = (
    "geometry",
    "experiment",
    "dic",
    "kinematics",
    "virtual_fields",
    "identification",
)


def save_vfm(document: SessionDocument, path: Path, result: Any = None) -> Path:
    path = Path(path)
    payload = document.model_dump(mode="json")
    with h5py.File(path, "w") as handle:
        handle.attrs["format"] = FORMAT_NAME
        handle.attrs["version"] = FORMAT_VERSION
        handle.create_dataset(
            "document_json",
            data=json.dumps(payload, indent=2).encode("utf-8"),
        )
        for name in GROUP_NAMES:
            group = handle.create_group(name)
            if name == "geometry" and document.experiments:
                mesh = document.experiments[0].mesh
                group.attrs["n_elements_x"] = mesh.n_elements_x
                group.attrs["n_elements_y"] = mesh.n_elements_y
            if name == "experiment":
                group.attrs["n_experiments"] = len(document.experiments)
        if result is not None:
            group = handle["identification"]
            theta = getattr(result, "final_parameters", None)
            if theta is not None:
                group.create_dataset("parameters", data=np.asarray(theta, dtype=float))
            covariance = getattr(result, "parameter_covariance", None)
            if covariance is not None:
                group.create_dataset(
                    "parameter_covariance",
                    data=np.asarray(covariance, dtype=float),
                )
    document.source_path = path.resolve()
    return path


def load_vfm(path: Path) -> SessionDocument:
    path = Path(path)
    with h5py.File(path, "r") as handle:
        raw = handle["document_json"][()]
        if isinstance(raw, bytes):
            text = raw.decode("utf-8")
        else:
            text = bytes(raw).decode("utf-8")
    payload = json.loads(text)
    document = SessionDocument.model_validate(payload)
    document.source_path = path.resolve()
    from vfmstudio.core.step_series import relocate_document_data_files

    relocate_document_data_files(document)
    return document


def load_any(path: Path) -> SessionDocument:
    path = Path(path)
    name = path.name.lower()
    if name.endswith(".vfmproj.json") or path.suffix.lower() in {".json", ".vfmproj"}:
        return load_vfmproj(path)
    if path.suffix.lower() in {".vfm", ".h5", ".hdf5"}:
        return load_vfm(path)
    try:
        return load_vfmproj(path)
    except Exception:
        return load_vfm(path)


def save_any(document: SessionDocument, path: Path, result: Optional[Any] = None) -> Path:
    path = Path(path)
    if path.suffix.lower() in {".vfm", ".h5", ".hdf5"}:
        return save_vfm(document, path, result=result)
    return save_vfmproj(document, path)
