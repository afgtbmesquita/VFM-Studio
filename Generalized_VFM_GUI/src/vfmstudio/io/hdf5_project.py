"""Native HDF5 ``.vfm`` container. JSON remains the solver interchange format."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import h5py

from vfmstudio.core.document import SessionDocument
from vfmstudio.io.result_store import dump_snapshot, load_snapshot
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
_VFM_SUFFIXES = {".vfm", ".h5", ".hdf5"}


def is_scratch_project_path(path: Optional[Path]) -> bool:
    """True for Identify temp copies (``vfmstudio_job_*/in.vfm``), not user files."""
    if path is None:
        return True
    return Path(path).parent.name.startswith("vfmstudio_job_")


def normalize_project_save_path(path: Path, selected_filter: str = "") -> Path:
    """Attach the suffix the Save-As filter asked for when Windows omits it."""
    path = Path(path)
    name = path.name.lower()
    if name.endswith(".vfmproj.json") or path.suffix.lower() in _VFM_SUFFIXES | {".vfmproj"}:
        return path
    selected = str(selected_filter).lower()
    want_json = "json" in selected and ".vfm)" not in selected
    if path.suffix.lower() == ".json" or want_json:
        stem = path.stem if path.suffix else path.name
        return path.with_name(f"{stem}.vfmproj.json")
    if not path.suffix:
        return path.with_suffix(".vfm")
    return path


@dataclass
class LoadedProject:
    document: SessionDocument
    result: Any = None
    identifiability: Any = None
    uncertainty: Any = None
    view_state: dict = field(default_factory=dict)


def save_vfm(
    document: SessionDocument,
    path: Path,
    result: Any = None,
    identifiability: Any = None,
    uncertainty: Any = None,
    view_state: Optional[dict] = None,
    *,
    bind_source: bool = True,
) -> Path:
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
        dump_snapshot(
            handle,
            result=result,
            identifiability=identifiability,
            uncertainty=uncertainty,
            view_state=view_state,
        )
    if bind_source:
        document.source_path = path.resolve()
    return path


def _document_from_handle(handle, path: Path, *, bind_source: bool = True) -> SessionDocument:
    raw = handle["document_json"][()]
    if isinstance(raw, bytes):
        text = raw.decode("utf-8")
    else:
        text = bytes(raw).decode("utf-8")
    payload = json.loads(text)
    document = SessionDocument.model_validate(payload)
    if bind_source or document.source_path is None:
        document.source_path = path.resolve()
    from vfmstudio.core.step_series import relocate_document_data_files

    relocate_document_data_files(document)
    return document


def load_vfm_project(path: Path, *, bind_source: bool = True) -> LoadedProject:
    path = Path(path)
    with h5py.File(path, "r") as handle:
        document = _document_from_handle(handle, path, bind_source=bind_source)
        result, identifiability, uncertainty, view_state = load_snapshot(handle)
    return LoadedProject(
        document=document,
        result=result,
        identifiability=identifiability,
        uncertainty=uncertainty,
        view_state=view_state,
    )


def load_vfm(path: Path) -> SessionDocument:
    return load_vfm_project(path).document


def load_project(path: Path) -> LoadedProject:
    path = Path(path)
    name = path.name.lower()
    if name.endswith(".vfmproj.json") or path.suffix.lower() in {".json", ".vfmproj"}:
        return LoadedProject(document=load_vfmproj(path))
    if path.suffix.lower() in {".vfm", ".h5", ".hdf5"}:
        return load_vfm_project(path)
    try:
        return LoadedProject(document=load_vfmproj(path))
    except Exception:
        return load_vfm_project(path)


def load_any(path: Path) -> SessionDocument:
    return load_project(path).document


def save_any(
    document: SessionDocument,
    path: Path,
    result: Optional[Any] = None,
    identifiability: Optional[Any] = None,
    uncertainty: Optional[Any] = None,
    view_state: Optional[dict] = None,
    *,
    bind_source: bool = True,
) -> Path:
    path = Path(path)
    if path.suffix.lower() in _VFM_SUFFIXES:
        return save_vfm(
            document,
            path,
            result=result,
            identifiability=identifiability,
            uncertainty=uncertainty,
            view_state=view_state,
            bind_source=bind_source,
        )
    return save_vfmproj(document, path, bind_source=bind_source)
