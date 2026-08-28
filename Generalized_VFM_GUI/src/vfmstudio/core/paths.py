"""Locate the sibling Generalized VFM solver without modifying it."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path


ENV_CORE_PATH = "GVFM_CORE_PATH"


def gui_root() -> Path:
    """``Generalized_VFM_GUI/`` (the application root)."""
    return Path(__file__).resolve().parents[3]


def workspace_root() -> Path:
    """Parent of both ``Generalized_VFM_GUI`` and ``Generalized VFM`` (VFM Studio)."""
    return gui_root().parent


def default_core_path() -> Path:
    return workspace_root() / "Generalized VFM"


def campaign_project_path() -> Path:
    return default_core_path() / "projects" / "campaign.vfmproj.json"


def samples_ab_all_project_path() -> Path:
    return default_core_path() / "projects" / "samples_ab_all.vfmproj.json"


def _looks_like_core(path: Path) -> bool:
    return path.is_dir() and (path / "vfm_project.py").is_file()


@lru_cache(maxsize=1)
def resolve_core_path() -> Path:
    candidates = []
    override = os.environ.get(ENV_CORE_PATH, "").strip()
    if override:
        candidates.append(Path(override).expanduser())
    candidates.append(default_core_path())
    for parent in gui_root().parents:
        candidates.append(parent / "Generalized VFM")
        candidates.append(parent / "VFM Studio" / "Generalized VFM")
    seen: set[str] = set()
    for candidate in candidates:
        try:
            path = candidate.resolve()
        except OSError:
            path = candidate
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        if _looks_like_core(path):
            return path
    raise FileNotFoundError(
        f"Generalized VFM solver folder not found. "
        f"Set {ENV_CORE_PATH} to the solver directory."
    )


def ensure_core_on_path() -> Path:
    """Insert the solver folder on ``sys.path`` so ``import vfm_project`` works."""
    core = resolve_core_path()
    text = str(core)
    if text not in sys.path:
        sys.path.insert(0, text)
    return core
