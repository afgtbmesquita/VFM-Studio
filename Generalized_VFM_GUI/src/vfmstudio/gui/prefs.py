"""QSettings helpers: recent files and window layout."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QByteArray, QSettings

from vfmstudio.branding import APP_AUTHOR, APP_NAME

RECENT_LIMIT = 8


def ui_persist_enabled() -> bool:
    return "PYTEST_CURRENT_TEST" not in os.environ


def make_settings() -> QSettings:
    return QSettings(APP_AUTHOR, APP_NAME)


def recent_files(settings: Optional[QSettings] = None) -> list[Path]:
    store = settings or make_settings()
    raw = store.value("recentFiles", [])
    if isinstance(raw, str):
        raw = [raw]
    paths: list[Path] = []
    seen = set()
    for item in raw or []:
        path = Path(str(item))
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            paths.append(path)
        if len(paths) >= RECENT_LIMIT:
            break
    return paths


def remember_file(path: Path, settings: Optional[QSettings] = None) -> list[Path]:
    store = settings or make_settings()
    path = Path(path)
    unique: list[Path] = [path]
    seen = {str(path)}
    raw = store.value("recentFiles", [])
    if isinstance(raw, str):
        raw = [raw]
    for item in raw or []:
        candidate = Path(str(item))
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
        if len(unique) >= RECENT_LIMIT:
            break
    store.setValue("recentFiles", [str(item) for item in unique])
    return unique


def load_geometry(settings: Optional[QSettings] = None) -> Optional[QByteArray]:
    store = settings or make_settings()
    value = store.value("geometry")
    return value if isinstance(value, QByteArray) and not value.isEmpty() else None


def load_window_state(settings: Optional[QSettings] = None) -> Optional[QByteArray]:
    store = settings or make_settings()
    value = store.value("windowState")
    return value if isinstance(value, QByteArray) and not value.isEmpty() else None


def save_window(geometry: QByteArray, state: QByteArray, settings: Optional[QSettings] = None) -> None:
    store = settings or make_settings()
    store.setValue("geometry", geometry)
    store.setValue("windowState", state)
