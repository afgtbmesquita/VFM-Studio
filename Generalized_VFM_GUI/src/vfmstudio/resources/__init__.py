"""Small SVG toolbar icons."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QIcon


def _icons_dir() -> Path:
    return Path(__file__).resolve().parent / "icons"


def icon(name: str) -> QIcon:
    path = _icons_dir() / f"{name}.svg"
    if not path.is_file():
        return QIcon()
    return QIcon(str(path))
