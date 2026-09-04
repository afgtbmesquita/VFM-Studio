"""Toolbar SVG icons and the application window icon."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap

_APP_ICON: QIcon | None = None


def _icons_dir() -> Path:
    return Path(__file__).resolve().parent / "icons"


def _resources_dir() -> Path:
    return Path(__file__).resolve().parent


def _app_icon_path() -> Path | None:
    resources = _resources_dir()
    if sys.platform == "win32":
        ico = resources / "abdXtract_Icon.ico"
        if ico.is_file():
            return ico
    png = resources / "abdXtract_Icon.png"
    if png.is_file():
        return png
    return None


def icon(name: str) -> QIcon:
    path = _icons_dir() / f"{name}.svg"
    if not path.is_file():
        return QIcon()
    return QIcon(str(path))


def app_icon() -> QIcon:
    """abdXtract window / taskbar icon (multi-size for Windows)."""
    global _APP_ICON
    if _APP_ICON is not None and not _APP_ICON.isNull():
        return _APP_ICON

    path = _app_icon_path()
    if path is None:
        _APP_ICON = QIcon()
        return _APP_ICON

    if path.suffix.lower() == ".ico":
        icon_obj = QIcon(str(path))
        if not icon_obj.isNull():
            _APP_ICON = icon_obj
            return _APP_ICON

    master = QPixmap(str(path))
    if master.isNull():
        _APP_ICON = QIcon()
        return _APP_ICON
    icon_obj = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        icon_obj.addPixmap(
            master.scaled(
                size,
                size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
    _APP_ICON = icon_obj
    return _APP_ICON
