"""QAction registry for the VFM Studio menus and toolbar."""

from __future__ import annotations

from typing import Callable, Dict, Optional

from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import QMainWindow


class ActionRegistry:
    def __init__(self, window: QMainWindow) -> None:
        self.window = window
        self._actions: Dict[str, QAction] = {}

    def add(
        self,
        name: str,
        text: str,
        slot: Callable,
        shortcut=None,
        icon: Optional[QIcon] = None,
        menu=None,
        checkable: bool = False,
        checked: bool = False,
    ) -> QAction:
        action = QAction(text, self.window)
        if icon is not None:
            action.setIcon(icon)
        if shortcut is not None:
            action.setShortcut(
                shortcut if isinstance(shortcut, QKeySequence) else QKeySequence(shortcut)
            )
        action.setCheckable(checkable)
        if checkable:
            action.setChecked(checked)
            action.toggled.connect(slot)
        else:
            action.triggered.connect(slot)
        if menu is not None:
            menu.addAction(action)
        self._actions[name] = action
        return action

    def get(self, name: str) -> QAction:
        return self._actions[name]
