"""Abaqus-style exclusive module switcher."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QToolBar, QToolButton

MODULES = ("model", "mesh", "bcs", "job", "results")

_LABELS = {
    "model": "Model",
    "mesh": "Mesh",
    "bcs": "BCs",
    "job": "Job",
    "results": "Results",
}


class ModuleBar(QToolBar):
    module_changed = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__("Modules", parent)
        self.setObjectName("moduleBar")
        self.setMovable(False)
        self.setFloatable(False)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._group = QActionGroup(self)
        self._group.setExclusive(True)
        self._actions: dict[str, QAction] = {}
        self._block = False
        for name in MODULES:
            action = QAction(_LABELS[name], self)
            action.setCheckable(True)
            action.setData(name)
            self._group.addAction(action)
            self.addAction(action)
            self._actions[name] = action
            action.toggled.connect(self._on_toggled)
        self._actions["model"].setChecked(True)
        for button in self.findChildren(QToolButton):
            button.setAutoRaise(False)

    def current(self) -> str:
        checked = self._group.checkedAction()
        if checked is None:
            return "model"
        return str(checked.data() or "model")

    def set_current(self, module: str) -> None:
        name = str(module or "model")
        if name not in self._actions:
            name = "model"
        if self.current() == name:
            return
        self._block = True
        self._actions[name].setChecked(True)
        self._block = False

    def _on_toggled(self, checked: bool) -> None:
        if self._block or not checked:
            return
        self.module_changed.emit(self.current())
