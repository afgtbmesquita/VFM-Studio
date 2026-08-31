"""Message / diagnostics log."""

from __future__ import annotations

from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QPlainTextEdit

from vfmstudio.branding import APP_NAME


class MessageLogDock(QPlainTextEdit):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(4000)
        self.log(f"{APP_NAME} ready.")

    def log(self, text: str) -> None:
        self.appendPlainText(text)
        self.moveCursor(QTextCursor.MoveOperation.End)
        self.ensureCursorVisible()
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())
