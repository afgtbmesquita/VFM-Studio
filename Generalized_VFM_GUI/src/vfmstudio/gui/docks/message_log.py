"""Message / diagnostics log."""

from __future__ import annotations

from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QPlainTextEdit


class MessageLogDock(QPlainTextEdit):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(4000)
        self.log("VFM Studio ready.")

    def log(self, text: str) -> None:
        self.appendPlainText(text)
        self.moveCursor(QTextCursor.MoveOperation.End)
        self.ensureCursorVisible()
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())
