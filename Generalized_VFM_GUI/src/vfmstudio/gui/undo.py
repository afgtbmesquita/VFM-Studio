"""Document-snapshot undo for GUI edits (not identification results)."""

from __future__ import annotations

import time
from typing import Callable, Optional

from PySide6.QtGui import QUndoCommand, QUndoStack

from vfmstudio.core.document import SessionDocument

_COALESCE_S = 0.40
_LIMIT = 50


class _SnapshotCommand(QUndoCommand):
    def __init__(
        self,
        apply_fn: Callable[[dict], None],
        before: dict,
        after: dict,
        text: str,
        coalesce_key: Optional[str] = None,
    ) -> None:
        super().__init__(text)
        self._apply = apply_fn
        self.before = before
        self.after = after
        self.coalesce_key = coalesce_key
        self._stamp = time.monotonic()
        self._fresh = True

    def id(self) -> int:  # noqa: A003 — Qt API
        if not self.coalesce_key:
            return -1
        return abs(hash(self.coalesce_key)) % 2_000_000_000

    def mergeWith(self, other: QUndoCommand) -> bool:  # noqa: N802
        if not isinstance(other, _SnapshotCommand):
            return False
        if not self.coalesce_key or self.coalesce_key != other.coalesce_key:
            return False
        if time.monotonic() - self._stamp > _COALESCE_S:
            return False
        self.after = other.after
        self._stamp = other._stamp
        return True

    def redo(self) -> None:
        if self._fresh:
            self._fresh = False
            return
        self._apply(self.after)

    def undo(self) -> None:
        self._apply(self.before)


class DocumentHistory:
    def __init__(self, apply_fn: Callable[[dict], None]) -> None:
        self._apply = apply_fn
        self.stack = QUndoStack()
        self.stack.setUndoLimit(_LIMIT)
        self._snapshot: dict = {}

    def capture(self, document: SessionDocument) -> None:
        self._snapshot = _dump(document)

    def reset(self, document: SessionDocument) -> None:
        self.stack.clear()
        self.capture(document)

    def commit(
        self,
        document: SessionDocument,
        label: str,
        coalesce_key: Optional[str] = None,
    ) -> bool:
        after = _dump(document)
        if after == self._snapshot:
            return False
        command = _SnapshotCommand(
            self._apply, self._snapshot, after, label, coalesce_key
        )
        self.stack.push(command)
        self._snapshot = after
        return True


def _dump(document: SessionDocument) -> dict:
    return document.model_dump(mode="json")


def document_from_dump(payload: dict) -> SessionDocument:
    return SessionDocument.model_validate(payload)
