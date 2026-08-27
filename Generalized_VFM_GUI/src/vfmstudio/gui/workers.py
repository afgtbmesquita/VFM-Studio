"""Background identification worker. Solver work stays off the GUI thread."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from vfmstudio.core.document import SessionDocument
from vfmstudio.core.jobs import run_identifiability, run_identification


class WorkerSignals(QObject):
    finished = Signal(object)
    failed = Signal(str)
    log = Signal(str)


class IdentifyRunnable(QRunnable):
    def __init__(self, document: SessionDocument) -> None:
        super().__init__()
        self.document = document
        self.signals = WorkerSignals()
        self.setAutoDelete(False)

    @Slot()
    def run(self) -> None:
        try:
            self.signals.log.emit("Building solver project and running FGLS…")
            result = run_identification(self.document, verbose=False)
            self.signals.finished.emit(result)
        except Exception as exc:
            self.signals.failed.emit(str(exc))


class IdentifiabilityRunnable(QRunnable):
    def __init__(self, document: SessionDocument, result: Any) -> None:
        super().__init__()
        self.document = document
        self.result = result
        self.signals = WorkerSignals()
        self.setAutoDelete(False)

    @Slot()
    def run(self) -> None:
        try:
            self.signals.log.emit("Computing identifiability (H, SVD, CID)…")
            diagnostics = run_identifiability(self.document, self.result)
            self.signals.finished.emit(diagnostics)
        except Exception as exc:
            self.signals.failed.emit(str(exc))
