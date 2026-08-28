"""Background identification worker. Solver work stays off the GUI thread."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from vfmstudio.core.document import SessionDocument
from vfmstudio.core.jobs import run_identifiability, run_identification, run_loo_refits, run_uncertainty


class WorkerSignals(QObject):
    finished = Signal(object)
    failed = Signal(str)
    log = Signal(str)


class IdentifyRunnable(QRunnable):
    def __init__(self, document: SessionDocument, kind: str = "CR2") -> None:
        super().__init__()
        self.document = document
        self.kind = kind
        self.signals = WorkerSignals()
        self.setAutoDelete(False)

    @Slot()
    def run(self) -> None:
        try:
            self.signals.log.emit("Building solver project and running FGLS…")
            result = run_identification(self.document, verbose=False)
            ident = None
            try:
                self.signals.log.emit("Computing identifiability (H, SVD, CID)…")
                ident = run_identifiability(self.document, result)
            except Exception as exc:
                self.signals.log.emit(f"Identifiability skipped: {exc}")
            uq = None
            try:
                self.signals.log.emit("Computing uncertainty quantification…")
                uq = run_uncertainty(self.document, result, kind=self.kind)
            except Exception as exc:
                self.signals.log.emit(f"Uncertainty quantification skipped: {exc}")
            self.signals.finished.emit((result, ident, uq))
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


class UncertaintyRunnable(QRunnable):
    def __init__(self, document: SessionDocument, result: Any, kind: str = "CR2") -> None:
        super().__init__()
        self.document = document
        self.result = result
        self.kind = kind
        self.signals = WorkerSignals()
        self.setAutoDelete(False)

    @Slot()
    def run(self) -> None:
        try:
            self.signals.log.emit(f"Computing uncertainty quantification ({self.kind})…")
            payload = run_uncertainty(self.document, self.result, kind=self.kind)
            self.signals.finished.emit(payload)
        except Exception as exc:
            self.signals.failed.emit(str(exc))


class LoocvRefitRunnable(QRunnable):
    def __init__(self, document: SessionDocument, uncertainty: Any, kind: str = "CR2") -> None:
        super().__init__()
        self.document = document
        self.uncertainty = uncertainty
        self.kind = kind
        self.signals = WorkerSignals()
        self.setAutoDelete(False)

    @Slot()
    def run(self) -> None:
        try:
            self.signals.log.emit("Leave-one-out FGLS refits…")
            payload = run_loo_refits(
                self.document,
                self.uncertainty,
                kind=self.kind,
                log=self.signals.log.emit,
            )
            self.signals.finished.emit(payload)
        except Exception as exc:
            self.signals.failed.emit(str(exc))
