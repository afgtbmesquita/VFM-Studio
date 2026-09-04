"""Background identification worker. Solver work stays off the GUI thread."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QRunnable, Signal, Slot

from vfmstudio.core.document import SessionDocument
from vfmstudio.core.jobs import run_identifiability, run_identification, run_loo_refits, run_uncertainty
from vfmstudio.platform.scipy_wdac import explain_native_import_error, prepare_scipy_native_imports


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
            prepare_scipy_native_imports()
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
            self.signals.failed.emit(explain_native_import_error(exc))


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
            self.signals.failed.emit(explain_native_import_error(exc))


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
            self.signals.failed.emit(explain_native_import_error(exc))


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
            self.signals.failed.emit(explain_native_import_error(exc))


class JobProcess(QObject):
    """Run a solver job in a child process so Cancel can kill it."""

    finished = Signal(object)
    failed = Signal(str)
    log = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._proc: Optional[QProcess] = None
        self._output: Optional[Path] = None
        self._killed = False

    def start(
        self,
        kind: str,
        document: SessionDocument,
        *,
        result: Any = None,
        identifiability: Any = None,
        uncertainty: Any = None,
        view_state: Optional[dict] = None,
        uq_kind: str = "CR2",
    ) -> None:
        from vfmstudio.io.hdf5_project import save_vfm

        tmp = Path(tempfile.mkdtemp(prefix="vfmstudio_job_"))
        inp = tmp / "in.vfm"
        self._output = tmp / "out.vfm"
        original = document.source_path
        try:
            save_vfm(
                document,
                inp,
                result=result,
                identifiability=identifiability,
                uncertainty=uncertainty,
                view_state=view_state,
                bind_source=False,
            )
        finally:
            document.source_path = original
        self._killed = False
        proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        src = str(Path(__import__("vfmstudio").__file__).resolve().parents[1])
        path = env.value("PYTHONPATH", "")
        parts = [item for item in path.split(os.pathsep) if item]
        if src not in parts:
            parts.insert(0, src)
        env.insert("PYTHONPATH", os.pathsep.join(parts))
        proc.setProcessEnvironment(env)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_output)
        proc.finished.connect(self._on_finished)
        self._proc = proc
        proc.start(
            sys.executable,
            [
                "-m",
                "vfmstudio.core.job_runner",
                kind,
                str(inp),
                str(self._output),
                str(uq_kind or "CR2"),
            ],
        )

    def kill(self) -> None:
        self._killed = True
        if self._proc is not None and self._proc.state() != QProcess.ProcessState.NotRunning:
            self._proc.kill()

    def _on_output(self) -> None:
        if self._proc is None:
            return
        text = bytes(self._proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in text.splitlines():
            line = line.strip()
            if line:
                self.log.emit(line)

    def _on_finished(self, code: int, _status) -> None:
        if self._killed:
            return
        if int(code) != 0:
            extra = ""
            if self._proc is not None:
                extra = bytes(self._proc.readAllStandardOutput()).decode(
                    "utf-8", errors="replace"
                ).strip()
            self.failed.emit(extra or f"Job failed (exit {code}).")
            return
        from vfmstudio.io.hdf5_project import load_vfm_project

        try:
            self.finished.emit(load_vfm_project(self._output))
        except Exception as exc:
            self.failed.emit(str(exc))
