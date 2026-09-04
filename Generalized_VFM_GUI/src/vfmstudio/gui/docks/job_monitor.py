"""Job monitor dock: queue, elapsed time, last log line. Cancel kills the solver process."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


def _elapsed_text(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:d}:{sec:02d}"


@dataclass
class JobRecord:
    token: int
    name: str
    status: str = "Running"
    started: datetime = field(default_factory=datetime.now)
    finished_at: Optional[datetime] = None
    last_log: str = ""

    def elapsed_s(self) -> float:
        end = self.finished_at or datetime.now()
        return max(0.0, (end - self.started).total_seconds())


class JobMonitorDock(QWidget):
    cancel_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._jobs: list[JobRecord] = []
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["Job", "Status", "Started", "Elapsed", "Log"]
        )
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._hint = QLabel("No jobs yet.")
        self._hint.setProperty("role", "hint")
        self._cancel = QPushButton("Cancel job")
        self._cancel.setEnabled(False)
        self._cancel.clicked.connect(self.cancel_requested.emit)
        row = QHBoxLayout()
        row.addWidget(self._hint, 1)
        row.addWidget(self._cancel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self._table, 1)
        layout.addLayout(row)
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._refresh_elapsed)

    def start_job(self, name: str, token: int) -> None:
        self._jobs.append(JobRecord(token=int(token), name=str(name)))
        if not self._timer.isActive():
            self._timer.start()
        self._rebuild()

    def log_line(self, text: str, token: Optional[int] = None) -> None:
        job = self._job(token)
        if job is None:
            return
        job.last_log = str(text).strip().splitlines()[-1] if text else ""
        self._rebuild()

    def finish_job(self, status: str, token: Optional[int] = None) -> None:
        job = self._job(token)
        if job is None:
            return
        job.status = str(status)
        job.finished_at = datetime.now()
        if not any(item.status == "Running" for item in self._jobs):
            self._timer.stop()
        self._rebuild()

    def mark_cancelled(self, token: Optional[int] = None) -> None:
        job = self._job(token)
        if job is None:
            if self._jobs:
                job = self._jobs[-1]
            else:
                return
        if job.status == "Running":
            job.status = "Cancelled"
            job.finished_at = datetime.now()
        if not any(item.status == "Running" for item in self._jobs):
            self._timer.stop()
        self._rebuild()

    def _job(self, token: Optional[int]) -> Optional[JobRecord]:
        if token is None:
            for item in reversed(self._jobs):
                if item.status == "Running":
                    return item
            return self._jobs[-1] if self._jobs else None
        for item in reversed(self._jobs):
            if item.token == int(token):
                return item
        return None

    def _refresh_elapsed(self) -> None:
        if any(job.status == "Running" for job in self._jobs):
            self._rebuild()

    def _rebuild(self) -> None:
        running = any(job.status == "Running" for job in self._jobs)
        self._cancel.setEnabled(running)
        if not self._jobs:
            self._hint.setText("No jobs yet.")
        elif running:
            self._hint.setText("A job is running. Cancel stops the solver process.")
        else:
            self._hint.setText("Idle.")
        self._table.setRowCount(len(self._jobs))
        for row, job in enumerate(reversed(self._jobs)):
            values = (
                job.name,
                job.status,
                job.started.strftime("%H:%M:%S"),
                _elapsed_text(job.elapsed_s()),
                job.last_log,
            )
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._table.setItem(row, column, item)
