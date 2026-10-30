"""Model build agent window: show the virtual model and the inputs the data did not decide.

The build reads load steps that are already imported. It does not import files,
and it writes the project only when the user applies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from vfmstudio.agents.model_build.apply import apply_to_document
from vfmstudio.agents.model_build.measure import ModelRefused
from vfmstudio.agents.model_build.session import ModelBuildRun, build_model
from vfmstudio.core.document import SessionDocument


class _Signals(QObject):
    finished = Signal(object)
    failed = Signal(object)


class _Task(QRunnable):
    def __init__(self, document: SessionDocument, workdir: Optional[Path]) -> None:
        super().__init__()
        self.document = document
        self.workdir = workdir
        self.signals = _Signals()
        self.setAutoDelete(False)

    @Slot()
    def run(self) -> None:
        try:
            self.signals.finished.emit(build_model(self.document, workdir=self.workdir))
        except Exception as exc:
            self.signals.failed.emit(exc)


class ModelAgentWindow(QMainWindow):
    """Non-modal window for the model build agent."""

    log_message = Signal(str)
    refused = Signal(str)
    ready = Signal(object)
    applied = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Model build agent")
        self.resize(760, 680)
        self._provider: Optional[Callable[[], SessionDocument]] = None
        self._workdir: Optional[Path] = None
        self._run: Optional[ModelBuildRun] = None
        self._task: Optional[_Task] = None
        self._generation = 0
        self._widgets: dict[str, QWidget] = {}
        self.pool = QThreadPool(self)

        self._status = QLabel("Import load steps, then build the virtual model.")
        self._status.setWordWrap(True)
        self._report = QTextBrowser()
        self._report.setOpenExternalLinks(False)
        self._form_host = QWidget()
        self._form = QFormLayout(self._form_host)
        self._form.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._form_host)
        scroll.setMinimumHeight(120)
        self._questions_title = QLabel("Left for you")
        self._questions_title.setProperty("role", "hint")

        self._apply = QPushButton("Apply model")
        self._apply.setEnabled(False)
        self._apply.clicked.connect(self._on_apply)
        self._rebuild = QPushButton("Rebuild")
        self._rebuild.clicked.connect(self.start)
        buttons = QHBoxLayout()
        buttons.addWidget(self._apply)
        buttons.addWidget(self._rebuild)
        buttons.addStretch(1)

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.addWidget(self._status)
        layout.addWidget(self._report, 3)
        layout.addWidget(self._questions_title)
        layout.addWidget(scroll, 2)
        layout.addLayout(buttons)
        self.setCentralWidget(body)

    def set_document_provider(self, provider: Callable[[], SessionDocument]) -> None:
        self._provider = provider

    def set_workdir(self, workdir: Optional[Path]) -> None:
        self._workdir = workdir

    @property
    def busy(self) -> bool:
        return self._task is not None

    @property
    def run(self) -> Optional[ModelBuildRun]:
        return self._run

    def start(self) -> None:
        """Build from the current project. Refuses when nothing has been imported."""
        if self._provider is None:
            self.refused.emit("The model build agent has no project to read.")
            return
        document = self._provider()
        self._generation += 1
        generation = self._generation
        self._apply.setEnabled(False)
        self._status.setText("Reading the imported load steps…")
        snapshot = document.model_copy(deep=True)
        task = _Task(snapshot, self._workdir)
        task.signals.finished.connect(lambda run, generation=generation: self._on_built(generation, run))
        task.signals.failed.connect(lambda exc, generation=generation: self._on_failed(generation, exc))
        self._task = task
        self.pool.start(task)

    def _on_built(self, generation: int, run: ModelBuildRun) -> None:
        if generation != self._generation:
            return
        self._task = None
        self._run = run
        self._report.setMarkdown(run.report_markdown)
        self._fill_questions(run)
        n_open = len(run.questions)
        if n_open:
            self._status.setText(
                f"Model built. {n_open} input(s) are left for you; everything else is already decided."
            )
        else:
            self._status.setText("Model built. Nothing is left to choose. Apply writes it into the project.")
        self._sync_apply()
        self.log_message.emit(f"Model build agent: built {len(run.builds)} experiment(s); facts in {run.facts_path}")
        self.ready.emit(run)

    def _on_failed(self, generation: int, exc: BaseException) -> None:
        if generation != self._generation:
            return
        self._task = None
        self._apply.setEnabled(False)
        if isinstance(exc, ModelRefused):
            text = "Refused: " + " ".join(exc.reasons)
        else:
            text = f"Model build failed: {exc}"
            self.log_message.emit(f"Model build agent: {text}")
        self._status.setText(text)
        self.refused.emit(text)

    def _clear_form(self) -> None:
        while self._form.rowCount():
            self._form.removeRow(0)
        self._widgets.clear()

    def _fill_questions(self, run: ModelBuildRun) -> None:
        self._clear_form()
        if not run.questions:
            self._questions_title.setText("Nothing left to choose.")
            return
        self._questions_title.setText("Left for you — only what the measurements do not decide.")
        for question in run.questions:
            widget = self._widget_for(question)
            self._widgets[question["id"]] = widget
            if isinstance(widget, QCheckBox):
                self._form.addRow(question["experiment"], widget)
            else:
                text = question["prompt"]
                if question.get("blocking"):
                    text = "Required. " + text
                caption = QLabel(text)
                caption.setWordWrap(True)
                self._form.addRow(caption, widget)
        self._sync_apply()

    def _widget_for(self, question: dict[str, Any]) -> QWidget:
        kind = question["kind"]
        if kind == "linear_window":
            return self._window_widget(question)
        if kind == "boundary_conditions":
            if question.get("options"):
                return self._combo_widget(question)
            note = QLabel("No training-library set carries over. Set them on the experiment.")
            note.setWordWrap(True)
            return note
        if kind == "reference":
            return self._combo_widget(question)
        if kind in {"replace_existing", "include"}:
            box = QCheckBox(question["prompt"])
            box.setChecked(bool(question.get("value")))
            box.toggled.connect(self._sync_apply)
            return box
        spin = QDoubleSpinBox()
        spin.setDecimals(4 if kind == "sample_area" else 2)
        lower = question["minimum"] if question.get("minimum") is not None else -1.0e9
        upper = question["maximum"] if question.get("maximum") is not None else 1.0e9
        spin.setRange(float(lower), float(upper))
        if question.get("suffix"):
            spin.setSuffix(str(question["suffix"]))
        if question.get("value") is not None:
            spin.setValue(float(question["value"]))
        spin.valueChanged.connect(lambda _value: self._sync_apply())
        return spin

    def _combo_widget(self, question: dict[str, Any]) -> QComboBox:
        combo = QComboBox()
        if question.get("blocking"):
            combo.addItem("Choose…", None)
        for option in question.get("options") or []:
            combo.addItem(str(option["label"]), option["value"])
        value = question.get("value")
        if value is not None or not question.get("blocking"):
            index = combo.findData(value)
            if index >= 0:
                combo.setCurrentIndex(index)
        combo.currentIndexChanged.connect(self._sync_apply)
        return combo

    def _window_widget(self, question: dict[str, Any]) -> QWidget:
        value = question.get("value") or {}
        host = QWidget()
        row = QHBoxLayout(host)
        row.setContentsMargins(0, 0, 0, 0)
        start = QComboBox()
        end = QComboBox()
        for option in question.get("options") or []:
            start.addItem(str(option["label"]), option["value"])
            end.addItem(str(option["label"]), option["value"])
        start.setCurrentIndex(max(0, start.findData(value.get("start"))))
        end.setCurrentIndex(max(0, end.findData(value.get("end"))))
        start.currentIndexChanged.connect(self._sync_apply)
        end.currentIndexChanged.connect(self._sync_apply)
        row.addWidget(QLabel("Start"))
        row.addWidget(start, 1)
        row.addWidget(QLabel("End"))
        row.addWidget(end, 1)
        host._start = start
        host._end = end
        return host

    def answers(self) -> dict[str, Any]:
        if self._run is None:
            return {}
        collected: dict[str, Any] = {}
        for question in self._run.questions:
            widget = self._widgets.get(question["id"])
            if widget is None:
                continue
            kind = question["kind"]
            if kind == "linear_window":
                collected[question["id"]] = {
                    "start": widget._start.currentData(),
                    "end": widget._end.currentData(),
                }
            elif isinstance(widget, QComboBox):
                collected[question["id"]] = widget.currentData()
            elif isinstance(widget, QCheckBox):
                collected[question["id"]] = widget.isChecked()
            elif isinstance(widget, QDoubleSpinBox):
                collected[question["id"]] = float(widget.value())
        return collected

    def _blocking_open(self) -> bool:
        if self._run is None:
            return True
        answers = self.answers()
        for question in self._run.questions:
            if not question.get("blocking"):
                continue
            value = answers.get(question["id"])
            if value is None or (question["kind"] == "sample_area" and not value):
                return True
        return False

    def _sync_apply(self, *_args) -> None:
        self._apply.setEnabled(self._run is not None and not self.busy and not self._blocking_open())

    def _on_apply(self) -> None:
        if self._provider is None or self._run is None or self._blocking_open():
            return
        notes = apply_to_document(self._provider(), self._run.builds, self.answers())
        text = "\n".join(notes)
        self._status.setText(text.replace("\n", " "))
        self.log_message.emit("Model build agent: applied the virtual model")
        self.applied.emit(text)
