"""Assistant side panel: chat with Claude agents through the local Claude Code CLI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QImage,
    QKeyEvent,
    QResizeEvent,
    QTextCursor,
    QTextImageFormat,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from vfmstudio.agents import (
    AgentEvent,
    AgentSpec,
    ClaudeRun,
    available_agents,
    build_arguments,
    build_command,
    find_claude_cli,
)
from vfmstudio.agents.claude_cli import denied_tool_names
from vfmstudio.agents.writing.interpreter import HUMAN_INTERPRETER, build_interpreter_prompt
from vfmstudio.agents.writing.present import prepare_markdown
from vfmstudio.gui.prefs import claude_cli_path, set_claude_cli_path

STREAM_FLUSH_MS = 60
STDERR_TAIL = 12


class _RunBridge(QObject):
    """Carries callbacks from the CLI reader thread onto the GUI thread."""

    event = Signal(int, object)
    stderr = Signal(int, str)
    exited = Signal(int, int, bool)


class ChatInput(QPlainTextEdit):
    submitted = Signal()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        enter = event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
        if enter and not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self.submitted.emit()
            return
        super().keyPressEvent(event)


class _RichBody(QTextBrowser):
    """Markdown body. Vertical scrolling stays with the chat; this widget scrolls sideways."""

    def wheelEvent(self, event: QWheelEvent) -> None:
        event.ignore()


class ChatMessage(QFrame):
    def __init__(
        self,
        role: str,
        text: str = "",
        caption: str = "",
        rich: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.role = role
        self._rich = rich
        self._source = ""
        self._base: Optional[Path] = None
        self._laying_out = False
        self.setObjectName("chatMessage")
        self.setProperty("role", role)
        layout = QVBoxLayout(self)
        margin = 2 if role == "tool" else 8
        layout.setContentsMargins(margin + 6, margin, margin + 6, margin)
        if caption:
            label = QLabel(caption)
            label.setObjectName("chatCaption")
            layout.addWidget(label)
        if rich:
            self.view = _RichBody()
            self.view.setObjectName("chatText")
            self.view.setProperty("role", role)
            self.view.setFrameShape(QFrame.Shape.NoFrame)
            self.view.setOpenExternalLinks(True)
            self.view.setOpenLinks(False)
            self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self.view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.view.document().setDefaultStyleSheet(
                "table { border-collapse: collapse; } "
                "th, td { padding: 3px 6px; border: 1px solid #D0D0D0; }"
            )
            self.label = None
            layout.addWidget(self.view)
        else:
            self.view = None
            self.label = QLabel()
            self.label.setObjectName("chatText")
            self.label.setProperty("role", role)
            self.label.setWordWrap(True)
            self.label.setTextFormat(
                Qt.TextFormat.MarkdownText if role in ("assistant", "report") else Qt.TextFormat.PlainText
            )
            self.label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
                | Qt.TextInteractionFlag.LinksAccessibleByMouse
            )
            self.label.setOpenExternalLinks(True)
            layout.addWidget(self.label)
        self.set_text(text)

    def set_text(self, text: str) -> None:
        self._source = text
        if self.view is not None:
            self.view.setMarkdown(text)
            self._layout_document(scale_images=False)
        elif self.label is not None:
            self.label.setText(text)

    def set_document(self, markdown: str, base: Optional[Path]) -> None:
        """Show finished Markdown, resolving images against ``base``."""
        self._source = markdown
        self._base = Path(base) if base is not None else None
        if self.view is None:
            if self.label is not None:
                self.label.setText(markdown)
            return
        if self._base is not None:
            self.view.document().setBaseUrl(QUrl.fromLocalFile(os.path.join(str(self._base), "")))
        self.view.setMarkdown(markdown)
        if self._base is not None:
            self.view.document().setBaseUrl(QUrl.fromLocalFile(os.path.join(str(self._base), "")))
        self._layout_document(scale_images=True)

    def text(self) -> str:
        return self._source

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if self.view is not None:
            self._layout_document(scale_images=True)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self.view is not None:
            event.ignore()
            return
        super().wheelEvent(event)

    def _layout_document(self, scale_images: bool) -> None:
        view = self.view
        if view is None or self._laying_out:
            return
        self._laying_out = True
        try:
            width = view.viewport().width()
            if width < 40:
                width = max(40, self.width() - 28)
            document = view.document()
            document.setTextWidth(width)
            if scale_images:
                self._scale_images(width)
                document.setTextWidth(width)
            height = int(document.size().height()) + view.frameWidth() * 2 + 6
            bar = view.horizontalScrollBar()
            if bar.maximum() > 0:
                height += bar.sizeHint().height()
            if view.height() != height:
                view.setFixedHeight(max(height, 1))
        finally:
            self._laying_out = False

    def _scale_images(self, width: int) -> None:
        view = self.view
        if view is None:
            return
        document = view.document()
        found: list[tuple[int, int, QTextImageFormat]] = []
        block = document.begin()
        while block.isValid():
            iterator = block.begin()
            while not iterator.atEnd():
                fragment = iterator.fragment()
                fmt = fragment.charFormat()
                if fragment.isValid() and fmt.isImageFormat():
                    found.append(
                        (fragment.position(), fragment.length(), QTextImageFormat(fmt.toImageFormat()))
                    )
                iterator += 1
            block = block.next()
        limit = max(1, width - 8)
        for position, length, image_fmt in found:
            image = self._load_image(image_fmt.name())
            if image is None or image.width() <= 0 or image.height() <= 0:
                continue
            display = min(image.width(), limit)
            image_fmt.setWidth(display)
            image_fmt.setHeight(display * image.height() / image.width())
            cursor = QTextCursor(document)
            cursor.setPosition(position)
            cursor.setPosition(position + length, QTextCursor.MoveMode.KeepAnchor)
            cursor.setCharFormat(image_fmt)

    def _load_image(self, name: str) -> Optional[QImage]:
        candidates = [Path(name)]
        if self._base is not None:
            candidates.append(self._base / name)
        url = QUrl(name)
        if url.isLocalFile():
            candidates.append(Path(url.toLocalFile()))
        for path in candidates:
            if path.is_file():
                image = QImage(str(path))
                if not image.isNull():
                    return image
        return None


class AgentChatPanel(QWidget):
    log_message = Signal(str)
    busy_changed = Signal(bool)
    # An agent with a launcher window was chosen; the main window opens that window.
    launcher_requested = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._agents: list[AgentSpec] = available_agents()
        self._agent: AgentSpec = self._agents[0]
        self._context_provider: Optional[Callable[[], str]] = None
        self._workdir_provider: Optional[Callable[[], Optional[Path]]] = None
        self._cli: Optional[Path] = None
        self._session_id: Optional[str] = None
        self._session_cwd: Optional[Path] = None
        self._run: Optional[ClaudeRun] = None
        self._run_id = 0
        self._got_result = False
        self._shown_text = False
        self._stderr_tail: list[str] = []
        self._reply: Optional[ChatMessage] = None
        self._reply_text = ""
        self._turn_text = ""
        self._stick_to_bottom = True
        # "specialist" buffers prose for the Human Interpreter; "interpreter" is that pass.
        self._phase: Optional[str] = None
        self._follow_up = False
        self._report_source = ""
        self._figures: dict[str, dict[str, str]] = {}
        self._specialist_text = ""

        self._agent_combo = QComboBox()
        for spec in self._agents:
            self._agent_combo.addItem(spec.name, spec.agent_id)
            self._agent_combo.setItemData(
                self._agent_combo.count() - 1, spec.description, Qt.ItemDataRole.ToolTipRole
            )
        self._agent_combo.currentIndexChanged.connect(self._on_agent_changed)
        self._new_chat = QToolButton()
        self._new_chat.setText("New chat")
        self._new_chat.setToolTip("Start a fresh conversation with the selected agent.")
        self._new_chat.clicked.connect(self.new_chat)
        self._cli_button = QToolButton()
        self._cli_button.setObjectName("chatCliButton")
        self._cli_button.setText("CLI ▾")
        self._cli_button.setToolTip("Choose which Claude CLI executable the panel runs.")
        self._cli_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        cli_menu = QMenu(self._cli_button)
        cli_menu.addAction("Locate Claude CLI…", self._choose_cli)
        cli_menu.addAction("Use auto-detected CLI", self._reset_cli)
        self._cli_button.setMenu(cli_menu)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(QLabel("Agent"))
        header.addWidget(self._agent_combo, 1)
        header.addWidget(self._new_chat)
        header.addWidget(self._cli_button)
        launchers = QHBoxLayout()
        launchers.setContentsMargins(0, 0, 0, 0)
        self._launch_buttons: dict[str, QPushButton] = {}
        for spec in self._agents:
            if not spec.launch_label:
                continue
            button = QPushButton(spec.launch_label)
            button.setObjectName("chatLaunchButton")
            button.setToolTip(spec.description)
            button.clicked.connect(lambda _checked=False, agent_id=spec.agent_id: self.use_agent(agent_id))
            launchers.addWidget(button)
            self._launch_buttons[spec.agent_id] = button
        launchers.addStretch(1)

        self._transcript = QWidget()
        self._transcript.setObjectName("chatTranscript")
        self._messages = QVBoxLayout(self._transcript)
        self._messages.setContentsMargins(6, 6, 6, 6)
        self._messages.setSpacing(6)
        self._messages.addStretch(1)
        self._scroll = QScrollArea()
        self._scroll.setObjectName("chatScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setWidget(self._transcript)
        bar = self._scroll.verticalScrollBar()
        bar.rangeChanged.connect(self._on_scroll_range)
        bar.valueChanged.connect(self._on_scroll_value)

        self._status = QLabel()
        self._status.setProperty("role", "hint")
        self._status.setWordWrap(True)

        self._input = ChatInput()
        self._input.setObjectName("chatInput")
        self._input.setPlaceholderText("Ask Claude…  (Enter to send, Shift+Enter for a new line)")
        self._input.setMaximumHeight(96)
        self._input.submitted.connect(self.send)
        self._send = QPushButton("Send")
        self._send.setProperty("role", "compact")
        self._send.clicked.connect(self._on_send_clicked)
        input_row = QHBoxLayout()
        input_row.setContentsMargins(0, 0, 0, 0)
        input_row.addWidget(self._input, 1)
        input_row.addWidget(self._send, 0, Qt.AlignmentFlag.AlignBottom)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.addLayout(header)
        if self._launch_buttons:
            layout.addLayout(launchers)
        layout.addWidget(self._scroll, 1)
        layout.addWidget(self._status)
        layout.addLayout(input_row)

        self._flush_timer = QTimer(self)
        self._flush_timer.setSingleShot(True)
        self._flush_timer.setInterval(STREAM_FLUSH_MS)
        self._flush_timer.timeout.connect(self._flush_reply)

        self._bridge = _RunBridge(self)
        self._bridge.event.connect(self._on_event)
        self._bridge.stderr.connect(self._on_stderr)
        self._bridge.exited.connect(self._on_exited)

        self._greet()
        self._set_status("Ready.")

    # -- public API --------------------------------------------------------

    def set_context_provider(self, provider: Optional[Callable[[], str]]) -> None:
        """Text appended to the system prompt each turn (current GUI state)."""
        self._context_provider = provider

    def set_workdir_provider(self, provider: Optional[Callable[[], Optional[Path]]]) -> None:
        """Working directory for a new chat; pinned for the rest of that chat."""
        self._workdir_provider = provider

    @property
    def agent(self) -> AgentSpec:
        return self._agent

    @property
    def busy(self) -> bool:
        return self._run is not None

    def messages(self) -> list[ChatMessage]:
        items = []
        for index in range(self._messages.count()):
            widget = self._messages.itemAt(index).widget()
            if isinstance(widget, ChatMessage):
                items.append(widget)
        return items

    def focus_input(self) -> None:
        self._input.setFocus()

    def new_chat(self) -> None:
        self.stop()
        for message in self.messages():
            self._messages.removeWidget(message)
            message.deleteLater()
        self._session_id = None
        self._session_cwd = None
        self._reply = None
        self._reply_text = ""
        self._phase = None
        self._follow_up = False
        self._report_source = ""
        self._figures = {}
        self._specialist_text = ""
        self._greet()
        self._set_status("Ready.")

    def use_agent(self, agent_id: str) -> None:
        """Switch to ``agent_id`` and open its launcher window, if it has one."""
        self._select_agent(agent_id)
        if self._agent.launcher:
            self.launcher_requested.emit(self._agent.launcher)

    def start_task(
        self,
        agent_id: str,
        prompt: str,
        display: str,
        workdir: Path,
        attachments: Sequence[tuple[str, str]] = (),
        figures: Optional[Mapping[str, Mapping[str, str]]] = None,
    ) -> bool:
        """Fresh chat with ``agent_id`` pinned to ``workdir``; ``display`` is shown instead of ``prompt``.

        ``attachments`` are (role, text) messages shown under the request, before the reply.
        A ``report`` attachment on an agent with ``interpret`` is source text for the Human
        Interpreter and is not posted. ``figures`` maps a test name to figure-key paths
        relative to ``workdir``.
        """
        self.stop()
        if self._agent.agent_id != agent_id:
            self._select_agent(agent_id)
        self.new_chat()
        self._session_cwd = Path(workdir)
        self._figures = {
            str(test): {str(key): str(path) for key, path in items.items()}
            for test, items in (figures or {}).items()
            if isinstance(items, Mapping)
        }
        return self.send(prompt, display=display, attachments=attachments)

    def post_message(self, role: str, text: str) -> None:
        self._add_message(role, text)

    def send(
        self,
        text: Optional[str] = None,
        display: Optional[str] = None,
        attachments: Sequence[tuple[str, str]] = (),
    ) -> bool:
        prompt = (text if text is not None else self._input.toPlainText()).strip()
        if not prompt or self.busy:
            return False
        if self._agent.launcher and self._session_cwd is None:
            self._add_message(
                "tool",
                self._agent.launch_hint
                or (
                    f"{self._agent.name} works on data loaded in its window. Load files there and "
                    "press Analyse; you can then ask follow-up questions here."
                ),
            )
            self.launcher_requested.emit(self._agent.launcher)
            return False
        cli = self._resolve_cli()
        if cli is None:
            self._add_message(
                "error",
                "Claude CLI not found. Install Claude Code, or use CLI ▸ Locate Claude CLI… "
                "to point at claude.exe.",
            )
            return False
        if text is None:
            self._input.clear()
        visible = list(attachments)
        self._phase = None
        if self._agent.interpret:
            reports = [body for role, body in attachments if role == "report" and body.strip()]
            visible = [(role, body) for role, body in attachments if role != "report"]
            if reports:
                self._report_source = "\n\n".join(reports)
                self._follow_up = False
            self._phase = "specialist"
        self._add_message("user", display or prompt)
        for role, body in visible:
            self._add_message(role, body)
        if self._session_cwd is None:
            self._session_cwd = self._initial_workdir()
        context = ""
        if self._agent.gui_context and self._context_provider is not None:
            try:
                context = self._context_provider() or ""
            except Exception as exc:
                self.log_message.emit(f"Assistant context unavailable: {exc}")
        arguments = build_arguments(
            self._agent, resume_session=self._session_id, extra_system_prompt=context
        )
        self._run_id += 1
        run_id = self._run_id
        self._got_result = False
        self._shown_text = False
        self._stderr_tail = []
        self._reply = None
        self._reply_text = ""
        self._turn_text = ""
        bridge = self._bridge
        self._run = ClaudeRun(
            build_command(cli, arguments),
            prompt,
            self._session_cwd,
            on_event=lambda event: bridge.event.emit(run_id, event),
            on_stderr=lambda line: bridge.stderr.emit(run_id, line),
            on_exit=lambda code, stopped: bridge.exited.emit(run_id, code, stopped),
        )
        self._set_busy(True)
        self._set_status(f"{self._agent.name} is working…")
        self._run.start()
        return True

    def stop(self) -> None:
        run = self._run
        if run is None:
            return
        self._run = None
        self._run_id += 1
        self._phase = None
        run.stop()
        self._finish_reply()
        self._add_message("tool", "Stopped.")
        self._set_busy(False)
        self._set_status("Stopped.")

    def shutdown(self) -> None:
        run = self._run
        self._run = None
        self._run_id += 1
        if run is not None:
            run.stop()

    # -- CLI discovery -----------------------------------------------------

    def _resolve_cli(self) -> Optional[Path]:
        if self._cli is None or not self._cli.is_file():
            self._cli = find_claude_cli(claude_cli_path())
        return self._cli

    def _choose_cli(self) -> None:
        start = str(self._cli.parent) if self._cli else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "Locate Claude CLI", start, "Claude CLI (claude.exe claude.cmd claude);;All files (*)"
        )
        if not path:
            return
        set_claude_cli_path(path)
        self._cli = Path(path)
        self._add_message("tool", f"Using Claude CLI: {path}")

    def _reset_cli(self) -> None:
        set_claude_cli_path(None)
        self._cli = None
        cli = self._resolve_cli()
        self._add_message("tool", f"Using Claude CLI: {cli}" if cli else "Claude CLI not found.")

    # -- stream handling ---------------------------------------------------

    def _on_event(self, run_id: int, event: AgentEvent) -> None:
        if run_id != self._run_id:
            return
        kind = event.kind
        if kind == "session":
            if self._phase != "interpreter":
                self._session_id = event.data.get("session_id") or self._session_id
        elif kind == "thinking":
            if self._phase == "interpreter":
                self._set_status("Human Interpreter is writing…")
            elif self._phase == "specialist":
                self._set_status(f"{self._agent.name} is working…")
            else:
                self._set_status(f"{self._agent.name} is thinking…")
        elif kind == "text_delta":
            self._reply_text += event.text
            self._turn_text += event.text
            self._shown_text = True
            if self._phase == "specialist":
                self._set_status(f"{self._agent.name} is working…")
                return
            if self._reply is None:
                rich = self._phase == "interpreter"
                caption = "Human Interpreter" if rich else ""
                self._reply = self._add_message("assistant", "", caption=caption, rich=rich)
            if not self._flush_timer.isActive():
                self._flush_timer.start()
            if self._phase == "interpreter":
                self._set_status("Human Interpreter is writing…")
            else:
                self._set_status(f"{self._agent.name} is writing…")
        elif kind == "tool_use":
            self._finish_reply()
            name = event.data.get("name") or "tool"
            detail = f" · {event.text}" if event.text else ""
            self._add_message("tool", f"⚙ {name}{detail}")
            self._set_status(f"Running {name}…")
        elif kind == "tool_result":
            if self._phase == "specialist":
                self._set_status(f"{self._agent.name} is working…")
            elif self._phase == "interpreter":
                self._set_status("Human Interpreter is writing…")
            else:
                self._set_status(f"{self._agent.name} is thinking…")
        elif kind == "result":
            if self._phase == "specialist":
                self._on_specialist_result(event)
            elif self._phase == "interpreter":
                self._on_interpreter_result(event)
            else:
                self._on_direct_result(event)

    def _on_direct_result(self, event: AgentEvent) -> None:
        self._got_result = True
        self._finish_reply()
        self._session_id = event.data.get("session_id") or self._session_id
        if event.data.get("is_error"):
            self._add_message("error", event.text or "Claude reported an error.")
        elif event.text and not self._shown_text:
            self._add_message("assistant", event.text)
            self._turn_text = event.text
        if not event.data.get("is_error"):
            self._audit_turn()
        self._show_denials(event)
        self._set_ready(event)

    def _on_specialist_result(self, event: AgentEvent) -> None:
        self._got_result = True
        self._finish_reply()
        self._session_id = event.data.get("session_id") or self._session_id
        text = self._turn_text.strip() or (event.text or "").strip()
        self._specialist_text = text
        self._show_denials(event)
        if event.data.get("is_error"):
            self._phase = None
            self._add_message("error", event.text or "Claude reported an error.")
            self._set_status("Failed.")
            return
        if not text:
            self._phase = None
            self._add_message("error", f"{self._agent.name} returned no text.")
            self._set_ready(event)
            return
        self._begin_interpreter()

    def _begin_interpreter(self) -> None:
        follow_up = self._follow_up
        prompt = build_interpreter_prompt(
            specialist_text=self._specialist_text,
            report_text="" if follow_up else self._report_source,
            figures=self._figures,
            follow_up=follow_up,
        )
        cli = self._resolve_cli()
        if cli is None:
            self._phase = None
            self._reveal_specialist(
                "Human Interpreter could not start because the Claude CLI was not found."
            )
            self._set_status("Failed.")
            return
        self._follow_up = True
        self._phase = "interpreter"
        arguments = build_arguments(HUMAN_INTERPRETER, resume_session=None)
        self._run_id += 1
        run_id = self._run_id
        self._got_result = False
        self._shown_text = False
        self._stderr_tail = []
        self._reply = None
        self._reply_text = ""
        self._turn_text = ""
        bridge = self._bridge
        self._run = ClaudeRun(
            build_command(cli, arguments),
            prompt,
            self._session_cwd,
            on_event=lambda item: bridge.event.emit(run_id, item),
            on_stderr=lambda line: bridge.stderr.emit(run_id, line),
            on_exit=lambda code, stopped: bridge.exited.emit(run_id, code, stopped),
        )
        self._set_busy(True)
        self._set_status("Human Interpreter is writing…")
        self._run.start()

    def _on_interpreter_result(self, event: AgentEvent) -> None:
        self._got_result = True
        raw = self._turn_text if self._shown_text else (event.text or "")
        bubble = self._reply
        self._finish_reply()
        self._show_denials(event)
        if event.data.get("is_error") or not raw.strip():
            if bubble is not None:
                self._remove_message(bubble)
            self._phase = None
            detail = (event.text or "").strip() or "Human Interpreter could not organize this reply."
            self._reveal_specialist(detail)
            self._set_status("Ready.")
            return
        try:
            presented = prepare_markdown(
                raw,
                workdir=self._session_cwd,
                figures=self._figures,
                sources=self._layout_sources(),
            )
        except Exception as exc:
            self.log_message.emit(f"Human Interpreter layout failed: {exc}")
            presented = raw
        if bubble is None:
            bubble = self._add_message("assistant", "", caption="Human Interpreter", rich=True)
        bubble.set_document(presented, self._session_cwd)
        self._turn_text = raw
        self._audit_turn()
        self._phase = None
        self._set_ready(event)

    def _reveal_specialist(self, error: str) -> None:
        self._add_message("error", error)
        if not self._specialist_text.strip():
            return
        self._add_message("assistant", self._specialist_text)
        self._turn_text = self._specialist_text
        self._audit_turn()

    def _layout_sources(self) -> str:
        return "\n\n".join(part for part in (self._report_source, self._specialist_text) if part)

    def _show_denials(self, event: AgentEvent) -> None:
        denied = denied_tool_names(event.data.get("permission_denials") or [])
        if denied:
            self._add_message(
                "tool",
                "Not permitted in this panel: " + ", ".join(denied)
                + ". Headless CLI runs cannot ask for approval; the agent's "
                "tool settings decide what it may do.",
            )

    def _set_ready(self, event: AgentEvent) -> None:
        cost = event.data.get("cost_usd")
        self._set_status(f"Ready · ${cost:.3f}" if isinstance(cost, (int, float)) else "Ready.")

    def _on_stderr(self, run_id: int, line: str) -> None:
        if run_id != self._run_id:
            return
        self._stderr_tail = (self._stderr_tail + [line])[-STDERR_TAIL:]
        self.log_message.emit(f"[claude] {line}")

    def _on_exited(self, run_id: int, code: int, stopped: bool) -> None:
        if run_id != self._run_id:
            return
        phase = self._phase
        bubble = self._reply
        self._run = None
        self._finish_reply()
        if not stopped and not self._got_result:
            detail = "\n".join(self._stderr_tail) or f"Claude CLI exited with code {code}."
            if phase == "interpreter":
                if bubble is not None:
                    self._remove_message(bubble)
                self._reveal_specialist(detail)
            else:
                self._add_message("error", detail)
            self._phase = None
            self._set_status("Failed.")
        self._set_busy(False)

    def _audit_turn(self) -> None:
        auditor = self._agent.auditor
        if auditor is None or not self._turn_text.strip():
            return
        try:
            note = auditor(self._turn_text, self._session_cwd)
        except Exception as exc:
            self.log_message.emit(f"Reply audit failed: {exc}")
            return
        if note:
            self._add_message("audit", note)

    def _flush_reply(self) -> None:
        if self._reply is not None:
            self._reply.set_text(self._reply_text)

    def _finish_reply(self) -> None:
        self._flush_timer.stop()
        self._flush_reply()
        self._reply = None
        self._reply_text = ""

    # -- widgets -----------------------------------------------------------

    def _greet(self) -> None:
        self._add_message(
            "tool",
            f"{self._agent.name}: {self._agent.description}".strip(),
        )

    def _add_message(
        self, role: str, text: str, caption: str = "", rich: bool = False
    ) -> ChatMessage:
        message = ChatMessage(role, text, caption=caption, rich=rich)
        self._messages.insertWidget(self._messages.count() - 1, message)
        self._stick_to_bottom = True
        return message

    def _remove_message(self, message: ChatMessage) -> None:
        self._messages.removeWidget(message)
        message.hide()
        message.deleteLater()
        if self._reply is message:
            self._reply = None
            self._reply_text = ""

    def _initial_workdir(self) -> Optional[Path]:
        if self._workdir_provider is None:
            return None
        try:
            path = self._workdir_provider()
        except Exception:
            return None
        return Path(path) if path is not None and Path(path).is_dir() else None

    def _set_busy(self, busy: bool) -> None:
        self._send.setText("Stop" if busy else "Send")
        self._agent_combo.setEnabled(not busy)
        self.busy_changed.emit(busy)

    def _set_status(self, text: str) -> None:
        self._status.setText(text)

    def _on_send_clicked(self) -> None:
        if self.busy:
            self.stop()
        else:
            self.send()

    def _select_agent(self, agent_id: str) -> None:
        for index, spec in enumerate(self._agents):
            if spec.agent_id == agent_id:
                self._agent_combo.blockSignals(True)
                self._agent_combo.setCurrentIndex(index)
                self._agent_combo.blockSignals(False)
                if spec is not self._agent:
                    self._agent = spec
                    self.new_chat()
                return
        raise KeyError(agent_id)

    def _on_agent_changed(self, index: int) -> None:
        if 0 <= index < len(self._agents) and self._agents[index] is not self._agent:
            self._agent = self._agents[index]
            self.new_chat()
            if self._agent.launcher:
                self.launcher_requested.emit(self._agent.launcher)

    def _on_scroll_range(self, _low: int, high: int) -> None:
        if self._stick_to_bottom:
            self._scroll.verticalScrollBar().setValue(high)

    def _on_scroll_value(self, value: int) -> None:
        bar = self._scroll.verticalScrollBar()
        self._stick_to_bottom = value >= bar.maximum() - 24
