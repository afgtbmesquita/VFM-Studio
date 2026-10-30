"""Assistant panel: Claude CLI stream parsing, runner, and chat dock (fake CLI)."""

from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("VFMSTUDIO_NO_VISTA", "1")

from vfmstudio.agents import (
    AgentSpec,
    ClaudeRun,
    build_arguments,
    find_claude_cli,
    parse_stream_line,
)
from vfmstudio.agents.claude_cli import ENV_CLAUDE_CLI, denied_tool_names

FAKE_CLI = textwrap.dedent(
    r'''
    import json, sys
    prompt = sys.stdin.read()
    sid = "sess-1"
    mode = "resumed" if "--resume" in sys.argv else "fresh"
    def emit(obj):
        print(json.dumps(obj), flush=True)
    def delta(text):
        emit({"type": "stream_event", "parent_tool_use_id": None,
              "event": {"type": "content_block_delta", "index": 0,
                        "delta": {"type": "text_delta", "text": text}}})
    print("warming up", file=sys.stderr, flush=True)
    emit({"type": "system", "subtype": "init", "session_id": sid, "model": "fake", "cwd": "."})
    delta("Looking ")
    delta("now.")
    emit({"type": "assistant", "parent_tool_use_id": None, "message": {"content": [
        {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "a.txt"}}]}})
    emit({"type": "user", "parent_tool_use_id": None, "message": {"content": [
        {"type": "tool_result", "tool_use_id": "t1", "is_error": False}]}})
    delta("Echo: " + prompt.strip() + " (" + mode + ")")
    emit({"type": "result", "subtype": "success", "is_error": False,
          "result": "Echo", "session_id": sid, "total_cost_usd": 0.001,
          "permission_denials": [{"tool_name": "Edit"}]})
    '''
)


@pytest.fixture()
def fake_cli(tmp_path) -> Path:
    script = tmp_path / "fake_claude.py"
    script.write_text(FAKE_CLI, encoding="utf-8")
    return script


def _line(obj) -> str:
    return json.dumps(obj)


def test_parse_init_delta_tool_and_result():
    (init,) = parse_stream_line(_line({"type": "system", "subtype": "init", "session_id": "s"}))
    assert init.kind == "session" and init.data["session_id"] == "s"
    (delta,) = parse_stream_line(
        _line(
            {
                "type": "stream_event",
                "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}},
            }
        )
    )
    assert delta.kind == "text_delta" and delta.text == "hi"
    (tool,) = parse_stream_line(
        _line(
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "tool_use", "name": "Bash", "input": {"command": "ls  -la"}}]
                },
            }
        )
    )
    assert tool.kind == "tool_use" and tool.data["name"] == "Bash" and tool.text == "ls -la"
    (result,) = parse_stream_line(
        _line({"type": "result", "result": "done", "is_error": False, "session_id": "s"})
    )
    assert result.kind == "result" and result.text == "done"


def test_parse_skips_subagent_and_garbage():
    sub = {
        "type": "stream_event",
        "parent_tool_use_id": "t9",
        "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "x"}},
    }
    assert parse_stream_line(_line(sub)) == []
    assert parse_stream_line("not json") == []
    assert parse_stream_line("") == []
    assert parse_stream_line(_line({"type": "rate_limit_event"})) == []


def test_build_arguments():
    spec = AgentSpec("x", "X", system_prompt="Be brief.", tools=(), model="sonnet")
    args = build_arguments(spec, resume_session="abc", extra_system_prompt="Ctx")
    assert args[:3] == ["--print", "--output-format", "stream-json"]
    assert args[args.index("--resume") + 1] == "abc"
    assert args[args.index("--tools") + 1] == ""
    assert args[args.index("--model") + 1] == "sonnet"
    assert args[args.index("--append-system-prompt") + 1] == "Be brief.\n\nCtx"
    plain = build_arguments(AgentSpec("y", "Y"))
    assert "--resume" not in plain and "--tools" not in plain
    assert "--append-system-prompt" not in plain


def test_find_claude_cli_prefers_explicit_then_env(tmp_path, monkeypatch):
    chosen = tmp_path / "claude.exe"
    chosen.write_bytes(b"")
    env_cli = tmp_path / "env_claude.exe"
    env_cli.write_bytes(b"")
    monkeypatch.setenv(ENV_CLAUDE_CLI, str(env_cli))
    assert find_claude_cli(chosen) == chosen
    assert find_claude_cli(tmp_path / "missing.exe") == env_cli


def test_denied_tool_names_unique():
    assert denied_tool_names([{"tool_name": "Edit"}, {"tool_name": "Edit"}, {"tool_name": "Bash"}]) == [
        "Edit",
        "Bash",
    ]


def test_claude_run_streams_events(fake_cli, tmp_path):
    events, errors, exits = [], [], []
    run = ClaudeRun(
        [sys.executable, str(fake_cli)],
        "ping",
        tmp_path,
        on_event=events.append,
        on_stderr=errors.append,
        on_exit=lambda code, stopped: exits.append((code, stopped)),
    )
    run.start()
    run.wait(30)
    assert exits == [(0, False)]
    assert errors == ["warming up"]
    kinds = [event.kind for event in events]
    assert kinds[0] == "session" and kinds[-1] == "result"
    assert "tool_use" in kinds and "tool_result" in kinds
    text = "".join(event.text for event in events if event.kind == "text_delta")
    assert text == "Looking now.Echo: ping (fresh)"


# -- GUI ---------------------------------------------------------------------

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _wait_idle(app, panel, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while panel.busy and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    assert not panel.busy


def _fake_panel(monkeypatch, fake_cli):
    from vfmstudio.gui.docks import agent_chat

    monkeypatch.setattr(
        agent_chat,
        "build_command",
        lambda _exe, args: [sys.executable, str(fake_cli), *args],
    )
    panel = agent_chat.AgentChatPanel()
    monkeypatch.setattr(panel, "_resolve_cli", lambda: Path(sys.executable))
    return panel


def test_panel_streams_reply_and_resumes(qapp, monkeypatch, fake_cli, tmp_path):
    panel = _fake_panel(monkeypatch, fake_cli)
    panel.set_workdir_provider(lambda: tmp_path)
    panel.set_context_provider(lambda: "Project: test")
    assert panel.send("hello")
    assert panel._send.text() == "Stop"
    _wait_idle(qapp, panel)
    roles = [(m.role, m.text()) for m in panel.messages()]
    assert roles[1] == ("user", "hello")
    assert ("tool", "⚙ Read · a.txt") in roles
    assistant = [text for role, text in roles if role == "assistant"]
    assert assistant == ["Looking now.", "Echo: hello (fresh)"]
    assert any(role == "tool" and "Edit" in text for role, text in roles)
    assert panel._session_id == "sess-1"
    assert panel._send.text() == "Send"

    assert panel.send("again")
    _wait_idle(qapp, panel)
    assert panel.messages()[-2].text() == "Echo: again (resumed)"

    panel.new_chat()
    assert panel._session_id is None
    assert [m.role for m in panel.messages()] == ["tool"]


def test_general_assistant_is_one_pass(qapp, monkeypatch, fake_cli, tmp_path):
    from vfmstudio.gui.docks import agent_chat

    calls = []

    def command(exe, args):
        calls.append(list(args))
        return [sys.executable, str(fake_cli), *args]

    monkeypatch.setattr(agent_chat, "build_command", command)
    panel = agent_chat.AgentChatPanel()
    monkeypatch.setattr(panel, "_resolve_cli", lambda: Path(sys.executable))
    panel.set_workdir_provider(lambda: tmp_path)
    assert panel.agent.interpret is False
    assert panel.send("hello")
    _wait_idle(qapp, panel)
    assert len(calls) == 1
    assistant = [message.text() for message in panel.messages() if message.role == "assistant"]
    assert assistant == ["Looking now.", "Echo: hello (fresh)"]


def test_panel_interprets_specialist_without_showing_the_report(qapp, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QLabel

    from vfmstudio.gui.docks import agent_chat

    script = tmp_path / "fake_two_pass.py"
    script.write_text(
        textwrap.dedent(
            r'''
            import json, os, sys
            prompt = sys.stdin.read()
            log = os.environ.get("VFM_FAKE_PROMPT_LOG")
            if log:
                with open(log, "a", encoding="utf-8") as handle:
                    argv = " ".join(part.replace("\n", " ") for part in sys.argv)
                    handle.write("ARGV " + argv + "\n")
                    handle.write(prompt)
                    handle.write("\n---END---\n")

            def emit(obj):
                print(json.dumps(obj), flush=True)

            def delta(text):
                emit({"type": "stream_event", "parent_tool_use_id": None,
                      "event": {"type": "content_block_delta", "index": 0,
                                "delta": {"type": "text_delta", "text": text}}})

            interpreter = "## Specialist reply" in prompt
            sid = "sess-interpreter" if interpreter else "sess-specialist"
            emit({"type": "system", "subtype": "init", "session_id": sid, "model": "fake", "cwd": "."})
            if interpreter:
                delta("Organized reply.")
            else:
                emit({"type": "assistant", "parent_tool_use_id": None, "message": {"content": [
                    {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "report.md"}}]}})
                delta("SPEC finding.")
            emit({"type": "result", "subtype": "success", "is_error": False,
                  "result": "done", "session_id": sid, "total_cost_usd": 0.001})
            '''
        ),
        encoding="utf-8",
    )
    log = tmp_path / "prompts.log"
    monkeypatch.setenv("VFM_FAKE_PROMPT_LOG", str(log))
    monkeypatch.setattr(
        agent_chat,
        "build_command",
        lambda _exe, args: [sys.executable, str(script), *args],
    )
    panel = agent_chat.AgentChatPanel()
    monkeypatch.setattr(panel, "_resolve_cli", lambda: Path(sys.executable))
    figures = {"Plate A": {"fields": "figures/a_fields.png"}}
    assert panel.start_task(
        "dic_data",
        "Read facts.json",
        "Interpret 1 test",
        tmp_path,
        attachments=[("report", "# Secret report")],
        figures=figures,
    )
    _wait_idle(qapp, panel)
    assert panel.send("what about the edge?")
    _wait_idle(qapp, panel)

    roles = [(message.role, message.text()) for message in panel.messages()]
    assert all(role != "report" for role, _text in roles)
    assert not any("# Secret report" in text or "SPEC finding." in text for _role, text in roles)
    assert [text for role, text in roles if role == "assistant"] == [
        "Organized reply.",
        "Organized reply.",
    ]
    assert any(role == "tool" and "Read" in text for role, text in roles)
    captions = [
        label.text()
        for message in panel.messages()
        if (label := message.findChild(QLabel, "chatCaption")) is not None
    ]
    assert captions == ["Human Interpreter", "Human Interpreter"]
    assert panel._session_id == "sess-specialist"

    blocks = []
    for part in log.read_text(encoding="utf-8").split("\n---END---\n"):
        if not part.strip():
            continue
        argv, _, body = part.partition("\n")
        blocks.append((argv, body))
    assert len(blocks) == 4
    (spec1, interp1, spec2, interp2) = blocks
    assert "--resume" not in spec1[0]
    assert "Read facts.json" in spec1[1]
    assert "# Secret report" not in interp1[1]
    assert "SPEC finding." in interp1[1]
    assert "{{figure:Plate A/fields}}" in interp1[1]
    assert "noise_maps" not in interp1[1]
    assert "--resume" not in interp1[0]
    assert "--resume" in spec2[0] and "sess-specialist" in spec2[0]
    assert "# Secret report" not in interp2[1]
    assert "Do not restate" in interp2[1]
    assert "SPEC finding." in interp2[1]
    assert "--resume" not in interp2[0]


def test_rich_message_keeps_image_aspect(qapp, tmp_path):
    from PySide6.QtGui import QImage, QTextImageFormat

    from vfmstudio.gui.docks.agent_chat import ChatMessage

    image = QImage(800, 200, QImage.Format.Format_RGB32)
    image.fill(0xFF225588)
    path = tmp_path / "dot.png"
    assert image.save(str(path))
    message = ChatMessage("assistant", "", caption="Human Interpreter", rich=True)
    message.resize(360, 80)
    message.show()
    qapp.processEvents()
    message.set_document(f"![plot]({path.name})", tmp_path)
    qapp.processEvents()
    assert message.view is not None and message.view.height() > 0
    widths = []
    heights = []
    block = message.view.document().begin()
    while block.isValid():
        iterator = block.begin()
        while not iterator.atEnd():
            fragment = iterator.fragment()
            fmt = fragment.charFormat()
            if fragment.isValid() and fmt.isImageFormat():
                image_fmt = QTextImageFormat(fmt.toImageFormat())
                widths.append(image_fmt.width())
                heights.append(image_fmt.height())
            iterator += 1
        block = block.next()
    assert len(widths) == 1
    assert 40 < widths[0] < 800
    assert heights[0] == pytest.approx(widths[0] * 200 / 800)

def test_panel_reports_missing_cli(qapp, monkeypatch):
    from vfmstudio.gui.docks.agent_chat import AgentChatPanel

    panel = AgentChatPanel()
    monkeypatch.setattr(panel, "_resolve_cli", lambda: None)
    assert panel.send("hello") is False
    assert panel.messages()[-1].role == "error"
    assert not panel.busy


def test_main_window_assistant_dock_toggles(qapp):
    from PySide6.QtWidgets import QToolButton

    from vfmstudio.gui.main_window import MainWindow

    window = MainWindow()
    window.show()
    qapp.processEvents()
    assert window.dock_agent.isHidden()
    toggle = window.dock_agent.toggleViewAction()
    texts = [button.text() for button in window._assistant_bar.findChildren(QToolButton)]
    assert "Assistant" in texts
    toggle.trigger()
    qapp.processEvents()
    assert window.dock_agent.isVisible()
    context = window._agent_context()
    assert "Active module" in context and "Experiments" in context
    assert window._agent_workdir().is_dir()
    toggle.trigger()
    qapp.processEvents()
    assert window.dock_agent.isHidden()
    window.session.dirty = False
    window.close()


def test_main_window_assistant_is_full_height_right_column(qapp):
    from PySide6.QtCore import Qt

    from vfmstudio.gui.main_window import LAYOUT_VERSION, MainWindow

    window = MainWindow()
    window.resize(1400, 900)
    window.show()
    qapp.processEvents()
    assert window.corner(Qt.Corner.BottomRightCorner) == Qt.DockWidgetArea.BottomDockWidgetArea
    assert not window.restoreState(window.saveState(0), LAYOUT_VERSION)

    window.dock_agent.toggleViewAction().trigger()
    for _ in range(3):
        qapp.processEvents()

    agent = window.dock_agent.geometry()
    props = window.dock_props.geometry()
    log = window.dock_log.geometry()
    assert window.corner(Qt.Corner.BottomRightCorner) == Qt.DockWidgetArea.RightDockWidgetArea
    assert agent.left() >= props.right()
    assert agent.right() >= window.centralWidget().geometry().right()
    assert log.right() < agent.left()
    assert agent.bottom() >= log.bottom() - 2
    assert agent.width() >= 320

    window.dock_agent.toggleViewAction().trigger()
    qapp.processEvents()
    assert window.corner(Qt.Corner.BottomRightCorner) == Qt.DockWidgetArea.BottomDockWidgetArea
    window.session.dirty = False
    window.close()
