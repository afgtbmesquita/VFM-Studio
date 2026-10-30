"""Drive the local Claude Code CLI in headless (``--print``) mode. Qt-free."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from vfmstudio.agents.registry import AgentSpec

ENV_CLAUDE_CLI = "VFMSTUDIO_CLAUDE_CLI"

_SUMMARY_KEYS = ("file_path", "path", "notebook_path", "pattern", "command", "url", "query", "description")


def _version_key(path: Path) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", path.name))


def _desktop_bundle_roots() -> list[Path]:
    """``claude-code`` folders shipped with the Claude desktop app (plain and MSIX installs)."""
    roots: list[Path] = []
    appdata = os.environ.get("APPDATA")
    if appdata:
        roots.append(Path(appdata) / "Claude" / "claude-code")
    local = os.environ.get("LOCALAPPDATA")
    if local:
        packages = Path(local) / "Packages"
        try:
            for package in packages.glob("Claude_*"):
                roots.append(package / "LocalCache" / "Roaming" / "Claude" / "claude-code")
        except OSError:
            pass
    return roots


def _bundled_executables() -> list[Path]:
    found: list[Path] = []
    for root in _desktop_bundle_roots():
        if not root.is_dir():
            continue
        versions = [item for item in root.iterdir() if item.is_dir()]
        for version in sorted(versions, key=_version_key, reverse=True):
            found.extend(sorted(version.glob("claude.exe")))
            found.extend(sorted(version.glob("*/claude.exe")))
    return found


def _standard_locations() -> list[Path]:
    home = Path.home()
    if sys.platform == "win32":
        candidates = [
            home / ".local" / "bin" / "claude.exe",
            home / ".claude" / "local" / "claude.exe",
        ]
        appdata = os.environ.get("APPDATA")
        if appdata:
            candidates.append(Path(appdata) / "npm" / "claude.cmd")
        return candidates
    return [
        home / ".local" / "bin" / "claude",
        home / ".claude" / "local" / "claude",
        Path("/usr/local/bin/claude"),
        Path("/opt/homebrew/bin/claude"),
    ]


def find_claude_cli(preferred: Optional[str | Path] = None) -> Optional[Path]:
    """First usable CLI: explicit choice, env override, PATH, standard installs, desktop bundle."""
    explicit = [preferred, os.environ.get(ENV_CLAUDE_CLI, "").strip() or None]
    for item in explicit:
        if item and Path(item).expanduser().is_file():
            return Path(item).expanduser()
    on_path = shutil.which("claude")
    if on_path:
        return Path(on_path)
    for candidate in _standard_locations():
        if candidate.is_file():
            return candidate
    bundled = _bundled_executables()
    return bundled[0] if bundled else None


def build_arguments(
    spec: AgentSpec,
    *,
    resume_session: Optional[str] = None,
    extra_system_prompt: str = "",
) -> list[str]:
    """CLI arguments for one turn. The prompt itself is written to stdin."""
    args = [
        "--print",
        "--output-format",
        "stream-json",
        "--verbose",
        "--include-partial-messages",
    ]
    if resume_session:
        args += ["--resume", resume_session]
    if spec.model:
        args += ["--model", spec.model]
    if spec.permission_mode:
        args += ["--permission-mode", spec.permission_mode]
    if spec.tools is not None:
        args += ["--tools", *(spec.tools or ("",))]
    if spec.allowed_tools:
        args += ["--allowedTools", *spec.allowed_tools]
    if spec.disallowed_tools:
        args += ["--disallowedTools", *spec.disallowed_tools]
    for directory in spec.add_dirs:
        args += ["--add-dir", str(directory)]
    args += list(spec.extra_args)
    system_prompt = "\n\n".join(part for part in (spec.system_prompt, extra_system_prompt) if part)
    if system_prompt:
        args += ["--append-system-prompt", system_prompt]
    return args


def build_command(executable: Path, arguments: Sequence[str]) -> list[str]:
    if sys.platform == "win32" and executable.suffix.lower() in {".cmd", ".bat"}:
        return ["cmd.exe", "/c", str(executable), *arguments]
    return [str(executable), *arguments]


@dataclass
class AgentEvent:
    """One normalized item from the CLI stream.

    kinds: ``session``, ``thinking``, ``text_delta``, ``text``, ``tool_use``,
    ``tool_result``, ``result``.
    """

    kind: str
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)


def summarize_tool_input(tool_input: Any, limit: int = 90) -> str:
    if not isinstance(tool_input, dict):
        return ""
    for key in _SUMMARY_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            text = " ".join(value.split())
            return text if len(text) <= limit else text[: limit - 1] + "…"
    return ""


def parse_stream_line(line: str) -> list[AgentEvent]:
    """Translate one ``stream-json`` line. Subagent traffic and unknown types are dropped."""
    line = line.strip()
    if not line:
        return []
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    kind = payload.get("type")
    if kind == "system" and payload.get("subtype") == "init":
        return [
            AgentEvent(
                "session",
                data={
                    "session_id": payload.get("session_id"),
                    "model": payload.get("model"),
                    "cwd": payload.get("cwd"),
                },
            )
        ]
    if kind == "result":
        return [
            AgentEvent(
                "result",
                text=str(payload.get("result") or ""),
                data={
                    "is_error": bool(payload.get("is_error")),
                    "subtype": payload.get("subtype"),
                    "session_id": payload.get("session_id"),
                    "cost_usd": payload.get("total_cost_usd"),
                    "num_turns": payload.get("num_turns"),
                    "permission_denials": payload.get("permission_denials") or [],
                },
            )
        ]
    if payload.get("parent_tool_use_id"):
        return []
    if kind == "stream_event":
        event = payload.get("event") or {}
        etype = event.get("type")
        if etype == "content_block_start":
            block = event.get("content_block") or {}
            if block.get("type") == "thinking":
                return [AgentEvent("thinking")]
        elif etype == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta" and delta.get("text"):
                return [AgentEvent("text_delta", text=delta["text"])]
        return []
    if kind == "assistant":
        events: list[AgentEvent] = []
        for block in (payload.get("message") or {}).get("content") or []:
            btype = block.get("type")
            if btype == "tool_use":
                events.append(
                    AgentEvent(
                        "tool_use",
                        text=summarize_tool_input(block.get("input")),
                        data={"name": block.get("name"), "id": block.get("id")},
                    )
                )
            elif btype == "text" and block.get("text"):
                events.append(AgentEvent("text", text=block["text"]))
        return events
    if kind == "user":
        events = []
        for block in (payload.get("message") or {}).get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                events.append(
                    AgentEvent(
                        "tool_result",
                        data={"id": block.get("tool_use_id"), "is_error": bool(block.get("is_error"))},
                    )
                )
        return events
    return []


def _child_environment() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("CLAUDECODE", None)
    return env


class ClaudeRun:
    """One CLI turn in a background thread. Callbacks fire on that thread."""

    def __init__(
        self,
        command: Sequence[str],
        prompt: str,
        cwd: Optional[Path],
        on_event: Callable[[AgentEvent], None],
        on_stderr: Callable[[str], None],
        on_exit: Callable[[int, bool], None],
    ) -> None:
        self.command = list(command)
        self.prompt = prompt
        self.cwd = cwd
        self._on_event = on_event
        self._on_stderr = on_stderr
        self._on_exit = on_exit
        self._proc: Optional[subprocess.Popen] = None
        self._stopped = False
        self._thread: Optional[threading.Thread] = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="claude-cli", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopped = True
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        if sys.platform == "win32":
            # Tool subprocesses (Bash, etc.) would outlive a plain kill.
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
                check=False,
            )
        else:
            proc.kill()

    def wait(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        try:
            self._proc = subprocess.Popen(
                self.command,
                cwd=str(self.cwd) if self.cwd else None,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=_child_environment(),
                creationflags=flags,
            )
        except OSError as exc:
            self._on_stderr(f"Could not start Claude CLI: {exc}")
            self._on_exit(-1, False)
            return
        proc = self._proc
        stderr_thread = threading.Thread(target=self._pump_stderr, args=(proc,), daemon=True)
        stderr_thread.start()
        try:
            assert proc.stdin is not None
            proc.stdin.write(self.prompt.encode("utf-8"))
            proc.stdin.close()
        except OSError:
            pass
        assert proc.stdout is not None
        for raw in proc.stdout:
            for event in parse_stream_line(raw.decode("utf-8", errors="replace")):
                self._on_event(event)
        code = proc.wait()
        stderr_thread.join(timeout=2.0)
        self._on_exit(int(code), self._stopped)

    def _pump_stderr(self, proc: subprocess.Popen) -> None:
        assert proc.stderr is not None
        for raw in proc.stderr:
            text = raw.decode("utf-8", errors="replace").rstrip()
            if text:
                self._on_stderr(text)


def denied_tool_names(denials: Iterable[Any]) -> list[str]:
    names: list[str] = []
    for item in denials:
        name = item.get("tool_name") if isinstance(item, dict) else None
        if name and name not in names:
            names.append(str(name))
    return names
