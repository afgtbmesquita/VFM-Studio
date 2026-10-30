"""Claude CLI agents for the Assistant panel (Qt-free)."""

from vfmstudio.agents.claude_cli import (
    AgentEvent,
    ClaudeRun,
    build_arguments,
    build_command,
    find_claude_cli,
    parse_stream_line,
)
from vfmstudio.agents.registry import AgentSpec, available_agents, get_agent, register_agent

__all__ = [
    "AgentEvent",
    "AgentSpec",
    "ClaudeRun",
    "available_agents",
    "build_arguments",
    "build_command",
    "find_claude_cli",
    "get_agent",
    "parse_stream_line",
    "register_agent",
]
