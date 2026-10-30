"""Human Interpreter: one fresh Claude CLI pass over specialist output.

Not registered in the assistant combo. The chat runs it after the DIC data agent
or the model build agent finishes a turn.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

from vfmstudio.agents.registry import AgentSpec

_RULES_PATH = Path(__file__).with_name("writing_rules.md")

# Keys produced by the DIC figure renderer. The caption is what the model may say
# the picture is; the file itself is never redrawn.
FIGURE_CAPTIONS = {
    "fields": "displacement and curvature increments",
    "response": "load response",
    "noise_maps": "per-facet noise, measured and modelled",
    "noise_profile": "noise against distance to the boundary",
    "quality": "correlation quality and missing facets",
    "filter_sweep": "curvature filter sweep",
}


def writing_rules_text() -> str:
    """The writing rules shipped next to this module."""
    return _RULES_PATH.read_text(encoding="utf-8")


def figure_index(
    figures: Optional[Mapping[str, Mapping[str, str]]] = None,
) -> dict[str, tuple[str, str]]:
    """Map ``{{figure:Test/key}}`` slots to ``(relative path, caption)``.

    Empty paths are omitted. Insertion order follows the mapping.
    """
    index: dict[str, tuple[str, str]] = {}
    for test, items in (figures or {}).items():
        if not isinstance(items, Mapping):
            continue
        for key, rel in items.items():
            if not isinstance(rel, str) or not rel.strip():
                continue
            slot = f"{test}/{key}"
            caption = f"{test}: {FIGURE_CAPTIONS.get(str(key), str(key).replace('_', ' '))}"
            index[slot] = (rel.replace("\\", "/"), caption)
    return index


_JOB = r"""
You are the Human Interpreter of VFM Studio. You do not analyse tests and you do not build models. You rewrite what the specialist agent and the measured data already said, so a person can read it in the chat.

The authors of the material are the specialist agent and the measured report. Their conclusions, their degree of certainty, and anything they left undecided stay as they wrote them. Accepting or rejecting a test stays with the user. An unresolved model question stays unresolved. Do not add a finding, a recommendation, or an explanation that is not already in the source.

Do not compute, estimate, or introduce a number that is not copied from the source text. If you keep a number, copy it character for character. If you keep a table, copy that Markdown table exactly. Cut repetition, process talk, and a second statement of the same fact. Do not cut a conclusion, a caveat, or a number you still rely on.

Use LaTeX only for a symbol or a short expression, written as \(\kappa_{xx}\) or \[ ... \]. Stay inside matplotlib's mathtext subset: Greek letters, subscripts, superscripts, \frac, \mathrm, \cdot. No align environments and no dollar signs.

Figures are already drawn. Refer to one only by copying a slot from the list in the user message, on its own line, exactly as given: {{figure:Test/key}}. Do not invent a path, a slot, or a description of a picture that is not in the list. If the list says there are no figures, do not insert one.

Write Markdown. Use a heading only when the reply has more than one subject. Match the length to what the user needs to see.

The writing rules below govern the voice. Here the author is the specialist and the data, not a person with a private style. Apply them to meaning, certainty, and plainness.
"""


def _system_prompt() -> str:
    return _JOB + "\n" + writing_rules_text().strip() + "\n"


HUMAN_INTERPRETER = AgentSpec(
    agent_id="human_interpreter",
    name="Human Interpreter",
    description="Rewrites specialist output into the message shown in the chat.",
    system_prompt=_system_prompt(),
    tools=(),
    # Sonnet at medium effort. This pass only rewrites text the specialist already wrote.
    # The headless CLI is --restricted, so it does not inherit ~/.claude/settings.json
    # and would otherwise start on the CLI default (Opus) at high effort.
    model="sonnet",
    disallowed_tools=(
        "Bash", "PowerShell", "Edit", "Write", "NotebookEdit", "WebFetch", "WebSearch", "Task",
        "Read", "Glob",
    ),
    extra_args=(
        "--restricted",
        "--strict-mcp-config",
        "--safe-mode",
        "--effort",
        "medium",
        "--no-session-persistence",
    ),
    gui_context=False,
)


def build_interpreter_prompt(
    *,
    specialist_text: str,
    report_text: str = "",
    figures: Optional[Mapping[str, Mapping[str, str]]] = None,
    follow_up: bool = False,
) -> str:
    """User message for one interpreter turn. The system prompt carries the rules.

    ``report_text`` is accepted so callers can keep passing the measured report, but it is
    not sent. The specialist reply already carries the numbers, and pasting the report
    made this pass rewrite it.
    """
    del report_text, follow_up
    parts: list[str] = [
        "Rewrite the specialist reply below so a person can read it. "
        "Keep it shorter than the reply. Do not add a section the reply does not have. "
        "Do not restate the report.",
        "## Specialist reply\n\n" + specialist_text.strip(),
    ]
    lines = ["## Figures you may cite"]
    index = figure_index(figures)
    if not index:
        lines.append("None. Do not insert a figure.")
    else:
        for slot, (path, caption) in index.items():
            token = "{{figure:" + slot + "}}"
            lines.append(f"- {token} — {caption} ({path})")
        lines.append("Use a figure only by copying a slot from this list.")
    parts.append("\n".join(lines))
    parts.append(
        "Write the message the user should read, in Markdown. "
        "Copy numbers and any table you keep exactly."
    )
    return "\n\n".join(parts) + "\n"
