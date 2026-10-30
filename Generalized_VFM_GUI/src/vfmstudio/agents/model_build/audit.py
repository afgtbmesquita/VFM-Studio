"""Deterministic audit of the model build agent's replies.

Numbers have to come from the report, the facts, the BC training cards, or the
data agent records in the build folder. The agent may ask only the questions
the build left open, and it does not accept or reject a test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from vfmstudio.agents.dic_data.audit import decision_wording, source_values, unsupported_numbers


def audit_reply(reply: str, workdir: Optional[Path]) -> Optional[str]:
    """A short note for the chat, or None when the reply passes."""
    if not reply.strip() or workdir is None:
        return None
    texts = []
    unresolved: list = []
    sources = [Path(workdir) / name for name in ("report.md", "facts.json", "bc_reference.json")]
    sources += sorted((Path(workdir) / "data_agent").glob("*/facts.json"))
    sources += sorted((Path(workdir) / "data_agent").glob("*/report.md"))
    for path in sources:
        if path.is_file():
            texts.append(path.read_text(encoding="utf-8"))
    facts_path = Path(workdir) / "facts.json"
    if facts_path.is_file():
        try:
            unresolved = list(json.loads(facts_path.read_text(encoding="utf-8")).get("unresolved") or [])
        except (OSError, json.JSONDecodeError):
            unresolved = []
    notes = []
    numbers = unsupported_numbers(reply, source_values(texts)) if texts else []
    if numbers:
        listed = ", ".join(numbers[:12])
        if len(numbers) > 12:
            listed += f" and {len(numbers) - 12} more"
        notes.append(
            f"Audit: {len(numbers)} number(s) in this reply are not in the model facts ({listed}). "
            "They are the agent's own estimates, not measurements."
        )
    words = decision_wording(reply)
    if words:
        notes.append(
            "Audit: the reply uses decision wording (" + ", ".join(words) + "). "
            "Keeping or discarding a test is your decision."
        )
    if not unresolved and "?" in reply:
        notes.append(
            "Audit: the reply asks a question, but the build left no input unresolved."
        )
    return "\n".join(notes) or None
