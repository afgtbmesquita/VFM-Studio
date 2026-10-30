"""Deterministic audit of the data agent's replies against its rules.

* Every number the agent states should exist in report.md or facts.json (rule 5).
  Numbers that do not are listed so the user knows they are readings of a figure
  or the agent's own estimate, not measurements.
* Decision wording (accept, reject, discard, …) outside a negation is listed (rule 1).

The audit never edits the reply; it only adds a note under it.
"""

from __future__ import annotations

import bisect
import math
import re
from pathlib import Path
from typing import Iterable, Optional

_NUMBER = re.compile(r"(?<![\w.,])[-+−]?((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?)(?![\w,]\d)(%?)")
_JSON_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_GROUPED = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?")
_DECISION = re.compile(
    r"\b(accept(?:ed|able)?|reject(?:ed)?|discard(?:ed)?|invalid|valid test|good test|bad test|"
    r"should (?:not )?be used|usable|unusable|pass(?:es|ed)? the test|fail(?:s|ed)? the test)\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(r"\b(not|never|no|nor|n't|cannot|neither|without|whether)\b", re.IGNORECASE)
SMALL_INTEGER = 20


def source_values(texts: Iterable[str]) -> list[float]:
    values: set[float] = set()
    for text in texts:
        tokens = [match.group() for match in _JSON_NUMBER.finditer(text)]
        tokens += [match.group().replace(",", "") for match in _GROUPED.finditer(text)]
        for token in tokens:
            try:
                value = abs(float(token))
            except ValueError:
                continue
            if math.isfinite(value):
                values.add(value)
    return sorted(values)


def _resolution(token: str) -> float:
    mantissa = token.lower().split("e")[0]
    exponent = int(token.lower().split("e")[1]) if "e" in token.lower() else 0
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    return 0.5 * 10.0 ** (exponent - decimals)


def _supported(value: float, resolution: float, sources: list[float]) -> bool:
    """Is some source value equal to ``value`` at the precision it was written with?"""
    low = bisect.bisect_left(sources, value - resolution * (1 + 1e-9))
    high = bisect.bisect_right(sources, value + resolution * (1 + 1e-9))
    return high > low


def unsupported_numbers(reply: str, sources: list[float]) -> list[str]:
    found: list[str] = []
    for match in _NUMBER.finditer(reply):
        token, percent = match.group(1).replace(",", ""), match.group(2)
        value = float(token)
        if not percent and "." not in token and "e" not in token.lower() and value <= SMALL_INTEGER:
            continue
        resolution = _resolution(token)
        if percent:
            supported = _supported(value / 100.0, resolution / 100.0, sources) or _supported(value, resolution, sources)
        else:
            supported = _supported(value, resolution, sources)
        if not supported:
            shown = token + percent
            if shown not in found:
                found.append(shown)
    return found


def decision_wording(reply: str) -> list[str]:
    hits: list[str] = []
    for match in _DECISION.finditer(reply):
        before = reply[max(0, match.start() - 60) : match.start()]
        sentence = re.split(r"[.!?\n]", before)[-1]
        if _NEGATION.search(sentence):
            continue
        word = match.group(0).lower()
        if word not in hits:
            hits.append(word)
    return hits


def audit_reply(reply: str, workdir: Optional[Path]) -> Optional[str]:
    """A short note for the chat, or None when the reply passes both checks."""
    if not reply.strip():
        return None
    texts = []
    for name in ("report.md", "facts.json"):
        path = Path(workdir) / name if workdir is not None else None
        if path is not None and path.is_file():
            texts.append(path.read_text(encoding="utf-8"))
    notes = []
    numbers = unsupported_numbers(reply, source_values(texts)) if texts else []
    if numbers:
        listed = ", ".join(numbers[:12]) + (f" and {len(numbers) - 12} more" if len(numbers) > 12 else "")
        notes.append(
            f"Audit: {len(numbers)} number(s) in this reply are not in the measured facts ({listed}). "
            "They are readings of a figure or the agent's own estimates, not measurements."
        )
    words = decision_wording(reply)
    if words:
        notes.append(
            "Audit: the reply uses decision wording (" + ", ".join(words) + "). "
            "Accepting or rejecting a test is your decision; the agent should only present facts."
        )
    return "\n".join(notes) or None
