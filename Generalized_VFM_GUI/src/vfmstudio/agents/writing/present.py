"""Turn Human Interpreter Markdown into something the chat can lay out.

Qt-free. Figure slots become image links, mathtext spans become PNGs, and a
table with a broken column count is replaced or flattened.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from vfmstudio.agents.writing.interpreter import figure_index

_SLOT = re.compile(r"\{\{figure:([^}]+)\}\}")
_MATH = re.compile(r"\\\[(.+?)\\\]|\\\((.+?)\\\)", re.DOTALL)
_SEP_CELL = re.compile(r":?-{3,}:?")


@dataclass(frozen=True)
class _Table:
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    raw: str
    ragged: bool
    end: int


def prepare_markdown(
    text: str,
    *,
    workdir: Optional[Path] = None,
    figures: Optional[Mapping[str, Mapping[str, str]]] = None,
    sources: str = "",
) -> str:
    """Expand figures, draw equations, and repair ragged tables."""
    rendered = _expand_figures(text, figures, workdir)
    rendered = _render_math(rendered, workdir)
    rendered = _repair_tables(rendered, sources)
    return rendered


def _expand_figures(
    text: str,
    figures: Optional[Mapping[str, Mapping[str, str]]],
    workdir: Optional[Path],
) -> str:
    index = figure_index(figures)

    def replace(match: re.Match[str]) -> str:
        item = index.get(match.group(1).strip())
        if item is None:
            return ""
        path, caption = item
        if workdir is not None and not (Path(workdir) / path).is_file():
            return ""
        return f"\n\n![{caption}]({path})\n\n"

    return _SLOT.sub(replace, text)


def _render_math(text: str, workdir: Optional[Path]) -> str:
    if not _MATH.search(text):
        return text
    folder = Path(workdir) / ".interpreter" if workdir is not None else None
    counter = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal counter
        expr = (match.group(1) if match.group(1) is not None else match.group(2)) or ""
        expr = expr.strip()
        if not expr:
            return ""
        if folder is None:
            return expr
        counter += 1
        path = folder / f"eq_{counter}.png"
        if not _math_png(expr, path):
            return expr
        return f"![equation]({path.relative_to(workdir).as_posix()})"

    return _MATH.sub(replace, text)


def _math_png(expr: str, path: Path) -> bool:
    """Draw one mathtext expression. False leaves the caller with plain text."""
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        path.parent.mkdir(parents=True, exist_ok=True)
        figure = Figure(figsize=(1.2, 0.4))
        FigureCanvasAgg(figure)
        figure.patch.set_facecolor("white")
        figure.text(0.5, 0.5, f"${expr}$", fontsize=13, ha="center", va="center")
        figure.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    except (OSError, ValueError, RuntimeError):
        return False
    return path.is_file() and path.stat().st_size > 0


def _repair_tables(text: str, sources: str) -> str:
    if "|" not in text:
        return text
    canonical = {
        table.header: table.raw
        for table in _parse_tables(sources)
        if not table.ragged
    }
    lines = text.splitlines()
    pieces: list[str] = []
    fenced = False
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped.startswith("```"):
            fenced = not fenced
            pieces.append(lines[index])
            index += 1
            continue
        if not fenced:
            table = _table_at(lines, index)
            if table is not None:
                if not table.ragged:
                    pieces.append(table.raw)
                elif table.header in canonical:
                    pieces.append(canonical[table.header])
                else:
                    pieces.append(_table_as_list(table))
                index = table.end
                continue
        pieces.append(lines[index])
        index += 1
    ending = "\n" if text.endswith("\n") else ""
    return "\n".join(pieces) + ending


def _parse_tables(text: str) -> list[_Table]:
    lines = text.splitlines()
    found: list[_Table] = []
    fenced = False
    index = 0
    while index < len(lines):
        if lines[index].strip().startswith("```"):
            fenced = not fenced
            index += 1
            continue
        if not fenced:
            table = _table_at(lines, index)
            if table is not None:
                found.append(table)
                index = table.end
                continue
        index += 1
    return found


def _table_at(lines: list[str], start: int) -> Optional[_Table]:
    if start + 1 >= len(lines):
        return None
    if not _is_row(lines[start]) or not _is_separator_line(lines[start + 1]):
        return None
    end = start + 2
    while end < len(lines) and _is_row(lines[end]):
        end += 1
    block = lines[start:end]
    rows = tuple(tuple(_split_row(line)) for line in block)
    header = rows[0]
    width = len(header)
    ragged = any(len(row) != width for row in rows)
    return _Table(
        header=header,
        rows=rows,
        raw="\n".join(block),
        ragged=ragged,
        end=end,
    )


def _table_as_list(table: _Table) -> str:
    header = table.rows[0]
    body = table.rows[2:]
    if not body:
        return " · ".join(cell for cell in header if cell)
    lines = []
    for row in body:
        pairs = []
        for index, cell in enumerate(row):
            name = header[index] if index < len(header) else f"Column {index + 1}"
            pairs.append(f"{name}: {cell}" if name else cell)
        lines.append("- " + "; ".join(pairs))
    return "\n".join(lines)


def _is_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2


def _is_separator_line(line: str) -> bool:
    if not _is_row(line):
        return False
    cells = _split_row(line)
    return bool(cells) and all(_SEP_CELL.fullmatch(cell) for cell in cells)


def _split_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]
