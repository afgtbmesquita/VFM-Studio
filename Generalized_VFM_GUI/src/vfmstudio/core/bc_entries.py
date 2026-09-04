"""Build one BoundaryEntry per selected DOF for the Add-BC dialog.

The solver stores each constrained DOF as its own group.  The dialog lets
the user tick several DOFs at once; this module expands that choice into
the same one-entry-per-DOF list the model tree and FGLS already expect.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from vfmstudio.core.document import BoundaryEntry

DOF_ORDER = ("W", "W_X", "W_Y", "W_XY", "U", "V")
EDGES = ("top", "bottom", "left", "right")


def default_dofs(kind: str) -> tuple[str, ...]:
    if str(kind) == "external":
        return ("W_Y",)
    return ("W",)


def ordered_dofs(dofs: Iterable[str]) -> List[str]:
    """De-duplicate and keep Kirchhoff order, then any extras."""
    wanted = []
    seen = set()
    raw = [str(item).strip().upper().replace(" ", "_") for item in dofs]
    for name in DOF_ORDER:
        if name in raw and name not in seen:
            wanted.append(name)
            seen.add(name)
    for name in raw:
        if name and name not in seen:
            wanted.append(name)
            seen.add(name)
    return wanted


def clean_node_ids(nodes: Optional[Sequence[int]]) -> List[int]:
    if not nodes:
        return []
    cleaned = []
    seen = set()
    for item in nodes:
        try:
            index = int(item)
        except (TypeError, ValueError):
            continue
        if index < 0 or index in seen:
            continue
        seen.add(index)
        cleaned.append(index)
    return cleaned


def format_nodes_short(nodes: Sequence[int], limit: int = 6) -> str:
    ids = [int(item) for item in nodes]
    if not ids:
        return "nodes"
    shown = ", ".join(str(item) for item in ids[:limit])
    if len(ids) > limit:
        shown += f", … (+{len(ids) - limit})"
    return shown


def bc_label(
    dof: str,
    *,
    edge: Optional[str] = None,
    nodes: Optional[Sequence[int]] = None,
    prefix: str = "",
) -> str:
    text = str(prefix or "").strip()
    if text:
        if dof and dof not in text.split():
            return f"{text}  {dof}"
        return text
    if edge:
        return f"{dof} @{edge}"
    ids = clean_node_ids(nodes)
    if ids:
        return f"{dof} @{format_nodes_short(ids)}"
    return str(dof)


def make_boundary_entries(
    kind: str,
    dofs: Iterable[str],
    *,
    edge: Optional[str] = None,
    nodes: Optional[Sequence[int]] = None,
    label: str = "",
    load_type: str = "distributed",
) -> List[BoundaryEntry]:
    """One ``BoundaryEntry`` per DOF.  Edge and node list are mutually exclusive."""
    selected = ordered_dofs(dofs)
    if not selected:
        raise ValueError("Select at least one DOF.")
    side = str(edge or "").strip().lower() or None
    node_ids = clean_node_ids(nodes)
    if side and side not in EDGES:
        raise ValueError(f"Unknown edge {edge!r}.")
    if side:
        node_ids = []
    elif not node_ids:
        raise ValueError("Choose an edge or at least one mesh node.")

    entries: List[BoundaryEntry] = []
    prefix = str(label or "").strip()
    for dof in selected:
        payload = {
            "label": bc_label(dof, edge=side, nodes=node_ids, prefix=prefix),
            "dof": dof,
            "edge": side,
            "nodes": list(node_ids) if node_ids else None,
            "value": 0.0 if kind != "external" else None,
        }
        if kind == "external":
            payload["loadType"] = str(load_type or "distributed")
            if side in {"top", "bottom"}:
                payload["sortDirection"] = "x"
            elif side in {"left", "right"}:
                payload["sortDirection"] = "y"
        entries.append(BoundaryEntry.model_validate(payload))
    return entries
