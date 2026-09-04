"""Parse viewport pick hits without depending on VTK."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from vfmstudio.gui.session import Selection

_BC_GROUPS = {"prescribed", "linking", "external"}
_EDGES = ("left", "right", "top", "bottom")


def bc_actor_name(experiment_id: int, group: str, index: int) -> str:
    return f"bc:{int(experiment_id)}:{group}:{int(index)}"


def parse_bc_actor_name(name: str) -> Optional[Selection]:
    token = str(name or "")
    if token.endswith("-label"):
        token = token[: -len("-label")]
    parts = token.split(":")
    if len(parts) != 4 or parts[0] != "bc":
        return None
    group = parts[2]
    if group not in _BC_GROUPS:
        return None
    try:
        experiment_id = int(parts[1])
        index = int(parts[3])
    except ValueError:
        return None
    return Selection(
        kind="bc",
        experiment_id=experiment_id,
        bc_kind=group,
        bc_index=index,
    )


def nearest_mesh_edge(point_xy: Sequence[float], mesh_xyz: np.ndarray) -> str:
    xyz = np.asarray(mesh_xyz, dtype=float)
    if xyz.size == 0:
        return "bottom"
    x, y = float(point_xy[0]), float(point_xy[1])
    xmin, ymin = float(np.min(xyz[:, 0])), float(np.min(xyz[:, 1]))
    xmax, ymax = float(np.max(xyz[:, 0])), float(np.max(xyz[:, 1]))
    dist = {
        "left": abs(x - xmin),
        "right": abs(x - xmax),
        "bottom": abs(y - ymin),
        "top": abs(y - ymax),
    }
    return min(dist, key=dist.get)


def nearest_node_index(point_xy: Sequence[float], mesh_xyz: np.ndarray) -> int:
    xyz = np.asarray(mesh_xyz, dtype=float)
    if xyz.size == 0:
        return 0
    delta = xyz[:, :2] - np.asarray(point_xy[:2], dtype=float)
    return int(np.argmin(np.einsum("ij,ij->i", delta, delta)))


def first_bc_on_edge(experiment, edge: str) -> Optional[Tuple[str, int]]:
    token = str(edge or "").lower()
    for kind in ("prescribed", "linking", "external"):
        entries = getattr(experiment, kind, None) or []
        for index, entry in enumerate(entries):
            if str(getattr(entry, "edge", "") or "").lower() == token:
                return kind, index
    return None
