"""Export the active mesh / field to VTK via meshio (optional PyVista)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def export_mesh_vtk(
    path: Path,
    points: np.ndarray,
    quads: np.ndarray,
    point_data: Optional[Dict[str, np.ndarray]] = None,
) -> Path:
    import meshio

    cells = [("quad", np.asarray(quads, dtype=np.int64))]
    mesh = meshio.Mesh(
        points=np.asarray(points, dtype=float),
        cells=cells,
        point_data=point_data or {},
    )
    path = Path(path)
    mesh.write(path)
    return path
