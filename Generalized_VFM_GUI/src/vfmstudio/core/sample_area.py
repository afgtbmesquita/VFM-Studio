"""Estimate VFM ``sample_area_total`` from a DIC point cloud.

Identification weights every measurement by ``A / n``.  For a structured
facet lattice that means ``A = n · dx · dy`` -- including tabbed plates,
L / C outlines, and interior holes -- because missing points are simply
absent from ``n``.  A bounding-box default is wrong as soon as the cloud
does not fill its hull.

The estimator itself now lives in the SOLVER, as
``vfm_data_io.estimate_measurement_area``, and this module is a thin
adapter.  It used to carry its own Delaunay implementation (reuse the
viewport's void-triangle filter, take twice the median kept-triangle area
as the facet cell, multiply by ``n``); that was correct and is preserved in
git history, but it duplicated an estimator the solver needs anyway for the
headless API and ``run_vfm.py``.  One implementation, one set of behaviours
to reason about.

The solver version recovers the two shortest independent lattice vectors
and takes ``n · |v1 x v2|``.  It is exact on the same cases the Delaunay
version handled -- anisotropic pitch (``dx != dy``, where an occupancy
``n · h²`` estimate is wrong because nearest-neighbour ``h`` is
``min(dx, dy)``), in-plane rotation, and holes / concavities, which only
change ``n``.  There is deliberately NO fallback path: if the cloud is not
a lattice the solver raises rather than guessing, because a silently wrong
area rescales every identified stiffness.  ``tests/test_sample_area.py``
pins the behaviour to 1e-9 relative.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from vfmstudio.core.contour_mesh import finite_xy
from vfmstudio.core.paths import ensure_core_on_path


def _solver_estimator():
    """Import the solver's estimator, putting the solver on sys.path first."""
    ensure_core_on_path()
    from vfm_data_io import estimate_measurement_area

    return estimate_measurement_area


def estimate_sample_area(x, y) -> Optional[float]:
    """Return ``n · dx · dy`` for a DIC cloud, or ``None`` if undefined.

    ``None`` means "could not measure it" and leaves the caller on whatever
    default it has.
    """
    xs, ys, _ = finite_xy(x, y)
    if xs.size < 4:
        return None
    try:
        area, _ = _solver_estimator()(np.column_stack([xs, ys]))
    except Exception:
        return None
    return float(area) if np.isfinite(area) and area > 0.0 else None
