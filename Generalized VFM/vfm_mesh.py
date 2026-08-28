# =============================================================================
#  plate / vfm_mesh.py
# -----------------------------------------------------------------------------
#  Analysis-type aware plate mesh and strain-displacement operator.
#
#  The mesh stores a single ``analysis_type`` string driving:
#
#    * ``n_dof_per_node``:
#         'in_plane'      -> 2     DOF order [u, v]
#         'out_of_plane'  -> 4     DOF order [w, w_x, w_y, w_xy]
#         'both'          -> 6     DOF order [u, v, w, w_x, w_y, w_xy]
#
#    * Per-mode strain-displacement operator (replaces the old, always-6x24
#      "plate B-matrix"):
#         'in_plane'      -> (N, 3, 8)   Q1 gradients only
#         'out_of_plane'  -> (N, 3, 16)  BFS Hessians only
#         'both'          -> (N, 6, 24)  composite (Q1 rows 0-2, BFS rows 3-5)
#
#    * DOF-label resolver -- ``_resolve_dof_component`` rejects DOF labels
#      that are not legal for the active analysis type (e.g. ``W`` in
#      ``in_plane`` mode).
#
#  DOF orderings per element (4 nodes, CCW [BL, BR, TR, TL]):
#      'in_plane':     [u1, v1, u2, v2, u3, v3, u4, v4]
#      'out_of_plane': [w1, wx1, wy1, wxy1, ..., w4, wx4, wy4, wxy4]
#      'both':         [u1, v1, w1, wx1, wy1, wxy1, ..., wxy4]
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from skfem import MeshQuad

from vfm_data_io import BoundaryConditions, PointMeasurements
from vfm_user_inputs import (
    Bounds,
    DOF_PER_NODE_BY_ANALYSIS,
    MeshSettings,
    dof_per_node_for_analysis,
)


# =============================================================================
#  Data containers
# =============================================================================

@dataclass
class MeshPackage:
    """Bundle holding the mesh and all derived bookkeeping."""
    skfem_mesh: MeshQuad
    node_coords: np.ndarray          # (n_nodes, 2)
    element_nodes: np.ndarray        # (n_elements, 4) -- CCW: [BL, BR, TR, TL]
    n_nodes: int
    n_elements: int
    n_dof_per_node: int
    n_dofs: int                      # n_nodes * n_dof_per_node
    analysis_type: str

    nx: int
    ny: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    dx: float
    dy: float
    element_grid_index: np.ndarray   # (nx, ny) -- int
    x_nodes: np.ndarray
    y_nodes: np.ndarray
    element_dx: np.ndarray           # (n_elements,) side length per element
    element_dy: np.ndarray           # (n_elements,) side length per element


@dataclass
class MeshPruneReport:
    """Summary of :func:`prune_mesh_by_measurement_points`."""
    n_elements_before: int
    n_elements_after: int
    n_elements_removed: int
    n_nodes_before: int
    n_nodes_after: int
    n_nodes_removed: int


@dataclass
class PointMapping:
    element_ids: np.ndarray
    local_xy: np.ndarray
    mapped_mask: np.ndarray

    @property
    def n_mapped(self) -> int:
        return int(self.mapped_mask.sum())

    @property
    def n_unmapped(self) -> int:
        return int(self.mapped_mask.size - self.mapped_mask.sum())


@dataclass
class ConstraintSystem:
    """Container for the homogeneous KA constraint matrix C * u = 0."""
    C: sp.csr_matrix
    n_prescribed: int
    n_linking: int
    n_inactive: int
    inactive_dofs: np.ndarray
    n_constraints: int


# =============================================================================
#  Per-analysis-type DOF tables
# -----------------------------------------------------------------------------
#  Offsets inside each node's DOF block.  Used by:
#    - the strain-displacement operator (where to write each row),
#    - the BC resolver (which labels are legal).
# =============================================================================

_DOF_OFFSETS_BY_ANALYSIS: Dict[str, Dict[str, int]] = {
    "in_plane": {
        "U": 0,
        "V": 1,
    },
    "out_of_plane": {
        "W":    0,
        "W_X":  1,
        "W_Y":  2,
        "W_XY": 3,
    },
    "both": {
        "U":    0,
        "V":    1,
        "W":    2,
        "W_X":  3,
        "W_Y":  4,
        "W_XY": 5,
    },
}


def _dof_offsets(analysis_type: str) -> Dict[str, int]:
    key = str(analysis_type).strip().lower()
    if key not in _DOF_OFFSETS_BY_ANALYSIS:
        raise ValueError(
            f"Unsupported analysis_type {analysis_type!r}; expected one of "
            f"{tuple(_DOF_OFFSETS_BY_ANALYSIS.keys())}."
        )
    return _DOF_OFFSETS_BY_ANALYSIS[key]


# =============================================================================
#  Membrane Q1 shape functions  (kept for documentation / one-off use)
# =============================================================================

def q1_shape_functions(xi: float, eta: float
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(N, dN/dxi, dN/deta)`` for the bilinear Q1 element."""
    one_minus_xi = 1.0 - xi
    one_plus_xi = 1.0 + xi
    one_minus_eta = 1.0 - eta
    one_plus_eta = 1.0 + eta

    N = 0.25 * np.array([
        one_minus_xi * one_minus_eta,
        one_plus_xi * one_minus_eta,
        one_plus_xi * one_plus_eta,
        one_minus_xi * one_plus_eta,
    ])
    dN_dxi = 0.25 * np.array([
        -one_minus_eta,
         one_minus_eta,
         one_plus_eta,
        -one_plus_eta,
    ])
    dN_deta = 0.25 * np.array([
        -one_minus_xi,
        -one_plus_xi,
         one_plus_xi,
         one_minus_xi,
    ])
    return N, dN_dxi, dN_deta


# =============================================================================
#  1D cubic Hermite basis functions on [0, a]
# =============================================================================

def bfs_1d_hermite_batch(s: np.ndarray, a: float
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched evaluator of the four 1D Hermite cubics and their derivatives."""
    if a <= 0.0:
        raise ValueError(f"BFS 1D side length 'a' must be positive; got {a}.")

    s = np.asarray(s, dtype=float)
    t = s / a
    t2 = t * t
    t3 = t2 * t

    H = np.empty(s.shape + (4,), dtype=float)
    H[..., 0] =  2.0 * t3 - 3.0 * t2 + 1.0
    H[..., 1] =  a * (t3 - 2.0 * t2 + t)
    H[..., 2] = -2.0 * t3 + 3.0 * t2
    H[..., 3] =  a * (t3 - t2)

    dH = np.empty(s.shape + (4,), dtype=float)
    dH[..., 0] = (6.0 * t2 - 6.0 * t) / a
    dH[..., 1] =  3.0 * t2 - 4.0 * t + 1.0
    dH[..., 2] = (-6.0 * t2 + 6.0 * t) / a
    dH[..., 3] =  3.0 * t2 - 2.0 * t

    d2H = np.empty(s.shape + (4,), dtype=float)
    d2H[..., 0] = (12.0 * t - 6.0) / (a * a)
    d2H[..., 1] = (6.0 * t - 4.0) / a
    d2H[..., 2] = (-12.0 * t + 6.0) / (a * a)
    d2H[..., 3] = (6.0 * t - 2.0) / a

    return H, dH, d2H


# =============================================================================
#  BFS bookkeeping
# =============================================================================

_H00, _H10, _H01, _H11 = 0, 1, 2, 3

# (alpha_index, beta_index) for each of the 16 BFS shape functions in the
# canonical corner order [BL, BR, TR, TL] x [w, w_x, w_y, w_xy].
_BFS_ALPHA_INDEX = np.array([
    _H00, _H10, _H00, _H10,   # node 1 (BL)
    _H01, _H11, _H01, _H11,   # node 2 (BR)
    _H01, _H11, _H01, _H11,   # node 3 (TR)
    _H00, _H10, _H00, _H10,   # node 4 (TL)
], dtype=np.int64)
_BFS_BETA_INDEX = np.array([
    _H00, _H00, _H10, _H10,   # node 1 (BL)
    _H00, _H00, _H10, _H10,   # node 2 (BR)
    _H01, _H01, _H11, _H11,   # node 3 (TR)
    _H01, _H01, _H11, _H11,   # node 4 (TL)
], dtype=np.int64)


# =============================================================================
#  Shared low-level helpers
# -----------------------------------------------------------------------------
#  ``_q1_membrane_gradients`` and ``_bfs_curvature_factors`` compute the
#  per-point physical derivatives needed by both the partial-mode and the
#  full-mode strain-displacement operators.  Extracting them keeps a single
#  source of truth.
# =============================================================================

def _q1_membrane_gradients(coords_per_point: np.ndarray,
                           xi: np.ndarray,
                           eta: np.ndarray):
    """Return (dN/dx, dN/dy, det_J) for the 4 Q1 nodes at every point."""
    one_minus_xi = 1.0 - xi
    one_plus_xi = 1.0 + xi
    one_minus_eta = 1.0 - eta
    one_plus_eta = 1.0 + eta

    dN_dxi = 0.25 * np.stack([
        -one_minus_eta,
         one_minus_eta,
         one_plus_eta,
        -one_plus_eta,
    ], axis=-1)
    dN_deta = 0.25 * np.stack([
        -one_minus_xi,
        -one_plus_xi,
         one_plus_xi,
         one_minus_xi,
    ], axis=-1)

    x_nodes = coords_per_point[..., 0]
    y_nodes = coords_per_point[..., 1]

    dx_dxi = (dN_dxi * x_nodes).sum(axis=-1)
    dy_dxi = (dN_dxi * y_nodes).sum(axis=-1)
    dx_deta = (dN_deta * x_nodes).sum(axis=-1)
    dy_deta = (dN_deta * y_nodes).sum(axis=-1)

    det_J = dx_dxi * dy_deta - dx_deta * dy_dxi
    if np.any(det_J <= 0.0):
        n_bad = int(np.count_nonzero(det_J <= 0.0))
        raise ValueError(
            f"{n_bad} Q1 Jacobian determinant(s) are non-positive; "
            "check element node ordering."
        )

    inv_det = 1.0 / det_J
    j11_inv = dy_deta * inv_det
    j12_inv = -dx_deta * inv_det
    j21_inv = -dy_dxi * inv_det
    j22_inv = dx_dxi * inv_det

    dN_dx = j11_inv[..., None] * dN_dxi + j12_inv[..., None] * dN_deta
    dN_dy = j21_inv[..., None] * dN_dxi + j22_inv[..., None] * dN_deta
    return dN_dx, dN_dy, det_J


def _bfs_curvature_factors(xi: np.ndarray,
                           eta: np.ndarray,
                           dx_per_point: np.ndarray,
                           dy_per_point: np.ndarray):
    """Return (N_xx, N_yy, N_xy) per BFS DOF at every point.

    Each output is shape ``(N, 16)`` where the column index runs over the
    16 BFS DOFs in canonical order [BL, BR, TR, TL] x [w, w_x, w_y, w_xy].
    """
    xp = 0.5 * (xi + 1.0) * dx_per_point
    yp = 0.5 * (eta + 1.0) * dy_per_point

    unique_dx = np.unique(dx_per_point)
    unique_dy = np.unique(dy_per_point)
    n_points = xi.size

    if unique_dx.size == 1 and unique_dy.size == 1:
        Hx, dHx, d2Hx = bfs_1d_hermite_batch(xp, float(unique_dx[0]))
        Hy, dHy, d2Hy = bfs_1d_hermite_batch(yp, float(unique_dy[0]))
    else:
        Hx = np.empty((n_points, 4), dtype=float)
        dHx = np.empty_like(Hx)
        d2Hx = np.empty_like(Hx)
        Hy = np.empty_like(Hx)
        dHy = np.empty_like(Hx)
        d2Hy = np.empty_like(Hx)
        for a_val in unique_dx:
            mask = (dx_per_point == a_val)
            Hx_m, dHx_m, d2Hx_m = bfs_1d_hermite_batch(xp[mask], float(a_val))
            Hx[mask] = Hx_m
            dHx[mask] = dHx_m
            d2Hx[mask] = d2Hx_m
        for a_val in unique_dy:
            mask = (dy_per_point == a_val)
            Hy_m, dHy_m, d2Hy_m = bfs_1d_hermite_batch(yp[mask], float(a_val))
            Hy[mask] = Hy_m
            dHy[mask] = dHy_m
            d2Hy[mask] = d2Hy_m

    Hx_alpha = Hx[:, _BFS_ALPHA_INDEX]
    dHx_alpha = dHx[:, _BFS_ALPHA_INDEX]
    d2Hx_alpha = d2Hx[:, _BFS_ALPHA_INDEX]
    Hy_beta = Hy[:, _BFS_BETA_INDEX]
    dHy_beta = dHy[:, _BFS_BETA_INDEX]
    d2Hy_beta = d2Hy[:, _BFS_BETA_INDEX]

    N_xx = d2Hx_alpha * Hy_beta         # d^2 N / dx^2
    N_yy = Hx_alpha * d2Hy_beta         # d^2 N / dy^2
    N_xy = dHx_alpha * dHy_beta         # d^2 N / (dx dy)
    return N_xx, N_yy, N_xy


# =============================================================================
#  Strain-displacement operator -- public dispatcher
# -----------------------------------------------------------------------------
#  Returns ``(B_all, det_J)`` where the shape of ``B_all`` depends on the
#  analysis type:
#      'in_plane'      -> (N, 3, 8)
#      'out_of_plane'  -> (N, 3, 16)
#      'both'          -> (N, 6, 24)
# =============================================================================

def strain_displacement_batch(analysis_type: str,
                              coords_per_point: np.ndarray,
                              xi: np.ndarray,
                              eta: np.ndarray,
                              dx_per_point: np.ndarray,
                              dy_per_point: np.ndarray
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """Batched analysis-aware strain-displacement operator evaluator.

    Output rows / columns are described in the module docstring.
    """
    xi = np.asarray(xi, dtype=float)
    eta = np.asarray(eta, dtype=float)
    coords_per_point = np.asarray(coords_per_point, dtype=float)
    dx_per_point = np.asarray(dx_per_point, dtype=float)
    dy_per_point = np.asarray(dy_per_point, dtype=float)

    if xi.shape != eta.shape:
        raise ValueError("xi and eta must have the same shape.")
    if coords_per_point.shape[:-2] != xi.shape:
        raise ValueError(
            f"coords_per_point shape {coords_per_point.shape} is incompatible "
            f"with xi shape {xi.shape}; expected leading dims to match."
        )
    if coords_per_point.shape[-2:] != (4, 2):
        raise ValueError(
            f"coords_per_point must have trailing shape (4, 2); got "
            f"{coords_per_point.shape}."
        )
    if dx_per_point.shape != xi.shape or dy_per_point.shape != xi.shape:
        raise ValueError(
            "dx_per_point / dy_per_point must have the same shape as xi."
        )

    key = str(analysis_type).strip().lower()
    if key == "in_plane":
        return _inplane_strain_displacement_batch(
            coords_per_point, xi, eta)
    if key == "out_of_plane":
        return _outofplane_strain_displacement_batch(
            coords_per_point, xi, eta, dx_per_point, dy_per_point)
    if key == "both":
        return _both_strain_displacement_batch(
            coords_per_point, xi, eta, dx_per_point, dy_per_point)

    raise ValueError(
        f"Unsupported analysis_type {analysis_type!r}; expected one of "
        "'in_plane', 'out_of_plane', 'both'."
    )


# -----------------------------------------------------------------------------
#  In-plane operator -- (N, 3, 8)
# -----------------------------------------------------------------------------
#  Element DOF layout: [u1, v1, u2, v2, u3, v3, u4, v4]
#      row 0 : eps_xx     =  du/dx
#      row 1 : eps_yy     =  dv/dy
#      row 2 : gamma_xy   =  du/dy + dv/dx
# -----------------------------------------------------------------------------

def _inplane_strain_displacement_batch(coords_per_point: np.ndarray,
                                       xi: np.ndarray,
                                       eta: np.ndarray
                                       ) -> Tuple[np.ndarray, np.ndarray]:
    dN_dx, dN_dy, det_J = _q1_membrane_gradients(coords_per_point, xi, eta)
    n_points = xi.size

    B = np.zeros((n_points, 3, 8), dtype=float)
    u_columns = np.array([0, 2, 4, 6], dtype=np.int64)   # u offset = 0
    v_columns = np.array([1, 3, 5, 7], dtype=np.int64)   # v offset = 1

    B[:, 0, u_columns] = dN_dx
    B[:, 1, v_columns] = dN_dy
    B[:, 2, u_columns] = dN_dy
    B[:, 2, v_columns] = dN_dx

    assert B.shape == (n_points, 3, 8), B.shape
    return B, det_J


# -----------------------------------------------------------------------------
#  Out-of-plane operator -- (N, 3, 16)
# -----------------------------------------------------------------------------
#  Element DOF layout: [w, w_x, w_y, w_xy] repeated for [BL, BR, TR, TL]
#      row 0 : kappa_xx   = -d^2 w / dx^2
#      row 1 : kappa_yy   = -d^2 w / dy^2
#      row 2 : kappa_xy   = -2 d^2 w / (dx dy)
# -----------------------------------------------------------------------------

def _outofplane_strain_displacement_batch(coords_per_point: np.ndarray,
                                          xi: np.ndarray,
                                          eta: np.ndarray,
                                          dx_per_point: np.ndarray,
                                          dy_per_point: np.ndarray
                                          ) -> Tuple[np.ndarray, np.ndarray]:
    _, _, det_J = _q1_membrane_gradients(coords_per_point, xi, eta)
    N_xx, N_yy, N_xy = _bfs_curvature_factors(
        xi, eta, dx_per_point, dy_per_point)
    n_points = xi.size

    # BFS DOFs occupy columns 0..15 (each node has [w, w_x, w_y, w_xy]).
    B = np.zeros((n_points, 3, 16), dtype=float)
    B[:, 0, :] = -N_xx
    B[:, 1, :] = -N_yy
    B[:, 2, :] = -2.0 * N_xy

    assert B.shape == (n_points, 3, 16), B.shape
    return B, det_J


# -----------------------------------------------------------------------------
#  Both operator -- (N, 6, 24)  (composite Q1 + BFS)
# -----------------------------------------------------------------------------
#  Element DOF layout: [u, v, w, w_x, w_y, w_xy] repeated for [BL, BR, TR, TL]
#      rows 0-2 : membrane strains (Q1 gradients)
#      rows 3-5 : curvatures        (BFS Hessians, CLT sign convention)
# -----------------------------------------------------------------------------

# Column index inside the 24-DOF element block, for each of the 16 BFS DOFs.
_BOTH_BFS_ELEMENT_COLUMN = np.array([
    (k // 4) * 6 + 2 + (k % 4) for k in range(16)
], dtype=np.int64)


def _both_strain_displacement_batch(coords_per_point: np.ndarray,
                                    xi: np.ndarray,
                                    eta: np.ndarray,
                                    dx_per_point: np.ndarray,
                                    dy_per_point: np.ndarray
                                    ) -> Tuple[np.ndarray, np.ndarray]:
    dN_dx, dN_dy, det_J = _q1_membrane_gradients(coords_per_point, xi, eta)
    N_xx, N_yy, N_xy = _bfs_curvature_factors(
        xi, eta, dx_per_point, dy_per_point)
    n_points = xi.size

    B = np.zeros((n_points, 6, 24), dtype=float)

    # Membrane rows (Q1 gradients).  Each 6-DOF node: u offset 0, v offset 1.
    u_columns = np.array([0, 6, 12, 18], dtype=np.int64)
    v_columns = np.array([1, 7, 13, 19], dtype=np.int64)
    B[:, 0, u_columns] = dN_dx
    B[:, 1, v_columns] = dN_dy
    B[:, 2, u_columns] = dN_dy
    B[:, 2, v_columns] = dN_dx

    # Bending rows (BFS Hessians).  Each 6-DOF node: w/wx/wy/wxy at offsets 2..5.
    bfs_columns = _BOTH_BFS_ELEMENT_COLUMN
    B[:, 3, bfs_columns] = -N_xx
    B[:, 4, bfs_columns] = -N_yy
    B[:, 5, bfs_columns] = -2.0 * N_xy

    assert B.shape == (n_points, 6, 24), B.shape
    return B, det_J


# =============================================================================
#  Mesh construction
# =============================================================================

def mesh_settings_from_node_points(points) -> MeshSettings:
    """Validate manual node coordinates and return structured mesh settings.

    ``points`` must be an iterable of ``(x, y)`` pairs forming every
    Cartesian product of the unique x and y coordinates.  This is the
    conformity rule for the GUI's Manual Nodes feature.
    """
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Manual nodes must be an (N, 2) array of x/y pairs.")
    if arr.shape[0] < 4:
        raise ValueError("Enter at least four nodes for one quadrilateral.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Manual node coordinates must be finite numbers.")

    tuples = [(float(x), float(y)) for x, y in arr]
    if len(set(tuples)) != len(tuples):
        raise ValueError("Duplicate node coordinates are not allowed.")

    xs = sorted({x for x, _y in tuples})
    ys = sorted({y for _x, y in tuples})
    if len(xs) < 2 or len(ys) < 2:
        raise ValueError("Need at least two unique x values and y values.")

    expected = {(x, y) for y in ys for x in xs}
    provided = set(tuples)
    missing = sorted(expected - provided)
    if missing:
        preview = ", ".join(f"({x:g}, {y:g})" for x, y in missing[:4])
        more = "" if len(missing) <= 4 else f" and {len(missing) - 4} more"
        raise ValueError(
            "Manual nodes must form a complete structured x-by-y grid; "
            f"missing {preview}{more}."
        )

    return MeshSettings(
        number_of_elements_x=len(xs) - 1,
        number_of_elements_y=len(ys) - 1,
        bounds=Bounds(
            x_min=float(xs[0]),
            x_max=float(xs[-1]),
            y_min=float(ys[0]),
            y_max=float(ys[-1]),
        ),
        x_nodes=[float(x) for x in xs],
        y_nodes=[float(y) for y in ys],
    )


def build_mesh(mesh_settings: MeshSettings,
               analysis_type: str = "both") -> MeshPackage:
    """Build a structured Q1 rectangular mesh with analysis-aware DOFs.

    ``n_dof_per_node`` is derived from ``analysis_type``:
        'in_plane'      -> 2
        'out_of_plane'  -> 4
        'both'          -> 6
    """
    n_dof_per_node = dof_per_node_for_analysis(analysis_type)
    analysis_key = str(analysis_type).strip().lower()

    xs, ys = _mesh_axis_nodes(mesh_settings)
    nx = int(xs.size - 1)
    ny = int(ys.size - 1)

    skfem_mesh = MeshQuad.init_tensor(xs, ys)

    node_coords = skfem_mesh.p.T.copy()
    n_nodes = node_coords.shape[0]

    n_elements = nx * ny
    element_nodes = np.zeros((n_elements, 4), dtype=np.int64)
    element_grid_index = np.zeros((nx, ny), dtype=np.int64)
    element_dx = np.zeros(n_elements, dtype=float)
    element_dy = np.zeros(n_elements, dtype=float)

    node_grid_index = _build_node_grid_index(node_coords, xs, ys)

    element_id = 0
    for iy in range(ny):
        for ix in range(nx):
            bl = node_grid_index[ix,     iy]
            br = node_grid_index[ix + 1, iy]
            tr = node_grid_index[ix + 1, iy + 1]
            tl = node_grid_index[ix,     iy + 1]
            element_nodes[element_id, :] = [bl, br, tr, tl]
            element_grid_index[ix, iy] = element_id
            element_dx[element_id] = xs[ix + 1] - xs[ix]
            element_dy[element_id] = ys[iy + 1] - ys[iy]
            element_id += 1

    return MeshPackage(
        skfem_mesh=skfem_mesh,
        node_coords=node_coords,
        element_nodes=element_nodes,
        n_nodes=n_nodes,
        n_elements=n_elements,
        n_dof_per_node=int(n_dof_per_node),
        n_dofs=n_nodes * int(n_dof_per_node),
        analysis_type=analysis_key,
        nx=nx,
        ny=ny,
        x_min=float(xs[0]),
        x_max=float(xs[-1]),
        y_min=float(ys[0]),
        y_max=float(ys[-1]),
        dx=float(np.min(np.diff(xs))),
        dy=float(np.min(np.diff(ys))),
        element_grid_index=element_grid_index,
        x_nodes=xs.copy(),
        y_nodes=ys.copy(),
        element_dx=element_dx,
        element_dy=element_dy,
    )


def _mesh_axis_nodes(mesh_settings: MeshSettings) -> Tuple[np.ndarray, np.ndarray]:
    x_nodes = getattr(mesh_settings, "x_nodes", None)
    y_nodes = getattr(mesh_settings, "y_nodes", None)
    has_manual_x = x_nodes is not None
    has_manual_y = y_nodes is not None
    if has_manual_x != has_manual_y:
        raise ValueError("Manual mesh settings require both x_nodes and y_nodes.")

    if has_manual_x and has_manual_y:
        xs = _validate_axis_nodes(x_nodes, "x_nodes")
        ys = _validate_axis_nodes(y_nodes, "y_nodes")
        return xs, ys

    if mesh_settings.number_of_elements_x < 1 \
            or mesh_settings.number_of_elements_y < 1:
        raise ValueError("Mesh element counts must be >= 1 in each direction.")

    bounds = mesh_settings.bounds
    if bounds.x_max <= bounds.x_min or bounds.y_max <= bounds.y_min:
        raise ValueError("Mesh bounds must define a non-degenerate rectangle.")

    nx = int(mesh_settings.number_of_elements_x)
    ny = int(mesh_settings.number_of_elements_y)
    return (
        np.linspace(bounds.x_min, bounds.x_max, nx + 1),
        np.linspace(bounds.y_min, bounds.y_max, ny + 1),
    )


def _validate_axis_nodes(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size < 2:
        raise ValueError(f"{name} must contain at least two coordinates.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite coordinates.")
    if np.unique(arr).size != arr.size:
        raise ValueError(f"{name} must not contain duplicate coordinates.")
    arr = np.sort(arr)
    diffs = np.diff(arr)
    if np.any(diffs <= 0.0):
        raise ValueError(f"{name} coordinates must be strictly increasing.")
    return arr


def _build_node_grid_index(node_coords: np.ndarray,
                           xs: np.ndarray,
                           ys: np.ndarray,
                           *,
                           require_complete: bool = True) -> np.ndarray:
    nx_p1 = xs.size
    ny_p1 = ys.size
    node_grid_index = -np.ones((nx_p1, ny_p1), dtype=np.int64)

    if nx_p1 >= 2 and ny_p1 >= 2:
        tolerance = max(
            1.0e-12,
            1.0e-9 * min(np.min(np.diff(xs)), np.min(np.diff(ys))),
        )
    else:
        tolerance = 1.0e-12

    for node_id, (x, y) in enumerate(node_coords):
        ix = _nearest_axis_index(xs, x)
        iy = _nearest_axis_index(ys, y)
        if abs(x - xs[ix]) > tolerance or abs(y - ys[iy]) > tolerance:
            raise RuntimeError(
                "Could not place node at "
                f"({x}, {y}) on the structured grid; check mesh inputs."
            )
        if node_grid_index[ix, iy] != -1:
            raise RuntimeError(
                f"Duplicate grid slot ({ix}, {iy}) assigned to nodes "
                f"{node_grid_index[ix, iy]} and {node_id}."
            )
        node_grid_index[ix, iy] = node_id

    if require_complete and (node_grid_index < 0).any():
        raise RuntimeError("Failed to assign every grid slot to a mesh node.")

    return node_grid_index


def _nearest_axis_index(axis: np.ndarray, value: float) -> int:
    pos = int(np.searchsorted(axis, value, side="left"))
    if pos <= 0:
        return 0
    if pos >= axis.size:
        return int(axis.size - 1)
    left = pos - 1
    if abs(value - axis[left]) <= abs(value - axis[pos]):
        return left
    return pos


# =============================================================================
#  Point-to-element mapping
# =============================================================================

def map_points_to_mesh(points: np.ndarray,
                       mesh_pkg: MeshPackage,
                       inside_tolerance: float = 1.0e-9) -> PointMapping:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("`points` must have shape (n_points, 2).")

    n_points = points.shape[0]
    element_ids = -np.ones(n_points, dtype=np.int64)
    local_xy = np.full((n_points, 2), np.nan, dtype=float)

    x = points[:, 0]
    y = points[:, 1]

    width = mesh_pkg.x_max - mesh_pkg.x_min
    height = mesh_pkg.y_max - mesh_pkg.y_min
    abs_tol_x = max(inside_tolerance, inside_tolerance * width)
    abs_tol_y = max(inside_tolerance, inside_tolerance * height)

    inside = (
        (x >= mesh_pkg.x_min - abs_tol_x)
        & (x <= mesh_pkg.x_max + abs_tol_x)
        & (y >= mesh_pkg.y_min - abs_tol_y)
        & (y <= mesh_pkg.y_max + abs_tol_y)
    )

    if not inside.any():
        return PointMapping(
            element_ids=element_ids,
            local_xy=local_xy,
            mapped_mask=element_ids >= 0,
        )

    x_in = x[inside]
    y_in = y[inside]

    x_nodes = np.asarray(getattr(mesh_pkg, "x_nodes", []), dtype=float)
    y_nodes = np.asarray(getattr(mesh_pkg, "y_nodes", []), dtype=float)
    if x_nodes.size != mesh_pkg.nx + 1:
        x_nodes = np.linspace(mesh_pkg.x_min, mesh_pkg.x_max, mesh_pkg.nx + 1)
    if y_nodes.size != mesh_pkg.ny + 1:
        y_nodes = np.linspace(mesh_pkg.y_min, mesh_pkg.y_max, mesh_pkg.ny + 1)

    ix = np.searchsorted(x_nodes, x_in, side="right") - 1
    iy = np.searchsorted(y_nodes, y_in, side="right") - 1
    ix = np.clip(ix, 0, mesh_pkg.nx - 1)
    iy = np.clip(iy, 0, mesh_pkg.ny - 1)

    x0 = x_nodes[ix]
    x1 = x_nodes[ix + 1]
    y0 = y_nodes[iy]
    y1 = y_nodes[iy + 1]

    xi = np.clip(2.0 * (x_in - x0) / (x1 - x0) - 1.0, -1.0, 1.0)
    eta = np.clip(2.0 * (y_in - y0) / (y1 - y0) - 1.0, -1.0, 1.0)

    grid_ids = mesh_pkg.element_grid_index[ix, iy]
    element_ids[inside] = np.where(grid_ids >= 0, grid_ids, -1)
    local_xy[inside, 0] = xi
    local_xy[inside, 1] = eta

    mapped_mask = element_ids >= 0
    return PointMapping(
        element_ids=element_ids,
        local_xy=local_xy,
        mapped_mask=mapped_mask,
    )


def subset_measurements_to_mapped_mesh(
    measurements: PointMeasurements,
    mesh_pkg: MeshPackage,
    mapping: Optional[PointMapping] = None,
) -> Tuple[PointMeasurements, PointMapping]:
    """Return measurements restricted to points mapped into the active mesh.

    Points outside the mesh rectangle or landing in pruned grid cells
    (``element_grid_index == -1``) are dropped so they do not enter the
    virtual-work sums.  ``area_per_point`` is left unchanged: the remaining
    points keep their original quadrature weight.
    """
    if mapping is None:
        mapping = map_points_to_mesh(measurements.points, mesh_pkg)

    active = np.asarray(mapping.mapped_mask, dtype=bool)
    if not active.any():
        raise ValueError(
            "No measurement points map into the active virtual mesh.  "
            "Check mesh bounds and pruning against the experimental cloud."
        )

    points = np.asarray(measurements.points, dtype=float)[active]
    strains = np.asarray(measurements.strains, dtype=float)[:, active]
    subset = PointMeasurements(
        points=points,
        strains=strains,
        applied_force=measurements.applied_force,
        area_per_point=float(measurements.area_per_point),
        data_bounds=dict(measurements.data_bounds),
        analysis_type=str(measurements.analysis_type),
    )
    element_ids = np.asarray(mapping.element_ids, dtype=np.int64)[active]
    local_xy = np.asarray(mapping.local_xy, dtype=float)[active]
    subset_mapping = PointMapping(
        element_ids=element_ids,
        local_xy=local_xy,
        mapped_mask=np.ones(points.shape[0], dtype=bool),
    )
    return subset, subset_mapping


def count_measurements_per_element(points: np.ndarray,
                                   mesh_pkg: MeshPackage) -> np.ndarray:
    mapping = map_points_to_mesh(points, mesh_pkg)
    n_el = int(mesh_pkg.n_elements)
    if mapping.n_mapped == 0:
        return np.zeros(n_el, dtype=np.int64)
    ids = mapping.element_ids[mapping.mapped_mask].astype(np.int64, copy=False)
    bc = np.bincount(ids, minlength=n_el)
    return np.asarray(bc[:n_el], dtype=np.int64)


def prune_mesh_by_measurement_points(
    mesh_pkg: MeshPackage,
    points: np.ndarray,
    *,
    min_points_per_element: int = 1,
) -> Tuple[MeshPackage, MeshPruneReport]:
    """Drop virtual elements with no mapped measurement points.

    The structured ``nx`` / ``ny`` bookkeeping and ``element_grid_index`` are
    preserved, but cells without data are marked ``-1`` in the grid index.
    Only nodes referenced by the surviving elements are kept; global node and
    element ids are renumbered compactly so removed elements do not allocate
    DOFs.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("`points` must have shape (n_points, 2).")

    n_elements_before = int(mesh_pkg.n_elements)
    n_nodes_before = int(mesh_pkg.n_nodes)

    counts = count_measurements_per_element(points, mesh_pkg)
    keep_elements = np.flatnonzero(counts >= int(min_points_per_element))
    if keep_elements.size == 0:
        raise ValueError(
            "No virtual mesh element contains a mapped measurement point.  "
            "Check mesh bounds against the experimental point cloud, or "
            "ensure at least one measurement lies inside the virtual mesh."
        )

    if keep_elements.size == n_elements_before:
        report = MeshPruneReport(
            n_elements_before=n_elements_before,
            n_elements_after=n_elements_before,
            n_elements_removed=0,
            n_nodes_before=n_nodes_before,
            n_nodes_after=n_nodes_before,
            n_nodes_removed=0,
        )
        return mesh_pkg, report

    old_element_nodes = mesh_pkg.element_nodes[keep_elements]
    element_dx = mesh_pkg.element_dx[keep_elements]
    element_dy = mesh_pkg.element_dy[keep_elements]

    used_nodes = np.unique(old_element_nodes.ravel())
    old_to_new_node = -np.ones(n_nodes_before, dtype=np.int64)
    old_to_new_node[used_nodes] = np.arange(used_nodes.size, dtype=np.int64)

    new_element_nodes = old_to_new_node[old_element_nodes]
    new_node_coords = mesh_pkg.node_coords[used_nodes].copy()

    n_elements_after = int(keep_elements.size)
    n_nodes_after = int(used_nodes.size)

    old_to_new_element = -np.ones(n_elements_before, dtype=np.int64)
    old_to_new_element[keep_elements] = np.arange(
        n_elements_after, dtype=np.int64)

    new_element_grid_index = -np.ones_like(mesh_pkg.element_grid_index)
    flat_old = mesh_pkg.element_grid_index.ravel()
    flat_new = new_element_grid_index.ravel()
    keep_mask = flat_old >= 0
    mapped = old_to_new_element[flat_old[keep_mask]]
    flat_new[keep_mask] = mapped
    new_element_grid_index = flat_new.reshape(mesh_pkg.element_grid_index.shape)

    new_skfem_mesh = MeshQuad(new_node_coords.T, new_element_nodes.T)

    pruned = MeshPackage(
        skfem_mesh=new_skfem_mesh,
        node_coords=new_node_coords,
        element_nodes=new_element_nodes,
        n_nodes=n_nodes_after,
        n_elements=n_elements_after,
        n_dof_per_node=int(mesh_pkg.n_dof_per_node),
        n_dofs=n_nodes_after * int(mesh_pkg.n_dof_per_node),
        analysis_type=mesh_pkg.analysis_type,
        nx=int(mesh_pkg.nx),
        ny=int(mesh_pkg.ny),
        x_min=float(mesh_pkg.x_min),
        x_max=float(mesh_pkg.x_max),
        y_min=float(mesh_pkg.y_min),
        y_max=float(mesh_pkg.y_max),
        dx=float(mesh_pkg.dx),
        dy=float(mesh_pkg.dy),
        element_grid_index=new_element_grid_index,
        x_nodes=np.asarray(mesh_pkg.x_nodes, dtype=float).copy(),
        y_nodes=np.asarray(mesh_pkg.y_nodes, dtype=float).copy(),
        element_dx=element_dx,
        element_dy=element_dy,
    )

    report = MeshPruneReport(
        n_elements_before=n_elements_before,
        n_elements_after=n_elements_after,
        n_elements_removed=n_elements_before - n_elements_after,
        n_nodes_before=n_nodes_before,
        n_nodes_after=n_nodes_after,
        n_nodes_removed=n_nodes_before - n_nodes_after,
    )
    return pruned, report


def build_mesh_with_measurements(
    mesh_settings: MeshSettings,
    points: np.ndarray,
    analysis_type: str = "both",
    *,
    min_points_per_element: int = 1,
) -> Tuple[MeshPackage, MeshPruneReport]:
    """Build the structured mesh and drop elements without measurement data."""
    mesh_pkg = build_mesh(mesh_settings, analysis_type=analysis_type)
    return prune_mesh_by_measurement_points(
        mesh_pkg,
        points,
        min_points_per_element=min_points_per_element,
    )


# =============================================================================
#  DOF helpers
# =============================================================================

def element_dof_indices(node_ids: np.ndarray, n_dof_per_node: int) -> np.ndarray:
    """Return the global DOF indices for one element in interleaved order."""
    node_ids = np.asarray(node_ids, dtype=np.int64).ravel()
    n_nodes_per_elem = node_ids.size
    dof_indices = np.empty(n_nodes_per_elem * n_dof_per_node, dtype=np.int64)
    for local_node, global_node in enumerate(node_ids):
        start = local_node * n_dof_per_node
        for dof_component in range(n_dof_per_node):
            dof_indices[start + dof_component] = \
                global_node * n_dof_per_node + dof_component
    return dof_indices


# =============================================================================
#  Boundary selectors
# =============================================================================

def _select_edge_nodes(mesh_pkg: MeshPackage, edge_name: str) -> np.ndarray:
    edge = str(edge_name).strip().lower()
    coords = mesh_pkg.node_coords
    tolerance = max(1.0e-12,
                    1.0e-10 * max(abs(mesh_pkg.x_min), abs(mesh_pkg.x_max),
                                  abs(mesh_pkg.y_min), abs(mesh_pkg.y_max),
                                  1.0))

    if edge == "left":
        mask = np.abs(coords[:, 0] - mesh_pkg.x_min) <= tolerance
    elif edge == "right":
        mask = np.abs(coords[:, 0] - mesh_pkg.x_max) <= tolerance
    elif edge == "bottom":
        mask = np.abs(coords[:, 1] - mesh_pkg.y_min) <= tolerance
    elif edge == "top":
        mask = np.abs(coords[:, 1] - mesh_pkg.y_max) <= tolerance
    else:
        raise ValueError(f"Unsupported edge name: {edge_name!r}")

    return np.where(mask)[0]


def _resolve_group_nodes(group: Dict[str, Any],
                         mesh_pkg: MeshPackage) -> np.ndarray:
    label = group.get("label", "<unlabeled>")
    selected: Optional[np.ndarray] = None

    if group.get("nodes"):
        node_list = np.unique(np.asarray(group["nodes"], dtype=np.int64))
        selected = node_list

    if group.get("edge"):
        edge_nodes = _select_edge_nodes(mesh_pkg, group["edge"])
        if selected is None:
            selected = edge_nodes
        else:
            selected = np.unique(np.concatenate([selected, edge_nodes]))

    if group.get("excludeEdges"):
        excludes = group["excludeEdges"]
        if isinstance(excludes, str):
            excludes = [excludes]
        excluded: List[int] = []
        for edge_name in excludes:
            excluded.extend(_select_edge_nodes(mesh_pkg, edge_name).tolist())
        if selected is not None and excluded:
            selected = np.setdiff1d(selected, np.asarray(excluded,
                                                          dtype=np.int64))

    if group.get("excludeNodes"):
        excluded_nodes = np.asarray(group["excludeNodes"], dtype=np.int64)
        if selected is not None:
            selected = np.setdiff1d(selected, excluded_nodes)

    if selected is None or selected.size == 0:
        raise ValueError(
            f"Boundary group {label!r} did not select any nodes."
        )

    return selected.astype(np.int64, copy=False)


def _sort_nodes(nodes: np.ndarray,
                mesh_pkg: MeshPackage,
                sort_direction: Optional[str]) -> np.ndarray:
    if not sort_direction:
        return nodes
    axis = str(sort_direction).strip().lower()
    if axis == "x":
        key = mesh_pkg.node_coords[nodes, 0]
    elif axis == "y":
        key = mesh_pkg.node_coords[nodes, 1]
    else:
        raise ValueError(f"Unsupported sortDirection: {sort_direction!r}")
    return nodes[np.argsort(key)]


def _resolve_dof_component(dof_label: str,
                           analysis_type: str) -> int:
    """Return the 0-based DOF offset within a node block for ``dof_label``.

    Raises ``ValueError`` if the label is not legal in the active analysis
    type (e.g. ``W`` in ``in_plane``).
    """
    offsets = _dof_offsets(analysis_type)
    key = str(dof_label).strip().upper()
    if key not in offsets:
        raise ValueError(
            f"DOF label {dof_label!r} is not allowed in analysis_type "
            f"'{analysis_type}'.  Allowed labels: {sorted(offsets.keys())}."
        )
    return offsets[key]


# =============================================================================
#  KA constraint matrix (Dirichlet + linking + inactive DOFs)
# =============================================================================

def build_kinematic_constraints(mesh_pkg: MeshPackage,
                                bc: BoundaryConditions,
                                inactive_dofs: np.ndarray
                                ) -> ConstraintSystem:
    rows_index: List[int] = []
    cols_index: List[int] = []
    values: List[float] = []
    row_counter = 0
    analysis_type = mesh_pkg.analysis_type

    # ------ (a) prescribed Dirichlet -------------------------------------
    n_prescribed = 0
    for group in bc.prescribed:
        prescribed_value = float(group.get("value", 0.0) or 0.0)
        if abs(prescribed_value) > 0.0:
            raise ValueError(
                "Only homogeneous prescribed values (=0) are supported. "
                f"Group {group.get('label', '<unlabeled>')!r} has "
                f"value={prescribed_value}."
            )

        dof_offset = _resolve_dof_component(group["dof"], analysis_type)
        node_ids = _resolve_group_nodes(group, mesh_pkg)

        for node_id in node_ids:
            rows_index.append(row_counter)
            cols_index.append(int(node_id) * mesh_pkg.n_dof_per_node
                              + dof_offset)
            values.append(1.0)
            row_counter += 1
            n_prescribed += 1

    # ------ (b) linking ties --------------------------------------------
    n_linking = 0
    for group in bc.linking:
        link_value = float(group.get("value", 0.0) or 0.0)
        if abs(link_value) > 0.0:
            raise ValueError(
                "Only homogeneous linking offsets (=0) are supported. "
                f"Group {group.get('label', '<unlabeled>')!r} has "
                f"value={link_value}."
            )

        dof_offset = _resolve_dof_component(group["dof"], analysis_type)
        node_ids = _resolve_group_nodes(group, mesh_pkg)
        node_ids = _sort_nodes(node_ids, mesh_pkg, group.get("sortDirection"))

        if node_ids.size <= 1:
            continue

        reference_node = int(node_ids[0])
        reference_dof = reference_node * mesh_pkg.n_dof_per_node + dof_offset

        for follower_node in node_ids[1:]:
            follower_dof = (int(follower_node) * mesh_pkg.n_dof_per_node
                            + dof_offset)
            rows_index.append(row_counter)
            cols_index.append(follower_dof)
            values.append(1.0)
            rows_index.append(row_counter)
            cols_index.append(reference_dof)
            values.append(-1.0)
            row_counter += 1
            n_linking += 1

    # ------ (c) inactive DOFs -------------------------------------------
    inactive_dofs = np.asarray(inactive_dofs, dtype=np.int64).ravel()
    for dof in inactive_dofs:
        rows_index.append(row_counter)
        cols_index.append(int(dof))
        values.append(1.0)
        row_counter += 1

    n_constraints = row_counter

    if n_constraints == 0:
        C = sp.csr_matrix((0, mesh_pkg.n_dofs))
    else:
        C = sp.csr_matrix(
            (values, (rows_index, cols_index)),
            shape=(n_constraints, mesh_pkg.n_dofs),
        )

    return ConstraintSystem(
        C=C,
        n_prescribed=n_prescribed,
        n_linking=n_linking,
        n_inactive=int(inactive_dofs.size),
        inactive_dofs=inactive_dofs,
        n_constraints=n_constraints,
    )


# =============================================================================
#  Admissible-space basis (null space of C)
# =============================================================================

def build_admissible_basis(C: sp.csr_matrix,
                           n_dofs: int,
                           svd_tolerance_factor: float,
                           rank_tolerance_floor: float
                           ) -> Tuple[np.ndarray, Dict[str, float]]:
    n_constraints = C.shape[0]
    if n_constraints == 0:
        return np.eye(n_dofs), {
            "rank": 0,
            "nullity": n_dofs,
            "tolerance": 0.0,
        }

    C_dense = np.asarray(C.todense())

    _, singular_values, Vt = np.linalg.svd(C_dense, full_matrices=True)
    sigma_max = float(singular_values.max()) if singular_values.size else 0.0
    tolerance = max(
        rank_tolerance_floor,
        svd_tolerance_factor
        * max(C_dense.shape) * np.finfo(float).eps * sigma_max,
    )
    rank = int(np.count_nonzero(singular_values > tolerance))

    if rank >= n_dofs:
        raise ValueError(
            "The KA constraints eliminate every virtual displacement DOF. "
            "Loosen the prescribed/linking groups or check the BC JSON."
        )

    R = Vt[rank:, :].T.copy()

    metadata = {
        "rank": rank,
        "nullity": int(R.shape[1]),
        "tolerance": float(tolerance),
    }
    return R, metadata


# =============================================================================
#  External-work operator
# =============================================================================

def build_external_work_matrix(mesh_pkg: MeshPackage,
                               bc: BoundaryConditions
                               ) -> Tuple[sp.csr_matrix, Dict[str, List[str]]]:
    analysis_type = mesh_pkg.analysis_type

    if not bc.external:
        external_work_matrix = sp.csr_matrix((mesh_pkg.n_dofs, 1))
        return external_work_matrix, {"labels": ["<fallback>"],
                                      "load_types": ["concentrated"]}

    n_groups = len(bc.external)
    rows: List[int] = []
    cols: List[int] = []
    values: List[float] = []
    labels: List[str] = []
    load_types: List[str] = []

    for group_index, group in enumerate(bc.external):
        labels.append(str(group.get("label", f"External group #{group_index}")))
        load_type = str(group.get("loadType", "concentrated")).lower()
        load_types.append(load_type)

        dof_offset = _resolve_dof_component(group["dof"], analysis_type)
        node_ids = _resolve_group_nodes(group, mesh_pkg)
        node_ids = _sort_nodes(node_ids, mesh_pkg, group.get("sortDirection"))

        weights = _external_group_weights(load_type, node_ids, mesh_pkg)

        for node_id, weight in zip(node_ids, weights):
            rows.append(int(node_id) * mesh_pkg.n_dof_per_node + dof_offset)
            cols.append(group_index)
            values.append(float(weight))

    external_work_matrix = sp.csr_matrix(
        (values, (rows, cols)),
        shape=(mesh_pkg.n_dofs, n_groups),
    )

    return external_work_matrix, {"labels": labels, "load_types": load_types}


def _external_group_weights(load_type: str,
                            node_ids: np.ndarray,
                            mesh_pkg: MeshPackage) -> np.ndarray:
    n = node_ids.size
    if load_type == "concentrated":
        return np.full(n, 1.0 / n)

    if load_type == "distributed":
        if n < 2:
            raise ValueError(
                "Distributed external work requires at least two nodes."
            )
        coords = mesh_pkg.node_coords[node_ids]
        segment_lengths = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        weights = np.zeros(n, dtype=float)
        weights[0] = segment_lengths[0] / 2.0
        weights[-1] = segment_lengths[-1] / 2.0
        if n > 2:
            weights[1:-1] = (segment_lengths[:-1] + segment_lengths[1:]) / 2.0
        return weights

    raise ValueError(f"Unsupported external load type: {load_type!r}")


# =============================================================================
#  BFS Kronecker-delta self-test  (run with ``python -m vfm_mesh``)
# =============================================================================

def _evaluate_bfs_shape_and_derivatives_at_point(
    s_x: float, s_y: float, dx: float, dy: float
) -> Dict[str, np.ndarray]:
    Hx, dHx, _d2Hx = bfs_1d_hermite_batch(np.array([s_x]), dx)
    Hy, dHy, _d2Hy = bfs_1d_hermite_batch(np.array([s_y]), dy)

    Hx = Hx[0]
    dHx = dHx[0]
    Hy = Hy[0]
    dHy = dHy[0]

    Hx_a = Hx[_BFS_ALPHA_INDEX]
    dHx_a = dHx[_BFS_ALPHA_INDEX]
    Hy_b = Hy[_BFS_BETA_INDEX]
    dHy_b = dHy[_BFS_BETA_INDEX]

    return {
        "value":   Hx_a * Hy_b,
        "d_dx":    dHx_a * Hy_b,
        "d_dy":    Hx_a * dHy_b,
        "d2_dxdy": dHx_a * dHy_b,
    }


def _bfs_kronecker_check(dx: float = 2.0, dy: float = 3.0) -> bool:
    corner_local = np.array([
        [0.0, 0.0],
        [dx,  0.0],
        [dx,  dy],
        [0.0, dy],
    ])
    M = np.zeros((16, 16), dtype=float)
    for corner_idx, (sx, sy) in enumerate(corner_local):
        evald = _evaluate_bfs_shape_and_derivatives_at_point(sx, sy, dx, dy)
        for dof_kind, key in enumerate(("value", "d_dx", "d_dy", "d2_dxdy")):
            j_global = corner_idx * 4 + dof_kind
            M[:, j_global] = evald[key]
    identity = np.eye(16)
    deviation = float(np.max(np.abs(M - identity)))
    return deviation < 1.0e-10


if __name__ == "__main__":
    # --- BFS Kronecker-delta self-test ------------------------------------
    ok = _bfs_kronecker_check(dx=2.0, dy=3.0)
    if ok:
        print("[plate / vfm_mesh] BFS Kronecker-delta self-test PASSED.")
    else:
        print("[plate / vfm_mesh] BFS Kronecker-delta self-test FAILED.")
        raise SystemExit(1)

    # --- Strain-displacement operator shape checks ------------------------
    coords = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 3.0], [0.0, 3.0]])
    xi = np.array([0.1])
    eta = np.array([-0.2])
    dxs = np.array([2.0])
    dys = np.array([3.0])

    B_in, _ = strain_displacement_batch(
        "in_plane", coords[None, :, :], xi, eta, dxs, dys)
    assert B_in.shape == (1, 3, 8), B_in.shape

    B_out, _ = strain_displacement_batch(
        "out_of_plane", coords[None, :, :], xi, eta, dxs, dys)
    assert B_out.shape == (1, 3, 16), B_out.shape

    B_both, _ = strain_displacement_batch(
        "both", coords[None, :, :], xi, eta, dxs, dys)
    assert B_both.shape == (1, 6, 24), B_both.shape

    print(
        "[plate / vfm_mesh] strain-displacement shapes OK: "
        f"in_plane={B_in.shape}, out_of_plane={B_out.shape}, "
        f"both={B_both.shape}"
    )
