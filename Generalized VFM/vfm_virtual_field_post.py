# =============================================================================
#  plate / vfm_virtual_field_post.py
# -----------------------------------------------------------------------------
#  Post-process optimal virtual nodal displacements for visualization.
#
#  Analysis-type aware:
#    * ``nodal_virtual_generalized_strains`` returns the nodal patch average
#      of the active generalised-strain vector (3 rows in_plane /
#      out_of_plane, 6 rows in both mode).
#    * ``nodal_field`` extracts the nodal values of any single DOF (u, v,
#      w) using the mesh's analysis-aware DOF table.
#    * ``nodal_w`` is kept as a thin wrapper around ``nodal_field`` for
#      callers that explicitly want the transverse deflection.
#    * ``nodal_deformed_xyz`` maps reference mid-plane ``(X, Y)`` plus the
#      active displacement DOFs to current-configuration ``(x, y, z)``.
#    * ``structured_surface_grids_from_nodal_xyz`` samples nodal ``(x,y,z)``
#      on the same structured grid used for ``plot_surface``.
# =============================================================================

from __future__ import annotations

import numpy as np

from vfm_mesh import (
    MeshPackage,
    _build_node_grid_index,
    _dof_offsets,
    strain_displacement_batch,
)


def nodal_virtual_generalized_strains(
    mesh_pkg: MeshPackage,
    u: np.ndarray,
) -> np.ndarray:
    """Nodal patch average of the virtual generalised strain.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_nodes, n_gen)`` where ``n_gen`` is 3 for the partial
        modes and 6 for ``'both'``.  Column order follows the per-mode B
        matrix.
    """
    u = np.asarray(u, dtype=float).ravel()
    if u.size != mesh_pkg.n_dofs:
        raise ValueError(
            f"u has length {u.size}; expected mesh_pkg.n_dofs={mesh_pkg.n_dofs}."
        )

    n_dof_node = mesh_pkg.n_dof_per_node
    n_nodes_elem = mesh_pkg.element_nodes.shape[1]
    n_dofs_elem = n_nodes_elem * n_dof_node

    # n_gen is implied by the mesh's analysis type.
    n_gen_by_mode = {"in_plane": 3, "out_of_plane": 3, "both": 6}
    n_gen = n_gen_by_mode[mesh_pkg.analysis_type]

    acc = np.zeros((mesh_pkg.n_nodes, n_gen), dtype=float)
    counts = np.zeros(mesh_pkg.n_nodes, dtype=float)

    corner_xi = np.array([-1.0, 1.0, 1.0, -1.0], dtype=float)
    corner_eta = np.array([-1.0, -1.0, 1.0, 1.0], dtype=float)

    for e in range(mesh_pkg.n_elements):
        nodes = mesh_pkg.element_nodes[e, :]
        elem_coords = mesh_pkg.node_coords[nodes, :]
        u_elem = np.empty(n_dofs_elem, dtype=float)
        for ln in range(n_nodes_elem):
            g0 = nodes[ln] * n_dof_node
            u_elem[ln * n_dof_node:(ln + 1) * n_dof_node] = u[g0:g0 + n_dof_node]

        for c in range(4):
            coords_pp = elem_coords[np.newaxis, :, :]
            xi = np.array([corner_xi[c]], dtype=float)
            eta = np.array([corner_eta[c]], dtype=float)
            dx_p = np.array([
                float(getattr(mesh_pkg, "element_dx",
                              np.full(mesh_pkg.n_elements, mesh_pkg.dx))[e])
            ], dtype=float)
            dy_p = np.array([
                float(getattr(mesh_pkg, "element_dy",
                              np.full(mesh_pkg.n_elements, mesh_pkg.dy))[e])
            ], dtype=float)
            B_all, _ = strain_displacement_batch(
                analysis_type=mesh_pkg.analysis_type,
                coords_per_point=coords_pp,
                xi=xi,
                eta=eta,
                dx_per_point=dx_p,
                dy_per_point=dy_p,
            )
            B = B_all[0]
            eps = B @ u_elem
            gnode = int(nodes[c])
            acc[gnode, :] += eps
            counts[gnode] += 1.0

    counts = np.maximum(counts, 1.0)
    return acc / counts[:, np.newaxis]


def structured_surface_grids(
    mesh_pkg: MeshPackage,
    w_per_node: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, Y, Z) 2D arrays for ``plot_surface`` on the structured grid."""
    w_per_node = np.asarray(w_per_node, dtype=float).ravel()
    if w_per_node.size != mesh_pkg.n_nodes:
        raise ValueError("w_per_node length must match n_nodes.")

    xs = np.asarray(
        getattr(mesh_pkg, "x_nodes",
                np.linspace(mesh_pkg.x_min, mesh_pkg.x_max, mesh_pkg.nx + 1)),
        dtype=float,
    )
    ys = np.asarray(
        getattr(mesh_pkg, "y_nodes",
                np.linspace(mesh_pkg.y_min, mesh_pkg.y_max, mesh_pkg.ny + 1)),
        dtype=float,
    )
    node_grid = _build_node_grid_index(
        mesh_pkg.node_coords, xs, ys, require_complete=False)

    ny1 = mesh_pkg.ny + 1
    nx1 = mesh_pkg.nx + 1
    X = np.full((ny1, nx1), np.nan, dtype=float)
    Y = np.full((ny1, nx1), np.nan, dtype=float)
    Z = np.full((ny1, nx1), np.nan, dtype=float)
    for iy in range(ny1):
        for ix in range(nx1):
            nid = int(node_grid[ix, iy])
            if nid < 0:
                continue
            X[iy, ix] = xs[ix]
            Y[iy, ix] = ys[iy]
            Z[iy, ix] = w_per_node[nid]

    return X, Y, Z


def structured_surface_grids_from_nodal_xyz(
    mesh_pkg: MeshPackage,
    nodal_x: np.ndarray,
    nodal_y: np.ndarray,
    nodal_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, Y, Z) 2D arrays for ``plot_surface`` from per-node positions.

    Uses the same structured corner-node layout as
    ``structured_surface_grids`` (via ``_build_node_grid_index``).
    """
    nodal_x = np.asarray(nodal_x, dtype=float).ravel()
    nodal_y = np.asarray(nodal_y, dtype=float).ravel()
    nodal_z = np.asarray(nodal_z, dtype=float).ravel()
    n = mesh_pkg.n_nodes
    if nodal_x.size != n or nodal_y.size != n or nodal_z.size != n:
        raise ValueError("nodal_x, nodal_y, nodal_z must have length n_nodes.")

    xs = np.asarray(
        getattr(mesh_pkg, "x_nodes",
                np.linspace(mesh_pkg.x_min, mesh_pkg.x_max, mesh_pkg.nx + 1)),
        dtype=float,
    )
    ys = np.asarray(
        getattr(mesh_pkg, "y_nodes",
                np.linspace(mesh_pkg.y_min, mesh_pkg.y_max, mesh_pkg.ny + 1)),
        dtype=float,
    )
    node_grid = _build_node_grid_index(
        mesh_pkg.node_coords, xs, ys, require_complete=False)

    ny1 = mesh_pkg.ny + 1
    nx1 = mesh_pkg.nx + 1
    X = np.full((ny1, nx1), np.nan, dtype=float)
    Y = np.full((ny1, nx1), np.nan, dtype=float)
    Z = np.full((ny1, nx1), np.nan, dtype=float)
    for iy in range(ny1):
        for ix in range(nx1):
            nid = int(node_grid[ix, iy])
            if nid < 0:
                continue
            X[iy, ix] = nodal_x[nid]
            Y[iy, ix] = nodal_y[nid]
            Z[iy, ix] = nodal_z[nid]

    return X, Y, Z


def nodal_field(mesh_pkg: MeshPackage,
                u: np.ndarray,
                dof_label: str) -> np.ndarray:
    """Return the nodal values of one DOF (``u``, ``v`` or ``w``).

    Raises ``ValueError`` if the requested DOF is not exposed by the active
    analysis type (e.g. asking for ``w`` in ``in_plane`` mode).
    """
    u = np.asarray(u, dtype=float).ravel()
    n = mesh_pkg.n_nodes
    dof = mesh_pkg.n_dof_per_node
    if u.size != n * dof:
        raise ValueError("u length mismatch for mesh_pkg.")
    offsets = _dof_offsets(mesh_pkg.analysis_type)
    key = str(dof_label).strip().upper()
    if key not in offsets:
        raise ValueError(
            f"DOF label {dof_label!r} is not available in analysis_type "
            f"'{mesh_pkg.analysis_type}'."
        )
    offset = offsets[key]
    return u[offset::dof].copy()


def nodal_deformed_xyz(
    mesh_pkg: MeshPackage,
    u: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Current-configuration mid-surface coordinates **x = X + u** (3D).

    Reference ``(X, Y)`` comes from ``mesh_pkg.node_coords``; the third
    coordinate is ``0`` in ``in_plane``, ``w`` in ``out_of_plane`` / ``both``.
    """
    u = np.asarray(u, dtype=float).ravel()
    if u.size != mesh_pkg.n_dofs:
        raise ValueError(
            f"u has length {u.size}; expected mesh_pkg.n_dofs={mesh_pkg.n_dofs}."
        )
    Xr = mesh_pkg.node_coords[:, 0].astype(float, copy=False)
    Yr = mesh_pkg.node_coords[:, 1].astype(float, copy=False)
    at = mesh_pkg.analysis_type
    if at == "in_plane":
        uu = nodal_field(mesh_pkg, u, "U")
        vv = nodal_field(mesh_pkg, u, "V")
        return Xr + uu, Yr + vv, np.zeros_like(Xr)
    if at == "out_of_plane":
        ww = nodal_field(mesh_pkg, u, "W")
        return Xr.copy(), Yr.copy(), ww
    if at == "both":
        uu = nodal_field(mesh_pkg, u, "U")
        vv = nodal_field(mesh_pkg, u, "V")
        ww = nodal_field(mesh_pkg, u, "W")
        return Xr + uu, Yr + vv, ww
    raise ValueError(f"Unknown analysis_type {at!r}")


def nodal_w(mesh_pkg: MeshPackage, u: np.ndarray) -> np.ndarray:
    """Out-of-plane virtual displacement w at each node."""
    return nodal_field(mesh_pkg, u, "W")
