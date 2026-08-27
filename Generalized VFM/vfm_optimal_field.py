# =============================================================================
#  plate / vfm_optimal_field.py
# -----------------------------------------------------------------------------
#  Analysis-type aware optimal-virtual-field solver.  Same five-block
#  algorithm as before; shape literals are read from the mesh package and
#  constitutive model so the solver works for:
#
#      analysis_type == 'in_plane'      ->  B (N, 3, 8), material 3x3
#      analysis_type == 'out_of_plane'  ->  B (N, 3, 16), material 3x3
#      analysis_type == 'both'          ->  B (N, 6, 24), material 6x6
#
#  Algorithm (unchanged):
#  ---------------------------------------------------------------------------
#  Block 1.  VIRTUAL STIFFNESS MATRIX  G
#      G(theta) = sum_p  w_p * B(x_p)^T * M^T * M * B(x_p)
#      where M is the active material matrix (A, D, or ABD).
#
#  Block 2.  VIRTUAL FORCE VECTOR  A
#      A[:, j] = sum_p  w_p * B(x_p)^T * (dM/dtheta_j) * eps_gen(x_p)
#
#  Block 3.  KA constraints  C u = 0  ->  null-space basis R.
#
#  Block 4.  G_KA = R^T G R,  A_KA = R^T A.
#
#  Block 5.  Saddle-point solve and lift
#      | G_KA    A_KA  |  | z_k   |     |  0  |
#      | A_KA^T  0     |  | -mu_k | =   | e_k |
#      ->  u_k = R z_k
#      The stationarity row is scaled in the same spirit as Pierron's
#      bending script to reduce numerical conditioning warnings.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from vfm_constitutive import ConstitutiveModel
from vfm_data_io import BoundaryConditions, PointMeasurements
from vfm_mesh import (
    ConstraintSystem,
    MeshPackage,
    PointMapping,
    build_admissible_basis,
    build_kinematic_constraints,
    map_points_to_mesh,
    strain_displacement_batch,
)


# =============================================================================
#  Helper data containers
# =============================================================================

@dataclass
class OptimalFieldSolverOptions:
    svd_tolerance_factor: float = 100.0
    rank_tolerance_floor: float = 1.0e-12
    inactive_dof_threshold: float = 1.0e-14
    use_pierron_saddle_scaling: bool = True


@dataclass
class OptimalFieldResult:
    nodal_variables_k: np.ndarray
    G: np.ndarray
    A: np.ndarray
    R: np.ndarray
    G_KA: np.ndarray
    A_KA: np.ndarray
    constraint_system: ConstraintSystem
    mapping: PointMapping
    diagnostics: Dict[str, float]


# =============================================================================
#  Public entry point
# =============================================================================

def solve_optimal_virtual_field(
    parameter_index_k: int,
    measurements: PointMeasurements,
    mesh_pkg: MeshPackage,
    boundary_conditions: BoundaryConditions,
    constitutive_model: ConstitutiveModel,
    options: OptimalFieldSolverOptions = None,
    mapping: PointMapping = None,
    payload: "_AssemblyPayload" = None,
) -> OptimalFieldResult:
    """Solve for the optimal plate virtual field associated with parameter k."""
    if options is None:
        options = OptimalFieldSolverOptions()

    n_parameters = constitutive_model.number_of_parameters
    if not (0 <= parameter_index_k < n_parameters):
        raise ValueError(
            f"parameter_index_k={parameter_index_k} is out of range; "
            f"expected 0..{n_parameters - 1}."
        )

    n_gen = constitutive_model.n_generalised_strain
    if constitutive_model.material_matrix.shape != (n_gen, n_gen):
        raise ValueError(
            "Constitutive material_matrix has shape "
            f"{constitutive_model.material_matrix.shape}; "
            f"expected ({n_gen}, {n_gen})."
        )
    if constitutive_model.basis_matrices.shape != (n_parameters, n_gen, n_gen):
        raise ValueError(
            "Constitutive basis_matrices has shape "
            f"{constitutive_model.basis_matrices.shape}; expected "
            f"({n_parameters}, {n_gen}, {n_gen})."
        )

    # Cross-check that the mesh and constitutive model agree on the analysis
    # type.  Mismatched modes would silently produce a wrongly-shaped B.
    if mesh_pkg.analysis_type != constitutive_model.analysis_type:
        raise ValueError(
            f"Mesh analysis_type {mesh_pkg.analysis_type!r} does not match "
            f"constitutive analysis_type "
            f"{constitutive_model.analysis_type!r}."
        )

    return solve_optimal_virtual_fields(
        measurements=measurements,
        mesh_pkg=mesh_pkg,
        boundary_conditions=boundary_conditions,
        constitutive_model=constitutive_model,
        options=options,
        mapping=mapping,
        payload=payload,
        parameter_indices=(parameter_index_k,),
    )[0]


def solve_optimal_virtual_fields(
    measurements: PointMeasurements,
    mesh_pkg: MeshPackage,
    boundary_conditions: BoundaryConditions,
    constitutive_model: ConstitutiveModel,
    options: OptimalFieldSolverOptions = None,
    mapping: PointMapping = None,
    payload: "_AssemblyPayload" = None,
    parameter_indices=None,
):
    """Solve the optimal virtual fields for several parameters at once.

    ``solve_optimal_virtual_field`` re-assembles G, A, R and the saddle-point
    matrix on every call, so identifying p parameters costs p identical
    assemblies per case per iteration.  This entry point assembles once and
    solves the p right-hand sides together.  The saddle-point matrix does not
    depend on ``parameter_index_k`` -- only the unit RHS does -- so the results
    are numerically identical to calling the single-parameter routine p times
    (same LU factorisation, one column per k).

    Returns a list of ``OptimalFieldResult``, one per entry of
    ``parameter_indices`` (default: every parameter, in order).
    """
    if options is None:
        options = OptimalFieldSolverOptions()

    n_parameters = constitutive_model.number_of_parameters
    if parameter_indices is None:
        parameter_indices = tuple(range(n_parameters))
    parameter_indices = tuple(int(k) for k in parameter_indices)
    for k in parameter_indices:
        if not (0 <= k < n_parameters):
            raise ValueError(
                f"parameter index {k} is out of range; "
                f"expected 0..{n_parameters - 1}."
            )

    n_gen = constitutive_model.n_generalised_strain
    if constitutive_model.material_matrix.shape != (n_gen, n_gen):
        raise ValueError(
            "Constitutive material_matrix has shape "
            f"{constitutive_model.material_matrix.shape}; "
            f"expected ({n_gen}, {n_gen})."
        )
    if constitutive_model.basis_matrices.shape != (n_parameters, n_gen, n_gen):
        raise ValueError(
            "Constitutive basis_matrices has shape "
            f"{constitutive_model.basis_matrices.shape}; expected "
            f"({n_parameters}, {n_gen}, {n_gen})."
        )
    if mesh_pkg.analysis_type != constitutive_model.analysis_type:
        raise ValueError(
            f"Mesh analysis_type {mesh_pkg.analysis_type!r} does not match "
            f"constitutive analysis_type "
            f"{constitutive_model.analysis_type!r}."
        )

    if mapping is None:
        mapping = map_points_to_mesh(measurements.points, mesh_pkg)

    if payload is None:
        payload = _collect_assembly_payload(
            measurements, mesh_pkg, mapping, n_gen)

    G = _assemble_virtual_stiffness_matrix(
        measurements=measurements,
        mesh_pkg=mesh_pkg,
        mapping=mapping,
        material_matrix=constitutive_model.material_matrix,
        payload=payload,
    )

    A, dofs_touched_by_data = _assemble_virtual_force_vector(
        measurements=measurements,
        mesh_pkg=mesh_pkg,
        mapping=mapping,
        basis_matrices=constitutive_model.basis_matrices,
        payload=payload,
    )

    row_activity = (
        np.abs(G).sum(axis=1) + np.abs(A).sum(axis=1)
    )
    inactive_mask = (row_activity <= options.inactive_dof_threshold) & \
        (~dofs_touched_by_data)
    inactive_dofs = np.where(inactive_mask)[0]

    constraint_system = build_kinematic_constraints(
        mesh_pkg=mesh_pkg,
        bc=boundary_conditions,
        inactive_dofs=inactive_dofs,
    )

    R, admissibility_metadata = build_admissible_basis(
        C=constraint_system.C,
        n_dofs=mesh_pkg.n_dofs,
        svd_tolerance_factor=options.svd_tolerance_factor,
        rank_tolerance_floor=options.rank_tolerance_floor,
    )

    G_KA = R.T @ G @ R
    G_KA = 0.5 * (G_KA + G_KA.T)
    A_KA = R.T @ A

    nullity = R.shape[1]

    saddle_scale = 1.0
    if options.use_pierron_saddle_scaling:
        max_g = float(np.max(np.abs(G_KA))) if G_KA.size else 0.0
        max_a = float(np.max(np.abs(A_KA))) if A_KA.size else 0.0
        if (np.isfinite(max_g) and np.isfinite(max_a)
                and max_g > 0.0 and max_a > 0.0):
            saddle_scale = max_a / max_g

    top_block = saddle_scale * np.hstack([G_KA, A_KA])
    bottom_block = np.hstack([A_KA.T, np.zeros((n_parameters, n_parameters))])
    K_VFM = np.vstack([top_block, bottom_block])

    F_VFM = np.zeros((nullity + n_parameters, len(parameter_indices)),
                     dtype=float)
    for column, k in enumerate(parameter_indices):
        F_VFM[nullity + k, column] = 1.0

    expected = nullity + n_parameters
    assert K_VFM.shape == (expected, expected), K_VFM.shape

    from scipy.linalg import solve as _solve_dense
    # Pierron's scaling multiplies only the stationarity equations, leaving
    # the speciality equations untouched.  That keeps the solution unchanged
    # because the top RHS is zero, but the matrix is no longer symmetric.
    solution = _solve_dense(K_VFM, F_VFM)

    z_all = solution[:nullity, :]
    nodal_variables_all = R @ z_all                      # (n_dofs, n_solved)

    diagnostics = {
        "n_dofs": mesh_pkg.n_dofs,
        "n_parameters": n_parameters,
        "n_mapped_points": mapping.n_mapped,
        "n_unmapped_points": mapping.n_unmapped,
        "n_inactive_dofs": int(inactive_dofs.size),
        "constraint_rank": admissibility_metadata["rank"],
        "admissible_nullity": admissibility_metadata["nullity"],
        "svd_tolerance": admissibility_metadata["tolerance"],
        "n_dofs_per_elem": payload.n_dofs_per_elem,
        "n_generalised_strain": n_gen,
        "pierron_saddle_scale": float(saddle_scale),
    }

    return [
        OptimalFieldResult(
            nodal_variables_k=nodal_variables_all[:, column],
            G=G,
            A=A,
            R=R,
            G_KA=G_KA,
            A_KA=A_KA,
            constraint_system=constraint_system,
            mapping=mapping,
            diagnostics=diagnostics,
        )
        for column in range(len(parameter_indices))
    ]


# =============================================================================
#  Vectorised assembly helpers
# =============================================================================

@dataclass
class _AssemblyPayload:
    n_mapped: int
    n_dofs_per_elem: int
    n_gen_strain: int
    element_dofs_all: np.ndarray   # (N, n_dofs_per_elem) int64
    B_all: np.ndarray              # (N, n_gen, n_dofs_per_elem) float
    measured_strains: np.ndarray   # (n_gen, N) float
    flat_idx: np.ndarray
    n_dofs: int


def collect_assembly_payload(measurements: PointMeasurements,
                             mesh_pkg: MeshPackage,
                             mapping: PointMapping,
                             n_gen_strain: int) -> "_AssemblyPayload":
    """Public alias: the per-point B operator and its global DOF scatter map.

    The GLS covariance propagation needs B(x_p) explicitly in order to build
    the generalised virtual kinematics E*_{t,p} = B_p U_t, so this assembly is
    no longer purely internal to the optimal-field solve.
    """
    return _collect_assembly_payload(
        measurements, mesh_pkg, mapping, n_gen_strain)


def _collect_assembly_payload(measurements: PointMeasurements,
                              mesh_pkg: MeshPackage,
                              mapping: PointMapping,
                              n_gen_strain: int) -> _AssemblyPayload:
    n_dof_per_node = mesh_pkg.n_dof_per_node
    n_nodes_per_elem = mesh_pkg.element_nodes.shape[1]
    n_dofs_per_elem = n_nodes_per_elem * n_dof_per_node

    mapped_mask = mapping.element_ids >= 0
    element_ids = mapping.element_ids[mapped_mask]
    xi = mapping.local_xy[mapped_mask, 0]
    eta = mapping.local_xy[mapped_mask, 1]
    n_mapped = element_ids.size

    nodes_per_point = mesh_pkg.element_nodes[element_ids]            # (N, 4)
    coords_per_point = mesh_pkg.node_coords[nodes_per_point]         # (N, 4, 2)

    element_dx = np.asarray(
        getattr(mesh_pkg, "element_dx", np.full(mesh_pkg.n_elements, mesh_pkg.dx)),
        dtype=float,
    )
    element_dy = np.asarray(
        getattr(mesh_pkg, "element_dy", np.full(mesh_pkg.n_elements, mesh_pkg.dy)),
        dtype=float,
    )
    dx_per_point = element_dx[element_ids]
    dy_per_point = element_dy[element_ids]

    B_all, _ = strain_displacement_batch(
        analysis_type=mesh_pkg.analysis_type,
        coords_per_point=coords_per_point,
        xi=xi,
        eta=eta,
        dx_per_point=dx_per_point,
        dy_per_point=dy_per_point,
    )

    element_dofs_all = np.empty((n_mapped, n_dofs_per_elem), dtype=np.int64)
    for local_node in range(n_nodes_per_elem):
        base_global = nodes_per_point[:, local_node] * n_dof_per_node
        for dof_component in range(n_dof_per_node):
            element_dofs_all[:,
                             local_node * n_dof_per_node + dof_component] \
                = base_global + dof_component

    measured_strains = measurements.strains[:, mapped_mask]
    if measured_strains.shape[0] != n_gen_strain:
        raise ValueError(
            f"Measured strains have {measured_strains.shape[0]} rows but the "
            f"active constitutive model expects {n_gen_strain}.  Check that "
            "the measurement file matches the analysis_type."
        )

    assert B_all.shape == (n_mapped, n_gen_strain, n_dofs_per_elem), B_all.shape
    assert measured_strains.shape == (n_gen_strain, n_mapped), \
        measured_strains.shape

    return _AssemblyPayload(
        n_mapped=n_mapped,
        n_dofs_per_elem=n_dofs_per_elem,
        n_gen_strain=n_gen_strain,
        element_dofs_all=element_dofs_all,
        B_all=B_all,
        measured_strains=measured_strains,
        flat_idx=element_dofs_all.ravel(),
        n_dofs=mesh_pkg.n_dofs,
    )


# =============================================================================
#  Block 1 helper: virtual stiffness matrix G
# =============================================================================

def _assemble_virtual_stiffness_matrix(
    measurements: PointMeasurements,
    mesh_pkg: MeshPackage,
    mapping: PointMapping,
    material_matrix: np.ndarray,
    payload: "_AssemblyPayload" = None,
) -> np.ndarray:
    if payload is None:
        payload = _collect_assembly_payload(
            measurements, mesh_pkg, mapping,
            n_gen_strain=material_matrix.shape[0])
    n_dofs = payload.n_dofs
    weight = measurements.area_per_point
    weighting_matrix = material_matrix.T @ material_matrix
    B_all = payload.B_all
    element_dofs_all = payload.element_dofs_all
    n_dofs_per_elem = payload.n_dofs_per_elem

    if payload.n_mapped == 0:
        return np.zeros((n_dofs, n_dofs), dtype=float)

    WB = np.einsum("ab,pbj->paj", weighting_matrix, B_all)
    local_G = weight * np.einsum("pai,paj->pij", B_all, WB)
    assert local_G.shape == (payload.n_mapped, n_dofs_per_elem, n_dofs_per_elem)

    rows = np.broadcast_to(element_dofs_all[:, :, None],
                           (payload.n_mapped,
                            n_dofs_per_elem, n_dofs_per_elem))
    cols = np.broadcast_to(element_dofs_all[:, None, :],
                           (payload.n_mapped,
                            n_dofs_per_elem, n_dofs_per_elem))
    flat_keys = rows.ravel() * n_dofs + cols.ravel()
    flat_vals = local_G.ravel()
    G_flat = np.bincount(flat_keys, weights=flat_vals,
                         minlength=n_dofs * n_dofs)
    G = G_flat.reshape(n_dofs, n_dofs)

    return 0.5 * (G + G.T)


# =============================================================================
#  Block 2 helper: virtual force vector A and the touched-DOF mask
# =============================================================================

def _assemble_virtual_force_vector(
    measurements: PointMeasurements,
    mesh_pkg: MeshPackage,
    mapping: PointMapping,
    basis_matrices: np.ndarray,
    payload: "_AssemblyPayload" = None,
):
    n_parameters, n_gen, _ = basis_matrices.shape
    if payload is None:
        payload = _collect_assembly_payload(
            measurements, mesh_pkg, mapping, n_gen_strain=n_gen)
    n_dofs = payload.n_dofs
    weight = measurements.area_per_point

    if basis_matrices.shape != (n_parameters, n_gen, n_gen):
        raise ValueError(
            f"Basis matrices have shape {basis_matrices.shape}; expected "
            f"({n_parameters}, {n_gen}, {n_gen})."
        )

    A = np.zeros((n_dofs, n_parameters), dtype=float)
    touched = np.zeros(n_dofs, dtype=bool)

    if payload.n_mapped > 0:
        B_all = payload.B_all
        measured_strains = payload.measured_strains

        constitutive_loads = np.einsum("jab,bp->paj",
                                       basis_matrices, measured_strains)

        local_A = weight * np.einsum("pai,paj->pij",
                                     B_all, constitutive_loads)
        assert local_A.shape == (payload.n_mapped,
                                  payload.n_dofs_per_elem,
                                  n_parameters)

        flat_idx = payload.flat_idx
        for j in range(n_parameters):
            A[:, j] = np.bincount(flat_idx,
                                  weights=local_A[..., j].ravel(),
                                  minlength=n_dofs)

        touched[np.unique(payload.element_dofs_all)] = True

    return A, touched
