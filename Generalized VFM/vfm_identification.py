# =============================================================================
#  plate / vfm_identification.py
# -----------------------------------------------------------------------------
#  Outer iteration loop of the plate VFM inverse problem.  Identical control
#  flow to the membrane code -- only the import paths and the parameter
#  labels are different.  The outer loop is generic in ``n_parameters`` so
#  the four-parameter isotropic-plate case works without any plate-specific
#  branching here.
#
#  Iteration body, per outer step:
#
#    1. Re-build the plate ``ConstitutiveModel`` with the current parameter
#       vector ``theta``.
#    2. Solve the optimal virtual field for every constitutive parameter
#       index ``k`` in ``[0, n_parameters)``.
#    3. Assemble the identification system
#           K[k, j] = (u_k)^T A_j
#           U[k]    = (external_work_matrix.T @ u_k) . applied_force
#    4. Solve  K theta_new = U  (then apply ``solution_scale`` to convert
#       units if necessary).
#    5. Check the legacy ``deltaQ`` convergence rule.
#
#  Applied force / moment resolution
#  ---------------------------------
#  The ``applied_force`` vector used in step 3 is resolved once, up front,
#  by :func:`_resolve_applied_force`:
#    * If the MAT file provided an ``F`` field (legacy path), it is used
#      directly (broadcast to the number of external groups if scalar).
#    * Otherwise, the per-entry ``value`` fields declared in the BC file's
#      ``EXTERNAL`` list (writable from the GUI) are used.  Every entry
#      must carry a finite ``value``; a missing one produces a clear
#      error pointing the user back to the BC file / GUI.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from vfm_constitutive import (
    ConstitutiveModel,
    define_constitutive_model,
    rotate_constitutive_model,
)
from vfm_data_io import BoundaryConditions, PointMeasurements
from vfm_mesh import (
    MeshPackage,
    PointMapping,
    build_external_work_matrix,
    map_points_to_mesh,
)
from vfm_optimal_field import (
    OptimalFieldResult,
    OptimalFieldSolverOptions,
    solve_optimal_virtual_field,
)
from vfm_user_inputs import IdentificationSettings, LinearAlgebraSettings

# =============================================================================
#  Where an experiment's applied load comes from
# -----------------------------------------------------------------------------
#  'mat'    -- the measurement file's F field wins; the BC 'value' entries are
#              the fallback.  The historical behaviour and still the default.
#  'manual' -- the per-time-step 'value' entries win; F is the fallback.  This
#              is what makes the load columns in the experiment dialog bite on
#              data that also ships an F.
# =============================================================================

APPLIED_LOAD_FROM_MAT = "mat"
APPLIED_LOAD_MANUAL = "manual"
APPLIED_LOAD_SOURCES = (APPLIED_LOAD_FROM_MAT, APPLIED_LOAD_MANUAL)


def normalise_applied_load_source(source) -> str:
    """Validate and canonicalise an applied-load source string."""
    if source is None:
        return APPLIED_LOAD_FROM_MAT
    key = str(source).strip().lower()
    if key in ("", "auto", "file", "mat"):
        return APPLIED_LOAD_FROM_MAT
    if key in ("manual", "table", "per_step", "per-step"):
        return APPLIED_LOAD_MANUAL
    raise ValueError(
        f"Unknown applied-load source {source!r}; expected one of "
        f"{APPLIED_LOAD_SOURCES}.")


# =============================================================================
#  Iteration record + final result containers
# =============================================================================

@dataclass
class IterationRecord:
    """One row of the identification history."""
    iteration: int
    parameters_before: np.ndarray
    parameters_after: np.ndarray
    delta_q: float
    residual_norm: float
    condition_number: float
    identity_deviation: float


@dataclass
class IdentificationCase:
    """One measurement/mesh/BC bundle in a stacked identification run.

    ``material_orientation_deg`` is the angle from this case's own measurement
    frame (+x) to the material 1-axis, counter-clockwise positive.  Its fields,
    mesh, boundary conditions and applied load all stay in that measurement
    frame; the constitutive matrix is rotated into it instead.  ``0.0`` -- the
    default -- means the case was measured in the material frame already, and
    is bit-identical to a run predating this field.
    """
    measurements: PointMeasurements
    mesh_pkg: MeshPackage
    boundary_conditions: BoundaryConditions
    label: str = ""
    data_file: Optional[Path] = None
    material_orientation_deg: float = 0.0
    #  'mat' (default) or 'manual' -- see APPLIED_LOAD_SOURCES above.
    applied_load_source: str = APPLIED_LOAD_FROM_MAT


@dataclass
class IdentificationResult:
    """Output of :func:`run_identification`."""
    initial_parameters: np.ndarray
    final_parameters: np.ndarray
    converged: bool
    iterations_completed: int
    history: List[IterationRecord] = field(default_factory=list)
    optimal_field_results: List[OptimalFieldResult] = field(default_factory=list)
    optimal_nodal_history: List[List[np.ndarray]] = field(default_factory=list)
    K_matrix: Optional[np.ndarray] = None
    U_vector: Optional[np.ndarray] = None
    case_K_matrices: List[np.ndarray] = field(default_factory=list)
    case_U_vectors: List[np.ndarray] = field(default_factory=list)
    case_labels: List[str] = field(default_factory=list)
    case_data_files: List[Optional[Path]] = field(default_factory=list)
    diagnostic_cases: List[IdentificationCase] = field(default_factory=list)
    constitutive_model: Optional[ConstitutiveModel] = None


# =============================================================================
#  Public entry point
# =============================================================================

def run_identification(
    measurements: PointMeasurements,
    mesh_pkg: MeshPackage,
    boundary_conditions: BoundaryConditions,
    material_type: str,
    analysis_type: str,
    initial_parameters,
    identification_settings: IdentificationSettings,
    linear_algebra_settings: LinearAlgebraSettings,
    verbose: bool = True,
) -> IdentificationResult:
    """Run the explicit ``K theta = U`` plate identification loop."""
    case = IdentificationCase(
        measurements=measurements,
        mesh_pkg=mesh_pkg,
        boundary_conditions=boundary_conditions,
        label="test 1",
    )
    return run_identification_cases(
        cases=[case],
        material_type=material_type,
        analysis_type=analysis_type,
        initial_parameters=initial_parameters,
        identification_settings=identification_settings,
        linear_algebra_settings=linear_algebra_settings,
        verbose=verbose,
    )


@dataclass
class _PreparedCase:
    case: IdentificationCase
    mapping: PointMapping
    external_work_matrix: np.ndarray
    applied_force: np.ndarray


def run_identification_cases(
    cases: Sequence[IdentificationCase],
    material_type: str,
    analysis_type: str,
    initial_parameters,
    identification_settings: IdentificationSettings,
    linear_algebra_settings: LinearAlgebraSettings,
    verbose: bool = True,
) -> IdentificationResult:
    """Run the plate identification loop over ordered same-type test cases."""
    cases = list(cases)
    if not cases:
        raise ValueError("run_identification_cases requires at least one case.")

    initial_parameters = np.asarray(initial_parameters, dtype=float).ravel().copy()
    current_parameters = initial_parameters.copy()

    optimal_field_options = OptimalFieldSolverOptions(
        svd_tolerance_factor=linear_algebra_settings.svd_tolerance_factor,
        rank_tolerance_floor=linear_algebra_settings.rank_tolerance_floor,
    )

    prepared_cases: List[_PreparedCase] = []
    for idx, case in enumerate(cases, start=1):
        if case.mesh_pkg.analysis_type != str(analysis_type).strip().lower():
            raise ValueError(
                f"Case {idx} mesh analysis_type {case.mesh_pkg.analysis_type!r} "
                f"does not match run analysis_type {analysis_type!r}."
            )
        if case.measurements.analysis_type != str(analysis_type).strip().lower():
            raise ValueError(
                f"Case {idx} measurement analysis_type "
                f"{case.measurements.analysis_type!r} does not match run "
                f"analysis_type {analysis_type!r}."
            )

        mapping = map_points_to_mesh(case.measurements.points, case.mesh_pkg)
        if verbose:
            label = case.label or f"test {idx}"
            print(
                f"[VFM-plate] case {idx}: {label} -- mapped "
                f"{mapping.n_mapped}/"
                f"{mapping.n_mapped + mapping.n_unmapped} "
                f"measurement points into the mesh "
                f"(mesh: {case.mesh_pkg.n_elements} elements, "
                f"{case.mesh_pkg.n_nodes} nodes, {case.mesh_pkg.n_dofs} DOFs "
                f"at {case.mesh_pkg.n_dof_per_node} DOFs/node)"
            )

        external_work_matrix, _ = build_external_work_matrix(
            mesh_pkg=case.mesh_pkg,
            bc=case.boundary_conditions,
        )
        applied_force = _resolve_applied_force(
            measurements=case.measurements,
            boundary_conditions=case.boundary_conditions,
            n_external_groups=external_work_matrix.shape[1],
            applied_load_source=case.applied_load_source,
        )
        if verbose:
            source = normalise_applied_load_source(case.applied_load_source)
            origin = ("per-step values" if source == APPLIED_LOAD_MANUAL
                      else "the .mat F field")
            print(
                f"[VFM-plate] case {idx}: applied force vector: "
                f"{applied_force}  (from {origin})"
            )
        prepared_cases.append(_PreparedCase(
            case=case,
            mapping=mapping,
            external_work_matrix=external_work_matrix,
            applied_force=applied_force,
        ))

    history: List[IterationRecord] = []
    optimal_nodal_history: List[List[np.ndarray]] = []
    last_field_results: List[OptimalFieldResult] = []
    last_K = None
    last_U = None
    last_case_K: List[np.ndarray] = []
    last_case_U: List[np.ndarray] = []
    final_model: Optional[ConstitutiveModel] = None
    converged = False
    stack_weight = float(np.sqrt(1.0 / len(prepared_cases)))

    for iteration in range(1, identification_settings.max_iterations + 1):
        # 1. Build the constitutive model.
        constitutive_model = define_constitutive_model(
            material_type=material_type,
            analysis_type=analysis_type,
            parameters=current_parameters,
        )
        n_parameters = constitutive_model.number_of_parameters

        if verbose:
            theta_str = np.array2string(current_parameters, precision=6)
            print(
                f"[VFM-plate] iteration "
                f"{iteration}/{identification_settings.max_iterations} "
                f"-- current theta = {theta_str}"
            )

        case_K_matrices: List[np.ndarray] = []
        case_U_vectors: List[np.ndarray] = []
        case_field_results: List[List[OptimalFieldResult]] = []

        for case_index, prepared in enumerate(prepared_cases, start=1):
            field_results, K_i, U_i = _assemble_case_system(
                prepared=prepared,
                constitutive_model=constitutive_model,
                optimal_field_options=optimal_field_options,
                n_parameters=n_parameters,
            )
            case_field_results.append(field_results)
            case_K_matrices.append(K_i)
            case_U_vectors.append(U_i)

            if verbose and len(prepared_cases) > 1:
                print(
                    f"[VFM-plate]    case {case_index}: "
                    f"cond(K_i) = {np.linalg.cond(K_i):.3e}"
                )

        K_matrix = stack_weight * np.vstack(case_K_matrices)
        U_vector = stack_weight * np.concatenate(case_U_vectors)

        condition_number = float(np.linalg.cond(K_matrix))
        identity = np.eye(n_parameters)
        if len(prepared_cases) == 1:
            identity_deviation = float(
                np.linalg.norm(K_matrix - identity, ord="fro")
            )
        else:
            normal_matrix = K_matrix.T @ K_matrix
            identity_deviation = float(
                np.linalg.norm(normal_matrix - identity, ord="fro")
            )

        # Store the first case in the legacy optimal-field containers.  The
        # viewer is single-mesh today; run_vfm skips it for multi-test runs.
        first_case_results = case_field_results[0]
        optimal_nodal_history.append(
            [res.nodal_variables_k.copy() for res in first_case_results]
        )

        # 4. Solve stacked K theta ~= U.
        raw_parameters, *_ = np.linalg.lstsq(K_matrix, U_vector, rcond=None)
        next_parameters = (
            identification_settings.solution_scale * raw_parameters
        )
        residual_norm = float(
            np.linalg.norm(K_matrix @ raw_parameters - U_vector)
        )

        # 5. deltaQ convergence.
        delta_q = float(np.sum(
            (current_parameters - next_parameters) ** 2
            / np.maximum(next_parameters ** 2, np.finfo(float).eps)
        ))

        history.append(IterationRecord(
            iteration=iteration,
            parameters_before=current_parameters.copy(),
            parameters_after=next_parameters.copy(),
            delta_q=delta_q,
            residual_norm=residual_norm,
            condition_number=condition_number,
            identity_deviation=identity_deviation,
        ))

        if verbose:
            print(
                f"[VFM-plate]    deltaQ = {delta_q:.3e}, "
                f"residual = {residual_norm:.3e}, "
                f"cond(K) = {condition_number:.3e}, "
                f"||K - I||_F = {identity_deviation:.3e}"
            )

        current_parameters = next_parameters
        last_field_results = first_case_results
        last_K = K_matrix
        last_U = U_vector
        last_case_K = [arr.copy() for arr in case_K_matrices]
        last_case_U = [arr.copy() for arr in case_U_vectors]
        final_model = constitutive_model

        if delta_q <= identification_settings.convergence_tolerance:
            converged = True
            if verbose:
                print(f"[VFM-plate] converged at iteration {iteration}.")
            break

    iterations_completed = len(history)
    if iterations_completed == 0:
        raise RuntimeError(
            "Identification stage did not complete any iteration -- check "
            "max_iterations in your user inputs."
        )

    final_model = define_constitutive_model(
        material_type=material_type,
        analysis_type=analysis_type,
        parameters=current_parameters,
    )

    assert len(optimal_nodal_history) == len(history) == iterations_completed

    return IdentificationResult(
        initial_parameters=initial_parameters,
        final_parameters=current_parameters,
        converged=converged,
        iterations_completed=iterations_completed,
        history=history,
        optimal_field_results=last_field_results,
        optimal_nodal_history=optimal_nodal_history,
        K_matrix=last_K,
        U_vector=last_U,
        case_K_matrices=last_case_K,
        case_U_vectors=last_case_U,
        case_labels=[
            case.label or f"test {idx}"
            for idx, case in enumerate(cases, start=1)
        ],
        case_data_files=[case.data_file for case in cases],
        diagnostic_cases=list(cases),
        constitutive_model=final_model,
    )


def _assemble_case_system(
    prepared: _PreparedCase,
    constitutive_model: ConstitutiveModel,
    optimal_field_options: OptimalFieldSolverOptions,
    n_parameters: int,
):
    """Solve optimal fields and assemble one case-level K_i, U_i block."""
    field_results: List[OptimalFieldResult] = []
    case = prepared.case
    #  This case's fields are in ITS OWN measurement frame, so the constitutive
    #  matrix is rotated into that frame.  theta itself is untouched -- it stays
    #  material-frame and shared, which is what lets differently-oriented cases
    #  stack into one system.  A zero angle returns the model unchanged.
    case_model = rotate_constitutive_model(constitutive_model,
                                           case.material_orientation_deg)
    for k in range(n_parameters):
        field_result = solve_optimal_virtual_field(
            parameter_index_k=k,
            measurements=case.measurements,
            mesh_pkg=case.mesh_pkg,
            boundary_conditions=case.boundary_conditions,
            constitutive_model=case_model,
            options=optimal_field_options,
            mapping=prepared.mapping,
        )
        field_results.append(field_result)

    A_matrix = field_results[0].A          # (n_dofs, n_parameters)
    K_matrix = np.zeros((n_parameters, n_parameters), dtype=float)
    U_vector = np.zeros(n_parameters, dtype=float)

    for k, field_result in enumerate(field_results):
        u_k = field_result.nodal_variables_k
        K_matrix[k, :] = u_k @ A_matrix

        generalised_displacement = prepared.external_work_matrix.T @ u_k
        U_vector[k] = float(
            np.dot(generalised_displacement, prepared.applied_force)
        )

    return field_results, K_matrix, U_vector


# =============================================================================
#  Applied force / moment resolution
# =============================================================================

def _resolve_applied_force(
    measurements: PointMeasurements,
    boundary_conditions: BoundaryConditions,
    n_external_groups: int,
    applied_load_source: str = APPLIED_LOAD_FROM_MAT,
) -> np.ndarray:
    """Build the per-external-group applied-force vector for one run.

    ``applied_load_source`` picks which of the two sources is authoritative:

    ``'mat'`` (the default, and the historical behaviour)
        1. ``measurements.applied_force`` (the MAT file's ``F`` field) when
           it is provided.  A scalar is broadcast to ``n_external_groups``.
        2. The per-entry ``value`` fields declared in the BC file's
           ``EXTERNAL`` list (writable from the GUI).

    ``'manual'``
        The two are swapped: the declared ``value`` fields win, and the MAT
        file's ``F`` is the fallback.  This is what makes the per-time-step
        load columns in the experiment dialog usable on data that ALSO ships
        an ``F`` -- without it the typed numbers are read, stored, and then
        silently ignored at solve time.

    Either way, if neither source has the data a ``ValueError`` says where to
    add it.  Returns a length-``n_external_groups`` 1-D float array.
    """
    if n_external_groups == 0:
        # An empty EXTERNAL list is unusual but we should not crash here:
        # the K-theta=U row count is still zero so the identification is
        # ill-posed regardless.  Let the caller fail later with a more
        # informative downstream message.
        return np.zeros(0, dtype=float)

    #  build_external_work_matrix() returns a SINGLE placeholder column when
    #  the EXTERNAL list is empty, so n_external_groups == 1 is ambiguous: it
    #  means either one real group or none at all.  Say which, or the user goes
    #  hunting for a size mismatch that is really a missing BC definition.
    if not boundary_conditions.external:
        if measurements.applied_force is None:
            supplied = "and no 'F' was supplied either"
        else:
            n_force = int(np.asarray(measurements.applied_force).size)
            supplied = (f"while the measurement file supplies an 'F' with "
                        f"{n_force} entr{'y' if n_force == 1 else 'ies'}")
        raise ValueError(
            "This experiment defines NO EXTERNAL work group, so there is "
            f"nothing for the applied load to act on -- {supplied}. Open the "
            "boundary-condition editor and add one EXTERNAL group per loaded "
            "edge (they are what the applied moment does work through). Note "
            "this is NOT a size mismatch: the count is zero, not one."
        )

    source = normalise_applied_load_source(applied_load_source)

    if source == APPLIED_LOAD_MANUAL:
        declared = _applied_force_from_bc(boundary_conditions,
                                          n_external_groups)
        if declared is not None:
            return declared
        if measurements.applied_force is None:
            #  Nothing anywhere.  Say so in the manual-mode vocabulary rather
            #  than the default one, or the message points at the wrong knob.
            raise ValueError(
                "This experiment is set to take its applied load from the "
                "per-time-step values, but not every EXTERNAL group of this "
                "step has one, and the measurement file has no 'F' field to "
                "fall back on.  Fill in every load column for this step, or "
                "switch the experiment back to using the .mat 'F' field."
            )

    # ---- Path 1: F from the MAT file ------------------------------------
    if measurements.applied_force is not None:
        applied_force = np.asarray(measurements.applied_force,
                                    dtype=float).ravel()
        if applied_force.size == 1 and n_external_groups > 1:
            applied_force = np.repeat(applied_force, n_external_groups)
        if applied_force.size != n_external_groups:
            raise ValueError(
                f"Applied force from the measurement file has "
                f"{applied_force.size} entries but the BC file declares "
                f"{n_external_groups} external-work group(s).  Either "
                f"fix the size of 'F' in the MAT file, or remove it and "
                "set the 'value' on each EXTERNAL entry instead."
            )
        return applied_force

    # ---- Path 2: per-entry 'value' from the BC file ---------------------
    n_bc_groups = len(boundary_conditions.external)
    if n_bc_groups != n_external_groups:
        # This should be impossible in practice (build_external_work_matrix
        # uses the same list) but be explicit anyway.
        raise ValueError(
            f"Internal inconsistency: BC file declares {n_bc_groups} "
            f"EXTERNAL group(s) but the external-work operator has "
            f"{n_external_groups} columns."
        )

    declared = _applied_force_from_bc(boundary_conditions, n_external_groups)
    if declared is not None:
        return declared

    labels = []
    for i, entry in enumerate(boundary_conditions.external):
        raw_value = entry.get("value", None) if isinstance(entry, dict) else None
        if raw_value is None or not _is_finite_number(raw_value):
            label = (entry.get("label", "<unlabeled>")
                     if isinstance(entry, dict) else "<invalid>")
            labels.append(f"#{i} '{label}'")
    raise ValueError(
        "No applied force/moment was provided: the measurement file "
        "is missing the 'F' field AND the following BC EXTERNAL "
        "entries do not declare a 'value': "
        + "; ".join(labels)
        + ".  Either add 'F' to the MAT file, or open the BC GUI "
        "and set a 'value' on every EXTERNAL entry."
    )


def _is_finite_number(raw_value) -> bool:
    try:
        return bool(np.isfinite(float(raw_value)))
    except (TypeError, ValueError):
        return False


def _applied_force_from_bc(boundary_conditions: BoundaryConditions,
                           n_external_groups: int) -> Optional[np.ndarray]:
    """The declared per-group ``value`` vector, or ``None`` if incomplete.

    All-or-nothing on purpose: a partially filled set would silently mix a
    typed load on one edge with the MAT file's on another, which is a
    different experiment from the one the user thinks they described.
    """
    external = boundary_conditions.external
    if len(external) != n_external_groups:
        return None

    values: List[float] = []
    for entry in external:
        if not isinstance(entry, dict):
            return None
        raw_value = entry.get("value", None)
        if raw_value is None or not _is_finite_number(raw_value):
            return None
        values.append(float(raw_value))
    return np.asarray(values, dtype=float)
