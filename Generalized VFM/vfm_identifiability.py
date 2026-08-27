# =============================================================================
#  plate / vfm_identifiability.py
# -----------------------------------------------------------------------------
#  Fisher-like constitutive identifiability diagnostics for the plate inverse
#  solver.  The formulas mirror the Q-matrix Pierron-style diagnostics, but
#  use the active plate ConstitutiveModel directly so iso / orthotropic /
#  anisotropic and in-plane / bending / ABD modes share one implementation.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from vfm_constitutive import ConstitutiveModel, rotate_constitutive_model
from vfm_data_io import PointMeasurements
from vfm_mesh import MeshPackage, subset_measurements_to_mapped_mesh


@dataclass
class InformationMatrixDiagnostics:
    """Matrix-level diagnostics for one Fisher-like information matrix."""
    H: np.ndarray
    singular_values: np.ndarray
    determinant: float
    slogdet_sign: float
    slogdet: float
    condition_number: float
    covariance: np.ndarray
    correlation: np.ndarray
    used_pseudoinverse: bool


@dataclass
class IdentifiabilityCaseDiagnostics:
    """Identifiability diagnostics for one measurement case."""
    label: str
    data_file: Optional[Path]
    points: np.ndarray
    CID: np.ndarray
    sensitivities: np.ndarray
    matrix: InformationMatrixDiagnostics
    area_per_point: float
    material_solver_used_pseudoinverse: bool = False
    mesh_pkg: Optional[MeshPackage] = None
    n_points_total: int = 0
    n_points_used: int = 0


@dataclass
class IdentifiabilityDiagnostics:
    """Complete per-test and aggregate identifiability diagnostics."""
    parameter_labels: List[str]
    material_type: str
    analysis_type: str
    aggregate: InformationMatrixDiagnostics
    cases: List[IdentifiabilityCaseDiagnostics] = field(default_factory=list)


def compute_identifiability_diagnostics(
    cases: Sequence,
    constitutive_model: ConstitutiveModel,
) -> IdentifiabilityDiagnostics:
    """Compute Pierron-style H, CID, and correlation diagnostics.

    Parameters
    ----------
    cases
        Sequence of objects with at least ``measurements`` plus optional
        ``label`` and ``data_file`` attributes.  This intentionally accepts
        ``vfm_identification.IdentificationCase`` without importing it here.
    constitutive_model
        Final active constitutive model for the run, in the MATERIAL frame.
        Each case's own ``material_orientation_deg`` (default 0) rotates it
        into that case's measurement frame before its sensitivities are taken,
        because the strains it is contracted against are in that frame.  The
        resulting ``H`` blocks are still indexed by the shared material-frame
        theta, so summing them across cases stays meaningful.
    """
    cases = list(cases)
    if not cases:
        raise ValueError("At least one identification case is required.")

    labels = list(constitutive_model.parameter_labels)
    case_results: List[IdentifiabilityCaseDiagnostics] = []
    H_total = np.zeros(
        (constitutive_model.number_of_parameters,
         constitutive_model.number_of_parameters),
        dtype=float,
    )

    for idx, case in enumerate(cases, start=1):
        measurements = getattr(case, "measurements", None)
        if measurements is None:
            raise ValueError(f"Case {idx} is missing measurements.")
        label = str(getattr(case, "label", "") or f"test {idx}")
        data_file = getattr(case, "data_file", None)
        mesh_pkg = getattr(case, "mesh_pkg", None)
        case_model = rotate_constitutive_model(
            constitutive_model,
            float(getattr(case, "material_orientation_deg", 0.0) or 0.0))
        result = compute_case_identifiability(
            measurements=measurements,
            constitutive_model=case_model,
            label=label,
            data_file=data_file,
            mesh_pkg=mesh_pkg,
        )
        case_results.append(result)
        H_total += result.matrix.H

    aggregate = _matrix_diagnostics(H_total)
    return IdentifiabilityDiagnostics(
        parameter_labels=labels,
        material_type=constitutive_model.material_type,
        analysis_type=constitutive_model.analysis_type,
        aggregate=aggregate,
        cases=case_results,
    )


def compute_case_identifiability(
    measurements: PointMeasurements,
    constitutive_model: ConstitutiveModel,
    label: str = "test 1",
    data_file: Optional[Path] = None,
    mesh_pkg: Optional[MeshPackage] = None,
) -> IdentifiabilityCaseDiagnostics:
    """Compute H and CID for one measurement field."""
    n_points_total = int(measurements.strains.shape[1])
    if mesh_pkg is not None:
        measurements, _ = subset_measurements_to_mapped_mesh(
            measurements, mesh_pkg)

    M = np.asarray(constitutive_model.material_matrix, dtype=float)
    basis = np.asarray(constitutive_model.basis_matrices, dtype=float)
    strains = np.asarray(measurements.strains, dtype=float)

    n_params = constitutive_model.number_of_parameters
    n_gen = constitutive_model.n_generalised_strain
    if M.shape != (n_gen, n_gen):
        raise ValueError(
            f"material_matrix shape {M.shape} does not match ({n_gen}, {n_gen})."
        )
    if basis.shape != (n_params, n_gen, n_gen):
        raise ValueError(
            f"basis_matrices shape {basis.shape} does not match "
            f"({n_params}, {n_gen}, {n_gen})."
        )
    if strains.ndim != 2 or strains.shape[0] != n_gen:
        raise ValueError(
            f"Measured strains have shape {strains.shape}; expected "
            f"({n_gen}, n_points)."
        )

    n_points = strains.shape[1]
    if n_points == 0:
        raise ValueError("Cannot compute identifiability with zero data points.")
    area_per_point = float(measurements.area_per_point)
    if not np.isfinite(area_per_point) or area_per_point <= 0.0:
        raise ValueError(
            f"measurements.area_per_point must be positive and finite; "
            f"got {measurements.area_per_point!r}."
        )

    sensitivities, used_material_pinv = _constitutive_sensitivities(
        material_matrix=M,
        basis_matrices=basis,
        strains=strains,
    )
    CID = np.einsum("gpk,gpk->kp", sensitivities, sensitivities)
    H = area_per_point * np.einsum("gpi,gpj->ij", sensitivities, sensitivities)
    H = 0.5 * (H + H.T)

    return IdentifiabilityCaseDiagnostics(
        label=label,
        data_file=Path(data_file) if data_file is not None else None,
        points=np.asarray(measurements.points, dtype=float).copy(),
        CID=CID,
        sensitivities=sensitivities,
        matrix=_matrix_diagnostics(H),
        area_per_point=area_per_point,
        material_solver_used_pseudoinverse=used_material_pinv,
        mesh_pkg=mesh_pkg,
        n_points_total=n_points_total,
        n_points_used=n_points,
    )


def _constitutive_sensitivities(
    material_matrix: np.ndarray,
    basis_matrices: np.ndarray,
    strains: np.ndarray,
):
    n_params = basis_matrices.shape[0]
    n_gen, n_points = strains.shape
    sensitivities = np.zeros((n_gen, n_points, n_params), dtype=float)
    used_pinv = False

    for j in range(n_params):
        rhs = basis_matrices[j] @ strains
        try:
            sensitivities[:, :, j] = -np.linalg.solve(material_matrix, rhs)
        except np.linalg.LinAlgError:
            used_pinv = True
            sensitivities[:, :, j] = -(np.linalg.pinv(material_matrix) @ rhs)

    return sensitivities, used_pinv


def _matrix_diagnostics(H: np.ndarray) -> InformationMatrixDiagnostics:
    H = np.asarray(H, dtype=float)
    H = 0.5 * (H + H.T)
    singular_values = np.linalg.svd(H, compute_uv=False)
    determinant = float(np.linalg.det(H))
    sign, logabsdet = np.linalg.slogdet(H)
    condition_number = float(np.linalg.cond(H))

    use_pinv = _should_use_pseudoinverse(H, singular_values)
    if use_pinv:
        covariance = np.linalg.pinv(H)
    else:
        try:
            covariance = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            use_pinv = True
            covariance = np.linalg.pinv(H)

    correlation = _correlation_from_covariance(covariance)
    return InformationMatrixDiagnostics(
        H=H,
        singular_values=singular_values,
        determinant=determinant,
        slogdet_sign=float(sign),
        slogdet=float(logabsdet),
        condition_number=condition_number,
        covariance=covariance,
        correlation=correlation,
        used_pseudoinverse=bool(use_pinv),
    )


def _should_use_pseudoinverse(H: np.ndarray,
                              singular_values: np.ndarray) -> bool:
    if H.size == 0 or singular_values.size == 0:
        return True
    max_sv = float(np.max(singular_values))
    min_sv = float(np.min(singular_values))
    if not np.isfinite(max_sv) or not np.isfinite(min_sv):
        return True
    if max_sv <= 0.0:
        return True
    tolerance = np.finfo(float).eps * max(H.shape) * max_sv
    return min_sv <= tolerance


def _correlation_from_covariance(covariance: np.ndarray) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    diag = np.diag(covariance)
    denom_sq = np.outer(diag, diag)
    denom = np.full_like(covariance, np.nan, dtype=float)
    positive = np.isfinite(denom_sq) & (denom_sq > 0.0)
    denom[positive] = np.sqrt(denom_sq[positive])
    correlation = np.full_like(covariance, np.nan, dtype=float)
    valid = np.isfinite(denom) & (denom > 0.0)
    np.divide(covariance, denom, out=correlation, where=valid)
    return correlation
