"""Leave-one-experiment-out diagnostics.

Information (U_ik) and influence (I_ik) are not the same thing and must never
be collapsed into one number.  A whole experiment is deleted, never a load step.

Fixed-weight shift (eq. 63) is exact for linear GLS at the current weights.
An optional FGLS refit re-estimates Gamma and the virtual fields; the gap is a
result, not a nuisance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

import numpy as np

from .assemble import EstimatingSystem, UncertaintyError, system_from_result
from . import sandwich as ee

U_MATERIAL = np.log(1.10)
I_LARGE = 1.0


def _log_ratio(deleted: float, full: float) -> float:
    if not np.isfinite(deleted) or not np.isfinite(full) or deleted <= 0 or full <= 0:
        return float("nan")
    return float(np.log(deleted / full))


@dataclass
class DeletionRow:
    dropped: str
    experiment_id: int
    n_kept: int
    well_posed: bool
    theta_full: np.ndarray
    theta_deleted_fixed: np.ndarray
    theta_deleted_refit: Optional[np.ndarray]
    covariance_deleted: Optional[np.ndarray]
    delta_fixed: np.ndarray
    delta_refit: Optional[np.ndarray]
    score: np.ndarray
    score_norm: float
    precision_contribution: np.ndarray
    volume_contribution: float
    influence: np.ndarray
    cook_distance: float
    verdicts: List[str]
    reference_covariance: str


def exact_shift(system: EstimatingSystem, dropped: str) -> np.ndarray:
    """θ_(-i) − θ̂ = −(A − A_i)^{-1} ψ_i(θ̂)."""
    index = system.cluster_names.index(dropped)
    cluster = system.clusters[index]
    remaining = ee.sensitivity(system) - cluster.sensitivity
    return -np.linalg.solve(remaining, cluster.score(system.theta))


@dataclass
class ExperimentShift:
    """Eq. (63) for one experiment inside a chosen subset of tests."""

    name: str
    experiment_id: int
    shift: np.ndarray


def _eq63_shift(remaining: np.ndarray, score: np.ndarray) -> np.ndarray:
    remaining = ee.symmetrise(remaining)
    score = np.asarray(score, dtype=float).reshape(-1)
    try:
        return -np.linalg.solve(remaining, score)
    except np.linalg.LinAlgError:
        values, vectors = np.linalg.eigh(remaining)
        vmax = max(float(np.max(np.abs(values))), 1.0)
        floor = 1.0e-12 * vmax
        inverse = vectors @ np.diag(1.0 / np.maximum(values, floor)) @ vectors.T
        inverse = 0.5 * (inverse + inverse.T)
        return -inverse @ score


def subset_deletion_shifts(
    result,
    included_ids: Sequence[int],
) -> List[ExperimentShift]:
    """Eq. (63) using only the ticked tests: A_S = sum_{i in S} A_i.

    For each included experiment i,

        θ̂_(-i) − θ̂ = −(A_S − A_i)^{-1} ψ_i(θ̂)

    in the original parameter units.  Unticked tests are left out of A_S
    and do not appear as series.
    """
    included = {int(eid) for eid in included_ids}
    indices = [
        i
        for i, eid in enumerate(result.cluster_ids)
        if int(eid) in included
    ]
    if len(indices) < 2:
        return []
    blocks = np.asarray(result.information.blocks, dtype=float)
    scores = np.asarray(result.scores, dtype=float)
    A_subset = ee.symmetrise(sum(blocks[i] for i in indices))
    rows: List[ExperimentShift] = []
    for i in indices:
        rows.append(
            ExperimentShift(
                name=str(result.cluster_names[i]),
                experiment_id=int(result.cluster_ids[i]),
                shift=np.asarray(
                    _eq63_shift(A_subset - blocks[i], scores[i]), dtype=float
                ),
            )
        )
    return rows


def verdict(precision: float, influence: float) -> str:
    if not np.isfinite(precision) or not np.isfinite(influence):
        return "undetermined"
    large_influence = influence >= I_LARGE
    if precision < -U_MATERIAL:
        return "harmful" if large_influence else "harmful_quiet"
    if precision > U_MATERIAL:
        return "in_tension" if large_influence else "valuable"
    return "in_tension_redundant" if large_influence else "redundant"


def analyse(
    system: EstimatingSystem,
    kind: str = "CR2",
    refit_fn: Optional[Callable[[int, str], Any]] = None,
) -> List[DeletionRow]:
    """Leave-one-experiment-out for the whole stacking."""
    kind = kind.upper()
    if system.n_clusters < 2:
        return []
    campaign_well_posed = system.robust_is_well_posed
    robust_full = (
        ee.sandwich_covariance(system, kind)
        if campaign_well_posed
        else system.covariance_work
    )
    reference = "V_robust eq. (34)" if campaign_well_posed else "V_work eq. (35)"
    variance_full = np.diag(robust_full)
    sigma_full = ee.standard_errors(robust_full)
    logdet_full = float(np.linalg.slogdet(robust_full)[1])
    bread_inverse = np.linalg.inv(ee.sensitivity(system))

    rows: List[DeletionRow] = []
    for cluster in system.clusters:
        reduced = system.without(cluster.name)
        well_posed = reduced.robust_is_well_posed
        covariance_deleted = (
            ee.sandwich_covariance(reduced, kind) if well_posed else None
        )

        shift = exact_shift(system, cluster.name)
        theta_fixed = system.theta + shift
        delta_fixed = system.theta - theta_fixed

        theta_refit = delta_refit = None
        if refit_fn is not None:
            refit_result = refit_fn(cluster.experiment_id, cluster.name)
            theta_refit = np.asarray(refit_result.final_parameters, dtype=float)
            delta_refit = system.theta - theta_refit
            if well_posed:
                try:
                    names = {
                        item.experiment_id: item.name
                        for item in reduced.clusters
                    }
                    refit_system = system_from_result(
                        refit_result,
                        labels=system.labels,
                        experiment_names=names,
                    )
                    covariance_deleted = ee.sandwich_covariance(refit_system, kind)
                except (UncertaintyError, Exception):
                    pass

        delta = delta_refit if delta_refit is not None else delta_fixed

        if covariance_deleted is not None:
            variance_deleted = np.diag(covariance_deleted)
            precision = np.array(
                [
                    _log_ratio(variance_deleted[k], variance_full[k])
                    for k in range(system.n_parameters)
                ]
            )
            volume = float(np.linalg.slogdet(covariance_deleted)[1]) - logdet_full
        else:
            precision = np.full(system.n_parameters, np.nan)
            volume = float("nan")

        influence = np.abs(delta) / np.clip(sigma_full, np.finfo(float).tiny, None)
        cook = float(
            delta @ np.linalg.solve(robust_full, delta) / system.n_parameters
        )
        score = cluster.score(system.theta)

        rows.append(
            DeletionRow(
                dropped=cluster.name,
                experiment_id=cluster.experiment_id,
                n_kept=system.n_clusters - 1,
                well_posed=well_posed,
                theta_full=system.theta.copy(),
                theta_deleted_fixed=theta_fixed,
                theta_deleted_refit=theta_refit,
                covariance_deleted=covariance_deleted,
                delta_fixed=delta_fixed,
                delta_refit=delta_refit,
                score=score,
                score_norm=float(score @ bread_inverse @ score),
                precision_contribution=precision,
                volume_contribution=volume,
                influence=influence,
                cook_distance=cook,
                verdicts=[
                    verdict(precision[k], influence[k])
                    for k in range(system.n_parameters)
                ],
                reference_covariance=reference,
            )
        )
    return rows
