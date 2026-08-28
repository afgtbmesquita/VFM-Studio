"""Pack a GLS identification result into estimating-equation form.

One cluster is one experiment, never a load step: the steps of one coupon share
a DIC reference frame and are correlated by construction.  The solver already
carries that as a dense within-experiment C_i.

Whitening uses the Cholesky of C_i, matching the solver: C^{-1} is never formed.

    H_i^T W_i H_i = G_i^T G_i        H_i^T W_i r_i = G_i^T rt_i
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from scipy.linalg import cholesky, solve_triangular

SOLVER_AGREEMENT_TOLERANCE = 1.0e-8
SCORE_SUM_TOLERANCE = 1.0e-8


class UncertaintyError(ValueError):
    """The identification result cannot be turned into a UQ system."""


@dataclass
class Cluster:
    """One experiment: the independent statistical unit."""

    name: str
    experiment_id: int
    n_steps: int
    n_rows: int
    design: np.ndarray
    observation: np.ndarray
    covariance: np.ndarray
    whitened_design: np.ndarray
    whitened_observation: np.ndarray

    @property
    def sensitivity(self) -> np.ndarray:
        return self.whitened_design.T @ self.whitened_design

    def whitened_residual(self, theta: np.ndarray) -> np.ndarray:
        return self.whitened_observation - self.whitened_design @ theta

    def score(self, theta: np.ndarray) -> np.ndarray:
        return self.whitened_design.T @ self.whitened_residual(theta)


@dataclass
class EstimatingSystem:
    """Stacked GLS system in (H_i, f_i, C_i) form."""

    labels: List[str]
    theta: np.ndarray
    clusters: List[Cluster]
    covariance_work: np.ndarray
    chi2: Dict[str, float]
    identity_gap: float = float("nan")
    solver_gap: float = float("nan")
    score_sum_gap: float = float("nan")
    additivity_gap: float = float("nan")
    result: object = field(default=None, repr=False)

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)

    @property
    def n_parameters(self) -> int:
        return len(self.labels)

    @property
    def n_rows(self) -> int:
        return int(sum(cluster.n_rows for cluster in self.clusters))

    @property
    def cluster_names(self) -> List[str]:
        return [cluster.name for cluster in self.clusters]

    @property
    def whitened_design(self) -> np.ndarray:
        return np.vstack([cluster.whitened_design for cluster in self.clusters])

    def scores(self, theta: Optional[np.ndarray] = None) -> np.ndarray:
        used = self.theta if theta is None else theta
        return np.vstack([cluster.score(used) for cluster in self.clusters])

    def cluster_sensitivities(self) -> np.ndarray:
        return np.stack([cluster.sensitivity for cluster in self.clusters])

    def without(self, dropped: str) -> "EstimatingSystem":
        """This stacking minus one experiment, at fixed weights."""
        kept = [cluster for cluster in self.clusters if cluster.name != dropped]
        if len(kept) == len(self.clusters):
            raise KeyError(f"{dropped!r} is not a cluster of this system")
        design = np.vstack([cluster.whitened_design for cluster in kept])
        observation = np.concatenate(
            [cluster.whitened_observation for cluster in kept]
        )
        bread = design.T @ design
        theta = np.linalg.solve(bread, design.T @ observation)
        return EstimatingSystem(
            labels=list(self.labels),
            theta=theta,
            clusters=kept,
            covariance_work=np.linalg.inv(bread),
            chi2=dict(self.chi2),
        )

    def rank_bound(self) -> int:
        """Largest possible rank of the CR0 meat: min(M - 1, p)."""
        return int(min(self.n_clusters - 1, self.n_parameters))

    @property
    def robust_is_well_posed(self) -> bool:
        return self.rank_bound() >= self.n_parameters


def _whiten_block(matrix: np.ndarray, vector: np.ndarray, covariance: np.ndarray):
    try:
        factor = cholesky(np.asarray(covariance, dtype=float), lower=True)
    except np.linalg.LinAlgError as error:
        raise UncertaintyError(
            f"Experiment covariance is not positive definite ({error})."
        ) from error
    return (
        solve_triangular(factor, np.asarray(matrix, dtype=float), lower=True),
        solve_triangular(factor, np.asarray(vector, dtype=float), lower=True),
    )


def _unique_name(preferred: str, experiment_id: int, taken: set[str]) -> str:
    name = str(preferred or "").strip() or f"Experiment {experiment_id}"
    if name not in taken:
        return name
    candidate = f"{name} ({experiment_id})"
    index = 2
    while candidate in taken:
        candidate = f"{name} ({experiment_id}.{index})"
        index += 1
    return candidate


def _labels_for(result: Any, labels: Optional[Sequence[str]]) -> List[str]:
    theta = np.asarray(getattr(result, "final_parameters"), dtype=float)
    n = int(theta.size)
    if labels:
        names = list(labels)
    else:
        model = getattr(result, "constitutive_model", None)
        names = list(getattr(model, "parameter_labels", ()) or ())
    while len(names) < n:
        names.append(f"P{len(names) + 1}")
    return names[:n]


def _chi2_from_clusters(clusters: Sequence[Cluster], theta: np.ndarray, n_par: int):
    total = float(
        sum(
            residual @ residual
            for residual in (cluster.whitened_residual(theta) for cluster in clusters)
        )
    )
    dof = int(sum(cluster.n_rows for cluster in clusters) - n_par)
    if dof <= 0:
        return {"chi2": total, "dof": dof, "reduced": float("nan"), "birge": float("nan")}
    reduced = total / dof
    return {
        "chi2": total,
        "dof": dof,
        "reduced": reduced,
        "birge": float(np.sqrt(max(reduced, 0.0))),
    }


def _build_clusters(
    reports: Sequence[Any],
    designs: Sequence[Any],
    observations: Sequence[Any],
    labels: Sequence[str],
    experiment_names: Optional[Mapping[int, str]],
) -> List[Cluster]:
    names: dict[int, str] = dict(experiment_names or {})
    taken: set[str] = set()
    clusters: List[Cluster] = []
    for index, (report, design, observation) in enumerate(
        zip(reports, designs, observations)
    ):
        if design is None or observation is None:
            raise UncertaintyError(
                "Identification result is missing a per-experiment design or observation."
            )
        covariance = getattr(report, "covariance", None)
        if covariance is None:
            raise UncertaintyError(
                "Identification result is missing a per-experiment covariance C_i."
            )
        design = np.asarray(design, dtype=float)
        observation = np.asarray(observation, dtype=float)
        covariance = np.asarray(covariance, dtype=float)
        if design.ndim != 2 or observation.ndim != 1:
            raise UncertaintyError(
                "Per-experiment blocks H_i, f_i have unexpected shape."
            )
        if design.shape[0] != observation.shape[0] or design.shape[1] != len(labels):
            raise UncertaintyError(
                "Per-experiment design does not match the observation or parameter count."
            )
        experiment_id = int(getattr(report, "experiment_id", index + 1))
        preferred = names.get(experiment_id) or str(getattr(report, "label", "") or "")
        name = _unique_name(preferred, experiment_id, taken)
        taken.add(name)
        whitened_design, whitened_observation = _whiten_block(
            design, observation, covariance
        )
        clusters.append(
            Cluster(
                name=name,
                experiment_id=experiment_id,
                n_steps=int(getattr(report, "n_steps", 1) or 1),
                n_rows=int(design.shape[0]),
                design=design,
                observation=observation,
                covariance=covariance,
                whitened_design=whitened_design,
                whitened_observation=whitened_observation,
            )
        )
    if not clusters:
        raise UncertaintyError(
            "Uncertainty quantification needs a GLS result with per-experiment blocks."
        )
    return clusters


def system_from_blocks(
    designs: Sequence[np.ndarray],
    observations: Sequence[np.ndarray],
    covariances: Sequence[np.ndarray],
    theta: np.ndarray,
    labels: Sequence[str],
    names: Optional[Sequence[str]] = None,
    *,
    verify: bool = True,
) -> EstimatingSystem:
    """Build a system from raw (H_i, f_i, C_i) blocks. Used by tests."""
    reports = []
    for index, covariance in enumerate(covariances):
        name = names[index] if names is not None else f"E{index + 1}"
        reports.append(
            type(
                "Report",
                (),
                {
                    "experiment_id": index + 1,
                    "label": name,
                    "n_steps": 1,
                    "covariance": covariance,
                },
            )()
        )
    clusters = _build_clusters(reports, designs, observations, labels, None)
    stacked = np.vstack([cluster.whitened_design for cluster in clusters])
    rhs = np.concatenate([cluster.whitened_observation for cluster in clusters])
    bread = stacked.T @ stacked
    theta = np.linalg.solve(bread, stacked.T @ rhs)
    system = EstimatingSystem(
        labels=list(labels),
        theta=theta,
        clusters=clusters,
        covariance_work=np.linalg.inv(bread),
        chi2=_chi2_from_clusters(clusters, theta, len(labels)),
    )
    if verify:
        _verify(system)
    return system


def system_from_result(
    result: Any,
    *,
    labels: Optional[Sequence[str]] = None,
    experiment_names: Optional[Mapping[int, str]] = None,
) -> EstimatingSystem:
    """Repack a GlsIdentificationResult into (H_i, f_i, C_i) clusters."""
    reports = list(getattr(result, "experiments", None) or [])
    designs = list(getattr(result, "case_K_matrices", None) or [])
    observations = list(getattr(result, "case_U_vectors", None) or [])
    if not reports or not designs or not observations:
        raise UncertaintyError(
            "Uncertainty quantification needs a GLS result with per-experiment "
            "blocks (H_i, f_i, C_i). Run FGLS identification first."
        )
    if not (len(reports) == len(designs) == len(observations)):
        raise UncertaintyError(
            "Per-experiment design, observation and report lists are misaligned."
        )
    if getattr(result, "parameter_covariance", None) is None:
        raise UncertaintyError("Identification result is missing the eq. (20) covariance.")

    used_labels = _labels_for(result, labels)
    theta = np.asarray(result.final_parameters, dtype=float)
    clusters = _build_clusters(
        reports, designs, observations, used_labels, experiment_names
    )
    system = EstimatingSystem(
        labels=used_labels,
        theta=theta,
        clusters=clusters,
        covariance_work=np.asarray(result.parameter_covariance, dtype=float),
        chi2=_chi2_from_clusters(clusters, theta, len(used_labels)),
        result=result,
    )
    _verify(system)
    return system


def _verify(system: EstimatingSystem) -> None:
    """Layer-2 guards: block additivity, A^{-1} = eq. (20), score sum = 0."""
    n_par = system.n_parameters
    sensitivity = system.cluster_sensitivities().sum(axis=0)
    stacked = system.whitened_design.T @ system.whitened_design
    system.additivity_gap = float(np.max(np.abs(sensitivity - stacked)))
    if system.additivity_gap > SOLVER_AGREEMENT_TOLERANCE:
        raise UncertaintyError(
            f"sum_i A_i disagrees with the stacked A "
            f"(max |gap| = {system.additivity_gap:.3e})."
        )

    inverse = np.linalg.inv(sensitivity)
    system.identity_gap = float(np.max(np.abs(sensitivity @ inverse - np.eye(n_par))))
    system.solver_gap = float(np.max(np.abs(inverse - system.covariance_work)))
    if system.solver_gap > SOLVER_AGREEMENT_TOLERANCE:
        raise UncertaintyError(
            f"A^{{-1}} disagrees with the solver eq. (20) covariance "
            f"(max |gap| = {system.solver_gap:.3e})."
        )

    scores = system.scores()
    total = np.abs(scores.sum(axis=0)).max()
    scale = max(float(np.abs(scores).max()), 1.0)
    system.score_sum_gap = float(total / scale)
    if system.score_sum_gap > SCORE_SUM_TOLERANCE:
        raise UncertaintyError(
            f"sum_i ψ_i(θ̂) is not zero (relative {system.score_sum_gap:.3e})."
        )
