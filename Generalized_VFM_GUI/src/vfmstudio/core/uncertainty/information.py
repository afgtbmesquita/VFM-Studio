"""Layer 1: information geometry of the working GLS sensitivity A.

The cluster sandwich is the quoted Cov(θ̂) only when M ≥ p + 1.  Below that,
replication spans an (M − 1)-dimensional subspace of R^p and a robust standard
deviation is not well posed.  The object to read then is A itself:

    A   = sum_i H_i^T W_i H_i
    A_i = H_i^T W_i H_i

cond(A) says which combinations of parameters the pooled design constrains.
The share A_i,kk / A_kk says which experiment supplies which direction.
cond(A − A_i) is what happens to that geometry if the experiment is dropped.
Residual χ²/dof is a warning that C is optimistic on average, not a
per-coefficient σ.

This is not the kinematic identifiability matrix H, and it is not Cov(θ).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .sandwich import sensitivity, symmetrise

RANK_TOLERANCE = 1.0e-10


def _eigvalsh(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.eigvalsh(symmetrise(matrix))


def spd_condition(matrix: np.ndarray) -> float:
    """λ_max / λ_min of a symmetric matrix; inf if not positive definite."""
    values = _eigvalsh(matrix)
    if values.size == 0:
        return float("inf")
    vmin, vmax = float(values.min()), float(values.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= 0.0 or vmin <= 0.0:
        return float("inf")
    return float(vmax / vmin)


def spd_rank(matrix: np.ndarray, *, scale: float) -> int:
    values = _eigvalsh(matrix)
    floor = RANK_TOLERANCE * max(float(scale), 1.0)
    return int(np.sum(values > floor))


@dataclass
class DesignContribution:
    """One experiment's share of A and the design left after dropping it."""

    name: str
    experiment_id: int
    n_steps: int
    sensitivity: np.ndarray
    share: np.ndarray
    remaining_condition: float
    remaining_rank: int
    remaining_min_eig: float
    condition_ratio: float


@dataclass
class InformationGeometry:
    """Pooled A, per-experiment A_i, and leave-one-out identifiability of A."""

    condition: float
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    A: np.ndarray
    A_relative: np.ndarray
    blocks: np.ndarray
    contributions: List[DesignContribution]
    chi2_reduced: float
    n_clusters: int
    n_parameters: int
    rank_bound: int
    well_posed: bool

    @property
    def min_clusters_for_sandwich(self) -> int:
        return int(self.n_parameters + 1)

    @property
    def shares(self) -> np.ndarray:
        if not self.contributions:
            return np.zeros((0, self.n_parameters))
        return np.vstack([row.share for row in self.contributions])

    @property
    def remaining_conditions(self) -> np.ndarray:
        if not self.contributions:
            return np.zeros(0)
        return np.array(
            [row.remaining_condition for row in self.contributions], dtype=float
        )


def analyse_information(system) -> InformationGeometry:
    """Working information A and each experiment's share of it."""
    bread = sensitivity(system)
    blocks = np.asarray(system.cluster_sensitivities(), dtype=float)
    values, vectors = np.linalg.eigh(symmetrise(bread))
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    scale = float(values[0]) if values.size else 1.0
    condition = spd_condition(bread)
    diagonal = np.clip(np.diag(bread), np.finfo(float).tiny, None)
    theta_scale = np.diag(np.abs(np.asarray(system.theta, dtype=float)))
    relative = theta_scale @ bread @ theta_scale

    contributions: List[DesignContribution] = []
    for cluster, block in zip(system.clusters, blocks):
        remaining = symmetrise(bread - block)
        remaining_values = _eigvalsh(remaining)
        remaining_rank = spd_rank(remaining, scale=scale)
        remaining_condition = spd_condition(remaining)
        ratio = (
            remaining_condition / condition
            if np.isfinite(remaining_condition) and np.isfinite(condition) and condition > 0
            else float("inf")
        )
        share = np.diag(block) / diagonal
        contributions.append(
            DesignContribution(
                name=cluster.name,
                experiment_id=int(cluster.experiment_id),
                n_steps=int(cluster.n_steps),
                sensitivity=np.asarray(block, dtype=float),
                share=np.asarray(share, dtype=float),
                remaining_condition=remaining_condition,
                remaining_rank=remaining_rank,
                remaining_min_eig=float(remaining_values.min()) if remaining_values.size else float("nan"),
                condition_ratio=ratio,
            )
        )

    chi = system.chi2.get("reduced", float("nan"))
    return InformationGeometry(
        condition=condition,
        eigenvalues=np.asarray(values, dtype=float),
        eigenvectors=np.asarray(vectors, dtype=float),
        A=np.asarray(bread, dtype=float),
        A_relative=np.asarray(relative, dtype=float),
        blocks=blocks,
        contributions=contributions,
        chi2_reduced=float(chi) if chi is not None else float("nan"),
        n_clusters=int(system.n_clusters),
        n_parameters=int(system.n_parameters),
        rank_bound=int(system.rank_bound()),
        well_posed=bool(system.robust_is_well_posed),
    )
