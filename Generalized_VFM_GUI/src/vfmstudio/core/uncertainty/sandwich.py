"""Sandwich covariance and Godambe information for multi-experiment VFM GLS.

Under a correctly specified likelihood A = B = F.  When the working density is
wrong they separate and inverse Fisher information stops being the uncertainty:

    Cov(θ̂) ~ A^{-1} B A^{-T}

Fisher is the special case A = B.  None of this requires Gaussian residuals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

LEVERAGE_FLOOR = 1.0e-10


def symmetrise(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    return 0.5 * (matrix + matrix.T)


def sensitivity(system) -> np.ndarray:
    """A = sum_i H_i^T W_i H_i."""
    return symmetrise(system.cluster_sensitivities().sum(axis=0))


def cluster_scores(system, theta: Optional[np.ndarray] = None) -> np.ndarray:
    """ψ_i = H_i^T W_i r_i.  Rows sum to zero at θ̂."""
    return system.scores(theta)


def leverage_blocks(system) -> list:
    inverse = np.linalg.inv(sensitivity(system))
    return [
        cluster.whitened_design @ inverse @ cluster.whitened_design.T
        for cluster in system.clusters
    ]


def _inverse_sqrt(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(symmetrise(matrix))
    values = np.clip(values, LEVERAGE_FLOOR, None)
    return vectors @ np.diag(values ** -0.5) @ vectors.T


def _inverse(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(symmetrise(matrix))
    values = np.clip(values, LEVERAGE_FLOOR, None)
    return vectors @ np.diag(1.0 / values) @ vectors.T


def adjusted_scores(system, kind: str) -> np.ndarray:
    kind = kind.upper()
    theta = system.theta
    if kind in ("CR0", "CR1"):
        return system.scores(theta)
    if kind not in ("CR2", "CR3"):
        raise ValueError(f"Unknown small-sample correction {kind!r}")

    scores = []
    for cluster, leverage in zip(system.clusters, leverage_blocks(system)):
        residual = cluster.whitened_residual(theta)
        gap = np.eye(leverage.shape[0]) - leverage
        adjust = _inverse_sqrt(gap) if kind == "CR2" else _inverse(gap)
        scores.append(cluster.whitened_design.T @ (adjust @ residual))
    return np.vstack(scores)


def meat(system, kind: str = "CR2") -> np.ndarray:
    """B = sum_i ψ_i ψ_i^T, cluster-robust.  Rank at most M − 1."""
    scores = adjusted_scores(system, kind)
    matrix = symmetrise(scores.T @ scores)
    if kind.upper() == "CR1":
        n_rows, n_par, n_cluster = system.n_rows, system.n_parameters, system.n_clusters
        scale = (n_cluster / max(n_cluster - 1, 1)) * (
            (n_rows - 1) / max(n_rows - n_par, 1)
        )
        matrix = scale * matrix
    return matrix


def model_covariance(system) -> np.ndarray:
    """V_work = A^{-1}."""
    return symmetrise(np.linalg.inv(sensitivity(system)))


def sandwich_covariance(system, kind: str = "CR2") -> np.ndarray:
    """V_sand = A^{-1} B A^{-T}."""
    bread = np.linalg.inv(sensitivity(system))
    return symmetrise(bread @ meat(system, kind) @ bread.T)


def godambe_information(system, kind: str = "CR2") -> Optional[np.ndarray]:
    """G = A^T B^{-1} A.  None when B is singular (M ≤ p)."""
    matrix = meat(system, kind)
    scale = max(float(np.abs(matrix).max()), 1.0)
    if np.linalg.matrix_rank(matrix, tol=1.0e-10 * scale) < system.n_parameters:
        return None
    bread = sensitivity(system)
    return symmetrise(bread.T @ np.linalg.inv(matrix) @ bread)


@dataclass
class MisfitSpectrum:
    """Eigen-decomposition of A^{1/2} V_sand A^{1/2}.

    λ_j is the robust / working variance ratio along principal direction c_j.
    """

    eigenvalues: np.ndarray
    directions: np.ndarray
    relative_loadings: np.ndarray
    anisotropy: float
    geometric_mean: float
    scalar_rescaling: float
    well_posed: bool


def misfit_spectrum(system, kind: str = "CR2") -> MisfitSpectrum:
    bread = sensitivity(system)
    values, vectors = np.linalg.eigh(bread)
    root = vectors @ np.diag(np.sqrt(np.clip(values, 0.0, None))) @ vectors.T

    lam, vecs = np.linalg.eigh(
        symmetrise(root @ sandwich_covariance(system, kind) @ root)
    )
    order = np.argsort(lam)[::-1]
    lam, vecs = lam[order], vecs[:, order]

    directions = root @ vecs
    directions = directions / np.linalg.norm(directions, axis=0, keepdims=True)
    loadings = np.diag(np.abs(system.theta)) @ directions
    norms = np.linalg.norm(loadings, axis=0, keepdims=True)
    loadings = loadings / np.clip(norms, np.finfo(float).tiny, None)

    well_posed = bool(
        system.robust_is_well_posed and lam.min() > 1.0e-12 * max(lam.max(), 1.0)
    )
    if well_posed:
        anisotropy = float(lam.max() / lam.min())
        geometric_mean = float(np.exp(np.mean(np.log(lam))))
    else:
        anisotropy = geometric_mean = float("nan")

    return MisfitSpectrum(
        eigenvalues=lam,
        directions=directions,
        relative_loadings=loadings,
        anisotropy=anisotropy,
        geometric_mean=geometric_mean,
        scalar_rescaling=float(system.chi2.get("birge", float("nan"))),
        well_posed=well_posed,
    )


def standard_errors(covariance: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(np.diag(np.asarray(covariance, dtype=float)), 0.0, None))


def correlation(covariance: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.clip(np.diag(covariance), np.finfo(float).tiny, None))
    return np.asarray(covariance, dtype=float) / np.outer(scale, scale)


def summarise(system, kind: str = "CR2") -> Dict[str, object]:
    bread = sensitivity(system)
    covariances = {
        "work": model_covariance(system),
        "CR0": sandwich_covariance(system, "CR0"),
        "CR1": sandwich_covariance(system, "CR1"),
        "CR2": sandwich_covariance(system, "CR2"),
        "CR3": sandwich_covariance(system, "CR3"),
    }
    return {
        "A": bread,
        "B": meat(system, kind),
        "G": godambe_information(system, kind),
        "covariances": covariances,
        "robust": covariances[kind],
        "scores": cluster_scores(system),
        "spectrum": misfit_spectrum(system, kind),
        "kind": kind.upper(),
        "rank_meat": int(np.linalg.matrix_rank(meat(system, "CR0"))),
        "rank_meat_numerical": int(np.linalg.matrix_rank(meat(system, kind))),
        "rank_bound": system.rank_bound(),
        "well_posed": system.robust_is_well_posed,
    }
