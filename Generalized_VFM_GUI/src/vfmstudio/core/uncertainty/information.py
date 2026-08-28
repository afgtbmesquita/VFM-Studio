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
from typing import List, Optional, Sequence

import numpy as np

from .sandwich import sensitivity, symmetrise

RANK_TOLERANCE = 1.0e-10
NEAR_ZERO_FRACTION = 0.05
INVERSE_EIG_FLOOR = 1.0e-12
G_HIGH = float(np.log(1.5))
R_HIGH = 0.05
_CHARACTERISTIC_LABELS = {"D11", "D22", "A11", "A22"}
_ADDED_TEST_TOKENS = ("twist", "torsion")
VERDICT_EXCELLENT = "excellent"
VERDICT_CONFLICTING = "conflicting"
VERDICT_REDUNDANT = "redundant"
VERDICT_PROBLEMATIC = "problematic"
VERDICT_UNDETERMINED = "undetermined"


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


def _normalise_label(label: str) -> str:
    return str(label or "").upper().replace("_", "").replace(" ", "")


def parameter_reference_scales(
    theta: np.ndarray,
    labels: Sequence[str],
    *,
    floor_fraction: float = NEAR_ZERO_FRACTION,
) -> np.ndarray:
    """s_i for A_α = Sᵀ A S.

    Nonzero coefficients use |θ̂_i|.  Near-zero couplings such as D16 / D26
    use a characteristic stiffness (median of the diagonal membrane/bending
    terms, otherwise the largest |θ̂|) so the normalisation stays meaningful.
    """
    theta = np.asarray(theta, dtype=float).reshape(-1)
    names = [_normalise_label(label) for label in labels]
    if len(names) < theta.size:
        names.extend(f"P{i + 1}" for i in range(len(names), theta.size))
    abs_theta = np.abs(theta)
    characteristic = [
        float(abs_theta[i])
        for i, name in enumerate(names[: theta.size])
        if name in _CHARACTERISTIC_LABELS and abs_theta[i] > 0.0
    ]
    if not characteristic:
        characteristic = [float(v) for v in abs_theta if v > 0.0]
    theta_char = float(np.median(characteristic)) if characteristic else 1.0
    floor = max(float(floor_fraction) * theta_char, np.finfo(float).tiny)
    return np.where(abs_theta >= floor, abs_theta, theta_char).astype(float)


def looks_like_added_test(name: str) -> bool:
    """True for names that are typically the extra test type (twist), not the baseline."""
    text = " ".join(str(name or "").lower().replace("_", " ").replace("-", " ").split())
    return any(token in text for token in _ADDED_TEST_TOKENS)


def default_baseline_flags(
    names: Sequence[str],
    experiment_ids: Sequence[int],
) -> dict:
    """Tick every test except those whose names look like twist/torsion."""
    return {
        int(eid): not looks_like_added_test(name)
        for name, eid in zip(names, experiment_ids)
    }


def assemble_sensitivity(
    source,
    experiment_ids: Sequence[int],
    *,
    n_parameters: Optional[int] = None,
) -> np.ndarray:
    """Sum A_i = H_iᵀ W_i H_i over the selected experiments."""
    if hasattr(source, "contributions"):
        n_parameters = int(source.n_parameters)
        rows = source.contributions
    else:
        rows = list(source)
        if n_parameters is None:
            if not rows:
                raise ValueError("n_parameters is required when there are no contributions.")
            n_parameters = int(np.asarray(rows[0].sensitivity).shape[0])
    wanted = {int(eid) for eid in experiment_ids}
    total = np.zeros((int(n_parameters), int(n_parameters)), dtype=float)
    for row in rows:
        if int(row.experiment_id) in wanted:
            total += np.asarray(row.sensitivity, dtype=float)
    return symmetrise(total)


def _working_inverse_diagonal(
    matrix: np.ndarray,
    *,
    rel_floor: float = INVERSE_EIG_FLOOR,
) -> tuple[np.ndarray, bool]:
    """diag(A⁻¹) with a relative eigenvalue floor.

    A Moore–Penrose pseudoinverse would put *zeros* on unidentified
    directions, which would make a missing excitation look strong.
    Flooring small eigenvalues instead makes those directions have a
    large working variance, which is the right comparison for G_i.
    """
    matrix = symmetrise(np.asarray(matrix, dtype=float))
    if matrix.size == 0:
        return np.zeros(0, dtype=float), False
    values, vectors = np.linalg.eigh(matrix)
    vmax = max(float(np.max(np.abs(values))), 1.0)
    floor = float(rel_floor) * vmax
    used_floor = bool(np.any(values < floor))
    inverse = vectors @ np.diag(1.0 / np.maximum(values, floor)) @ vectors.T
    inverse = 0.5 * (inverse + inverse.T)
    return np.asarray(np.diag(inverse), dtype=float), used_floor


def relative_deletion_influence(
    deletions: Sequence,
    scales: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """ρ_k = max_i |θ̂_(-i) − θ̂|_k / s_k from the fixed-weight shift (eq. 63)."""
    scales = np.asarray(scales, dtype=float).reshape(-1)
    n = int(scales.size)
    rho = np.full(n, np.nan)
    names = [""] * n
    if not deletions or n == 0:
        return rho, tuple(names)
    tiny = np.finfo(float).tiny
    scales = np.clip(scales, tiny, None)
    for k in range(n):
        best = float("-inf")
        best_name = ""
        for row in deletions:
            delta = np.asarray(getattr(row, "delta_fixed"), dtype=float).reshape(-1)
            if k >= delta.size:
                continue
            value = abs(float(delta[k])) / float(scales[k])
            if not np.isfinite(value) or value <= best:
                continue
            best = value
            best_name = str(getattr(row, "dropped", "") or "")
        if np.isfinite(best):
            rho[k] = best
            names[k] = best_name
    return rho, tuple(names)


def robustness_verdict(
    gain: float,
    influence: float,
    *,
    g_high: float = G_HIGH,
    r_high: float = R_HIGH,
) -> str:
    """Joint reading of information gain G_k and deletion influence ρ_k."""
    if not np.isfinite(influence) or not np.isfinite(gain):
        return VERDICT_UNDETERMINED
    high_g = float(gain) >= float(g_high)
    high_r = float(influence) >= float(r_high)
    if high_g and not high_r:
        return VERDICT_EXCELLENT
    if high_g and high_r:
        return VERDICT_CONFLICTING
    if not high_g and not high_r:
        return VERDICT_REDUNDANT
    return VERDICT_PROBLEMATIC


@dataclass
class IdentifiabilityStrength:
    """Information gain G_k, deletion influence ρ_k, and their joint reading.

    G_k asks whether added tests improve the geometry of A.  ρ_k asks whether
    experiments actually agree on parameter k, via the residual score in
    eq. (63).  Neither is a check that the mechanical model is correct.
    """

    scales: np.ndarray
    A_alpha: np.ndarray
    working_strength: np.ndarray
    information_gain: np.ndarray
    deletion_influence: np.ndarray
    pulling_experiment: tuple
    verdicts: tuple
    coupling_quality: np.ndarray
    used_eigenvalue_floor: bool

    @property
    def relative_strength(self) -> np.ndarray:
        return self.working_strength

    @property
    def used_pseudoinverse(self) -> bool:
        return self.used_eigenvalue_floor


def identifiability_strength(
    A: np.ndarray,
    theta: np.ndarray,
    labels: Sequence[str],
    *,
    A_baseline: Optional[np.ndarray] = None,
    deletions: Optional[Sequence] = None,
    floor_fraction: float = NEAR_ZERO_FRACTION,
) -> IdentifiabilityStrength:
    """Information gain G_k and deletion influence ρ_k.

    With S = diag(s_i) and A_α = Sᵀ A S, using the same S for every matrix,

        G_k = log( (A_{α,base}^{-1})_{kk} / (A_{α,all}^{-1})_{kk} )

    asks whether added tests improve the mathematical identifiability of
    parameter k.  It does not care whether the tests agree on the value.

    The fixed-weight deletion shift (eq. 63)

        θ̂_(-i) − θ̂ = −(A − A_i)^{-1} ψ_i(θ̂)

    does: ρ_k = max_i |θ̂_(-i) − θ̂|_k / s_k is large when some experiment
    pulls parameter k.  Read the two together:

        high G, low ρ   useful and consistent
        high G, high ρ  informative but conflicting
        low G,  low ρ   redundant / not useful
        low G,  high ρ  problematic test
    """
    A = symmetrise(np.asarray(A, dtype=float))
    theta = np.asarray(theta, dtype=float).reshape(-1)
    n = int(A.shape[0])
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    if theta.size != n:
        raise ValueError(f"theta has {theta.size} entries but A is {n}×{n}.")
    scales = parameter_reference_scales(theta, labels, floor_fraction=floor_fraction)
    scale_matrix = np.diag(scales)
    a_alpha = scale_matrix.T @ A @ scale_matrix
    inverse_diag, used_floor_a = _working_inverse_diagonal(A)
    inverse_alpha_diag, used_floor_alpha = _working_inverse_diagonal(a_alpha)
    tiny = np.finfo(float).tiny
    inverse_diag = np.clip(inverse_diag, tiny, None)
    inverse_alpha_diag = np.clip(inverse_alpha_diag, tiny, None)
    working = 1.0 / inverse_alpha_diag
    diagonal = np.clip(np.diag(A), tiny, None)
    coupling = 1.0 / (diagonal * inverse_diag)
    used_floor_base = False
    if A_baseline is None:
        gain = np.full(n, np.nan)
    else:
        A_baseline = symmetrise(np.asarray(A_baseline, dtype=float))
        if A_baseline.shape != A.shape:
            raise ValueError(
                f"A_baseline is {A_baseline.shape[0]}×{A_baseline.shape[1]} "
                f"but A is {n}×{n}."
            )
        a_base_alpha = scale_matrix.T @ A_baseline @ scale_matrix
        var_base, used_floor_base = _working_inverse_diagonal(a_base_alpha)
        var_base = np.clip(var_base, tiny, None)
        gain = np.log(var_base / inverse_alpha_diag)
    rho, pullers = relative_deletion_influence(deletions or (), scales)
    verdicts = tuple(
        robustness_verdict(float(g), float(r)) for g, r in zip(gain, rho)
    )
    return IdentifiabilityStrength(
        scales=np.asarray(scales, dtype=float),
        A_alpha=np.asarray(a_alpha, dtype=float),
        working_strength=np.asarray(working, dtype=float),
        information_gain=np.asarray(gain, dtype=float),
        deletion_influence=np.asarray(rho, dtype=float),
        pulling_experiment=pullers,
        verdicts=verdicts,
        coupling_quality=np.clip(np.asarray(coupling, dtype=float), 0.0, 1.0),
        used_eigenvalue_floor=bool(used_floor_a or used_floor_alpha or used_floor_base),
    )
