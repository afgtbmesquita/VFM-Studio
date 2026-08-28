"""All non-empty test combinations in frozen-weight GLS parameter space.

After one Identify, each experiment already has its last-iteration blocks
(H_i, b_i, C_i). Optimal virtual fields are not recomputed: they stay those of
the all-tests solve. Every subset S is the closed-form GLS estimator

    A_S = Σ_{i∈S} H̃_iᵀ H̃_i ,   s_S = Σ_{i∈S} H̃_iᵀ f̃_i ,   θ_S = A_S⁻¹ s_S

which is additive in the tests, so a bitmask sweep is enough. The empty set is
excluded. The all-tests mask recovers the Identify θ̂.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from vfmstudio.core.uncertainty.assemble import (
    EstimatingSystem,
    UncertaintyError,
    system_from_result,
)
from vfmstudio.core.uncertainty.sandwich import symmetrise

RANK_RELATIVE = 1.0e-10
COND_LIMIT = 1.0e12
PREFERRED_LEFT = ("D11", "D22")
PREFERRED_RIGHT = ("D12", "D66")


@dataclass
class ParameterSpaceResult:
    """θ_S for every non-empty combination of the identified tests."""

    labels: List[str]
    names: List[str]
    experiment_ids: List[int]
    theta_full: np.ndarray
    masks: np.ndarray
    n_tests: np.ndarray
    theta: np.ndarray
    cond: np.ndarray
    well_posed: np.ndarray
    full_gap: float = float("nan")
    system: Optional[EstimatingSystem] = field(default=None, repr=False)

    @property
    def n_clusters(self) -> int:
        return len(self.names)

    @property
    def n_parameters(self) -> int:
        return len(self.labels)

    @property
    def n_subsets(self) -> int:
        return int(self.masks.size)

    @property
    def full_mask(self) -> int:
        return (1 << self.n_clusters) - 1

    def subset_names(self, mask: int) -> List[str]:
        return [self.names[i] for i in range(self.n_clusters) if mask & (1 << i)]

    def subset_ids(self, mask: int) -> List[int]:
        return [
            self.experiment_ids[i]
            for i in range(self.n_clusters)
            if mask & (1 << i)
        ]

    def index_of_mask(self, mask: int) -> int:
        found = np.flatnonzero(self.masks == int(mask))
        if found.size == 0:
            raise KeyError(f"mask {mask} is not in this parameter-space sweep")
        return int(found[0])


def default_axis_labels(labels: Sequence[str]) -> Tuple[str, str, str, str]:
    """Left: D11 vs D22. Right: D12 vs D66. Fall back to the first labels."""
    names = [str(name) for name in labels]
    if not names:
        return ("", "", "", "")
    lookup = {name.upper(): name for name in names}

    def pick(preferred: str, fallback: int) -> str:
        if preferred in lookup:
            return lookup[preferred]
        return names[min(max(fallback, 0), len(names) - 1)]

    return (
        pick(PREFERRED_LEFT[0], 0),
        pick(PREFERRED_LEFT[1], 2),
        pick(PREFERRED_RIGHT[0], 1),
        pick(PREFERRED_RIGHT[1], 3),
    )


def resolve_axis_index(labels: Sequence[str], chosen: str, fallback: int) -> int:
    names = [str(name) for name in labels]
    if chosen in names:
        return names.index(chosen)
    upper = str(chosen).upper()
    for index, name in enumerate(names):
        if name.upper() == upper:
            return index
    if not names:
        return 0
    return min(max(int(fallback), 0), len(names) - 1)


def resolve_plot_axes(
    labels: Sequence[str],
    left_x: str = "",
    left_y: str = "",
    right_x: str = "",
    right_y: str = "",
) -> Tuple[int, int, int, int]:
    defaults = default_axis_labels(labels)
    return (
        resolve_axis_index(labels, left_x or defaults[0], 0),
        resolve_axis_index(labels, left_y or defaults[1], 2),
        resolve_axis_index(labels, right_x or defaults[2], 1),
        resolve_axis_index(labels, right_y or defaults[3], 3),
    )


def mask_for_experiment_ids(
    space: ParameterSpaceResult,
    experiment_ids: Sequence[int],
) -> Optional[int]:
    """Bitmask for this set of tests, or None if it is not a stored subset."""
    wanted = {int(experiment_id) for experiment_id in experiment_ids}
    if not wanted:
        return None
    mask = 0
    seen: set[int] = set()
    for index, experiment_id in enumerate(space.experiment_ids):
        key = int(experiment_id)
        if key in wanted:
            mask |= 1 << index
            seen.add(key)
    if seen != wanted or mask == 0:
        return None
    return int(mask)


def theta_for_experiment_ids(
    space: ParameterSpaceResult,
    experiment_ids: Sequence[int],
) -> Optional[np.ndarray]:
    """Frozen GLS θ for this combination, or None if Identify did not include it."""
    mask = mask_for_experiment_ids(space, experiment_ids)
    if mask is None:
        return None
    return np.asarray(space.theta[space.index_of_mask(mask)], dtype=float)


def is_full_combination(
    space: ParameterSpaceResult,
    experiment_ids: Sequence[int],
) -> bool:
    mask = mask_for_experiment_ids(space, experiment_ids)
    return mask is not None and int(mask) == int(space.full_mask)


def assign_highlight_groups(
    space: ParameterSpaceResult,
    groups: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Most specific group contained in the combination, else -1.

    A combination matches a group when it includes every ticked test in that
    group (intersection of the "contains this test" clouds). Ticking more
    tests therefore highlights fewer points. If several groups match, the
    largest group wins, then the earlier group in ``groups``.
    """
    assigned = np.full(int(space.n_subsets), -1, dtype=int)
    parsed: list[tuple[int, int, set[int]]] = []
    for index, group in enumerate(groups or []):
        ids = {int(experiment_id) for experiment_id in (group.get("experiment_ids") or [])}
        if ids:
            parsed.append((len(ids), index, ids))
    parsed.sort(key=lambda item: (-item[0], item[1]))
    for row, mask in enumerate(np.asarray(space.masks, dtype=int).ravel()):
        combo = set(space.subset_ids(int(mask)))
        if not combo:
            continue
        for _size, group_index, required in parsed:
            if required <= combo:
                assigned[row] = group_index
                break
    return assigned


def group_member_mask(
    space: ParameterSpaceResult,
    experiment_ids: Sequence[int],
) -> np.ndarray:
    """True where the combination includes every ticked test."""
    required = {int(experiment_id) for experiment_id in experiment_ids}
    n = int(space.n_subsets)
    if not required or n == 0:
        return np.zeros(n, dtype=bool)
    matched = np.zeros(n, dtype=bool)
    for row, mask in enumerate(np.asarray(space.masks, dtype=int).ravel()):
        combo = set(space.subset_ids(int(mask)))
        matched[row] = bool(combo) and required <= combo
    return matched


def group_statistics(
    space: ParameterSpaceResult,
    experiment_ids: Sequence[int],
) -> dict:
    """Mean and sample σ of θ over combinations that match the group."""
    labels = list(space.labels)
    n_par = int(len(labels))
    pick = group_member_mask(space, experiment_ids)
    n = int(np.sum(pick))
    mean = np.full(n_par, np.nan, dtype=float)
    std = np.full(n_par, np.nan, dtype=float)
    if n >= 1:
        values = np.asarray(space.theta, dtype=float)[pick]
        mean = np.mean(values, axis=0)
        if n >= 2:
            std = np.std(values, axis=0, ddof=1)
    return {"n": n, "mean": mean, "std": std, "labels": labels}


def _cluster_normal(cluster) -> Tuple[np.ndarray, np.ndarray]:
    design = np.asarray(cluster.whitened_design, dtype=float)
    observation = np.asarray(cluster.whitened_observation, dtype=float).reshape(-1)
    sensitivity = symmetrise(cluster.sensitivity)
    rhs = design.T @ observation
    return sensitivity, rhs


def _condition_and_rank(matrix: np.ndarray, n_par: int) -> Tuple[float, int, bool]:
    values = np.linalg.eigvalsh(symmetrise(matrix))
    scale = max(float(np.max(np.abs(values))), 1.0)
    floor = RANK_RELATIVE * scale
    positive = values[values > floor]
    rank = int(positive.size)
    if rank == 0:
        return float("inf"), 0, False
    cond = float(np.max(positive) / np.min(positive))
    well = rank >= int(n_par) and np.isfinite(cond) and cond < COND_LIMIT
    return cond, rank, well


def _solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    matrix = symmetrise(matrix)
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(matrix, rhs, rcond=None)[0]


def from_system(system: EstimatingSystem) -> ParameterSpaceResult:
    """Sweep every non-empty subset of ``system.clusters``."""
    clusters = list(system.clusters)
    labels = list(system.labels)
    n_clusters = len(clusters)
    n_par = int(len(labels))
    theta_full = np.asarray(system.theta, dtype=float).reshape(-1)
    names = list(system.cluster_names)
    experiment_ids = [int(cluster.experiment_id) for cluster in clusters]
    if n_clusters == 0 or n_par == 0:
        return ParameterSpaceResult(
            labels=labels,
            names=names,
            experiment_ids=experiment_ids,
            theta_full=theta_full,
            masks=np.zeros(0, dtype=int),
            n_tests=np.zeros(0, dtype=int),
            theta=np.zeros((0, n_par), dtype=float),
            cond=np.zeros(0, dtype=float),
            well_posed=np.zeros(0, dtype=bool),
            full_gap=float("nan"),
            system=system,
        )

    n_masks = 1 << n_clusters
    bread = np.zeros((n_masks, n_par, n_par), dtype=float)
    rhs = np.zeros((n_masks, n_par), dtype=float)
    for index, cluster in enumerate(clusters):
        bit = 1 << index
        bread[bit], rhs[bit] = _cluster_normal(cluster)

    for mask in range(1, n_masks):
        bit = mask & -mask
        if mask == bit:
            continue
        parent = mask ^ bit
        bread[mask] = bread[parent] + bread[bit]
        rhs[mask] = rhs[parent] + rhs[bit]

    masks = np.arange(1, n_masks, dtype=int)
    n_subsets = int(masks.size)
    theta = np.empty((n_subsets, n_par), dtype=float)
    cond = np.empty(n_subsets, dtype=float)
    well_posed = np.empty(n_subsets, dtype=bool)
    n_tests = np.array([int(mask.bit_count()) for mask in masks], dtype=int)
    for row, mask in enumerate(masks):
        cond[row], _rank, well_posed[row] = _condition_and_rank(bread[mask], n_par)
        theta[row] = _solve(bread[mask], rhs[mask])

    full_index = n_subsets - 1
    full_gap = float(np.max(np.abs(theta[full_index] - theta_full)))
    return ParameterSpaceResult(
        labels=labels,
        names=names,
        experiment_ids=experiment_ids,
        theta_full=theta_full,
        masks=masks,
        n_tests=n_tests,
        theta=theta,
        cond=cond,
        well_posed=well_posed,
        full_gap=full_gap,
        system=system,
    )


def compute_parameter_space(
    source: Any,
    *,
    labels: Optional[Sequence[str]] = None,
    experiment_names: Optional[Mapping[int, str]] = None,
) -> Optional[ParameterSpaceResult]:
    """Build the sweep from a GLS result or an already-packed system."""
    if source is None:
        return None
    if isinstance(source, EstimatingSystem):
        return from_system(source)
    try:
        system = system_from_result(
            source, labels=labels, experiment_names=experiment_names
        )
    except (UncertaintyError, TypeError, ValueError):
        return None
    return from_system(system)


def space_from_document(result: Any, document: Any = None) -> Optional[ParameterSpaceResult]:
    labels = None
    names = None
    if document is not None:
        labels = document.material.labels()
        names = {
            int(experiment.experiment_id): (
                experiment.name or f"Experiment {experiment.experiment_id}"
            )
            for experiment in document.experiments
        }
    return compute_parameter_space(result, labels=labels, experiment_names=names)


def selected_experiment_ids(document: Any) -> List[int]:
    if document is None:
        return []
    return [
        int(experiment.experiment_id)
        for experiment in document.experiments_in_identification()
    ]
