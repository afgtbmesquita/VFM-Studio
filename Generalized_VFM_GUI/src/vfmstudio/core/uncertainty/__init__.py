"""Robust uncertainty quantification for a GLS identification result.

Ports the notes' three analyses into the GUI, without the campaign-specific
I/O of ``Samples_A_B.robust_information``:

    Cov(θ)                     sandwich / working covariance of the pooled θ̂
    Experiment uncertainty     cluster scores ψ_i and meat B
    Information geometry       A, A_i, cond(A − A_i); the object to read if M < p+1
    Leave-one-out              deletion of a whole experiment, never a step

The GLS point estimate is untouched.  Only the covariance formula and the
diagnostics change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .assemble import (
    Cluster,
    EstimatingSystem,
    UncertaintyError,
    system_from_blocks,
    system_from_result,
)
from .deletion import DeletionRow, analyse as analyse_deletion
from .information import (
    DesignContribution,
    InformationGeometry,
    analyse_information,
)
from .sandwich import (
    MisfitSpectrum,
    correlation,
    misfit_spectrum,
    sandwich_covariance,
    standard_errors,
    summarise,
)

__all__ = [
    "Cluster",
    "DeletionRow",
    "DesignContribution",
    "EstimatingSystem",
    "InformationGeometry",
    "MisfitSpectrum",
    "UncertaintyError",
    "UncertaintyResult",
    "analyse",
    "analyse_information",
    "attach_refits",
    "correlation",
    "misfit_spectrum",
    "sandwich_covariance",
    "standard_errors",
    "summarise",
    "system_from_blocks",
    "system_from_result",
]


@dataclass
class UncertaintyResult:
    """Everything the post-processing pages need, computed once."""

    kind: str
    labels: List[str]
    theta: np.ndarray
    n_clusters: int
    n_parameters: int
    n_rows: int
    chi2: Dict[str, float]
    well_posed: bool
    rank_meat: int
    rank_bound: int
    additivity_gap: float
    solver_gap: float
    score_sum_gap: float
    A: np.ndarray
    B: np.ndarray
    G: Optional[np.ndarray]
    covariances: Dict[str, np.ndarray]
    robust: np.ndarray
    sigma_work: np.ndarray
    sigma_robust: np.ndarray
    sigma_cr3: np.ndarray
    scores: np.ndarray
    score_norms: np.ndarray
    cluster_names: List[str]
    cluster_ids: List[int]
    cluster_n_steps: List[int]
    spectrum: MisfitSpectrum
    deletions: List[DeletionRow]
    information: InformationGeometry
    system: EstimatingSystem = field(repr=False)


def analyse(
    result: Any,
    *,
    labels: Optional[Sequence[str]] = None,
    experiment_names: Optional[Mapping[int, str]] = None,
    kind: str = "CR2",
) -> UncertaintyResult:
    """Sandwich, scores, and fixed-weight leave-one-out from a GLS result."""
    system = system_from_result(
        result, labels=labels, experiment_names=experiment_names
    )
    return _from_system(system, kind=kind, deletions=analyse_deletion(system, kind=kind))


def attach_refits(
    current: UncertaintyResult,
    refit_fn: Callable[[int, str], Any],
    *,
    kind: Optional[str] = None,
) -> UncertaintyResult:
    """Recompute leave-one-out with FGLS refits; sandwich/scores stay put."""
    used = (kind or current.kind).upper()
    rows = analyse_deletion(current.system, kind=used, refit_fn=refit_fn)
    return _from_system(current.system, kind=used, deletions=rows)


def _from_system(
    system: EstimatingSystem,
    *,
    kind: str,
    deletions: List[DeletionRow],
) -> UncertaintyResult:
    kind = kind.upper()
    layer = summarise(system, kind)
    inverse = np.linalg.inv(layer["A"])
    scores = np.asarray(layer["scores"], dtype=float)
    norms = np.array([float(score @ inverse @ score) for score in scores])
    sigma = {
        key: standard_errors(value) for key, value in layer["covariances"].items()
    }
    return UncertaintyResult(
        kind=kind,
        labels=list(system.labels),
        theta=np.asarray(system.theta, dtype=float),
        n_clusters=system.n_clusters,
        n_parameters=system.n_parameters,
        n_rows=system.n_rows,
        chi2=dict(system.chi2),
        well_posed=bool(layer["well_posed"]),
        rank_meat=int(layer["rank_meat"]),
        rank_bound=int(layer["rank_bound"]),
        additivity_gap=float(system.additivity_gap),
        solver_gap=float(system.solver_gap),
        score_sum_gap=float(system.score_sum_gap),
        A=np.asarray(layer["A"], dtype=float),
        B=np.asarray(layer["B"], dtype=float),
        G=None if layer["G"] is None else np.asarray(layer["G"], dtype=float),
        covariances=layer["covariances"],
        robust=np.asarray(layer["robust"], dtype=float),
        sigma_work=sigma["work"],
        sigma_robust=sigma[kind],
        sigma_cr3=sigma["CR3"],
        scores=scores,
        score_norms=norms,
        cluster_names=list(system.cluster_names),
        cluster_ids=[c.experiment_id for c in system.clusters],
        cluster_n_steps=[c.n_steps for c in system.clusters],
        spectrum=layer["spectrum"],
        deletions=deletions,
        information=analyse_information(system),
        system=system,
    )
