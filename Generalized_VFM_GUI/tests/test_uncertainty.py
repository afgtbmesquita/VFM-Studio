"""Core uncertainty quantification: sandwich, scores, leave-one-out."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vfmstudio.core.uncertainty import (
    UncertaintyError,
    analyse,
    sandwich_covariance,
    standard_errors,
    summarise,
    system_from_blocks,
    system_from_result,
)
from vfmstudio.core.uncertainty.deletion import analyse as analyse_deletion
from vfmstudio.core.uncertainty.deletion import exact_shift
from vfmstudio.core.uncertainty.sandwich import godambe_information, model_covariance


def _random_system(n_clusters: int, n_par: int, n_steps: int, seed: int = 0, noise: float = 1.0):
    rng = np.random.default_rng(seed)
    theta_true = rng.normal(size=n_par)
    designs, observations, covariances = [], [], []
    for _ in range(n_clusters):
        design = rng.normal(size=(n_steps, n_par))
        root = rng.normal(size=(n_steps, n_steps)) / np.sqrt(n_steps)
        covariance = root @ root.T + np.eye(n_steps)
        factor = np.linalg.cholesky(covariance)
        residual = noise * factor @ rng.standard_t(df=5, size=n_steps) / np.sqrt(5.0 / 3.0)
        observation = design @ theta_true + residual
        designs.append(design)
        observations.append(observation)
        covariances.append(covariance)
    labels = [f"P{i + 1}" for i in range(n_par)]
    names = [f"E{i + 1}" for i in range(n_clusters)]
    system = system_from_blocks(
        designs, observations, covariances, theta_true, labels, names
    )
    return system, theta_true


def test_score_sum_is_zero():
    system, _ = _random_system(6, 3, 4, seed=1)
    scores = system.scores()
    scale = max(float(np.abs(scores).max()), np.finfo(float).tiny)
    assert float(np.abs(scores.sum(axis=0)).max() / scale) < 1.0e-10


def test_deletion_identity_matches_reduced_gls():
    system, _ = _random_system(6, 3, 4, seed=2)
    dropped = system.cluster_names[1]
    shift = exact_shift(system, dropped)
    reduced = system.without(dropped)
    assert np.allclose(system.theta + shift, reduced.theta, atol=1.0e-10)


def test_rank_bound_refuses_robust_when_m_leq_p():
    system, _ = _random_system(3, 4, 5, seed=3)
    assert system.rank_bound() == 2
    assert system.robust_is_well_posed is False
    assert godambe_information(system, "CR2") is None
    layer = summarise(system, "CR2")
    assert layer["well_posed"] is False


def test_sandwich_collapses_when_weight_is_true_covariance():
    rng = np.random.default_rng(4)
    n_clusters, n_par, n_steps = 120, 3, 5
    theta_true = np.array([2.0, -1.0, 0.5])
    designs, observations, covariances = [], [], []
    for _ in range(n_clusters):
        design = rng.normal(size=(n_steps, n_par))
        root = rng.normal(size=(n_steps, n_steps)) / np.sqrt(n_steps)
        covariance = root @ root.T + np.eye(n_steps)
        factor = np.linalg.cholesky(covariance)
        residual = factor @ rng.standard_t(df=5, size=n_steps) / np.sqrt(5.0 / 3.0)
        designs.append(design)
        observations.append(design @ theta_true + residual)
        covariances.append(covariance)
    labels = ["P1", "P2", "P3"]
    system = system_from_blocks(
        designs, observations, covariances, theta_true, labels
    )
    work = model_covariance(system)
    robust = sandwich_covariance(system, "CR0")
    gap = float(np.max(np.abs(robust - work)) / np.max(np.abs(work)))
    assert gap < 0.45


def test_noiseless_sandwich_is_zero():
    system, _ = _random_system(5, 3, 4, seed=5, noise=0.0)
    robust = sandwich_covariance(system, "CR0")
    assert np.allclose(robust, 0.0, atol=1.0e-10)
    assert np.all(np.isfinite(standard_errors(system.covariance_work)))


def test_system_from_result_round_trip():
    system, _ = _random_system(5, 2, 3, seed=6)
    reports = []
    designs, observations = [], []
    for cluster in system.clusters:
        reports.append(
            SimpleNamespace(
                experiment_id=cluster.experiment_id,
                label=cluster.name,
                n_steps=cluster.n_steps,
                covariance=cluster.covariance,
            )
        )
        designs.append(cluster.design)
        observations.append(cluster.observation)
    result = SimpleNamespace(
        experiments=reports,
        case_K_matrices=designs,
        case_U_vectors=observations,
        final_parameters=system.theta,
        parameter_covariance=system.covariance_work,
        constitutive_model=SimpleNamespace(parameter_labels=system.labels),
    )
    rebuilt = system_from_result(result, labels=system.labels)
    assert rebuilt.n_clusters == 5
    assert np.allclose(rebuilt.theta, system.theta)
    payload = analyse(result, labels=system.labels, kind="CR2")
    assert payload.n_clusters == 5
    assert len(payload.deletions) == 5
    assert payload.deletions[0].theta_deleted_refit is None
    assert payload.information.n_clusters == 5
    assert payload.information.shares.shape == (5, 2)


def test_system_from_result_requires_gls_blocks():
    result = SimpleNamespace(
        experiments=[],
        case_K_matrices=[],
        case_U_vectors=[],
        final_parameters=np.ones(2),
        parameter_covariance=np.eye(2),
    )
    with pytest.raises(UncertaintyError):
        system_from_result(result)


def test_leave_one_out_rows_cover_every_cluster():
    system, _ = _random_system(5, 2, 4, seed=7)
    rows = analyse_deletion(system, kind="CR2")
    assert [row.dropped for row in rows] == system.cluster_names
    assert all(row.theta_deleted_refit is None for row in rows)
    assert all(row.influence.shape == (2,) for row in rows)


def test_leave_one_out_needs_two_experiments():
    system, _ = _random_system(1, 2, 6, seed=8)
    assert analyse_deletion(system, kind="CR2") == []


def test_information_shares_sum_to_one():
    system, _ = _random_system(6, 3, 4, seed=9)
    from vfmstudio.core.uncertainty import analyse_information

    info = analyse_information(system)
    shares = info.shares
    assert shares.shape == (6, 3)
    assert np.allclose(shares.sum(axis=0), 1.0)
    assert np.isfinite(info.condition)
    assert info.well_posed is True
    assert info.min_clusters_for_sandwich == 4


def test_information_condition_when_a_direction_is_unique():
    theta = np.array([1.0, 2.0])
    designs = [
        np.array([[1.0, 0.0], [2.0, 0.0]]),
        np.array([[1.5, 0.0], [0.5, 0.0]]),
        np.array([[0.0, 1.0], [0.0, 2.0]]),
    ]
    observations = [design @ theta for design in designs]
    covariances = [np.eye(2) for _ in designs]
    system = system_from_blocks(
        designs, observations, covariances, theta, ["P1", "P2"], ["E1", "E2", "E3"]
    )
    from vfmstudio.core.uncertainty import analyse_information

    info = analyse_information(system)
    by_name = {row.name: row for row in info.contributions}
    assert by_name["E3"].remaining_rank < 2
    assert not np.isfinite(by_name["E3"].remaining_condition)
    assert by_name["E1"].remaining_rank == 2
    assert by_name["E2"].remaining_rank == 2
    shares_p2 = {row.name: float(row.share[1]) for row in info.contributions}
    assert shares_p2["E3"] == pytest.approx(1.0)
    assert shares_p2["E1"] == pytest.approx(0.0)
    assert info.well_posed is True


def test_information_exists_when_sandwich_is_refused():
    system, _ = _random_system(3, 4, 5, seed=10)
    payload = summarise(system, "CR2")
    assert payload["well_posed"] is False
    from vfmstudio.core.uncertainty import analyse_information

    info = analyse_information(system)
    assert info.well_posed is False
    assert info.n_clusters == 3
    assert info.min_clusters_for_sandwich == 5
    assert info.shares.shape == (3, 4)
