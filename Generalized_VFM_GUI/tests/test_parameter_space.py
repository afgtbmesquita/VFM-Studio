"""Frozen-weight GLS combinations for the Parameter space tab."""

from __future__ import annotations

import numpy as np

from vfmstudio.core.parameter_space import (
    compute_parameter_space,
    default_axis_labels,
    from_system,
    assign_highlight_groups,
    group_statistics,
    is_full_combination,
    mask_for_experiment_ids,
    resolve_plot_axes,
    theta_for_experiment_ids,
)
from vfmstudio.core.uncertainty import system_from_blocks


def _random_system(n_clusters: int, n_par: int, n_steps: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    theta_true = rng.normal(size=n_par)
    designs, observations, covariances = [], [], []
    for _ in range(n_clusters):
        design = rng.normal(size=(n_steps, n_par))
        root = rng.normal(size=(n_steps, n_steps)) / np.sqrt(n_steps)
        covariance = root @ root.T + np.eye(n_steps)
        factor = np.linalg.cholesky(covariance)
        residual = factor @ rng.standard_normal(n_steps)
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


def _cluster_gls(cluster):
    bread = cluster.whitened_design.T @ cluster.whitened_design
    rhs = cluster.whitened_design.T @ cluster.whitened_observation
    return np.linalg.solve(bread, rhs)


def test_three_clusters_make_seven_subsets():
    system, _ = _random_system(3, 4, 5, seed=1)
    space = from_system(system)
    assert space.n_subsets == 7
    assert space.n_clusters == 3
    assert set(space.masks.tolist()) == set(range(1, 8))
    assert space.full_mask == 7
    assert np.all(space.n_tests >= 1)
    assert int(space.n_tests.max()) == 3


def test_full_mask_matches_stacked_gls():
    system, _ = _random_system(3, 4, 6, seed=2)
    space = from_system(system)
    full = space.theta[space.index_of_mask(space.full_mask)]
    assert np.allclose(full, system.theta, atol=1.0e-10)
    assert space.full_gap < 1.0e-10


def test_leave_one_out_matches_system_without():
    system, _ = _random_system(3, 4, 5, seed=3)
    space = from_system(system)
    dropped = system.cluster_names[1]
    reduced = system.without(dropped)
    mask = space.full_mask ^ (1 << 1)
    theta = space.theta[space.index_of_mask(mask)]
    assert space.subset_names(mask) == ["E1", "E3"]
    assert np.allclose(theta, reduced.theta, atol=1.0e-10)


def test_singleton_matches_cluster_gls():
    system, _ = _random_system(3, 4, 5, seed=4)
    space = from_system(system)
    for index, cluster in enumerate(system.clusters):
        mask = 1 << index
        theta = space.theta[space.index_of_mask(mask)]
        assert np.allclose(theta, _cluster_gls(cluster), atol=1.0e-10)
        assert space.subset_names(mask) == [cluster.name]


def test_eight_clusters_make_255_subsets():
    system, _ = _random_system(8, 4, 3, seed=5)
    space = from_system(system)
    assert space.n_subsets == 255
    assert space.full_mask == 255
    full = space.theta[space.index_of_mask(255)]
    assert np.allclose(full, system.theta, atol=1.0e-9)


def test_compute_parameter_space_accepts_system_and_none():
    system, _ = _random_system(2, 3, 4, seed=6)
    assert compute_parameter_space(None) is None
    space = compute_parameter_space(system)
    assert space is not None
    assert space.n_subsets == 3


def test_default_axes_prefer_d_pairs():
    labels = ["D11", "D12", "D22", "D66"]
    assert default_axis_labels(labels) == ("D11", "D22", "D12", "D66")
    axes = resolve_plot_axes(labels, "", "", "", "")
    assert axes == (0, 2, 1, 3)
    axes = resolve_plot_axes(labels, "D12", "D66", "D11", "D22")
    assert axes == (1, 3, 0, 2)


def test_default_axes_fallback_when_names_differ():
    labels = ["A11", "A12", "A22"]
    left_x, left_y, right_x, right_y = default_axis_labels(labels)
    assert left_x == "A11"
    assert left_y == "A22"
    assert right_x == "A12"
    assert right_y == "A22"


def test_theta_lookup_by_experiment_ids():
    system, _ = _random_system(3, 4, 5, seed=7)
    space = from_system(system)
    ids = list(space.experiment_ids)
    assert is_full_combination(space, ids)
    full = theta_for_experiment_ids(space, ids)
    assert full is not None
    assert np.allclose(full, system.theta, atol=1.0e-10)
    subset = theta_for_experiment_ids(space, ids[:2])
    mask = mask_for_experiment_ids(space, ids[:2])
    assert mask is not None
    assert np.allclose(subset, space.theta[space.index_of_mask(mask)], atol=1.0e-12)
    assert theta_for_experiment_ids(space, [ids[0], 99]) is None
    assert mask_for_experiment_ids(space, []) is None
    assert is_full_combination(space, ids[:2]) is False


def test_assign_highlight_groups_is_intersection():
    system, _ = _random_system(3, 4, 5, seed=8)
    space = from_system(system)
    ids = list(space.experiment_ids)
    one = assign_highlight_groups(
        space, [{"name": "E1", "color": "#D55E00", "experiment_ids": ids[:1]}]
    )
    two = assign_highlight_groups(
        space, [{"name": "E1∩E2", "color": "#0072B2", "experiment_ids": ids[:2]}]
    )
    assert int(np.sum(one == 0)) == 4
    assert int(np.sum(two == 0)) == 2
    assert one[space.index_of_mask(mask_for_experiment_ids(space, ids[:1]))] == 0
    assert two[space.index_of_mask(mask_for_experiment_ids(space, ids[:1]))] == -1
    assert two[space.index_of_mask(mask_for_experiment_ids(space, ids[:2]))] == 0
    assert two[space.index_of_mask(space.full_mask)] == 0


def test_assign_highlight_groups_prefers_largest_match():
    system, _ = _random_system(4, 4, 5, seed=9)
    space = from_system(system)
    ids = list(space.experiment_ids)
    groups = [
        {"name": "first", "color": "#D55E00", "experiment_ids": ids[:1]},
        {"name": "first two", "color": "#0072B2", "experiment_ids": ids[:2]},
    ]
    assigned = assign_highlight_groups(space, groups)
    mask_1 = mask_for_experiment_ids(space, ids[:1])
    mask_12 = mask_for_experiment_ids(space, ids[:2])
    mask_123 = mask_for_experiment_ids(space, ids[:3])
    mask_14 = mask_for_experiment_ids(space, [ids[0], ids[3]])
    mask_4 = mask_for_experiment_ids(space, ids[3:])
    assert assigned[space.index_of_mask(mask_1)] == 0
    assert assigned[space.index_of_mask(mask_12)] == 1
    assert assigned[space.index_of_mask(mask_123)] == 1
    assert assigned[space.index_of_mask(mask_14)] == 0
    assert assigned[space.index_of_mask(mask_4)] == -1


def test_assign_highlight_groups_skips_empty():
    system, _ = _random_system(2, 3, 4, seed=9)
    space = from_system(system)
    assigned = assign_highlight_groups(
        space, [{"name": "empty", "color": "#000000", "experiment_ids": []}]
    )
    assert np.all(assigned == -1)


def test_group_statistics_match_intersection_cloud():
    system, _ = _random_system(3, 4, 5, seed=11)
    space = from_system(system)
    ids = list(space.experiment_ids)
    empty = group_statistics(space, [])
    assert empty["n"] == 0
    assert np.all(np.isnan(empty["mean"]))
    one = group_statistics(space, ids[:1])
    assert one["n"] == 4
    two = group_statistics(space, ids[:2])
    assert two["n"] == 2
    pick = np.array(
        [ids[0] in space.subset_ids(int(mask)) for mask in space.masks],
        dtype=bool,
    )
    values = space.theta[pick]
    assert np.allclose(one["mean"], np.mean(values, axis=0))
    assert np.allclose(one["std"], np.std(values, axis=0, ddof=1))
