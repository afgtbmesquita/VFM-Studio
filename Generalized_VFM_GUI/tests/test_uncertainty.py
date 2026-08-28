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


def test_identifiability_strength_uncoupled_is_one():
    from vfmstudio.core.uncertainty import identifiability_strength

    A = np.diag([2.0, 5.0, 8.0])
    theta = np.array([10.0, 20.0, 40.0])
    metrics = identifiability_strength(A, theta, ["D11", "D22", "D12"])
    assert np.allclose(metrics.coupling_quality, 1.0)
    assert np.allclose(metrics.scales, np.abs(theta))
    assert np.allclose(metrics.working_strength, np.abs(theta) ** 2 * np.diag(A))
    assert np.all(np.isnan(metrics.information_gain))


def test_identifiability_strength_near_zero_uses_characteristic_scale():
    from vfmstudio.core.uncertainty import identifiability_strength

    A = np.eye(3)
    theta = np.array([100.0, 80.0, 1.0e-12])
    metrics = identifiability_strength(A, theta, ["D11", "D22", "D16"])
    assert metrics.scales[2] == pytest.approx(90.0)
    assert metrics.scales[2] > 1.0


def test_coupling_quality_is_scale_invariant():
    from vfmstudio.core.uncertainty import identifiability_strength

    rng = np.random.default_rng(11)
    root = rng.normal(size=(3, 3))
    A = root @ root.T + np.eye(3)
    theta = np.array([12.0, 7.0, 3.0])
    labels = ["D11", "D22", "D12"]
    first = identifiability_strength(A, theta, labels)
    scaled = identifiability_strength(4.0 * A, theta, labels)
    assert np.allclose(first.coupling_quality, scaled.coupling_quality)


def test_information_gain_is_zero_when_baseline_is_all():
    from vfmstudio.core.uncertainty import identifiability_strength

    A = np.diag([2.0, 5.0, 8.0, 3.0])
    theta = np.array([10.0, 20.0, 4.0, 8.0])
    labels = ["D11", "D22", "D12", "D66"]
    metrics = identifiability_strength(A, theta, labels, A_baseline=A)
    assert np.allclose(metrics.information_gain, 0.0, atol=1e-12)


def test_information_gain_grows_when_an_informative_block_is_added():
    from vfmstudio.core.uncertainty import identifiability_strength

    A_bend = np.diag([10.0, 8.0, 5.0, 1.0e-18])
    A_twist = np.diag([0.2, 0.2, 0.1, 40.0])
    A_all = A_bend + A_twist
    theta = np.array([100.0, 80.0, 30.0, 20.0])
    labels = ["D11", "D22", "D12", "D66"]
    metrics = identifiability_strength(A_all, theta, labels, A_baseline=A_bend)
    assert np.all(metrics.information_gain > 0.0)
    assert metrics.information_gain[3] > metrics.information_gain[0]
    assert metrics.information_gain[3] > 1.0


def test_assemble_sensitivity_sums_selected_blocks():
    from types import SimpleNamespace

    from vfmstudio.core.uncertainty import assemble_sensitivity

    rows = [
        SimpleNamespace(experiment_id=1, sensitivity=np.diag([1.0, 2.0])),
        SimpleNamespace(experiment_id=2, sensitivity=np.diag([10.0, 0.0])),
        SimpleNamespace(experiment_id=3, sensitivity=np.diag([0.0, 7.0])),
    ]
    baseline = assemble_sensitivity(rows, [1, 2], n_parameters=2)
    assert np.allclose(baseline, np.diag([11.0, 2.0]))


def test_default_baseline_unticks_twist():
    from vfmstudio.core.uncertainty import default_baseline_flags

    flags = default_baseline_flags(
        ["Specimen A1 Pure Bending", "Specimen B1 Twist"],
        [1, 2],
    )
    assert flags[1] is True
    assert flags[2] is False


def test_deletion_influence_tracks_the_parameter_that_shifts():
    from types import SimpleNamespace

    from vfmstudio.core.uncertainty import identifiability_strength

    A = np.diag([2.0, 5.0, 8.0])
    theta = np.array([10.0, 20.0, 40.0])
    deletions = [
        SimpleNamespace(dropped="bending A", delta_fixed=np.array([0.1, 0.2, 8.0])),
        SimpleNamespace(dropped="twist", delta_fixed=np.array([0.05, 0.1, 0.4])),
    ]
    metrics = identifiability_strength(
        A, theta, ["D11", "D22", "D12"], deletions=deletions
    )
    assert metrics.deletion_influence[2] == pytest.approx(8.0 / 40.0)
    assert metrics.deletion_influence[0] == pytest.approx(0.1 / 10.0)
    assert metrics.pulling_experiment[2] == "bending A"


def test_robustness_verdict_table():
    from vfmstudio.core.uncertainty.information import (
        G_HIGH,
        R_HIGH,
        robustness_verdict,
    )

    assert robustness_verdict(G_HIGH + 1.0, R_HIGH / 2.0) == "excellent"
    assert robustness_verdict(G_HIGH + 1.0, R_HIGH + 0.1) == "conflicting"
    assert robustness_verdict(0.0, R_HIGH / 2.0) == "redundant"
    assert robustness_verdict(0.0, R_HIGH + 0.1) == "problematic"


def test_conflicting_d12_is_high_gain_and_high_influence():
    from types import SimpleNamespace

    from vfmstudio.core.uncertainty import identifiability_strength

    A_bend = np.diag([10.0, 8.0, 1.0e-18])
    A_all = A_bend + np.diag([0.2, 0.2, 40.0])
    theta = np.array([100.0, 80.0, 20.0])
    labels = ["D11", "D22", "D12"]
    deletions = [
        SimpleNamespace(dropped="bending", delta_fixed=np.array([0.2, 0.3, 6.0])),
        SimpleNamespace(dropped="other", delta_fixed=np.array([0.1, 0.1, 0.2])),
    ]
    metrics = identifiability_strength(
        A_all, theta, labels, A_baseline=A_bend, deletions=deletions
    )
    assert metrics.information_gain[2] > 1.0
    assert metrics.deletion_influence[2] == pytest.approx(6.0 / 20.0)
    assert metrics.verdicts[2] == "conflicting"
    assert metrics.verdicts[0] == "redundant"


def test_subset_deletion_shifts_is_eq63_in_original_units():
    from types import SimpleNamespace

    from vfmstudio.core.uncertainty import subset_deletion_shifts

    result = SimpleNamespace(
        cluster_ids=[1, 2, 3],
        cluster_names=["bend A", "bend B", "twist"],
        scores=np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]]),
        information=SimpleNamespace(
            blocks=np.stack(
                [
                    np.diag([4.0, 1.0]),
                    np.diag([1.0, 4.0]),
                    np.diag([2.0, 2.0]),
                ]
            )
        ),
    )
    bending = subset_deletion_shifts(result, [1, 2])
    assert [row.name for row in bending] == ["bend A", "bend B"]
    assert np.allclose(bending[0].shift, [-2.0, 0.0])
    assert np.allclose(bending[1].shift, [0.0, -3.0])
    all_tests = subset_deletion_shifts(result, [1, 2, 3])
    assert len(all_tests) == 3
    assert not np.allclose(all_tests[0].shift, bending[0].shift)
    assert subset_deletion_shifts(result, [1]) == []


def test_identifiability_strength_plot_is_two_panels():
    from matplotlib.figure import Figure

    from vfmstudio.core.uncertainty import ExperimentShift, identifiability_strength
    from vfmstudio.gui.results.identifiability_strength import (
        plot_identifiability_strength,
    )

    A_bend = np.diag([2.0, 5.0, 8.0, 1.0e-18])
    A_all = A_bend + np.diag([0.1, 0.1, 0.1, 12.0])
    theta = np.array([10.0, 20.0, 4.0, 8.0])
    labels = ["D11", "D22", "D12", "D66"]
    metrics = identifiability_strength(A_all, theta, labels, A_baseline=A_bend)
    shifts = [
        ExperimentShift("bend A", 1, np.array([0.2, 0.4, 1.2, 0.05])),
        ExperimentShift("bend B", 2, np.array([-0.1, 0.2, -0.8, 0.01])),
    ]
    figure = Figure()
    plot_identifiability_strength(figure, labels, metrics, shifts, theta=theta)
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    FigureCanvasAgg(figure).draw()
    panels = [ax for ax in figure.axes if ax.axison]
    assert len(panels) == 2
    x0 = [ax.get_position().x0 for ax in panels]
    assert x0[1] > x0[0]
    widths = [ax.get_position().width for ax in panels]
    assert min(widths) > 0.25
    assert "Δθ" in panels[0].get_ylabel()
    assert r"$G_k$" in panels[1].get_ylabel()

    relative = Figure()
    plot_identifiability_strength(
        relative, labels, metrics, shifts, theta=theta, relative=True
    )
    FigureCanvasAgg(relative).draw()
    rel_panels = [ax for ax in relative.axes if ax.axison]
    assert "Δθ / |θ̂|" in rel_panels[0].get_ylabel()


def test_scaled_deletion_shifts_divide_by_abs_theta():
    from vfmstudio.core.uncertainty import ExperimentShift
    from vfmstudio.gui.results.identifiability_strength import scaled_deletion_shifts

    shifts = [ExperimentShift("bend A", 1, np.array([2.0, -4.0]))]
    theta = np.array([10.0, 20.0])
    raw = scaled_deletion_shifts(shifts, theta, relative=False)
    relative = scaled_deletion_shifts(shifts, theta, relative=True)
    assert np.allclose(raw[0], [2.0, -4.0])
    assert np.allclose(relative[0], [0.2, -0.2])


def test_deletion_shifts_are_a_single_row():
    from matplotlib.figure import Figure

    from vfmstudio.gui.results.uncertainty_plots import plot_deletion_shifts

    system, _ = _random_system(4, 4, 3, seed=12)
    reports, designs, observations = [], [], []
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
    payload = analyse(result, labels=system.labels, kind="CR2")
    figure = Figure()
    plot_deletion_shifts(figure, payload)
    panels = [ax for ax in figure.axes if ax.get_visible() and ax.axison]
    assert len(panels) == 4
    y0 = [ax.get_position().y0 for ax in panels]
    assert max(y0) - min(y0) < 0.05
    assert figure.legends
