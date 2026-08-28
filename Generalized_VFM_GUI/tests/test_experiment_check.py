"""Per-experiment load-step coefficient extraction and plots."""

from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

from vfmstudio.gui.results.experiment_check import (
    collect_experiment_steps,
    plot_by_experiment,
    plot_coefficients_vs_step,
    plot_deviation,
)


def _result():
    return SimpleNamespace(
        final_parameters=np.array([10.0, 20.0, 3.0, 8.0]),
        constitutive_model=SimpleNamespace(
            parameter_labels=["D11", "D22", "D12", "D66"]
        ),
        experiments=[
            SimpleNamespace(
                experiment_id=1,
                label="Longitudinal",
                step_labels=["a", "b", "c"],
                step_estimates=np.array(
                    [
                        [11.0, 19.0, 2.8, 8.2],
                        [9.5, 21.0, 3.1, 7.7],
                        [10.2, 20.4, 3.0, 8.0],
                    ]
                ),
            ),
            SimpleNamespace(
                experiment_id=2,
                label="Transverse",
                step_labels=["a", "b"],
                step_estimates=np.array(
                    [
                        [12.0, 18.5, 2.5, 8.4],
                        [9.0, 21.5, 3.4, 7.6],
                    ]
                ),
            ),
        ],
    )


def test_collect_experiment_steps_shapes():
    series, labels, theta = collect_experiment_steps(_result())
    assert labels == ["D11", "D22", "D12", "D66"]
    assert np.allclose(theta, [10.0, 20.0, 3.0, 8.0])
    assert [item.name for item in series] == ["Longitudinal", "Transverse"]
    assert series[0].values.shape == (3, 4)
    assert series[1].values.shape == (2, 4)
    assert series[0].step_labels == ["a", "b", "c"]


def test_collect_reshapes_flat_step_estimates():
    result = _result()
    result.experiments[0].step_estimates = np.arange(8, dtype=float)
    series, _, _ = collect_experiment_steps(result)
    assert series[0].values.shape == (2, 4)


def test_collect_skips_missing_estimates():
    result = _result()
    result.experiments[1].step_estimates = None
    series, _, _ = collect_experiment_steps(result)
    assert len(series) == 1
    assert series[0].name == "Longitudinal"


def test_step_coefficient_plots_draw():
    from matplotlib.figure import Figure

    series, labels, theta = collect_experiment_steps(_result())
    for drawer in (
        plot_coefficients_vs_step,
        plot_by_experiment,
        plot_deviation,
    ):
        figure = Figure()
        drawer(figure, series, labels, theta)
        assert figure.axes
