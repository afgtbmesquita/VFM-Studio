"""GLS parameter evolution: the FGLS outer loop, and the per-load-step estimates.

Plots THIS study's campaign (longitudinal + transverse + edge twist).  The
plotting helpers are shared -- ../twist_four_coupon/plot_evolution.py imports
them from here rather than duplicating them, and each study writes only into
its own figures/ and data/.

Two panels per campaign:

  LEFT   theta after each FGLS iteration.  The loop alternates
         theta -> C(theta) -> theta, so this is the convergence of the joint
         (parameter, noise-covariance) fixed point, not a line search.

  RIGHT  the per-load-step estimates U_t that the covariance weights together.
         Because H_i = I, the FGLS answer is a matrix-weighted mean of these,
         so the scatter here IS the information the weighting has to work with.

Run:  python -m studies.twist_four_coupon.plot_evolution
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from vfm_gls_identification import run_gls_identification_cases   # noqa: E402
from vfm_user_inputs import (GlsSettings, IdentificationSettings,  # noqa: E402
                             LinearAlgebraSettings)
from workflows.campaign_common import (ANALYSIS_TYPE,              # noqa: E402
                                       INITIAL_PARAMETERS,
                                       MATERIAL_TYPE,
                                       TESTS_WITH_TWIST_RAW,
                                       build_campaign)
from workflows.make_figures import apply_style, math_label, soft_grid  # noqa: E402

LABELS = ["D11", "D12", "D22", "D66"]
COLOURS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]


def run(tests):
    cases, ids, _ = build_campaign(tests, verbose=False)
    return run_gls_identification_cases(
        cases=cases, material_type=MATERIAL_TYPE, analysis_type=ANALYSIS_TYPE,
        initial_parameters=INITIAL_PARAMETERS,
        identification_settings=IdentificationSettings(
            max_iterations=40, convergence_tolerance=1.0e-8),
        linear_algebra_settings=LinearAlgebraSettings(),
        experiment_ids=ids, gls_settings=GlsSettings(weighting="gls"),
        verbose=False)


def iteration_table(result):
    """(n_iterations+1, 4) -- the initial guess followed by each iterate."""
    rows = [np.asarray(result.history[0].parameters_before, dtype=float)]
    rows += [np.asarray(record.parameters_after, dtype=float)
             for record in result.history]
    return np.vstack(rows)


def panel_iterations(axis, table, final):
    steps = np.arange(table.shape[0])
    for index, (label, colour) in enumerate(zip(LABELS, COLOURS)):
        axis.plot(steps, table[:, index], "-o", color=colour, markersize=2.6,
                  linewidth=1.2, label=math_label(label))
        axis.axhline(final[index], color=colour, linestyle=":", linewidth=0.8)
    axis.set_xlabel("FGLS iteration")
    axis.set_ylabel("stiffness  (N mm)")
    axis.set_title("Parameter evolution through the FGLS loop")
    axis.legend(ncol=4, fontsize=6.5, loc="best")
    soft_grid(axis, which="both")


def panel_steps(axis, result):
    offset = 0
    for report in result.experiments:
        estimates = report.step_estimates
        if estimates is None:
            continue
        estimates = np.asarray(estimates, dtype=float)
        x = np.arange(offset, offset + estimates.shape[0])
        for index, colour in enumerate(COLOURS):
            axis.plot(x, estimates[:, index], "o", color=colour, markersize=3,
                      alpha=0.85)
        axis.axvline(offset - 0.5, color="0.8", linewidth=0.7)
        axis.text(offset + 0.1, axis.get_ylim()[1], report.label,
                  fontsize=5.5, va="top", rotation=90, color="0.35")
        offset += estimates.shape[0]
    for index, (label, colour) in enumerate(zip(LABELS, COLOURS)):
        axis.axhline(result.final_parameters[index], color=colour,
                     linestyle="--", linewidth=1.0,
                     label=f"{math_label(label)} (FGLS)")
    axis.set_xlabel("load step, grouped by experiment")
    axis.set_ylabel("per-step estimate  (N mm)")
    axis.set_title("What the covariance weights together")
    axis.legend(ncol=2, fontsize=6, loc="best")
    soft_grid(axis, which="both")


def figure_for(result, title, path, figures_dir=None):
    table = iteration_table(result)
    final = np.asarray(result.final_parameters, dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    panel_iterations(axes[0], table, final)
    panel_steps(axes[1], result)
    figure.suptitle(title, y=1.03)
    figure.tight_layout()
    (figures_dir or path.parent).mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".pdf"))
    figure.savefig(path.with_suffix(".png"), dpi=600)
    plt.close(figure)
    print(f"  wrote {path.name}.pdf + .png")
    return table


def main() -> None:
    apply_style()
    result = run(TESTS_WITH_TWIST_RAW)
    table = figure_for(
        result,
        "Longitudinal + transverse + edge twist  (FGLS)",
        HERE / "figures" / "fig4_parameter_evolution")
    print(f"  converged={result.converged} in {result.iterations_completed} "
          f"iterations, theta = "
          + ", ".join(f"{l} {v:.3f}"
                      for l, v in zip(LABELS, result.final_parameters)))

    (HERE / "data").mkdir(parents=True, exist_ok=True)
    out = HERE / "data" / "parameter_evolution.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration"] + LABELS)
        for iteration, values in enumerate(table):
            writer.writerow([iteration] + [f"{v:.6f}" for v in values])
    print(f"  wrote data/{out.name}")


if __name__ == "__main__":
    main()
