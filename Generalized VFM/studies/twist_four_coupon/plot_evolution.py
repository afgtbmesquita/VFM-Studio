"""GLS parameter evolution for the four-coupon campaign.

The plotting helpers live in ../twist_three_coupon/plot_evolution.py -- that
study establishes the twist BC, so it owns the shared code.  This module only
selects the campaign and writes into THIS study's figures/ and data/.

Run:  python -m studies.twist_four_coupon.plot_evolution
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from studies.twist_three_coupon.plot_evolution import (   # noqa: E402
    LABELS, figure_for, run)
from workflows.campaign_common import TESTS_WITH_45_TWIST_RAW   # noqa: E402
from workflows.make_figures import apply_style                  # noqa: E402


def main() -> None:
    apply_style()
    result = run(TESTS_WITH_45_TWIST_RAW)
    table = figure_for(
        result,
        "Longitudinal + transverse + 45 deg + edge twist  (FGLS)",
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
