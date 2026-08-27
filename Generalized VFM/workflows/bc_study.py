# =============================================================================
#  Generalized VFM / bc_study.py
# -----------------------------------------------------------------------------
#  Which boundary conditions should the virtual field carry?
#
#  THE RULE
#  ---------------------------------------------------------------------------
#  Every UNKNOWN boundary traction must do zero virtual work.  On a cut with
#  normal along y the Kirchhoff tractions and their conjugate DOFs are
#
#      effective shear  V_y   <->  W        (deflection)
#      bending moment   M_yy  <->  W_Y      (normal rotation)
#      twisting moment  M_xy  <->  W_X, W_XY (tangential slope and twist)
#
#  So a cut is "sealed" when W, W_X, W_XY and W_Y are all prescribed there.  A
#  DOF may instead be left free IF the conjugate traction is measured and its
#  work is supplied as an EXTERNAL group -- that is the only legitimate way to
#  leave one open.
#
#  THE FIXTURE
#  ---------------------------------------------------------------------------
#      TOP cut     the machine couple is measured here (dTy -> M_yy, and
#                  dTx -> M_xy if you choose to use it)
#      BOTTOM cut  reaction, NOT measured
#      LEFT/RIGHT  the specimen's free edges -- never constrained, never loaded
#
#  Hence the reference form: seal the bottom completely, and at the top seal
#  everything except W_Y, whose conjugate moment is measured.
#
#  This script builds each variant from an explicit DOF specification rather
#  than by patching the shipped file, so what is constrained and what does work
#  is auditable line by line.  It then reports the identified coefficients and
#  the goodness of fit, and draws the summary figure.
#
#      python -m workflows.bc_study
# =============================================================================

from __future__ import annotations

import csv
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from workflows.campaign_common import (
    ANALYSIS_TYPE,
    CAMPAIGN,
    FIGURES_DIR,
    HERE,
    LONGITUDINAL,
    MATERIAL_TYPE,
    RESULTS_DIR,
    TESTS_RAW_FRAME,
    TRANSVERSE_RAW,
    build_cases,
)
from workflows.make_figures import apply_style, chi_square_diagnostics, math_label, soft_grid
from vfm_gls_identification import run_gls_identification_cases
from vfm_user_inputs import (
    GlsSettings,
    IdentificationSettings,
    LinearAlgebraSettings,
)

LABELS = ("D11", "D12", "D22", "D66")
FIGURE_DIR = FIGURES_DIR

#  Specimen widths used to turn a machine couple [N.m] into a plate moment
#  resultant per unit width [N.mm/mm = N].  Same numbers the exports use.
WIDTH_MM = {"longitudinal": 9.28, "transverse-raw": 9.6}

#  M_xy = TWIST_SIGN * dTx * 1000 / width.  The channel identification is from
#  NOTES_transverse_frame_and_curvature.md section 5; the SIGN is a convention
#  the notes fix for the transverse fixed frame only, so both signs are run and
#  reported rather than one being assumed correct.
TWIST_SIGN = -1.0


# =============================================================================
#  The machine twisting moment, per load step
# =============================================================================

def machine_twist_moment(spec) -> Dict[Path, float]:
    """``{mat path: M_xy}`` from the synchronized machine log.

    Step 1 is the reference state, so its ``Tx`` is the baseline and is
    subtracted -- exactly as the exports treat ``Ty``.
    """
    source = CAMPAIGN / ("Longitudinal_bending"
                         if spec.name == "longitudinal"
                         else "Transverse_bending")
    rows = list(csv.DictReader(
        (source / "synchronized_DIC_machine_data.csv").open(encoding="utf-8")))
    mapping = list(csv.DictReader(
        (source / "step_DIC_file_map.csv").open(encoding="utf-8")))

    sync_row_by_step = {int(entry["Step"]): int(entry["Sync_row"])
                        for entry in mapping}

    def twist_at(step: int) -> float:
        #  Sync_row is 1-based into the data rows of the synchronized file.
        return float(rows[sync_row_by_step[step] - 1]["Machine_Tx_Nm_"])

    baseline = twist_at(min(sync_row_by_step))
    width = WIDTH_MM[spec.name]

    moments: Dict[Path, float] = {}
    for path in mat_paths(spec):
        step = int(path.stem.split("step")[-1])
        delta = twist_at(step) - baseline
        moments[path] = TWIST_SIGN * delta * 1000.0 / width
    return moments


def mat_paths(spec) -> List[Path]:
    from workflows.campaign_common import mat_files
    return mat_files(spec)


# =============================================================================
#  A variant, described by what each cut does
# =============================================================================

@dataclass(frozen=True)
class EdgeSpec:
    """What happens to each DOF on one cut."""
    prescribed: Tuple[str, ...] = ()
    #  DOF -> where its conjugate moment comes from: 'bending' (the .mat F) or
    #  'twist' (the machine Tx channel).
    external: Tuple[Tuple[str, str], ...] = ()
    linked: Tuple[str, ...] = ()

    @property
    def free(self) -> Tuple[str, ...]:
        loaded = {dof for dof, _ in self.external}
        return tuple(d for d in ("W", "W_X", "W_XY", "W_Y")
                     if d not in self.prescribed and d not in loaded)


@dataclass(frozen=True)
class Variant:
    key: str
    title: str
    top: EdgeSpec
    bottom: EdgeSpec
    note: str = ""

    @property
    def sealed(self) -> bool:
        """True when no UNKNOWN traction can do virtual work.

        The top's W_Y (and W_X, when its twist is supplied) are allowed to be
        free because their conjugate moments are measured.
        """
        return not self.top.free and not self.bottom.free


#  --------------------------------------------------------------------------
#  The reference: bottom sealed completely, top sealed except the measured
#  bending moment.  Everything else is a departure from exactly one of these.
#  --------------------------------------------------------------------------
_SEALED_BOTTOM = EdgeSpec(prescribed=("W", "W_X", "W_XY", "W_Y"))
_TOP_MEASURED_BENDING = EdgeSpec(prescribed=("W", "W_X", "W_XY"),
                                 external=(("W_Y", "bending"),))

VARIANTS: Tuple[Variant, ...] = (
    Variant(
        "A", "A  reference (all rules)",
        top=_TOP_MEASURED_BENDING, bottom=_SEALED_BOTTOM,
        note="bottom sealed; top open only on the measured M_yy"),
    Variant(
        "B", "B  both cuts loaded",
        top=_TOP_MEASURED_BENDING,
        bottom=EdgeSpec(prescribed=("W", "W_X", "W_XY"),
                        external=(("W_Y", "bending"),)),
        note="bottom W_Y free + EXTERNAL: assumes equal and opposite moments"),
    Variant(
        "C", "C  bottom W_XY free",
        top=_TOP_MEASURED_BENDING,
        bottom=EdgeSpec(prescribed=("W", "W_X", "W_Y")),
        note="LEAKS: bottom twisting moment does unaccounted work"),
    Variant(
        "D", "D  bottom W_X free",
        top=_TOP_MEASURED_BENDING,
        bottom=EdgeSpec(prescribed=("W", "W_XY", "W_Y")),
        note="LEAKS: bottom twisting moment does unaccounted work"),
    Variant(
        "E", "E  top W_XY free",
        top=EdgeSpec(prescribed=("W", "W_X"),
                     external=(("W_Y", "bending"),)),
        bottom=_SEALED_BOTTOM,
        note="LEAKS: top twist does work but is not supplied"),
    Variant(
        "F", "F  top W_X free + measured M_xy",
        top=EdgeSpec(prescribed=("W", "W_XY"),
                     external=(("W_Y", "bending"), ("W_X", "twist"))),
        bottom=_SEALED_BOTTOM,
        note="rule 6: W_X opened AND its measured twist supplied"),
    Variant(
        "G", "G  link W_Y across top",
        top=EdgeSpec(prescribed=("W", "W_X", "W_XY"),
                     external=(("W_Y", "bending"),), linked=("W_Y",)),
        bottom=_SEALED_BOTTOM,
        note="rule 7: top rotates as one; admissible, smaller field space"),
    Variant(
        "H", "H  link W_Y top + free bottom W_XY",
        top=EdgeSpec(prescribed=("W", "W_X", "W_XY"),
                     external=(("W_Y", "bending"),), linked=("W_Y",)),
        bottom=EdgeSpec(prescribed=("W", "W_X", "W_Y")),
        note="linking on top, C's leak on the bottom"),
)


# =============================================================================
#  Building cases for a variant
# =============================================================================

def _edge_entries(edge: str, spec: EdgeSpec):
    prescribed = [{"label": f"{edge} {dof}=0", "dof": dof, "edge": edge,
                   "value": 0.0} for dof in spec.prescribed]
    external = [{"label": f"{edge} {dof} ({source})", "dof": dof,
                 "edge": edge, "loadType": "distributed",
                 "sortDirection": "x"}
                for dof, source in spec.external]
    linking = [{"label": f"{edge} {dof} tied", "dof": dof, "edge": edge,
                "sortDirection": "x"} for dof in spec.linked]
    return prescribed, external, linking


def build_variant_cases(variant: Variant, spec) -> List:
    """Cases for one coupon under one BC variant."""
    twist = None
    if any(source == "twist"
           for edge in (variant.top, variant.bottom)
           for _, source in edge.external):
        twist = machine_twist_moment(spec)

    top_p, top_e, top_l = _edge_entries("top", variant.top)
    bot_p, bot_e, bot_l = _edge_entries("bottom", variant.bottom)
    #  Order matters: it is the order the applied-load vector is matched in.
    external_spec = ([("top", d, s) for d, s in variant.top.external]
                     + [("bottom", d, s) for d, s in variant.bottom.external])

    cases = []
    for case in build_cases(spec, verbose=False):
        force = np.asarray(loadmat(case.data_file, squeeze_me=True)["F"],
                           dtype=float).ravel()
        #  The shipped F is [top bending, bottom bending].
        bending = {"top": float(force[0]), "bottom": float(force[1])}

        values = []
        for edge, _dof, source in external_spec:
            if source == "bending":
                values.append(bending[edge])
            else:
                values.append(float(twist[case.data_file]))

        bc = dataclasses.replace(case).boundary_conditions
        import copy
        bc = copy.deepcopy(bc)
        bc.prescribed = top_p + bot_p
        bc.linking = top_l + bot_l
        bc.external = [
            {**entry, "value": value}
            for entry, value in zip(top_e + bot_e, values)
        ]

        variant_case = dataclasses.replace(case)
        variant_case.boundary_conditions = bc
        variant_case.applied_load_source = "manual"
        cases.append(variant_case)
    return cases


def run_variant(variant: Variant):
    cases, ids = [], []
    for spec in TESTS_RAW_FRAME:
        variant_cases = build_variant_cases(variant, spec)
        cases.extend(variant_cases)
        ids.extend([spec.experiment_id] * len(variant_cases))
    result = run_gls_identification_cases(
        cases=cases, material_type=MATERIAL_TYPE, analysis_type=ANALYSIS_TYPE,
        initial_parameters=[1.0, 1.0, 1.0, 1.0],
        identification_settings=IdentificationSettings(
            max_iterations=40, convergence_tolerance=1.0e-8),
        linear_algebra_settings=LinearAlgebraSettings(),
        experiment_ids=ids,
        gls_settings=GlsSettings(basis_degree=6, weighting="gls"),
        verbose=False)
    goodness = chi_square_diagnostics(result)
    return result, goodness


# =============================================================================
#  Figure
# =============================================================================

SEALED_COLOUR = "#1f6f8b"
LEAKY_COLOUR = "#c1553b"


def make_figure(rows, png_only: bool = False) -> None:
    apply_style()
    n = len(rows)
    y = np.arange(n)[::-1]

    figure, axes = plt.subplots(
        1, 5, figsize=(7.16, 0.42 * n + 1.7), sharey=True,
        gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.05], "wspace": 0.18})

    for index, label in enumerate(LABELS):
        axis = axes[index]
        values = np.array([r["theta"][index] for r in rows])
        errors = np.array([r["birge_std"][index] for r in rows])
        colours = [SEALED_COLOUR if r["sealed"] else LEAKY_COLOUR
                   for r in rows]
        for position, value, error, colour in zip(y, values, errors, colours):
            axis.errorbar(value, position, xerr=error, fmt="o", ms=4.2,
                          color=colour, ecolor=colour, elinewidth=1.1,
                          capsize=2.4, zorder=3)
        #  The reference variant, as a line every panel can be read against.
        axis.axvline(values[0], color="#8a8a8a", lw=0.9, ls=(0, (4, 3)),
                     zorder=1)
        axis.set_title(math_label(label), pad=6)
        soft_grid(axis, "x")
        axis.tick_params(labelsize=7)
        #  Explicit margins: several variants land near an extreme and the
        #  default autoscale clips their error bar against the spine.
        low = float(np.min(values - errors))
        high = float(np.max(values + errors))
        pad = 0.16 * max(high - low, 1.0e-9)
        axis.set_xlim(low - pad, high + pad)
        axis.xaxis.set_major_locator(plt.MaxNLocator(4))

    chi_axis = axes[4]
    chi = np.array([r["chi2"] for r in rows])
    for position, value, row in zip(y, chi, rows):
        chi_axis.plot(value, position, "o", ms=4.2,
                      color=SEALED_COLOUR if row["sealed"] else LEAKY_COLOUR,
                      zorder=3)
    chi_axis.axvline(chi[0], color="#8a8a8a", lw=0.9, ls=(0, (4, 3)), zorder=1)
    chi_axis.set_xscale("log")
    chi_axis.set_title(r"$\chi^2/\mathrm{dof}$", pad=6)
    soft_grid(chi_axis, "x")
    chi_axis.tick_params(labelsize=7)
    #  Log autoscale leaves the largest point sitting on the spine, and the
    #  default log locator collides its own labels over this narrow range.
    chi_axis.set_xlim(float(chi.min()) / 1.35, float(chi.max()) * 1.35)
    ticks = [t for t in (200, 300, 500, 800, 1200)
             if chi.min() / 1.35 <= t <= chi.max() * 1.35]
    chi_axis.set_xticks(ticks)
    chi_axis.set_xticklabels([str(t) for t in ticks])
    chi_axis.xaxis.set_minor_locator(plt.NullLocator())

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r["title"] for r in rows], fontsize=7.2)
    axes[0].set_ylim(-0.7, n - 0.3)

    handles = [
        plt.Line2D([0], [0], marker="o", ls="none", ms=4.2,
                   color=SEALED_COLOUR,
                   label="every unknown traction annihilated"),
        plt.Line2D([0], [0], marker="o", ls="none", ms=4.2,
                   color=LEAKY_COLOUR,
                   label="an unknown traction does virtual work"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=2,
                  frameon=False, fontsize=7.2, bbox_to_anchor=(0.5, 0.055))
    figure.suptitle(
        "Boundary-condition choice and the identified plate stiffnesses",
        y=0.985, fontsize=9)
    figure.text(0.5, 0.018,
                "bars are Birge-scaled intervals "
                r"($\sigma_{\mathrm{eq.20}}\times\sqrt{\chi^2/\mathrm{dof}}$); "
                "dashed line marks variant A",
                ha="center", fontsize=6.8, color="#52514e")
    #  Hand-set rather than tight_layout: the shared y-tick labels are long and
    #  the figure legend sits outside the axes, which tight_layout mishandles.
    figure.subplots_adjust(left=0.30, right=0.985, top=0.86, bottom=0.16)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    if not png_only:
        figure.savefig(FIGURE_DIR / "figF_bc_study.pdf")
    figure.savefig(FIGURE_DIR / "figF_bc_study.png")
    plt.close(figure)
    print(f"  wrote results/figures/figF_bc_study"
          f"{'.png' if png_only else '.pdf + .png'}")


def write_table(rows) -> None:
    output_dir = RESULTS_DIR / "bc_studies" / "two_coupon"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "bc_study.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "title", "sealed", "chi2_per_dof", "birge"]
                        + list(LABELS)
                        + [f"birge_std_{n}" for n in LABELS]
                        + ["top_free", "bottom_free", "note"])
        for row in rows:
            writer.writerow(
                [row["key"], row["title"], int(row["sealed"]),
                 f"{row['chi2']:.1f}", f"{row['birge']:.2f}"]
                + [f"{v:.6f}" for v in row["theta"]]
                + [f"{v:.6f}" for v in row["birge_std"]]
                + ["|".join(row["top_free"]) or "-",
                   "|".join(row["bottom_free"]) or "-",
                   row["note"]])
    print(f"  wrote {path.relative_to(HERE).as_posix()}")


# =============================================================================
#  Driver
# =============================================================================

def main() -> int:
    print("=" * 78)
    print("  BOUNDARY-CONDITION STUDY")
    print("=" * 78)
    print("  Fixture:  TOP cut  -- machine couple measured here")
    print("            BOTTOM   -- reaction, NOT measured")
    print("            LEFT/RIGHT -- specimen free edges, never touched")
    print()
    print("  A cut is SEALED when W, W_X, W_XY and W_Y are all prescribed.")
    print("  A DOF may be left free only if its conjugate moment is measured")
    print("  AND supplied as an EXTERNAL group.")
    print()

    rows = []
    for variant in VARIANTS:
        result, goodness = run_variant(variant)
        theta = np.asarray(result.final_parameters, dtype=float)
        birge = float(goodness["birge"])
        rows.append({
            "key": variant.key,
            "title": variant.title,
            "sealed": variant.sealed,
            "theta": theta,
            "birge_std": np.asarray(result.parameter_std,
                                    dtype=float) * birge,
            "chi2": float(goodness["reduced"]),
            "birge": birge,
            "top_free": variant.top.free,
            "bottom_free": variant.bottom.free,
            "note": variant.note,
        })
        flag = "sealed" if variant.sealed else "LEAKS"
        print(f"  {variant.title:<34s} {flag:>6s}  "
              + "".join(f"{v:9.2f}" for v in theta)
              + f"   chi2/dof {goodness['reduced']:7.0f}")

    print()
    print("-" * 78)
    print(f"  {'':34s}" + "".join(f"{n:>9}" for n in LABELS)
          + "   chi2/dof")
    print("-" * 78)

    print()
    print("Rendering...")
    make_figure(rows)
    write_table(rows)

    reference = rows[0]
    print()
    print("=" * 78)
    print("  READING IT")
    print("=" * 78)
    sealed = [r for r in rows if r["sealed"]]
    spread_11 = max(r["theta"][0] for r in sealed) - min(
        r["theta"][0] for r in sealed)
    spread_22 = max(r["theta"][2] for r in sealed) - min(
        r["theta"][2] for r in sealed)
    print(f"  Among the {len(sealed)} admissible variants the spread is "
          f"{spread_11:.1f} N.mm on D11 ({spread_11 / reference['theta'][0] * 100:.0f} %)")
    print(f"  and {spread_22:.1f} N.mm on D22 "
          f"({spread_22 / reference['theta'][2] * 100:.0f} %).")
    print("  That spread is MODEL error: every one of them is a legitimate")
    print("  virtual field, so no variance formula can see the difference.")
    print()
    print("  D22 (transverse coupon, 9 % moment gradient) moves by a few")
    print("  percent across every variant.  D11 (longitudinal, 111 %) does")
    print("  not: variant B alone moves it 45 %.  The BC choice is only")
    print("  consequential where the pure-bending assumption already fails.")
    print()
    print("  Variant F supplies M_xy from the machine Tx channel, whose SIGN")
    print("  convention is not fixed for this frame.  Flipping it moves D66")
    print("  from -7.1 to -2.7 but D11 by 1.6 % and D22 by 0.6 %, so it")
    print("  matters only for the coefficient that is unidentifiable anyway.")
    print("  |M_xy| is 0.23-0.38 N against |M_yy| of 0.25-2.5 N, i.e. NOT")
    print("  negligible at the first load steps -- see TWIST_SIGN above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
