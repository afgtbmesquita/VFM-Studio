# =============================================================================
#  Generalized VFM / make_audit_figures.py
# -----------------------------------------------------------------------------
#  Figures for the assumption audit: which equations behind the FGLS derivation
#  actually hold on this campaign, and does each violation move the ANSWER or
#  only the ERROR BAR.
#
#      figA_assumption_scorecard   every assumption, its status, its effect
#      figB_uncertainty_ladder     eq. (20) against five wider estimates
#      figC_noise_model_fit        chi2 test: does Gamma explain the residuals?
#      figD_spatial_whiteness      the measured correlogram against eq. (2)
#      figE_D66_is_bias_not_noise  why D66's tight error bar is meaningless
#
#  Run audit_assumptions.py first (it writes results/audit/audit_results.npz).
# =============================================================================

from __future__ import annotations

import numpy as np
from scipy.stats import chi2 as chi2_dist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from workflows.campaign_common import (
    ANALYSIS_TYPE,
    HERE,
    INITIAL_PARAMETERS,
    MATERIAL_TYPE,
    RESULTS_DIR,
    build_campaign,
)
from workflows.make_figures import (
    DOUBLE_COLUMN,
    GRID_COLOUR,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SERIES_COLOUR,
    SERIES_MARKER,
    apply_style,
    chi_square_diagnostics,
    math_label,
    pad_limits,
    save,
    soft_grid,
)
from vfm_gls_identification import run_gls_identification_cases
from vfm_user_inputs import (
    GlsSettings,
    IdentificationSettings,
    LinearAlgebraSettings,
)

#  Status palette.  Reserved for state, never reused as a series colour, and
#  every chip carries its word as well as its colour.
STATUS = {
    "HOLDS": "#1baf7a",
    "HARMLESS": "#2a78d6",
    "PARTIAL": "#eda100",
    "VIOLATED": "#e8663c",
    "DOMINANT": "#c62f1e",
    "VACUOUS": "#8f8e88",
}


def reference_run(spatial_correction: bool = False):
    cases, ids, _ = build_campaign(verbose=False)
    result = run_gls_identification_cases(
        cases=cases, material_type=MATERIAL_TYPE, analysis_type=ANALYSIS_TYPE,
        initial_parameters=INITIAL_PARAMETERS,
        identification_settings=IdentificationSettings(
            max_iterations=30, convergence_tolerance=1.0e-9),
        linear_algebra_settings=LinearAlgebraSettings(), experiment_ids=ids,
        gls_settings=GlsSettings(spatial_correction=spatial_correction),
        verbose=False)
    return result


# =============================================================================
#  Figure A -- the scorecard
# =============================================================================

def figure_scorecard(data, gof, png_only) -> None:
    labels = [str(v) for v in data["labels"]]
    kappa = data["kappa"]
    rho = data["nearest_rho"]
    jack_ratio = data["jackknife_std"] / data["eq20"]
    tension = data["tension"]
    white_ratio = data["white_std"] / data["eq20"]

    rows = [
        ("§2, §3",
         "$b$ deterministic; noise perturbs $\\mathbf{H}$",
         "HARMLESS",
         "speciality pins $\\mathbf{H}=\\mathbf{I}$ to $10^{-15}$, so\n"
         "$\\delta\\mathbf{H}=0$ and ALL randomness is in $b$",
         "none", "none"),
        ("eq. (2)",
         "noise spatially uncorrelated, $\\delta_{pq}\\Gamma$",
         "VIOLATED",
         f"$\\rho={rho[0]:.2f}$ / {rho[1]:.2f} at one grid step;\n"
         f"$\\kappa={kappa[0]:.1f}$ / {kappa[1]:.1f} correlated points",
         "$+2\\%$ on $D_{22}$", f"$\\times{np.sqrt(kappa.mean()):.1f}$"),
        ("eq. (2)",
         "$\\Gamma$ homogeneous over the ROI",
         "PARTIAL",
         "correlation is anisotropic:\n"
         "$\\rho=0.73$ along $x$ vs $0.89$ along $y$",
         "not isolated", "in $\\kappa$"),
        ("§8",
         "$\\mathbf{M}\\mathbf{Y}\\approx0$: basis captures field",
         "PARTIAL",
         "$\\sigma$ tracks the load ($r=0.88$–$0.99$), so\n"
         "$\\Gamma$ is EFFECTIVE error, not DIC noise",
         "$\\leq1\\%$", "$\\times1$"),
        #  Status set from the measurement, not asserted: if the bootstrap
        #  under eq. (2)'s own noise model reproduces eq. (20), the first-order
        #  plug-in is adequate; if it runs wide, it is not.
        ("§14",
         "first-order plug-in; $\\mathbf{E}^*(\\theta)$ from data",
         "HOLDS" if white_ratio.max() < 1.6 else "PARTIAL",
         "bootstrap under eq. (2)'s own noise model\n"
         f"vs eq. (20), $\\kappa=1$: ratio "
         f"{white_ratio.min():.1f}–{white_ratio.max():.1f}",
         "none detected",
         f"$\\times{white_ratio.mean():.1f}$"),
        ("§19",
         "eq. (20) treats $\\hat\\Gamma$, $\\mathbf{C}$ as fixed",
         "PARTIAL",
         "the derivation says so itself;\n"
         f"jackknife/eq.(20) $={jack_ratio[0]:.1f}$, "
         f"{jack_ratio[1]:.1f}, {jack_ratio[2]:.1f}, {jack_ratio[3]:.1f}",
         "none", "$\\times2$–$6$"),
        ("§1",
         "rank$(\\mathbf{H}_{i,t})=p$ per load step",
         "VACUOUS",
         "$\\mathbf{H}=\\mathbf{I}$ BY CONSTRUCTION, so the rank\n"
         "test always passes and detects nothing",
         "hides $D_{12}$, $D_{66}$\nblindness", "none"),
        ("eq. (12)",
         "experiments independent",
         "VACUOUS",
         "untestable with two coupons, no repeats;\n"
         "a shared calibration error would not show",
         "unknown", "unknown"),
        ("§1",
         "ONE common $\\theta$ fits all experiments",
         "VIOLATED",
         f"the coupons disagree on $D_{{66}}$ by "
         f"{tension[3]:.1f}$\\sigma$,\non $D_{{22}}$ by "
         f"{tension[2]:.0f}$\\sigma$ of their own error bars",
         "dominant for\n$D_{66}$", "not\nrepresented"),
        ("implicit",
         "no model error: plate model exact",
         "DOMINANT",
         f"$\\chi^2/\\mathrm{{dof}}={gof['reduced']:.0f}$: the noise model\n"
         f"explains {100 / gof['reduced']:.1f}% of the residual variance",
         "this IS the\n$D_{66}$ error",
         f"$\\times{np.sqrt(gof['reduced']):.1f}$ min."),
    ]

    height = 1.15 + 0.60 * len(rows)
    figure = plt.figure(figsize=(DOUBLE_COLUMN, height))
    axis = figure.add_axes([0, 0, 1, 1])
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    columns = (0.010, 0.078, 0.352, 0.452, 0.775, 0.900)
    headers = ("eq.", "assumption", "status", "what was measured",
               "moves $\\hat\\theta$?", "moves $\\sigma$?")

    top = 1.0 - 0.52 / height
    row_height = 0.60 / height

    for x, header in zip(columns, headers):
        axis.text(x, top, header, fontsize=7.5, color=INK_PRIMARY,
                  fontweight="bold", va="bottom")
    axis.plot([0.008, 0.992], [top - 0.10 * row_height] * 2,
              color=INK_SECONDARY, linewidth=0.8)

    for index, (reference, assumption, status, measured, on_theta,
                on_sigma) in enumerate(rows):
        centre = top - (index + 0.72) * row_height
        if index % 2 == 0:
            axis.add_patch(plt.Rectangle(
                (0.008, centre - 0.42 * row_height), 0.984, 0.84 * row_height,
                facecolor="#f6f5f1", edgecolor="none", zorder=0))

        axis.text(columns[0], centre, reference, fontsize=6.8,
                  color=INK_SECONDARY, va="center", family="monospace")
        axis.text(columns[1], centre, assumption, fontsize=7.0,
                  color=INK_PRIMARY, va="center")

        colour = STATUS[status]
        axis.add_patch(FancyBboxPatch(
            (columns[2], centre - 0.19 * row_height), 0.082,
            0.38 * row_height, boxstyle="round,pad=0.002,rounding_size=0.004",
            facecolor=colour, edgecolor="none", zorder=2))
        axis.text(columns[2] + 0.041, centre, status, fontsize=6.1,
                  color="white", ha="center", va="center", zorder=3,
                  fontweight="bold")

        axis.text(columns[3], centre, measured, fontsize=6.6,
                  color=INK_SECONDARY, va="center", linespacing=1.5)
        axis.text(columns[4], centre, on_theta, fontsize=6.6,
                  color=INK_PRIMARY, va="center", linespacing=1.5)
        axis.text(columns[5], centre, on_sigma, fontsize=6.6,
                  color=INK_PRIMARY, va="center", linespacing=1.5)

    axis.text(0.008, top + 0.28 / height,
              "Assumptions behind eqs. (1)–(20), audited on the campaign data",
              fontsize=9.5, color=INK_PRIMARY, va="bottom")
    axis.text(0.008, 0.40 / height,
              "Read the last two columns. Only two rows move the ANSWER, and "
              "the one that moves it most — the bottom row — is the single "
              "assumption\nthe derivation never writes down. Everything above "
              "it changes the error bar but not the number.",
              fontsize=7, color=INK_MUTED, va="center", linespacing=1.6)
    save(figure, "figA_assumption_scorecard", png_only)


# =============================================================================
#  Figure B -- the uncertainty ladder
# =============================================================================

def figure_ladder(data, gof, png_only) -> None:
    labels = [str(v) for v in data["labels"]]
    theta = data["theta"]
    sources = [
        (r"eq. (20), eq. (3) verbatim  $\leftarrow$ reported", data["eq20"],
         STATUS["HOLDS"]),
        ("bootstrap, spatially white noise", data["white_std"],
         STATUS["HOLDS"]),
        ("bootstrap, measured spatial correlation", data["correlated_std"],
         STATUS["PARTIAL"]),
        ("basis degree 2–10 spread",
         data["by_degree"].std(axis=0, ddof=1), STATUS["PARTIAL"]),
        ("jackknife over load steps", data["jackknife_std"],
         STATUS["VIOLATED"]),
        ("eq. (20) $\\times\\sqrt{\\chi^2/\\mathrm{dof}}$  "
         "$\\leftarrow$ quote this", data["eq20"] * float(data["birge"]),
         STATUS["DOMINANT"]),
    ]
    #  kappa is NOT a bar here.  It is an approximation to eq. (1) that the
    #  solver does not use, and giving it a bar alongside measured quantities
    #  would imply it is one.  Its size is stated in the caption instead.
    #  |longitudinal - transverse| is also deliberately absent: for D11 and D22
    #  that gap is non-identifiability, not model error.  See figure E.

    figure, axes = plt.subplots(1, 4, figsize=(DOUBLE_COLUMN, 3.0),
                               constrained_layout=True)
    positions = np.arange(len(sources))[::-1]

    for index, axis in enumerate(axes):
        values = np.array([s[1][index] for s in sources])
        colours = [s[2] for s in sources]
        axis.barh(positions, values, color=colours, height=0.66, zorder=3)
        for position, value in zip(positions, values):
            axis.annotate(f"{value:.2f}", xy=(value, position),
                          xytext=(3, 0), textcoords="offset points",
                          va="center", fontsize=6.3, color=INK_SECONDARY)
        axis.set_xscale("log")
        axis.set_yticks(positions)
        axis.set_yticklabels([s[0] for s in sources] if index == 0 else [],
                             fontsize=6.6)
        axis.set_title(f"{math_label(labels[index])} $= {theta[index]:.2f}$",
                       fontsize=8, pad=4,
                       color=SERIES_COLOUR[index])
        axis.set_xlabel("$\\sigma$  (N$\\cdot$mm)", fontsize=7.5)
        axis.grid(True, axis="x", zorder=0)
        axis.set_axisbelow(True)
        axis.set_xlim(values.min() * 0.35, values.max() * 6.0)
        axis.tick_params(labelsize=6.6)

    figure.suptitle(
        "How large the error bar becomes as each violated assumption is "
        "accounted for", fontsize=9, y=1.04)
    figure.text(0.5, -0.075,
                "Rows 1–2 compare eq. (20) against a full bootstrap under "
                "eq. (2)'s OWN noise model: the formula is right to a factor "
                "of about two, so the shortfall is not in the algebra.\n"
                "No $\\kappa$ bar appears: $\\kappa$ approximates eq. (1), the "
                "solver does not use it, and placing it beside measured "
                "quantities would imply it is one. Applying it would move "
                "row 1 to\n0.40 on $D_{11}$ — still far below every empirical "
                "row, which is the point.",
                ha="center", fontsize=7, color=INK_MUTED, linespacing=1.5)
    save(figure, "figB_uncertainty_ladder", png_only)


# =============================================================================
#  Figure C -- does the noise model explain the residuals?
# =============================================================================

def figure_noise_model_fit(data, gof, png_only) -> None:
    labels = [str(v) for v in data["labels"]]
    figure, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN, 2.5),
                               constrained_layout=True,
                               gridspec_kw={"width_ratios": [1.0, 1.0, 1.25]})

    # ---- (a) chi2/dof per parameter --------------------------------------
    reduced = gof["per_parameter"]
    positions = np.arange(len(labels))
    colours = [STATUS["DOMINANT"] if v > 3 else
               (STATUS["PARTIAL"] if v > 1.5 else STATUS["HOLDS"])
               for v in reduced]
    axes[0].bar(positions, reduced, width=0.62, color=colours, zorder=3)
    axes[0].axhline(1.0, color=INK_PRIMARY, linewidth=0.9,
                    linestyle=(0, (4, 3)), zorder=4)
    for position, value in zip(positions, reduced):
        axes[0].annotate(f"{value:.1f}", xy=(position, value),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", fontsize=6.8, color=INK_SECONDARY)
    axes[0].set_yscale("log")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels([math_label(l) for l in labels])
    axes[0].set_ylabel("$\\chi^2/\\mathrm{dof}$")
    axes[0].set_title("(a) By coefficient", pad=4)
    axes[0].annotate("model fits", xy=(len(labels) - 0.5, 1.0),
                     xytext=(0, -10), textcoords="offset points",
                     ha="right", fontsize=6.5, color=INK_PRIMARY)
    soft_grid(axes[0])

    # ---- (b) what the spatial correction buys, and what survives it ------
    stages = [("eq. (3)\nverbatim\n(used)", gof["reduced"],
               STATUS["DOMINANT"]),
              ("with $\\kappa$\n(not used)",
               gof["reduced_with_kappa"], STATUS["VACUOUS"]),
              ("a correct\nnoise\nmodel", 1.0, STATUS["HOLDS"])]
    positions = np.arange(len(stages))
    axes[1].bar(positions, [s[1] for s in stages], width=0.6,
                color=[s[2] for s in stages], zorder=3)
    axes[1].axhline(1.0, color=INK_PRIMARY, linewidth=0.9,
                    linestyle=(0, (4, 3)), zorder=4)
    for position, (_, value, _) in zip(positions, stages):
        axes[1].annotate(f"{value:.0f}" if value >= 10 else f"{value:.0f}",
                         xy=(position, value), xytext=(0, 3),
                         textcoords="offset points", ha="center",
                         fontsize=6.8, color=INK_SECONDARY)
    axes[1].set_yscale("log")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([s[0] for s in stages], fontsize=6.5)
    axes[1].set_ylabel("$\\chi^2/\\mathrm{dof}$")
    axes[1].set_title("(b) Why $\\kappa$ is not the answer", pad=4)
    #  Upper right: the tallest bar is on the left and the shortest on the
    #  right, so this corner is the only clear one.
    axes[1].annotate(
        f"$\\kappa$ would remove a factor\n"
        f"of {gof['reduced'] / gof['reduced_with_kappa']:.0f}, leaving "
        f"{gof['reduced_with_kappa']:.0f} — still a\nrejected model. Measured,\n"
        "reported, NOT applied.",
        xy=(0.98, 0.985), xycoords="axes fraction", ha="right", va="top",
        fontsize=6.2, color=INK_SECONDARY, linespacing=1.5)
    axes[1].set_ylim(top=gof["reduced"] * 25.0)
    soft_grid(axes[1])

    # ---- (c) whitened residuals against N(0,1) ---------------------------
    whitened = gof["whitened"]
    edges = np.linspace(-max(4.0, np.abs(whitened).max() * 1.05),
                        max(4.0, np.abs(whitened).max() * 1.05), 34)
    axes[2].hist(whitened, bins=edges, density=True, color=SERIES_COLOUR[0],
                 alpha=0.55, zorder=3, label="whitened residuals")
    span = np.linspace(edges[0], edges[-1], 400)
    axes[2].plot(span, np.exp(-0.5 * span ** 2) / np.sqrt(2 * np.pi),
                 color=INK_PRIMARY, linewidth=1.3, zorder=4,
                 label="$\\mathcal{N}(0,1)$, what the model predicts")
    axes[2].set_xlabel("$(\\mathbf{L}^{-1}\\mathbf{r})_i$")
    axes[2].set_ylabel("density")
    axes[2].set_title("(c) Are the residuals the size the model says?", pad=4)
    axes[2].legend(fontsize=6.5, loc="upper left")
    axes[2].annotate(f"observed s.d. = {whitened.std(ddof=1):.1f}\n"
                     "predicted = 1.0", xy=(0.97, 0.62),
                     xycoords="axes fraction", ha="right", fontsize=6.8,
                     color=STATUS["DOMINANT"])
    soft_grid(axes[2])

    figure.suptitle(
        f"Goodness of fit of the noise model: $\\chi^2/\\mathrm{{dof}} = "
        f"{gof['reduced']:.0f}$ against an expected 1",
        fontsize=9, y=1.06)
    save(figure, "figC_noise_model_fit", png_only)


# =============================================================================
#  Figure D -- the spatial-whiteness assumption
# =============================================================================

def wide_correlogram(max_lag: float = 3.0, n_bins: int = 30):
    """rho(r) over a range wide enough to SEE the decay.

    vfm_noise_covariance measures correlation by neighbour rank, which on this
    dense cloud only reaches 0.31 mm -- fine for estimating kappa, useless as a
    picture.  Here the pairs are binned by true distance instead.
    """
    from scipy.spatial import cKDTree
    from vfm_noise_covariance import _orthonormal_basis, _standardise

    cases, ids, _ = build_campaign(verbose=False)
    out = []
    for experiment in (1, 2):
        group = [c for c, e in zip(cases, ids) if e == experiment]
        points = group[0].measurements.points
        basis = _orthonormal_basis(points, 6)
        stack = np.stack([c.measurements.strains.T for c in group])
        residual = stack - np.einsum("ij,tjc->tic", basis @ basis.T, stack)
        standard = _standardise(residual)

        tree = cKDTree(points)
        pairs = tree.query_pairs(max_lag, output_type="ndarray")
        lag = np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1)
        product = np.einsum("tpc,tpc->p",
                            standard[:, pairs[:, 0], :],
                            standard[:, pairs[:, 1], :])
        product /= (residual.shape[0] * residual.shape[2])

        edges = np.linspace(0.0, max_lag, n_bins + 1)
        which = np.clip(np.digitize(lag, edges) - 1, 0, n_bins - 1)
        counts = np.bincount(which, minlength=n_bins)
        keep = counts > 50
        centres = np.bincount(which, weights=lag,
                              minlength=n_bins)[keep] / counts[keep]
        values = np.bincount(which, weights=product,
                             minlength=n_bins)[keep] / counts[keep]
        out.append((centres, values))
    return out


def figure_spatial(data, png_only) -> None:
    kappa = data["kappa"]
    names = ["longitudinal", "transverse"]
    correlograms = wide_correlogram()

    figure, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, 2.5),
                               constrained_layout=True)

    for index, name in enumerate(names):
        centres, values = correlograms[index]
        axes[0].plot(centres, values, color=SERIES_COLOUR[index],
                     marker=SERIES_MARKER[index], markersize=3.2,
                     markeredgecolor="white", markeredgewidth=0.4,
                     label=name, zorder=3)
    #  What eq. (2) actually asserts, drawn to the same axes: unity at zero
    #  separation and identically zero everywhere else.
    axes[0].plot([0.0, 0.0], [0.0, 1.0], color=INK_PRIMARY, linewidth=1.8,
                 zorder=5)
    axes[0].plot([0.0], [1.0], marker="o", color=INK_PRIMARY, markersize=4.5,
                 zorder=6)
    axes[0].axhline(0.0, color=INK_PRIMARY, linewidth=1.2, zorder=4)
    axes[0].annotate("what eq. (2) assumes:\n$\\rho=\\delta_{pq}$ — unity at "
                     "zero,\nzero everywhere else",
                     xy=(0.30, 0.52), xycoords="axes fraction", fontsize=6.8,
                     color=INK_PRIMARY, va="top", linespacing=1.5)
    axes[0].legend(fontsize=7, loc="upper right", title="what is measured",
                   title_fontsize=7)
    axes[0].set_xlabel("separation between measurement points (mm)")
    axes[0].set_ylabel("residual correlation $\\rho$")
    axes[0].set_title("(a) The measured correlogram", pad=4)
    axes[0].set_ylim(-0.08, 1.06)
    axes[0].set_xlim(-0.06, 3.35)
    soft_grid(axes[0])

    positions = np.arange(2)
    axes[1].bar(positions, kappa, width=0.5, color=STATUS["VIOLATED"],
                zorder=3)
    axes[1].axhline(1.0, color=INK_PRIMARY, linewidth=0.9,
                    linestyle=(0, (4, 3)), zorder=4)
    for position, value in zip(positions, kappa):
        axes[1].annotate(f"{value:.1f}", xy=(position, value),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", fontsize=7, color=INK_SECONDARY)
    axes[1].annotate("$\\kappa=1$ is the spatially white value",
                     xy=(1.45, 1.0), xytext=(0, 6),
                     textcoords="offset points", ha="right", fontsize=6.8,
                     color=INK_PRIMARY)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(names, fontsize=7)
    axes[1].set_ylabel("$\\kappa=\\overline{\\sum_q \\rho_{pq}}$")
    axes[1].set_title("(b) Effectively correlated points", pad=4)
    axes[1].set_ylim(0, kappa.max() * 1.25)
    soft_grid(axes[1])

    figure.suptitle("Assumption eq. (2): spatially uncorrelated measurement "
                    "noise", fontsize=9, y=1.05)
    figure.text(0.5, -0.05,
                "Each DIC point is worth far less than one independent sample. "
                "This inflates $\\sigma$ by $\\sqrt{\\kappa}$, but a factor "
                "common to both experiments cancels from $\\hat\\theta$ "
                "exactly, because every $\\mathbf{H}_i$ is the identity.",
                ha="center", fontsize=7, color=INK_MUTED)
    save(figure, "figD_spatial_whiteness", png_only)


# =============================================================================
#  Figure E -- D66
# =============================================================================

def figure_d66(data, gof, png_only) -> None:
    labels = [str(v) for v in data["labels"]]
    theta = data["theta"]
    eq20 = data["eq20"]
    jack = data["jackknife_std"]
    long_theta, long_std = data["long_theta"], data["long_std"]
    trans_theta, trans_std = data["trans_theta"], data["trans_std"]
    tension = data["tension"]
    index = labels.index("D66")

    figure, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, 2.7),
                               constrained_layout=True,
                               gridspec_kw={"width_ratios": [1.15, 1.0]})

    # ---- (a) the three D66 answers, and the feasible set -----------------
    entries = [
        ("longitudinal alone", long_theta[index], long_std[index],
         SERIES_COLOUR[0]),
        ("transverse alone", trans_theta[index], trans_std[index],
         SERIES_COLOUR[1]),
        ("FGLS pooled, eq. (20)", theta[index], eq20[index], INK_PRIMARY),
        ("FGLS pooled, jackknife", theta[index], jack[index], INK_PRIMARY),
    ]
    positions = np.arange(len(entries))[::-1]
    axes[0].axvspan(0.0, 60.0, color=STATUS["HOLDS"], alpha=0.10, zorder=0)
    axes[0].axvline(0.0, color=STATUS["HOLDS"], linewidth=1.2, zorder=2)
    axes[0].annotate("physically admissible: $D_{66}>0$",
                     xy=(0.6, positions.max() + 0.42), fontsize=6.8,
                     color="#12805a", ha="left", va="center")
    for position, (name, value, sigma, colour) in zip(positions, entries):
        axes[0].errorbar(value, position, xerr=sigma, fmt="o", color=colour,
                         markersize=4.5, capsize=2.5, linewidth=1.3,
                         markeredgecolor="white", markeredgewidth=0.6,
                         zorder=4)
        axes[0].annotate(f"{value:.2f} $\\pm$ {sigma:.2f}",
                         xy=(value, position), xytext=(0, 7),
                         textcoords="offset points", ha="center",
                         fontsize=6.6, color=colour)
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels([e[0] for e in entries], fontsize=7)
    axes[0].set_xlabel("$D_{66}$  (N$\\cdot$mm)")
    axes[0].set_title("(a) Every route says $D_{66}<0$, confidently", pad=4)
    axes[0].set_xlim(-14.5, 6.0)
    axes[0].set_ylim(-0.6, positions.max() + 0.75)
    axes[0].grid(True, axis="x", zorder=1)
    axes[0].set_axisbelow(True)

    distance = abs(theta[index]) / jack[index]
    axes[0].annotate(
        f"{distance:.0f}$\\sigma$ outside the feasible set",
        xy=(theta[index], positions[3]), xytext=(-8, -14),
        textcoords="offset points", ha="right", fontsize=6.8,
        color=STATUS["DOMINANT"])

    # ---- (b) tension between the two experiments -------------------------
    positions = np.arange(len(labels))
    colours = [STATUS["DOMINANT"] if v > 3 else STATUS["HOLDS"]
               for v in tension]
    axes[1].bar(positions, np.maximum(tension, 1e-3), width=0.62,
                color=colours, zorder=3)
    axes[1].axhline(3.0, color=INK_PRIMARY, linewidth=0.9,
                    linestyle=(0, (4, 3)), zorder=4)
    for position, value in zip(positions, tension):
        axes[1].annotate(f"{value:.0f}" if value >= 1 else f"{value:.2f}",
                         xy=(position, max(value, 1e-3)),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", fontsize=6.8, color=INK_SECONDARY)
    axes[1].set_yscale("log")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([math_label(l) for l in labels])
    axes[1].set_ylabel("$|\\theta_L-\\theta_T|\\,/\\,\\sqrt{\\sigma_L^2+"
                       "\\sigma_T^2}$")
    axes[1].set_title("(b) Do the stated error bars survive\ncontact with "
                      "the other coupon?", pad=4, fontsize=7.5)
    axes[1].annotate("$3\\sigma$", xy=(len(labels) - 0.45, 3.0),
                     xytext=(0, 3), textcoords="offset points", ha="right",
                     fontsize=6.8, color=INK_PRIMARY)
    soft_grid(axes[1])

    figure.suptitle("$D_{66}$: the error bar is small and the answer is still "
                    "wrong", fontsize=9, y=1.05)
    figure.text(0.5, -0.075,
                "A variance is a statement about repeatability, not about "
                "truth. In (b) every coefficient fails by a wide margin, but "
                "for two different reasons: $D_{11}$ and $D_{22}$ because one "
                "coupon\ncannot see them yet still quotes a tight $\\sigma$, "
                "$D_{66}$ because neither coupon can see it and the two "
                "nonetheless disagree by $7.1$ N$\\cdot$mm — larger than "
                "$|D_{66}|$ itself.",
                ha="center", fontsize=7, color=INK_MUTED, linespacing=1.5)
    save(figure, "figE_D66_is_bias_not_noise", png_only)


def main() -> None:
    apply_style()
    path = RESULTS_DIR / "audit" / "audit_results.npz"
    if not path.exists():
        raise SystemExit("Run audit_assumptions.py first.")
    data = dict(np.load(path, allow_pickle=False))

    print("Recomputing the goodness-of-fit diagnostics...")
    #  Primary = eq. (3) verbatim, which is what the solver reports.  The
    #  kappa variant is kept only to show how much of the misfit spatial
    #  correlation could account for -- it is never used for a reported number.
    gof = chi_square_diagnostics(reference_run(spatial_correction=False))
    gof["reduced_with_kappa"] = chi_square_diagnostics(
        reference_run(spatial_correction=True))["reduced"]
    p_value = float(chi2_dist.sf(gof["chi2"], gof["dof"]))
    print(f"  chi2/dof = {gof['reduced']:.0f}  (p = {p_value:.2e}) with eq. (3)"
          f" verbatim; the kappa variant would give "
          f"{gof['reduced_with_kappa']:.0f}")

    print("Rendering audit figures...")
    figure_scorecard(data, gof, False)
    figure_ladder(data, gof, False)
    figure_noise_model_fit(data, gof, False)
    figure_spatial(data, False)
    figure_d66(data, gof, False)


if __name__ == "__main__":
    main()
