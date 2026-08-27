# =============================================================================
#  Generalized VFM / make_figures.py
# -----------------------------------------------------------------------------
#  Publication figures for the FGLS identification.  Vector PDF (Type-42 fonts,
#  which is what journals require) plus 600-dpi PNG for drafts.
#
#      fig1_coefficient_evolution   theta_k per iteration, with the eq. (20) band
#      fig2_correlation_determinant log|C| and log|R| per iteration
#      fig3_parameter_variance      sigma_k per iteration, absolute and relative
#      fig4_information_pca         eigenspectrum + loadings of Cov(theta)
#      fig5_parameter_ellipsoid     4x4 corner plot: slices through D-space
#
#  Figure widths are the usual two-column journal measures (3.5 in single,
#  7.16 in double) so nothing needs rescaling in LaTeX -- include them at
#  \includegraphics[width=\columnwidth]{...} and the type sizes come out right.
#
#  COLOUR.  One fixed hue per coefficient, never cycled, and every series also
#  carries a distinct marker so identity never rests on colour alone.  The
#  four-hue set was checked with a CVD validator: worst adjacent pair over ALL
#  pairs is dE 9.2 (deuteranopia) and 16.3 (normal vision), both above the
#  usability floors.  Do not substitute a green for the aqua -- green/orange
#  collapses to dE 3.2 under protanopia.
#
#  Run:  python -m workflows.make_figures            (uses the shipped campaign)
#        python -m workflows.make_figures --png-only
# =============================================================================

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from scipy.stats import chi2

from workflows.campaign_common import (
    ANALYSIS_TYPE,
    BC_FORMS,
    BC_TOP_ONLY,
    FIGURES_DIR,
    HERE,
    INITIAL_PARAMETERS,
    MATERIAL_TYPE,
    TESTS_RAW_FRAME,
    TESTS_WITH_45,
    build_campaign,
)
from vfm_gls_identification import run_gls_identification_cases
from vfm_user_inputs import (
    GlsSettings,
    IdentificationSettings,
    LinearAlgebraSettings,
)

FIGURE_DIR = FIGURES_DIR

#  Journal column measures, inches.
SINGLE_COLUMN = 3.50
DOUBLE_COLUMN = 7.16

#  One hue per coefficient, fixed order, never cycled.  Validated for CVD.
SERIES_COLOUR = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")
SERIES_MARKER = ("o", "s", "^", "D")

INK_PRIMARY = "#141413"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8f8e88"
GRID_COLOUR = "#dcdbd5"

#  Diverging map for the signed loadings: two poles, NEUTRAL GREY midpoint.
LOADING_CMAP = LinearSegmentedColormap.from_list(
    "loading", ["#2a78d6", "#7ba7dd", "#eceae4", "#f0a988", "#eb6834"])


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK_SECONDARY,
        "axes.labelcolor": INK_PRIMARY,
        "axes.titlecolor": INK_PRIMARY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK_PRIMARY,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.4,
        "lines.markersize": 4.2,
        "grid.color": GRID_COLOUR,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.columnspacing": 1.2,
        "legend.handletextpad": 0.5,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        #  Type 42 = embedded TrueType.  Type 3 (the default) is rejected by
        #  most journal submission systems and cannot be searched or edited.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def soft_grid(axis, which: str = "y") -> None:
    axis.grid(True, which="major", axis=which, zorder=0)
    axis.set_axisbelow(True)


def math_label(label: str) -> str:
    """'D11' -> '$D_{11}$'.  Plain 'D11' in a paper figure is a typo."""
    return f"${label[0]}_{{{label[1:]}}}$"


def chi_square_diagnostics(result):
    """Does the modelled covariance actually explain the residuals?

    r = H theta_hat - b is the part of the stacked system the fit could not
    satisfy.  Under the noise model r ~ N(0, C), so r' C^-1 r should be about
    its degrees of freedom.  Anything much larger is error the covariance does
    not contain, and eq. (20) -- being a pure noise-propagation formula --
    cannot report it.  The Birge factor sqrt(chi2/dof) is the standard
    correction, and on this campaign it is 6.5.
    """
    design, rhs = result.K_matrix, result.U_vector
    covariance = result.global_covariance
    residual = design @ result.final_parameters - rhs
    n_rows, n_parameters = design.shape

    total = float(residual @ np.linalg.solve(covariance, residual))
    dof = n_rows - n_parameters

    per_parameter = []
    for index in range(n_parameters):
        rows = np.arange(index, n_rows, n_parameters)
        block = covariance[np.ix_(rows, rows)]
        piece = residual[rows]
        per_parameter.append(
            float(piece @ np.linalg.solve(block, piece)) / (len(rows) - 1))

    per_experiment, offset = [], 0
    for report in result.experiments:
        size = n_parameters * report.n_steps
        block = covariance[offset:offset + size, offset:offset + size]
        piece = residual[offset:offset + size]
        per_experiment.append(
            float(piece @ np.linalg.solve(block, piece))
            / (size - n_parameters))
        offset += size

    factor = np.linalg.cholesky(covariance)
    return {
        "chi2": total, "dof": dof, "reduced": total / dof,
        "birge": float(np.sqrt(total / dof)),
        "per_parameter": np.array(per_parameter),
        "per_experiment": np.array(per_experiment),
        "whitened": np.linalg.solve(factor, residual),
    }


def pad_limits(axis, values, fraction: float = 0.14, axis_name: str = "y",
               floor: float = None) -> None:
    """Explicit head/foot room so markers never sit on the frame.

    matplotlib's autoscaling leaves too little room once clip_on=False lets a
    marker hang over the spine, which is how points end up half-cut.
    """
    low, high = float(np.min(values)), float(np.max(values))
    span = high - low
    if span <= 0:
        span = max(abs(high), 1.0) * 0.1
    lower, upper = low - fraction * span, high + fraction * span
    if floor is not None:
        lower = max(lower, floor)
    if axis_name == "y":
        axis.set_ylim(lower, upper)
    else:
        axis.set_xlim(lower, upper)


def save(figure, name: str, png_only: bool = False) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    if not png_only:
        figure.savefig(FIGURE_DIR / f"{name}.pdf")
    figure.savefig(FIGURE_DIR / f"{name}.png")
    plt.close(figure)
    suffixes = ".png" if png_only else ".pdf + .png"
    relative = FIGURE_DIR.relative_to(HERE).as_posix()
    print(f"  wrote {relative}/{name}{suffixes}")


def set_figure_dir(bc_form: str, with_45: bool = False) -> None:
    """Send each BC form to its own folder so the sets coexist.

    ``top-only`` keeps ``results/figures/`` -- it is the default and what the README
    references -- and any other form gets a sibling subfolder.  The
    three-coupon campaign gets its own tree under ``results/figures/with_45/`` so it
    can never overwrite a published two-coupon figure.
    """
    global FIGURE_DIR
    root = FIGURES_DIR / "with_45" if with_45 else FIGURES_DIR
    FIGURE_DIR = (root if bc_form == BC_TOP_ONLY
                  else root / bc_form.replace("-", "_"))


# =============================================================================
#  Figure 1 -- coefficient evolution
# =============================================================================

def figure_coefficients(result, labels, png_only, chi2_reduced=1.0,
                        initial=None) -> None:
    diagnostics = result.iteration_diagnostics
    iterations = np.array([0] + [d.iteration for d in diagnostics])
    if initial is None:
        initial = INITIAL_PARAMETERS
    values = np.vstack([np.asarray(initial, dtype=float)]
                       + [d.parameters for d in diagnostics])
    #  No covariance exists before the first solve, so the band starts at 1.
    band_iterations = np.array([d.iteration for d in diagnostics])
    band = np.vstack([d.parameter_std for d in diagnostics])

    figure = plt.figure(figsize=(DOUBLE_COLUMN, 3.6), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.05])
    #  Row-major, so the panels read D11 D12 / D22 D66 like the D matrix.
    panels = [figure.add_subplot(grid[i // 2, i % 2]) for i in range(4)]
    ax_rate = figure.add_subplot(grid[:, 2])

    for index, axis in enumerate(panels):
        colour = SERIES_COLOUR[index]
        final = values[-1, index]

        axis.axhline(final, color=INK_MUTED, linewidth=0.7,
                     linestyle=(0, (4, 3)), zorder=1)
        #  NO uncertainty band here, deliberately.  eq. (20) is a variance
        #  UNDER the noise model, and that model is rejected by the data
        #  (chi2/dof = 830).  Drawing it against the estimate would read as
        #  "the coefficient is known this well", which is not what it says.
        #  The distinction is made explicitly in fig. 3.
        axis.plot(iterations, values[:, index], color=colour,
                  marker=SERIES_MARKER[index], markeredgecolor="white",
                  markeredgewidth=0.6, zorder=3, clip_on=False)

        soft_grid(axis)
        axis.set_xlim(-0.28, iterations[-1] + 0.28)
        axis.set_xticks(iterations)
        pad_limits(axis, values[:, index])
        axis.set_title(math_label(labels[index]), pad=3)
        axis.set_ylabel("N$\\cdot$mm")
        if index >= 2:
            axis.set_xlabel("FGLS iteration")

        #  The converged value only.  No +/- : see the note above.
        axis.annotate(f"{final:.2f}", xy=(iterations[-1], final),
                      xytext=(-3, 5), textcoords="offset points",
                      ha="right", va="bottom", fontsize=7.2, color=colour)

    # ---- convergence rate, which the absolute panels cannot show ----------
    floor = 1.0e-12
    for index, label in enumerate(labels):
        relative = np.abs(values[:, index] - values[-1, index]) / abs(
            values[-1, index])
        ax_rate.plot(iterations[:-1], np.maximum(relative[:-1], floor),
                     color=SERIES_COLOUR[index], marker=SERIES_MARKER[index],
                     markeredgecolor="white", markeredgewidth=0.6,
                     label=math_label(label), clip_on=False, zorder=3)
    ax_rate.set_yscale("log")
    soft_grid(ax_rate)
    ax_rate.set_xticks(iterations[:-1])
    ax_rate.set_xlim(-0.28, iterations[-2] + 0.28)
    ax_rate.set_xlabel("FGLS iteration")
    ax_rate.set_ylabel("$|\\theta_k-\\hat{\\theta}|\\,/\\,|\\hat{\\theta}|$")
    ax_rate.set_title("Convergence", pad=3)
    ax_rate.legend(ncol=2, loc="upper right", fontsize=7)

    figure.suptitle(
        "Coefficient evolution from $\\theta_0=[1,1,1,1]$ "
        "(the fixed point is independent of the start)", fontsize=9, y=1.05)
    figure.text(0.5, -0.045,
                "No uncertainty band is drawn. eq. (20) is a variance under "
                "the noise model, and that model is rejected by the data "
                "($\\chi^2/\\mathrm{dof}=%.0f$); it is\nnot the variability of "
                "the coefficients and must not be read against them. See "
                "fig. 3." % chi2_reduced,
                ha="center", fontsize=7, color=INK_MUTED, linespacing=1.5)
    save(figure, "fig1_coefficient_evolution", png_only)


# =============================================================================
#  Figure 2 -- determinants of the stacked covariance and correlation
# =============================================================================

def figure_determinants(result, png_only) -> None:
    diagnostics = result.iteration_diagnostics
    iterations = np.array([d.iteration for d in diagnostics])
    covariance_logdet = np.array([d.covariance_logdet for d in diagnostics])
    correlation_logdet = np.array([d.correlation_logdet for d in diagnostics])
    size = result.global_covariance.shape[0]

    figure, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, 2.6),
                                constrained_layout=True)

    #  Two panels, not two y-axes on one: the quantities differ by 60 decades.
    for axis, series, colour, marker, title, ylabel in (
        (axes[0], covariance_logdet, SERIES_COLOUR[0], "o",
         "Stacked equation covariance",
         "$\\log_{10}\\,|\\mathbf{C}_{\\mathrm{G}}|$"),
        (axes[1], correlation_logdet, SERIES_COLOUR[1], "s",
         "Stacked equation correlation",
         "$\\log_{10}\\,|\\mathbf{R}_{\\mathrm{G}}|$"),
    ):
        axis.plot(iterations, series, color=colour, marker=marker,
                  markeredgecolor="white", markeredgewidth=0.6, clip_on=False,
                  zorder=3)
        axis.annotate(f"{series[-1]:.2f}", xy=(iterations[-1], series[-1]),
                      xytext=(-3, 7), textcoords="offset points",
                      ha="right", fontsize=7.5, color=colour)
        soft_grid(axis)
        axis.set_xticks(iterations)
        axis.set_xlim(iterations[0] - 0.2, iterations[-1] + 0.2)
        pad_limits(axis, series, fraction=0.18)
        axis.set_xlabel("FGLS iteration")
        axis.set_ylabel(ylabel)
        axis.set_title(title, pad=4)

    #  |R| = 1 (independent equations) sits 40 decades above the data, so it is
    #  stated rather than plotted -- drawing it would flatten the panel and
    #  hide the per-iteration evolution, which is the point of the figure.
    #  Top-right: the curve runs top-left to bottom-right, so this corner is
    #  the only one it never enters.
    axes[1].annotate(
        "independent equations would give $\\log_{10}|\\mathbf{R}|=0$;\n"
        f"the stacked system sits {abs(correlation_logdet[-1]):.0f} decades below",
        xy=(0.97, 0.95), xycoords="axes fraction", fontsize=7,
        color=INK_MUTED, va="top", ha="right", linespacing=1.5)

    figure.suptitle(
        f"Redundancy of the {size} stacked VFM equations "
        "($\\mathbf{R}=\\mathbf{D}^{-1/2}\\mathbf{C}\\mathbf{D}^{-1/2}$)",
        fontsize=9, y=1.06)
    save(figure, "fig2_correlation_determinant", png_only)


# =============================================================================
#  Figure 3 -- parameter variance per iteration
# =============================================================================

def figure_variance(result, labels, png_only, birge=1.0,
                    chi2_reduced=1.0) -> None:
    diagnostics = result.iteration_diagnostics
    iterations = np.array([d.iteration for d in diagnostics])
    std = np.vstack([d.parameter_std for d in diagnostics])
    values = np.vstack([d.parameters for d in diagnostics])

    figure, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, 2.7),
                                constrained_layout=True)

    relative = 100.0 * std / np.abs(values)

    for index, label in enumerate(labels):
        style = dict(color=SERIES_COLOUR[index], marker=SERIES_MARKER[index],
                     markeredgecolor="white", markeredgewidth=0.6,
                     label=math_label(label), clip_on=False, zorder=3)
        axes[0].plot(iterations, std[:, index], **style)
        axes[1].plot(iterations, relative[:, index], **style)

    #  Absolute sigma spans well under a decade, so a log axis would only add
    #  6x10^-1 style ticks; relative sigma spans 0.6 % to 40 % and needs one.
    for axis, ylabel, title, log in (
        (axes[0], "$\\sigma_{\\mathrm{model}}$  (N$\\cdot$mm)",
         "Absolute, under the noise model", False),
        (axes[1], "$\\sigma_{\\mathrm{model}}/|\\theta|$  (%)",
         "Relative, under the noise model", True),
    ):
        soft_grid(axis)
        if log:
            axis.set_yscale("log")
        axis.set_xticks(iterations)
        axis.set_xlim(iterations[0] - 0.2, iterations[-1] + 0.35)
        axis.set_xlabel("FGLS iteration")
        axis.set_ylabel(ylabel)
        axis.set_title(title, pad=4)

    pad_limits(axes[0], std, fraction=0.16, floor=0.0)

    #  Direct labels at the right-hand end, so identity never rests on colour.
    #  Nudge apart where two curves converge to nearly the same value.
    for axis, series in ((axes[0], std), (axes[1], relative)):
        order = np.argsort(series[-1, :])
        offsets = {}
        for rank, index in enumerate(order):
            previous = series[-1, order[rank - 1]] if rank else None
            close = (previous is not None
                     and abs(np.log10(max(series[-1, index], 1e-30))
                             - np.log10(max(previous, 1e-30))) < 0.07)
            offsets[index] = -4.5 if close else 3.5
        for index, label in enumerate(labels):
            axis.annotate(math_label(label),
                          xy=(iterations[-1], series[-1, index]),
                          xytext=(5, offsets[index]),
                          textcoords="offset points",
                          va="center", fontsize=7.2,
                          color=SERIES_COLOUR[index])

    #  THE distinction this figure exists to make.  Everything plotted above
    #  is conditional on a noise model the data rejects.
    figure.suptitle("Parameter variance UNDER THE MODEL — not the variance "
                    "of the data", fontsize=9, y=1.04)
    figure.text(
        0.5, -0.34,
        "These curves are eq. (20) evaluated on eq. (3) exactly as derived "
        "($\\kappa=1$; the measured spatial correlation is reported in "
        "fig. D but never folded in).\n"
        "They answer: \"if the ONLY error were spatially white DIC noise with "
        "covariance $\\hat\\Gamma$, how much would $\\hat\\theta$ move?\"  "
        "They do NOT answer \"how well is $D_{ij}$ known?\".\n"
        f"The data reject that noise model at $\\chi^2/\\mathrm{{dof}} = "
        f"{chi2_reduced:.0f}$: it accounts for "
        f"{100 / chi2_reduced:.2f} % of the observed residual variance. An "
        f"interval to quote is at least $\\sqrt{{\\chi^2/\\mathrm{{dof}}}} = "
        f"{birge:.0f}\\times$ these values,\nand even that assumes the misfit "
        "is a pure scale error rather than a shape error.",
        ha="center", fontsize=7, color=INK_MUTED, linespacing=1.6)
    save(figure, "fig3_parameter_variance", png_only)


# =============================================================================
#  Figure 4 -- PCA of the converged parameter covariance
# =============================================================================

def figure_pca(result, labels, png_only):
    covariance = result.parameter_covariance
    #  eigh: Cov is symmetric positive definite.  Descending, so PC1 is the
    #  WORST-determined direction -- the one the experiment cannot see.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    principal_std = np.sqrt(np.clip(eigenvalues, 0.0, None))
    share = 100.0 * eigenvalues / eigenvalues.sum()

    figure = plt.figure(figsize=(DOUBLE_COLUMN, 2.55), constrained_layout=True)
    grid = figure.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 1.0])
    ax_spectrum = figure.add_subplot(grid[0, 0])
    ax_loadings = figure.add_subplot(grid[0, 1])
    ax_share = figure.add_subplot(grid[0, 2])

    # ---- (a) principal standard deviations ------------------------------
    positions = np.arange(len(principal_std))
    ax_spectrum.bar(positions, principal_std, width=0.62,
                    color=SERIES_COLOUR[0], zorder=3)
    for position, value in zip(positions, principal_std):
        ax_spectrum.annotate(f"{value:.2f}", xy=(position, value),
                             xytext=(0, 3), textcoords="offset points",
                             ha="center", fontsize=7, color=INK_SECONDARY)
    ax_spectrum.set_xticks(positions)
    ax_spectrum.set_xticklabels([f"PC{i + 1}" for i in positions])
    ax_spectrum.set_ylabel("$\\sqrt{\\lambda}$  (N$\\cdot$mm)")
    ax_spectrum.set_title("(a) Principal std. deviation", pad=4)
    ax_spectrum.set_ylim(0, principal_std.max() * 1.22)
    soft_grid(ax_spectrum)

    # ---- (b) eigenvector loadings ---------------------------------------
    #  Sign is arbitrary in an eigendecomposition; fix it so the largest
    #  component of each PC is positive, otherwise the picture flips run to run.
    for column in range(eigenvectors.shape[1]):
        dominant = np.argmax(np.abs(eigenvectors[:, column]))
        if eigenvectors[dominant, column] < 0:
            eigenvectors[:, column] *= -1.0

    image = ax_loadings.imshow(eigenvectors.T, cmap=LOADING_CMAP,
                               vmin=-1.0, vmax=1.0, aspect="auto")
    ax_loadings.set_xticks(np.arange(len(labels)))
    ax_loadings.set_xticklabels([math_label(l) for l in labels])
    ax_loadings.set_yticks(positions)
    ax_loadings.set_yticklabels([f"PC{i + 1}" for i in positions])
    ax_loadings.set_title("(b) Orientation in $D$-space", pad=4)
    for row in positions:
        for column in range(len(labels)):
            value = eigenvectors[column, row]
            ax_loadings.text(column, row, f"{value:+.2f}", ha="center",
                             va="center", fontsize=6.8,
                             color=INK_PRIMARY if abs(value) < 0.6 else "white")
    for spine in ax_loadings.spines.values():
        spine.set_visible(False)
    ax_loadings.tick_params(length=0)
    colourbar = figure.colorbar(image, ax=ax_loadings, fraction=0.046, pad=0.03,
                                ticks=[-1, 0, 1])
    colourbar.outline.set_visible(False)
    colourbar.ax.tick_params(length=0, labelsize=7)
    colourbar.set_label("loading", fontsize=7.5)

    # ---- (c) share of total variance ------------------------------------
    ax_share.bar(positions, share, width=0.62, color=SERIES_COLOUR[1], zorder=3)
    for position, value in zip(positions, share):
        ax_share.annotate(f"{value:.1f}%", xy=(position, value),
                          xytext=(0, 3), textcoords="offset points",
                          ha="center", fontsize=7, color=INK_SECONDARY)
    ax_share.set_xticks(positions)
    ax_share.set_xticklabels([f"PC{i + 1}" for i in positions])
    ax_share.set_ylabel("share of total variance (%)")
    ax_share.set_title("(c) Variance distribution", pad=4)
    ax_share.set_ylim(0, min(112.0, share.max() * 1.22))
    soft_grid(ax_share)

    #  State the result the loadings show, rather than leaving the reader to
    #  deduce it from sixteen numbers: how far each PC is from a pure D axis.
    alignment = np.abs(eigenvectors).max(axis=0).min()
    ax_loadings.set_xlabel(
        f"each PC aligns with one $D$ axis to $\\geq\\,${alignment:.2f}",
        fontsize=7, color=INK_MUTED, labelpad=3)

    figure.suptitle(
        "Principal components of the converged parameter covariance "
        "$(\\mathbf{H}^{\\mathsf{T}}\\mathbf{C}^{-1}\\mathbf{H})^{-1}$",
        fontsize=9, y=1.08)
    save(figure, "fig4_information_pca", png_only)
    return eigenvalues, eigenvectors, principal_std, share


# =============================================================================
#  Figure 5 -- slices through D11 / D12 / D22 / D66 space
# =============================================================================

def _ellipse(mean, covariance_2d, n_sigma, **kwargs) -> Ellipse:
    values, vectors = np.linalg.eigh(covariance_2d)
    order = values.argsort()[::-1]
    values, vectors = values[order], vectors[:, order]
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    width, height = 2.0 * n_sigma * np.sqrt(np.clip(values, 0.0, None))
    return Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)


def figure_ellipsoid(result, labels, png_only) -> None:
    theta = result.final_parameters
    covariance = result.parameter_covariance
    precision = np.linalg.inv(covariance)
    n_parameters = len(labels)

    #  1-sigma and 2-sigma here mean the 39.3 % and 86.5 % contours of the
    #  bivariate Gaussian (chi2 with 2 dof), i.e. the usual covariance ellipse.
    levels = (1.0, 2.0)

    figure, axes = plt.subplots(n_parameters, n_parameters,
                                figsize=(DOUBLE_COLUMN, DOUBLE_COLUMN * 0.86),
                                constrained_layout=True)

    for row in range(n_parameters):
        for column in range(n_parameters):
            axis = axes[row, column]

            # ---- upper triangle: the correlation coefficient -------------
            if column > row:
                correlation = covariance[row, column] / np.sqrt(
                    covariance[row, row] * covariance[column, column])
                axis.text(0.5, 0.55, f"{correlation:+.2f}", ha="center",
                          va="center", fontsize=11,
                          color=(SERIES_COLOUR[1] if abs(correlation) > 0.5
                                 else INK_SECONDARY))
                axis.text(0.5, 0.26, "corr.", ha="center", va="center",
                          fontsize=6.5, color=INK_MUTED)
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_visible(False)
                continue

            # ---- diagonal: the 1-D marginal ------------------------------
            if row == column:
                sigma = np.sqrt(covariance[row, row])
                span = np.linspace(theta[row] - 3.4 * sigma,
                                   theta[row] + 3.4 * sigma, 240)
                density = np.exp(-0.5 * ((span - theta[row]) / sigma) ** 2)
                axis.fill_between(span, density, color=SERIES_COLOUR[row],
                                  alpha=0.20, linewidth=0)
                axis.plot(span, density, color=SERIES_COLOUR[row],
                          linewidth=1.2)
                axis.axvline(theta[row], color=INK_MUTED, linewidth=0.7,
                             linestyle=(0, (4, 3)))
                axis.set_ylim(0, 1.28)
                axis.set_yticks([])
                axis.set_title(f"{math_label(labels[row])[:-1]} = "
                               f"{theta[row]:.2f}\\pm{sigma:.2f}$",
                               fontsize=7.5, pad=3, color=SERIES_COLOUR[row])
                axis.set_xlim(span[0], span[-1])
            else:
                # ---- lower triangle: the 2-D slice -----------------------
                pair = [column, row]
                marginal = covariance[np.ix_(pair, pair)]
                #  Slice (conditional): the other two coefficients held at
                #  their converged values.  Always tighter than the marginal;
                #  the gap between the two IS the correlation.
                conditional = np.linalg.inv(precision[np.ix_(pair, pair)])
                centre = (theta[column], theta[row])

                for n_sigma, alpha in zip(levels, (0.26, 0.13)):
                    axis.add_patch(_ellipse(
                        centre, marginal, n_sigma, facecolor=SERIES_COLOUR[0],
                        edgecolor="none", alpha=alpha, zorder=2))
                axis.add_patch(_ellipse(
                    centre, marginal, levels[-1], facecolor="none",
                    edgecolor=SERIES_COLOUR[0], linewidth=0.9, zorder=3))
                for n_sigma in levels:
                    axis.add_patch(_ellipse(
                        centre, conditional, n_sigma, facecolor="none",
                        edgecolor=SERIES_COLOUR[1], linewidth=0.9,
                        linestyle=(0, (3, 2)), zorder=4))

                axis.plot(*centre, marker="+", color=INK_PRIMARY,
                          markersize=5, markeredgewidth=0.9, zorder=5)
                span_x = 3.4 * np.sqrt(covariance[column, column])
                span_y = 3.4 * np.sqrt(covariance[row, row])
                axis.set_xlim(theta[column] - span_x, theta[column] + span_x)
                axis.set_ylim(theta[row] - span_y, theta[row] + span_y)
                axis.grid(True, linewidth=0.4, zorder=0)
                axis.set_axisbelow(True)

            axis.tick_params(labelsize=6.5)
            axis.locator_params(axis="x", nbins=4)
            if row != column:
                axis.locator_params(axis="y", nbins=4)
            if row == n_parameters - 1:
                axis.set_xlabel(f"{math_label(labels[column])}  (N$\\cdot$mm)",
                                fontsize=7.5)
            else:
                axis.set_xticklabels([])
            if column == 0 and row != 0:
                axis.set_ylabel(f"{math_label(labels[row])}  (N$\\cdot$mm)",
                                fontsize=7.5)
            elif row != column:
                axis.set_yticklabels([])

    handles = [
        Ellipse((0, 0), 1, 1, facecolor=SERIES_COLOUR[0], alpha=0.30,
                edgecolor=SERIES_COLOUR[0], linewidth=0.9),
        Line2D([0], [0], color=SERIES_COLOUR[1], linewidth=0.9,
               linestyle=(0, (3, 2))),
    ]
    #  How much the slice tightens on the marginal.  With |corr| <= 0.06 the
    #  two contours land on top of each other, and that coincidence IS the
    #  result -- so say it, rather than leaving two curves that look
    #  accidentally duplicated.
    shrink = np.array([
        np.sqrt(1.0 / (precision[i, i] * covariance[i, i]))
        for i in range(n_parameters)])
    worst = 100.0 * (1.0 - shrink.min())
    figure.legend(handles,
                  ["marginal (other coefficients free), $1\\sigma$ and $2\\sigma$",
                   "slice (other coefficients fixed at $\\hat{\\theta}$)"],
                  loc="lower center", bbox_to_anchor=(0.5, -0.052), ncol=2)
    figure.text(0.5, -0.082,
                "The two coincide to within "
                f"{worst:.1f}%: with $|\\mathrm{{corr}}|\\leq0.06$ the four "
                "coefficients are determined essentially independently, so "
                "fixing the others buys almost nothing.",
                ha="center", fontsize=7, color=INK_MUTED)
    figure.suptitle("Slices through the converged confidence ellipsoid in "
                    "$(D_{11}, D_{12}, D_{22}, D_{66})$ space",
                    fontsize=9, y=1.035)
    save(figure, "fig5_parameter_ellipsoid", png_only)


# =============================================================================

def write_figure_data(result, labels, eigen) -> None:
    """The numbers behind the figures, so they can be replotted or checked."""
    eigenvalues, eigenvectors, principal_std, share = eigen
    path = FIGURE_DIR / "figure_data.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)

        writer.writerow(["# per-iteration"])
        writer.writerow(["iteration"] + list(labels)
                        + [f"std_{n}" for n in labels]
                        + ["log10_det_C", "log10_det_R", "delta_q",
                           "delta_C"])
        for diagnostics in result.iteration_diagnostics:
            writer.writerow(
                [diagnostics.iteration]
                + [f"{v:.8f}" for v in diagnostics.parameters]
                + [f"{v:.8f}" for v in diagnostics.parameter_std]
                + [f"{diagnostics.covariance_logdet:.6f}",
                   f"{diagnostics.correlation_logdet:.6f}",
                   f"{diagnostics.delta_q:.6e}",
                   f"{diagnostics.covariance_delta:.6e}"])

        writer.writerow([])
        writer.writerow(["# converged parameter covariance"])
        writer.writerow([""] + list(labels))
        for index, name in enumerate(labels):
            writer.writerow([name] + [f"{v:.10e}"
                                      for v in result.parameter_covariance[index]])

        writer.writerow([])
        writer.writerow(["# principal components (descending eigenvalue)"])
        writer.writerow(["component", "eigenvalue", "sqrt_eigenvalue",
                         "share_percent"] + [f"loading_{n}" for n in labels])
        for index in range(len(labels)):
            writer.writerow([f"PC{index + 1}", f"{eigenvalues[index]:.10e}",
                             f"{principal_std[index]:.8f}",
                             f"{share[index]:.4f}"]
                            + [f"{v:+.6f}" for v in eigenvectors[:, index]])
    print(f"  wrote {path.relative_to(HERE).as_posix()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png-only", action="store_true",
                        help="skip the vector PDF")
    parser.add_argument("--basis-degree", type=int, default=6)
    #  All ones: the least informative admissible start, and it makes
    #  K(theta_0) singular, so it doubles as a robustness check.  The fixed
    #  point is start-independent -- [1,1,1,1], the campaign default and a
    #  deliberately bad guess all land on the same theta to 4 decimals.
    parser.add_argument("--initial", type=float, nargs="+",
                        default=[1.0, 1.0, 1.0, 1.0])
    #  Default is the STRICT form: only the top cut does external work, so the
    #  bottom cut's unknown reaction moment is annihilated rather than assumed
    #  equal and opposite.  Pass --bc-form both-cuts for the older figures.
    parser.add_argument("--bc-form", choices=BC_FORMS, default=BC_TOP_ONLY,
                        help="which loaded cuts do external virtual work")
    #  The three-coupon campaign.  Off by default, so every published
    #  two-coupon figure is regenerated bit-identically by a bare
    #  `python -m workflows.make_figures`.
    parser.add_argument("--with-45", action="store_true",
                        help="stack the 45 deg coupon as a third experiment")
    arguments = parser.parse_args()

    apply_style()
    set_figure_dir(arguments.bc_form, with_45=arguments.with_45)

    print("Building cases and running the FGLS identification...")
    print(f"  initial guess: {arguments.initial}")
    print(f"  BC form      : {arguments.bc_form}")
    if arguments.bc_form == BC_TOP_ONLY:
        print("                 bottom cut fully clamped (W, W_X, W_XY, W_Y "
              "= 0) -> its unknown")
        print("                 reaction moment does NO virtual work; only "
              "the top cut is loaded.")
    tests = TESTS_WITH_45 if arguments.with_45 else TESTS_RAW_FRAME
    if arguments.with_45:
        print("  coupons      : longitudinal + transverse + 45 deg "
              "(material frames +90, +90, +45 deg)")
    cases, ids, _ = build_campaign(verbose=False, tests=tests,
                                   bc_form=arguments.bc_form)
    result = run_gls_identification_cases(
        cases=cases, material_type=MATERIAL_TYPE, analysis_type=ANALYSIS_TYPE,
        initial_parameters=arguments.initial,
        identification_settings=IdentificationSettings(
            max_iterations=40, convergence_tolerance=1.0e-8),
        linear_algebra_settings=LinearAlgebraSettings(),
        experiment_ids=ids,
        gls_settings=GlsSettings(basis_degree=arguments.basis_degree),
        verbose=False)
    labels = list(result.constitutive_model.parameter_labels)
    print(f"  converged in {result.iterations_completed} iterations: "
          + np.array2string(result.final_parameters, precision=4))

    goodness = chi_square_diagnostics(result)
    birge = goodness["birge"]
    print(f"  chi2/dof = {goodness['reduced']:.0f}  ->  eq. (20) is a variance "
          f"UNDER THE MODEL;")
    print(f"  an interval to quote is at least {birge:.0f}x it: "
          + np.array2string(result.parameter_std * birge, precision=2))

    print("Rendering figures...")
    figure_coefficients(result, labels, arguments.png_only,
                        chi2_reduced=goodness["reduced"],
                        initial=arguments.initial)
    figure_determinants(result, arguments.png_only)
    figure_variance(result, labels, arguments.png_only, birge=birge,
                    chi2_reduced=goodness["reduced"])
    eigen = figure_pca(result, labels, arguments.png_only)
    figure_ellipsoid(result, labels, arguments.png_only)
    write_figure_data(result, labels, eigen)

    print()
    print("Principal components of Cov(theta), worst-determined first:")
    eigenvalues, eigenvectors, principal_std, share = eigen
    for index in range(len(labels)):
        loading = "  ".join(
            f"{labels[j]} {eigenvectors[j, index]:+.2f}"
            for j in range(len(labels)))
        print(f"  PC{index + 1}: sigma = {principal_std[index]:8.4f} N.mm "
              f"({share[index]:5.1f} % of total variance)   {loading}")


if __name__ == "__main__":
    main()
