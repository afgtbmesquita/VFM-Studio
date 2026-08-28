"""Publication-style plots for the three UQ pages."""

from __future__ import annotations

import numpy as np

from vfmstudio.core.document import parameter_math_label, parameter_unit_math
from vfmstudio.core.uncertainty import UncertaintyResult, correlation
from vfmstudio.core.uncertainty.deletion import I_LARGE, U_MATERIAL
from vfmstudio.gui.results.plot_style import (
    GRID_COLOUR,
    INK_MUTED,
    INK_PRIMARY,
    SERIES_COLOUR,
    SERIES_MARKER,
    annotate_heatmap,
    draw_matrix,
    empty_message,
    experiment_colour,
    parameter_tick_labels,
    style_axis,
    style_heatmap_axis,
)

WORK = "#2a78d6"
ROBUST = "#1baf7a"
CR3 = "#eb6834"


def _panel_grid(figure, n: int, *, extra: bool = False):
    n_rows = max(1, (n + 1) // 2)
    n_cols = 2 if n > 1 else 1
    if extra:
        grid = figure.add_gridspec(
            n_rows,
            n_cols + 1,
            width_ratios=[1.0] * n_cols + [1.12],
            left=0.10,
            right=0.97,
            top=0.88,
            bottom=0.12,
            wspace=0.50,
            hspace=0.48,
        )
        panels = [
            figure.add_subplot(grid[index // n_cols, index % n_cols])
            for index in range(n)
        ]
        extra_ax = figure.add_subplot(grid[:, n_cols])
        return panels, extra_ax
    if n <= 2:
        axes = figure.subplots(1, n) if n > 1 else [figure.add_subplot(111)]
        figure.subplots_adjust(left=0.10, right=0.96, top=0.86, bottom=0.18, wspace=0.45)
        return list(np.atleast_1d(axes)), None
    axes = figure.subplots(n_rows, 2)
    figure.subplots_adjust(left=0.10, right=0.96, top=0.88, bottom=0.12, wspace=0.45, hspace=0.48)
    flat = list(np.ravel(axes))
    for extra in flat[n:]:
        extra.set_visible(False)
    return flat[:n], None


def plot_theta_bands(figure, result: UncertaintyResult) -> None:
    n = result.n_parameters
    panels, _ = _panel_grid(figure, n)
    for index, axis in enumerate(panels):
        theta = float(result.theta[index])
        work = float(result.sigma_work[index])
        robust = float(result.sigma_robust[index]) if result.well_posed else float("nan")
        bands = [(r"working, eq. (35)", work, WORK, 2.2)]
        if np.isfinite(robust):
            bands.append((r"robust, eq. (34)", robust, ROBUST, 3.0))
        for slot, (_, half, colour, weight) in enumerate(bands):
            y = len(bands) - 1 - slot
            axis.plot(
                [theta - half, theta + half],
                [y, y],
                color=colour,
                linewidth=weight,
                solid_capstyle="round",
                zorder=4,
            )
            for edge in (theta - half, theta + half):
                axis.plot(
                    [edge, edge],
                    [y - 0.13, y + 0.13],
                    color=colour,
                    linewidth=weight * 0.7,
                    zorder=4,
                )
            axis.text(
                theta + half,
                y + 0.22,
                f"$\\pm${half:.3g}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=colour,
            )
        axis.plot(
            [theta],
            [0],
            marker="o",
            markersize=5.2,
            color=INK_PRIMARY,
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=6,
        )
        axis.axvline(theta, color=INK_PRIMARY, linewidth=0.8, linestyle="--", alpha=0.55, zorder=2)
        widest = max(band[1] for band in bands) if bands else abs(theta) * 0.05
        axis.set_xlim(theta - 1.45 * max(widest, abs(theta) * 1e-6), theta + 1.45 * max(widest, abs(theta) * 1e-6))
        axis.set_ylim(-0.45, max(len(bands) - 0.5, 0.6))
        axis.set_yticks(range(len(bands)))
        axis.set_yticklabels([label for label, *_ in bands][::-1], fontsize=7)
        axis.tick_params(length=0, axis="y")
        style_axis(axis, grid="x")
        unit = parameter_unit_math(result.labels[index])
        title = parameter_math_label(result.labels[index])
        if np.isfinite(robust) and work > 0:
            title = rf"{title}  $=$  {theta:.3g}" + (
                rf"      (robust / working $=$ {robust / work:.1f}$\times$)"
            )
        axis.set_title(title, pad=8)
        if unit:
            axis.set_xlabel(unit, labelpad=6)


def plot_uncertainty_ladder(figure, result: UncertaintyResult) -> None:
    axis = figure.add_subplot(111)
    style_axis(axis)
    labels = result.labels
    x = np.arange(len(labels), dtype=float)
    theta = np.clip(np.abs(result.theta), np.finfo(float).tiny, None)
    series = [
        (100.0 * result.sigma_work / theta, "eq. (35)  working", WORK),
        (100.0 * result.sigma_robust / theta, f"eq. (34)  sandwich {result.kind}", ROBUST),
        (100.0 * result.sigma_cr3 / theta, "eq. (34)  sandwich CR3", CR3),
    ]
    width = 0.22
    offsets = (np.arange(len(series)) - 1.0) * width
    for shift, (values, label, colour) in zip(offsets, series):
        faded = not result.well_posed and "sandwich" in label
        axis.bar(
            x + shift,
            np.nan_to_num(values),
            width=0.9 * width,
            color=colour,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
            alpha=0.25 if faded else 1.0,
            label=label,
            hatch="//" if faded else None,
        )
    axis.set_xticks(x)
    axis.set_xticklabels(parameter_tick_labels(labels))
    axis.set_yscale("log")
    axis.set_ylabel(r"relative uncertainty $\sigma/|\theta|$  (%)")
    axis.set_title(r"The uncertainty ladder: what each $\sigma$ actually answers", pad=8)
    axis.legend(loc="upper left", fontsize=8)
    figure.tight_layout(pad=1.2)


def plot_covariance_matrix(figure, result: UncertaintyResult, which: str) -> None:
    if which == "work":
        matrix = result.covariances["work"]
        title = r"Working covariance $A^{-1}$  (eq. 35)"
        cmap = "YlGnBu"
        vmin = vmax = None
    elif which == "robust":
        if not result.well_posed:
            empty_message(figure, "Robust Cov(θ) is refused: M < p + 1 independent experiments.")
            return
        matrix = result.robust
        title = rf"Robust covariance  ({result.kind}, eq. 34)"
        cmap = "YlGnBu"
        vmin = vmax = None
    else:
        source = result.robust if result.well_posed else result.covariances["work"]
        matrix = correlation(source)
        title = "Robust correlation" if result.well_posed else "Working correlation"
        cmap = "RdBu_r"
        vmin, vmax = -1.0, 1.0
    axis = figure.add_subplot(111)
    image = draw_matrix(
        axis, matrix, result.labels, title=title, cmap=cmap, vmin=vmin, vmax=vmax
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(r"$\mathrm{Cov}(\theta)$" if which != "corr" else r"$\rho$")
    colorbar.ax.tick_params(colors="#52514e", labelsize=8)
    figure.tight_layout(pad=1.2)


def plot_misfit_spectrum(figure, result: UncertaintyResult) -> None:
    if not result.spectrum.well_posed:
        empty_message(
            figure,
            "Misfit spectrum needs a non-singular robust covariance (M ≥ p + 1).",
        )
        return
    axis = figure.add_subplot(111)
    style_axis(axis)
    lam = np.asarray(result.spectrum.eigenvalues, dtype=float)
    x = np.arange(lam.size, dtype=float)
    axis.scatter(
        x, lam, s=42, color=SERIES_COLOUR[0], zorder=4, edgecolor="white", linewidth=0.6
    )
    chi = result.chi2.get("reduced", float("nan"))
    if np.isfinite(chi):
        axis.axhline(chi, color=CR3, linewidth=1.6, zorder=3, label=r"$\chi^2/\mathrm{dof}$")
    axis.axhline(1.0, color=INK_MUTED, linewidth=1.0, linestyle=":", zorder=2, label=r"$\lambda=1$")
    axis.set_xticks(x)
    axis.set_xticklabels([rf"$c_{{{i}}}$" for i in range(lam.size)])
    axis.set_yscale("log")
    axis.set_ylabel(r"$\lambda_j$ of $A^{1/2}V_{\mathrm{sand}}A^{1/2}$")
    axis.set_title(r"Could one scalar replace eq. (34)?", pad=8)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout(pad=1.2)


def plot_cluster_scores(figure, result: UncertaintyResult) -> None:
    axis = figure.add_subplot(111)
    style_axis(axis)
    names = result.cluster_names
    x = np.arange(len(names), dtype=float)
    colors = [experiment_colour(i) for i in range(len(names))]
    axis.bar(
        x,
        np.maximum(result.score_norms, 1.0e-30),
        color=colors,
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )
    axis.set_xticks(x)
    axis.set_xticklabels(names, rotation=30, ha="right")
    axis.set_yscale("log")
    axis.set_ylabel(r"$\hat\psi_i^{\mathrm{T}}A^{-1}\hat\psi_i$")
    axis.set_title(r"How hard does each experiment pull?  (eq. 32)", pad=8)
    figure.tight_layout(pad=1.2)


def plot_score_components(figure, result: UncertaintyResult) -> None:
    n = result.n_parameters
    panels, _ = _panel_grid(figure, n)
    names = result.cluster_names
    x = np.arange(len(names), dtype=float)
    for index, axis in enumerate(panels):
        style_axis(axis)
        axis.axhline(0.0, color=INK_MUTED, linewidth=0.7, zorder=1)
        axis.bar(
            x,
            result.scores[:, index],
            color=[experiment_colour(i) for i in range(len(names))],
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        axis.set_xticks(x)
        axis.set_xticklabels(names if index >= n - 2 or n <= 2 else [], rotation=30, ha="right")
        axis.set_title(parameter_math_label(result.labels[index]), pad=8)
        axis.set_ylabel(r"$\psi_{ik}$")
    figure.tight_layout(pad=1.2)


def plot_sandwich_anatomy(figure, result: UncertaintyResult) -> None:
    panels = [
        (result.A, r"$A=\sum_i H_i^{\mathrm{T}}W_i H_i$", "eq. (31)"),
        (result.B, r"$B=\sum_i \hat\psi_i\hat\psi_i^{\mathrm{T}}$", "eq. (33)"),
    ]
    if result.G is not None:
        panels.append((result.G, r"$G=A^{\mathrm{T}}B^{-1}A$", "eq. (26)"))
    axes = figure.subplots(1, len(panels))
    axes = np.atleast_1d(axes)
    for axis, (matrix, title, equation) in zip(axes, panels):
        scale = float(np.max(np.abs(matrix))) or 1.0
        image = draw_matrix(
            axis,
            matrix / scale,
            result.labels,
            title=f"{title}\n{equation}",
            cmap="PuOr_r",
            vmin=-1.0,
            vmax=1.0,
        )
    figure.colorbar(image, ax=list(axes), fraction=0.025, pad=0.02, label="scaled to panel max")
    figure.suptitle("Anatomy of the sandwich (each panel scaled to its own maximum)", fontsize=10, y=0.98)
    figure.tight_layout(pad=1.2)


def plot_ab_matrices(figure, result: UncertaintyResult, which: str = "both") -> None:
    """Unscaled A and/or B so the identification's actual entries are readable."""
    panels = []
    if which in ("both", "A"):
        panels.append(
            (result.A, r"$A=\sum_i H_i^{\mathrm{T}}W_i H_i$", "eq. (31)", r"$A$")
        )
    if which in ("both", "B"):
        panels.append(
            (result.B, r"$B=\sum_i \hat\psi_i\hat\psi_i^{\mathrm{T}}$", "eq. (33)", r"$B$")
        )
    axes = figure.subplots(1, len(panels)) if len(panels) > 1 else [figure.add_subplot(111)]
    axes = list(np.atleast_1d(axes))
    for axis, (matrix, title, equation, bar_label) in zip(axes, panels):
        span = float(np.max(np.abs(matrix))) or 1.0
        image = draw_matrix(
            axis,
            matrix,
            result.labels,
            title=f"{title}\n{equation}",
            cmap="PuOr_r",
            vmin=-span,
            vmax=span,
            fmt="{:.3g}",
        )
        colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label(bar_label)
        colorbar.ax.tick_params(colors="#52514e", labelsize=8)
    figure.suptitle("Sensitivity $A$ and variability $B$ (actual values)", fontsize=10, y=0.98)
    figure.tight_layout(pad=1.2)


def plot_deletion_shifts(figure, result: UncertaintyResult) -> None:
    rows = result.deletions
    if not rows:
        empty_message(figure, "No leave-one-out diagnostics yet.")
        return
    names = [row.dropped for row in rows]
    n = result.n_parameters
    panels, _ = _panel_grid(figure, n)
    y = np.arange(len(names), dtype=float)
    for index, axis in enumerate(panels):
        style_axis(axis, grid="x")
        full = float(result.theta[index])
        work = float(result.sigma_work[index])
        robust = float(result.sigma_robust[index]) if result.well_posed else float("nan")
        if np.isfinite(robust):
            axis.axvspan(full - robust, full + robust, color=ROBUST, alpha=0.16, zorder=1)
        axis.axvspan(full - work, full + work, color=WORK, alpha=0.22, zorder=2)
        axis.axvline(full, color=INK_PRIMARY, linewidth=1.0, zorder=3)
        fixed = np.array([row.theta_deleted_fixed[index] for row in rows])
        axis.scatter(
            fixed, y, s=22, marker="x", color=INK_MUTED, zorder=4, linewidth=0.9
        )
        refit = np.array(
            [
                row.theta_deleted_refit[index]
                if row.theta_deleted_refit is not None
                else np.nan
                for row in rows
            ]
        )
        if np.isfinite(refit).any():
            axis.scatter(
                refit,
                y,
                s=30,
                color=[experiment_colour(i) for i in range(len(names))],
                edgecolor="white",
                linewidth=0.5,
                zorder=5,
            )
        axis.set_yticks(y)
        axis.set_yticklabels(names if index % 2 == 0 else [], fontsize=7)
        axis.set_title(parameter_math_label(result.labels[index]), pad=8)
        unit = parameter_unit_math(result.labels[index])
        if unit:
            axis.set_xlabel(unit, labelpad=4)
    figure.tight_layout(pad=1.2)


def _deletion_grid(result: UncertaintyResult, attr: str) -> np.ndarray:
    rows = result.deletions
    matrix = np.full((len(rows), result.n_parameters), np.nan)
    for i, row in enumerate(rows):
        matrix[i, :] = np.asarray(getattr(row, attr), dtype=float)
    return matrix


def plot_precision_heatmap(figure, result: UncertaintyResult) -> None:
    if not result.deletions:
        empty_message(figure, "No leave-one-out diagnostics yet.")
        return
    matrix = _deletion_grid(result, "precision_contribution")
    axis = figure.add_subplot(111)
    style_heatmap_axis(axis)
    span = float(np.nanmax(np.abs(matrix))) or 1.0
    image = axis.imshow(
        matrix, cmap="RdBu_r", vmin=-span, vmax=span, origin="upper", aspect="auto"
    )
    axis.set_xticks(np.arange(result.n_parameters))
    axis.set_xticklabels(parameter_tick_labels(result.labels))
    axis.set_yticks(np.arange(len(result.deletions)))
    axis.set_yticklabels([row.dropped for row in result.deletions], fontsize=8)
    annotate_heatmap(axis, matrix, fmt="{:+.2f}", vmin=-span, vmax=span)
    axis.set_title(r"Precision contribution $U_{ik}$  (eq. 57)", pad=8)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label=r"$U_{ik}$")
    figure.tight_layout(pad=1.2)


def plot_influence_heatmap(figure, result: UncertaintyResult) -> None:
    if not result.deletions:
        empty_message(figure, "No leave-one-out diagnostics yet.")
        return
    matrix = _deletion_grid(result, "influence")
    axis = figure.add_subplot(111)
    style_heatmap_axis(axis)
    finite = matrix[np.isfinite(matrix) & (matrix > 0)]
    if finite.size:
        import matplotlib

        norm = matplotlib.colors.LogNorm(
            vmin=max(float(finite.min()), 1.0e-3), vmax=float(finite.max())
        )
    else:
        norm = None
    image = axis.imshow(
        matrix, cmap="magma_r", norm=norm, origin="upper", aspect="auto"
    )
    axis.set_xticks(np.arange(result.n_parameters))
    axis.set_xticklabels(parameter_tick_labels(result.labels))
    axis.set_yticks(np.arange(len(result.deletions)))
    axis.set_yticklabels([row.dropped for row in result.deletions], fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            cell = matrix[i, j]
            text = "—" if not np.isfinite(cell) else f"{cell:.3g}"
            dark = np.isfinite(cell) and norm is not None and float(norm(cell)) > 0.55
            axis.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=7,
                color="#f7f6f2" if dark else INK_PRIMARY,
            )
    axis.set_title(r"Standardized deletion influence $I_{ik}$  (eq. 61)", pad=8)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label=r"$I_{ik}$")
    figure.tight_layout(pad=1.2)


def plot_information_vs_influence(figure, result: UncertaintyResult) -> None:
    if not result.deletions:
        empty_message(figure, "No leave-one-out diagnostics yet.")
        return
    axis = figure.add_subplot(111)
    style_axis(axis, grid="both")
    for i, row in enumerate(result.deletions):
        for k, label in enumerate(result.labels):
            u = float(row.precision_contribution[k])
            influence = float(row.influence[k])
            if not (np.isfinite(u) and np.isfinite(influence)):
                continue
            axis.scatter(
                u,
                influence,
                s=44,
                color=experiment_colour(i),
                marker=SERIES_MARKER[k % len(SERIES_MARKER)],
                edgecolor="white",
                linewidth=0.6,
                zorder=4,
            )
    axis.axvline(U_MATERIAL, color=GRID_COLOUR, linewidth=0.8, zorder=1)
    axis.axvline(-U_MATERIAL, color=GRID_COLOUR, linewidth=0.8, zorder=1)
    axis.axhline(I_LARGE, color=GRID_COLOUR, linewidth=0.8, zorder=1)
    limits = axis.get_xlim()
    axis.axvspan(limits[0], -U_MATERIAL, color=CR3, alpha=0.06, zorder=0)
    axis.set_xlim(limits)
    axis.set_xlabel(r"precision contribution  $U_{ik}$   (eq. 57)")
    axis.set_ylabel(r"deletion influence  $I_{ik}$   (eq. 61)")
    axis.set_title("Information and influence are not the same thing", pad=8)
    from matplotlib.lines import Line2D

    coupon_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=5,
            color=experiment_colour(i),
            label=row.dropped,
        )
        for i, row in enumerate(result.deletions)
    ]
    parameter_handles = [
        Line2D(
            [],
            [],
            marker=SERIES_MARKER[k % len(SERIES_MARKER)],
            linestyle="",
            markersize=5,
            color=INK_MUTED,
            label=parameter_math_label(label),
        )
        for k, label in enumerate(result.labels)
    ]
    axis.legend(
        handles=coupon_handles + parameter_handles,
        fontsize=7,
        ncol=2,
        loc="best",
    )
    figure.tight_layout(pad=1.2)


def plot_deletion_condition(figure, result: UncertaintyResult) -> None:
    info = result.information
    rows = info.contributions
    if not rows:
        empty_message(figure, "Need at least one experiment to inspect A.")
        return
    axis = figure.add_subplot(111)
    style_axis(axis, grid="y")
    names = [row.name for row in rows]
    x = np.arange(len(names), dtype=float)
    values = np.array(
        [
            row.remaining_condition if np.isfinite(row.remaining_condition) else np.nan
            for row in rows
        ],
        dtype=float,
    )
    plot_values = np.where(
        np.isfinite(values),
        values,
        np.nanmax(values[np.isfinite(values)]) if np.isfinite(values).any() else 1.0,
    )
    axis.bar(
        x,
        plot_values,
        color=[experiment_colour(i) for i in range(len(names))],
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )
    for index, row in enumerate(rows):
        if not np.isfinite(row.remaining_condition):
            axis.text(
                x[index],
                plot_values[index],
                "∞",
                ha="center",
                va="bottom",
                fontsize=8,
                color=INK_PRIMARY,
            )
    if np.isfinite(info.condition) and info.condition > 0:
        axis.axhline(
            info.condition,
            color=WORK,
            linewidth=1.4,
            linestyle="--",
            zorder=4,
            label=rf"$\mathrm{{cond}}(A)={info.condition:.3g}$",
        )
        axis.legend(loc="best", fontsize=8)
    axis.set_xticks(x)
    axis.set_xticklabels(names, rotation=30, ha="right")
    axis.set_yscale("log")
    axis.set_ylabel(r"$\mathrm{cond}(A-A_i)$")
    axis.set_title(r"What happens to $A$ if a test is dropped", pad=8)
    figure.tight_layout(pad=1.2)


def plot_information_share(figure, result: UncertaintyResult) -> None:
    info = result.information
    if not info.contributions:
        empty_message(figure, "Need at least one experiment to inspect A.")
        return
    matrix = info.shares
    axis = figure.add_subplot(111)
    style_heatmap_axis(axis)
    image = axis.imshow(
        matrix, cmap="YlGnBu", vmin=0.0, vmax=1.0, origin="upper", aspect="auto"
    )
    axis.set_xticks(np.arange(result.n_parameters))
    axis.set_xticklabels(parameter_tick_labels(result.labels))
    axis.set_yticks(np.arange(len(info.contributions)))
    axis.set_yticklabels([row.name for row in info.contributions], fontsize=8)
    annotate_heatmap(axis, matrix, fmt="{:.2f}", vmin=0.0, vmax=1.0)
    axis.set_title(r"Which coupon supplies which direction  $(A_i)_{kk}/A_{kk}$", pad=8)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="share of $A_{kk}$")
    figure.tight_layout(pad=1.2)


def plot_a_spectrum(figure, result: UncertaintyResult) -> None:
    info = result.information
    axis = figure.add_subplot(111)
    style_axis(axis)
    lam = np.asarray(info.eigenvalues, dtype=float)
    x = np.arange(lam.size, dtype=float)
    axis.bar(
        x,
        np.maximum(lam, np.finfo(float).tiny),
        color=WORK,
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )
    axis.set_xticks(x)
    axis.set_xticklabels([rf"$a_{{{i + 1}}}$" for i in range(lam.size)])
    axis.set_yscale("log")
    axis.set_ylabel(r"eigenvalue of $A$")
    cond = info.condition
    cond_text = rf"$\mathrm{{cond}}(A)={cond:.3g}$" if np.isfinite(cond) else r"$\mathrm{cond}(A)=\infty$"
    axis.set_title(rf"Which combinations of $D_{{ij}}$ the pooled design constrains  ({cond_text})", pad=8)
    figure.tight_layout(pad=1.2)


def plot_a_directions(figure, result: UncertaintyResult) -> None:
    info = result.information
    matrix = np.asarray(info.eigenvectors, dtype=float)
    axis = figure.add_subplot(111)
    style_heatmap_axis(axis)
    span = float(np.max(np.abs(matrix))) or 1.0
    image = axis.imshow(
        matrix, cmap="RdBu_r", vmin=-span, vmax=span, origin="upper", aspect="auto"
    )
    axis.set_xticks(np.arange(matrix.shape[1]))
    axis.set_xticklabels([rf"$a_{{{i + 1}}}$" for i in range(matrix.shape[1])])
    axis.set_yticks(np.arange(result.n_parameters))
    axis.set_yticklabels(parameter_tick_labels(result.labels))
    annotate_heatmap(axis, matrix, fmt="{:.2f}", vmin=-span, vmax=span)
    axis.set_title(r"Eigenvectors of $A$: constrained combinations of $D_{ij}$", pad=8)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="loading")
    figure.tight_layout(pad=1.2)
