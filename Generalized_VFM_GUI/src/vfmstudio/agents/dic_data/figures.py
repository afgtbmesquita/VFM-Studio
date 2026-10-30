"""PNG figures of one analysed test (matplotlib object API, safe off the GUI thread)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure

from vfmstudio.agents.dic_data import noise_model as nm
from vfmstudio.agents.dic_data.analysis import TestAnalysis

DPI = 100
SAVE_DPI = 220
_LABELS = {"kxx": "κxx", "kyy": "κyy", "kxy": "κxy"}


def _figure(width: float, height: float) -> Figure:
    figure = Figure(figsize=(width, height), dpi=DPI, layout="constrained")
    FigureCanvasAgg(figure)
    return figure


def _save(figure: Figure, path: Path) -> Path:
    figure.savefig(path, dpi=SAVE_DPI)
    return path


def _show(ax, image, extent, title, cmap="viridis", diverging=False, vmin=None, vmax=None, unit="", smooth=True):
    # Interpolating colours (not data) keeps NaN facets out of the blend: holes stay holes.
    style = {
        "origin": "lower",
        "extent": extent,
        "interpolation": "bilinear" if smooth else "nearest",
        "interpolation_stage": "rgba",
    }
    if diverging:
        finite = image[np.isfinite(image)]
        bound = float(np.percentile(np.abs(finite), 99)) if finite.size else 1.0
        bound = bound or 1.0
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-bound, vmax=bound)
        artist = ax.imshow(image, cmap="RdBu_r", norm=norm, **style)
    else:
        artist = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, **style)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("X [mm]", fontsize=8)
    ax.set_ylabel("Y [mm]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal")
    bar = ax.figure.colorbar(artist, ax=ax, shrink=0.85)
    bar.ax.tick_params(labelsize=7)
    if unit:
        bar.set_label(unit, fontsize=7)


def _panel_size(analysis: TestAnalysis, columns: int) -> tuple[float, float]:
    extent = analysis.grid.extent
    aspect = (extent[3] - extent[2]) / max(1e-9, extent[1] - extent[0])
    width = 3.0 * columns
    # ~1.6 in of each 3 in panel is image; the rest is colour bar and labels.
    height = min(8.0, max(2.6, 1.6 * aspect + 1.0))
    return width, height


def fields_figure(analysis: TestAnalysis, path: Path) -> Path:
    final = analysis.facts["deformation"]["final_step_analysed"]
    width, height = _panel_size(analysis, 4)
    figure = _figure(width, height)
    axes = figure.subplots(1, 4)
    extent = analysis.grid.extent
    _show(axes[0], analysis.images["dW"], extent, "ΔW", diverging=True, unit="mm")
    for ax, comp in zip(axes[1:], nm.COMPONENTS):
        _show(ax, analysis.images[f"d{comp}"], extent, f"Δ{_LABELS[comp]}", diverging=True, unit="1/mm")
    figure.suptitle(f"{analysis.name}: increments from step 1 to step {final}", fontsize=10)
    return _save(figure, path)


def response_figure(analysis: TestAnalysis, path: Path) -> Path:
    figure = _figure(9.0, 3.4)
    left, right = figure.subplots(1, 2)
    load = analysis.series["load"]
    label = str(analysis.series["load_label"][0])
    mean_dk = analysis.series["mean_dk"]
    for i, comp in enumerate(nm.COMPONENTS):
        left.plot(load, mean_dk[:, i], marker="o", ms=3, label=f"Δ{_LABELS[comp]}")
    left.set_xlabel(label, fontsize=8)
    left.set_ylabel("interior mean Δκ [1/mm]", fontsize=8)
    left.legend(fontsize=7)
    left.grid(alpha=0.3)
    left.tick_params(labelsize=7)
    steps = analysis.facts["deformation"]["steps"]
    machine_names = sorted({name for row in steps for name in row["machine"]})
    plotted = 0
    for name in machine_names:
        if not (name.split(".")[-1].startswith("T") or name.split(".")[-1].startswith("θ")):
            continue
        values = [row["machine"].get(name) for row in steps]
        if any(value is None for value in values) or max(values) - min(values) == 0:
            continue
        right.plot(range(1, len(values) + 1), values, marker="o", ms=3, label=name)
        plotted += 1
    right.set_xlabel("step", fontsize=8)
    right.set_ylabel("moment [N·mm] / rotation [deg]", fontsize=8)
    if plotted:
        right.legend(fontsize=7, ncol=2)
    else:
        right.text(0.5, 0.5, "no machine data", ha="center", va="center", transform=right.transAxes)
    right.grid(alpha=0.3)
    right.tick_params(labelsize=7)
    figure.suptitle(f"{analysis.name}: load response", fontsize=10)
    return _save(figure, path)


def noise_maps_figure(analysis: TestAnalysis, path: Path) -> Path | None:
    if "measured_kxx" not in analysis.images:
        return None
    width, height = _panel_size(analysis, 3)
    figure = _figure(width, height * 2 - 0.4)
    axes = figure.subplots(2, 3)
    extent = analysis.grid.extent
    for col, comp in enumerate(nm.COMPONENTS):
        measured = analysis.images[f"measured_{comp}"]
        model = analysis.images[f"model_{comp}"]
        finite = measured[np.isfinite(measured)]
        vmax = float(np.percentile(finite, 98)) if finite.size else None
        _show(axes[0, col], measured, extent, f"measured σ {_LABELS[comp]}", cmap="magma", vmin=0, vmax=vmax, unit="1/mm")
        _show(axes[1, col], model, extent, f"model σ {_LABELS[comp]}", cmap="magma", vmin=0, vmax=vmax, unit="1/mm")
    figure.suptitle(f"{analysis.name}: per-facet noise, measured (top) vs model (bottom)", fontsize=10)
    return _save(figure, path)


def noise_profile_figure(analysis: TestAnalysis, path: Path) -> Path | None:
    if not analysis.profiles:
        return None
    figure = _figure(9.0, 3.0)
    axes = figure.subplots(1, 3)
    for ax, (index, comp) in zip(axes, enumerate(nm.COMPONENTS)):
        profile = analysis.profiles[comp]
        distance = np.arange(len(profile["measured"]))
        measured = np.array([np.nan if v is None else v for v in profile["measured"]])
        model = np.array([np.nan if v is None else v for v in profile["model"]])
        ax.plot(distance, measured, "o-", ms=3, label="measured")
        ax.plot(distance, model, "s--", ms=3, label="model")
        ax.axhline(nm.LEGACY_ENVELOPE_INV_MM[index], color="grey", ls=":", label="legacy envelope")
        ax.set_title(_LABELS[comp], fontsize=9)
        ax.set_xlabel("facets from nearest edge (12 = ≥12)", fontsize=8)
        ax.set_ylabel("median σ [1/mm]", fontsize=8)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7)
    figure.suptitle(f"{analysis.name}: noise vs distance to the boundary", fontsize=10)
    return _save(figure, path)


def filter_sweep_figure(analysis: TestAnalysis, path: Path) -> Path | None:
    params = analysis.facts.get("dic_parameters") or {}
    sweep = params.get("curvature_filter_sweep")
    if not sweep or not sweep["rows"]:
        return None

    def column(key, sub=None):
        out = []
        for row in sweep["rows"]:
            value = row.get(key)
            if sub is not None:
                value = (value or {}).get(sub)
            out.append(np.nan if value is None else value)
        return np.array(out, dtype=float)

    sizes = column("filter_size_facets")
    identified = sweep["identified_filter_size_facets"]
    figure = _figure(12.0, 3.2)
    axes = figure.subplots(1, 4)
    ax = axes[0]
    if sweep["noise_available"]:
        for comp in nm.COMPONENTS:
            ax.plot(sizes, column("noise_interior_median_inv_mm", comp), "o-", ms=3, label=_LABELS[comp])
        anchor = column("noise_interior_median_inv_mm", sweep["dominant_component"])
        at = np.flatnonzero(sizes == identified)
        if at.size and np.isfinite(anchor[at[0]]):
            ax.plot(sizes, anchor[at[0]] * column("white_noise_sigma_relative_to_identified"), "k:",
                    label="uncorrelated-noise scaling")
        ax.set_yscale("log")
        ax.legend(fontsize=7)
    ax.set_ylabel("interior noise [1/mm]", fontsize=8)
    ax.set_title("noise vs filter size", fontsize=9)
    axes[1].plot(sizes, column("snr_interior"), "o-", ms=3)
    axes[1].set_ylabel(f"SNR (|Δ{_LABELS[sweep['dominant_component']]}| / noise)", fontsize=8)
    axes[1].set_title("signal-to-noise", fontsize=9)
    axes[2].plot(sizes, column("smoothing_bias_index"), "o-", ms=3, label="smoothing bias index")
    axes[2].plot(sizes, column("full_window_facet_fraction"), "s--", ms=3, label="full-window facet fraction")
    axes[2].plot(sizes, column("mismatch_to_export"), "^-.", ms=3, label="mismatch to export")
    axes[2].set_ylim(0, max(1.5, float(np.nanmax(column("smoothing_bias_index"))) * 1.05)
                     if np.isfinite(column("smoothing_bias_index")).any() else None)
    axes[2].legend(fontsize=7)
    axes[2].set_title("bias, coverage, match to export", fontsize=9)
    for ax in axes[:3]:
        ax.axvline(identified, color="grey", ls="--", lw=0.8)
        ax.set_xlabel("curvature filter [facets]", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    subset = params.get("subset") or {}
    ax = axes[3]
    curves = subset.get("autocorrelation")
    if curves:
        lags = np.array(subset["lags_facets"])
        for key, values in curves.items():
            ax.plot(lags, [np.nan if v is None else v for v in values], lw=1, label=key.replace("_along_", " along "))
        ax.axhline(0.5, color="grey", ls=":", lw=0.8)
        ax.legend(fontsize=6, ncol=2)
        estimate = subset.get("subset_size_estimate_px")
        ax.set_title(f"noise correlation (subset ≈ {estimate:g} px)" if estimate else "noise correlation", fontsize=9)
    ax.set_xlabel("separation [facets]", fontsize=8)
    ax.set_ylabel("autocorrelation", fontsize=8)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=7)
    figure.suptitle(f"{analysis.name}: curvature filter sweep, {sweep['weighting']} "
                    "(dashed line: filter inferred from the export)", fontsize=10)
    return _save(figure, path)


def quality_figure(analysis: TestAnalysis, path: Path) -> Path:
    width, height = _panel_size(analysis, 2)
    figure = _figure(width, height)
    axes = figure.subplots(1, 2)
    extent = analysis.grid.extent
    vic = analysis.images["vic_sigma"]
    finite = vic[np.isfinite(vic)]
    _show(axes[0], vic, extent, "Vic-3D sigma (median over steps)", cmap="magma", vmin=0,
          vmax=float(np.percentile(finite, 99)) if finite.size else None, unit="px")
    missing = analysis.images["missing_steps"]
    _show(axes[1], missing, extent, "steps in which the facet is missing", cmap="Reds", vmin=0,
          vmax=max(1.0, float(np.nanmax(missing))), unit="steps", smooth=False)
    figure.suptitle(f"{analysis.name}: read quality", fontsize=10)
    return _save(figure, path)


def render_all(analysis: TestAnalysis, folder: Path, slug: str) -> dict[str, str]:
    folder.mkdir(parents=True, exist_ok=True)
    out = {
        "fields": fields_figure(analysis, folder / f"{slug}_fields.png"),
        "response": response_figure(analysis, folder / f"{slug}_response.png"),
        "noise_maps": noise_maps_figure(analysis, folder / f"{slug}_noise_maps.png"),
        "noise_profile": noise_profile_figure(analysis, folder / f"{slug}_noise_profile.png"),
        "quality": quality_figure(analysis, folder / f"{slug}_quality.png"),
        "filter_sweep": filter_sweep_figure(analysis, folder / f"{slug}_filter_sweep.png"),
    }
    return {key: str(value) for key, value in out.items() if value is not None}
