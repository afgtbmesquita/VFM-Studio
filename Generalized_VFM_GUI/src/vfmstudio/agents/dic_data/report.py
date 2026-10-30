"""Deterministic part of the data agent's report, rendered from facts.json alone.

Everything here is transcription, arithmetic or a fixed threshold rule, so it is
done by code: exact, reproducible and free. What needs judgement (describing the
deformation pattern from the figures, weighing what matters, the three key
points, follow-up questions) is left to the language model.

Thresholds in :data:`CHECK_THRESHOLDS` are reporting thresholds only. They decide
what is listed, never whether a test is acceptable.
"""

from __future__ import annotations

import math
from typing import Any, Optional

COMPONENTS = ("kxx", "kyy", "kxy")
LABEL = {"kxx": "κxx", "kyy": "κyy", "kxy": "κxy"}

CHECK_THRESHOLDS = {
    "noise_ratio_high": 1.5,
    "noise_ratio_low": 0.67,
    "snr_low": 3.0,
    "snr_below_noise": 1.0,
    "residual_dof_low": 3,
    "linearity_r2_low": 0.95,
    "secondary_ratio": 0.25,
    "map_correlation_low": 0.5,
    "facets_lost_fraction": 0.01,
    "vic_sigma_growth": 2.0,
    "filter_stencil_mismatch_gap": 0.2,
    "refit_noise_ratio_low": 0.8,
    "refit_noise_ratio_high": 1.25,
    "smoothing_bias_high": 1.1,
    "snr_target": 10.0,
    "full_window_fraction_low": 0.5,
}


def _num(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    if isinstance(value, int):
        return f"{value:,}"
    if value == 0:
        return "0"
    return f"{value:.{digits}g}"


def _signed(value: Any, digits: int = 3) -> str:
    return "—" if value is None else f"{value:+.{digits}g}"


def _ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return a / b if a is not None and b else None


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


# -- derived numbers --------------------------------------------------------------


def summarise(test: dict) -> dict:
    """Numbers derived from one test's facts (ranges, ratios, evolutions)."""
    deformation = test["deformation"]
    steps = deformation["steps"]
    noise = test["noise"]
    quality = test["read_quality"]
    times = [row["time_s"] for row in steps if row.get("time_s") is not None]
    names = sorted({name for row in steps for name in row["machine"]})
    channels = {}
    for name in names:
        values = [row["machine"].get(name) for row in steps]
        if any(value is None for value in values):
            continue
        channels[name] = {
            "first": values[0],
            "last": values[-1],
            "min": min(values),
            "max": max(values),
        }
    interior_ratio = {}
    if noise.get("available"):
        for comp, stats in noise["components"].items():
            interior_ratio[comp] = _ratio(
                stats.get("measured_interior_median_inv_mm"), stats.get("model_interior_median_inv_mm")
            )
    per_step = quality["per_step"]
    zone_ratios = [zone["noise_to_model_ratio"] for zone in quality["zones"] if zone.get("noise_to_model_ratio")]
    worst_dropout = max(quality["zones"], key=lambda zone: zone["dropout_fraction"] or 0, default=None)
    return {
        "duration_s": (times[-1] - times[0]) if len(times) >= 2 else None,
        "machine_channel_ranges": channels,
        "noise_interior_measured_to_model": interior_ratio,
        "vic_sigma_first_last_px": [per_step[0]["vic_sigma_median_px"], per_step[-1]["vic_sigma_median_px"]],
        "vic_sigma_p95_first_last_px": [per_step[0]["vic_sigma_p95_px"], per_step[-1]["vic_sigma_p95_px"]],
        "sigma_z_first_last_mm": [per_step[0]["sigma_z_median_mm"], per_step[-1]["sigma_z_median_mm"]],
        "zone_noise_ratio_range": [min(zone_ratios), max(zone_ratios)] if zone_ratios else None,
        "worst_zone_dropout": (
            {"zone": worst_dropout["zone"], "fraction": worst_dropout["dropout_fraction"]}
            if worst_dropout and worst_dropout["dropout_fraction"]
            else None
        ),
        "facets_lost_fraction": (
            quality["facets_lost_in_some_step"] / quality["facets_union"] if quality["facets_union"] else None
        ),
        "dic_parameters": summarise_dic_parameters(test),
    }


def _first_size(rows: list[dict], key: str, test, default=None) -> Optional[int]:
    for row in rows:
        value = row.get(key)
        if value is not None and test(value):
            return row["filter_size_facets"]
    return default


def summarise_dic_parameters(test: dict) -> dict:
    """Threshold crossings and headline values of the curvature filter sweep."""
    params = test.get("dic_parameters") or {}
    filt = params.get("curvature_filter") or {}
    sweep = params.get("curvature_filter_sweep") or {}
    rows = sweep.get("rows") or []
    t = CHECK_THRESHOLDS
    out: dict = {
        "subset_size_estimate_px": (params.get("subset") or {}).get("subset_size_estimate_px"),
        "inferred_filter_size_facets": None,
        "inferred_filter_weighting": None,
        "inferred_filter_mismatch": None,
        "noise_model_stencil_mismatch": None,
        "noise_model_stencil_mismatch_minus_inferred": None,
    }
    if not filt.get("available"):
        return out
    best = filt["best"]
    model = filt.get("noise_model_assumption") or {}
    out.update({
        "inferred_filter_size_facets": best["filter_size_facets"],
        "inferred_filter_weighting": best["weighting"],
        "inferred_filter_mismatch": best["mismatch_mean"],
        "noise_model_stencil_mismatch": model.get("mismatch_mean"),
        "noise_model_stencil_mismatch_minus_inferred": (
            round(model["mismatch_mean"] - best["mismatch_mean"], 3)
            if model.get("mismatch_mean") is not None and best["mismatch_mean"] is not None else None
        ),
    })
    at = next((row for row in rows if row["filter_size_facets"] == sweep.get("identified_filter_size_facets")), None)
    if at:
        out["at_inferred_filter"] = {
            "snr_interior": at.get("snr_interior"),
            "noise_interior_median_inv_mm": at.get("noise_interior_median_inv_mm"),
            "full_window_facet_fraction": at.get("full_window_facet_fraction"),
            "smoothing_bias_index": at.get("smoothing_bias_index"),
        }
    out["smallest_filter_with_snr_at_least"] = {
        f"{t['snr_low']:g}": _first_size(rows, "snr_interior", lambda v: v >= t["snr_low"]),
        f"{t['snr_target']:g}": _first_size(rows, "snr_interior", lambda v: v >= t["snr_target"]),
    }
    out["smallest_filter_with_smoothing_bias_above"] = {
        f"{t['smoothing_bias_high']:g}": _first_size(
            rows, "smoothing_bias_index", lambda v: v > t["smoothing_bias_high"]
        ),
    }
    out["smallest_filter_with_full_window_fraction_below"] = {
        f"{t['full_window_fraction_low']:g}": _first_size(
            rows, "full_window_facet_fraction", lambda v: v < t["full_window_fraction_low"]
        ),
    }
    if rows:
        out["sweep_range"] = {
            "filter_sizes_facets": [rows[0]["filter_size_facets"], rows[-1]["filter_size_facets"]],
            "snr_interior": [rows[0].get("snr_interior"), rows[-1].get("snr_interior")],
            "full_window_facet_fraction": [rows[0]["full_window_facet_fraction"],
                                           rows[-1]["full_window_facet_fraction"]],
            "smoothing_bias_index": [rows[0].get("smoothing_bias_index"), rows[-1].get("smoothing_bias_index")],
        }
    return out


def automatic_checks(test: dict, summary: Optional[dict] = None) -> list[dict]:
    """Fixed-threshold observations. Each states a measured value against a stated threshold."""
    summary = summary or summarise(test)
    t = CHECK_THRESHOLDS
    deformation = test["deformation"]
    noise = test["noise"]
    checks: list[dict] = []

    def add(check: str, observation: str, value: Any = None, threshold: Any = None) -> None:
        checks.append({"check": check, "observation": observation, "value": value, "threshold": threshold})

    frame = deformation.get("curvature_frame_check", {})
    verdict = str(frame.get("verdict", ""))
    if verdict.startswith("INCONSISTENT"):
        add("curvature frame", verdict, frame.get("relative_mismatch_as_exported"))
    elif verdict.startswith("not testable"):
        add("curvature frame", verdict)

    if not noise.get("available"):
        add("noise", f"noise not measured: {noise.get('reason', 'unknown reason')}")
    else:
        dof = noise.get("residual_dof_per_facet")
        if dof is not None and dof <= t["residual_dof_low"]:
            add("noise", f"only {dof} residual degrees of freedom per facet; per-facet noise values are "
                "weakly constrained", dof, t["residual_dof_low"])
        for comp, ratio in summary["noise_interior_measured_to_model"].items():
            if ratio is not None and ratio > t["noise_ratio_high"]:
                add("noise", f"{LABEL[comp]} interior noise is {ratio:.2f}× the model", ratio, t["noise_ratio_high"])
            elif ratio is not None and ratio < t["noise_ratio_low"]:
                add("noise", f"{LABEL[comp]} interior noise is {ratio:.2f}× the model (below)", ratio, t["noise_ratio_low"])
        for comp, stats in noise["components"].items():
            corr = stats.get("pointwise_map_correlation")
            if corr is not None and corr < t["map_correlation_low"]:
                add("noise", f"{LABEL[comp]} noise map correlates with the model map at only {corr:.2f}",
                    corr, t["map_correlation_low"])
        for comp, snr in (noise.get("signal_to_noise_at_max_load") or {}).items():
            if snr is None:
                continue
            if snr < t["snr_below_noise"]:
                add("signal", f"{LABEL[comp]} signal at the final step is below the noise (SNR {snr:.2f})",
                    snr, t["snr_below_noise"])
            elif snr < t["snr_low"]:
                add("signal", f"{LABEL[comp]} signal-to-noise at the final step is {snr:.2f}", snr, t["snr_low"])

    if deformation.get("dominant_machine_moment") is None:
        add("loads", "no machine moment found for these frames; the load variable is a curvature proxy")
    dominant = deformation["dominant_curvature_component"]
    r2 = (deformation.get("curvature_linearity_vs_load") or {}).get(dominant, {}).get("r2")
    if r2 is not None and r2 < t["linearity_r2_low"]:
        add("deformation", f"dominant {LABEL[dominant]} follows the load with R² = {r2:.3f}", r2, t["linearity_r2_low"])
    for comp, ratio in (deformation.get("secondary_to_dominant_curvature_ratios") or {}).items():
        if ratio is not None and abs(ratio) > t["secondary_ratio"]:
            add("deformation", f"{LABEL[comp]} is {ratio:+.2f}× the dominant {LABEL[dominant]}",
                ratio, t["secondary_ratio"])

    params = test.get("dic_parameters") or {}
    filt = params.get("curvature_filter") or {}
    sweep = params.get("curvature_filter_sweep") or {}
    dic = summary.get("dic_parameters") or {}
    subset = params.get("subset") or {}
    if params and not subset.get("subset_size_estimate_px"):
        add("subset", "subset size not estimated: " + subset.get(
            "reason", "the noise correlation stays above 0.5 out to the largest lag"))
    if params and not filt.get("available"):
        add("curvature filter", f"curvature filter not identified: {filt.get('reason', 'unknown reason')}")
    elif filt:
        best = filt["best"]
        if not filt["verdict"].startswith("reproduced closely"):
            add("curvature filter", f"exported curvature is {filt['verdict'].split(':')[0]} by the best candidate "
                f"({best['filter_size_facets']} facets, {best['weighting']}; mismatch {best['mismatch_mean']:.2f})",
                best["mismatch_mean"])
        gap = dic.get("noise_model_stencil_mismatch_minus_inferred")
        if gap is not None and gap > t["filter_stencil_mismatch_gap"]:
            add("curvature filter",
                f"the noise model's stencil (uniform {filt['noise_model_assumption']['filter_size_facets']} facets) "
                f"reproduces the exported curvature with mismatch {dic['noise_model_stencil_mismatch']:.2f}, against "
                f"{best['mismatch_mean']:.2f} for the inferred filter ({best['filter_size_facets']} facets, "
                f"{best['weighting']})", gap, t["filter_stencil_mismatch_gap"])
        for comp, ratio in (sweep.get("refit_to_export_interior_noise_ratio") or {}).items():
            if ratio is not None and not t["refit_noise_ratio_low"] <= ratio <= t["refit_noise_ratio_high"]:
                add("curvature filter", f"{LABEL[comp]} interior noise of the refit at the inferred filter is "
                    f"{ratio:.2f}× the exported noise", ratio,
                    [t["refit_noise_ratio_low"], t["refit_noise_ratio_high"]])
        at = dic.get("at_inferred_filter") or {}
        bias = at.get("smoothing_bias_index")
        if bias is not None and bias > t["smoothing_bias_high"]:
            add("curvature filter", f"smoothing bias index at the inferred filter is {bias:.2f}", bias,
                t["smoothing_bias_high"])
        coverage = at.get("full_window_facet_fraction")
        if coverage is not None and coverage < t["full_window_fraction_low"]:
            add("curvature filter", f"only {coverage:.0%} of facets have their full filter window inside the "
                "measured region at the inferred filter", coverage, t["full_window_fraction_low"])

    quality = test["read_quality"]
    lost = summary.get("facets_lost_fraction")
    if lost is not None and lost > t["facets_lost_fraction"]:
        add("read quality", f"{lost:.1%} of facets are missing in at least one step", lost, t["facets_lost_fraction"])
    first, last = summary["vic_sigma_first_last_px"]
    if first and last and last / first > t["vic_sigma_growth"]:
        add("read quality", f"Vic-3D correlation sigma grows {last / first:.1f}× from the first to the last step",
            last / first, t["vic_sigma_growth"])
    for zone in quality["zones"]:
        if zone["flags"]:
            add("read quality", f"zone {zone['zone']}: " + "; ".join(zone["flags"]))
    for note in test["specimen"].get("companion_notes", []):
        add("companion files", note)
    return checks


def enrich(test: dict) -> dict:
    """Add the derived summary and automatic checks to one test's facts (in place)."""
    test["summary"] = summarise(test)
    test["automatic_checks"] = automatic_checks(test, test["summary"])
    return test


# -- rendering ------------------------------------------------------------------


def _comparison(tests: list[dict]) -> str:
    rows = []
    for test in tests:
        deformation = test["deformation"]
        dominant = deformation["dominant_curvature_component"]
        means = deformation["interior_mean_dk_at_final_step_inv_mm"]
        ratios = test["summary"]["noise_interior_measured_to_model"]
        verdict = str(deformation.get("curvature_frame_check", {}).get("verdict", "—")).split(":")[0]
        dic = test["summary"].get("dic_parameters") or {}
        size = dic.get("inferred_filter_size_facets")
        rows.append([
            test["test_name"],
            f"{test['specimen']['n_steps']} / {test['read_quality']['facets_union']:,}",
            f"{LABEL[dominant]} {_signed(means[dominant])}",
            verdict,
            " / ".join(_num(ratios.get(comp), 3) for comp in COMPONENTS) if ratios else "—",
            f"{size} ({_num(dic.get('inferred_filter_mismatch'))})" if size else "—",
            _num(dic.get("subset_size_estimate_px")),
            str(len(test["read_quality"]["flagged_zones"])),
            str(len(test["automatic_checks"])),
        ])
    return _table(
        ["Test", "Steps / facets", "Dominant Δκ at final step [1/mm]", "Frame check",
         "Interior noise ÷ model (κxx/κyy/κxy)", "Inferred curvature filter [facets] (mismatch)",
         "Subset ≈ [px]", "Flagged zones", "Automatic checks"],
        rows,
    )


def _specimen_section(test: dict) -> str:
    spec = test["specimen"]
    summary = test["summary"]
    holes = spec["internal_holes"]
    lines = [
        f"- **Folder:** {spec['grandparent_folder']} / {spec['parent_folder']} / {spec['folder']}",
        f"- **Measured region:** {_num(spec['measured_span_mm']['x'])} × {_num(spec['measured_span_mm']['y'])} mm "
        f"({_num(spec['measured_area_mm2'])} mm²); facet pitch {_num(spec['facet_pitch_mm']['x'])} × "
        f"{_num(spec['facet_pitch_mm']['y'])} mm ({spec['facet_step_px']} px step); camera {spec['camera_image_axes']}",
        f"- **Internal holes:** {holes['count']}"
        + (f" ({_num(holes['area_mm2'])} mm²)" if holes["count"] else "")
        + " (cut-outs open to an edge are not counted)",
        f"- **Width along Y (bottom→top) [mm]:** {', '.join(_num(v) for v in spec['width_profile_along_y_mm'])}",
    ]
    duration = summary.get("duration_s")
    load_line = f"- **Steps:** {spec['n_steps']}" + (f" over {_num(duration)} s" if duration is not None else "")
    lines.append(load_line)
    ranges = summary["machine_channel_ranges"]
    moving = []
    for name, span in ranges.items():
        kind = name.split(".")[-1][:1]
        if kind not in ("T", "θ"):
            continue
        moving.append([name, _num(span["first"]), _num(span["last"]), _num(span["min"]), _num(span["max"])])
    if moving:
        lines.append("")
        lines.append(_table(["Machine channel", "Step 1", "Last step", "Min", "Max"], moving))
        lines.append("\nMoments in N·mm, rotations in degrees.")
    else:
        lines.append("- **Machine loads:** none matched to these DIC frames")
    return "\n".join(lines)


def _deformation_section(test: dict) -> str:
    d = test["deformation"]
    dominant = d["dominant_curvature_component"]
    means = d["interior_mean_dk_at_final_step_inv_mm"]
    lines = [
        f"- **Reference / final step:** {d['reference_step']} → {d['final_step_analysed']} ({d['final_step_rule']})",
        f"- **Dominant machine moment / rotation:** {d['dominant_machine_moment'] or '—'} / {d['dominant_machine_rotation'] or '—'}",
        "- **Interior mean Δκ [1/mm]:** " + ", ".join(f"{LABEL[c]} {_signed(means[c])}" for c in COMPONENTS)
        + f" (dominant {LABEL[dominant]})",
    ]
    ratios = d.get("secondary_to_dominant_curvature_ratios") or {}
    if ratios:
        lines.append("- **Secondary ÷ dominant:** " + ", ".join(f"{LABEL[c]} {_signed(v)}" for c, v in ratios.items()))
    linearity = d.get("curvature_linearity_vs_load") or {}
    if linearity:
        lines.append("- **Linearity vs load (R²):** " + ", ".join(f"{LABEL[c]} {_num(v['r2'], 3)}" for c, v in linearity.items()))
    w = d["w_quadratic_fit_final_step"]
    lines.append(
        f"- **ΔW quadratic fit [1/mm]:** κxx {_signed(w['kxx_from_W_inv_mm'])}, κyy {_signed(w['kyy_from_W_inv_mm'])}, "
        f"κxy {_signed(w['kxy_from_W_inv_mm'])}; tilt about X {_num(w['tilt_about_x_deg'])}°, about Y {_num(w['tilt_about_y_deg'])}°"
    )
    frame = d.get("curvature_frame_check", {})
    if frame:
        lines.append(
            f"- **Curvature frame check:** {frame['verdict']} (mismatch as exported "
            f"{_num(frame['relative_mismatch_as_exported'])}, with xx/yy swapped {_num(frame['relative_mismatch_xx_yy_swapped'])})"
        )
    bands = d.get("bands_along_y_at_final_step") or []
    if bands:
        lines.append("")
        lines.append(_table(
            ["Band (along Y)", "Width [mm]", "Δκxx", "Δκyy", "Δκxy", "ΔW [mm]"],
            [[b["band"], _num(b["width_mm"]), _signed(b["mean_dkxx_inv_mm"]), _signed(b["mean_dkyy_inv_mm"]),
              _signed(b["mean_dkxy_inv_mm"]), _signed(b["mean_dW_mm"])] for b in bands],
        ))
    return "\n".join(lines)


def _noise_section(test: dict) -> str:
    noise = test["noise"]
    if not noise.get("available"):
        return f"Not measured: {noise.get('reason', 'unknown reason')}."
    ratios = test["summary"]["noise_interior_measured_to_model"]
    snr = noise.get("signal_to_noise_at_max_load") or {}
    rows = []
    for comp in COMPONENTS:
        s = noise["components"][comp]
        rows.append([
            LABEL[comp],
            _num(s["measured_interior_median_inv_mm"]),
            _num(s["model_interior_median_inv_mm"]),
            _num(ratios.get(comp)),
            _num(s["measured_to_model_ratio"]),
            f"{_num(s['measured_edge_amplification'])} / {_num(s['model_edge_amplification'])}",
            _num(s["legacy_to_measured_ratio"]),
            _num(s["pointwise_map_correlation"], 2),
            _num(snr.get(comp)),
        ])
    table = _table(
        ["", "Measured interior [1/mm]", "Model interior [1/mm]", "Interior ratio", "All-facet ratio",
         "Edge amp. meas / model", "Legacy envelope ÷ measured", "Map corr.", "SNR final step"],
        rows,
    )
    return (
        f"Load variable {noise['load_variable']}; {noise['residual_dof_per_facet']} residual dof per facet "
        "(median-unbiased).\n\n" + table
    )


def _dic_parameters_section(test: dict) -> str:
    params = test.get("dic_parameters")
    if not params:
        return "Not measured."
    rec = params["recorded_in_export"]
    lines = [
        f"- **Recorded in the export:** facet step {rec['facet_step_px']} px; pitch "
        f"{_num(rec['facet_pitch_mm']['x'])} × {_num(rec['facet_pitch_mm']['y'])} mm; image scale "
        f"{_num(rec['image_scale_mm_per_px']['x'])} × {_num(rec['image_scale_mm_per_px']['y'])} mm/px",
    ]
    subset = params["subset"]
    if subset.get("subset_size_estimate_px"):
        lags = subset["half_correlation_lag_facets"]
        lines.append(
            f"- **Subset size (estimated):** ≈ {_num(subset['subset_size_estimate_px'])} px from the noise "
            "correlation length (half-correlation lag [facets] "
            + ", ".join(f"{key.replace('_along_', ' along ')} {_num(value)}" for key, value in lags.items())
            + ")"
        )
    else:
        lines.append(f"- **Subset size:** not estimated ({subset.get('reason', 'noise correlation too long')})")
    filt = params["curvature_filter"]
    if not filt.get("available"):
        lines.append(f"- **Curvature filter:** not identified ({filt.get('reason')})")
        return "\n".join(lines)
    best = filt["best"]
    sigma = f", σ {_num(best['gaussian_sigma_mm'])} mm" if best["gaussian_sigma_mm"] else ""
    lines.append(
        f"- **Curvature filter (inferred):** {best['filter_size_facets']} facets ({_num(best['window_mm']['x'])} × "
        f"{_num(best['window_mm']['y'])} mm), {best['weighting']}{sigma}; mismatch to the exported Δκ "
        f"{_num(best['mismatch_mean'])} ({_num(best['explained_variance_fraction'])} of its variance reproduced): "
        f"{filt['verdict']}"
    )
    alternatives = [f"{item['filter_size_facets']} facets {item['weighting']} ({_num(item['mismatch_mean'])})"
                    for item in filt["next_best"]]
    if filt.get("best_uniform"):
        alternatives.append(f"best uniform {filt['best_uniform']['filter_size_facets']} facets "
                            f"({_num(filt['best_uniform']['mismatch_mean'])})")
    if filt.get("noise_model_assumption"):
        alternatives.append(f"noise-model stencil, uniform {filt['noise_model_assumption']['filter_size_facets']} "
                            f"facets ({_num(filt['noise_model_assumption']['mismatch_mean'])})")
    lines.append("- **Other candidates (mismatch):** " + "; ".join(alternatives))
    sweep = params["curvature_filter_sweep"]
    ratio = sweep.get("refit_to_export_interior_noise_ratio")
    if ratio:
        lines.append("- **Refit ÷ exported interior noise at the inferred filter:** "
                     + ", ".join(f"{LABEL[c]} {_num(v)}" for c, v in ratio.items()))
    if sweep.get("noise_vs_size_exponent_measured") is not None:
        lines.append(
            f"- **Noise ∝ size^p:** measured p = {_num(sweep['noise_vs_size_exponent_measured'])}, "
            f"uncorrelated W noise would give {_num(sweep['noise_vs_size_exponent_white_noise'])}"
        )
    dic = test["summary"].get("dic_parameters") or {}

    def crossing(key: str, phrase: str) -> list[str]:
        out = []
        for threshold, size in (dic.get(key) or {}).items():
            out.append(f"{phrase} {threshold}: " + (f"{size} facets" if size else "none in the sweep"))
        return out

    crossings = (crossing("smallest_filter_with_snr_at_least", "SNR ≥")
                 + crossing("smallest_filter_with_smoothing_bias_above", "smoothing bias index >")
                 + crossing("smallest_filter_with_full_window_fraction_below", "full-window facet fraction <"))
    if crossings:
        lines.append("- **Smallest filter size reaching:** " + "; ".join(crossings))
    rows = []
    for row in sweep["rows"]:
        noise = row.get("noise_interior_median_inv_mm") or {}
        mark = " (inferred)" if row["filter_size_facets"] == sweep["identified_filter_size_facets"] else ""
        rows.append([
            f"{row['filter_size_facets']}{mark}",
            _num(row["window_mm"]["x"]),
            _num(row["full_window_facet_fraction"], 2),
            " / ".join(_num(noise.get(c)) for c in COMPONENTS),
            _num(row.get("snr_interior")),
            _num(row.get("smoothing_bias_index")),
            _num(row.get("mismatch_to_export")),
        ])
    lines += ["", f"Curvature filter sweep ({sweep['weighting']}; signal = interior mean "
              f"Δ{LABEL[sweep['dominant_component']]} at step {sweep['final_step_analysed']}):", "",
              _table(["Filter [facets]", "Window x [mm]", "Full-window facets", "Interior noise κxx / κyy / κxy [1/mm]",
                      "SNR", "Smoothing bias index", "Mismatch to export"], rows)]
    return "\n".join(lines)


def _quality_section(test: dict) -> str:
    q = test["read_quality"]
    s = test["summary"]
    lines = [
        f"- **Facets:** {q['facets_union']:,} read in at least one step, {q['facets_in_every_step']:,} in every step",
        f"- **Vic-3D sigma median [px]:** {_num(s['vic_sigma_first_last_px'][0])} at step 1 → "
        f"{_num(s['vic_sigma_first_last_px'][1])} at the last step (p95 {_num(s['vic_sigma_p95_first_last_px'][0])} → "
        f"{_num(s['vic_sigma_p95_first_last_px'][1])}); σZ median {_num(s['sigma_z_first_last_mm'][0])} → "
        f"{_num(s['sigma_z_first_last_mm'][1])} mm",
    ]
    if s["zone_noise_ratio_range"]:
        low, high = s["zone_noise_ratio_range"]
        lines.append(f"- **Zone noise ÷ model:** {_num(low)} to {_num(high)} ({q['zone_layout']})")
    flagged = [zone for zone in q["zones"] if zone["flags"]]
    if flagged:
        lines.append("")
        lines.append(_table(
            ["Poorly read zone", "Facets", "Dropout", "Vic sigma [px]", "Noise ÷ model", "Flags"],
            [[z["zone"], f"{z['facets']:,}", f"{(z['dropout_fraction'] or 0):.1%}", _num(z["vic_sigma_median_px"]),
              _num(z["noise_to_model_ratio"]), "; ".join(z["flags"])] for z in flagged],
        ))
    else:
        lines.append("- **Poorly read zones:** none crosses the dropout, correlation-sigma or noise thresholds")
    return "\n".join(lines)


def _checks_section(test: dict) -> str:
    checks = test["automatic_checks"]
    if not checks:
        return "None of the reporting thresholds is crossed."
    return "\n".join(f"- **{c['check']}:** {c['observation']}" for c in checks)


def render_report(facts: dict) -> str:
    """Markdown report of everything measurable. Expects tests enriched by :func:`enrich`."""
    tests = facts["tests"]
    fmt = facts["format_check"]
    parts = [
        "# DIC data report (measured, scripted)",
        f"Generated {facts['generated_at']}. Every number below is computed by VFM Studio from the raw files; "
        "thresholds only decide what is listed, not whether a test is acceptable.",
        "## Overview",
        _comparison(tests),
        "## Dataset and format check",
        "\n".join([
            f"- **Files:** {fmt['files_checked']} raw Vic-3D exports in {len(tests)} test(s); all match the "
            "expected 33-column format",
            f"- **Cross-file checks:** {fmt['cross_file_checks']}",
            f"- **Conversion:** {facts['noise_model']['curvature_convention']}; ε = (exx, eyy, 2·exy)",
            f"- **Increments:** {facts['conventions']['increments']}",
        ]),
    ]
    for test in tests:
        parts += [
            f"## {test['test_name']}",
            "### Specimen and test",
            _specimen_section(test),
            "### Deformation (measured)",
            _deformation_section(test),
            "### Noise vs model",
            _noise_section(test),
            "### DIC processing parameters and curvature filter",
            _dic_parameters_section(test),
            "### Read quality",
            _quality_section(test),
            "### Automatic checks",
            _checks_section(test),
        ]
    model = facts["noise_model"]
    parts += [
        "## Noise model used",
        f"{model['equation']}. Interior floor [1/mm]: "
        + ", ".join(f"{LABEL[c]} {_num(v)}" for c, v in model["interior_floor_inv_mm"].items())
        + f". Calibrated on {', '.join(model['calibrated_on'])}. Known limits: {'; '.join(model['known_limits'])}.",
    ]
    return "\n\n".join(parts) + "\n"
