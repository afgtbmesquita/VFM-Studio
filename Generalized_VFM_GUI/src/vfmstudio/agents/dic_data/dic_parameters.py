"""DIC processing parameters that the raw export does not record, inferred from the data.

The raw Vic-3D facet export carries the facet step (pixel grid) but not the subset
size or the strain/curvature filter. Both leave deterministic fingerprints:

* the exported curvature (strain) is a local polynomial fit of W (U, V) over the
  filter window, so re-fitting the exported displacements with candidate windows
  and finding the one that reproduces the exported field identifies the filter;
* neighbouring facets share pixels of their subsets, so displacement noise is
  spatially correlated over roughly one subset width.

The same window fit, swept over sizes, gives noise, signal and smoothing misfit as
a function of the curvature filter size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.signal import fftconvolve

from vfmstudio.agents.dic_data import noise_model as nm

# Basis order of the local quadratic fit: 1, y, x, y^2, xy, x^2 (as noise_model).
_N_TERMS = 6
_MIN_POINTS = 2 * _N_TERMS
SWEEP_HALVES = tuple(range(2, 16))
EDGE_WEIGHTS = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02)
NOISE_MODEL_FILTER = 2 * nm.FILTER_HALF_WIDTH_STEPS + 1
MIN_COMPARE_FACETS = 200
REPRODUCED_CLOSELY = 0.35
REPRODUCED_ROUGHLY = 0.7
AUTOCORR_MAX_LAG = 10


def _offsets(half: int) -> np.ndarray:
    return np.arange(-half, half + 1, dtype=float) / half


def window_weights(half: int, edge_weight: float = 1.0) -> np.ndarray:
    """1-D weights; ``edge_weight`` < 1 is a Gaussian whose value at the window edge is that fraction."""
    u = _offsets(half)
    if edge_weight >= 1.0:
        return np.ones_like(u)
    sigma = 1.0 / math.sqrt(-2.0 * math.log(edge_weight))
    return np.exp(-0.5 * (u / sigma) ** 2)


def _basis(half: int) -> list[np.ndarray]:
    u = _offsets(half)
    gx, gy = np.meshgrid(u, u, indexing="xy")
    return [np.ones_like(gx), gy, gx, gy**2, gx * gy, gx**2]


def _correlate(images: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return fftconvolve(images, kernel[::-1, ::-1][(None,) * (images.ndim - 2)], mode="same",
                       axes=(-2, -1))


@dataclass
class WindowFit:
    """Weighted local quadratic fit over a (2·half+1)² facet window, truncated at the mask."""

    mask: np.ndarray
    half: int
    edge_weight: float
    pitch_x_mm: float
    pitch_y_mm: float
    kernels: list[np.ndarray]
    weight2d: np.ndarray
    inverse: np.ndarray
    usable: np.ndarray

    @classmethod
    def build(cls, mask: np.ndarray, half: int, pitch_x_mm: float, pitch_y_mm: float,
              edge_weight: float = 1.0) -> "WindowFit":
        mask = np.asarray(mask, dtype=bool)
        w1 = window_weights(half, edge_weight)
        weight2d = np.outer(w1, w1)
        basis = _basis(half)
        kernels = [weight2d * b for b in basis]
        rows, cols = np.nonzero(mask)
        normal = np.empty((rows.size, _N_TERMS, _N_TERMS))
        m = mask.astype(float)
        for p in range(_N_TERMS):
            for q in range(p, _N_TERMS):
                value = _correlate(m, weight2d * basis[p] * basis[q])[rows, cols]
                normal[:, p, q] = value
                normal[:, q, p] = value
        count = _correlate(m, (np.abs(weight2d) > 0).astype(float))[rows, cols]
        usable = count >= _MIN_POINTS - 0.5
        usable &= np.linalg.cond(normal) < 1e10
        inverse = np.full_like(normal, np.nan)
        if usable.any():
            inverse[usable] = np.linalg.inv(normal[usable])
        return cls(mask, half, edge_weight, pitch_x_mm, pitch_y_mm, kernels, weight2d, inverse, usable)

    def coefficients(self, images: np.ndarray, misfit: bool = False) -> dict[str, np.ndarray]:
        """Physical derivatives at every mask facet for ``images`` (..., H, W) on the mask.

        Returns ``value`` (smoothed field), ``dx, dy`` (first derivatives), ``dxx, dyy, dxy``
        (second derivatives) as arrays (..., n_mask_facets); with ``misfit`` also the weighted
        RMS fit residual.
        """
        images = np.where(self.mask, np.nan_to_num(images), 0.0)
        rows, cols = np.nonzero(self.mask)
        rhs = np.stack([_correlate(images, kernel)[..., rows, cols] for kernel in self.kernels], axis=-1)
        coef = np.einsum("npq,...nq->...np", self.inverse, rhs)
        lx = self.half * self.pitch_x_mm
        ly = self.half * self.pitch_y_mm
        out = {
            "value": coef[..., 0],
            "dy": coef[..., 1] / ly,
            "dx": coef[..., 2] / lx,
            "dyy": 2.0 * coef[..., 3] / ly**2,
            "dxy": coef[..., 4] / (lx * ly),
            "dxx": 2.0 * coef[..., 5] / lx**2,
        }
        if misfit:
            total = _correlate(images**2, self.weight2d)[..., rows, cols]
            weight = _correlate(self.mask.astype(float), self.weight2d)[rows, cols]
            residual = np.clip(total - np.sum(coef * rhs, axis=-1), 0.0, None)
            out["rms_misfit"] = np.sqrt(residual / weight)
        return out

    def curvatures(self, w_images: np.ndarray) -> np.ndarray:
        """(κxx, κyy, κxy) = (−∂²W/∂x², −∂²W/∂y², −2·∂²W/∂x∂y), stacked on axis −2."""
        c = self.coefficients(w_images)
        return np.stack([-c["dxx"], -c["dyy"], -2.0 * c["dxy"]], axis=-2)

    def strains(self, u_images: np.ndarray, v_images: np.ndarray) -> np.ndarray:
        """Small-strain (εxx, εyy, γxy) from U and V, stacked on axis −2."""
        cu = self.coefficients(u_images)
        cv = self.coefficients(v_images)
        return np.stack([cu["dx"], cv["dy"], cu["dy"] + cv["dx"]], axis=-2)

    def stencil_norms(self) -> np.ndarray:
        """‖w_c‖ (unit-variance white input) for κxx, κyy, κxy at every mask facet, (3, n)."""
        lx = self.half * self.pitch_x_mm
        ly = self.half * self.pitch_y_mm
        # For weighted least squares the coefficient variance is (N⁻¹ M N⁻¹), M built with w².
        rows, cols = np.nonzero(self.mask)
        basis = _basis(self.half)
        m = self.mask.astype(float)
        w2 = self.weight2d**2
        second = np.empty_like(self.inverse)
        for p in range(_N_TERMS):
            for q in range(p, _N_TERMS):
                value = _correlate(m, w2 * basis[p] * basis[q])[rows, cols]
                second[:, p, q] = value
                second[:, q, p] = value
        cov = np.einsum("npq,nqr,nrs->nps", self.inverse, second, self.inverse)
        return np.stack([
            2.0 * np.sqrt(np.abs(cov[:, 5, 5])) / lx**2,
            2.0 * np.sqrt(np.abs(cov[:, 3, 3])) / ly**2,
            2.0 * np.sqrt(np.abs(cov[:, 4, 4])) / (lx * ly),
        ])


def gaussian_sigma_facets(half: int, edge_weight: float) -> Optional[float]:
    if edge_weight >= 1.0:
        return None
    return half / math.sqrt(-2.0 * math.log(edge_weight))


def weighting_label(edge_weight: float) -> str:
    if edge_weight >= 1.0:
        return "uniform"
    return f"gaussian, edge weight {edge_weight:.0%} of centre"


def _mismatch(ours: np.ndarray, vic: np.ndarray, select: np.ndarray) -> np.ndarray:
    """RMS(refit − exported) / SD(exported) per component on ``select``."""
    out = np.full(3, np.nan)
    for i in range(3):
        a, b = ours[i, select], vic[i, select]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() >= MIN_COMPARE_FACETS and np.std(b[ok]) > 0:
            out[i] = np.sqrt(np.mean((a[ok] - b[ok]) ** 2)) / np.std(b[ok])
    return out


def _autocorrelation(images: np.ndarray, detrend: WindowFit, max_lag: int) -> dict[str, np.ndarray]:
    """Normalised spatial autocorrelation along X (columns) and Y (rows) after a wide local-quadratic high-pass."""
    mask = detrend.mask
    smooth = detrend.coefficients(images)["value"]
    sums = {"x": np.zeros((max_lag + 1, 3)), "y": np.zeros((max_lag + 1, 3))}
    for image, trend in zip(images, smooth):
        z = np.full(mask.shape, np.nan)
        z[mask] = np.where(detrend.usable, image[mask] - trend, np.nan)
        for axis, key in ((1, "x"), (0, "y")):
            for lag in range(max_lag + 1):
                if lag >= z.shape[axis]:
                    break
                a = z if lag == 0 else (z[:, :-lag] if axis == 1 else z[:-lag, :])
                b = z if lag == 0 else (z[:, lag:] if axis == 1 else z[lag:, :])
                both = np.isfinite(a) & np.isfinite(b)
                sums[key][lag] += [np.sum(a[both] * b[both]), np.sum(a[both] ** 2), np.sum(b[both] ** 2)]
    with np.errstate(all="ignore"):
        return {key: s[:, 0] / np.sqrt(s[:, 1] * s[:, 2]) for key, s in sums.items()}


def _half_lag(curve: np.ndarray) -> Optional[float]:
    for lag in range(1, curve.size):
        if np.isfinite(curve[lag]) and curve[lag] < 0.5:
            above, below = curve[lag - 1], curve[lag]
            return float(lag - 1 + (above - 0.5) / (above - below))
    return None


def _subset_facts(disp_residual_images: np.ndarray, detrend: WindowFit, step_px: int, sig) -> dict:
    """``disp_residual_images`` is (steps, 3 [U, V, W], H, W) of temporal residuals."""
    curves = {}
    halves = {}
    for index, name in enumerate(("U", "V", "W")):
        corr = _autocorrelation(disp_residual_images[:, index], detrend, AUTOCORR_MAX_LAG)
        for axis in ("x", "y"):
            curves[f"{name}_along_{axis}"] = [sig(value, 3) for value in corr[axis]]
            halves[f"{name}_along_{axis}"] = _half_lag(corr[axis])
    in_plane = [value for key, value in halves.items() if key[0] in "UV" and value is not None]
    estimate = 2.0 * float(np.median(in_plane)) * step_px if in_plane else None
    return {
        "method": (
            "spatial autocorrelation of the per-facet temporal residual of U, V, W (the same "
            "load-polynomial residual the noise estimator uses), high-passed per step by subtracting "
            f"a local quadratic over {2 * SWEEP_HALVES[-1] + 1}×{2 * SWEEP_HALVES[-1] + 1} facets "
            "(removes rigid-body jitter and other smooth motion the load polynomial cannot follow). "
            "Overlapping subsets share pixels, so their displacement errors correlate roughly in "
            "proportion to the shared area; for a uniformly weighted square subset that falls to 0.5 "
            "at a separation of half a subset, so subset ≈ 2 × (half-correlation lag) × facet step"
        ),
        "lags_facets": list(range(AUTOCORR_MAX_LAG + 1)),
        "autocorrelation": curves,
        "half_correlation_lag_facets": {key: sig(value, 3) for key, value in halves.items()},
        "subset_size_estimate_px": sig(estimate, 3),
        "estimate_from": "median of the U and V half-correlation lags along X and Y",
        "caveat": (
            "an estimate, not a recorded setting: subset weighting, interpolation and correlated "
            "speckle or lighting changes shift it by about a facet step either way"
        ),
    }


def dic_parameter_facts(
    mask: np.ndarray,
    pitch_x_mm: float,
    pitch_y_mm: float,
    step_px: int,
    disp_steps: np.ndarray,
    vic_dk_steps: np.ndarray,
    final: int,
    dominant: int,
    load_series: Optional[np.ndarray],
    min_steps_for_noise: int,
) -> dict:
    """Recorded and inferred DIC parameters, and noise/signal against curvature filter size.

    ``disp_steps``: (steps, 3, H, W) images of ΔU, ΔV, ΔW on ``mask``.
    ``vic_dk_steps``: (steps, 3, n) exported Δκ in ``mask`` raster order (``np.nonzero``).
    """
    from vfmstudio.agents.dic_data.analysis import _median, _sig, temporal_residual_sigma, temporal_residuals

    distance = nm.edge_distance(mask)[mask]
    mm_per_px = (pitch_x_mm / step_px, pitch_y_mm / step_px) if step_px else (None, None)
    recorded = {
        "facet_step_px": int(step_px),
        "facet_pitch_mm": {"x": _sig(pitch_x_mm), "y": _sig(pitch_y_mm)},
        "image_scale_mm_per_px": {"x": _sig(mm_per_px[0]), "y": _sig(mm_per_px[1])},
        "not_in_export": "subset size, strain/curvature filter size and weighting, correlation "
        "criterion and interpolation are not written to the raw facet export; they are inferred below",
    }
    facts: dict = {"recorded_in_export": recorded}

    noise_ok = load_series is not None and disp_steps.shape[0] >= min_steps_for_noise
    w_residual = None
    if noise_ok:
        try:
            residual, _dof = temporal_residuals(load_series, disp_steps[..., mask])
            images = np.full(disp_steps.shape, np.nan)
            images[..., mask] = residual
            detrend = WindowFit.build(mask, SWEEP_HALVES[-1], pitch_x_mm, pitch_y_mm)
            facts["subset"] = _subset_facts(images, detrend, step_px, _sig)
            w_residual = images[:, 2]
        except ValueError as exc:
            noise_ok = False
            facts["subset"] = {"available": False, "reason": str(exc)}
    else:
        facts["subset"] = {
            "available": False,
            "reason": f"needs at least {min_steps_for_noise} steps with a load variable",
        }

    dw_steps = disp_steps[:, 2]
    dw_final = dw_steps[final]
    vic_final = vic_dk_steps[final]
    table: dict[float, dict[int, np.ndarray]] = {}
    for edge_weight in EDGE_WEIGHTS:
        table[edge_weight] = {}
        for half in SWEEP_HALVES:
            select = distance >= half
            if select.sum() < MIN_COMPARE_FACETS:
                continue
            fit = WindowFit.build(mask, half, pitch_x_mm, pitch_y_mm, edge_weight)
            table[edge_weight][half] = _mismatch(fit.curvatures(dw_final), vic_final, select & fit.usable)
    ranked = sorted(
        ((float(np.nanmean(m)), ew, half) for ew, row in table.items() for half, m in row.items()
         if np.isfinite(m).any()),
    )
    if not ranked:
        facts["curvature_filter"] = {
            "available": False,
            "reason": f"needs a non-zero final-step Δκ and at least {MIN_COMPARE_FACETS} facets with a "
            "full window for some candidate size",
        }
        return facts
    best_mean, best_ew, best_half = ranked[0]
    best = table[best_ew][best_half]

    def describe(ew: float, half: int) -> dict:
        sigma = gaussian_sigma_facets(half, ew)
        return {
            "filter_size_facets": 2 * half + 1,
            "weighting": weighting_label(ew),
            "edge_weight": ew,
            "gaussian_sigma_facets": _sig(sigma, 3),
            "gaussian_sigma_mm": _sig(sigma * math.sqrt(pitch_x_mm * pitch_y_mm), 3) if sigma else None,
            "window_mm": {"x": _sig((2 * half + 1) * pitch_x_mm, 3), "y": _sig((2 * half + 1) * pitch_y_mm, 3)},
            "mismatch_mean": _sig(float(np.nanmean(table[ew][half])), 3),
        }

    uniform = [(float(np.nanmean(m)), h) for h, m in table[1.0].items() if np.isfinite(m).any()]
    model_half = nm.FILTER_HALF_WIDTH_STEPS
    if best_mean <= REPRODUCED_CLOSELY:
        verdict = "reproduced closely"
    elif best_mean <= REPRODUCED_ROUGHLY:
        verdict = "reproduced roughly: the exported curvature follows a smoothing of this width, but "
        verdict += "Vic-3D's computation differs from a weighted quadratic fit of ΔW in detail"
    else:
        verdict = "not reproduced: no window in the candidate family matches the exported curvature"
    sizes = [2 * h + 1 for h in SWEEP_HALVES]
    facts["curvature_filter"] = {
        "available": True,
        "method": (
            "the final-step ΔW is re-fitted with a local quadratic over every candidate window "
            f"({sizes[0]}…{sizes[-1]} facets; uniform and gaussian weights), truncated at the facet "
            "mask, and the resulting Δκ is compared with the exported Δκ on facets whose full "
            "window lies inside the mask. mismatch = RMS(refit − exported) / SD(exported); 0 is an "
            "exact reproduction, 1 means the refit explains none of the exported scatter"
        ),
        "best": {**describe(best_ew, best_half),
                 "mismatch": dict(zip(nm.COMPONENTS, [_sig(v, 3) for v in best])),
                 "explained_variance_fraction": _sig(1.0 - best_mean**2, 3)},
        "next_best": [describe(ew, half) for _, ew, half in ranked[1:4]],
        "best_uniform": describe(1.0, min(uniform)[1]) if uniform else None,
        "noise_model_assumption": describe(1.0, model_half) if model_half in table[1.0] else None,
        "verdict": verdict,
        "note": "size and weighting trade off along a valley (a wider window with faster decay "
        "smooths like a narrower uniform one); the gaussian sigma is better determined than the size",
        "mismatch_by_size": {
            "filter_sizes": sizes,
            **{weighting_label(ew): [_sig(np.nanmean(row[h]), 3) if h in row else None for h in SWEEP_HALVES]
               for ew, row in table.items()},
        },
    }

    sweep = []
    gains = {}
    ours_sigma_at_best = None
    for half in SWEEP_HALVES:
        fit = WindowFit.build(mask, half, pitch_x_mm, pitch_y_mm, best_ew)
        interior = (distance >= half) & fit.usable
        if interior.sum() < MIN_COMPARE_FACETS:
            continue
        edge = (distance <= nm.EDGE_DISTANCE_FACETS) & fit.usable
        final_fit = fit.coefficients(dw_final, misfit=True)
        k_final = np.stack([-final_fit["dxx"], -final_fit["dyy"], -2.0 * final_fit["dxy"]])
        signal = float(np.mean(k_final[dominant, interior]))
        gains[half] = np.nanmedian(fit.stencil_norms()[:, interior], axis=1)
        misfit = _median(final_fit["rms_misfit"][interior])
        row = {
            "filter_size_facets": 2 * half + 1,
            "window_mm": {"x": _sig((2 * half + 1) * pitch_x_mm, 3), "y": _sig((2 * half + 1) * pitch_y_mm, 3)},
            "full_window_facet_fraction": _sig(float(np.mean(distance >= half)), 3),
            "unfittable_facet_fraction": _sig(float(np.mean(~fit.usable)), 3),
            "mismatch_to_export": _sig(float(np.nanmean(table[best_ew][half])), 3) if half in table[best_ew] else None,
            "signal_interior_mean_inv_mm": _sig(signal),
            "dW_fit_rms_misfit_um": _sig(1e3 * misfit, 3) if misfit is not None else None,
        }
        if noise_ok:
            noise_misfit = fit.coefficients(w_residual, misfit=True)["rms_misfit"]
            noise_misfit = _median(np.sqrt(np.mean(noise_misfit[:, interior] ** 2, axis=0)))
            row["noise_only_fit_rms_misfit_um"] = _sig(1e3 * noise_misfit, 3) if noise_misfit else None
            row["misfit_to_noise_ratio"] = _sig(misfit / noise_misfit, 3) if misfit and noise_misfit else None
            sigma = temporal_residual_sigma(load_series, fit.curvatures(dw_steps))[0]
            interior_noise = [_median(sigma[i, interior]) for i in range(3)]
            edge_noise = [_median(sigma[i, edge]) if edge.any() else None for i in range(3)]
            row["noise_interior_median_inv_mm"] = dict(zip(nm.COMPONENTS, [_sig(v) for v in interior_noise]))
            row["noise_all_facets_median_inv_mm"] = dict(
                zip(nm.COMPONENTS, [_sig(_median(sigma[i, fit.usable])) for i in range(3)])
            )
            row["edge_amplification"] = dict(zip(nm.COMPONENTS, [
                _sig(e / n, 3) if e and n else None for e, n in zip(edge_noise, interior_noise)
            ]))
            level = interior_noise[dominant]
            row["snr_interior"] = _sig(abs(signal) / level, 3) if level else None
            if half == best_half:
                ours_sigma_at_best = interior_noise
        sweep.append(row)
    base_ratio = next((row["misfit_to_noise_ratio"] for row in sweep if row.get("misfit_to_noise_ratio")), None)
    for row in sweep:
        half = (row["filter_size_facets"] - 1) // 2
        if base_ratio and row.get("misfit_to_noise_ratio"):
            row["smoothing_bias_index"] = _sig(row["misfit_to_noise_ratio"] / base_ratio, 3)
        if best_half in gains:
            row["white_noise_sigma_relative_to_identified"] = _sig(
                float(np.mean(gains[half] / gains[best_half])), 3
            )

    def exponent(key: str) -> Optional[float]:
        pairs = []
        for row in sweep:
            value = row.get(key)
            if isinstance(value, dict):
                value = float(np.mean([v for v in value.values() if v])) if all(value.values()) else None
            if value:
                pairs.append((math.log(row["filter_size_facets"]), math.log(value)))
        if len(pairs) < 3:
            return None
        x, y = np.array(pairs).T
        return _sig(float(np.polyfit(x, y, 1)[0]), 3)

    export_noise = None
    if noise_ok and ours_sigma_at_best is not None:
        interior_best = distance >= best_half
        vic_sigma = temporal_residual_sigma(load_series, vic_dk_steps)[0]
        export_noise = [_median(vic_sigma[i, interior_best]) for i in range(3)]
    facts["curvature_filter_sweep"] = {
        "weighting": weighting_label(best_ew),
        "identified_filter_size_facets": 2 * best_half + 1,
        "dominant_component": nm.COMPONENTS[dominant],
        "final_step_analysed": final + 1,
        "noise_available": noise_ok,
        "definitions": (
            "each row re-fits ΔW with that filter size (weighting fixed at the identified one). "
            "noise: same temporal-residual estimator as the noise section, on the refitted curvature; "
            "interior = facets whose full window lies inside the mask; edge = ≤ "
            f"{nm.EDGE_DISTANCE_FACETS:g} facets from a boundary. signal: interior mean Δκ of the "
            "dominant component at the final step; snr_interior = |signal| / interior noise. "
            "dW_fit_rms_misfit_um: median RMS residual of the local quadratic fit of the final-step ΔW; "
            "noise_only_fit_rms_misfit_um: the same fit applied to the temporal-residual W fields "
            "(noise alone); misfit_to_noise_ratio = their quotient (its absolute level depends on how "
            "the frames share noise, so only its change with size matters); smoothing_bias_index = that "
            "ratio divided by its value at the smallest size: 1 means the larger window follows ΔW as "
            "well as the smallest, a rise marks smoothing bias (the window is wider than the deformation "
            "features). white_noise_sigma_relative_to_identified: noise ratio expected if W noise "
            "were uncorrelated between facets"
        ),
        "noise_vs_size_exponent_measured": exponent("noise_interior_median_inv_mm") if noise_ok else None,
        "noise_vs_size_exponent_white_noise": exponent("white_noise_sigma_relative_to_identified"),
        "refit_to_export_interior_noise_ratio": dict(zip(nm.COMPONENTS, [
            _sig(o / e, 3) if o and e else None for o, e in zip(ours_sigma_at_best, export_noise)
        ])) if export_noise else None,
        "rows": sweep,
    }
    return facts
