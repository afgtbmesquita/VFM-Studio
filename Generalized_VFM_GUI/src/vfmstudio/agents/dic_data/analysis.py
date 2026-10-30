"""Raw Vic-3D exports -> solver inputs, plus deformation, noise and read-quality facts.

Raw files are only ever read. Solver payloads are built in memory with the same
conversion the Import-steps dialog uses (:mod:`vfmstudio.io.csv_import`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import chi2

from vfmstudio.agents.dic_data import dic_parameters as dp
from vfmstudio.agents.dic_data import noise_model as nm
from vfmstudio.agents.dic_data.intake import FolderCompanions, TestGroup, frame_match
from vfmstudio.agents.dic_data.raw_format import RAW_DIC_COLUMNS, RawFileInfo
from vfmstudio.io.csv_import import (
    KEY_STRIDE,
    VIC3D_CONVENTION_NOTE,
    _aligned_frames,
    _convert_kinematics,
    default_import_name,
    long_path,
)

LOAD_POLYNOMIAL_DEGREE = 2
MIN_STEPS_FOR_NOISE = LOAD_POLYNOMIAL_DEGREE + 3
MIN_FACET_OVERLAP = 0.5
RADIAL_MAX_FACETS = 12
CURVATURE = (("kxx", "cxx", -2.0), ("kyy", "cyy", -2.0), ("kxy", "cxy", 2.0))
STRAIN = (("exx", "exx", 1.0), ("eyy", "eyy", 1.0), ("gxy", "exy", 2.0))
DROPOUT_FLAG = 0.02
VIC_SIGMA_FLAG = 1.5
NOISE_RATIO_FLAG = 1.5

FIELD_DOC = {
    "nodes": ("mm", "facet coordinates [X, Y] (Layout A)"),
    "X": ("mm", "facet X coordinate"),
    "Y": ("mm", "facet Y coordinate"),
    "Z": ("mm", "facet Z coordinate"),
    "U": ("mm", "displacement along X"),
    "V": ("mm", "displacement along Y"),
    "W": ("mm", "out-of-plane displacement"),
    "SK1grid": ("1/mm", "κxx = −2·cxx"),
    "SK2grid": ("1/mm", "κyy = −2·cyy"),
    "SK3grid": ("1/mm", "κxy = +2·cxy (engineering twist)"),
    "SE1grid": ("–", "εxx = exx"),
    "SE2grid": ("–", "εyy = eyy"),
    "SE3grid": ("–", "γxy = 2·exy (engineering shear)"),
    "F": ("N·mm", "machine moments at this frame"),
    "F_labels": ("", "names of the F entries"),
}


class DataRefused(Exception):
    """The input does not match the raw Vic-3D export; nothing was analysed."""

    def __init__(self, reasons: Sequence[str]) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = list(reasons)


def _sig(value, digits: int = 4):
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    if value == 0.0:
        return 0.0
    return float(f"{value:.{digits}g}")


def _sig_list(values, digits: int = 4) -> list:
    return [_sig(item, digits) for item in np.asarray(values, dtype=float).ravel()]


def _median(values) -> Optional[float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else None


@dataclass
class Grid:
    """Facet lattice oriented so that columns run along +X and rows along +Y."""

    rows: np.ndarray
    cols: np.ndarray
    shape: tuple[int, int]
    pitch_x_mm: float
    pitch_y_mm: float
    step_px: int
    extent: tuple[float, float, float, float]
    swapped_axes: bool
    image_axes: str = ""

    def image(self, values: np.ndarray, index: Optional[np.ndarray] = None) -> np.ndarray:
        out = np.full(self.shape, np.nan)
        rows = self.rows if index is None else self.rows[index]
        cols = self.cols if index is None else self.cols[index]
        out[rows, cols] = values
        return out

    def mask(self, index: Optional[np.ndarray] = None) -> np.ndarray:
        out = np.zeros(self.shape, dtype=bool)
        rows = self.rows if index is None else self.rows[index]
        cols = self.cols if index is None else self.cols[index]
        out[rows, cols] = True
        return out


@dataclass
class TestAnalysis:
    name: str
    files: list[RawFileInfo]
    step_labels: list[str]
    payloads: list[dict[str, np.ndarray]]
    facts: dict
    images: dict[str, np.ndarray] = field(default_factory=dict)
    profiles: dict[str, dict] = field(default_factory=dict)
    series: dict[str, np.ndarray] = field(default_factory=dict)
    grid: Optional[Grid] = None
    companions: FolderCompanions = field(default_factory=lambda: FolderCompanions(folder=Path(".")))


# -- reading ---------------------------------------------------------------


def read_raw_frame(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(long_path(path), skipinitialspace=True)
    except Exception as exc:
        raise DataRefused([f"{path.name}: could not be parsed as CSV ({exc})"]) from exc
    frame.columns = [str(name).strip().strip('"') for name in frame.columns]
    if tuple(frame.columns) != RAW_DIC_COLUMNS:
        raise DataRefused([f"{path.name}: columns differ from the raw Vic-3D export"])
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    bad = int(numeric.isna().to_numpy().sum())
    if bad:
        raise DataRefused([f"{path.name}: {bad} blank or non-numeric values"])
    pixels = numeric[["x", "y"]].to_numpy()
    if not np.allclose(pixels, np.round(pixels)):
        raise DataRefused([f"{path.name}: facet pixel coordinates x, y are not integers"])
    numeric["key"] = (
        numeric["x"].round().astype(np.int64) * KEY_STRIDE + numeric["y"].round().astype(np.int64)
    )
    if numeric["key"].duplicated().any():
        raise DataRefused([f"{path.name}: duplicate facet pixel positions"])
    return numeric


def _facet_step(frame: pd.DataFrame) -> int:
    steps = []
    for column in ("x", "y"):
        values = np.unique(frame[column].round().astype(np.int64).to_numpy())
        diffs = np.diff(values)
        if diffs.size:
            steps.append(int(np.gcd.reduce(diffs)))
    return int(np.gcd.reduce(steps)) if steps else 0


def check_group_consistency(group: TestGroup, frames: Sequence[pd.DataFrame]) -> None:
    reasons: list[str] = []
    steps = [_facet_step(frame) for frame in frames]
    if len(set(steps)) > 1:
        detail = ", ".join(f"{info.path.name}: {step} px" for info, step in zip(group.files, steps))
        reasons.append(f"test '{group.name}': facet step differs between files ({detail})")
    reference = set(frames[0]["key"].tolist())
    for info, frame in zip(group.files[1:], frames[1:]):
        keys = set(frame["key"].tolist())
        overlap = len(reference & keys) / max(1, min(len(reference), len(keys)))
        if overlap < MIN_FACET_OVERLAP:
            reasons.append(
                f"test '{group.name}': {info.path.name} shares only {overlap:.0%} of its facets "
                f"with {group.files[0].path.name} (different specimen or region of interest?)"
            )
    if reasons:
        raise DataRefused(reasons)


# -- geometry --------------------------------------------------------------


def build_grid(union_keys: np.ndarray, frame0: pd.DataFrame, step_px: int) -> Grid:
    px = (union_keys // KEY_STRIDE).astype(np.int64)
    py = (union_keys % KEY_STRIDE).astype(np.int64)
    ix = (px - px.min()) // step_px
    iy = (py - py.min()) // step_px
    fx = ((frame0["x"].round().astype(np.int64) - px.min()) // step_px).to_numpy()
    fy = ((frame0["y"].round().astype(np.int64) - py.min()) // step_px).to_numpy()
    design = np.column_stack([np.ones_like(fx, dtype=float), fx, fy])
    coef_x, *_ = np.linalg.lstsq(design, frame0["X"].to_numpy(), rcond=None)
    coef_y, *_ = np.linalg.lstsq(design, frame0["Y"].to_numpy(), rcond=None)
    dx_dix, dx_diy = coef_x[1], coef_x[2]
    dy_dix, dy_diy = coef_y[1], coef_y[2]
    swapped = abs(dx_dix) + abs(dy_diy) < abs(dx_diy) + abs(dy_dix)
    if not swapped:
        cols = ix if dx_dix >= 0 else ix.max() - ix
        rows = iy if dy_diy >= 0 else iy.max() - iy
        pitch_x = math.hypot(dx_dix, dy_dix)
        pitch_y = math.hypot(dx_diy, dy_diy)
    else:
        cols = iy if dx_diy >= 0 else iy.max() - iy
        rows = ix if dy_dix >= 0 else ix.max() - ix
        pitch_x = math.hypot(dx_diy, dy_diy)
        pitch_y = math.hypot(dx_dix, dy_dix)
    if not swapped:
        image_axes = (f"image x → {'+' if dx_dix >= 0 else '−'}X, "
                      f"image y → {'+' if dy_diy >= 0 else '−'}Y")
    else:
        image_axes = (f"image x → {'+' if dy_dix >= 0 else '−'}Y, "
                      f"image y → {'+' if dx_diy >= 0 else '−'}X")
    shape = (int(rows.max()) + 1, int(cols.max()) + 1)
    x_all = frame0["X"].to_numpy()
    y_all = frame0["Y"].to_numpy()
    extent = (
        float(x_all.min() - pitch_x / 2),
        float(x_all.max() + pitch_x / 2),
        float(y_all.min() - pitch_y / 2),
        float(y_all.max() + pitch_y / 2),
    )
    return Grid(
        rows=rows.astype(int),
        cols=cols.astype(int),
        shape=shape,
        pitch_x_mm=float(pitch_x),
        pitch_y_mm=float(pitch_y),
        step_px=int(step_px),
        extent=extent,
        swapped_axes=bool(swapped),
        image_axes=image_axes,
    )


def _frame_check(grid: Grid, from_w: np.ndarray, from_vic: np.ndarray) -> dict:
    """Do the Vic curvature tensor and the ΔW surface agree, as exported or with xx/yy swapped?"""
    scale = max(float(np.max(np.abs(from_w[:2]))), 3 * max(nm.INTERIOR_FLOOR_INV_MM[:2]))

    def mismatch(vic_xx: float, vic_yy: float) -> float:
        return float(max(abs(vic_xx - from_w[0]), abs(vic_yy - from_w[1])) / scale)

    as_exported = mismatch(from_vic[0], from_vic[1])
    swapped = mismatch(from_vic[1], from_vic[0])
    resolvable = float(np.max(np.abs(from_w[:2]))) >= 3 * max(nm.INTERIOR_FLOOR_INV_MM[:2])
    if not resolvable:
        verdict = "not testable: the ΔW curvature is below 3× the noise floor"
    elif as_exported <= 0.5 and as_exported <= swapped:
        verdict = "consistent: the Vic curvature tensor matches the ΔW surface as exported"
    elif swapped <= 0.5 and swapped < as_exported:
        verdict = (
            "INCONSISTENT: the ΔW surface matches the Vic curvature only with κxx and κyy swapped; "
            "the Vic tensor appears to follow the image axes, which are rotated 90° to X/Y here, "
            "so the solver arrays built with the standard conversion have κxx and κyy (and εxx, εyy) "
            "in each other's place"
        )
    else:
        verdict = "INCONSISTENT: the ΔW surface matches the Vic curvature neither as exported nor swapped"
    return {
        "image_axes": grid.image_axes,
        "image_axes_rotated_90deg": grid.swapped_axes,
        "kxx_kyy_from_W_inv_mm": _sig_list(from_w[:2]),
        "kxx_kyy_from_vic_mean_inv_mm": _sig_list(from_vic[:2]),
        "relative_mismatch_as_exported": _sig(as_exported, 3),
        "relative_mismatch_xx_yy_swapped": _sig(swapped, 3),
        "verdict": verdict,
        "solver_data_note": "solver arrays are built with the standard conversion used by the VFM Studio "
        "importer; nothing is re-oriented automatically",
    }


def _width_profile(mask: np.ndarray, pitch_across: float, stations: int = 12) -> list:
    counts = mask.sum(axis=1)
    occupied = np.flatnonzero(counts)
    if occupied.size == 0:
        return []
    picks = np.linspace(occupied[0], occupied[-1], stations).round().astype(int)
    return [_sig(counts[index] * pitch_across, 3) for index in picks]


def _zone_layout(shape: tuple[int, int]) -> tuple[int, int, list[str], list[str]]:
    n_rows, n_cols = shape
    aspect = n_rows / max(1, n_cols)
    if aspect >= 1.8:
        return 5, 3, ["bottom", "lower", "middle", "upper", "top"], ["left", "centre", "right"]
    if aspect <= 1 / 1.8:
        return 3, 5, ["bottom", "middle", "top"], ["far-left", "left", "centre", "right", "far-right"]
    return (
        5,
        5,
        ["bottom", "lower", "middle", "upper", "top"],
        ["far-left", "left", "centre", "right", "far-right"],
    )


# -- noise -----------------------------------------------------------------


def temporal_residual_sigma(load: np.ndarray, series: np.ndarray) -> tuple[np.ndarray, int]:
    """Per-point scatter about a degree-2 polynomial in load. ``series`` is (steps, comps, points)."""
    residual, dof = temporal_residuals(load, series)
    return np.sqrt(np.sum(residual**2, axis=0) / dof) / median_bias(dof), dof


def temporal_residuals(load: np.ndarray, series: np.ndarray) -> tuple[np.ndarray, int]:
    """Residual of every point about its degree-2 polynomial in load, and the residual dof."""
    load = np.asarray(load, dtype=float)
    spread = load.std()
    scaled = (load - load.mean()) / (spread if spread > 0 else 1.0)
    design = np.vander(scaled, N=LOAD_POLYNOMIAL_DEGREE + 1, increasing=True)
    rank = int(np.linalg.matrix_rank(design))
    dof = len(load) - rank
    if dof < 1:
        raise ValueError("not enough distinct load levels for the residual estimator")
    projector = design @ np.linalg.pinv(design)
    return series - np.einsum("ij,jcp->icp", projector, series), dof


def median_bias(dof: int) -> float:
    """Median of sqrt(chi2_dof / dof): dividing by it makes per-facet sigma median-unbiased.

    The calibration medians come from 35-frame static stacks (factor ≈ 0.99), so loaded
    tests with few steps need this to be compared like for like.
    """
    return math.sqrt(chi2.median(dof) / dof)


def _radial(values: np.ndarray, distance: np.ndarray) -> list:
    out = []
    for d in range(RADIAL_MAX_FACETS + 1):
        if d < RADIAL_MAX_FACETS:
            pick = np.abs(distance - d) < 0.5
        else:
            pick = distance >= RADIAL_MAX_FACETS
        chosen = values[pick]
        chosen = chosen[np.isfinite(chosen)]
        out.append(_sig(np.median(chosen)) if chosen.size >= 20 else None)
    return out


def _noise_component_stats(measured: np.ndarray, model: np.ndarray, distance: np.ndarray, index: int) -> dict:
    interior = distance >= nm.INTERIOR_DISTANCE_FACETS
    edge = distance <= nm.EDGE_DISTANCE_FACETS
    meas_med = _median(measured)
    model_med = _median(model)
    meas_int = _median(measured[interior]) if interior.sum() >= 20 else None
    model_int = _median(model[interior]) if interior.sum() >= 20 else None
    meas_edge = _median(measured[edge])
    model_edge = _median(model[edge])
    finite = np.isfinite(measured) & np.isfinite(model)
    correlation = None
    if finite.sum() > 30 and np.std(model[finite]) > 0 and np.std(measured[finite]) > 0:
        correlation = float(np.corrcoef(measured[finite], model[finite])[0, 1])
    return {
        "measured_median_inv_mm": _sig(meas_med),
        "model_median_inv_mm": _sig(model_med),
        "measured_to_model_ratio": _sig(meas_med / model_med if meas_med and model_med else None, 3),
        "measured_interior_median_inv_mm": _sig(meas_int),
        "model_interior_median_inv_mm": _sig(model_int),
        "measured_edge_median_inv_mm": _sig(meas_edge),
        "model_edge_median_inv_mm": _sig(model_edge),
        "measured_edge_amplification": _sig(meas_edge / meas_int if meas_edge and meas_int else None, 3),
        "model_edge_amplification": _sig(model_edge / model_int if model_edge and model_int else None, 3),
        "pointwise_map_correlation": _sig(correlation, 3),
        "campaign_region_median_floor_inv_mm": nm.REGION_MEDIAN_FLOOR_INV_MM[index],
        "measured_to_campaign_floor_ratio": _sig(
            meas_med / nm.REGION_MEDIAN_FLOOR_INV_MM[index] if meas_med else None, 3
        ),
        "legacy_envelope_inv_mm": nm.LEGACY_ENVELOPE_INV_MM[index],
        "legacy_to_measured_ratio": _sig(
            nm.LEGACY_ENVELOPE_INV_MM[index] / meas_med if meas_med else None, 3
        ),
        "radial_profile_measured_inv_mm": _radial(measured, distance),
        "radial_profile_model_inv_mm": _radial(model, distance),
    }


# -- loads -----------------------------------------------------------------


def _step_loads(group: TestGroup) -> tuple[list[dict[str, float]], FolderCompanions]:
    match = frame_match(group)
    if not match.ready:
        raise DataRefused([f"{group.name}: {problem}" for problem in match.problems])
    companions = group.companions
    return [companions.loads_for(info.dic_number) for info in group.files], companions


def _dominant(loads: list[dict[str, float]], companions: FolderCompanions, kind: str) -> Optional[str]:
    if not loads or not all(loads):
        return None
    names = {
        item.name
        for item in companions.channels
        if item.kind == kind and all(item.name in values for values in loads)
    }
    if not names:
        return None

    def spread(name: str) -> float:
        series = [values[name] for values in loads]
        return max(series) - min(series)

    return max(sorted(names), key=spread)


def _channel_unit(companions: FolderCompanions, name: str) -> str:
    channel = companions.channel(name)
    return channel.unit if channel is not None else ""


def _force_vectors(
    loads: list[dict[str, float]], companions: FolderCompanions
) -> tuple[Optional[list[np.ndarray]], list[str]]:
    if not loads or not all(loads):
        return None, []
    channels = [item for item in companions.channels if item.kind == "moment"]
    names = [item.name for item in channels if all(item.name in values for values in loads)]
    if not names:
        return None, []
    labels = [default_import_name(companions.channel(name).column) for name in names]
    vectors = [np.asarray([values[name] for name in names], dtype=float) for values in loads]
    return vectors, labels


# -- main ------------------------------------------------------------------


def analyse_group(
    group: TestGroup, progress: Callable[[str], None] = lambda _text: None
) -> TestAnalysis:
    if not group.files:
        raise DataRefused([f"test '{group.name}' has no files"])
    refused = [f"{info.path.name}: {'; '.join(info.problems)}" for info in group.files if not info.ok]
    if refused:
        raise DataRefused(refused)
    frames = []
    for index, info in enumerate(group.files, start=1):
        progress(f"{group.name}: reading {info.path.name} ({index}/{len(group.files)})")
        frames.append(read_raw_frame(info.path))
    check_group_consistency(group, frames)
    progress(f"{group.name}: building solver data")
    n_steps = len(frames)
    step_px = _facet_step(frames[0])
    union_keys = np.unique(np.concatenate([frame["key"].to_numpy() for frame in frames]))
    present = np.zeros((n_steps, union_keys.size), dtype=bool)
    for s, frame in enumerate(frames):
        present[s, np.searchsorted(union_keys, frame["key"].to_numpy())] = True
    common_mask = present.all(axis=0)
    common_keys = union_keys[common_mask]
    grid = build_grid(union_keys, frames[0], step_px)
    common_index = np.flatnonzero(common_mask)

    aligned = _aligned_frames(frames, intersect=True)
    loads, companions = _step_loads(group)
    force_vectors, force_labels = _force_vectors(loads, companions)
    payloads = []
    for s, frame in enumerate(aligned):
        payload = _convert_kinematics(frame, vic3d_conventions=True)
        if force_vectors is not None:
            payload["F"] = force_vectors[s]
            payload["F_labels"] = np.array(force_labels, dtype=object)
        payloads.append(payload)

    progress(f"{group.name}: measuring deformation, noise and read quality")
    curv = np.stack(
        [np.stack([factor * frame[col].to_numpy() for _, col, factor in CURVATURE]) for frame in aligned]
    )
    strain = np.stack(
        [np.stack([factor * frame[col].to_numpy() for _, col, factor in STRAIN]) for frame in aligned]
    )
    disp = np.stack([np.stack([frame[c].to_numpy() for c in ("U", "V", "W")]) for frame in aligned])
    d_curv = curv - curv[0]
    d_strain = strain - strain[0]
    d_disp = disp - disp[0]
    x_mm = aligned[0]["X"].to_numpy()
    y_mm = aligned[0]["Y"].to_numpy()

    mask_common = grid.mask(common_index)
    mask_union = grid.mask()
    distance_img = nm.edge_distance(mask_common)
    distance = distance_img[grid.rows[common_index], grid.cols[common_index]]

    moment = _dominant(loads, companions, "moment")
    rotation = _dominant(loads, companions, "rotation")
    if moment is None:
        raise DataRefused([
            f"{group.name}: the machine data has no moment channel (Tx, Ty or Tz) on every DIC frame"
        ])
    load_series = np.asarray([values[moment] for values in loads], dtype=float)
    final = int(np.argmax(np.abs(load_series - load_series[0])))
    load_source = f"machine channel {moment} [{_channel_unit(companions, moment)}]"

    interior = distance >= nm.INTERIOR_DISTANCE_FACETS
    core = interior if interior.sum() >= 50 else np.ones_like(interior)
    final_means = d_curv[final][:, core].mean(axis=1)
    dominant_component = int(np.argmax(np.abs(final_means)))

    # noise
    noise_facts: dict = {"available": False}
    measured_maps: dict[str, np.ndarray] = {}
    model_full = nm.model_sigma_maps(mask_common, grid.pitch_x_mm, grid.pitch_y_mm)
    model_maps = {
        comp: model_full[comp][grid.rows[common_index], grid.cols[common_index]]
        for comp in nm.COMPONENTS
    }
    if n_steps >= MIN_STEPS_FOR_NOISE and load_series is not None:
        try:
            sigma, dof = temporal_residual_sigma(load_series, curv)
            measured_maps = {comp: sigma[i] for i, comp in enumerate(nm.COMPONENTS)}
            noise_facts = {
                "available": True,
                "estimator": (
                    "pointwise temporal residual: at every facet, each curvature component is "
                    f"regressed on {{1, T, T²}} across the {n_steps} steps and the residual "
                    f"scatter is taken ({dof} degrees of freedom per facet), divided by "
                    f"{median_bias(dof):.3f} (median of sqrt(chi2/dof)) so that medians compare "
                    "like for like with the calibration, which used 35-frame static stacks. "
                    "Anything the quadratic in load "
                    "cannot follow (slip, local damage, re-correlation jumps) also counts as noise, "
                    "so this is an upper bound on sensor noise."
                ),
                "load_variable": load_source,
                "residual_dof_per_facet": dof,
                "components": {
                    comp: _noise_component_stats(measured_maps[comp], model_maps[comp], distance, i)
                    for i, comp in enumerate(nm.COMPONENTS)
                },
            }
        except ValueError as exc:
            noise_facts = {"available": False, "reason": str(exc)}
    else:
        noise_facts["reason"] = (
            f"needs at least {MIN_STEPS_FOR_NOISE} steps with a load variable; "
            f"this test has {n_steps}"
        )
    noise_facts["model_prediction_on_this_mask"] = {
        comp: {
            "median_inv_mm": _sig(_median(model_maps[comp])),
            "interior_inv_mm": nm.INTERIOR_FLOOR_INV_MM[i],
            "edge_median_inv_mm": _sig(_median(model_maps[comp][distance <= nm.EDGE_DISTANCE_FACETS])),
        }
        for i, comp in enumerate(nm.COMPONENTS)
    }
    if measured_maps:
        snr = {}
        for i, comp in enumerate(nm.COMPONENTS):
            signal = abs(final_means[i])
            level = _median(measured_maps[comp])
            snr[comp] = _sig(signal / level if level else None, 3)
        noise_facts["signal_to_noise_at_max_load"] = snr

    progress(f"{group.name}: inferring DIC processing parameters and sweeping the curvature filter")
    raster_order = np.lexsort((grid.cols[common_index], grid.rows[common_index]))
    disp_images = np.full((n_steps, 3, *grid.shape), np.nan)
    disp_images[..., grid.rows[common_index], grid.cols[common_index]] = d_disp
    dic_params = dp.dic_parameter_facts(
        mask_common,
        grid.pitch_x_mm,
        grid.pitch_y_mm,
        grid.step_px,
        disp_images,
        d_curv[..., raster_order],
        final,
        dominant_component,
        load_series if noise_facts.get("available") else None,
        MIN_STEPS_FOR_NOISE,
    )

    # read quality (union of facets)
    vic_sigma = np.full((n_steps, union_keys.size), np.nan)
    sigma_z = np.full((n_steps, union_keys.size), np.nan)
    for s, frame in enumerate(frames):
        where = np.searchsorted(union_keys, frame["key"].to_numpy())
        vic_sigma[s, where] = frame["sigma"].to_numpy()
        sigma_z[s, where] = frame["SIGMA_Z"].to_numpy()
    with np.errstate(all="ignore"):
        vic_facet = np.nanmedian(vic_sigma, axis=0)
    missing_steps = (~present).sum(axis=0)
    global_vic = _median(vic_facet)

    ratio_points = None
    if measured_maps:
        ratio_points = np.mean(
            [measured_maps[comp] / model_maps[comp] for comp in nm.COMPONENTS], axis=0
        )

    n_zr, n_zc, row_names, col_names = _zone_layout(grid.shape)
    zone_r = np.minimum((grid.rows * n_zr) // grid.shape[0], n_zr - 1)
    zone_c = np.minimum((grid.cols * n_zc) // grid.shape[1], n_zc - 1)
    zones = []
    for zr in range(n_zr):
        for zc in range(n_zc):
            in_zone = (zone_r == zr) & (zone_c == zc)
            if in_zone.sum() < 30:
                continue
            in_common = in_zone[common_index]
            dropout = float((missing_steps[in_zone] > 0).mean())
            zone_vic = _median(vic_facet[in_zone])
            zone_ratio = _median(ratio_points[in_common]) if ratio_points is not None and in_common.any() else None
            flags = []
            if dropout > DROPOUT_FLAG:
                flags.append(f"{dropout:.0%} of facets lost in at least one step")
            if zone_vic and global_vic and zone_vic > VIC_SIGMA_FLAG * global_vic:
                flags.append(f"Vic-3D correlation uncertainty {zone_vic / global_vic:.1f}× the specimen median")
            if zone_ratio and zone_ratio > NOISE_RATIO_FLAG:
                flags.append(f"measured noise {zone_ratio:.1f}× the model")
            means = d_curv[final][:, in_common].mean(axis=1) if in_common.any() else [np.nan] * 3
            zones.append(
                {
                    "zone": f"{row_names[zr]}-{col_names[zc]}",
                    "facets": int(in_zone.sum()),
                    "area_fraction": _sig(in_zone.mean(), 3),
                    "dropout_fraction": _sig(dropout, 3),
                    "vic_sigma_median_px": _sig(zone_vic, 3),
                    "noise_to_model_ratio": _sig(zone_ratio, 3),
                    "final_mean_dkxx_inv_mm": _sig(means[0]),
                    "final_mean_dkyy_inv_mm": _sig(means[1]),
                    "final_mean_dkxy_inv_mm": _sig(means[2]),
                    "final_mean_dW_mm": _sig(d_disp[final][2][in_common].mean()) if in_common.any() else None,
                    "flags": flags,
                }
            )

    lost = union_keys.size - common_keys.size
    quality = {
        "facets_union": int(union_keys.size),
        "facets_in_every_step": int(common_keys.size),
        "facets_lost_in_some_step": int(lost),
        "per_step": [
            {
                "step": s + 1,
                "facets": int(present[s].sum()),
                "vic_sigma_median_px": _sig(np.nanmedian(vic_sigma[s])),
                "vic_sigma_p95_px": _sig(np.nanpercentile(vic_sigma[s], 95)),
                "sigma_z_median_mm": _sig(np.nanmedian(sigma_z[s])),
            }
            for s in range(n_steps)
        ],
        "vic_sigma_specimen_median_px": _sig(global_vic),
        "zones": zones,
        "flagged_zones": [zone["zone"] for zone in zones if zone["flags"]],
        "zone_layout": f"{n_zr} bands along Y (bottom→top) × {n_zc} along X (left→right), "
        "on the bounding box of the measured region",
    }

    # deformation
    load_table = []
    for s in range(n_steps):
        info = group.files[s]
        row = {
            "step": s + 1,
            "file": info.path.name,
            "dic_frame": info.dic_number,
            "campaign_step": companions.step_of_dic.get(info.dic_number) if info.dic_number is not None else None,
            "time_s": companions.time_of_dic.get(info.dic_number) if info.dic_number is not None else None,
            "machine": {name: _sig(value) for name, value in loads[s].items()},
            "mean_dkxx_inv_mm": _sig(d_curv[s][0].mean()),
            "mean_dkyy_inv_mm": _sig(d_curv[s][1].mean()),
            "mean_dkxy_inv_mm": _sig(d_curv[s][2].mean()),
            "mean_dexx": _sig(d_strain[s][0].mean()),
            "mean_deyy": _sig(d_strain[s][1].mean()),
            "mean_dgxy": _sig(d_strain[s][2].mean()),
            "mean_dU_mm": _sig(d_disp[s][0].mean()),
            "mean_dV_mm": _sig(d_disp[s][1].mean()),
            "mean_dW_mm": _sig(d_disp[s][2].mean()),
            "dW_p2_p98_mm": _sig_list(np.percentile(d_disp[s][2], [2, 98])),
        }
        load_table.append(row)

    linearity = {}
    if load_series is not None and n_steps >= 3 and moment is not None:
        for i, comp in enumerate(nm.COMPONENTS):
            series = d_curv[:, i][:, core].mean(axis=1)
            if np.std(series) > 0:
                fit = np.polyfit(load_series, series, 1)
                predicted = np.polyval(fit, load_series)
                r2 = 1 - np.sum((series - predicted) ** 2) / np.sum((series - series.mean()) ** 2)
                linearity[comp] = {
                    "slope_per_load_unit": _sig(fit[0]),
                    "r2": _sig(r2, 5),
                }

    xc = x_mm - x_mm.mean()
    yc = y_mm - y_mm.mean()
    basis = np.column_stack([np.ones_like(xc), xc, yc, xc**2, xc * yc, yc**2])
    coef, *_ = np.linalg.lstsq(basis, d_disp[final][2], rcond=None)
    w_fit = {
        "tilt_about_y_deg": _sig(math.degrees(math.atan(coef[1])), 3),
        "tilt_about_x_deg": _sig(math.degrees(math.atan(coef[2])), 3),
        "kxx_from_W_inv_mm": _sig(-2 * coef[3]),
        "kyy_from_W_inv_mm": _sig(-2 * coef[5]),
        "kxy_from_W_inv_mm": _sig(-2 * coef[4]),
        "note": "quadratic fit of the final-step ΔW; agreement in sign and size with the mean Δκ "
        "checks that the curvature convention and the displacement field tell the same story",
    }
    frame_check = _frame_check(
        grid,
        np.array([-2 * coef[3], -2 * coef[5], -2 * coef[4]]),
        d_curv[final].mean(axis=1),
    )

    bands = []
    n_bands = 5
    band_of = np.minimum((grid.rows[common_index] * n_bands) // grid.shape[0], n_bands - 1)
    for b, label in enumerate(["bottom", "lower", "middle", "upper", "top"]):
        pick = band_of == b
        if pick.sum() < 20:
            continue
        bands.append(
            {
                "band": label,
                "facets": int(pick.sum()),
                "width_mm": _sig(
                    mask_union[
                        (np.arange(grid.shape[0]) * n_bands) // grid.shape[0] == b
                    ].sum(axis=1).max()
                    * grid.pitch_x_mm,
                    3,
                ),
                "mean_dkxx_inv_mm": _sig(d_curv[final][0][pick].mean()),
                "mean_dkyy_inv_mm": _sig(d_curv[final][1][pick].mean()),
                "mean_dkxy_inv_mm": _sig(d_curv[final][2][pick].mean()),
                "mean_dW_mm": _sig(d_disp[final][2][pick].mean()),
            }
        )

    others = [i for i in range(3) if i != dominant_component]
    deformation = {
        "reference_step": 1,
        "final_step_analysed": final + 1,
        "final_step_rule": "step with the largest change of the dominant machine moment"
        if moment is not None
        else "last step",
        "dominant_machine_moment": moment,
        "dominant_machine_rotation": rotation,
        "dominant_curvature_component": nm.COMPONENTS[dominant_component],
        "interior_mean_dk_at_final_step_inv_mm": dict(zip(nm.COMPONENTS, _sig_list(final_means))),
        "secondary_to_dominant_curvature_ratios": {
            nm.COMPONENTS[i]: _sig(final_means[i] / final_means[dominant_component], 3)
            if final_means[dominant_component]
            else None
            for i in others
        },
        "curvature_linearity_vs_load": linearity,
        "w_quadratic_fit_final_step": w_fit,
        "curvature_frame_check": frame_check,
        "bands_along_y_at_final_step": bands,
        "steps": load_table,
    }

    filled = ndimage.binary_fill_holes(mask_union)
    holes, n_holes = ndimage.label(filled & ~mask_union)
    hole_sizes = ndimage.sum(np.ones_like(holes), holes, index=range(1, n_holes + 1)) if n_holes else []
    big_holes = [size for size in np.atleast_1d(hole_sizes) if size >= 4]
    folder = group.files[0].path.parent
    specimen = {
        "test_name": group.name,
        "folder": folder.name,
        "parent_folder": folder.parent.name,
        "grandparent_folder": folder.parent.parent.name,
        "measured_region_x_mm": _sig_list([grid.extent[0], grid.extent[1]]),
        "measured_region_y_mm": _sig_list([grid.extent[2], grid.extent[3]]),
        "measured_span_mm": {
            "x": _sig(grid.extent[1] - grid.extent[0], 3),
            "y": _sig(grid.extent[3] - grid.extent[2], 3),
        },
        "measured_area_mm2": _sig(union_keys.size * grid.pitch_x_mm * grid.pitch_y_mm, 3),
        "facet_pitch_mm": {"x": _sig(grid.pitch_x_mm), "y": _sig(grid.pitch_y_mm)},
        "camera_image_axes": grid.image_axes,
        "facet_step_px": grid.step_px,
        "internal_holes": {
            "count": len(big_holes),
            "area_mm2": _sig(sum(big_holes) * grid.pitch_x_mm * grid.pitch_y_mm, 3),
        },
        "width_profile_along_y_mm": _width_profile(mask_union, grid.pitch_x_mm),
        "height_profile_along_x_mm": _width_profile(mask_union.T, grid.pitch_y_mm),
        "n_steps": n_steps,
        "machine_channels": sorted(item.name for item in companions.channels),
        "companion_files": sorted(
            path.name for path in (companions.step_map_file, companions.sync_file) if path is not None
        ),
        "companion_notes": list(companions.notes),
    }

    solver = {
        "layout": "Layout A (nodes) — one .mat per load step, read by vfm_data_io",
        "conversion": VIC3D_CONVENTION_NOTE,
        "facets_per_step": int(common_keys.size),
        "steps": n_steps,
        "fields": sorted(payloads[0].keys()),
        "force_vector": force_labels or None,
        "note": "values are absolute per frame, as the Import-steps dialog writes them; the solver "
        "works with increments relative to its reference step",
    }

    facts = {
        "test_name": group.name,
        "files_in_order": [info.path.name for info in group.files],
        "specimen": specimen,
        "solver_data": solver,
        "deformation": deformation,
        "noise": noise_facts,
        "dic_parameters": dic_params,
        "read_quality": quality,
    }

    images = {
        "dW": grid.image(d_disp[final][2], common_index),
        "dkxx": grid.image(d_curv[final][0], common_index),
        "dkyy": grid.image(d_curv[final][1], common_index),
        "dkxy": grid.image(d_curv[final][2], common_index),
        "vic_sigma": grid.image(vic_facet),
        "missing_steps": grid.image(missing_steps.astype(float)),
    }
    for comp in nm.COMPONENTS:
        images[f"model_{comp}"] = grid.image(model_maps[comp], common_index)
        if comp in measured_maps:
            images[f"measured_{comp}"] = grid.image(measured_maps[comp], common_index)
    profiles = {}
    if noise_facts.get("available"):
        profiles = {
            comp: {
                "measured": noise_facts["components"][comp]["radial_profile_measured_inv_mm"],
                "model": noise_facts["components"][comp]["radial_profile_model_inv_mm"],
            }
            for comp in nm.COMPONENTS
        }
    series = {
        "load": np.asarray(load_series) if load_series is not None else np.arange(n_steps, dtype=float),
        "mean_dk": d_curv[:, :, core].mean(axis=2),
    }
    labels = [
        f"step {s + 1} — {info.path.name}" for s, info in enumerate(group.files)
    ]
    return TestAnalysis(
        name=group.name,
        files=list(group.files),
        step_labels=labels,
        payloads=payloads,
        facts=facts,
        images=images,
        profiles=profiles,
        series={**series, "load_label": np.array([load_source or "step index"], dtype=object)},
        grid=grid,
        companions=companions,
    )


def noise_model_facts() -> dict:
    return {
        "reference": nm.MODEL_REFERENCE,
        "equation": "σ_c(x) = σ_int,c · s_c(x),  s_c = 1 + (g_c − 1)/c,  g_c = ‖w_c(x)‖/‖w_c(full)‖ "
        "(Vic-3D 15×15 local quadratic stencil, truncated at every boundary of the facet mask)",
        "interior_floor_inv_mm": dict(zip(nm.COMPONENTS, nm.INTERIOR_FLOOR_INV_MM)),
        "campaign_region_median_floor_inv_mm": dict(zip(nm.COMPONENTS, nm.REGION_MEDIAN_FLOOR_INV_MM)),
        "damping_constant": nm.DAMPING_CONSTANT,
        "damping_95pct_interval": list(nm.DAMPING_CONFIDENCE_INTERVAL),
        "legacy_envelope_inv_mm": dict(zip(nm.COMPONENTS, nm.LEGACY_ENVELOPE_INV_MM)),
        "calibrated_on": list(nm.CALIBRATION_SPECIMENS),
        "known_limits": [
            "calibrated on 45° laminate coupons; other laminates, speckle patterns or camera "
            "set-ups are extrapolations",
            "measured interior levels differ by about 2× between specimens in the calibration set",
            "fitted on plain coupons alone c = 1.244, on cutouts alone 1.582",
        ],
        "curvature_convention": "κ = (−2·cxx, −2·cyy, +2·cxy); κxy is the engineering twist",
    }
