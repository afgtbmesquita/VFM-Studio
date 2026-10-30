"""Read imported measurement files and find the linear load-step window.

Numbers come from the MAT files already on the experiment. Nothing is fitted
by a language model, and nothing is written back to those files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from vfmstudio.core.axes_align import (
    alignment_rotation_deg,
    rotate_engineering_fields,
    rotate_xy,
)
from vfmstudio.core.document import ExperimentModel
from vfmstudio.core.sample_area import estimate_sample_area

COMPONENTS = ("kxx", "kyy", "kxy")
SK_KEYS = ("SK1grid", "SK2grid", "SK3grid")

# A cloud more than this far from the lab axes is rotated onto them.
ALIGNMENT_MIN_DEG = 1.0
# Longest run whose curvature-magnitude-vs-load fit is at least this linear.
R2_LINEAR = 0.995
MIN_LINEAR_STEPS = 3
# An unloaded step: small next to the load reached inside the linear run.
BASELINE_LOAD_FRACTION = 0.15


class ModelRefused(Exception):
    """The project has no imported measurement files the agent can build on."""

    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = list(reasons)
        super().__init__("; ".join(self.reasons))


@dataclass
class StepMeasurement:
    """One imported load step, reduced to what the model builder needs."""

    number: int
    label: str
    path: str
    readable: bool
    problem: str = ""
    curvature_mean: Optional[list[float]] = None
    force: Optional[list[float]] = None
    force_labels: list[str] = field(default_factory=list)
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None


@dataclass
class ExperimentMeasurements:
    experiment_id: int
    name: str
    steps: list[StepMeasurement]
    alignment_deg: float = 0.0
    fit_data_to_system_axes: bool = False
    sample_area: Optional[float] = None
    sample_area_source: str = ""
    load_index: Optional[int] = None
    load_name: Optional[str] = None
    load_resolved: bool = False
    load_reason: str = ""
    # Channel the linearity scan actually used. Equals load_index when that
    # channel is settled; otherwise the channel with the largest range.
    probe_load_index: Optional[int] = None
    linear_start: Optional[int] = None
    linear_end: Optional[int] = None
    linear_r2: Optional[float] = None
    full_series_r2: Optional[float] = None
    linear_resolved: bool = False
    reference_step: Optional[int] = None
    reference_reason: str = ""
    # R² of each curvature mean against the load, over the reported window.
    curvature_r2: dict[str, Optional[float]] = field(default_factory=dict)


def _finite_pair(x, y) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    n = min(x_arr.size, y_arr.size)
    x_arr, y_arr = x_arr[:n], y_arr[:n]
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    return x_arr[mask], y_arr[mask]


def _load_raw(path: Path) -> Optional[dict]:
    try:
        from scipy.io import loadmat
    except ImportError:
        return None
    try:
        return loadmat(str(path), squeeze_me=True, struct_as_record=False)
    except Exception:
        return None


def _as_labels(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (str, bytes, np.str_)):
        text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
        text = text.strip()
        return [text] if text else []
    try:
        values = np.atleast_1d(np.asarray(raw, dtype=object)).ravel()
    except (TypeError, ValueError):
        return []
    labels = []
    for item in values:
        if isinstance(item, bytes):
            text = item.decode("utf-8", "replace").strip()
        else:
            text = str(item).strip()
        if text and text != "None":
            labels.append(text)
    return labels


def _xy_curvature_force(raw: dict) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], list[str]]:
    if raw is None:
        return None, None, None, []
    xy = None
    nodes = raw.get("nodes")
    if nodes is not None:
        arr = np.asarray(nodes, dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            xy = arr[:, :2]
    if xy is None and "X" in raw and "Y" in raw:
        x, y = _finite_pair(raw["X"], raw["Y"])
        if x.size:
            xy = np.column_stack([x, y])
    curves = []
    for key in SK_KEYS:
        if key not in raw:
            curves = []
            break
        curves.append(np.asarray(raw[key], dtype=float).ravel())
    curvature = np.vstack(curves) if len(curves) == 3 else None
    force = None
    if "F" in raw:
        try:
            force = np.atleast_1d(np.asarray(raw["F"], dtype=float)).ravel()
        except (TypeError, ValueError):
            force = None
        if force is not None and (force.size == 0 or not np.any(np.isfinite(force))):
            force = None
    return xy, curvature, force, _as_labels(raw.get("F_labels"))


def _r2(load: np.ndarray, values: np.ndarray) -> Optional[float]:
    if load.size < 2 or np.std(load) == 0.0 or np.std(values) == 0.0:
        return None
    slope, intercept = np.polyfit(load, values, 1)
    predicted = slope * load + intercept
    residual = float(np.sum((values - predicted) ** 2))
    total = float(np.sum((values - np.mean(values)) ** 2))
    if total <= 0.0:
        return None
    return float(1.0 - residual / total)


def linear_window(load: np.ndarray, values: np.ndarray, usable: np.ndarray) -> tuple[Optional[tuple[int, int, float]], Optional[tuple[int, int, float]]]:
    """Best qualifying window, and the best window of any quality.

    Indices are into ``load`` / ``values``. A window may not cross a step
    that is not ``usable``. Each result is ``(start, end, r2)`` inclusive,
    or None when no window of :data:`MIN_LINEAR_STEPS` exists.
    """
    n = int(load.size)
    qualified: list[tuple[int, float, int, int, int]] = []
    any_window: list[tuple[float, int, int, int]] = []
    peak = int(np.nanargmax(np.abs(load))) if np.any(np.isfinite(load)) else 0
    for start in range(n):
        for end in range(start + MIN_LINEAR_STEPS - 1, n):
            if not np.all(usable[start : end + 1]):
                break
            score = _r2(load[start : end + 1], values[start : end + 1])
            if score is None:
                continue
            contains_peak = 1 if start <= peak <= end else 0
            any_window.append((score, end - start, start, end))
            if score + 1.0e-12 >= R2_LINEAR:
                qualified.append((end - start, score, contains_peak, -start, end))
    best_any = None
    if any_window:
        score, _length, start, end = max(any_window)
        best_any = (start, end, score)
    best = None
    if qualified:
        _length, score, _peak, neg_start, end = max(qualified)
        best = (-neg_start, end, score)
    return best, best_any


def _peel_reference(load: np.ndarray, values: np.ndarray, usable: np.ndarray, start: int, end: int) -> tuple[Optional[int], str]:
    """0-based reference index, or None when no unloaded step sits against the run."""
    window = np.abs(load[start : end + 1])
    window = window[np.isfinite(window)]
    if window.size == 0 or float(np.max(window)) <= 0.0:
        return None, "the load does not leave zero, so no unloaded reference was separated"
    scale = float(np.max(window))

    def _is_baseline(index: int) -> bool:
        return bool(usable[index]) and np.isfinite(load[index]) and abs(float(load[index])) <= BASELINE_LOAD_FRACTION * scale

    if _is_baseline(start) and end - start >= MIN_LINEAR_STEPS:
        rest = _r2(load[start + 1 : end + 1], values[start + 1 : end + 1])
        if rest is not None and rest + 1.0e-12 >= R2_LINEAR:
            return start, (
                f"step {start + 1} is unloaded next to the linear run "
                f"(|load| ≤ {BASELINE_LOAD_FRACTION:.0%} of the run) and is the reference"
            )
    if start > 0 and _is_baseline(start - 1):
        return start - 1, (
            f"step {start} is unloaded and sits just before the linear run, so it is the reference"
        )
    return None, "no unloaded step sits against the linear run; identification uses the absolute fields"


def _magnitude(means: np.ndarray) -> np.ndarray:
    """Curvature magnitude at each step. Does not pick a component."""
    series = np.sqrt(np.nansum(np.square(means), axis=1))
    series[~np.all(np.isfinite(means), axis=1)] = np.nan
    return series


def _component_r2(
    load: np.ndarray,
    means: np.ndarray,
    usable: np.ndarray,
    start: int,
    end: int,
) -> dict[str, Optional[float]]:
    window = np.zeros(usable.shape, dtype=bool)
    window[start : end + 1] = True
    scores: dict[str, Optional[float]] = {}
    for index, name in enumerate(COMPONENTS):
        mask = usable & window & np.isfinite(load) & np.isfinite(means[:, index])
        scores[name] = _r2(load[mask], means[mask, index])
    return scores


def _choose_load(
    forces: np.ndarray, labels: Sequence[str], rows: np.ndarray
) -> tuple[Optional[int], Optional[str], bool, str]:
    """Which ``F`` entry the linearity scan can use.

    A channel is settled only when it is the only one, or the only one that
    changes. A larger range is not treated as a reason to ignore the others.
    """
    if forces.size == 0 or forces.shape[1] == 0 or rows.size == 0:
        return None, None, False, "the measurement files have no F"
    block = forces[rows]
    ranges = np.nanmax(block, axis=0) - np.nanmin(block, axis=0)
    ranges = np.nan_to_num(ranges, nan=0.0)
    names = [labels[index] if index < len(labels) else f"F[{index}]" for index in range(forces.shape[1])]
    if forces.shape[1] == 1:
        return 0, names[0], True, f"{names[0]} is the only load channel"
    moving = [index for index, span in enumerate(ranges) if span > 0.0]
    if len(moving) == 1:
        index = moving[0]
        return index, names[index], True, f"{names[index]} is the only load channel that changes"
    if not moving:
        return None, None, False, "no F channel changes across the load steps"
    listed = ", ".join(names[index] for index in moving)
    return (
        None,
        None,
        False,
        f"more than one load channel changes ({listed}), so the load-step window is left for you to set",
    )


def _read_step(number: int, label: str, path: Path, angle: float, rotate: bool) -> StepMeasurement:
    if not path.is_file():
        return StepMeasurement(number, label, str(path), False, "measurement file not found")
    raw = _load_raw(path)
    xy, curvature, force, labels = _xy_curvature_force(raw or {})
    if xy is None or xy.shape[0] < 4:
        return StepMeasurement(number, label or path.name, str(path), False, "no coordinate cloud in the MAT file")
    if curvature is None or curvature.shape[1] != xy.shape[0]:
        return StepMeasurement(number, label or path.name, str(path), False, "curvature grids SK1-3 are missing or the wrong length")
    x, y = xy[:, 0], xy[:, 1]
    means = np.array([float(np.nanmean(curvature[i])) for i in range(3)])
    if rotate and abs(angle) > 1.0e-12:
        x, y = rotate_xy(x, y, angle)
        means = rotate_engineering_fields(means.reshape(3, 1), angle).ravel()
    finite = np.isfinite(means)
    if not np.all(finite):
        return StepMeasurement(number, label or path.name, str(path), False, "curvature means are not finite")
    force_list = None
    if force is not None and np.all(np.isfinite(force)):
        force_list = [float(value) for value in force]
    return StepMeasurement(
        number=number,
        label=label or path.name,
        path=str(path),
        readable=True,
        curvature_mean=[float(value) for value in means],
        force=force_list,
        force_labels=labels,
        x_min=float(np.min(x)),
        x_max=float(np.max(x)),
        y_min=float(np.min(y)),
        y_max=float(np.max(y)),
    )


def measure_experiment(experiment: ExperimentModel) -> ExperimentMeasurements:
    """Curvatures, loads, alignment, area, and the linear window for one experiment."""
    name = experiment.name or f"Experiment {experiment.experiment_id}"
    first_xy = None
    for step in experiment.steps:
        path = Path(step.data_file)
        raw = _load_raw(path) if path.is_file() else None
        xy, _curvature, _force, _labels = _xy_curvature_force(raw or {})
        if xy is not None and xy.shape[0] >= 4:
            first_xy = xy
            break
    angle = 0.0 if first_xy is None else float(alignment_rotation_deg(first_xy[:, 0], first_xy[:, 1]))
    fit = bool(experiment.fit_data_to_system_axes) or abs(angle) > ALIGNMENT_MIN_DEG
    rotate = fit and abs(angle) > 1.0e-12
    steps = [
        _read_step(index, step.label, Path(step.data_file), angle, rotate)
        for index, step in enumerate(experiment.steps, start=1)
    ]
    measured = ExperimentMeasurements(
        experiment_id=int(experiment.experiment_id),
        name=name,
        steps=steps,
        alignment_deg=angle,
        fit_data_to_system_axes=fit,
    )
    readable = [step for step in steps if step.readable and step.curvature_mean is not None]
    if not readable:
        measured.load_reason = "no load step could be read"
        measured.reference_reason = measured.load_reason
        return measured

    means = np.full((len(steps), 3), np.nan)
    for step in readable:
        means[step.number - 1] = step.curvature_mean
    usable = np.array([step.readable for step in steps])
    rows = np.flatnonzero(usable)

    width = max((len(step.force) for step in readable if step.force), default=0)
    forces = np.full((len(steps), width), np.nan)
    labels: list[str] = []
    for step in readable:
        if step.force_labels and not labels:
            labels = list(step.force_labels)
        if not step.force or width == 0:
            continue
        if len(step.force) != width:
            measured.load_resolved = False
            measured.load_reason = "F has a different length from one load step to the next"
            width = 0
            forces = np.zeros((len(steps), 0))
            break
        forces[step.number - 1, : len(step.force)] = step.force
    if width:
        index, load_name, resolved, reason = _choose_load(forces, labels, rows)
        measured.load_index = index
        measured.load_name = load_name
        measured.load_resolved = resolved
        measured.load_reason = reason
    elif not measured.load_reason:
        measured.load_reason = "the measurement files have no F"

    series = _magnitude(means)
    probe = measured.load_index
    if probe is None and forces.shape[1] >= 1:
        ranges = np.nanmax(forces, axis=0) - np.nanmin(forces, axis=0)
        if np.any(np.isfinite(ranges)):
            probe = int(np.nanargmax(np.nan_to_num(ranges, nan=-1.0)))
    measured.probe_load_index = probe
    load = np.full(len(steps), np.nan)
    if probe is not None and forces.shape[1] > probe:
        load = forces[:, probe]
    load_usable = usable & np.isfinite(load) & np.isfinite(series)
    if int(load_usable.sum()) >= MIN_LINEAR_STEPS:
        best, suggestion = linear_window(load, series, load_usable)
        chosen = best if best is not None else suggestion
        if chosen is not None:
            start, end, score = chosen
            measured.linear_start = start + 1
            measured.linear_end = end + 1
            measured.linear_r2 = float(score)
            measured.linear_resolved = best is not None and measured.load_resolved
            if best is not None and measured.load_resolved:
                reference, why = _peel_reference(load, series, load_usable, start, end)
                measured.reference_step = None if reference is None else reference + 1
                measured.reference_reason = why
            elif best is None:
                measured.reference_reason = (
                    f"the straightest run has R² = {score:.4f}, below {R2_LINEAR}, "
                    "so the load-step window is left for you to set"
                )
            else:
                measured.reference_step = None
                measured.reference_reason = (
                    f"the straightest run has R² = {score:.4f}, but {measured.load_reason}, "
                    "so the load-step window is left for you to set"
                )
        full = _r2(load[load_usable], series[load_usable])
        measured.full_series_r2 = None if full is None else float(full)
    else:
        measured.reference_reason = (
            "fewer than three load steps have both a curvature and a load, "
            "so a linear range cannot be separated"
        )
        if steps:
            measured.linear_start = 1
            measured.linear_end = len(steps)
    if (
        measured.linear_start is not None
        and measured.linear_end is not None
        and probe is not None
        and forces.shape[1] > probe
    ):
        measured.curvature_r2 = _component_r2(
            load, means, usable, measured.linear_start - 1, measured.linear_end - 1
        )

    area_already = getattr(experiment, "sample_area_total", None)
    if area_already not in (None, 0, 0.0):
        measured.sample_area = float(area_already)
        measured.sample_area_source = "already set on the experiment"
    else:
        cloud = first_xy
        if rotate and cloud is not None:
            rx, ry = rotate_xy(cloud[:, 0], cloud[:, 1], angle)
            cloud = np.column_stack([rx, ry])
        if cloud is not None:
            measured.sample_area = estimate_sample_area(cloud[:, 0], cloud[:, 1])
            measured.sample_area_source = (
                "lattice cell area of the DIC cloud" if measured.sample_area else ""
            )
    return measured
