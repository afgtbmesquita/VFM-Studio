"""Per-step load / curvature series for plotting and analysis windows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from vfmstudio.core.document import ExperimentModel, LoadStepModel
from vfmstudio.core.fields import (
    SCALAR_LABELS_IN_PLANE,
    SCALAR_LABELS_OUT_OF_PLANE,
    conjugate_curvature_name,
    conjugate_moment_math,
    experiment_alignment_angle,
    force_index_for_group,
    load_step_fields,
)


_STUDY_FOLDERS = (
    "Longitudinal_bending",
    "Transverse_bending",
    "45deg_bending",
    "Twist",
)


def campaign_root(source: Optional[Path]) -> Optional[Path]:
    if source is None:
        return None
    for parent in [source, *source.parents]:
        if parent.name == "Conference_campaign_bending":
            return parent
    return None


def with_pre_conference(path: Path) -> Optional[Path]:
    """Rewrite a pre-move coupon path onto ``Pre_Conference_Studies``.

    Inserting ``Pre_Conference_Studies`` immediately after the campaign folder
    is wrong once Generalized VFM lives under ``VFM Studio``: that would look
    for ``Pre_Conference_Studies/VFM Studio/Longitudinal_bending/...``.  Jump
    to the study-folder name instead.
    """
    parts = list(path.parts)
    try:
        index = parts.index("Conference_campaign_bending")
    except ValueError:
        return None
    rest = parts[index + 1 :]
    if rest and rest[0] == "Pre_Conference_Studies":
        return None
    for offset, part in enumerate(rest):
        if part in _STUDY_FOLDERS:
            return Path(*parts[: index + 1]) / "Pre_Conference_Studies" / Path(
                *rest[offset:]
            )
    return None


def resolve_step_data_file(
    data_file: Path,
    source_path: Optional[Path] = None,
) -> Optional[Path]:
    """Find a measurement MAT on disk, including campaign fallbacks."""
    path = Path(data_file)
    candidates = [path]
    if source_path is not None:
        candidates.append(Path(source_path).parent / path)
    extras: List[Path] = []
    for candidate in list(candidates):
        absolute = candidate
        if not candidate.is_absolute() and source_path is not None:
            absolute = Path(source_path).parent / candidate
        alt = with_pre_conference(absolute)
        if alt is not None:
            extras.append(alt)
    root = campaign_root(Path(source_path) if source_path is not None else None)
    if root is None:
        root = campaign_root(path if path.is_absolute() else None)
    if root is not None:
        for study, export in (
            ("Longitudinal_bending", "VFM_export"),
            ("Transverse_bending", "VFM_export"),
            ("Transverse_bending", "VFM_export_raw_frame"),
            ("45deg_bending", "VFM_export"),
            ("Twist", "VFM_export"),
        ):
            extras.append(
                root / "Pre_Conference_Studies" / study / export / "mat" / path.name
            )
    seen: set[str] = set()
    for candidate in [*candidates, *extras]:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        if resolved.is_file():
            return resolved
    return None


def relocate_document_data_files(document) -> int:
    """Point missing load-step MATs at Pre_Conference_Studies when possible."""
    source = getattr(document, "source_path", None)
    moved = 0
    for experiment in getattr(document, "experiments", []) or []:
        for step in getattr(experiment, "steps", []) or []:
            if not getattr(step, "data_file", None):
                continue
            current = Path(step.data_file)
            if current.is_file():
                continue
            found = resolve_step_data_file(current, source)
            if found is not None:
                step.data_file = found
                moved += 1
    return moved


def step_display_label(step: LoadStepModel) -> str:
    raw = str(step.label or getattr(step.data_file, "name", "") or "step")
    if ":" in raw:
        raw = raw.split(":")[-1]
    raw = raw.replace("\\", "/").split("/")[-1]
    if raw.lower().endswith(".mat"):
        raw = raw[:-4]
    return raw


def _finite_load_values(raw) -> np.ndarray:
    if raw is None:
        return np.zeros(0, dtype=float)
    arr = np.atleast_1d(np.asarray(raw, dtype=float)).ravel()
    return arr


def viewport_caption(
    experiment: ExperimentModel,
    step: Optional[LoadStepModel] = None,
    *,
    step_index: Optional[int] = None,
    applied_force=None,
) -> str:
    """Bottom-right 3D overlay: experiment, current step, and applied loads."""
    name = str(experiment.name or "").strip() or f"Experiment {experiment.experiment_id}"
    lines = [name]
    n_steps = len(experiment.steps)
    if step is not None and n_steps:
        if step_index is None:
            try:
                step_index = experiment.steps.index(step)
            except ValueError:
                step_index = 0
        step_index = max(0, min(int(step_index), n_steps - 1))
        lines.append(
            f"Step {step_index + 1}/{n_steps}: {step_display_label(step)}"
        )
        reference = getattr(experiment, "reference_step", None)
        if reference is not None:
            lines.append(f"Relative to step {int(reference)}")
        values = _finite_load_values(applied_force)
        if values.size == 0:
            values = _finite_load_values(getattr(step, "external_values", None))
        n_force = max(int(values.size), len(experiment.external), 1)
        channels = force_components(experiment, n_force)
        load_lines = []
        for channel in channels:
            if channel.index >= values.size:
                continue
            value = float(values[channel.index])
            if not np.isfinite(value):
                continue
            load_lines.append(f"{channel.combo_label} = {value:.4g} N·mm")
        if load_lines:
            lines.extend(load_lines)
        else:
            lines.append("Load: —")
    elif not n_steps:
        lines.append("No load steps")
    return "\n".join(lines)


def curvature_field_names(analysis_type: str) -> List[str]:
    key = str(analysis_type).lower()
    if key in {"in_plane", "inplane"}:
        pairs = SCALAR_LABELS_IN_PLANE
    elif key in {"both"}:
        pairs = SCALAR_LABELS_IN_PLANE + SCALAR_LABELS_OUT_OF_PLANE
    else:
        pairs = SCALAR_LABELS_OUT_OF_PLANE
    return [name for name, _label in pairs]


@dataclass
class ForceComponent:
    """One applied-load channel plotted on the Y axis (N·mm, never N)."""

    index: int
    edge: Optional[str]
    combo_label: str
    axis_math: str
    dof: Optional[str] = None
    curvature_name: Optional[str] = None


@dataclass
class SpatialSelection:
    """How curvature is reduced across the specimen for the load plot."""

    kind: str = "all"  # all | point | roi
    x: float = 0.0
    y: float = 0.0
    x_min: float = 0.0
    x_max: float = 0.0
    y_min: float = 0.0
    y_max: float = 0.0


def _moment_symbol(group) -> str:
    """Kirchhoff traction on this edge, from the DOF — not from W_X ⇒ M_xx.

    ``W_X`` on a top/bottom cut is the twisting moment ``M_xy`` (tangential
    slope).  Calling it ``M_xx`` is what made twist tests look like a
    positive bending couple on the loaded edge.
    """
    dof = str(getattr(group, "dof", "") or "")
    edge = str(getattr(group, "edge", "") or "") or None
    if conjugate_curvature_name(dof, edge) is not None or str(dof).upper() in {
        "W", "U", "V",
    }:
        return conjugate_moment_math(dof, edge)
    raw = str(getattr(group, "label", "") or "").strip()
    token = raw.split()[0] if raw else ""
    if token.lower().startswith("m") or token.lower().startswith("f"):
        body = token.replace("M_", "").replace("m_", "")
        if token[0] in "Mm":
            subscript = body[1:] if body.lower().startswith("m") else body
            subscript = subscript.replace("{", "").replace("}", "")
            return rf"M_{{{subscript}}}" if subscript else r"M"
        subscript = body[1:] if body.lower().startswith("f") else body
        subscript = subscript.replace("{", "").replace("}", "")
        return rf"F_{{{subscript}}}" if subscript else r"F"
    return r"M"


def force_components(experiment: ExperimentModel, n_force: int) -> List[ForceComponent]:
    """Y-axis load channels. Bottom-edge moments are omitted (not measured)."""
    items: List[ForceComponent] = []
    groups = list(experiment.external or [])
    for index, group in enumerate(groups):
        edge = str(group.edge or "").strip().lower() or None
        if edge == "bottom":
            continue
        force_index = force_index_for_group(group, index)
        if n_force == 1:
            force_index = 0
        elif n_force <= 0 or force_index < 0 or force_index >= n_force:
            continue
        symbol = _moment_symbol(group)
        plain = symbol.replace("{", "").replace("}", "")
        edge_note = f" ({edge})" if edge else ""
        items.append(
            ForceComponent(
                index=force_index,
                edge=edge,
                combo_label=f"{plain}{edge_note}",
                axis_math=rf"${symbol}\ (\mathrm{{N\cdot mm}})$",
                dof=str(getattr(group, "dof", "") or "") or None,
                curvature_name=conjugate_curvature_name(
                    getattr(group, "dof", ""), edge
                ),
            )
        )
    if not items and n_force > 0:
        for index in range(n_force):
            items.append(
                ForceComponent(
                    index=index,
                    edge=None,
                    combo_label="Moment" if n_force == 1 else f"M{index + 1}",
                    axis_math=r"$M\ (\mathrm{N\cdot mm})$",
                )
            )
    if not items:
        items.append(
            ForceComponent(
                index=0,
                edge=None,
                combo_label="Moment",
                axis_math=r"$M\ (\mathrm{N\cdot mm})$",
            )
        )
    return items


def force_component_labels(experiment: ExperimentModel, n_force: int) -> List[str]:
    return [item.combo_label for item in force_components(experiment, n_force)]


def selection_mask(
    x: np.ndarray,
    y: np.ndarray,
    selection: SpatialSelection,
) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if selection.kind == "point":
        if x.size == 0:
            return np.zeros(0, dtype=bool)
        nearest = int(np.nanargmin((x - selection.x) ** 2 + (y - selection.y) ** 2))
        mask = np.zeros(x.size, dtype=bool)
        mask[nearest] = True
        return mask
    if selection.kind == "roi":
        x_min, x_max = sorted((selection.x_min, selection.x_max))
        y_min, y_max = sorted((selection.y_min, selection.y_max))
        return (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    return np.ones(x.size, dtype=bool)


def field_from_selection(
    x: Optional[np.ndarray],
    y: Optional[np.ndarray],
    values: Optional[np.ndarray],
    selection: SpatialSelection,
) -> float:
    if values is None:
        return float("nan")
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0:
        return float("nan")
    if x is None or y is None or selection.kind == "all":
        return float(np.nanmean(values)) if values.size else float("nan")
    mask = selection_mask(x, y, selection)
    if mask.size != values.size:
        return float(np.nanmean(values))
    chosen = values[mask]
    if chosen.size == 0:
        return float("nan")
    return float(np.nanmean(chosen))


@dataclass
class StepSeriesPoint:
    index: int
    label: str
    force: Optional[np.ndarray]
    field_means: Dict[str, float] = field(default_factory=dict)
    fields: Dict[str, np.ndarray] = field(default_factory=dict)
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    path: Optional[Path] = None
    error: Optional[str] = None

    def force_value(self, component: int) -> float:
        if self.force is None:
            return float("nan")
        vector = np.atleast_1d(np.asarray(self.force, dtype=float)).ravel()
        if component < 0 or component >= vector.size:
            return float("nan")
        return float(vector[component])

    def field_mean(self, name: str) -> float:
        value = self.field_means.get(name)
        if value is None:
            return float("nan")
        return float(value)

    def field_value(self, name: str, selection: Optional[SpatialSelection] = None) -> float:
        if selection is None or selection.kind == "all":
            return self.field_mean(name)
        return field_from_selection(self.x, self.y, self.fields.get(name), selection)


@dataclass
class StepSeries:
    points: List[StepSeriesPoint]
    force_labels: List[str]
    field_names: List[str]
    force_channels: List[ForceComponent] = field(default_factory=list)


def collect_step_series(
    experiment: ExperimentModel,
    analysis_type: str,
    source_path: Optional[Path] = None,
    cache: Optional[dict] = None,
) -> StepSeries:
    """Load MAT force vectors and kinematic fields, referenced if a step is set."""
    field_names = curvature_field_names(analysis_type)
    points: List[StepSeriesPoint] = []
    n_force = 0
    for index, step in enumerate(experiment.steps):
        label = step_display_label(step)
        path = resolve_step_data_file(step.data_file, source_path)
        if path is None:
            points.append(
                StepSeriesPoint(
                    index=index,
                    label=label,
                    force=None,
                    error=f"Measurement file not found: {step.data_file}",
                )
            )
            continue
        cache_key = (
            str(path),
            str(analysis_type),
            experiment.sample_area_total,
            bool(experiment.fit_data_to_system_axes),
        )
        dataset = None if cache is None else cache.get(cache_key)
        try:
            if dataset is None:
                angle = experiment_alignment_angle(
                    experiment, analysis_type, source_path
                )
                dataset = load_step_fields(
                    path,
                    analysis_type,
                    experiment.sample_area_total,
                    align_to_axes=bool(experiment.fit_data_to_system_axes),
                    alignment_angle_deg=angle if experiment.fit_data_to_system_axes else None,
                )
                if cache is not None:
                    cache[cache_key] = dataset
            force_raw = dataset.attrs.get("applied_force")
            force = None
            if force_raw is not None:
                force = np.atleast_1d(np.asarray(force_raw, dtype=float)).ravel()
                if force.size == 0:
                    force = None
                else:
                    n_force = max(n_force, int(force.size))
            means: Dict[str, float] = {}
            stored: Dict[str, np.ndarray] = {}
            x_vals = np.asarray(dataset["x"].values, dtype=float)
            y_vals = np.asarray(dataset["y"].values, dtype=float)
            for name in field_names:
                if name in dataset:
                    values = np.asarray(dataset[name].values, dtype=float)
                    stored[name] = values
                    means[name] = float(np.nanmean(values)) if values.size else float("nan")
            points.append(
                StepSeriesPoint(
                    index=index,
                    label=label,
                    force=force,
                    field_means=means,
                    fields=stored,
                    x=x_vals,
                    y=y_vals,
                    path=path,
                )
            )
        except Exception as exc:
            points.append(
                StepSeriesPoint(
                    index=index,
                    label=label,
                    force=None,
                    path=path,
                    error=str(exc),
                )
            )
    _subtract_series_reference(experiment, points)
    channels = force_components(experiment, n_force)
    return StepSeries(
        points=points,
        force_labels=[item.combo_label for item in channels],
        field_names=field_names,
        force_channels=channels,
    )


def _subtract_series_reference(
    experiment: ExperimentModel,
    points: List[StepSeriesPoint],
) -> None:
    """Make loaded series incremental vs the experiment reference step."""
    if experiment.reference_step is None or not points:
        return
    ref_index = int(experiment.reference_step) - 1
    if ref_index < 0 or ref_index >= len(points):
        return
    reference = points[ref_index]
    if reference.error:
        return
    ref_force = (
        None
        if reference.force is None
        else np.atleast_1d(np.asarray(reference.force, dtype=float)).ravel().copy()
    )
    ref_fields = {
        name: np.asarray(values, dtype=float).copy()
        for name, values in (reference.fields or {}).items()
    }
    for point in points:
        if point.error:
            continue
        if point.force is not None and ref_force is not None:
            current = np.atleast_1d(np.asarray(point.force, dtype=float)).ravel()
            if current.shape == ref_force.shape:
                point.force = current - ref_force
        for name, values in list(point.fields.items()):
            baseline = ref_fields.get(name)
            if baseline is None:
                continue
            current = np.asarray(values, dtype=float)
            if current.shape != baseline.shape:
                continue
            updated = current - baseline
            point.fields[name] = updated
            point.field_means[name] = (
                float(np.nanmean(updated)) if updated.size else float("nan")
            )


def series_xy(
    series: StepSeries,
    force_component: int,
    field_name: str,
    selection: Optional[SpatialSelection] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, y, index) arrays for one force / curvature pair."""
    xs: List[float] = []
    ys: List[float] = []
    indices: List[int] = []
    for point in series.points:
        x_value = point.field_value(field_name, selection)
        y_value = point.force_value(force_component)
        if not np.isfinite(x_value) or not np.isfinite(y_value):
            continue
        xs.append(x_value)
        ys.append(y_value)
        indices.append(point.index)
    return (
        np.asarray(xs, dtype=float),
        np.asarray(ys, dtype=float),
        np.asarray(indices, dtype=int),
    )
