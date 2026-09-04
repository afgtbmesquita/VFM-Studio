"""Reconstruct plate stress resultants at DIC points from identified stiffness.

Generalised stress is C @ ε with the engineering-shear (Voigt) layout used by
the solver: row 3 of each 3-block is γ_xy / κ_xy, not tensor shear.

Kirchhoff resultants are per unit width:

    N = ∫ σ dz      →  N/mm     (A in N/mm, ε dimensionless)
    M = ∫ σ z dz    →  N        (D in N·mm, κ in mm⁻¹)

That N for M_x, M_y, M_xy is not the total edge couple (N·mm) shown on the
load plot. It is moment per unit width, N·mm/mm.

Identified θ lives in the material frame. Measured fields stay in the test
frame; C is rotated into that frame (the same route as identification).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional, Sequence

import numpy as np

from vfmstudio.core import bridge
from vfmstudio.core.axes_align import apply_alignment_to_measurements
from vfmstudio.core.document import SessionDocument
from vfmstudio.core.fields import experiment_alignment_angle
from vfmstudio.core.jobs import subtract_reference_measurements
from vfmstudio.core.step_series import resolve_step_data_file, step_display_label

ALL_STEPS = -1
COLOR_NONE = "None"
COLOR_STEP = "Load step"

MEMBRANE_KEYS = ("Nx", "Ny", "Nxy")
BENDING_KEYS = ("Mx", "My", "Mxy")


@dataclass(frozen=True)
class ResultantSpec:
    key: str
    math: str
    unit_math: str
    row: int

    def axis_label(self) -> str:
        return f"{self.math} ({self.unit_math})"


_MEMBRANE = (
    ResultantSpec("Nx", r"$N_{x}$", r"N/mm", 0),
    ResultantSpec("Ny", r"$N_{y}$", r"N/mm", 1),
    ResultantSpec("Nxy", r"$N_{xy}$", r"N/mm", 2),
)
_BENDING = (
    ResultantSpec("Mx", r"$M_{x}$", "N", 0),
    ResultantSpec("My", r"$M_{y}$", "N", 1),
    ResultantSpec("Mxy", r"$M_{xy}$", "N", 2),
)
_BENDING_BOTH = (
    ResultantSpec("Mx", r"$M_{x}$", "N", 3),
    ResultantSpec("My", r"$M_{y}$", "N", 4),
    ResultantSpec("Mxy", r"$M_{xy}$", "N", 5),
)


def normalize_analysis_type(analysis_type: str | None) -> str:
    key = str(analysis_type or "out_of_plane").strip().lower().replace("-", "_")
    if key in {"in_plane", "inplane"}:
        return "in_plane"
    if key in {"both", "abbd", "abd"}:
        return "both"
    return "out_of_plane"


def available_resultants(analysis_type: str | None) -> tuple[ResultantSpec, ...]:
    key = normalize_analysis_type(analysis_type)
    if key == "in_plane":
        return _MEMBRANE
    if key == "both":
        return _MEMBRANE + _BENDING_BOTH
    return _BENDING


def resultant_spec(analysis_type: str | None, key: str) -> Optional[ResultantSpec]:
    wanted = str(key or "").strip()
    for spec in available_resultants(analysis_type):
        if spec.key == wanted:
            return spec
    return None


def default_plot_axes(analysis_type: str | None) -> dict[str, str]:
    keys = [spec.key for spec in available_resultants(analysis_type)]
    if "Mx" in keys:
        return {
            "x2d": "Mx",
            "y2d": "My",
            "x3d": "Mx",
            "y3d": "My",
            "z3d": "Mxy",
            "color": "Mxy",
        }
    return {
        "x2d": "Nx",
        "y2d": "Ny",
        "x3d": "Nx",
        "y3d": "Ny",
        "z3d": "Nxy",
        "color": "Nxy",
    }


def resolve_axis_key(analysis_type: str | None, key: str, fallback: str) -> str:
    allowed = {spec.key for spec in available_resultants(analysis_type)}
    wanted = str(key or "").strip()
    if wanted in allowed:
        return wanted
    if fallback in allowed:
        return fallback
    return next(iter(allowed), "")


def color_choices(analysis_type: str | None) -> list[str]:
    return [COLOR_NONE, COLOR_STEP, *[spec.key for spec in available_resultants(analysis_type)]]


def resolve_color_key(analysis_type: str | None, key: str, fallback: str) -> str:
    allowed = set(color_choices(analysis_type))
    wanted = str(key or "").strip()
    if wanted in allowed:
        return wanted
    if fallback in allowed:
        return fallback
    return COLOR_NONE


@dataclass
class StressSpaceCloud:
    values: dict[str, np.ndarray]
    step_index: np.ndarray
    step_labels: list[str]
    experiment_name: str
    analysis_type: str
    n_points: int
    warning: Optional[str] = None

    def column(self, key: str) -> Optional[np.ndarray]:
        if key == COLOR_STEP:
            return np.asarray(self.step_index, dtype=float)
        array = self.values.get(str(key))
        if array is None:
            return None
        return np.asarray(array, dtype=float)


@dataclass(frozen=True)
class StressSpaceTarget:
    experiment_id: int
    experiment_name: str
    step_indices: tuple[int, ...]
    step_labels: tuple[str, ...]


def constitutive_model_from_result(document: Optional[SessionDocument], result: Any):
    """Material-frame constitutive model from identified θ̂.

    HDF5 snapshots store labels only, so this rebuilds A/D/ABD from
    ``final_parameters`` whenever the document (or a full model) is available.
    """
    if result is None:
        return None
    constitutive = bridge.vfm_constitutive()
    theta = getattr(result, "final_parameters", None)
    material_type = None
    analysis_type = None
    if document is not None:
        material_type = document.material.material_type
        analysis_type = document.material.analysis_type
    stored = getattr(result, "constitutive_model", None)
    if material_type is None and stored is not None:
        material_type = getattr(stored, "material_type", None)
        analysis_type = getattr(stored, "analysis_type", None)
    if theta is not None and material_type and analysis_type:
        try:
            return constitutive.define_constitutive_model(
                material_type, analysis_type, theta
            )
        except Exception:
            pass
    matrix = getattr(stored, "material_matrix", None) if stored is not None else None
    if matrix is not None:
        return stored
    return None


def reconstruct_resultants(
    strains,
    constitutive_model,
    orientation_deg: float = 0.0,
) -> dict[str, np.ndarray]:
    """Pointwise N/M = C_test @ ε_test. ``strains`` is (n_gen, n_points)."""
    constitutive = bridge.vfm_constitutive()
    fields = np.asarray(strains, dtype=float)
    if fields.ndim != 2 or fields.size == 0:
        return {}
    model = constitutive_model
    angle = float(orientation_deg or 0.0)
    current = float(getattr(model, "orientation_deg", 0.0) or 0.0)
    if abs(angle) > 1.0e-12 and abs(current) <= 1.0e-12:
        model = constitutive.rotate_constitutive_model(model, angle)
    matrix = np.asarray(getattr(model, "material_matrix"), dtype=float)
    n_gen = int(matrix.shape[0])
    if fields.shape[0] < n_gen:
        raise ValueError(
            f"Strain array has {fields.shape[0]} rows; constitutive matrix "
            f"expects {n_gen}."
        )
    resultants = matrix @ fields[:n_gen]
    analysis = normalize_analysis_type(getattr(model, "analysis_type", None))
    out: dict[str, np.ndarray] = {}
    for spec in available_resultants(analysis):
        if 0 <= spec.row < resultants.shape[0]:
            out[spec.key] = np.asarray(resultants[spec.row], dtype=float).ravel()
    return out


def list_stress_space_targets(
    document: Optional[SessionDocument],
) -> list[StressSpaceTarget]:
    if document is None:
        return []
    targets: list[StressSpaceTarget] = []
    for experiment in document.experiments:
        indices = list(experiment.included_step_indices())
        if not indices:
            indices = list(range(len(experiment.steps)))
        labels = []
        kept: list[int] = []
        for index in indices:
            if index < 0 or index >= len(experiment.steps):
                continue
            kept.append(int(index))
            labels.append(step_display_label(experiment.steps[index]))
        if not kept:
            continue
        name = experiment.name or f"Experiment {experiment.experiment_id}"
        targets.append(
            StressSpaceTarget(
                experiment_id=int(experiment.experiment_id),
                experiment_name=name,
                step_indices=tuple(kept),
                step_labels=tuple(labels),
            )
        )
    return targets


def resolve_stress_space_target(
    document: Optional[SessionDocument],
    experiment_id: Optional[int],
    step_index: Optional[int],
) -> tuple[Optional[StressSpaceTarget], int]:
    """Return (target, step_index) with ALL_STEPS meaning every included step."""
    targets = list_stress_space_targets(document)
    if not targets:
        return None, ALL_STEPS
    chosen = targets[0]
    if experiment_id is not None:
        for item in targets:
            if item.experiment_id == int(experiment_id):
                chosen = item
                break
    if step_index is None:
        return chosen, ALL_STEPS
    wanted = int(step_index)
    if wanted == ALL_STEPS:
        return chosen, ALL_STEPS
    if wanted in chosen.step_indices:
        return chosen, wanted
    return chosen, int(chosen.step_indices[0])


def load_experiment_clouds(
    document: Optional[SessionDocument],
    result: Any,
    experiment_id: Optional[int],
    step_index: Optional[int],
) -> StressSpaceCloud:
    analysis = "out_of_plane"
    if document is not None:
        analysis = normalize_analysis_type(document.material.analysis_type)
    empty = StressSpaceCloud(
        values={},
        step_index=np.zeros(0, dtype=int),
        step_labels=[],
        experiment_name="",
        analysis_type=analysis,
        n_points=0,
    )
    if result is None:
        return replace(empty, warning="Run Identify first.")
    model = constitutive_model_from_result(document, result)
    if model is None:
        return replace(
            empty,
            warning="Identification result has no constitutive model.",
        )
    analysis = normalize_analysis_type(getattr(model, "analysis_type", analysis))
    empty = replace(empty, analysis_type=analysis)
    target, resolved_step = resolve_stress_space_target(
        document, experiment_id, step_index
    )
    if target is None or document is None:
        return replace(empty, warning="Add an experiment with load steps.")
    experiment = document.experiment_by_id(target.experiment_id)
    if experiment is None:
        return replace(empty, warning="Experiment not found.")

    if resolved_step == ALL_STEPS:
        step_ids = list(target.step_indices)
    else:
        step_ids = [int(resolved_step)]

    chunks: list[dict[str, np.ndarray]] = []
    step_ids_per_point: list[np.ndarray] = []
    labels_used: list[str] = []
    errors: list[str] = []
    orientation = float(getattr(experiment, "material_orientation_deg", 0.0) or 0.0)
    reference = _load_reference_measurements(document, experiment, analysis)
    if experiment.reference_step is not None and reference is None:
        errors.append(
            f"Could not load reference step {experiment.reference_step}; "
            "showing unsubtracted fields."
        )

    for local in step_ids:
        measurements, label, error = _load_step_measurements(
            document, experiment, local, analysis, reference
        )
        if error:
            errors.append(error)
            continue
        try:
            values = reconstruct_resultants(
                measurements.strains, model, orientation
            )
        except Exception as exc:
            errors.append(f"{label}: {exc}")
            continue
        n = _cloud_length(values)
        if n <= 0:
            errors.append(f"{label}: no DIC points.")
            continue
        chunks.append(values)
        step_ids_per_point.append(np.full(n, int(local), dtype=int))
        labels_used.append(label)

    if not chunks:
        warning = errors[0] if errors else "No DIC points to plot."
        return StressSpaceCloud(
            values={},
            step_index=np.zeros(0, dtype=int),
            step_labels=[],
            experiment_name=target.experiment_name,
            analysis_type=analysis,
            n_points=0,
            warning=warning,
        )
    merged = _concat_values(chunks)
    step_arr = np.concatenate(step_ids_per_point) if step_ids_per_point else np.zeros(0, dtype=int)
    warning = None
    if errors:
        warning = "; ".join(errors[:3])
    return StressSpaceCloud(
        values=merged,
        step_index=step_arr,
        step_labels=labels_used,
        experiment_name=target.experiment_name,
        analysis_type=analysis,
        n_points=int(_cloud_length(merged)),
        warning=warning,
    )


def _cloud_length(values: dict[str, np.ndarray]) -> int:
    for array in values.values():
        return int(np.asarray(array).ravel().size)
    return 0


def _concat_values(chunks: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = list(chunks[0].keys())
    merged: dict[str, np.ndarray] = {}
    for key in keys:
        parts = [np.asarray(chunk[key], dtype=float).ravel() for chunk in chunks if key in chunk]
        if parts:
            merged[key] = np.concatenate(parts)
    return merged


def _load_reference_measurements(document, experiment, analysis_type: str):
    if experiment.reference_step is None:
        return None
    reference_index = int(experiment.reference_step) - 1
    if reference_index < 0 or reference_index >= len(experiment.steps):
        return None
    measurements, _label, error = _read_step_file(
        document, experiment, reference_index, analysis_type
    )
    if error or measurements is None:
        return None
    return measurements


def _load_step_measurements(
    document,
    experiment,
    step_index: int,
    analysis_type: str,
    reference,
):
    label = (
        step_display_label(experiment.steps[step_index])
        if 0 <= step_index < len(experiment.steps)
        else f"Step {step_index + 1}"
    )
    measurements, _file_label, error = _read_step_file(
        document, experiment, step_index, analysis_type
    )
    if error or measurements is None:
        return None, label, error
    reference_index = (
        None
        if experiment.reference_step is None
        else int(experiment.reference_step) - 1
    )
    if reference is not None and reference_index is not None and step_index != reference_index:
        try:
            subtract_reference_measurements(measurements, reference)
        except Exception as exc:
            return None, label, f"{label}: {exc}"
    return measurements, label, None


def _read_step_file(document, experiment, step_index: int, analysis_type: str):
    if step_index < 0 or step_index >= len(experiment.steps):
        return None, "", f"Load step {step_index + 1} is missing."
    step = experiment.steps[step_index]
    label = step_display_label(step)
    path = resolve_step_data_file(step.data_file, document.source_path)
    if path is None or not path.is_file():
        return None, label, f"Measurement file not found: {step.data_file}"
    try:
        measurements = bridge.load_point_measurements(
            path,
            analysis_type=analysis_type,
            total_sample_area=experiment.sample_area_total,
        )
    except Exception as exc:
        return None, label, f"{label}: {exc}"
    if experiment.fit_data_to_system_axes:
        angle = experiment_alignment_angle(
            experiment, analysis_type, document.source_path
        )
        apply_alignment_to_measurements(measurements, angle)
    return measurements, label, None
