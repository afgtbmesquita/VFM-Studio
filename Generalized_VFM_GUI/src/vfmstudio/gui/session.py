"""Application session held by the main window (document + results)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from vfmstudio.core.document import SessionDocument, empty_document


def is_post_kind(kind: str) -> bool:
    return str(kind).startswith("post_")


def is_uncertainty_kind(kind: str) -> bool:
    return str(kind).startswith("post_uncertainty")


_TARGET_KINDS = {
    "experiment",
    "mesh",
    "bcs",
    "bc_group",
    "bc",
    "steps",
    "step",
}


@dataclass
class Selection:
    kind: str = "project"
    experiment_id: Optional[int] = None
    step_index: Optional[int] = None
    bc_kind: Optional[str] = None
    bc_index: Optional[int] = None

    def targets_experiment(self) -> bool:
        return self.kind in _TARGET_KINDS and self.experiment_id is not None


@dataclass
class AppSession:
    document: SessionDocument = field(default_factory=empty_document)
    dirty: bool = False
    result: Any = None
    identifiability: Any = None
    uncertainty: Any = None
    uncertainty_kind: str = "CR2"
    selection: Selection = field(default_factory=Selection)
    current_scalar: str = "kappa_yy"
    display_mode: str = "contour"
    warp_scale: float = 1.0
    show_xy_axes: bool = False
    show_element_point_counts: bool = False
    experiment_check_visible: dict = field(default_factory=dict)
    experiment_check_colors: dict = field(default_factory=dict)
    identifiability_baseline: dict = field(default_factory=dict)
    identifiability_shift_scale: str = "units"
    parameter_space_left_x: str = ""
    parameter_space_left_y: str = ""
    parameter_space_right_x: str = ""
    parameter_space_right_y: str = ""
    parameter_space_groups: list = field(default_factory=list)
    stress_space_experiment_id: Optional[int] = None
    stress_space_step_index: Optional[int] = None
    stress_space_2d_x: str = ""
    stress_space_2d_y: str = ""
    stress_space_3d_x: str = ""
    stress_space_3d_y: str = ""
    stress_space_3d_z: str = ""
    stress_space_color: str = ""
    status_n_points: int = 0
    available_scalars: tuple = ()
    module: str = "model"
    last_post_kind: str = "post_plots"
    pick_mode: bool = False
    pick_nodes: bool = False
    preview_node_ids: tuple = ()
    preview_all_nodes: bool = False
    free_mesh_mode: bool = False
    drag_mesh_mode: bool = False
    free_mesh_points: list = field(default_factory=list)
    plots_fill_central: bool = False
    vf_experiment_id: Optional[int] = None
    vf_step_index: Optional[int] = None
    vf_view_mode: str = "3D surface"
    vf_quantity: str = "w*"
    vf_parameter_index: int = 0
    show_virtual_field: bool = False
    results_unsaved: bool = False
    scalar_clim: Optional[tuple] = None
    plot_modes: dict = field(default_factory=dict)

    @property
    def path(self) -> Optional[Path]:
        return self.document.source_path

    def mark_dirty(self) -> None:
        self.dirty = True

    def selected_experiment(self):
        """Experiment for display (viewport, status). Falls back to the first coupon."""
        if self.selection.experiment_id is None:
            if self.document.experiments:
                return self.document.experiments[0]
            return None
        return self.document.experiment_by_id(self.selection.experiment_id)

    def targeted_experiment(self):
        """Experiment the tree is actually on. None if Model / Material / Post / …"""
        if not self.selection.targets_experiment():
            return None
        return self.document.experiment_by_id(self.selection.experiment_id)

    def selected_step(self):
        experiment = self.selected_experiment()
        if experiment is None or not experiment.steps:
            return None
        index = self.selection.step_index if self.selection.step_index is not None else 0
        index = max(0, min(index, len(experiment.steps) - 1))
        return experiment.steps[index]

    def targeted_step_index(self) -> Optional[int]:
        experiment = self.targeted_experiment()
        if experiment is None or self.selection.kind != "step":
            return None
        if self.selection.step_index is None:
            return None
        index = int(self.selection.step_index)
        if index < 0 or index >= len(experiment.steps):
            return None
        return index

    def view_state(self) -> dict:
        return {
            "vf_experiment_id": self.vf_experiment_id,
            "vf_step_index": self.vf_step_index,
            "vf_view_mode": self.vf_view_mode,
            "vf_quantity": self.vf_quantity,
            "vf_parameter_index": self.vf_parameter_index,
            "uncertainty_kind": self.uncertainty_kind,
            "stress_space_experiment_id": self.stress_space_experiment_id,
            "stress_space_step_index": self.stress_space_step_index,
            "stress_space_2d_x": self.stress_space_2d_x,
            "stress_space_2d_y": self.stress_space_2d_y,
            "stress_space_3d_x": self.stress_space_3d_x,
            "stress_space_3d_y": self.stress_space_3d_y,
            "stress_space_3d_z": self.stress_space_3d_z,
            "stress_space_color": self.stress_space_color,
        }

    def apply_view_state(self, state: Optional[dict]) -> None:
        if not state:
            return
        if state.get("vf_experiment_id") is not None:
            self.vf_experiment_id = int(state["vf_experiment_id"])
        if state.get("vf_step_index") is not None:
            self.vf_step_index = int(state["vf_step_index"])
        if state.get("vf_view_mode"):
            self.vf_view_mode = str(state["vf_view_mode"])
        if state.get("vf_quantity"):
            self.vf_quantity = str(state["vf_quantity"])
        if state.get("vf_parameter_index") is not None:
            self.vf_parameter_index = int(state["vf_parameter_index"])
        if state.get("uncertainty_kind"):
            self.uncertainty_kind = str(state["uncertainty_kind"])
        if state.get("stress_space_experiment_id") is not None:
            self.stress_space_experiment_id = int(state["stress_space_experiment_id"])
        if state.get("stress_space_step_index") is not None:
            self.stress_space_step_index = int(state["stress_space_step_index"])
        for name in (
            "stress_space_2d_x",
            "stress_space_2d_y",
            "stress_space_3d_x",
            "stress_space_3d_y",
            "stress_space_3d_z",
            "stress_space_color",
        ):
            if state.get(name):
                setattr(self, name, str(state[name]))
