"""Application session held by the main window (document + results)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from vfmstudio.core.document import SessionDocument, empty_document


@dataclass
class Selection:
    kind: str = "project"
    experiment_id: Optional[int] = None
    step_index: Optional[int] = None
    bc_kind: Optional[str] = None
    bc_index: Optional[int] = None


@dataclass
class AppSession:
    document: SessionDocument = field(default_factory=empty_document)
    dirty: bool = False
    result: Any = None
    identifiability: Any = None
    selection: Selection = field(default_factory=Selection)
    current_scalar: str = "kappa_yy"
    display_mode: str = "contour"
    warp_scale: float = 1.0
    show_xy_axes: bool = False
    show_element_point_counts: bool = False
    status_n_points: int = 0
    available_scalars: tuple = ()

    @property
    def path(self) -> Optional[Path]:
        return self.document.source_path

    def mark_dirty(self) -> None:
        self.dirty = True

    def selected_experiment(self):
        if self.selection.experiment_id is None:
            if self.document.experiments:
                return self.document.experiments[0]
            return None
        return self.document.experiment_by_id(self.selection.experiment_id)

    def selected_step(self):
        experiment = self.selected_experiment()
        if experiment is None or not experiment.steps:
            return None
        index = self.selection.step_index if self.selection.step_index is not None else 0
        index = max(0, min(index, len(experiment.steps) - 1))
        return experiment.steps[index]
