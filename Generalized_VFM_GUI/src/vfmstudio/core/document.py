"""Pydantic session document. Widgets edit this; the solver never sees Qt."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from vfmstudio.core import bridge


class BoundsModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0

    def is_unit_square(self) -> bool:
        return (
            abs(self.x_min - 0.0) <= 1.0e-12
            and abs(self.x_max - 1.0) <= 1.0e-12
            and abs(self.y_min - 0.0) <= 1.0e-12
            and abs(self.y_max - 1.0) <= 1.0e-12
        )


class MeshModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    n_elements_x: int = Field(default=2, ge=1)
    n_elements_y: int = Field(default=2, ge=1)
    bounds: BoundsModel = Field(default_factory=BoundsModel)
    x_nodes: Optional[List[float]] = None
    y_nodes: Optional[List[float]] = None
    free_nodes: Optional[List[List[float]]] = None
    keep_cells: Optional[List[List[int]]] = None

    def clear_free_and_manual(self) -> None:
        self.x_nodes = None
        self.y_nodes = None
        self.free_nodes = None
        self.keep_cells = None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "number_of_elements_x": int(self.n_elements_x),
            "number_of_elements_y": int(self.n_elements_y),
            "bounds": self.bounds.model_dump(),
        }
        if self.x_nodes:
            payload["x_nodes"] = [float(v) for v in self.x_nodes]
        if self.y_nodes:
            payload["y_nodes"] = [float(v) for v in self.y_nodes]
        if self.free_nodes:
            payload["free_nodes"] = [
                [float(pair[0]), float(pair[1])] for pair in self.free_nodes
            ]
        if self.keep_cells:
            payload["keep_cells"] = [
                [int(cell[0]), int(cell[1])] for cell in self.keep_cells
            ]
        return payload

    @classmethod
    def from_payload(cls, payload: Optional[dict]) -> "MeshModel":
        payload = dict(payload or {})
        raw_bounds = payload.get("bounds") or {}
        free_nodes = _pairs_from_payload(payload.get("free_nodes"), float)
        keep_cells = _pairs_from_payload(payload.get("keep_cells"), int)
        return cls(
            n_elements_x=int(
                payload.get("number_of_elements_x")
                or payload.get("n_elements_x")
                or payload.get("nx")
                or 2
            ),
            n_elements_y=int(
                payload.get("number_of_elements_y")
                or payload.get("n_elements_y")
                or payload.get("ny")
                or 2
            ),
            bounds=BoundsModel(
                x_min=float(raw_bounds.get("x_min", 0.0)),
                x_max=float(raw_bounds.get("x_max", 1.0)),
                y_min=float(raw_bounds.get("y_min", 0.0)),
                y_max=float(raw_bounds.get("y_max", 1.0)),
            ),
            x_nodes=payload.get("x_nodes"),
            y_nodes=payload.get("y_nodes"),
            free_nodes=free_nodes,
            keep_cells=keep_cells,
        )


def _pairs_from_payload(raw, caster) -> Optional[List[list]]:
    if not raw:
        return None
    pairs: List[list] = []
    for item in raw:
        if item is None or len(item) < 2:
            continue
        pairs.append([caster(item[0]), caster(item[1])])
    return pairs or None


def apply_detected_mesh_bounds(
    experiment: "ExperimentModel",
    detected: BoundsModel,
    *,
    replace: bool = False,
) -> bool:
    """Replace the default 0–1 mesh, or expand it, so it covers measured data."""
    if detected.x_max <= detected.x_min or detected.y_max <= detected.y_min:
        return False
    mesh = experiment.mesh
    if mesh.x_nodes or mesh.y_nodes or mesh.keep_cells:
        return False
    current = mesh.bounds
    if replace or current.is_unit_square() or len(experiment.steps) <= 1:
        new_bounds = detected
    else:
        new_bounds = BoundsModel(
            x_min=min(current.x_min, detected.x_min),
            x_max=max(current.x_max, detected.x_max),
            y_min=min(current.y_min, detected.y_min),
            y_max=max(current.y_max, detected.y_max),
        )
    if new_bounds.model_dump() == current.model_dump():
        return False
    mesh.bounds = new_bounds
    return True


class BoundaryEntry(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    label: str = ""
    dof: str = "W"
    edge: Optional[str] = None
    nodes: Optional[List[int]] = None
    value: Optional[float] = None
    load_type: Optional[str] = Field(default=None, alias="loadType")
    sort_direction: Optional[str] = Field(default=None, alias="sortDirection")
    force_index: Optional[int] = Field(default=None, alias="forceIndex")

    def to_solver_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        dumped = self.model_dump(by_alias=True, exclude_none=True)
        for key, value in dumped.items():
            data[key] = value
        return data


class LoadStepModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    data_file: Path
    label: str = ""
    description: str = ""
    external_values: List[Optional[float]] = Field(default_factory=list)

    @field_validator("data_file", mode="before")
    @classmethod
    def _as_path(cls, value: Any) -> Path:
        return Path(value)


class ExperimentModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    experiment_id: int = 1
    name: str = ""
    material_orientation_deg: float = 0.0
    sample_area_total: Optional[float] = None
    applied_load_source: str = "mat"
    mesh: MeshModel = Field(default_factory=MeshModel)
    prescribed: List[BoundaryEntry] = Field(default_factory=list)
    linking: List[BoundaryEntry] = Field(default_factory=list)
    external: List[BoundaryEntry] = Field(default_factory=list)
    steps: List[LoadStepModel] = Field(default_factory=list)
    analysis_start: Optional[int] = None
    analysis_end: Optional[int] = None
    reference_step: Optional[int] = None
    fit_data_to_system_axes: bool = False
    include_in_identification: bool = True
    show_deformed_configuration: bool = False
    deformation_scale: float = 1.0

    @field_validator("deformation_scale", mode="before")
    @classmethod
    def _clamp_deformation_scale(cls, value: Any) -> float:
        try:
            scale = float(value)
        except (TypeError, ValueError):
            return 1.0
        if scale != scale:  # NaN
            return 1.0
        return min(50.0, max(0.0, scale))

    def resolved_analysis_start(self) -> int:
        n_steps = len(self.steps)
        if n_steps == 0:
            return 1
        value = 1 if self.analysis_start is None else int(self.analysis_start)
        return max(1, min(value, n_steps))

    def resolved_analysis_end(self) -> int:
        n_steps = len(self.steps)
        if n_steps == 0:
            return 1
        value = n_steps if self.analysis_end is None else int(self.analysis_end)
        return max(1, min(value, n_steps))

    def analysis_window_indices(self) -> List[int]:
        """0-based indices in the inclusive [start, end] window."""
        if not self.steps:
            return []
        start = self.resolved_analysis_start()
        end = self.resolved_analysis_end()
        if start > end:
            return []
        return list(range(start - 1, end))

    def included_step_indices(self) -> List[int]:
        """0-based indices that enter identification (window minus reference)."""
        window = self.analysis_window_indices()
        if self.reference_step is None:
            return window
        reference = int(self.reference_step) - 1
        return [index for index in window if index != reference]

    def boundary_entries(self, kind: str) -> List[BoundaryEntry]:
        if kind not in {"prescribed", "linking", "external"}:
            raise KeyError(kind)
        return getattr(self, kind)

    def remove_boundary_entry(self, kind: str, index: int) -> Optional[BoundaryEntry]:
        """Drop one prescribed, linking, or external BC.

        External groups are indexed in parallel with each load step's
        ``external_values``; those slots are removed so remaining loads stay
        attached to the right group.
        """
        entries = self.boundary_entries(kind)
        if index < 0 or index >= len(entries):
            return None
        removed = entries.pop(index)
        if kind == "external":
            for step in self.steps:
                values = list(step.external_values)
                if index < len(values):
                    values.pop(index)
                    step.external_values = values
        return removed

    def clear_mesh_and_dependencies(self) -> None:
        """Reset the mesh and drop every BC that references it."""
        self.mesh = MeshModel()
        self.prescribed = []
        self.linking = []
        self.external = []
        for step in self.steps:
            step.external_values = []

    def _shift_step_index(self, value: Optional[int], removed_index: int) -> Optional[int]:
        """Remap a 1-based analysis index after deleting 0-based ``removed_index``."""
        if value is None:
            return None
        removed_one = int(removed_index) + 1
        current = int(value)
        if current == removed_one:
            return None
        if current > removed_one:
            return current - 1
        return current

    def _clamp_analysis_window(self) -> None:
        n_steps = len(self.steps)
        if n_steps == 0:
            self.analysis_start = None
            self.analysis_end = None
            self.reference_step = None
            return
        if self.analysis_start is not None:
            self.analysis_start = max(1, min(int(self.analysis_start), n_steps))
        if self.analysis_end is not None:
            self.analysis_end = max(1, min(int(self.analysis_end), n_steps))
        if (
            self.analysis_start is not None
            and self.analysis_end is not None
            and int(self.analysis_start) > int(self.analysis_end)
        ):
            self.analysis_end = int(self.analysis_start)
        if self.reference_step is not None:
            reference = int(self.reference_step)
            if reference < 1 or reference > n_steps:
                self.reference_step = None

    def remove_step(self, index: int) -> Optional[LoadStepModel]:
        """Drop one load step and keep the analysis window pointing at remaining files."""
        if index < 0 or index >= len(self.steps):
            return None
        removed = self.steps.pop(index)
        self.analysis_start = self._shift_step_index(self.analysis_start, index)
        self.analysis_end = self._shift_step_index(self.analysis_end, index)
        self.reference_step = self._shift_step_index(self.reference_step, index)
        self._clamp_analysis_window()
        return removed

    def move_step(self, index: int, delta: int) -> bool:
        """Swap a load step with its neighbour. Analysis indices follow the files."""
        other = int(index) + int(delta)
        if index < 0 or index >= len(self.steps):
            return False
        if other < 0 or other >= len(self.steps):
            return False
        self.steps[index], self.steps[other] = self.steps[other], self.steps[index]
        first, second = index + 1, other + 1

        def remap(value: Optional[int]) -> Optional[int]:
            if value is None:
                return None
            current = int(value)
            if current == first:
                return second
            if current == second:
                return first
            return current

        self.analysis_start = remap(self.analysis_start)
        self.analysis_end = remap(self.analysis_end)
        self.reference_step = remap(self.reference_step)
        return True


class MaterialModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    material_type: str = "orthotropic"
    analysis_type: str = "out_of_plane"
    initial_parameters: List[float] = Field(
        default_factory=lambda: [1.0, 1.0, 1.0, 1.0]
    )

    def labels(self) -> List[str]:
        try:
            return bridge.parameter_labels(self.material_type, self.analysis_type)
        except Exception:
            return [f"P{i + 1}" for i in range(len(self.initial_parameters))]

    def sync_parameters(self, default: float = 1.0) -> List[float]:
        """Keep θ the length the constitutive model expects. Never drop extras
        unless the model shrank; pad missing entries with ``default``."""
        labels = self.labels()
        n = len(labels)
        values = [float(v) for v in (self.initial_parameters or [])]
        if len(values) < n:
            values.extend([float(default)] * (n - len(values)))
        elif len(values) > n:
            values = values[:n]
        self.initial_parameters = values
        return self.initial_parameters


def parameter_math_label(label: str) -> str:
    """'D11' -> '$D_{11}$'."""
    raw = str(label).strip()
    if len(raw) >= 2 and raw[0].isalpha() and raw[1:].isdigit():
        return f"${raw[0]}_{{{raw[1:]}}}$"
    return raw


def parameter_unit_plain(label: str) -> str:
    """A_ij in N/mm (membrane); D_ij in N·mm (bending)."""
    key = str(label).strip().upper()
    if key.startswith("A"):
        return "N/mm"
    if key.startswith("D"):
        return "N·mm"
    return ""


def parameter_unit_math(label: str) -> str:
    """Matplotlib axis unit: A in N/mm, D in N·mm."""
    plain = parameter_unit_plain(label)
    if plain == "N·mm":
        return r"N$\cdot$mm"
    return plain


class SolverModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    max_iterations: int = 40
    convergence_tolerance: float = 1.0e-8
    basis_degree: int = 6
    weighting: str = "gls"
    spatial_correction: bool = False
    covariance_tolerance: float = 1.0e-4
    #  Two-axis noise weighting.  Empty string = "use the single rule above",
    #  which reproduces the historical behaviour bit-for-bit.  Load steps of
    #  one test are repeat measurements of the SAME specimen and are strongly
    #  correlated, so combining them is a different job from combining
    #  independent tests; one rule cannot be right for both.
    load_step_weighting: str = ""
    experiment_weighting: str = ""


class SessionDocument(BaseModel):
    """GUI source of truth. Converted to VfmProject only at I/O and solve time."""

    model_config = ConfigDict(extra="ignore")
    description: str = ""
    created: str = ""
    source_path: Optional[Path] = None
    material: MaterialModel = Field(default_factory=MaterialModel)
    solver: SolverModel = Field(default_factory=SolverModel)
    experiments: List[ExperimentModel] = Field(default_factory=list)

    @field_validator("source_path", mode="before")
    @classmethod
    def _source_path(cls, value: Any) -> Optional[Path]:
        if value in (None, ""):
            return None
        return Path(value)

    def experiment_by_id(self, experiment_id: int) -> Optional[ExperimentModel]:
        for experiment in self.experiments:
            if experiment.experiment_id == experiment_id:
                return experiment
        return None

    def experiments_in_identification(self) -> List[ExperimentModel]:
        """Experiments that enter FGLS (every load step of each, after the window)."""
        return [
            experiment
            for experiment in self.experiments
            if experiment.include_in_identification
        ]

    def next_experiment_id(self) -> int:
        if not self.experiments:
            return 1
        return max(item.experiment_id for item in self.experiments) + 1

    def duplicate_experiment(self, experiment_id: int) -> Optional[ExperimentModel]:
        source = self.experiment_by_id(experiment_id)
        if source is None:
            return None
        clone = source.model_copy(deep=True)
        clone.experiment_id = self.next_experiment_id()
        base = source.name or f"Experiment {source.experiment_id}"
        clone.name = f"{base} copy"
        self.experiments.append(clone)
        return clone

    def copy_boundary_conditions(
        self, source_id: int, target_id: int
    ) -> Optional[ExperimentModel]:
        """Copy prescribed, linking, and external BCs from one experiment to another."""
        source = self.experiment_by_id(source_id)
        target = self.experiment_by_id(target_id)
        if source is None or target is None or source is target:
            return None
        target.prescribed = [item.model_copy(deep=True) for item in source.prescribed]
        target.linking = [item.model_copy(deep=True) for item in source.linking]
        target.external = [item.model_copy(deep=True) for item in source.external]
        n_ext = len(target.external)
        for step in target.steps:
            values = list(step.external_values or [])
            if len(values) < n_ext:
                values.extend([None] * (n_ext - len(values)))
            step.external_values = values[:n_ext] if n_ext else []
        return target


def empty_document() -> SessionDocument:
    return SessionDocument(
        description="New VFM Studio project",
        created=date.today().isoformat(),
        experiments=[],
    )


def document_from_vfm_project(project: Any) -> SessionDocument:
    names = dict(getattr(project, "experiment_names", {}) or {})
    grouped = bridge.group_by_experiment(project)
    experiments: List[ExperimentModel] = []
    for exp_id, name, tests in grouped:
        reference = tests[0]
        steps: List[LoadStepModel] = []
        for test in tests:
            externals = list(getattr(test, "external", []) or [])
            steps.append(
                LoadStepModel(
                    data_file=Path(getattr(test, "data_file")),
                    label=str(getattr(test, "label", "") or ""),
                    description=str(getattr(test, "description", "") or ""),
                    external_values=[
                        entry.get("value") if isinstance(entry, dict) else None
                        for entry in externals
                    ],
                )
            )
        topology = [
            BoundaryEntry.model_validate(entry)
            for entry in (getattr(reference, "external", []) or [])
        ]
        for entry in topology:
            entry.value = None
        experiments.append(
            ExperimentModel(
                experiment_id=int(exp_id),
                name=str(names.get(exp_id, name)),
                material_orientation_deg=float(
                    getattr(reference, "material_orientation_deg", 0.0) or 0.0
                ),
                sample_area_total=getattr(reference, "sample_area_total", None),
                applied_load_source=str(
                    getattr(reference, "applied_load_source", "mat") or "mat"
                ),
                mesh=MeshModel.from_payload(getattr(reference, "mesh_settings", None)),
                prescribed=[
                    BoundaryEntry.model_validate(entry)
                    for entry in (getattr(reference, "prescribed", []) or [])
                ],
                linking=[
                    BoundaryEntry.model_validate(entry)
                    for entry in (getattr(reference, "linking", []) or [])
                ],
                external=topology,
                steps=steps,
            )
        )
    source = getattr(project, "source_path", None)
    document = SessionDocument(
        description=str(getattr(project, "description", "") or ""),
        created=str(getattr(project, "created", "") or ""),
        source_path=Path(source) if source else None,
        material=MaterialModel(
            material_type=str(getattr(project, "material_type", "orthotropic")),
            analysis_type=str(getattr(project, "analysis_type", "out_of_plane")),
            initial_parameters=[
                float(v)
                for v in (
                    getattr(project, "initial_parameters", None) or [1.0, 1.0, 1.0, 1.0]
                )
            ],
        ),
        solver=SolverModel(
            max_iterations=int(getattr(project, "max_iterations", 40)),
            convergence_tolerance=float(
                getattr(project, "convergence_tolerance", 1.0e-8)
            ),
            basis_degree=int(getattr(project, "basis_degree", 6)),
            weighting=str(getattr(project, "weighting", "gls")),
            load_step_weighting=str(
                getattr(project, "load_step_weighting", "") or ""),
            experiment_weighting=str(
                getattr(project, "experiment_weighting", "") or ""),
            spatial_correction=bool(getattr(project, "spatial_correction", False)),
            covariance_tolerance=float(
                getattr(project, "covariance_tolerance", 1.0e-4)
            ),
        ),
        experiments=experiments,
    )
    document.material.sync_parameters()
    return document


def document_to_vfm_project(document: SessionDocument):
    module = bridge.vfm_project()
    ProjectTest = getattr(module, "ProjectTest")
    VfmProject = getattr(module, "VfmProject")
    document.material.sync_parameters()
    tests = []
    names: Dict[int, str] = {}
    for experiment in document.experiments:
        names[int(experiment.experiment_id)] = (
            experiment.name or f"Experiment {experiment.experiment_id}"
        )
        mesh_payload = experiment.mesh.to_payload()
        for step in experiment.steps:
            externals = []
            for index, entry in enumerate(experiment.external):
                payload = entry.to_solver_dict()
                if (
                    index < len(step.external_values)
                    and step.external_values[index] is not None
                ):
                    payload["value"] = step.external_values[index]
                externals.append(payload)
            tests.append(
                ProjectTest(
                    data_file=Path(step.data_file),
                    experiment_id=int(experiment.experiment_id),
                    label=step.label,
                    description=step.description,
                    prescribed=[item.to_solver_dict() for item in experiment.prescribed],
                    linking=[item.to_solver_dict() for item in experiment.linking],
                    external=externals,
                    mesh_settings=mesh_payload,
                    sample_area_total=experiment.sample_area_total,
                    material_orientation_deg=float(experiment.material_orientation_deg),
                    applied_load_source=experiment.applied_load_source,
                )
            )
    return VfmProject(
        tests=tests,
        material_type=document.material.material_type,
        analysis_type=document.material.analysis_type,
        initial_parameters=list(document.material.initial_parameters),
        max_iterations=document.solver.max_iterations,
        convergence_tolerance=document.solver.convergence_tolerance,
        basis_degree=document.solver.basis_degree,
        weighting=document.solver.weighting,
        load_step_weighting=(document.solver.load_step_weighting or None),
        experiment_weighting=(document.solver.experiment_weighting or None),
        spatial_correction=document.solver.spatial_correction,
        covariance_tolerance=document.solver.covariance_tolerance,
        description=document.description,
        created=document.created,
        source_path=document.source_path,
        experiment_names=names,
    )
