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

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "number_of_elements_x": int(self.n_elements_x),
            "number_of_elements_y": int(self.n_elements_y),
            "bounds": self.bounds.model_dump(),
        }
        if self.x_nodes:
            payload["x_nodes"] = list(self.x_nodes)
        if self.y_nodes:
            payload["y_nodes"] = list(self.y_nodes)
        return payload

    @classmethod
    def from_payload(cls, payload: Optional[dict]) -> "MeshModel":
        payload = dict(payload or {})
        raw_bounds = payload.get("bounds") or {}
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
        )


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
    if mesh.x_nodes or mesh.y_nodes:
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

    def next_experiment_id(self) -> int:
        if not self.experiments:
            return 1
        return max(item.experiment_id for item in self.experiments) + 1


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
    return SessionDocument(
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
            spatial_correction=bool(getattr(project, "spatial_correction", False)),
            covariance_tolerance=float(
                getattr(project, "covariance_tolerance", 1.0e-4)
            ),
        ),
        experiments=experiments,
    )


def document_to_vfm_project(document: SessionDocument):
    module = bridge.vfm_project()
    ProjectTest = getattr(module, "ProjectTest")
    VfmProject = getattr(module, "VfmProject")
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
        spatial_correction=document.solver.spatial_correction,
        covariance_tolerance=document.solver.covariance_tolerance,
        description=document.description,
        created=document.created,
        source_path=document.source_path,
        experiment_names=names,
    )
