"""Qt-free import shim for the untouched Generalized VFM modules."""

from __future__ import annotations

from types import ModuleType
from typing import Any, List, Optional, Sequence


def first_attr(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if obj is None:
            break
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


def _load(name: str) -> ModuleType:
    from vfmstudio.core.paths import ensure_core_on_path

    ensure_core_on_path()
    return __import__(name)


def vfm_project() -> ModuleType:
    return _load("vfm_project")


def vfm_mesh() -> ModuleType:
    return _load("vfm_mesh")


def vfm_data_io() -> ModuleType:
    return _load("vfm_data_io")


def vfm_user_inputs() -> ModuleType:
    return _load("vfm_user_inputs")


def vfm_constitutive() -> ModuleType:
    return _load("vfm_constitutive")


def vfm_identifiability() -> ModuleType:
    return _load("vfm_identifiability")


def vfm_virtual_field_post() -> ModuleType:
    return _load("vfm_virtual_field_post")


def parameter_labels(material_type: str, analysis_type: str) -> List[str]:
    module = vfm_user_inputs()
    fn = getattr(module, "parameter_labels_for", None)
    if callable(fn):
        return list(fn(material_type, analysis_type))
    table = getattr(module, "PARAMETER_LABELS_BY_MODEL", {}) or {}
    return list(table.get((material_type, analysis_type), ("P1", "P2", "P3", "P4")))


def load_project(path) -> Any:
    return vfm_project().load_project(path)


def save_project(project, path) -> Any:
    return vfm_project().save_project(project, path)


def run_project(project, verbose: bool = False) -> Any:
    return vfm_project().run_project(project, verbose=verbose)


def project_to_cases(project, verbose: bool = False):
    return vfm_project().project_to_cases(project, verbose=verbose)


def run_gls_identification_cases(
    project,
    cases,
    experiment_ids,
    verbose: bool = False,
) -> Any:
    """Run FGLS on pre-built cases (same settings as ``run_project``)."""
    gls = _load("vfm_gls_identification")
    user = vfm_user_inputs()
    return gls.run_gls_identification_cases(
        cases=cases,
        material_type=project.material_type,
        analysis_type=project.analysis_type,
        initial_parameters=project.initial_parameters,
        identification_settings=user.IdentificationSettings(
            max_iterations=project.max_iterations,
            convergence_tolerance=project.convergence_tolerance,
        ),
        linear_algebra_settings=user.LinearAlgebraSettings(),
        experiment_ids=experiment_ids,
        gls_settings=user.GlsSettings(
            basis_degree=project.basis_degree,
            weighting=project.weighting,
            spatial_correction=project.spatial_correction,
            covariance_tolerance=project.covariance_tolerance,
        ),
        verbose=verbose,
    )


def group_by_experiment(project):
    fn = getattr(vfm_project(), "group_by_experiment", None)
    if callable(fn):
        return fn(project)
    tests = list(getattr(project, "tests", []))
    order: List[int] = []
    groups: dict = {}
    for test in tests:
        exp_id = int(getattr(test, "experiment_id"))
        if exp_id not in groups:
            groups[exp_id] = []
            order.append(exp_id)
        groups[exp_id].append(test)
    names = getattr(project, "experiment_names", {}) or {}
    return [(i, names.get(i, f"Experiment {i}"), groups[i]) for i in order]


def check_experiment_consistency(project) -> List[str]:
    fn = getattr(vfm_project(), "check_experiment_consistency", None)
    if callable(fn):
        return list(fn(project))
    return []


def check_load_orientation(project) -> List[str]:
    fn = getattr(vfm_project(), "check_load_orientation", None)
    if callable(fn):
        return list(fn(project))
    return []


def apply_experiment_definition(
    project,
    experiment_id: int,
    prescribed: Sequence,
    linking: Sequence,
    external: Sequence,
    mesh_settings: Optional[dict],
    sample_area_total,
    material_orientation_deg: float = 0.0,
    applied_load_source: str = "mat",
) -> None:
    fn = getattr(vfm_project(), "apply_experiment_definition", None)
    if callable(fn):
        fn(
            project,
            experiment_id,
            prescribed,
            linking,
            external,
            mesh_settings,
            sample_area_total,
            material_orientation_deg=material_orientation_deg,
            applied_load_source=applied_load_source,
        )


def build_mesh(mesh_settings, analysis_type: str):
    return vfm_mesh().build_mesh(mesh_settings, analysis_type=analysis_type)


def mesh_settings_class():
    return vfm_user_inputs().MeshSettings


def bounds_class():
    return vfm_user_inputs().Bounds


def load_point_measurements(data_file, analysis_type: str, total_sample_area=None):
    fn = vfm_data_io().load_point_measurements
    try:
        return fn(
            data_file,
            analysis_type=analysis_type,
            total_sample_area=total_sample_area,
        )
    except TypeError:
        return fn(data_file, analysis_type=analysis_type)


def compute_identifiability(cases, constitutive_model):
    return vfm_identifiability().compute_identifiability_diagnostics(
        cases, constitutive_model
    )
