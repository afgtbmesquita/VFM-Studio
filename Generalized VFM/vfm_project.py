# =============================================================================
#  Generalized VFM / vfm_project.py
# -----------------------------------------------------------------------------
#  A single self-contained run description: every data file, its experiment
#  number, its virtual mesh, its boundary conditions and its sample area, plus
#  the constitutive model, the initial guess and the solver/GLS settings.
#
#  Load one of these in the GUI (Run setup -> "Load project...") and press run,
#  and you get exactly the numbers run_campaign_gls.py prints.  That is the
#  point: the GUI path and the headless path read the SAME file, so they cannot
#  drift.  make_gui_project.py builds the campaign project and then verifies
#  that claim numerically rather than asserting it.
#
#  WHY JSON AND NOT A .py BC FILE PER TEST
#  ---------------------------------------------------------------------------
#  The shipped BC files are one-per-test and carry no notion of which test
#  belongs to which experiment, nor the material model, nor the initial guess.
#  A campaign is not reproducible from them alone.  This format holds the whole
#  run in one file, with paths stored RELATIVE to the project so the folder can
#  be moved or shared.
#
#  It is deliberately a superset of the BC schema, not a replacement: each
#  test's boundary conditions are the same three lists (PRESCRIBED, LINKING,
#  EXTERNAL) that vfm_data_io.load_boundary_conditions understands, and
#  BcDocument round-trips them unchanged.
#
#  This module imports NO Qt, so the project can be built, loaded and run
#  headlessly.
# =============================================================================

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

FORMAT_NAME = "vfm-generalized-project"
FORMAT_VERSION = 1


# =============================================================================
#  Containers
# =============================================================================

@dataclass
class ProjectTest:
    """One measurement file, with everything needed to use it."""
    data_file: Path
    experiment_id: int = 1
    label: str = ""
    prescribed: List[Dict[str, Any]] = field(default_factory=list)
    linking: List[Dict[str, Any]] = field(default_factory=list)
    external: List[Dict[str, Any]] = field(default_factory=list)
    mesh_settings: Optional[Dict[str, Any]] = None
    sample_area_total: Optional[float] = None
    #  Angle from THIS test's measurement frame (+x) to the material 1-axis,
    #  counter-clockwise positive.  The fields, mesh, BCs and applied load all
    #  stay in the measurement frame; the D matrix is rotated by this angle
    #  instead.  0.0 means the coupon was measured in the material frame.
    #  Shared by every time step of an experiment, like sample_area_total.
    material_orientation_deg: float = 0.0
    #  Where this experiment's applied load comes from: 'mat' (the measurement
    #  file's F field wins) or 'manual' (the per-time-step EXTERNAL 'value'
    #  entries win).  Experiment-level, like the mesh -- see
    #  vfm_identification.APPLIED_LOAD_SOURCES.
    applied_load_source: str = "mat"
    description: str = ""

    def __post_init__(self) -> None:
        from vfm_identification import normalise_applied_load_source

        self.data_file = Path(self.data_file)
        self.experiment_id = int(self.experiment_id)
        self.material_orientation_deg = float(self.material_orientation_deg)
        self.applied_load_source = normalise_applied_load_source(
            self.applied_load_source)


@dataclass
class VfmProject:
    """A complete, reproducible identification run."""
    tests: List[ProjectTest] = field(default_factory=list)
    material_type: str = "orthotropic"
    analysis_type: str = "out_of_plane"
    initial_parameters: List[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    max_iterations: int = 40
    convergence_tolerance: float = 1.0e-8
    basis_degree: int = 6
    weighting: str = "gls"
    spatial_correction: bool = False
    covariance_tolerance: float = 1.0e-4
    description: str = ""
    created: str = ""
    source_path: Optional[Path] = None

    #  Optional display names, keyed by experiment id.  Purely cosmetic.
    experiment_names: Dict[int, str] = field(default_factory=dict)

    @property
    def experiment_ids(self) -> List[int]:
        return [t.experiment_id for t in self.tests]

    @property
    def data_files(self) -> List[Path]:
        return [t.data_file for t in self.tests]

    def experiment_count(self) -> int:
        return len(set(self.experiment_ids))


# =============================================================================
#  Persistence
# =============================================================================

_STUDY_FOLDERS = (
    "Longitudinal_bending",
    "Transverse_bending",
    "45deg_bending",
    "Twist",
)


def _relative_to(path: Path, root: Path) -> str:
    """Store paths relative to the project when possible, so it travels."""
    path = Path(path).resolve()
    try:
        return path.relative_to(root.resolve()).as_posix()
    except ValueError:
        try:
            import os
            return Path(os.path.relpath(path, root.resolve())).as_posix()
        except (ValueError, OSError):
            return path.as_posix()


def _campaign_root_from(path: Path) -> Optional[Path]:
    for parent in [path, *path.parents]:
        if parent.name == "Conference_campaign_bending":
            return parent
    return None


def resolve_measurement_file(project_dir: Path, data_file) -> Path:
    """Resolve a project MAT path, including the Pre_Conference_Studies move.

    Relative entries are first joined to the project folder.  If that file is
    missing, study-folder names in the path are rewritten under
    ``Pre_Conference_Studies`` so projects still load after Generalized VFM
    moved into ``VFM Studio``.
    """
    raw = Path(data_file)
    resolved = raw if raw.is_absolute() else (Path(project_dir) / raw)
    try:
        resolved = resolved.resolve()
    except OSError:
        pass
    if resolved.is_file():
        return resolved
    campaign = _campaign_root_from(resolved) or _campaign_root_from(Path(project_dir))
    if campaign is None:
        return resolved
    parts = list(resolved.parts)
    for name in _STUDY_FOLDERS:
        if name not in parts:
            continue
        index = parts.index(name)
        candidate = campaign / "Pre_Conference_Studies" / Path(*parts[index:])
        if candidate.is_file():
            return candidate
    return resolved


def save_project(project: VfmProject, path: Path) -> Path:
    """Write the project as JSON with paths relative to its own folder."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = path.parent

    payload = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "description": project.description,
        "created": project.created or datetime.now().strftime("%Y-%m-%d"),
        "experiment_names": {str(k): v
                             for k, v in project.experiment_names.items()},
        "material_type": project.material_type,
        "analysis_type": project.analysis_type,
        "initial_parameters": [float(v) for v in project.initial_parameters],
        "identification": {
            "max_iterations": int(project.max_iterations),
            "convergence_tolerance": float(project.convergence_tolerance),
        },
        "gls": {
            "basis_degree": int(project.basis_degree),
            "weighting": project.weighting,
            "spatial_correction": bool(project.spatial_correction),
            "covariance_tolerance": float(project.covariance_tolerance),
        },
        "tests": [
            {
                "data_file": _relative_to(test.data_file, root),
                "experiment_id": int(test.experiment_id),
                "label": test.label,
                "description": test.description,
                "sample_area_total": (None if test.sample_area_total is None
                                      else float(test.sample_area_total)),
                "material_orientation_deg": float(
                    test.material_orientation_deg),
                "applied_load_source": str(test.applied_load_source),
                "mesh_settings": test.mesh_settings,
                "boundary_conditions": {
                    "prescribed": test.prescribed,
                    "linking": test.linking,
                    "external": test.external,
                },
            }
            for test in project.tests
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_project(path: Path) -> VfmProject:
    """Read a project file, resolving data paths against its own folder."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Project file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{path.name} is not valid JSON: {error}") from error

    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} does not contain a JSON object.")
    fmt = payload.get("format")
    if fmt != FORMAT_NAME:
        raise ValueError(
            f"{path.name} has format {fmt!r}; expected {FORMAT_NAME!r}. "
            "This is not a Generalized VFM project file.")
    version = int(payload.get("version", 0))
    if version > FORMAT_VERSION:
        raise ValueError(
            f"{path.name} is format version {version}, but this build only "
            f"understands up to {FORMAT_VERSION}.")

    raw_tests = payload.get("tests") or []
    if not raw_tests:
        raise ValueError(f"{path.name} lists no tests.")

    root = path.parent
    tests: List[ProjectTest] = []
    for index, raw in enumerate(raw_tests, start=1):
        data_file = raw.get("data_file")
        if not data_file:
            raise ValueError(f"{path.name}: test {index} has no data_file.")
        resolved = resolve_measurement_file(root, data_file)
        boundary = raw.get("boundary_conditions") or {}
        tests.append(ProjectTest(
            data_file=resolved,
            experiment_id=int(raw.get("experiment_id", 1)),
            label=str(raw.get("label", "")),
            description=str(raw.get("description", "")),
            prescribed=list(boundary.get("prescribed") or []),
            linking=list(boundary.get("linking") or []),
            external=list(boundary.get("external") or []),
            mesh_settings=raw.get("mesh_settings"),
            sample_area_total=raw.get("sample_area_total"),
            #  Absent in files written before per-experiment orientation
            #  existed; 0.0 reproduces those runs exactly, so no version bump.
            material_orientation_deg=float(
                raw.get("material_orientation_deg", 0.0) or 0.0),
            applied_load_source=raw.get("applied_load_source") or "mat",
        ))

    identification = payload.get("identification") or {}
    gls = payload.get("gls") or {}
    return VfmProject(
        tests=tests,
        material_type=str(payload.get("material_type", "orthotropic")),
        analysis_type=str(payload.get("analysis_type", "out_of_plane")),
        initial_parameters=[float(v) for v in
                            payload.get("initial_parameters", [1, 1, 1, 1])],
        max_iterations=int(identification.get("max_iterations", 40)),
        convergence_tolerance=float(
            identification.get("convergence_tolerance", 1.0e-8)),
        basis_degree=int(gls.get("basis_degree", 6)),
        weighting=str(gls.get("weighting", "gls")),
        spatial_correction=bool(gls.get("spatial_correction", False)),
        covariance_tolerance=float(gls.get("covariance_tolerance", 1.0e-4)),
        experiment_names={int(k): str(v) for k, v in
                          (payload.get("experiment_names") or {}).items()},
        description=str(payload.get("description", "")),
        created=str(payload.get("created", "")),
        source_path=path,
    )


# =============================================================================
#  Experiment grouping
# -----------------------------------------------------------------------------
#  An EXPERIMENT is one physical test.  Its TIME STEPS are the load steps of
#  that test.  What may and may not vary between them is not a UI choice, it is
#  fixed by the mathematics:
#
#    SHARED by every time step of an experiment
#      * the boundary-condition topology.  The virtual field is built to
#        annihilate specific unknown boundary terms; if the constraint set
#        changed between steps the steps would not be measuring the same thing.
#      * the virtual mesh.  C_{i,ts} = U_t^T G(Gamma_ts) U_s couples the
#        virtual fields of steps t and s, so U_t and U_s must live in the same
#        DOF space.  Two different meshes make the cross-time block undefined.
#      * the measurement ROI, so that Gamma_ts = R_t^T R_s indexes the same
#        material points.
#
#    FREE to vary between time steps
#      * the applied force / moment on each EXTERNAL group -- that is the load
#        ramp, and it is the whole reason for having several steps.
# =============================================================================

def group_by_experiment(project: VfmProject):
    """[(experiment_id, name, [tests…])…] in first-seen order."""
    order: List[int] = []
    groups: Dict[int, List[ProjectTest]] = {}
    for test in project.tests:
        if test.experiment_id not in groups:
            groups[test.experiment_id] = []
            order.append(test.experiment_id)
        groups[test.experiment_id].append(test)
    return [(exp_id,
             project.experiment_names.get(exp_id, f"Experiment {exp_id}"),
             groups[exp_id])
            for exp_id in order]


def external_topology(entry: Dict[str, Any]) -> Dict[str, Any]:
    """An EXTERNAL entry stripped of its per-time-step magnitude."""
    return {k: v for k, v in entry.items() if k != "value"}


def apply_experiment_definition(project: VfmProject,
                                experiment_id: int,
                                prescribed: Sequence[Dict[str, Any]],
                                linking: Sequence[Dict[str, Any]],
                                external: Sequence[Dict[str, Any]],
                                mesh_settings: Optional[Dict[str, Any]],
                                sample_area_total: Optional[float],
                                material_orientation_deg: float = 0.0,
                                applied_load_source: str = "mat") -> None:
    """Push one BC topology and mesh onto every time step of an experiment.

    Per-step EXTERNAL magnitudes are PRESERVED where the group still exists,
    matched by position, so editing the topology does not silently wipe a load
    ramp the user already typed in.

    ``material_orientation_deg`` is an experiment-level property for the same
    reason the mesh is: the coupon does not change orientation between its own
    load steps.
    """
    for test in project.tests:
        if test.experiment_id != experiment_id:
            continue
        previous = {index: entry.get("value")
                    for index, entry in enumerate(test.external)}
        test.prescribed = [dict(e) for e in prescribed]
        test.linking = [dict(e) for e in linking]
        rebuilt = []
        for index, entry in enumerate(external):
            merged = external_topology(entry)
            carried = previous.get(index, entry.get("value"))
            if carried is not None:
                merged["value"] = carried
            elif entry.get("value") is not None:
                merged["value"] = entry["value"]
            rebuilt.append(merged)
        test.external = rebuilt
        test.mesh_settings = (dict(mesh_settings) if mesh_settings else None)
        test.sample_area_total = sample_area_total
        test.material_orientation_deg = float(material_orientation_deg)
        test.applied_load_source = str(applied_load_source)


def _is_finite(value) -> bool:
    try:
        return float(value) == float(value) and abs(float(value)) != float("inf")
    except (TypeError, ValueError):
        return False


def check_experiment_consistency(project: VfmProject) -> List[str]:
    """Problems that would make the GLS ill-posed.  Empty list means fine."""
    problems: List[str] = []
    for exp_id, name, tests in group_by_experiment(project):
        if not tests:
            continue
        reference = tests[0]
        if not reference.external and not reference.prescribed:
            problems.append(f"{name}: no boundary conditions defined.")
        elif not reference.external:
            #  Prescribed-but-no-EXTERNAL used to slip through here and fail
            #  much later, inside the solver, as a confusing "F has 2 entries
            #  but the BC file declares 1 external-work group" -- because the
            #  external-work operator falls back to a single placeholder column
            #  when the list is empty.  Catch it here, where it can be named.
            problems.append(
                f"{name}: has prescribed boundary conditions but NO EXTERNAL "
                "work group. The applied moment does its virtual work through "
                "an EXTERNAL group, so without one there is no right-hand "
                "side and the run cannot identify anything. Add one EXTERNAL "
                "group per loaded edge.")
        if not reference.mesh_settings:
            problems.append(f"{name}: no virtual mesh defined.")
        if reference.applied_load_source == "manual":
            #  Catch an unfilled load table here, where the message can name
            #  the step and the group, instead of at solve time.
            if not reference.external:
                problems.append(
                    f"{name}: set to take its applied load from the per-time-"
                    "step values, but it has no EXTERNAL groups to hold them.")
            blanks = []
            for test in tests:
                for position, entry in enumerate(test.external):
                    value = entry.get("value")
                    if value is None or not _is_finite(value):
                        blanks.append(
                            f"'{test.data_file.name}' / "
                            f"{entry.get('label', f'group {position + 1}')}")
            if blanks:
                shown = "; ".join(blanks[:6])
                more = ("" if len(blanks) <= 6
                        else f" (and {len(blanks) - 6} more)")
                problems.append(
                    f"{name}: takes its applied load from the per-time-step "
                    f"values, but these are blank: {shown}{more}. Fill every "
                    "load cell, or switch the experiment back to the .mat 'F' "
                    "field.")
        for test in tests[1:]:
            if len(test.prescribed) != len(reference.prescribed) or \
                    len(test.linking) != len(reference.linking) or \
                    len(test.external) != len(reference.external):
                problems.append(
                    f"{name}: '{test.data_file.name}' has a different "
                    "boundary-condition topology from the first time step.")
                break
            if test.mesh_settings != reference.mesh_settings:
                problems.append(
                    f"{name}: '{test.data_file.name}' uses a different virtual "
                    "mesh. All time steps of one experiment must share a mesh, "
                    "because the cross-time covariance couples their virtual "
                    "fields.")
                break
            if test.applied_load_source != reference.applied_load_source:
                problems.append(
                    f"{name}: '{test.data_file.name}' takes its applied load "
                    f"from {test.applied_load_source!r} but the first time "
                    f"step uses {reference.applied_load_source!r}. One "
                    "experiment takes its loads from one place.")
                break
            if float(test.material_orientation_deg) != \
                    float(reference.material_orientation_deg):
                problems.append(
                    f"{name}: '{test.data_file.name}' declares material "
                    f"orientation {test.material_orientation_deg} deg but the "
                    f"first time step declares "
                    f"{reference.material_orientation_deg} deg. One coupon in "
                    "one fixture has one orientation for the whole ramp.")
                break
    return problems


#  Kirchhoff traction on a named edge.  W_X / W_Y are slopes, not moment
#  names: W_X on top/bottom is the twisting couple M_xy (kappa_xy), not M_xx.
def _conjugate_curvature(dof: str, edge: str = "") -> Optional[str]:
    key = str(dof or "").strip().upper().replace(" ", "_")
    side = str(edge or "").strip().lower()
    if key == "W_XY":
        return "kappa_xy"
    if key == "W_X":
        return "kappa_xy" if side in {"top", "bottom"} else "kappa_xx"
    if key == "W_Y":
        return "kappa_xy" if side in {"left", "right"} else "kappa_yy"
    return None


#  Back-compat for callers that only have a DOF (no edge): the normal-moment
#  pairing, which is correct on left/right for W_X and on top/bottom for W_Y.
_DOF_TO_CURVATURE = {"W_X": "kappa_xx", "W_Y": "kappa_yy"}

#  Below this dominance ratio the coupon is genuinely biaxial (an off-axis or
#  torsion test) and "which curvature dominates" carries no verdict, so the
#  check stays silent rather than crying wolf on a 45 deg coupon.
_ORIENTATION_MARGIN = 5.0


def check_load_orientation(project: VfmProject) -> List[str]:
    """Warn when the declared load DOF does not match the measured bending.

    ADVISORY, never blocking.  ``run_campaign_gls.py`` has always guarded this
    with ``assert_orientation()``, but the project/GUI route bypassed it -- so
    an EXTERNAL group left on the wrong DOF produced a run that completed
    happily and reported the OTHER coefficient.  That failure is silent and
    looks exactly like "the results came out swapped".

    Returns a list of human-readable warnings; empty means nothing suspicious.
    """
    from scipy.io import loadmat

    warnings: List[str] = []
    for _exp_id, name, tests in group_by_experiment(project):
        reference = tests[-1]
        if not reference.external or not reference.data_file.is_file():
            continue

        expected = {
            _conjugate_curvature(str(e.get("dof", "")), str(e.get("edge", "")))
            for e in reference.external
        }
        expected.discard(None)
        dofs = {str(e.get("dof", "")).upper() for e in reference.external}
        if not expected:
            continue

        try:
            raw = loadmat(reference.data_file, squeeze_me=True)
            k_xx = float(np.mean(np.asarray(raw["SK1grid"], dtype=float)))
            k_yy = float(np.mean(np.asarray(raw["SK2grid"], dtype=float)))
            k_xy = float(np.mean(np.asarray(raw["SK3grid"], dtype=float)))
        except Exception:                  # pragma: no cover -- diagnostic only
            continue

        if expected == {"kappa_xy"}:
            bending = max(abs(k_xx), abs(k_yy))
            if abs(k_xy) > _ORIENTATION_MARGIN * max(bending, 1e-30):
                continue
            if bending > _ORIENTATION_MARGIN * max(abs(k_xy), 1e-30):
                measured = "kappa_yy" if abs(k_yy) >= abs(k_xx) else "kappa_xx"
                warnings.append(
                    f"{name}: the applied load is declared on "
                    f"{'/'.join(sorted(dofs))} on a top/bottom (or left/right "
                    f"W_Y) cut, which is the twisting couple conjugate to "
                    f"kappa_xy -- but the measured field is dominated by "
                    f"{measured} (mean kappa_xx = {k_xx:.3e}, "
                    f"kappa_yy = {k_yy:.3e}, kappa_xy = {k_xy:.3e}). "
                    f"A W_X group on the top edge is M_xy, not M_xx."
                )
            continue

        big, small = max(abs(k_xx), abs(k_yy)), min(abs(k_xx), abs(k_yy))
        if small <= 0.0 or big / small < _ORIENTATION_MARGIN:
            continue                        # genuinely biaxial -- no verdict

        measured = "kappa_xx" if abs(k_xx) > abs(k_yy) else "kappa_yy"
        if measured in expected:
            continue

        warnings.append(
            f"{name}: the applied load is declared on "
            f"{'/'.join(sorted(dofs))}, which is conjugate to "
            f"{'/'.join(sorted(expected))} -- but the measured field is "
            f"dominated by {measured} "
            f"(mean kappa_xx = {k_xx:.3e}, kappa_yy = {k_yy:.3e}, "
            f"{big / small:.0f}x). Either the EXTERNAL group is on the wrong "
            f"DOF/edge, or the data is not in the frame this experiment "
            f"assumes. As written, this run will report the OTHER stiffness "
            f"coefficient than you expect."
        )
    return warnings


def describe_project(project: VfmProject) -> str:
    """One-paragraph human summary, used by the GUI and the CLI alike."""
    groups: Dict[int, int] = {}
    for test in project.tests:
        groups[test.experiment_id] = groups.get(test.experiment_id, 0) + 1
    parts = ", ".join(f"experiment {k}: {v} load step(s)"
                      for k, v in sorted(groups.items()))
    return (f"{len(project.tests)} data file(s) in "
            f"{project.experiment_count()} experiment(s) ({parts}); "
            f"{project.material_type} / {project.analysis_type}; "
            f"initial {list(project.initial_parameters)}")


# =============================================================================
#  Turning a project into something the solver can run
# =============================================================================

def project_to_cases(project: VfmProject, verbose: bool = False):
    """Build (cases, experiment_ids) exactly as the campaign driver does.

    Every mesh is rebuilt from the project's own MESH_SETTINGS and pruned by
    the measurement cloud, and every BC list is fed through the same
    BoundaryConditions container the .py loader produces -- so a project run
    and a campaign_common run go through identical code.
    """
    from vfm_data_io import BoundaryConditions, load_point_measurements
    from vfm_identification import IdentificationCase
    from vfm_mesh import (
        build_mesh,
        map_points_to_mesh,
        prune_mesh_by_measurement_points,
        subset_mesh_elements,
        subset_measurements_to_mapped_mesh,
    )
    from vfm_user_inputs import Bounds, MeshSettings

    cases, experiment_ids = [], []
    for index, test in enumerate(project.tests, start=1):
        if not test.data_file.is_file():
            raise FileNotFoundError(
                f"Test {index} ({test.label or test.data_file.name}): "
                f"measurement file not found at {test.data_file}")
        if not test.mesh_settings:
            raise ValueError(
                f"Test {index} ({test.label or test.data_file.name}) has no "
                "mesh_settings; the project cannot rebuild its virtual mesh.")

        payload = dict(test.mesh_settings)
        bounds = payload.get("bounds") or {}
        mesh_settings = MeshSettings(
            number_of_elements_x=int(payload["number_of_elements_x"]),
            number_of_elements_y=int(payload["number_of_elements_y"]),
            bounds=Bounds(**{k: float(v) for k, v in bounds.items()}),
            x_nodes=payload.get("x_nodes"),
            y_nodes=payload.get("y_nodes"),
        )

        measurements = load_point_measurements(
            test.data_file,
            analysis_type=project.analysis_type,
            total_sample_area=test.sample_area_total)

        mesh_pkg = build_mesh(mesh_settings,
                              analysis_type=project.analysis_type)
        keep_cells = payload.get("keep_cells") or []
        if keep_cells:
            grid = mesh_pkg.element_grid_index
            keep_ids = []
            for cell in keep_cells:
                if cell is None or len(cell) < 2:
                    continue
                ix, iy = int(cell[0]), int(cell[1])
                if 0 <= ix < grid.shape[0] and 0 <= iy < grid.shape[1]:
                    eid = int(grid[ix, iy])
                    if eid >= 0:
                        keep_ids.append(eid)
            if keep_ids and len(set(keep_ids)) < int(mesh_pkg.n_elements):
                mesh_pkg, _ = subset_mesh_elements(mesh_pkg, keep_ids)
        mapping = map_points_to_mesh(measurements.points, mesh_pkg)
        if mapping.n_unmapped:
            n_before = int(mapping.mapped_mask.size)
            print(
                f"[VFM-plate] Test {index} ({test.data_file.name}): "
                f"{mapping.n_unmapped} of {n_before} measurement points "
                "fall outside the virtual mesh and will be excluded from "
                "the identification."
            )
            measurements, mapping = subset_measurements_to_mapped_mesh(
                measurements, mesh_pkg, mapping)
        mesh_pkg, _ = prune_mesh_by_measurement_points(
            mesh_pkg, measurements.points)

        boundary = BoundaryConditions(
            prescribed=list(test.prescribed),
            linking=list(test.linking),
            external=list(test.external),
            analysis_type=project.analysis_type,
        )

        label = test.label or f"exp{test.experiment_id}:{test.data_file.stem}"
        cases.append(IdentificationCase(
            measurements=measurements,
            mesh_pkg=mesh_pkg,
            boundary_conditions=boundary,
            label=label,
            data_file=test.data_file,
            material_orientation_deg=test.material_orientation_deg,
            applied_load_source=test.applied_load_source,
        ))
        experiment_ids.append(test.experiment_id)

        if verbose:
            print(f"  [{label}] exp {test.experiment_id}, "
                  f"{measurements.points.shape[0]} points, mesh "
                  f"{mesh_settings.number_of_elements_x}x"
                  f"{mesh_settings.number_of_elements_y}, area "
                  f"{test.sample_area_total}")

    return cases, experiment_ids


def run_project(project: VfmProject, verbose: bool = False):
    """Run the identification described by the project."""
    from vfm_gls_identification import run_gls_identification_cases
    from vfm_user_inputs import (
        GlsSettings,
        IdentificationSettings,
        LinearAlgebraSettings,
    )

    #  Always checked, printed whether or not the run is verbose: a silently
    #  swapped load DOF costs more than a few lines of console noise.
    for warning in check_load_orientation(project):
        print(f"[VFM-plate] ORIENTATION WARNING -- {warning}")

    cases, experiment_ids = project_to_cases(project, verbose=verbose)
    return run_gls_identification_cases(
        cases=cases,
        material_type=project.material_type,
        analysis_type=project.analysis_type,
        initial_parameters=project.initial_parameters,
        identification_settings=IdentificationSettings(
            max_iterations=project.max_iterations,
            convergence_tolerance=project.convergence_tolerance),
        linear_algebra_settings=LinearAlgebraSettings(),
        experiment_ids=experiment_ids,
        gls_settings=GlsSettings(
            basis_degree=project.basis_degree,
            weighting=project.weighting,
            spatial_correction=project.spatial_correction,
            covariance_tolerance=project.covariance_tolerance),
        verbose=verbose,
    )


# =============================================================================
#  Building a project from BC files already on disk
# =============================================================================

def project_test_from_bc_file(data_file: Path,
                              bc_file: Path,
                              experiment_id: int,
                              label: str = "",
                              material_orientation_deg: float = 0.0
                              ) -> ProjectTest:
    """Absorb a shipped ``boundary_conditions/*.py`` file into a project test.

    Reads the same three lists plus MESH_SETTINGS / SOLVER_SETTINGS that
    run_vfm.py would, so the project inherits exactly the boundary treatment
    that produced the published numbers.
    """
    from bc_gui.bc_model import BcDocument, mesh_settings_to_payload

    document = BcDocument.from_any_file(Path(bc_file))
    solver_settings = document.solver_settings or {}
    return ProjectTest(
        data_file=Path(data_file),
        experiment_id=int(experiment_id),
        label=label,
        description=document.description,
        prescribed=[e.to_python_dict() for e in document.prescribed],
        linking=[e.to_python_dict() for e in document.linking],
        external=[e.to_python_dict() for e in document.external],
        mesh_settings=mesh_settings_to_payload(document.mesh_settings),
        sample_area_total=solver_settings.get("sample_area_total"),
        material_orientation_deg=float(material_orientation_deg),
    )


def project_from_documents(tests: Sequence[ProjectTest],
                           material_type: str,
                           analysis_type: str,
                           initial_parameters: Sequence[float],
                           **kwargs) -> VfmProject:
    """Assemble a project from prepared tests plus the run-level settings."""
    return VfmProject(
        tests=list(tests),
        material_type=material_type,
        analysis_type=analysis_type,
        initial_parameters=[float(v) for v in initial_parameters],
        **kwargs,
    )
