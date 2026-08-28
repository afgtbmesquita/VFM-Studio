"""Session document round-trip against the solver project containers."""

from pathlib import Path

from vfmstudio.core.document import (
    BoundaryEntry,
    ExperimentModel,
    LoadStepModel,
    MeshModel,
    document_from_vfm_project,
    document_to_vfm_project,
    empty_document,
    parameter_math_label,
    parameter_unit_math,
    parameter_unit_plain,
)


def _sample_document(tmp_path: Path):
    mat = tmp_path / "dummy.mat"
    mat.write_bytes(b"not-a-real-mat")
    document = empty_document()
    document.material.material_type = "orthotropic"
    document.material.analysis_type = "out_of_plane"
    document.experiments.append(
        ExperimentModel(
            experiment_id=1,
            name="longitudinal",
            material_orientation_deg=90.0,
            sample_area_total=144.0,
            applied_load_source="mat",
            mesh=MeshModel(n_elements_x=2, n_elements_y=2),
            prescribed=[
                BoundaryEntry(label="top w", dof="W", edge="top", value=0.0)
            ],
            external=[
                BoundaryEntry.model_validate(
                    {
                        "label": "M_yy",
                        "dof": "W_Y",
                        "edge": "top",
                        "loadType": "distributed",
                        "sortDirection": "x",
                    }
                )
            ],
            steps=[
                LoadStepModel(data_file=mat, label="step-01"),
                LoadStepModel(data_file=mat, label="step-02"),
                LoadStepModel(data_file=mat, label="step-03"),
            ],
        )
    )
    return document


def test_empty_document_defaults():
    document = empty_document()
    assert document.material.material_type == "orthotropic"
    assert document.solver.weighting == "gls"


def test_parameter_math_and_units():
    assert parameter_math_label("D11") == r"$D_{11}$"
    assert parameter_math_label("A12") == r"$A_{12}$"
    assert parameter_unit_plain("D22") == "N·mm"
    assert parameter_unit_plain("A11") == "N/mm"
    assert parameter_unit_math("D11") == r"N$\cdot$mm"
    assert parameter_unit_math("A66") == "N/mm"


def test_document_to_vfm_and_back(tmp_path):
    document = _sample_document(tmp_path)
    project = document_to_vfm_project(document)
    assert len(project.tests) == 3
    assert project.tests[0].experiment_id == 1
    assert project.material_type == "orthotropic"
    restored = document_from_vfm_project(project)
    assert restored.experiments[0].name == "longitudinal"
    assert restored.experiments[0].steps[2].label == "step-03"
    assert restored.experiments[0].prescribed[0].dof == "W"
    assert restored.experiments[0].mesh.n_elements_x == 2
    payload = restored.experiments[0].mesh.to_payload()
    assert payload["number_of_elements_x"] == 2
    assert restored.experiments[0].included_step_indices() == [0, 1, 2]


def test_campaign_json_round_trip():
    from vfmstudio.core.paths import campaign_project_path
    from vfmstudio.io.vfmproj_json import load_vfmproj

    path = campaign_project_path()
    if not path.is_file():
        return
    document = load_vfmproj(path)
    assert document.experiments
    assert document.material.material_type == "orthotropic"
    assert document.experiments[0].mesh.n_elements_x >= 1
    project = document_to_vfm_project(document)
    assert len(project.tests) == sum(len(exp.steps) for exp in document.experiments)


def test_hdf5_round_trip(tmp_path):
    from vfmstudio.io.hdf5_project import load_vfm, save_vfm

    document = _sample_document(tmp_path)
    out = tmp_path / "session.vfm"
    save_vfm(document, out)
    restored = load_vfm(out)
    assert restored.experiments[0].name == "longitudinal"
    assert restored.experiments[0].mesh.n_elements_x == 2


def test_validate_empty_has_error():
    from vfmstudio.core.jobs import validate_document

    report = validate_document(empty_document())
    assert report.errors


def test_analysis_window_indices(tmp_path):
    document = _sample_document(tmp_path)
    experiment = document.experiments[0]
    assert experiment.included_step_indices() == [0, 1, 2]
    experiment.analysis_start = 2
    experiment.analysis_end = 3
    assert experiment.analysis_window_indices() == [1, 2]
    assert experiment.included_step_indices() == [1, 2]
    experiment.reference_step = 2
    assert experiment.included_step_indices() == [2]
    experiment.reference_step = 1
    assert experiment.included_step_indices() == [1, 2]
    experiment.analysis_start = 3
    experiment.analysis_end = 1
    assert experiment.analysis_window_indices() == []
    assert experiment.included_step_indices() == []


def test_analysis_window_validation(tmp_path):
    from vfmstudio.core.jobs import validate_analysis_window

    document = _sample_document(tmp_path)
    experiment = document.experiments[0]
    experiment.analysis_start = 2
    experiment.analysis_end = 3
    experiment.reference_step = 2
    assert validate_analysis_window(experiment) == []
    experiment.reference_step = 2
    experiment.analysis_start = 2
    experiment.analysis_end = 2
    errors = validate_analysis_window(experiment)
    assert errors
    experiment.analysis_start = 9
    experiment.analysis_end = 3
    experiment.reference_step = None
    errors = validate_analysis_window(experiment)
    assert any("outside" in item for item in errors)


def test_vfmproj_analysis_sidecar(tmp_path):
    from vfmstudio.io.vfmproj_json import load_vfmproj, save_vfmproj

    document = _sample_document(tmp_path)
    document.experiments[0].analysis_start = 2
    document.experiments[0].analysis_end = 3
    document.experiments[0].reference_step = 1
    document.experiments[0].fit_data_to_system_axes = True
    out = tmp_path / "session.vfmproj.json"
    save_vfmproj(document, out)
    payload = out.read_text(encoding="utf-8")
    assert '"vfmstudio"' in payload
    assert '"reference_step": 1' in payload
    restored = load_vfmproj(out)
    assert restored.experiments[0].analysis_start == 2
    assert restored.experiments[0].analysis_end == 3
    assert restored.experiments[0].reference_step == 1
    assert restored.experiments[0].fit_data_to_system_axes is True
    assert restored.experiments[0].included_step_indices() == [1, 2]


def test_hdf5_keeps_analysis_window(tmp_path):
    from vfmstudio.io.hdf5_project import load_vfm, save_vfm

    document = _sample_document(tmp_path)
    document.experiments[0].analysis_start = 1
    document.experiments[0].analysis_end = 2
    document.experiments[0].reference_step = 1
    out = tmp_path / "session.vfm"
    save_vfm(document, out)
    restored = load_vfm(out)
    assert restored.experiments[0].analysis_end == 2
    assert restored.experiments[0].reference_step == 1


def test_detected_bounds_replace_unit_square():
    from vfmstudio.core.document import BoundsModel, apply_detected_mesh_bounds

    document = empty_document()
    document.experiments.append(ExperimentModel(experiment_id=1, name="exp"))
    experiment = document.experiments[0]
    assert experiment.mesh.bounds.is_unit_square()
    changed = apply_detected_mesh_bounds(
        experiment, BoundsModel(x_min=1.5, x_max=8.25, y_min=-2.0, y_max=4.5)
    )
    assert changed is True
    assert experiment.mesh.bounds.x_min == 1.5
    assert experiment.mesh.bounds.x_max == 8.25
    assert experiment.mesh.bounds.y_min == -2.0
    assert experiment.mesh.bounds.y_max == 4.5


def test_detected_bounds_union_when_multiple_steps(tmp_path):
    from vfmstudio.core.document import BoundsModel, apply_detected_mesh_bounds

    document = _sample_document(tmp_path)
    experiment = document.experiments[0]
    experiment.mesh.bounds = BoundsModel(x_min=0.0, x_max=5.0, y_min=0.0, y_max=5.0)
    apply_detected_mesh_bounds(
        experiment, BoundsModel(x_min=-1.0, x_max=2.0, y_min=4.0, y_max=9.0)
    )
    assert experiment.mesh.bounds.x_min == -1.0
    assert experiment.mesh.bounds.x_max == 5.0
    assert experiment.mesh.bounds.y_min == 0.0
    assert experiment.mesh.bounds.y_max == 9.0


def test_detect_xy_bounds_from_mat(tmp_path):
    import numpy as np
    from scipy.io import savemat

    from vfmstudio.core.fields import detect_xy_bounds, sync_mesh_bounds_from_file

    path = tmp_path / "cloud.mat"
    savemat(
        path,
        {
            "x": np.array([2.0, 8.0, 2.0, 8.0]),
            "y": np.array([-1.0, -1.0, 4.0, 4.0]),
        },
    )
    detected = detect_xy_bounds(path)
    assert detected is not None
    assert detected.x_min == 2.0
    assert detected.x_max == 8.0
    assert detected.y_min == -1.0
    assert detected.y_max == 4.0

    experiment = ExperimentModel(experiment_id=1, name="exp")
    experiment.steps.append(LoadStepModel(data_file=path, label="s1"))
    assert sync_mesh_bounds_from_file(experiment, path) is True
    assert experiment.mesh.bounds.x_min == 2.0
    assert experiment.mesh.bounds.y_max == 4.0

