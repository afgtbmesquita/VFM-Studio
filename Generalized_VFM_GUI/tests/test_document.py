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


def test_free_nodes_payload_round_trip():
    mesh = MeshModel(
        n_elements_x=2,
        n_elements_y=2,
        x_nodes=[0.0, 1.0, 2.0],
        y_nodes=[0.0, 1.0, 2.0],
        free_nodes=[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0]],
        keep_cells=[[0, 0], [1, 0]],
    )
    restored = MeshModel.from_payload(mesh.to_payload())
    assert restored.free_nodes == [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0]]
    assert restored.keep_cells == [[0, 0], [1, 0]]
    assert restored.x_nodes == [0.0, 1.0, 2.0]
    payload = restored.to_payload()
    assert payload["free_nodes"][0] == [0.0, 0.0]
    assert payload["keep_cells"] == [[0, 0], [1, 0]]


def test_samples_ab_all_json_round_trip():
    from vfmstudio.core.paths import samples_ab_all_project_path
    from vfmstudio.io.vfmproj_json import load_vfmproj

    path = samples_ab_all_project_path()
    if not path.is_file():
        return
    document = load_vfmproj(path)
    assert len(document.experiments) == 8
    assert [item.name for item in document.experiments] == [
        "A1_PB",
        "A2_PB",
        "A3_PB",
        "B1_PB",
        "B2_PB",
        "B3_PB",
        "A1_Twist",
        "B1_Twist",
    ]
    n_steps = sum(len(experiment.steps) for experiment in document.experiments)
    assert n_steps == 58
    for experiment in document.experiments:
        assert len(experiment.external) == 1
        assert str(experiment.external[0].edge).lower() == "top"
        assert not any(
            str(entry.edge).lower() == "bottom" for entry in experiment.external
        )
        bottom_dofs = {
            entry.dof
            for entry in experiment.prescribed
            if str(entry.edge).lower() == "bottom"
        }
        assert {"W", "W_X", "W_Y", "W_XY"} <= bottom_dofs
        for step in experiment.steps:
            assert Path(step.data_file).is_file(), step.data_file
    assert document.experiments[0].material_orientation_deg == 90.0
    assert document.experiments[3].material_orientation_deg == 0.0
    assert document.experiments[0].applied_load_source == "manual"
    assert document.experiments[0].steps[0].external_values
    assert document.experiments[0].steps[0].external_values[0] is not None
    assert document.experiments[6].applied_load_source == "mat"
    assert len(document.experiments[6].linking) == 1


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


def test_remove_boundary_entry_shifts_external_values():
    experiment = ExperimentModel(
        experiment_id=1,
        prescribed=[
            BoundaryEntry(label="fix W", dof="W", edge="bottom"),
            BoundaryEntry(label="fix RX", dof="W_X", edge="bottom"),
        ],
        linking=[
            BoundaryEntry(label="link", dof="W", edge="top"),
        ],
        external=[
            BoundaryEntry.model_validate(
                {"label": "M1", "dof": "W_Y", "edge": "top", "loadType": "distributed"}
            ),
            BoundaryEntry.model_validate(
                {"label": "M2", "dof": "W_Y", "edge": "bottom", "loadType": "distributed"}
            ),
        ],
        steps=[
            LoadStepModel(data_file="a.mat", external_values=[10.0, 20.0]),
            LoadStepModel(data_file="b.mat", external_values=[11.0]),
        ],
    )
    first = experiment.remove_boundary_entry("prescribed", 0)
    assert first is not None and first.label == "fix W"
    assert [entry.label for entry in experiment.prescribed] == ["fix RX"]
    assert experiment.remove_boundary_entry("linking", 0) is not None
    assert experiment.linking == []
    removed = experiment.remove_boundary_entry("external", 0)
    assert removed is not None and removed.label == "M1"
    assert [entry.label for entry in experiment.external] == ["M2"]
    assert experiment.steps[0].external_values == [20.0]
    assert experiment.steps[1].external_values == []
    assert experiment.remove_boundary_entry("prescribed", 9) is None


def test_clear_mesh_and_dependencies():
    experiment = ExperimentModel(
        experiment_id=1,
        mesh=MeshModel(
            n_elements_x=4,
            n_elements_y=3,
            x_nodes=[0.0, 1.0, 2.0],
            y_nodes=[0.0, 1.0],
            free_nodes=[[0.0, 0.0], [1.0, 0.0]],
            keep_cells=[[0, 0]],
        ),
        prescribed=[BoundaryEntry(label="fix W", dof="W", edge="bottom")],
        linking=[BoundaryEntry(label="link", dof="W", edge="top")],
        external=[
            BoundaryEntry.model_validate(
                {"label": "M1", "dof": "W_Y", "edge": "top", "loadType": "distributed"}
            )
        ],
        steps=[LoadStepModel(data_file="a.mat", external_values=[10.0])],
    )
    experiment.clear_mesh_and_dependencies()
    assert experiment.mesh.n_elements_x == 2
    assert experiment.mesh.n_elements_y == 2
    assert experiment.mesh.x_nodes is None
    assert experiment.mesh.free_nodes is None
    assert experiment.mesh.keep_cells is None
    assert experiment.mesh.bounds.is_unit_square()
    assert experiment.prescribed == []
    assert experiment.linking == []
    assert experiment.external == []
    assert experiment.steps[0].external_values == []


def test_remove_and_move_step_keeps_analysis_window(tmp_path):
    document = _sample_document(tmp_path)
    experiment = document.experiments[0]
    experiment.analysis_start = 2
    experiment.analysis_end = 3
    experiment.reference_step = 1
    labels = [step.label for step in experiment.steps]
    removed = experiment.remove_step(0)
    assert removed is not None and removed.label == labels[0]
    assert [step.label for step in experiment.steps] == labels[1:]
    assert experiment.analysis_start == 1
    assert experiment.analysis_end == 2
    assert experiment.reference_step is None
    assert experiment.move_step(1, -1) is True
    assert [step.label for step in experiment.steps] == [labels[2], labels[1]]
    assert experiment.move_step(0, -1) is False


def test_duplicate_experiment_copies_model_with_new_id(tmp_path):
    document = _sample_document(tmp_path)
    original = document.experiments[0]
    original.include_in_identification = False
    clone = document.duplicate_experiment(1)
    assert clone is not None
    assert clone.experiment_id == 2
    assert clone.name == "longitudinal copy"
    assert clone.mesh.n_elements_x == original.mesh.n_elements_x
    assert [step.label for step in clone.steps] == [step.label for step in original.steps]
    assert original.include_in_identification is False
    assert clone.include_in_identification is False
    original.steps[0].label = "changed"
    assert clone.steps[0].label != "changed"


def test_hdf5_identification_snapshot_round_trip(tmp_path):
    from types import SimpleNamespace

    import numpy as np

    from vfmstudio.core.uncertainty import system_from_result
    from vfmstudio.io.hdf5_project import load_vfm_project, save_vfm

    document = _sample_document(tmp_path)
    theta = np.array([400.0, 20.0, 300.0, 50.0])
    design = np.eye(4)
    observation = design @ theta
    covariance = np.eye(4)
    result = SimpleNamespace(
        final_parameters=theta,
        initial_parameters=np.ones(4),
        parameter_covariance=np.eye(4),
        global_covariance=np.eye(4),
        K_matrix=design,
        U_vector=observation,
        constitutive_model=SimpleNamespace(parameter_labels=["D11", "D12", "D22", "D66"]),
        iteration_diagnostics=[SimpleNamespace(iteration=1, parameters=theta)],
        optimal_field_results=[SimpleNamespace(nodal_variables_k=np.arange(8.0))],
        case_K_matrices=[design],
        case_U_vectors=[observation],
        experiments=[
            SimpleNamespace(
                experiment_id=1,
                label="longitudinal",
                n_steps=3,
                covariance=covariance,
                step_labels=["a", "b", "c"],
                step_estimates=np.tile(theta, (3, 1)),
            )
        ],
    )
    ident = SimpleNamespace(
        parameter_labels=["D11", "D12", "D22", "D66"],
        aggregate=SimpleNamespace(
            condition_number=12.0,
            determinant=0.5,
            used_pseudoinverse=False,
            singular_values=np.array([4.0, 3.0, 2.0, 1.0]),
        ),
        cases=[
            SimpleNamespace(
                label="longitudinal",
                CID=np.array([0.1, 0.2, 0.3]),
                points=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            )
        ],
    )
    out = tmp_path / "session.vfm"
    save_vfm(document, out, result=result, identifiability=ident)
    loaded = load_vfm_project(out)
    np.testing.assert_allclose(loaded.result.final_parameters, theta)
    np.testing.assert_allclose(loaded.result.case_K_matrices[0], design)
    np.testing.assert_allclose(
        loaded.result.optimal_field_results[0].nodal_variables_k, np.arange(8.0)
    )
    system = system_from_result(loaded.result, verify=False)
    assert system.n_clusters == 1
    np.testing.assert_allclose(system.theta, theta)
    assert loaded.identifiability.aggregate.condition_number == 12.0
    np.testing.assert_allclose(loaded.identifiability.cases[0].CID, [0.1, 0.2, 0.3])


def test_copy_boundary_conditions_is_a_deep_copy(tmp_path):
    document = _sample_document(tmp_path)
    source = document.experiments[0]
    target = document.duplicate_experiment(1)
    assert target is not None
    target.prescribed = []
    target.linking = []
    target.external = []
    copied = document.copy_boundary_conditions(source.experiment_id, target.experiment_id)
    assert copied is target
    assert len(copied.prescribed) == len(source.prescribed)
    assert copied.prescribed[0].label == source.prescribed[0].label
    copied.prescribed[0].label = "changed"
    assert source.prescribed[0].label != "changed"


def test_hdf5_stores_uncertainty_and_view_state(tmp_path):
    from types import SimpleNamespace

    import numpy as np

    from vfmstudio.io.hdf5_project import load_vfm_project, save_vfm

    document = _sample_document(tmp_path)
    theta = np.array([1.0, 2.0, 3.0, 4.0])
    result = SimpleNamespace(
        final_parameters=theta,
        constitutive_model=SimpleNamespace(parameter_labels=["D11", "D12", "D22", "D66"]),
        optimal_field_results=[SimpleNamespace(nodal_variables_k=np.arange(8.0))],
        experiments=[],
        iteration_diagnostics=[],
        case_K_matrices=[],
        case_U_vectors=[],
    )
    uq = SimpleNamespace(kind="CR2", theta=theta)
    view_state = {
        "vf_view_mode": "2D contour",
        "vf_quantity": "kx*",
        "vf_parameter_index": 2,
        "uncertainty_kind": "CR2",
        "stress_space_2d_x": "Mx",
        "stress_space_3d_z": "Mxy",
        "stress_space_step_index": -1,
    }
    out = tmp_path / "with_uq.vfm"
    save_vfm(
        document,
        out,
        result=result,
        uncertainty=uq,
        view_state=view_state,
    )
    loaded = load_vfm_project(out)
    assert loaded.uncertainty is not None
    assert loaded.uncertainty.kind == "CR2"
    np.testing.assert_allclose(loaded.uncertainty.theta, theta)
    assert loaded.view_state["vf_view_mode"] == "2D contour"
    assert loaded.view_state["vf_quantity"] == "kx*"
    assert int(loaded.view_state["vf_parameter_index"]) == 2
    assert loaded.view_state["stress_space_2d_x"] == "Mx"
    assert loaded.view_state["stress_space_3d_z"] == "Mxy"
    assert int(loaded.view_state["stress_space_step_index"]) == -1


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
    document.experiments[0].include_in_identification = False
    document.experiments[0].show_deformed_configuration = True
    document.experiments[0].deformation_scale = 8.5
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
    assert restored.experiments[0].include_in_identification is False
    assert restored.experiments[0].show_deformed_configuration is True
    assert restored.experiments[0].deformation_scale == 8.5
    assert restored.experiments[0].included_step_indices() == [1, 2]


def test_covariance_tolerance_and_bc_value_json_round_trip(tmp_path):
    from vfmstudio.io.vfmproj_json import load_vfmproj, save_vfmproj

    document = _sample_document(tmp_path)
    document.solver.covariance_tolerance = 3.0e-3
    document.experiments[0].prescribed[0].value = 1.25
    out = tmp_path / "session.vfmproj.json"
    save_vfmproj(document, out)
    restored = load_vfmproj(out)
    assert restored.solver.covariance_tolerance == 3.0e-3
    assert restored.experiments[0].prescribed[0].value == 1.25


def test_hdf5_keeps_analysis_window(tmp_path):
    from vfmstudio.io.hdf5_project import load_vfm, save_vfm

    document = _sample_document(tmp_path)
    document.experiments[0].analysis_start = 1
    document.experiments[0].analysis_end = 2
    document.experiments[0].reference_step = 1
    document.experiments[0].include_in_identification = False
    document.experiments[0].show_deformed_configuration = True
    document.experiments[0].deformation_scale = 12.0
    out = tmp_path / "session.vfm"
    save_vfm(document, out)
    restored = load_vfm(out)
    assert restored.experiments[0].analysis_end == 2
    assert restored.experiments[0].reference_step == 1
    assert restored.experiments[0].include_in_identification is False
    assert restored.experiments[0].show_deformed_configuration is True
    assert restored.experiments[0].deformation_scale == 12.0


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


def test_read_applied_force_from_file(tmp_path):
    import numpy as np
    from scipy.io import savemat

    from vfmstudio.core.fields import (
        force_index_for_group,
        mapped_force_component,
        mat_force_component,
        read_applied_force_from_file,
    )

    path = tmp_path / "load.mat"
    savemat(path, {"F": np.array([-12.5, 40.0])})
    force = read_applied_force_from_file(path)
    assert force is not None
    np.testing.assert_allclose(force, [-12.5, 40.0])
    assert mat_force_component(force, 0, 2) == -12.5
    assert mat_force_component(force, 1, 2) == 40.0
    assert mat_force_component(np.array([7.0]), 1, 2) == 7.0
    first = BoundaryEntry(label="A", dof="W_Y")
    second = BoundaryEntry(label="B", dof="W_X", force_index=0)
    assert force_index_for_group(first, 0) == 0
    assert force_index_for_group(first, 1) == 1
    assert force_index_for_group(second, 1) == 0
    swapped = BoundaryEntry(label="A", dof="W_Y", force_index=1)
    assert mapped_force_component(force, swapped, 0) == 40.0
    assert mapped_force_component(force, first, 0) == -12.5
    empty = tmp_path / "no_f.mat"
    savemat(empty, {"x": np.array([1.0])})
    assert read_applied_force_from_file(empty) is None
    assert read_applied_force_from_file(tmp_path / "missing.mat") is None


def test_solver_maps_force_index_from_mat():
    import numpy as np

    from vfmstudio.core.paths import ensure_core_on_path

    ensure_core_on_path()
    from vfm_identification import _map_mat_force_to_groups

    identity = _map_mat_force_to_groups(
        np.array([-12.5, 40.0]),
        [{"label": "A"}, {"label": "B"}],
        2,
    )
    np.testing.assert_allclose(identity, [-12.5, 40.0])
    swapped = _map_mat_force_to_groups(
        np.array([-12.5, 40.0]),
        [
            {"label": "A", "forceIndex": 1},
            {"label": "B", "forceIndex": 0},
        ],
        2,
    )
    np.testing.assert_allclose(swapped, [40.0, -12.5])
    scalar = _map_mat_force_to_groups(np.array([7.0]), [{}, {}], 2)
    np.testing.assert_allclose(scalar, [7.0, 7.0])


def test_anisotropic_sync_pads_missing_d16_d26():
    from vfmstudio.core.document import MaterialModel

    material = MaterialModel(
        material_type="anisotropic",
        analysis_type="out_of_plane",
        initial_parameters=[400.0, 50.0, 400.0, 100.0],
    )
    assert material.labels() == ["D11", "D12", "D22", "D66", "D16", "D26"]
    material.sync_parameters()
    assert material.initial_parameters == [400.0, 50.0, 400.0, 100.0, 1.0, 1.0]


def test_anisotropic_project_round_trip_keeps_six_parameters():
    from vfmstudio.core.document import MaterialModel, SessionDocument

    document = empty_document()
    document.material = MaterialModel(
        material_type="anisotropic",
        analysis_type="out_of_plane",
        initial_parameters=[400.0, 50.0, 400.0, 100.0, 50.0, 50.0],
    )
    project = document_to_vfm_project(document)
    assert list(project.initial_parameters) == [400.0, 50.0, 400.0, 100.0, 50.0, 50.0]
    restored = document_from_vfm_project(project)
    assert restored.material.material_type == "anisotropic"
    assert restored.material.initial_parameters == [
        400.0, 50.0, 400.0, 100.0, 50.0, 50.0,
    ]


def test_scratch_job_save_does_not_rebind_source_path(tmp_path):
    from vfmstudio.io.hdf5_project import (
        is_scratch_project_path,
        load_vfm_project,
        normalize_project_save_path,
        save_vfm,
    )

    document = _sample_document(tmp_path)
    project = tmp_path / "coupon.vfm"
    save_vfm(document, project)
    assert document.source_path.resolve() == project.resolve()

    scratch_dir = tmp_path / "vfmstudio_job_xyz"
    scratch_dir.mkdir()
    scratch = scratch_dir / "in.vfm"
    save_vfm(document, scratch, bind_source=False)
    assert document.source_path.resolve() == project.resolve()
    assert is_scratch_project_path(scratch)
    assert not is_scratch_project_path(project)

    loaded = load_vfm_project(scratch, bind_source=False)
    assert Path(loaded.document.source_path).resolve() == project.resolve()


def test_normalize_project_save_path_fixes_windows_json_suffix():
    from vfmstudio.io.hdf5_project import normalize_project_save_path

    json_filter = "VFM project JSON (*.vfmproj.json)"
    vfm_filter = "VFM HDF5 with results (*.vfm)"
    folder = Path("C:/data")
    assert normalize_project_save_path(folder / "in.json", json_filter) == folder / "in.vfmproj.json"
    assert normalize_project_save_path(folder / "coupon", json_filter) == folder / "coupon.vfmproj.json"
    assert normalize_project_save_path(folder / "coupon", vfm_filter) == folder / "coupon.vfm"
    assert normalize_project_save_path(folder / "coupon.vfm", vfm_filter) == folder / "coupon.vfm"
    assert (
        normalize_project_save_path(folder / "coupon.vfmproj.json", json_filter)
        == folder / "coupon.vfmproj.json"
    )


