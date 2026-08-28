"""Analysis-window filtering and reference subtraction (no MAT files)."""

from pathlib import Path

import numpy as np

from vfmstudio.core.document import (
    ExperimentModel,
    LoadStepModel,
    empty_document,
)
from vfmstudio.core.jobs import (
    analysis_load_indices,
    apply_reference_subtraction,
    project_for_identification,
    subtract_reference_measurements,
    validate_analysis_window,
    validate_document,
)


class _Meas:
    def __init__(self, strains, force):
        self.strains = np.asarray(strains, dtype=float)
        self.applied_force = np.asarray(force, dtype=float)


class _Case:
    def __init__(self, strains, force):
        self.measurements = _Meas(strains, force)


def _document(tmp_path: Path, n_steps: int = 4):
    mat = tmp_path / "dummy.mat"
    mat.write_bytes(b"not-a-real-mat")
    document = empty_document()
    document.experiments.append(
        ExperimentModel(
            experiment_id=1,
            name="window",
            steps=[
                LoadStepModel(data_file=mat, label=f"step-{index + 1:02d}")
                for index in range(n_steps)
            ],
        )
    )
    return document


def test_analysis_load_indices_include_reference_outside_window(tmp_path):
    experiment = _document(tmp_path).experiments[0]
    experiment.analysis_start = 3
    experiment.analysis_end = 4
    experiment.reference_step = 1
    assert experiment.analysis_window_indices() == [2, 3]
    assert experiment.included_step_indices() == [2, 3]
    assert analysis_load_indices(experiment) == [2, 3, 0]


def test_project_for_identification_filters_steps(tmp_path):
    document = _document(tmp_path)
    document.experiments[0].analysis_start = 2
    document.experiments[0].analysis_end = 3
    project = project_for_identification(document)
    assert len(project.tests) == 2
    assert [test.label for test in project.tests] == ["step-02", "step-03"]
    document.experiments[0].reference_step = 1
    project = project_for_identification(document)
    assert [test.label for test in project.tests] == ["step-02", "step-03", "step-01"]


def test_project_for_identification_skips_excluded_experiments(tmp_path):
    document = _document(tmp_path, n_steps=2)
    other_mat = tmp_path / "other.mat"
    other_mat.write_bytes(b"not-a-real-mat")
    document.experiments.append(
        ExperimentModel(
            experiment_id=2,
            name="other",
            include_in_identification=False,
            steps=[
                LoadStepModel(data_file=other_mat, label="b-01"),
                LoadStepModel(data_file=other_mat, label="b-02"),
            ],
        )
    )
    assert document.experiments[0].include_in_identification is True
    project = project_for_identification(document)
    assert [test.experiment_id for test in project.tests] == [1, 1]
    assert [test.label for test in project.tests] == ["step-01", "step-02"]


def test_validate_requires_an_included_experiment(tmp_path):
    document = _document(tmp_path)
    document.experiments[0].include_in_identification = False
    report = validate_document(document)
    assert any("included" in item.lower() for item in report.errors)


def test_reference_subtraction_skips_experiments_not_loaded(tmp_path):
    document = _document(tmp_path, n_steps=3)
    document.experiments[0].reference_step = 1
    document.experiments.append(
        ExperimentModel(
            experiment_id=2,
            name="excluded",
            reference_step=1,
            include_in_identification=False,
            steps=list(document.experiments[0].steps),
        )
    )
    cases = [
        _Case([[1.0, 1.0]], [10.0]),
        _Case([[3.0, 5.0]], [30.0]),
        _Case([[4.0, 6.0]], [40.0]),
    ]
    kept, exp_ids, steps = apply_reference_subtraction(
        document, cases, [1, 1, 1], [0, 1, 2]
    )
    assert steps == [1, 2]
    assert exp_ids == [1, 1]
    assert np.allclose(kept[0].measurements.applied_force, [20.0])


def test_subtract_reference_measurements():
    target = _Meas([[2.0, 4.0], [6.0, 8.0]], [11.0, 21.0])
    reference = _Meas([[1.0, 1.0], [1.0, 1.0]], [1.0, 1.0])
    subtract_reference_measurements(target, reference)
    assert np.allclose(target.strains, [[1.0, 3.0], [5.0, 7.0]])
    assert np.allclose(target.applied_force, [10.0, 20.0])


def test_apply_reference_subtraction_drops_reference(tmp_path):
    document = _document(tmp_path, n_steps=3)
    document.experiments[0].reference_step = 1
    cases = [
        _Case([[1.0, 1.0]], [10.0]),
        _Case([[3.0, 5.0]], [30.0]),
        _Case([[4.0, 6.0]], [40.0]),
    ]
    kept, exp_ids, steps = apply_reference_subtraction(
        document, cases, [1, 1, 1], [0, 1, 2]
    )
    assert steps == [1, 2]
    assert exp_ids == [1, 1]
    assert np.allclose(kept[0].measurements.strains, [[2.0, 4.0]])
    assert np.allclose(kept[0].measurements.applied_force, [20.0])
    assert np.allclose(kept[1].measurements.applied_force, [30.0])


def test_reference_only_window_is_invalid(tmp_path):
    experiment = _document(tmp_path, n_steps=2).experiments[0]
    experiment.analysis_start = 1
    experiment.analysis_end = 1
    experiment.reference_step = 1
    errors = validate_analysis_window(experiment)
    assert errors


def test_collect_step_series_records_missing_files(tmp_path):
    from vfmstudio.core.step_series import collect_step_series, curvature_field_names

    document = _document(tmp_path, n_steps=2)
    series = collect_step_series(document.experiments[0], "out_of_plane")
    assert curvature_field_names("out_of_plane") == ["kappa_xx", "kappa_yy", "kappa_xy"]
    assert len(series.points) == 2
    assert all(point.error for point in series.points)
    assert series.force_labels == ["Moment"]
    assert "N\\cdot mm" in series.force_channels[0].axis_math


def test_force_components_skip_bottom_edge():
    from vfmstudio.core.document import BoundaryEntry
    from vfmstudio.core.step_series import force_components

    experiment = ExperimentModel(
        experiment_id=1,
        external=[
            BoundaryEntry(label="M_yy on top edge", dof="W_Y", edge="top"),
            BoundaryEntry(label="M_yy on bottom edge", dof="W_Y", edge="bottom"),
        ],
    )
    channels = force_components(experiment, n_force=2)
    assert len(channels) == 1
    assert channels[0].index == 0
    assert channels[0].edge == "top"
    assert "bottom" not in channels[0].combo_label.lower()
    assert r"N\cdot mm" in channels[0].axis_math


def test_field_from_selection_point_and_roi():
    from vfmstudio.core.fields import field_axis_math
    from vfmstudio.core.step_series import SpatialSelection, field_from_selection

    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    values = np.array([1.0, 3.0, 5.0, 7.0])
    whole = field_from_selection(x, y, values, SpatialSelection(kind="all"))
    assert whole == 4.0
    point = field_from_selection(
        x, y, values, SpatialSelection(kind="point", x=1.0, y=1.0)
    )
    assert point == 7.0
    roi = field_from_selection(
        x,
        y,
        values,
        SpatialSelection(kind="roi", x_min=-0.1, x_max=0.1, y_min=-0.1, y_max=1.1),
    )
    assert roi == 3.0
    assert r"mm}^{-1}" in field_axis_math("kappa_yy")
