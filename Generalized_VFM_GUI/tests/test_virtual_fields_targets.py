"""Virtual-field experiment / load-step selection."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("VFMSTUDIO_NO_VISTA", "1")

from vfmstudio.core.document import ExperimentModel, LoadStepModel, empty_document
from vfmstudio.core.virtual_fields import (
    list_virtual_field_targets,
    resolve_virtual_field_target,
    virtual_fields_for_target,
)


def _document_two_experiments():
    document = empty_document()
    document.experiments = [
        ExperimentModel(
            experiment_id=1,
            name="A1_Twist",
            reference_step=1,
            analysis_start=2,
            analysis_end=4,
            steps=[
                LoadStepModel(data_file=Path("step01.mat"), label="step01"),
                LoadStepModel(data_file=Path("step02.mat"), label="step02"),
                LoadStepModel(data_file=Path("step03.mat"), label="step03"),
                LoadStepModel(data_file=Path("step04.mat"), label="step04"),
            ],
        ),
        ExperimentModel(
            experiment_id=3,
            name="A3_PB",
            steps=[
                LoadStepModel(data_file=Path("pb_a.mat"), label="pb_a"),
                LoadStepModel(data_file=Path("pb_b.mat"), label="pb_b"),
            ],
        ),
    ]
    return document


def test_list_targets_uses_identified_steps_not_reference():
    document = _document_two_experiments()
    result = SimpleNamespace(
        experiments=[
            SimpleNamespace(
                experiment_id=1,
                label="A1_Twist",
                n_steps=3,
                step_labels=["a", "b", "c"],
            ),
            SimpleNamespace(
                experiment_id=3,
                label="A3_PB",
                n_steps=2,
                step_labels=["x", "y"],
            ),
        ]
    )
    targets = list_virtual_field_targets(document, result)
    assert [(item.experiment_id, item.step_index, item.document_step_index) for item in targets] == [
        (1, 0, 1),
        (1, 1, 2),
        (1, 2, 3),
        (3, 0, 0),
        (3, 1, 1),
    ]
    assert targets[0].step_label.startswith("Step 02")
    assert "A3_PB" in {item.experiment_name for item in targets}


def test_resolve_keeps_experiment_and_step():
    document = _document_two_experiments()
    result = SimpleNamespace(
        experiments=[
            SimpleNamespace(experiment_id=1, label="A1", n_steps=3, step_labels=[]),
            SimpleNamespace(experiment_id=3, label="A3", n_steps=2, step_labels=[]),
        ]
    )
    chosen = resolve_virtual_field_target(document, result, 3, 1)
    assert chosen is not None
    assert chosen.experiment_id == 3
    assert chosen.step_index == 1
    assert chosen.document_step_index == 1


def test_stored_fields_are_one_per_unknown():
    labels = ["D11", "D12", "D22", "D66", "D16", "D26"]
    result = SimpleNamespace(
        optimal_field_results=[
            SimpleNamespace(nodal_variables_k=np.arange(8.0) + i)
            for i in range(6)
        ],
        constitutive_model=SimpleNamespace(parameter_labels=labels),
        experiments=[
            SimpleNamespace(experiment_id=1, label="A", n_steps=4, step_labels=[]),
        ],
    )
    payload = virtual_fields_for_target(None, result, experiment_id=1, step_index=2)
    assert payload is not None
    assert len(payload.u_list) == 6
    assert payload.labels == labels
    assert payload.source == "stored"


def test_property_page_lists_experiments_and_steps():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from vfmstudio.gui.main_window import MainWindow
    from vfmstudio.gui.session import Selection

    app = QApplication.instance() or QApplication([])
    _ = app
    window = MainWindow()
    window.session.document = _document_two_experiments()
    window.session.result = SimpleNamespace(
        optimal_field_results=[SimpleNamespace(nodal_variables_k=np.zeros(8)) for _ in range(6)],
        constitutive_model=SimpleNamespace(parameter_labels=["D11"] * 6),
        experiments=[
            SimpleNamespace(experiment_id=1, label="A1_Twist", n_steps=3, step_labels=[]),
            SimpleNamespace(experiment_id=3, label="A3_PB", n_steps=2, step_labels=[]),
        ],
    )
    window._on_tree_selection(Selection(kind="post_virtual_fields"))
    assert window.props._vf_experiment.count() == 2
    assert window.props._vf_experiment.itemText(0) == "A1_Twist"
    assert window.props._vf_step.count() == 3
    window.props._vf_experiment.setCurrentIndex(1)
    assert window.session.vf_experiment_id == 3
    assert window.props._vf_step.count() == 2
    window.props._vf_step.setCurrentIndex(1)
    assert window.session.vf_step_index == 1
    assert [window.props._vf_mode.itemText(i) for i in range(window.props._vf_mode.count())] == [
        "3D surface",
        "2D contour",
    ]
    assert [window.props._vf_quantity.itemText(i) for i in range(window.props._vf_quantity.count())] == [
        "w*",
        "kx*",
        "ky*",
        "kxy*",
    ]
    window.props._vf_mode.setCurrentText("2D contour")
    assert window.session.vf_view_mode == "2D contour"
    window.props._vf_quantity.setCurrentText("kx*")
    assert window.session.vf_quantity == "kx*"
    window.session.dirty = False
    window.close()
