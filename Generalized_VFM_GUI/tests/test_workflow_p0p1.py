"""P0/P1 workflow: targeting, save prompt, jobs, import dialog, steps."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("VFMSTUDIO_NO_VISTA", "1")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QMessageBox

from vfmstudio.core.document import ExperimentModel, LoadStepModel
from vfmstudio.gui.main_window import MainWindow
from vfmstudio.gui.session import Selection


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _yes(*args, **kwargs):
    return QMessageBox.StandardButton.Yes


def _no(*args, **kwargs):
    return QMessageBox.StandardButton.No


def test_delete_experiment_requires_target(qapp, monkeypatch):
    window = MainWindow()
    window.session.document.experiments = [
        ExperimentModel(experiment_id=1, name="keep"),
        ExperimentModel(experiment_id=2, name="other"),
    ]
    window._refresh_all()
    window._on_tree_selection(Selection(kind="material"))
    monkeypatch.setattr(QMessageBox, "question", _yes)
    window.delete_experiment()
    assert [item.name for item in window.session.document.experiments] == ["keep", "other"]
    window._on_tree_selection(Selection(kind="experiment", experiment_id=1))
    window.delete_experiment()
    assert [item.name for item in window.session.document.experiments] == ["other"]
    window.session.dirty = False
    window.close()


def test_maybe_save_cancel_save_as_blocks(qapp, monkeypatch, tmp_path):
    window = MainWindow()
    window.session.dirty = True
    window.session.document.source_path = None
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *a, **k: QMessageBox.StandardButton.Save,
    )
    monkeypatch.setattr(
        "vfmstudio.gui.main_window.QFileDialog.getSaveFileName",
        lambda *a, **k: ("", ""),
    )
    assert window._maybe_save() is False
    window.session.dirty = False
    window.close()


def test_save_overwrites_current_project_not_scratch(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    project = tmp_path / "coupon.vfm"
    assert window._write(project) is True
    assert window.session.path.resolve() == project.resolve()
    window.session.document.description = "edited"
    assert window.save_project() is True
    assert window.session.path.resolve() == project.resolve()
    assert list(tmp_path.glob("in.*")) == []

    scratch_dir = tmp_path / "vfmstudio_job_abc"
    scratch_dir.mkdir()
    window.session.document.source_path = scratch_dir / "in.vfm"
    restored = tmp_path / "restored.vfm"
    monkeypatch.setattr(
        "vfmstudio.gui.main_window.QFileDialog.getSaveFileName",
        lambda *a, **k: (str(restored), "VFM HDF5 with results (*.vfm)"),
    )
    assert window.save_project() is True
    assert window.session.path.resolve() == restored.resolve()
    window.session.dirty = False
    window.close()


def test_add_step_uses_add_load_steps_dialog(qapp, monkeypatch, tmp_path):
    from vfmstudio.gui.dialogs.import_steps import AddLoadStepsDialog

    mat = tmp_path / "step03.mat"
    mat.write_bytes(b"x")
    window = MainWindow()
    experiment = ExperimentModel(experiment_id=1, name="exp")
    window.session.document.experiments = [experiment]
    window.session.selection = Selection(kind="experiment", experiment_id=1)
    window._refresh_all()

    class FakeDialog(AddLoadStepsDialog):
        def __init__(self, parent=None, start_directory="", source_path=None):
            super().__init__(
                parent, start_directory=start_directory, source_path=source_path
            )
            self._result_steps = [LoadStepModel(data_file=mat, label="step03")]

    monkeypatch.setattr("vfmstudio.gui.main_window.AddLoadStepsDialog", FakeDialog)
    monkeypatch.setattr(window, "_exec_dialog", lambda dialog, name: 1)
    window._add_step(1)
    assert len(experiment.steps) == 1
    assert experiment.steps[0].label == "step03"
    window.session.dirty = False
    window.close()


def test_import_mat_uses_import_mat_dialog(qapp, monkeypatch, tmp_path):
    from vfmstudio.gui.dialogs.import_mat import ImportMatDialog

    mat = tmp_path / "step03.mat"
    mat.write_bytes(b"x")
    window = MainWindow()
    experiment = ExperimentModel(experiment_id=1, name="exp")
    window.session.document.experiments = [experiment]
    window.session.selection = Selection(kind="experiment", experiment_id=1)
    window._refresh_all()

    class FakeDialog(ImportMatDialog):
        def __init__(self, parent=None, start_directory=""):
            super().__init__(parent, start_directory=start_directory)
            self._paths = [mat]

    monkeypatch.setattr("vfmstudio.gui.main_window.ImportMatDialog", FakeDialog)
    monkeypatch.setattr(window, "_exec_dialog", lambda dialog, name: 1)
    window._import_mat(1)
    assert len(experiment.steps) == 1
    assert experiment.steps[0].label == "step03"
    window.session.dirty = False
    window.close()


def test_add_step_mat_radio_appends_steps(qapp, monkeypatch, tmp_path):
    from vfmstudio.gui.dialogs.import_steps import AddLoadStepsDialog

    mat = tmp_path / "step02.mat"
    mat.write_bytes(b"x")
    window = MainWindow()
    experiment = ExperimentModel(experiment_id=1, name="exp")
    window.session.document.experiments = [experiment]
    window.session.selection = Selection(kind="experiment", experiment_id=1)
    window._refresh_all()

    class FakeDialog(AddLoadStepsDialog):
        def __init__(self, parent=None, start_directory="", source_path=None):
            super().__init__(
                parent, start_directory=start_directory, source_path=source_path
            )
            self.set_source_mode("mat")
            self._mat_list.set_paths([mat])
            self._result_steps = self.steps()

    monkeypatch.setattr("vfmstudio.gui.main_window.AddLoadStepsDialog", FakeDialog)
    monkeypatch.setattr(window, "_exec_dialog", lambda dialog, name: 1)
    window._add_step(1)
    assert [step.label for step in experiment.steps] == ["step02"]
    window.session.dirty = False
    window.close()


def test_job_guard_ignores_stale_result(qapp, monkeypatch):
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: QMessageBox.StandardButton.Ok)
    window = MainWindow()
    window.session.document.experiments = [ExperimentModel(experiment_id=1, name="e")]
    assert window._begin_job("Identifying…") is True
    assert window._begin_job("Identifying…") is False
    token = window._job_token
    window.session.result = "keep"
    window.cancel_job()
    window._on_identify_finished(("stale", None, None), token)
    assert window.session.result == "keep"
    window.close()


def test_workflow_text_matches_current_ui():
    from vfmstudio.gui.dialogs.workflow import WORKFLOW_TEXT

    assert "Cancel stops the solver process" in WORKFLOW_TEXT
    assert "Fit mesh to DIC" in WORKFLOW_TEXT
    assert ".vfm" in WORKFLOW_TEXT
    assert "Properties" in WORKFLOW_TEXT
    assert "Import MAT files" in WORKFLOW_TEXT
    assert "Add load steps" in WORKFLOW_TEXT


def test_job_runner_usage_returns_error():
    from vfmstudio.core.job_runner import main

    assert main([]) == 2


def test_duplicate_and_delete_step_from_window(qapp, monkeypatch, tmp_path):
    mat_a = tmp_path / "a.mat"
    mat_b = tmp_path / "b.mat"
    mat_a.write_bytes(b"x")
    mat_b.write_bytes(b"y")
    window = MainWindow()
    experiment = ExperimentModel(
        experiment_id=1,
        name="A1",
        steps=[
            LoadStepModel(data_file=mat_a, label="a"),
            LoadStepModel(data_file=mat_b, label="b"),
        ],
        analysis_start=1,
        analysis_end=2,
    )
    window.session.document.experiments = [experiment]
    window._refresh_all()
    window.duplicate_experiment(1)
    assert len(window.session.document.experiments) == 2
    assert window.session.document.experiments[1].name == "A1 copy"
    monkeypatch.setattr(QMessageBox, "question", _yes)
    window._delete_step(1, 0)
    assert [step.label for step in experiment.steps] == ["b"]
    window.session.dirty = False
    window.close()
