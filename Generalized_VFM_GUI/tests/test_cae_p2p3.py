"""P2/P3 CAE organization: modules, jobs dock, undo, recent files, picking."""

from __future__ import annotations

import os

from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("VFMSTUDIO_NO_VISTA", "1")

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QMessageBox

from vfmstudio.core.document import ExperimentModel, LoadStepModel
from vfmstudio.gui.main_window import MainWindow
from vfmstudio.gui.session import Selection
from vfmstudio.gui.viewport.picking import parse_bc_actor_name


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_results_module_keeps_viewport(qapp):
    window = MainWindow()
    window.session.document.experiments = [ExperimentModel(experiment_id=1, name="A")]
    window._refresh_all()
    assert window.post.isHidden()
    window.set_module("results")
    assert window.session.module == "results"
    assert not window.viewport.isHidden()
    assert not window.post.isHidden()
    window.set_module("model")
    assert window.post.isHidden()
    assert not window.viewport.isHidden()
    window.session.dirty = False
    window.close()


def test_plots_fill_central_hides_viewport(qapp):
    window = MainWindow()
    window.session.document.experiments = [ExperimentModel(experiment_id=1, name="A")]
    window._refresh_all()
    window.set_module("results")
    assert not window.viewport.isHidden()
    assert not window.post.isHidden()
    assert window._splitter.opaqueResize() is False
    window.actions.get("plots_fill").setChecked(True)
    assert window.session.plots_fill_central is True
    assert window.viewport.isHidden()
    assert not window.post.isHidden()
    window.actions.get("plots_fill").setChecked(False)
    assert not window.viewport.isHidden()
    window.set_module("model")
    assert window.post.isHidden()
    window.session.dirty = False
    window.close()


def test_viewport_resize_helpers_are_safe_without_vista(qapp):
    window = MainWindow()
    window.viewport.pause_live_render()
    window.viewport.resume_live_render()
    window.session.dirty = False
    window.close()


def test_help_examples_not_on_file_menu(qapp):
    window = MainWindow()
    file_texts = [action.text() for action in window._file_menu.actions()]
    assert "Open campaign example" not in file_texts
    assert "Open Samples A/B (all)" not in file_texts
    assert window.actions.get("workflow").text() == "Workflow"
    assert window.actions.get("open_campaign").text() == "Open campaign example"
    window.close()


def test_page_down_advances_step(qapp, tmp_path):
    mat_a = tmp_path / "a.mat"
    mat_b = tmp_path / "b.mat"
    mat_a.write_bytes(b"x")
    mat_b.write_bytes(b"y")
    window = MainWindow()
    experiment = ExperimentModel(
        experiment_id=1,
        name="A",
        steps=[
            LoadStepModel(data_file=mat_a, label="a"),
            LoadStepModel(data_file=mat_b, label="b"),
        ],
    )
    window.session.document.experiments = [experiment]
    window.session.selection = Selection(kind="step", experiment_id=1, step_index=0)
    window._refresh_all()
    window._step_delta(1)
    assert window.session.selection.step_index == 1
    window._step_delta(1)
    assert window.session.selection.step_index == 1
    window._step_delta(-1)
    assert window.session.selection.step_index == 0
    window.session.dirty = False
    window.close()


def test_undo_restores_deleted_experiment_not_result(qapp, monkeypatch):
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    window = MainWindow()
    window.session.document.experiments = [
        ExperimentModel(experiment_id=1, name="keep"),
        ExperimentModel(experiment_id=2, name="gone"),
    ]
    window._reset_history()
    window.session.result = "identify-keep"
    window.session.selection = Selection(kind="experiment", experiment_id=2)
    window._apply_selection()
    window.delete_experiment()
    assert [item.name for item in window.session.document.experiments] == ["keep"]
    assert window._history.stack.canUndo()
    window._history.stack.undo()
    names = [item.name for item in window.session.document.experiments]
    assert "gone" in names
    assert window.session.result == "identify-keep"
    window.session.dirty = False
    window.close()


def test_recent_files_skip_missing(tmp_path):
    from vfmstudio.gui.prefs import recent_files, remember_file

    settings = QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat)
    existing = tmp_path / "ok.vfmproj.json"
    existing.write_text("{}", encoding="utf-8")
    missing = tmp_path / "missing.vfm"
    remember_file(missing, settings)
    remember_file(existing, settings)
    listed = recent_files(settings)
    assert existing in listed
    assert missing not in listed


def test_parse_bc_actor_name():
    selection = parse_bc_actor_name("bc:1:prescribed:0")
    assert selection is not None
    assert selection.kind == "bc"
    assert selection.experiment_id == 1
    assert selection.bc_kind == "prescribed"
    assert selection.bc_index == 0
    assert parse_bc_actor_name("bc:1:prescribed:0-label") == selection
    assert parse_bc_actor_name("mesh") is None


def test_export_vtk_passes_point_data(qapp, tmp_path, monkeypatch):
    import numpy as np
    from PySide6.QtWidgets import QFileDialog

    window = MainWindow()
    window.session.document.experiments = [ExperimentModel(experiment_id=1, name="A")]
    window.session.selection = Selection(kind="experiment", experiment_id=1)
    captured = {}

    def fake_export(path, points, quads, point_data=None):
        captured["point_data"] = point_data
        Path(path).write_bytes(b"vtk")
        return Path(path)

    monkeypatch.setattr("vfmstudio.gui.main_window.export_mesh_vtk", fake_export)
    monkeypatch.setattr(
        window.viewport,
        "displayed_point_data",
        lambda: {"kappa_yy": np.array([1.0, 2.0, 3.0, 4.0])},
    )
    monkeypatch.setattr(
        window.viewport,
        "_build_mesh",
        lambda document, experiment: object(),
    )
    monkeypatch.setattr(
        "vfmstudio.gui.main_window.mesh_xyz",
        lambda mesh: np.zeros((4, 3)),
    )
    monkeypatch.setattr(
        "vfmstudio.gui.main_window.mesh_quads",
        lambda mesh: np.array([[0, 1, 2, 3]]),
    )
    out = tmp_path / "field.vtk"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *a, **k: (str(out), "VTK (*.vtk)"),
    )
    window.export_vtk()
    assert "kappa_yy" in (captured.get("point_data") or {})
    window.session.dirty = False
    window.close()


def test_show_virtual_field_calls_viewport(qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: QMessageBox.StandardButton.Ok)
    window = MainWindow()
    window.session.result = type(
        "R", (), {"optimal_field_results": [object()]}
    )()
    called = {}

    def fake_show(session, parameter_index=0):
        called["session"] = session
        return None

    monkeypatch.setattr(window.viewport, "show_virtual_field", fake_show)
    window.show_virtual_field()
    assert called["session"] is window.session
    assert window.session.module == "results"
    window.session.dirty = False
    window.close()


def test_splash_minimum_is_one_second():
    from vfmstudio.app import SPLASH_MINIMUM_S, _hold_splash
    import inspect

    assert SPLASH_MINIMUM_S == 1.0
    default = inspect.signature(_hold_splash).parameters["minimum_s"].default
    assert default == 1.0


def test_job_dock_cancel_uses_token(qapp, monkeypatch):
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: QMessageBox.StandardButton.Ok)
    window = MainWindow()
    window.session.document.experiments = [ExperimentModel(experiment_id=1, name="e")]
    assert window._begin_job("Identifying…") is True
    window.jobs.start_job("FGLS identification", window._job_token)
    assert window.jobs._jobs[-1].status == "Running"
    token = window._job_token
    window.session.result = "keep"
    window.cancel_job()
    assert window.jobs._jobs[-1].status == "Cancelled"
    window._on_identify_finished(("stale", None, None), token)
    assert window.session.result == "keep"
    window.close()


def test_results_module_hides_idle_log_and_jobs(qapp):
    window = MainWindow()
    window.set_module("results")
    assert window.dock_jobs.isHidden()
    assert window.dock_log.isHidden()
    window.set_module("job")
    assert not window.dock_jobs.isHidden()
    window.set_module("model")
    assert window.dock_jobs.isHidden()
    assert not window.dock_log.isHidden()
    window.session.dirty = False
    window.close()


def test_session_declares_scalar_clim():
    from vfmstudio.gui.session import AppSession

    session = AppSession()
    assert session.scalar_clim is None
    assert "scalar_clim" in session.__dataclass_fields__
    assert session.results_unsaved is False


def test_identify_marks_results_unsaved(qapp, monkeypatch):
    window = MainWindow()
    monkeypatch.setattr(window, "_refresh_all", lambda: None)
    monkeypatch.setattr(window, "_finish_job", lambda *a, **k: None)
    window._job_busy = True
    window._job_token = 3
    result = type("R", (), {"optimal_field_results": []})()
    window._on_identify_finished((result, None, None), 3)
    assert window.session.results_unsaved is True
    assert window.session.dirty is True
    window.session.dirty = False
    window.session.results_unsaved = False
    window.close()


def test_maybe_save_prompts_when_results_unsaved(qapp, monkeypatch):
    window = MainWindow()
    window.session.dirty = False
    window.session.results_unsaved = True
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Discard
    )
    assert window._maybe_save() is True
    window.session.results_unsaved = False
    window.close()
