"""Model build agent: imported-data gate, linear window, mesh bounds."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("VFMSTUDIO_NO_VISTA", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
# Builds here must not depend on the campaign's saved projects.
os.environ["VFMSTUDIO_PRECEDENTS"] = "off"

from vfmstudio.agents import build_arguments, get_agent
from vfmstudio.agents.model_build.apply import apply_to_document
from vfmstudio.agents.model_build.measure import ModelRefused
from vfmstudio.agents.model_build.session import agent_request, build_model
from vfmstudio.core.bc_entries import make_boundary_entries
from vfmstudio.core.document import ExperimentModel, LoadStepModel, MaterialModel, SessionDocument
from vfmstudio.io.csv_import import write_step_mat

LOADS = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0)
K_PER_LOAD = 2.0e-4


def _cloud(nx: int = 11, ny: int = 21, pitch: float = 1.0):
    x, y = np.meshgrid(np.arange(nx) * pitch, np.arange(ny) * pitch)
    return x.ravel().astype(float), y.ravel().astype(float)


def write_steps(folder: Path, loads=LOADS, *, kxx_scale: float = 0.0, saturate: bool = True, labels=("Tx", "Ty")) -> list[Path]:
    """Uniform bending about X (κyy follows Ty), with an optional nonlinear last step."""
    folder.mkdir(parents=True, exist_ok=True)
    x, y = _cloud()
    paths = []
    peak = max(abs(value) for value in loads[:-1]) if saturate and len(loads) > 1 else None
    for index, load in enumerate(loads, start=1):
        used = load
        if saturate and peak is not None and index == len(loads):
            used = peak
        kyy = K_PER_LOAD * used
        kxx = kxx_scale * K_PER_LOAD * used
        payload = {
            "nodes": np.column_stack([x, y]),
            "X": x,
            "Y": y,
            "SK1grid": np.full(x.size, kxx),
            "SK2grid": np.full(x.size, kyy),
            "SK3grid": np.zeros(x.size),
        }
        path = folder / f"step{index:02d}.mat"
        write_step_mat(path, payload, force=np.array([0.0, load]), force_labels=list(labels))
        paths.append(path)
    return paths


def _document(paths: list[Path], *, name: str = "Longitudinal", material: str = "orthotropic") -> tuple[SessionDocument, ExperimentModel]:
    experiment = ExperimentModel(
        experiment_id=1,
        name=name,
        steps=[
            LoadStepModel(data_file=path, label=path.stem)
            for path in paths
        ],
    )
    document = SessionDocument(
        material=MaterialModel(
            material_type=material,
            analysis_type="out_of_plane",
            initial_parameters=[1.0, 1.0, 1.0, 1.0] if material != "iso" else [1.0, 1.0],
        ),
        experiments=[experiment],
    )
    return document, experiment


def test_refuses_until_data_is_imported(tmp_path):
    with pytest.raises(ModelRefused) as empty:
        build_model(SessionDocument(), workdir=tmp_path / "empty")
    assert "no experiments" in empty.value.reasons[0].lower()
    bare = ExperimentModel(experiment_id=1, name="Bare", steps=[])
    with pytest.raises(ModelRefused) as no_steps:
        build_model(SessionDocument(experiments=[bare]), workdir=tmp_path / "nosteps")
    assert "no load steps" in no_steps.value.reasons[0].lower()
    missing = ExperimentModel(
        experiment_id=1,
        name="Missing",
        steps=[LoadStepModel(data_file=tmp_path / "gone.mat", label="gone")],
    )
    with pytest.raises(ModelRefused) as gone:
        build_model(SessionDocument(experiments=[missing]), workdir=tmp_path / "gone")
    assert "were found" in gone.value.reasons[0].lower()
    assert not (tmp_path / "empty").exists()


def test_fits_bounds_and_linear_window_without_choosing_a_model(tmp_path):
    paths = write_steps(tmp_path / "mats")
    document, experiment = _document(paths)
    experiment.mesh.n_elements_x = 4
    experiment.mesh.n_elements_y = 3
    before = experiment.model_dump()
    run = build_model(document, workdir=tmp_path / "work")
    assert experiment.model_dump() == before
    (built,) = run.builds
    assert built.measurements["linear_resolved"]
    assert "curvature_class" not in built.measurements
    assert built.analysis_start == 1
    assert built.analysis_end == 7
    assert built.reference_step == 1
    assert built.included_steps == [2, 3, 4, 5, 6, 7]
    assert built.mesh["number_of_elements_x"] == 4
    assert built.mesh["number_of_elements_y"] == 3
    assert built.mesh["bounds"]["x_min"] == pytest.approx(0.0)
    assert built.mesh["bounds"]["x_max"] == pytest.approx(10.0)
    assert built.mesh["bounds"]["y_max"] == pytest.approx(20.0)
    assert built.sample_area == pytest.approx(11 * 21)
    assert built.mesh_check["n_nodes"] == 20
    assert built.mesh_check["n_elements"] == 12
    assert "admissible_nullity" not in built.mesh_check
    assert not built.prescribed and not built.external
    assert not built.checks
    assert any("boundary conditions" in item for item in built.warnings)
    assert not run.solver_errors
    kinds = {question["kind"] for question in built.questions}
    assert kinds == {"material_orientation", "boundary_conditions"}
    lowered = run.report_markdown.lower()
    assert "4×3" in run.report_markdown
    assert "dominance" not in lowered and "curvature class" not in lowered
    assert "bending axis" in lowered
    prompt, display = agent_request(run)
    assert "facts.json" in prompt and "Longitudinal" in display
    assert "boundary conditions" in prompt
    facts = run.facts_path.read_text(encoding="utf-8")
    assert "2 by 2" not in facts
    assert run.facts_path.is_file() and run.report_path.is_file()

    notes = apply_to_document(document, run.builds, {"1.material_orientation": 90.0})
    assert "wrote" in notes[0]
    assert experiment.material_orientation_deg == pytest.approx(90.0)
    assert experiment.analysis_start == 1 and experiment.analysis_end == 7
    assert experiment.reference_step == 1
    assert experiment.mesh.n_elements_x == 4 and experiment.mesh.n_elements_y == 3
    assert not experiment.prescribed and not experiment.external
    assert experiment.mesh.bounds.x_max == pytest.approx(10.0)
    assert experiment.sample_area_total == pytest.approx(11 * 21)


def test_two_curvature_components_do_not_select_boundary_conditions(tmp_path):
    paths = write_steps(tmp_path / "mats", kxx_scale=0.9, saturate=False)
    document, experiment = _document(paths, name="Off axis")
    run = build_model(document, workdir=tmp_path / "work")
    (built,) = run.builds
    assert built.measurements["linear_resolved"]
    scores = built.measurements["curvature_r2"]
    assert scores["kxx"] > 0.99 and scores["kyy"] > 0.99
    assert not any(question["kind"] == "pattern" for question in built.questions)
    assert not built.prescribed and not built.external
    notes = apply_to_document(document, run.builds, {"1.material_orientation": 45.0})
    assert "wrote" in notes[0]
    assert not experiment.prescribed and not experiment.external
    assert experiment.material_orientation_deg == pytest.approx(45.0)


def test_several_load_channels_leave_the_window_open(tmp_path):
    folder = tmp_path / "both"
    folder.mkdir()
    x, y = _cloud()
    paths = []
    for index, load in enumerate(LOADS, start=1):
        payload = {
            "nodes": np.column_stack([x, y]),
            "SK1grid": np.full(x.size, K_PER_LOAD * load),
            "SK2grid": np.zeros(x.size),
            "SK3grid": np.zeros(x.size),
        }
        path = folder / f"step{index:02d}.mat"
        write_step_mat(path, payload, force=np.array([load, 0.5 * load]), force_labels=["Tx", "Ty"])
        paths.append(path)
    document, _experiment = _document(paths)
    run = build_model(document, workdir=tmp_path / "work")
    (built,) = run.builds
    assert not built.measurements["load_resolved"]
    assert not built.measurements["linear_resolved"]
    assert any(question["kind"] == "linear_window" for question in built.questions)
    assert not built.external


def test_hand_mesh_is_replaced_only_when_asked(tmp_path):
    paths = write_steps(tmp_path / "mats")
    document, experiment = _document(paths)
    experiment.mesh.n_elements_x = 4
    experiment.mesh.n_elements_y = 3
    experiment.mesh.x_nodes = [0.0, 5.0, 10.0]
    experiment.prescribed = make_boundary_entries("prescribed", ("W",), edge="left")
    run = build_model(document, workdir=tmp_path / "work")
    (built,) = run.builds
    assert any(question["kind"] == "replace_existing" and question["value"] is False for question in built.questions)
    assert not any(question["kind"] == "boundary_conditions" for question in built.questions)
    apply_to_document(document, run.builds, {})
    assert experiment.mesh.x_nodes == [0.0, 5.0, 10.0]
    assert len(experiment.prescribed) == 1 and experiment.prescribed[0].edge == "left"
    assert not experiment.external
    apply_to_document(document, run.builds, {"1.replace_existing": True, "1.material_orientation": 0.0})
    assert experiment.mesh.x_nodes is None
    assert experiment.mesh.n_elements_x == 4 and experiment.mesh.n_elements_y == 3
    assert experiment.mesh.bounds.x_max == pytest.approx(10.0)
    assert len(experiment.prescribed) == 1 and experiment.prescribed[0].edge == "left"
    assert not experiment.external
    assert built.mesh_check.get("admissible_nullity", 0) > 0


def test_iso_coupon_still_does_not_invent_boundary_conditions(tmp_path):
    folder = tmp_path / "scalar"
    folder.mkdir()
    x, y = _cloud()
    paths = []
    for index, load in enumerate(LOADS[:-1], start=1):
        payload = {
            "nodes": np.column_stack([x, y]),
            "SK1grid": np.zeros(x.size),
            "SK2grid": np.full(x.size, K_PER_LOAD * load),
            "SK3grid": np.zeros(x.size),
        }
        path = folder / f"step{index:02d}.mat"
        write_step_mat(path, payload, force=np.array([load]), force_labels=["Ty"])
        paths.append(path)
    document, experiment = _document(paths, material="iso")
    experiment.external = make_boundary_entries("external", ("W_Y",), edge="top")
    run = build_model(document, workdir=tmp_path / "work")
    (built,) = run.builds
    assert built.measurements["linear_resolved"] and not built.questions
    assert len(built.external) == 1 and built.external[0]["dof"] == "W_Y"
    from vfmstudio.agents.model_build.audit import audit_reply

    assert audit_reply("Nothing is left to choose.", run.workdir) is None
    note = audit_reply("Should the fiber angle be 12.5 deg?", run.workdir)
    assert "12.5" in note and "no input unresolved" in note
    apply_to_document(document, run.builds, {})
    assert len(experiment.external) == 1 and not experiment.prescribed


def test_precedent_from_a_saved_project_is_offered_and_adopted(tmp_path):
    from vfmstudio.agents.model_build.audit import audit_reply
    from vfmstudio.agents.model_build.precedents import harvest
    from vfmstudio.io.hdf5_project import save_any

    past, past_experiment = _document(write_steps(tmp_path / "past_mats"), name="Past")
    past_experiment.prescribed = make_boundary_entries("prescribed", ("W", "W_X", "W_Y", "W_XY"), edge="bottom")
    past_experiment.external = make_boundary_entries("external", ("W_Y",), edge="top")
    past_experiment.mesh.n_elements_x = 3
    past_experiment.mesh.n_elements_y = 5
    (tmp_path / "projects" / "vfmstudio_job_1").mkdir(parents=True)
    save_any(past, tmp_path / "projects" / "past.vfm")
    save_any(past, tmp_path / "projects" / "vfmstudio_job_1" / "in.vfm")
    library = harvest([tmp_path / "projects"])
    assert len(library["projects"]) == 1
    (precedent,) = library["precedents"]
    assert precedent["name"] == "Past" and precedent["signature"]["moving_channels"] == ["Ty"]

    document, experiment = _document(write_steps(tmp_path / "mats"), name="New")
    run = build_model(document, workdir=tmp_path / "work", library=library)
    (built,) = run.builds
    (scheme,) = library["schemes"]
    assert built.bc_reference[0]["id"] == scheme["id"]
    assert built.bc_reference[0]["comparison"]["curvature_direction_match"] == pytest.approx(1.0)
    assert ".vfm" not in json.dumps(built.bc_reference)
    (candidate,) = built.bc_candidates
    assert candidate["transferable"] and candidate["element_counts"] == {"x": 3, "y": 5}
    assert candidate["external"][0]["forceIndex"] == 1
    assert candidate["check"]["admissible_nullity"] > 0 and not candidate["check"]["problems"]
    (question,) = [item for item in built.questions if item["kind"] == "boundary_conditions"]
    assert [option["value"] for option in question["options"]] == [None, candidate["id"]]
    assert (run.workdir / "bc_reference.json").is_file()
    assert not (run.workdir / "precedents.json").exists()
    reference = (run.workdir / "bc_reference.json").read_text(encoding="utf-8")
    assert "past.vfm" not in reference and "projects" not in json.loads(reference)
    prompt, _display = agent_request(run)
    assert "Boundary-condition training cards" in run.report_markdown
    assert "bc_reference.json" in prompt and "Do not open .vfm" in prompt
    assert audit_reply(f"Set {candidate['id']} gives admissible virtual DOFs {candidate['check']['admissible_nullity']}.", run.workdir) is None

    apply_to_document(document, run.builds, {"1.material_orientation": 0.0})
    assert not experiment.prescribed and not experiment.external
    notes = apply_to_document(document, run.builds, {"1.material_orientation": 0.0, question["id"]: candidate["id"]})
    assert candidate["id"] in notes[0]
    assert [entry.dof for entry in experiment.prescribed] == ["W", "W_X", "W_Y", "W_XY"]
    assert experiment.external[0].edge == "top" and experiment.external[0].force_index == 1
    assert experiment.mesh.n_elements_x == 3 and experiment.mesh.n_elements_y == 5


def test_model_agent_is_locked_down():
    spec = get_agent("model_build")
    args = build_arguments(spec)
    tools = args[args.index("--tools") + 1 : args.index("--allowedTools")]
    assert tools == ["Read", "Glob"]
    denied = args[args.index("--disallowedTools") + 1 :]
    for name in ("Bash", "PowerShell", "Edit", "Write", "WebFetch", "WebSearch"):
        assert name in denied
    assert "--restricted" in args and "--strict-mcp-config" in args
    prompt = args[args.index("--append-system-prompt") + 1]
    assert "Ask the user only for the items in unresolved" in prompt
    assert "data_agent/<experiment>/facts.json" in prompt and "bc_reference.json" in prompt
    assert "precedents.json" not in prompt
    assert "Do not open .vfm" in prompt
    assert "2 by 2" not in prompt
    assert "dominant" not in prompt.lower()
    assert "clamped" not in prompt.lower()
    assert spec.launch_label == "Use model agent"
    assert spec.model is None
    assert not spec.gui_context


def test_training_library_is_a_compact_bc_reference(tmp_path):
    from vfmstudio.agents.model_build.library import (
        as_training_library,
        load_library,
        rank_schemes,
        record_feedback,
    )
    from vfmstudio.agents.model_build.precedents import harvest
    from vfmstudio.io.hdf5_project import save_any

    past, past_experiment = _document(write_steps(tmp_path / "past_mats"), name="Past")
    past_experiment.prescribed = make_boundary_entries("prescribed", ("W", "W_X", "W_Y", "W_XY"), edge="bottom")
    past_experiment.external = make_boundary_entries("external", ("W_Y",), edge="top")
    past_experiment.mesh.n_elements_x = 3
    past_experiment.mesh.n_elements_y = 5
    (tmp_path / "projects").mkdir()
    save_any(past, tmp_path / "projects" / "past.vfm")
    harvested = harvest([tmp_path / "projects"])
    assert any(str(path).endswith(".vfm") for path in [item["project"] for item in harvested["projects"]])

    trained = as_training_library(harvested)
    blob = json.dumps(trained)
    assert trained["schemes"] and ".vfm" not in blob and "G:\\\\" not in blob
    (scheme,) = trained["schemes"]
    assert "Hold" in scheme["lesson"] and scheme["typical_mesh"] == {"x": 3, "y": 5}
    assert scheme["use_when"]["moving_channels"] == ["Ty"]

    measured = build_model(
        _document(write_steps(tmp_path / "mats"), name="New")[0],
        workdir=tmp_path / "work",
        library=trained,
    )
    reference = json.loads((measured.workdir / "bc_reference.json").read_text(encoding="utf-8"))
    assert "past.vfm" not in json.dumps(reference)
    assert len(reference["experiments"]["New"]) <= 3

    store = tmp_path / "bc_library.json"
    store.write_text(json.dumps(trained), encoding="utf-8")
    os.environ["VFMSTUDIO_PRECEDENTS"] = str(store)
    try:
        record_feedback(scheme["id"], "adoption")
        record_feedback(scheme["id"], "adoption")
        record_feedback(scheme["id"], "decline")
        reloaded = load_library()
        scores = reloaded["schemes"][0]["scores"]
        assert scores["adoptions"] == 2 and scores["declines"] == 1
        assert scores["reward"] == scores["identifications"] + 2 * 2 - 2 * 1
        cards = rank_schemes(reloaded, measured.builds[0].signature, "orthotropic")
        assert cards[0]["id"] == scheme["id"]
    finally:
        os.environ["VFMSTUDIO_PRECEDENTS"] = "off"


# -- GUI ---------------------------------------------------------------------

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_window_apply_and_main_window_gate(qapp, tmp_path):
    import time

    from vfmstudio.gui.main_window import MainWindow
    from vfmstudio.gui.model_agent_window import ModelAgentWindow

    empty = MainWindow()
    empty.show()
    qapp.processEvents()
    empty.agent_chat._launch_buttons["model_build"].click()
    qapp.processEvents()
    assert empty.model_agent_window is None
    assert "imported" in empty.agent_chat.messages()[-1].text().lower()
    empty.session.dirty = False
    empty.close()

    paths = write_steps(tmp_path / "mats")
    document, experiment = _document(paths)
    window = ModelAgentWindow()
    window.set_document_provider(lambda: document)
    window.set_workdir(tmp_path / "work")
    window.start()
    deadline = time.monotonic() + 120.0
    while window.busy and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()
    assert not window.busy
    assert window.run is not None
    assert "Longitudinal" in window._report.toPlainText()
    assert window._apply.isEnabled()
    window._apply.click()
    qapp.processEvents()
    assert not experiment.external and not experiment.prescribed
    assert experiment.reference_step == 1
    assert experiment.mesh.bounds.y_max == pytest.approx(20.0)
    window.close()
