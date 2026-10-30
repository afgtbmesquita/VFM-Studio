"""DIC data agent: raw-format gate, intake order, analysis, read-only guarantees, window."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("VFMSTUDIO_NO_VISTA", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

from vfmstudio.agents import build_arguments, get_agent
from vfmstudio.agents.dic_data.analysis import DataRefused
from vfmstudio.agents.dic_data.intake import frame_match, group_by_folder, inspect_inputs, prepare_groups, prepare_submission
from vfmstudio.agents.dic_data.raw_format import RAW_DIC_COLUMNS
from vfmstudio.agents.dic_data.report import render_report
from vfmstudio.agents.dic_data.session import agent_request, export_solver_mats, run_analysis

NX, NY, STEP_PX, PITCH = 30, 80, 7, 0.2
SIGMA_C = 2.5e-4  # per Vic cxx/cyy/cxy; curvature noise is 2 * SIGMA_C
KYY_PER_NMM = 2e-4
LOADS = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0)


def write_raw_test(folder: Path, seed: int = 0, loads=LOADS, rotated: bool = False) -> list[Path]:
    """Raw Vic-3D exports of uniform bending about X (κyy ∝ Ty), plus companion files."""
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    i, j = np.meshgrid(np.arange(NX), np.arange(NY))
    px = (100 + STEP_PX * i).ravel().astype(float)
    py = (100 + STEP_PX * j).ravel().astype(float)
    if rotated:
        X, Y = (py - 100) / STEP_PX * PITCH, (px - 100) / STEP_PX * PITCH
    else:
        X, Y = (px - 100) / STEP_PX * PITCH, (NY - 1) * PITCH - (py - 100) / STEP_PX * PITCH
    Yc = Y - Y.mean()
    n = X.size
    files, step_rows, sync_rows = [], [], []
    for step, load in enumerate(loads, start=1):
        dic = 10 + 20 * step
        kyy = KYY_PER_NMM * load
        image_kyy_is_cxx = rotated
        cols = {name: np.zeros(n) for name in RAW_DIC_COLUMNS}
        cols.update(X=X, Y=Y, x=px, y=py, q=px, r=py, W=-0.5 * kyy * Yc**2)
        cols["cxx"] = (-0.5 * kyy if image_kyy_is_cxx else 0.0) + rng.normal(0, SIGMA_C, n)
        cols["cyy"] = (0.0 if image_kyy_is_cxx else -0.5 * kyy) + rng.normal(0, SIGMA_C, n)
        cols["cxy"] = rng.normal(0, SIGMA_C, n)
        cols["sigma"] = np.full(n, 0.01)
        cols["SIGMA_Z"] = np.full(n, 0.001)
        path = folder / f"exp-{dic:08d}_0.csv"
        header = ",".join(f'" {name}"' if k else f'"{name}"' for k, name in enumerate(RAW_DIC_COLUMNS))
        table = np.column_stack([cols[name] for name in RAW_DIC_COLUMNS])
        np.savetxt(path, table, delimiter=",", header=header, comments="", fmt="%.10g")
        files.append(path)
        step_rows.append(f"{step},{dic},{10.0 * step}")
        sync_rows.append(f"{dic},{load},{-0.5 * load}")
    (folder / "step_DIC_file_map.csv").write_text(
        "Step,DIC_file_number,Target_time_s\n" + "\n".join(step_rows) + "\n", encoding="utf-8"
    )
    (folder / "synchronized_DIC_machine_data.csv").write_text(
        "DIC_file_number,Machine_Ty_Nmm,Machine_DeltaTheta_y_deg\n" + "\n".join(sync_rows) + "\n",
        encoding="utf-8",
    )
    return files


def snapshot(folder: Path) -> dict[str, tuple[str, int]]:
    return {
        path.name: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in sorted(folder.iterdir())
    }


@pytest.fixture(scope="module")
def raw_test(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("raw") / "Specimen_Synthetic_PB"
    write_raw_test(folder)
    return folder


def test_raw_format_gate(tmp_path, raw_test):
    infos = inspect_inputs([raw_test])
    assert len(infos) == len(LOADS) and all(info.ok for info in infos)
    assert infos[0].n_facets == NX * NY
    bad = tmp_path / "exp-00000001_0.csv"
    bad.write_text("X,Y,Z,U,V,W\n1,2,3,4,5,6\n", encoding="utf-8")
    shuffled = tmp_path / "exp-00000002_0.csv"
    reordered = list(RAW_DIC_COLUMNS[1:]) + [RAW_DIC_COLUMNS[0]]
    shuffled.write_text(",".join(reordered) + "\n" + ",".join(["0"] * 33) + "\n", encoding="utf-8")
    other = tmp_path / "steps.txt"
    other.write_text("hello", encoding="utf-8")
    checks = {info.path.name: info for info in inspect_inputs([bad, shuffled, other])}
    assert "columns differ" in checks[bad.name].problems[0]
    assert "different order" in checks[shuffled.name].problems[0]
    assert not checks[other.name].ok
    with pytest.raises(DataRefused) as refused:
        run_analysis(group_by_folder(inspect_inputs([raw_test, bad])), workdir=tmp_path / "out")
    assert any(bad.name in reason for reason in refused.value.reasons)
    assert not (tmp_path / "out").exists()


def test_intake_orders_by_campaign_step(raw_test, tmp_path):
    files = sorted(raw_test.glob("exp-*.csv"), reverse=True)
    (group,) = group_by_folder(inspect_inputs(files))
    assert group.name == raw_test.name
    assert [info.dic_number for info in group.files] == [10 + 20 * s for s in range(1, len(LOADS) + 1)]
    second = tmp_path / "Second_test"
    write_raw_test(second, seed=3, loads=LOADS[:3])
    groups = group_by_folder(inspect_inputs([second, raw_test]))
    assert {g.name for g in groups} == {"Second_test", raw_test.name}


def test_machine_data_is_used_only_when_the_user_submits_it(raw_test, tmp_path):
    exports = sorted(raw_test.glob("exp-*.csv"))
    dic_only = prepare_submission(exports)
    assert dic_only.groups[0].companions.sync_file is None
    assert dic_only.groups[0].companions.step_map_file is None
    match = frame_match(dic_only.groups[0])
    assert not match.ready
    assert any("Machine data has not been submitted" in problem for problem in match.problems)
    with pytest.raises(DataRefused) as refused:
        run_analysis(dic_only.groups, workdir=tmp_path / "no-machine")
    assert any("Machine data" in reason for reason in refused.value.reasons)
    assert not (tmp_path / "no-machine").exists()

    prepared = prepare_submission([raw_test])
    assert prepared.unassigned_machines == []
    assert frame_match(prepared.groups[0]).ready
    assert prepared.groups[0].companions.sync_file.name == "synchronized_DIC_machine_data.csv"
    assert prepared.groups[0].companions.step_map_file.name == "step_DIC_file_map.csv"

    trimmed = tmp_path / "synchronized_DIC_machine_data.csv"
    rows = (raw_test / "synchronized_DIC_machine_data.csv").read_text(encoding="utf-8").splitlines()
    trimmed.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    mismatched = prepare_submission([*exports, trimmed])
    match = frame_match(mismatched.groups[0])
    assert not match.ready
    assert match.only_dic == (10 + 20 * len(LOADS),)
    assert mismatched.groups[0].companions.step_map_file is None


def test_step_map_does_not_order_files_unless_it_was_submitted(tmp_path):
    folder = tmp_path / "OutOfOrder"
    folder.mkdir()
    for name in ("exp-00000300_0.csv", "exp-00000100_0.csv"):
        (folder / name).write_text("not a dic export\n", encoding="utf-8")
    (folder / "step_DIC_file_map.csv").write_text(
        "Step,DIC_file_number,Target_time_s\n1,300,1\n2,100,2\n", encoding="utf-8"
    )
    exports = [folder / "exp-00000300_0.csv", folder / "exp-00000100_0.csv"]
    (dic_only,) = prepare_groups(exports)
    assert [info.dic_number for info in dic_only.files] == [100, 300]
    assert dic_only.companions.step_map_file is None
    (submitted,) = prepare_groups([*exports, folder / "step_DIC_file_map.csv"])
    assert [info.dic_number for info in submitted.files] == [300, 100]
    assert submitted.companions.time_of_dic[300] == 1.0


def test_analysis_builds_solver_data_and_leaves_raw_untouched(raw_test, tmp_path):
    before = snapshot(raw_test)
    run = run_analysis(prepare_groups([raw_test]), workdir=tmp_path / "work")
    assert snapshot(raw_test) == before
    (analysis,) = run.analyses
    payload = analysis.payloads[-1]
    for key in ("nodes", "X", "Y", "W", "SK1grid", "SK2grid", "SK3grid", "SE1grid", "F", "F_labels"):
        assert key in payload
    assert payload["nodes"].shape == (NX * NY, 2)
    kyy = KYY_PER_NMM * LOADS[-1]
    assert np.mean(payload["SK2grid"]) == pytest.approx(kyy, abs=5e-5)
    facts = analysis.facts
    deformation = facts["deformation"]
    assert deformation["dominant_curvature_component"] == "kyy"
    assert deformation["dominant_machine_moment"] == "Ty"
    assert deformation["curvature_frame_check"]["verdict"].startswith("consistent")
    noise = facts["noise"]
    assert noise["available"] and noise["residual_dof_per_facet"] == len(LOADS) - 3
    measured = noise["components"]["kxx"]["measured_interior_median_inv_mm"]
    assert measured == pytest.approx(2 * SIGMA_C, rel=0.12)
    assert facts["read_quality"]["facets_lost_in_some_step"] == 0
    assert run.facts_path.is_file()
    for relative in run.figures[analysis.name].values():
        assert (run.workdir / relative).is_file()
    prompt, display = agent_request(run)
    assert "report.md" in prompt and "facts.json" in prompt and raw_test.name in display


def test_scripted_report_is_reproducible_from_facts(raw_test, tmp_path):
    run = run_analysis(prepare_groups([raw_test]), workdir=tmp_path / "work")
    facts = json.loads(run.facts_path.read_text(encoding="utf-8"))
    assert render_report(facts) == run.report_markdown == run.report_path.read_text(encoding="utf-8")
    (test,) = facts["tests"]
    stats = test["noise"]["components"]["kyy"]
    ratio = test["summary"]["noise_interior_measured_to_model"]["kyy"]
    assert ratio == pytest.approx(stats["measured_interior_median_inv_mm"] / stats["model_interior_median_inv_mm"])
    assert test["summary"]["duration_s"] == pytest.approx(10.0 * (len(LOADS) - 1))
    assert test["summary"]["machine_channel_ranges"]["Ty"]["max"] == pytest.approx(LOADS[-1])
    for heading in ("## Overview", "## Dataset and format check", "### Noise vs model",
                    "### DIC processing parameters and curvature filter",
                    "### Read quality", "### Automatic checks", raw_test.name):
        assert heading in run.report_markdown
    params = test["dic_parameters"]
    dic = test["summary"]["dic_parameters"]
    assert params["recorded_in_export"]["facet_step_px"] == STEP_PX
    assert dic["inferred_filter_size_facets"] == params["curvature_filter"]["best"]["filter_size_facets"]
    rows = params["curvature_filter_sweep"]["rows"]
    snr_size = dic["smallest_filter_with_snr_at_least"]["3"]
    assert snr_size is None or all(
        row["snr_interior"] < 3 for row in rows if row["filter_size_facets"] < snr_size
    )
    assert "filter_sweep" in run.figures[test["test_name"]]
    assert "Inferred curvature filter" in run.report_markdown
    for word in ("accept", "reject", "valid"):
        assert word not in run.report_markdown.lower().replace("acceptable", "")


def test_frame_check_flags_rotated_camera(tmp_path):
    folder = tmp_path / "Rotated"
    write_raw_test(folder, rotated=True)
    run = run_analysis(prepare_groups([folder]), workdir=tmp_path / "work")
    check = run.analyses[0].facts["deformation"]["curvature_frame_check"]
    assert check["image_axes_rotated_90deg"]
    assert check["verdict"].startswith("INCONSISTENT") and "swapped" in check["verdict"]
    checks = run.analyses[0].facts["automatic_checks"]
    assert checks[0]["check"] == "curvature frame"
    assert "INCONSISTENT" in run.report_markdown


def test_curvature_filter_is_identified_from_the_export():
    from scipy import ndimage

    from vfmstudio.agents.dic_data.dic_parameters import WindowFit, dic_parameter_facts

    rng = np.random.default_rng(3)
    ny, nx, pitch = 70, 40, 0.2
    mask = np.ones((ny, nx), dtype=bool)
    mask[30:38, 15:22] = False
    yy, xx = np.mgrid[0:ny, 0:nx] * pitch
    loads = np.linspace(0.0, 1.0, 8)
    disp = np.stack([
        np.stack([ndimage.gaussian_filter(rng.normal(0, 1e-3, (ny, nx)), 1.2) for _ in range(3)])
        for _ in loads
    ])
    disp[:, 2] += loads[:, None, None] * (-0.5e-2 * (yy - yy.mean()) ** 2)
    disp -= disp[0]
    disp[..., ~mask] = np.nan
    exported = WindowFit.build(mask, 5, pitch, pitch, edge_weight=0.1).curvatures(disp[:, 2])
    facts = dic_parameter_facts(mask, pitch, pitch, 7, disp, exported, len(loads) - 1, 1, loads, 5)
    best = facts["curvature_filter"]["best"]
    assert (best["filter_size_facets"], best["edge_weight"]) == (11, 0.1)
    assert best["mismatch_mean"] < 1e-6 and facts["curvature_filter"]["verdict"] == "reproduced closely"
    sweep = facts["curvature_filter_sweep"]
    noise = [row["noise_interior_median_inv_mm"]["kyy"] for row in sweep["rows"]]
    assert noise == sorted(noise, reverse=True)
    assert abs(sweep["rows"][0]["signal_interior_mean_inv_mm"] - 1e-2) < 1e-3
    assert facts["subset"]["subset_size_estimate_px"] > 0


def test_script_runs_without_gui_or_llm(raw_test, tmp_path, capsys):
    from vfmstudio.agents.dic_data.__main__ import main

    assert main([str(raw_test), "--list"]) == 0
    assert "exp-00000030_0.csv" in capsys.readouterr().out
    out = tmp_path / "report"
    assert main([str(raw_test), "--out", str(out), "--export-mat", str(tmp_path / "mats")]) == 0
    assert (out / "report.md").is_file() and (out / "facts.json").is_file()
    assert len(list((tmp_path / "mats").rglob("*.mat"))) == len(LOADS)
    bad = tmp_path / "bad" / "exp-00000001_0.csv"
    bad.parent.mkdir()
    bad.write_text("X,Y\n1,2\n", encoding="utf-8")
    assert main([str(raw_test), str(bad), "--out", str(tmp_path / "never")]) == 2
    assert not (tmp_path / "never").exists()
    assert main([str(raw_test), "--export-mat", str(raw_test)]) == 2


def test_script_survives_a_legacy_console_code_page():
    import subprocess
    import sys

    code = ("from vfmstudio.agents.dic_data.__main__ import _utf8_console; "
            "_utf8_console(); print('θy κxx σZ Δκ ≈ ×')")
    env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True)
    assert done.returncode == 0, done.stderr.decode(errors="replace")


def test_export_refuses_raw_folders_and_overwrites(raw_test, tmp_path):
    run = run_analysis(prepare_groups([raw_test]), workdir=tmp_path / "work")
    before = snapshot(raw_test)
    with pytest.raises(DataRefused):
        export_solver_mats(run, raw_test)
    with pytest.raises(DataRefused):
        export_solver_mats(run, raw_test.parent)
    assert snapshot(raw_test) == before
    written = export_solver_mats(run, tmp_path / "export")
    assert len(written) == len(LOADS) and all(path.is_file() for path in written)
    with pytest.raises(DataRefused):
        export_solver_mats(run, tmp_path / "export")
    from vfmstudio.agents.dic_data.session import add_exported_experiments
    from vfmstudio.core.document import MeshModel, SessionDocument

    document = SessionDocument()
    (experiment,) = add_exported_experiments(document, run, written)
    assert document.experiments == [experiment] and experiment.name == raw_test.name
    assert [step.data_file for step in experiment.steps] == written
    assert experiment.steps[0].label == written[0].stem and "campaign step" in experiment.steps[0].description
    assert experiment.mesh.model_dump() != MeshModel().model_dump()
    (again,) = add_exported_experiments(document, run, written)
    assert again.name == f"{raw_test.name} (2)" and again.experiment_id == experiment.experiment_id + 1
    with pytest.raises(ValueError):
        add_exported_experiments(document, run, written[:-1])
    from scipy.io import loadmat

    data = loadmat(written[-1])
    assert data["SK2grid"].size == NX * NY


def test_model_build_reads_the_data_agent_record_of_exported_steps(raw_test, tmp_path):
    from vfmstudio.agents.dic_data.session import add_exported_experiments
    from vfmstudio.agents.model_build.audit import audit_reply
    from vfmstudio.agents.model_build.session import agent_request, build_model
    from vfmstudio.core.document import ExperimentModel, LoadStepModel, SessionDocument

    run = run_analysis(prepare_groups([raw_test]), workdir=tmp_path / "work")
    written = export_solver_mats(run, tmp_path / "export")
    record = written[0].parent / "data_agent"
    facts = json.loads((record / "facts.json").read_text(encoding="utf-8"))
    assert [test["test_name"] for test in facts["tests"]] == [raw_test.name]
    assert facts["exported_with"] == [path.stem for path in written]
    figures = facts["tests"][0]["figures"]
    assert figures and all((record / path).is_file() for path in figures.values())
    assert (record / "report.md").is_file()

    document = SessionDocument()
    (experiment,) = add_exported_experiments(document, run, written)
    imported = ExperimentModel(
        experiment_id=99, name="Imported by hand",
        steps=[LoadStepModel(data_file=path, label=path.stem) for path in written[:2]],
    )
    imported.steps.append(LoadStepModel(data_file=tmp_path / "elsewhere.mat", label="elsewhere"))
    document.experiments.append(imported)
    built_run = build_model(document, workdir=tmp_path / "model", library={"precedents": []})
    built, other = built_run.builds
    staged = built.data_agent
    assert staged["available"] and staged["test_name"] == raw_test.name
    assert (built_run.workdir / staged["facts"]).is_file()
    assert staged["specimen"]["machine_channels"] and len(staged["machine_per_step"]) == len(LOADS)
    assert all((built_run.workdir / path).is_file() for path in staged["figures"].values())
    assert built_run.figures[experiment.name] == staged["figures"]
    assert not other.data_agent["available"] and "does not cover" in other.data_agent["reason"]
    assert any("no DIC data agent record" in item for item in other.warnings)
    assert "DIC data agent record" in built_run.report_markdown
    prompt, _display = agent_request(built_run)
    assert f"data_agent/ for: {experiment.name}" in prompt and "no data agent record for: Imported by hand" in prompt
    span = staged["specimen"]["measured_span_mm"]["y"]
    assert audit_reply(f"The DIC spans {span} mm along Y.", built_run.workdir) is None


def test_reply_audit_flags_unmeasured_numbers_and_decision_words(raw_test, tmp_path):
    from vfmstudio.agents.dic_data.audit import (
        audit_reply,
        decision_wording,
        source_values,
        unsupported_numbers,
    )

    sources = source_values(['{"kyy": 0.004987, "facets": 2400, "lost": 0.0213}', "7,097 facets"])
    reply = ("Δκyy is 0.00499 1/mm over 2,400 facets (7097 read); 2.1% lost. "
             "The patch near Y 37.25 mm is 12.5% wider, in step 3.")
    assert unsupported_numbers(reply, sources) == ["37.25", "12.5%"]
    run = run_analysis(prepare_groups([raw_test]), workdir=tmp_path / "work")
    measured = run.analyses[0].facts["deformation"]["interior_mean_dk_at_final_step_inv_mm"]["kyy"]
    reply = f"Δκyy is {measured:.3g} 1/mm over {NX * NY:,} facets; the patch near Y 37.25 mm is wider."
    assert decision_wording("Whether to accept it is yours. It is not rejected. I would discard step 3.") == [
        "discard"
    ]
    note = audit_reply(reply + " The test is usable.", run.workdir)
    assert "37.25" in note and "usable" in note
    assert audit_reply(f"Δκyy is {measured:.3g} 1/mm.", run.workdir) is None


def test_data_agent_is_locked_down():
    spec = get_agent("dic_data")
    args = build_arguments(spec)
    tools = args[args.index("--tools") + 1 : args.index("--allowedTools")]
    assert tools == ["Read", "Glob"]
    denied = args[args.index("--disallowedTools") + 1 :]
    for name in ("Bash", "PowerShell", "Edit", "Write", "WebFetch", "WebSearch"):
        assert name in denied
    assert "--restricted" in args and "--strict-mcp-config" in args
    prompt = args[args.index("--append-system-prompt") + 1]
    assert "Never decide whether a test is accepted" in prompt
    assert "read-only" in prompt and "internet" in prompt
    assert "Do not repeat or re-derive" in prompt and "Do not compute new" in prompt
    assert spec.launch_label == "Use data agent" and not spec.gui_context


# -- GUI ---------------------------------------------------------------------

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _settle(app, window, timeout_s: float = 120.0) -> None:
    deadline = time.monotonic() + timeout_s
    while window.busy and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    assert not window.busy


def test_window_intake_refusal_and_analysis(qapp, raw_test, tmp_path, monkeypatch):
    from PySide6.QtCore import Qt

    from vfmstudio.agents.dic_data import session
    from vfmstudio.gui.data_agent_window import FILES, REPORT, DataAgentWindow

    monkeypatch.setattr(session, "scratch_root", lambda: tmp_path / "scratch")
    window = DataAgentWindow()
    refused, ready = [], []
    window.refused.connect(refused.append)
    window.analysis_ready.connect(ready.append)

    window.add_paths([str(raw_test)])
    _settle(qapp, window)
    (group,) = window.groups()
    assert [info.dic_number for info in group.files] == [10 + 20 * s for s in range(1, len(LOADS) + 1)]
    item = window.tree.topLevelItem(0)
    assert item.child(0).text(1) == "1" and item.child(0).text(7).startswith("✓")
    assert "Ty" in item.child(1).text(5)
    assert window._block_reason == "" and window._primary.isEnabled()
    requirement_rows = [
        (window.requirements.topLevelItem(0).child(row).text(0), window.requirements.topLevelItem(0).child(row).text(2))
        for row in range(window.requirements.topLevelItem(0).childCount())
    ]
    assert ("DIC exports", "Submitted") in requirement_rows
    assert ("Machine data", "Submitted") in requirement_rows
    assert ("Frame match", "Matched") in requirement_rows

    bad = tmp_path / "exp-00000999_0.csv"
    bad.write_text("X,Y\n1,2\n", encoding="utf-8")
    window.add_paths([str(bad)])
    _settle(qapp, window)
    assert window.tree.topLevelItemCount() == 2
    window.analyse()
    assert refused and bad.name in refused[0]
    assert not ready and window._banner.isVisibleTo(window)

    window.tree.clearSelection()
    window.tree.topLevelItem(1).setSelected(True)
    window._remove_selection()
    window.tree.clearSelection()
    first = window.tree.topLevelItem(0).child(0)
    first.setSelected(True)
    window._move_selection(1)
    assert window.groups()[0].files[1].dic_number == 30
    window._auto_order()
    assert window.groups()[0].files[0].dic_number == 30

    window.analyse()
    _settle(qapp, window)
    assert ready and ready[0].analyses[0].name == raw_test.name
    assert window.structure.topLevelItemCount() == 1
    assert all(window.steps.item(i).flags() & Qt.ItemFlag.ItemIsEnabled for i in range(window.steps.count()))
    assert window.pages.currentIndex() == REPORT
    assert window._primary.text().startswith("Next: Deformation")
    window.show()
    qapp.processEvents()
    width = window.steps.viewport().width()
    for index, card in enumerate(window._cards):
        assert window.steps.item(index).sizeHint().height() >= card.heightForWidth(width) > 0
        assert window.steps.visualItemRect(window.steps.item(index)).height() >= card.heightForWidth(width)

    # Analysing again is offered only when the file list differs from what was analysed.
    window.go_to(FILES)
    assert window.results_current() and window._primary.text().startswith("Next")
    window.analyse()
    assert not window.busy and len(ready) == 1
    window.tree.clearSelection()
    window.tree.topLevelItem(0).child(0).setSelected(True)
    window._move_selection(1)
    assert not window.results_current()
    assert window._primary.text() == "Analyse again ▶" and window._primary.isEnabled()
    assert "Outdated" in window._cards[REPORT].detail.text()
    window._move_selection(-1)
    assert window.results_current() and window._primary.text().startswith("Next")
    window.tree.topLevelItem(0).setText(0, "renamed")
    assert not window.results_current()
    window.close()


def test_window_blocks_until_machine_data_matches(qapp, raw_test):
    from vfmstudio.gui.data_agent_window import DataAgentWindow

    window = DataAgentWindow()
    window.add_paths([str(path) for path in sorted(raw_test.glob("exp-*.csv"))])
    _settle(qapp, window)
    assert not window._primary.isEnabled()
    assert "Machine data" in window._block_reason
    rows = {
        window.requirements.topLevelItem(0).child(row).text(0): window.requirements.topLevelItem(0).child(row).text(2)
        for row in range(window.requirements.topLevelItem(0).childCount())
    }
    assert rows["DIC exports"] == "Submitted"
    assert rows["Machine data"] == "Required"
    assert rows["Frame match"] == "Not matched"
    window.analyse()
    assert window._banner.isVisibleTo(window) and "Machine data" in window._banner.text()

    window.add_paths([str(raw_test / "synchronized_DIC_machine_data.csv")])
    _settle(qapp, window)
    assert window._block_reason == "" and window._primary.isEnabled()
    assert "Ty" in window.tree.topLevelItem(0).child(0).text(5)
    window.close()


def test_figures_are_high_resolution(raw_test, tmp_path):
    from PIL import Image

    from vfmstudio.agents.dic_data.figures import SAVE_DPI

    run = run_analysis(prepare_groups([raw_test]), workdir=tmp_path / "work")
    fields = run.workdir / run.figures[run.analyses[0].name]["fields"]
    with Image.open(fields) as image:
        dpi = image.info.get("dpi", (0, 0))[0]
        assert round(dpi) == SAVE_DPI
        assert image.width >= 2000


def test_panel_data_agent_flow(qapp, monkeypatch, tmp_path):
    from vfmstudio.gui.docks.agent_chat import AgentChatPanel

    panel = AgentChatPanel()
    launched = []
    panel.launcher_requested.connect(launched.append)
    panel._launch_buttons["dic_data"].click()
    assert panel.agent.agent_id == "dic_data" and launched == ["dic_data"]
    assert panel.send("what is the noise?") is False
    assert launched == ["dic_data", "dic_data"]
    assert "window" in panel.messages()[-1].text()
    monkeypatch.setattr(panel, "_resolve_cli", lambda: None)
    assert not panel.start_task("dic_data", "Read facts.json", "Interpret 1 test", tmp_path,
                                attachments=[("report", "# Report")])
    assert panel._session_cwd == tmp_path
    sent = []
    monkeypatch.setattr(
        panel, "send",
        lambda prompt, display=None, attachments=(): sent.append((prompt, display, list(attachments))) or True,
    )
    assert panel.start_task("dic_data", "Read facts.json", "Interpret 1 test", tmp_path,
                            attachments=[("report", "# Report")])
    assert sent == [("Read facts.json", "Interpret 1 test", [("report", "# Report")])]
    assert panel.agent.auditor is not None
    panel._turn_text = "I would reject this test."
    panel._audit_turn()
    assert panel.messages()[-1].role == "audit" and "reject" in panel.messages()[-1].text()


def test_main_window_opens_data_agent(qapp, raw_test, tmp_path):
    from vfmstudio.gui.main_window import MainWindow

    window = MainWindow()
    window.show()
    qapp.processEvents()
    window.agent_chat._launch_buttons["dic_data"].click()
    qapp.processEvents()
    assert window.dock_agent.isVisible()
    assert window.data_agent_window is not None and window.data_agent_window.isVisible()
    window.data_agent_window.refused.emit("Refused: test")
    assert window.agent_chat.messages()[-1].role == "error"
    run = run_analysis(prepare_groups([raw_test]), workdir=tmp_path / "work")
    written = export_solver_mats(run, tmp_path / "export")
    before = len(window.session.document.experiments)
    window.data_agent_window.exported.emit(run, written)
    experiments = window.session.document.experiments
    assert len(experiments) == before + 1 and experiments[-1].name == raw_test.name
    assert [step.data_file for step in experiments[-1].steps] == written
    assert window.session.selection.experiment_id == experiments[-1].experiment_id
    window.session.dirty = False
    window.close()
    assert not window.data_agent_window.isVisible()
