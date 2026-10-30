"""Human Interpreter prompt and the layout pass that follows it."""

from __future__ import annotations

import re

from vfmstudio.agents import available_agents
from vfmstudio.agents.registry import get_agent
from vfmstudio.agents.writing.interpreter import HUMAN_INTERPRETER, build_interpreter_prompt
from vfmstudio.agents.writing.present import prepare_markdown


def test_interpreter_is_not_a_selectable_agent():
    ids = [spec.agent_id for spec in available_agents()]
    assert "human_interpreter" not in ids
    assert get_agent("dic_data").interpret is True
    assert get_agent("model_build").interpret is True
    assert get_agent("claude").interpret is False
    assert HUMAN_INTERPRETER.tools == ()
    assert HUMAN_INTERPRETER.model == "sonnet"
    assert "--restricted" in HUMAN_INTERPRETER.extra_args
    assert "--safe-mode" in HUMAN_INTERPRETER.extra_args
    assert HUMAN_INTERPRETER.extra_args[HUMAN_INTERPRETER.extra_args.index("--effort") + 1] == "medium"


def test_prompt_keeps_rules_numbers_and_only_given_figures():
    prompt_rules = HUMAN_INTERPRETER.system_prompt
    assert "Write for humans, not for an abstract standard" in prompt_rules
    assert "Do not compute, estimate, or introduce a number" in prompt_rules
    figures = {"Plate A": {"fields": "figures/a_fields.png", "missing": ""}}
    prompt = build_interpreter_prompt(
        specialist_text="The ratio is 1.2.",
        report_text="# Report\n\nAll measured.",
        figures=figures,
        follow_up=False,
    )
    assert "{{figure:Plate A/fields}}" in prompt
    assert "noise_maps" not in prompt
    assert "1.2" in prompt
    assert "# Report" not in prompt
    assert "All measured." not in prompt
    follow = build_interpreter_prompt(
        specialist_text="Still 1.2.",
        report_text="# Report",
        figures={"Plate A": {"fields": "figures/a_fields.png"}},
        follow_up=True,
    )
    assert "# Report" not in follow
    assert "Do not restate" in follow
    assert "{{figure:Plate A/fields}}" in follow
    assert "Still 1.2." in follow


def test_prepare_expands_figures_math_and_repairs_tables(tmp_path):
    figure = tmp_path / "figures" / "a_fields.png"
    figure.parent.mkdir()
    figure.write_bytes(b"png")
    shown = prepare_markdown(
        "Curve {{figure:Plate A/fields}} then {{figure:Nope/fields}}.",
        workdir=tmp_path,
        figures={"Plate A": {"fields": "figures/a_fields.png"}},
    )
    assert "![Plate A: displacement and curvature increments](figures/a_fields.png)" in shown
    assert "Nope" not in shown
    assert "{{figure:" not in shown

    math = prepare_markdown(r"Curvature \(\kappa_{xx}\) at the edge.", workdir=tmp_path)
    assert r"\kappa_{xx}" not in math
    assert ".interpreter/eq_1.png" in math
    image = tmp_path / ".interpreter" / "eq_1.png"
    assert image.is_file() and image.stat().st_size > 0

    source = "| A | B |\n| --- | --- |\n| 1 | 2 |"
    broken = "| A | B |\n| --- | --- |\n| 1 |"
    repaired = prepare_markdown(broken, sources=source)
    assert "| 1 | 2 |" in repaired
    assert re.search(r"(?m)^\| 1 \|\s*$", repaired) is None

    listed = prepare_markdown(broken, sources="no table here")
    assert "| --- |" not in listed
    assert "- A: 1" in listed

    kept = prepare_markdown(source + "\n", sources="")
    assert kept == source + "\n"
