"""Multi-file MAT load-step helpers."""

from pathlib import Path

from vfmstudio.gui.dialogs.import_mat import load_steps_from_paths, natural_path_key


def test_natural_path_key_orders_step_numbers():
    names = [
        Path("A1_PB_step10.mat"),
        Path("A1_PB_step2.mat"),
        Path("A1_PB_step03.mat"),
        Path("A1_PB_step9.mat"),
    ]
    ordered = sorted(names, key=natural_path_key)
    assert [p.name for p in ordered] == [
        "A1_PB_step2.mat",
        "A1_PB_step03.mat",
        "A1_PB_step9.mat",
        "A1_PB_step10.mat",
    ]


def test_load_steps_from_paths_sorts_and_dedupes(tmp_path: Path):
    a = tmp_path / "step10.mat"
    b = tmp_path / "step2.mat"
    c = tmp_path / "step03.mat"
    for path in (a, b, c):
        path.write_bytes(b"x")
    steps = load_steps_from_paths([a, b, c, a])
    assert [step.label for step in steps] == ["step2", "step03", "step10"]
    assert [Path(step.data_file).name for step in steps] == [
        "step2.mat",
        "step03.mat",
        "step10.mat",
    ]


def test_load_steps_from_paths_skips_existing(tmp_path: Path):
    a = tmp_path / "step03.mat"
    b = tmp_path / "step04.mat"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    steps = load_steps_from_paths([a, b], skip=[a])
    assert len(steps) == 1
    assert Path(steps[0].data_file).name == "step04.mat"
