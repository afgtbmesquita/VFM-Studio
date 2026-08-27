"""Path discovery after Generalized VFM moved under VFM Studio."""

from pathlib import Path

from vfmstudio.core.paths import default_core_path, resolve_core_path, workspace_root
from vfmstudio.core.step_series import resolve_step_data_file, with_pre_conference


def test_workspace_root_is_vfm_studio():
    assert workspace_root().name == "VFM Studio"
    assert default_core_path().name == "Generalized VFM"
    assert (default_core_path() / "vfm_project.py").is_file()


def test_resolve_core_path_finds_solver_even_with_stale_env(monkeypatch):
    monkeypatch.setenv(
        "GVFM_CORE_PATH",
        str(workspace_root().parent / "Generalized VFM"),
    )
    resolve_core_path.cache_clear()
    try:
        path = resolve_core_path()
        assert path.is_dir()
        assert (path / "vfm_project.py").is_file()
        assert path.name == "Generalized VFM"
        assert path.parent.name == "VFM Studio"
    finally:
        resolve_core_path.cache_clear()


def test_with_pre_conference_skips_vfm_studio_prefix():
    campaign = Path("G:/campaign/Conference_campaign_bending")
    nested = (
        campaign
        / "VFM Studio"
        / "Longitudinal_bending"
        / "VFM_export"
        / "mat"
        / "Sample_6_step03.mat"
    )
    remapped = with_pre_conference(nested)
    assert remapped is not None
    assert remapped.parts[-5:] == (
        "Pre_Conference_Studies",
        "Longitudinal_bending",
        "VFM_export",
        "mat",
        "Sample_6_step03.mat",
    )
    assert "VFM Studio" not in remapped.parts


def test_campaign_common_finds_pre_conference_exports():
    from vfmstudio.core.paths import ensure_core_on_path

    ensure_core_on_path()
    from workflows.campaign_common import CAMPAIGN, LONG_DIR, TRANS_DIR

    assert CAMPAIGN.name == "Pre_Conference_Studies"
    assert LONG_DIR.is_dir()
    assert TRANS_DIR.is_dir()


def test_campaign_mat_resolves_under_pre_conference():
    from vfmstudio.core.paths import campaign_project_path

    project = campaign_project_path()
    stale = (
        project.parent.parent.parent
        / "Longitudinal_bending"
        / "VFM_export"
        / "mat"
        / "Sample_6_step03.mat"
    )
    found = resolve_step_data_file(stale, project)
    assert found is not None
    assert found.is_file()
    assert "Pre_Conference_Studies" in found.parts
    assert found.name == "Sample_6_step03.mat"
