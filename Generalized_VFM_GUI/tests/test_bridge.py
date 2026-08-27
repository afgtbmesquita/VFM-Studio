"""Solver bridge smoke test — imports the sibling Generalized VFM tree."""

from vfmstudio.core.paths import campaign_project_path, ensure_core_on_path, resolve_core_path


def test_core_folder_exists():
    path = resolve_core_path()
    assert path.is_dir()
    assert (path / "vfm_project.py").is_file()
    assert (path / "vfm_mesh.py").is_file()


def test_import_vfm_project():
    ensure_core_on_path()
    import vfm_project

    assert hasattr(vfm_project, "load_project")
    assert hasattr(vfm_project, "run_project")
    assert hasattr(vfm_project, "check_experiment_consistency")


def test_bridge_load_campaign():
    from vfmstudio.core import bridge

    path = campaign_project_path()
    if not path.is_file():
        return
    project = bridge.load_project(path)
    assert project.tests
    assert project.material_type == "orthotropic"
    assert project.analysis_type == "out_of_plane"
    missing = [test.data_file for test in project.tests if not test.data_file.is_file()]
    assert missing == []
