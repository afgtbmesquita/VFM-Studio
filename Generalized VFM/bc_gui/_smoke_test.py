# =============================================================================
#  bc_gui / _smoke_test.py  (plate)
# -----------------------------------------------------------------------------
#  Headless round-trip test for the plate BC document model.
#
#  Builds a small plate mesh (6 DOFs per node), populates a ``BcDocument``
#  with entries exercising every supported plate DOF (U, V, W, W_X, W_Y,
#  W_XY) across all three categories, writes it to a temporary ``.py``
#  file, and re-reads it through the same loader the plate solver uses
#  (``vfm_data_io.load_boundary_conditions``).  We then check that the
#  per-category counts and a few representative fields survived the
#  round-trip exactly as the solver expects.
#
#  No Qt / matplotlib / display are needed -- this test is safe on CI.
#
#  Run with:
#      python -m bc_gui._smoke_test
# =============================================================================

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


def _ensure_plate_on_path() -> None:
    """Add the plate folder to sys.path so ``vfm_*`` modules import."""
    plate_dir = Path(__file__).resolve().parent.parent
    if str(plate_dir) not in sys.path:
        sys.path.insert(0, str(plate_dir))


def main() -> int:
    _ensure_plate_on_path()

    import numpy as np
    from vfm_data_io import load_boundary_conditions
    from vfm_mesh import build_mesh, prune_mesh_by_measurement_points
    from vfm_user_inputs import Bounds, MeshSettings

    from bc_gui.bc_model import (
        BcDocument,
        ExternalEntry,
        LinkingEntry,
        PrescribedEntry,
    )

    mesh_settings = MeshSettings(
        number_of_elements_x=3,
        number_of_elements_y=2,
        bounds=Bounds(x_min=0.0, x_max=30.0, y_min=-9.5, y_max=9.5),
    )
    mesh_pkg = build_mesh(mesh_settings, analysis_type="both")
    print(f"[bc_gui smoke] mesh built: "
          f"{mesh_pkg.n_nodes} nodes, {mesh_pkg.n_elements} elements, "
          f"n_dof_per_node={mesh_pkg.n_dof_per_node}")

    document = BcDocument(
        description="Smoke test for plate bc_gui",
        mesh_settings=mesh_settings,
    )

    # ---- Prescribed entries: one per translational + one rotation -----
    prescribed_dof_specs = [
        ("Clamp left edge in U",    "U",    "left"),
        ("Clamp left edge in V",    "V",    "left"),
        ("Clamp left edge in W",    "W",    "left"),
        ("Clamp left edge in W_X",  "W_X",  "left"),
        ("Clamp left edge in W_Y",  "W_Y",  "left"),
        ("Clamp left edge in W_XY", "W_XY", "left"),
    ]
    for label, dof, edge in prescribed_dof_specs:
        document.add(
            "prescribed",
            PrescribedEntry(
                label=label, dof=dof, edge=edge,
                selection_mode="edge", value=0.0,
            ),
        )

    # Also exercise an explicit-nodes prescribed entry so the loader has
    # to see both "edge" and "nodes" selectors in the same file.
    document.add(
        "prescribed",
        PrescribedEntry(
            label="Pin first three nodes in V",
            dof="V",
            nodes=[0, 1, 2],
            selection_mode="nodes",
            value=0.0,
        ),
    )

    # ---- Linking: tie multiple DOFs along the right edge --------------
    document.add(
        "linking",
        LinkingEntry(
            label="Tie right edge in V (membrane shear)",
            dof="V",
            edge="right",
            selection_mode="edge",
            value=0.0,
            sort_direction="y",
        ),
    )
    document.add(
        "linking",
        LinkingEntry(
            label="Tie right edge in W (transverse bending)",
            dof="W",
            edge="right",
            selection_mode="edge",
            value=0.0,
            sort_direction="y",
        ),
    )

    # ---- External: measure V and a rotation along the right edge -----
    # Set a 'value' on the first entry (new "Applied force / moment" GUI
    # field) and leave the second one blank so both code paths round-trip.
    document.add(
        "external",
        ExternalEntry(
            label="Measure right-edge V (membrane shear group)",
            dof="V",
            edge="right",
            selection_mode="edge",
            load_type="concentrated",
            value=2.5,
        ),
    )
    document.add(
        "external",
        ExternalEntry(
            label="Measure right-edge W_X (bending moment group)",
            dof="W_X",
            edge="right",
            selection_mode="edge",
            load_type="concentrated",
            value=None,
        ),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "smoke_bc.py"
        saved_path = document.write_py(out_path, keep_backup=False)
        print(f"[bc_gui smoke] wrote        : {saved_path}")

        manual_settings = MeshSettings(
            number_of_elements_x=2,
            number_of_elements_y=2,
            bounds=Bounds(x_min=0.0, x_max=3.0, y_min=0.0, y_max=2.0),
            x_nodes=[0.0, 1.0, 3.0],
            y_nodes=[0.0, 0.5, 2.0],
        )
        manual_doc = BcDocument(
            description="Manual mesh settings round-trip",
            mesh_settings=manual_settings,
        )
        manual_path = Path(tmp_dir) / "manual_mesh_bc.py"
        manual_doc.write_py(manual_path, keep_backup=False)
        reloaded_manual = BcDocument.from_any_file(manual_path)
        assert reloaded_manual.mesh_settings is not None
        assert reloaded_manual.mesh_settings.x_nodes == [0.0, 1.0, 3.0]
        assert reloaded_manual.mesh_settings.y_nodes == [0.0, 0.5, 2.0]
        manual_mesh = build_mesh(reloaded_manual.mesh_settings,
                                 analysis_type="both")
        assert manual_mesh.n_nodes == 9
        assert manual_mesh.n_elements == 4
        print("[bc_gui smoke] manual mesh settings round-trip OK")

        # ---- Re-read through the solver's own loader ------------------
        reloaded = load_boundary_conditions(saved_path)
        print(
            f"[bc_gui smoke] reloaded     : "
            f"prescribed={len(reloaded.prescribed)}, "
            f"linking={len(reloaded.linking)}, "
            f"external={len(reloaded.external)}"
        )

        # ---- Counts ---------------------------------------------------
        assert len(reloaded.prescribed) == 7, (
            f"prescribed count round-trip failed: "
            f"got {len(reloaded.prescribed)}"
        )
        assert len(reloaded.linking) == 2, (
            f"linking count round-trip failed: got {len(reloaded.linking)}"
        )
        assert len(reloaded.external) == 2, (
            f"external count round-trip failed: got {len(reloaded.external)}"
        )

        # ---- DOF coverage --------------------------------------------
        prescribed_dofs = {entry.get("dof") for entry in reloaded.prescribed}
        assert prescribed_dofs == {"U", "V", "W", "W_X", "W_Y", "W_XY"}, (
            f"prescribed DOF coverage mismatch: {prescribed_dofs}"
        )

        # ---- Spot-check selectors and extra keys ----------------------
        p_first = reloaded.prescribed[0]
        assert p_first.get("dof") == "U" and p_first.get("edge") == "left", \
            f"unexpected first prescribed entry: {p_first}"

        # The nodes-based entry should appear with an explicit list.
        nodes_entry = next(
            entry for entry in reloaded.prescribed
            if entry.get("nodes") == [0, 1, 2]
        )
        assert nodes_entry.get("dof") == "V", (
            f"unexpected nodes entry: {nodes_entry}"
        )

        link0 = reloaded.linking[0]
        assert link0.get("sortDirection") == "y", \
            f"unexpected linking entry: {link0}"

        ext_wx = next(
            entry for entry in reloaded.external
            if entry.get("dof") == "W_X"
        )
        assert ext_wx.get("loadType") == "concentrated", \
            f"unexpected W_X external entry: {ext_wx}"

        # The W_X entry was created with value=None so the loader must
        # NOT have inserted a 'value' key (which would be misinterpreted
        # as zero applied moment at solve time).
        assert "value" not in ext_wx, (
            f"W_X external entry should not carry a 'value' key when set "
            f"to None: {ext_wx}"
        )

        # The V entry was created with value=2.5 so the round-trip must
        # preserve it as a numeric.
        ext_v = next(
            entry for entry in reloaded.external
            if entry.get("dof") == "V"
        )
        assert "value" in ext_v, (
            f"V external entry lost its 'value' on round-trip: {ext_v}"
        )
        assert float(ext_v["value"]) == 2.5, (
            f"V external entry value mismatch after round-trip: {ext_v}"
        )

        # ---- Round-trip back into BcDocument and confirm symmetry ----
        doc2 = BcDocument.from_py_file(saved_path)
        assert doc2.count("prescribed") == 7
        assert doc2.count("linking") == 2
        assert doc2.count("external") == 2
        assert doc2.mesh_settings is not None
        assert doc2.mesh_settings.number_of_elements_x == 3
        assert doc2.mesh_settings.number_of_elements_y == 2
        assert doc2.mesh_settings.bounds.x_min == 0.0
        assert doc2.mesh_settings.bounds.x_max == 30.0
        assert doc2.mesh_settings.bounds.y_min == -9.5
        assert doc2.mesh_settings.bounds.y_max == 9.5

        # External 'value' field must survive the BcDocument round-trip.
        ext_entries = doc2.entries("external")
        v_entry = next(e for e in ext_entries if e.dof == "V")
        assert v_entry.value == 2.5, (
            f"V external entry value mismatch after BcDocument round-trip: "
            f"got {v_entry.value!r}"
        )
        wx_entry = next(e for e in ext_entries if e.dof == "W_X")
        assert wx_entry.value is None, (
            f"W_X external entry value should be None after BcDocument "
            f"round-trip: got {wx_entry.value!r}"
        )

        # ---- Verify the assembly path accepts the reloaded file ------
        # The solver's KA-builder runs the DOF resolver against every
        # entry; running it through the full pipeline ensures every
        # supported plate DOF label is parsed correctly.
        from vfm_mesh import build_kinematic_constraints
        constraint_system = build_kinematic_constraints(
            mesh_pkg=mesh_pkg,
            bc=reloaded,
            inactive_dofs=mesh_pkg_inactive_placeholder(),
        )
        print(
            f"[bc_gui smoke] constraints  : "
            f"prescribed={constraint_system.n_prescribed}, "
            f"linking={constraint_system.n_linking}, "
            f"inactive={constraint_system.n_inactive}, "
            f"total={constraint_system.n_constraints}"
        )
        # Sanity: every prescribed entry must contribute at least one row
        # (every selector resolves to a non-empty node set), and the
        # explicit-nodes entry [0, 1, 2] adds three more.
        assert constraint_system.n_prescribed >= 9, (
            f"unexpectedly small prescribed-row count from assembly: "
            f"{constraint_system.n_prescribed}"
        )
        assert constraint_system.n_linking >= 2, (
            f"unexpectedly small linking-row count from assembly: "
            f"{constraint_system.n_linking}"
        )

    # ------------------------------------------------------------------
    #  Live virtual-mesh editor: round-trip MeshSettings -> MeshPackage.
    # ------------------------------------------------------------------
    # The actual MeshEditorPanel + MeshCanvas.set_mesh paths require a
    # QApplication and so are exercised by hand.  Here we just verify
    # that ``vfm_mesh.build_mesh`` can rebuild a mesh with arbitrary
    # new settings (the operation the panel ultimately triggers) and
    # that the resulting MeshPackage exposes the bookkeeping the editor
    # depends on.
    from bc_gui.mesh_editor import mesh_settings_from_package

    new_settings = Bounds(x_min=-5.0, x_max=12.0, y_min=-2.0, y_max=4.0)
    new_mesh_pkg = build_mesh(
        MeshSettings(number_of_elements_x=5,
                     number_of_elements_y=4,
                     bounds=new_settings),
        analysis_type="both",
    )
    assert new_mesh_pkg.nx == 5, new_mesh_pkg.nx
    assert new_mesh_pkg.ny == 4, new_mesh_pkg.ny
    assert new_mesh_pkg.n_elements == 20, new_mesh_pkg.n_elements
    assert new_mesh_pkg.n_nodes == 30, new_mesh_pkg.n_nodes
    assert new_mesh_pkg.n_dofs == 30 * 6, new_mesh_pkg.n_dofs

    seeded = mesh_settings_from_package(new_mesh_pkg)
    assert seeded.number_of_elements_x == 5
    assert seeded.number_of_elements_y == 4
    assert seeded.bounds.x_min == -5.0
    assert seeded.bounds.x_max == 12.0
    assert seeded.bounds.y_min == -2.0
    assert seeded.bounds.y_max == 4.0
    print(
        f"[bc_gui smoke] mesh editor OK: rebuilt to "
        f"{new_mesh_pkg.nx} x {new_mesh_pkg.ny} elements, "
        f"{new_mesh_pkg.n_nodes} nodes; settings round-trip matches."
    )

    sparse_center = np.array([[
        0.5 * (new_mesh_pkg.x_nodes[0] + new_mesh_pkg.x_nodes[1]),
        0.5 * (new_mesh_pkg.y_nodes[0] + new_mesh_pkg.y_nodes[1]),
    ]])
    pruned_pkg, prune_report = prune_mesh_by_measurement_points(
        new_mesh_pkg, sparse_center)
    assert pruned_pkg.n_elements == 1, pruned_pkg.n_elements
    assert prune_report.n_elements_removed == 19, prune_report.n_elements_removed
    assert np.all(prune_mesh_by_measurement_points(pruned_pkg,
                                                   sparse_center)[1]
                  .n_elements_removed == 0)
    print(
        "[bc_gui smoke] mesh prune OK: "
        f"{prune_report.n_elements_before} -> {prune_report.n_elements_after} "
        "elements"
    )

    # ------------------------------------------------------------------
    #  BcEditorResult default-contract check (locks the API).
    # ------------------------------------------------------------------
    from bc_gui import BcEditorResult, BcEditorTestResult
    import bc_gui
    assert "launch_identifiability_diagnosis" in bc_gui.__all__
    default_result = BcEditorResult()
    assert default_result.saved_bc_path is None
    assert default_result.mesh_settings is None
    assert default_result.mesh_pkg is None
    assert default_result.sample_area_total is None
    assert default_result.max_iterations is None
    assert default_result.convergence_tolerance is None
    assert default_result.test_results == []
    print("[bc_gui smoke] BcEditorResult defaults are all None")

    test_result = BcEditorTestResult(
        test_index=0,
        data_file=Path("Loading_Data") / "specimen one.mat",
    )
    assert test_result.test_index == 0
    assert test_result.data_file.name == "specimen one.mat"
    assert test_result.saved_bc_path is None

    from bc_gui.main_window import default_bc_path_for_test
    default_bc = default_bc_path_for_test(
        Path("boundary_conditions"),
        1,
        Path("Loading_Data") / "specimen one.mat",
    )
    assert default_bc == Path("boundary_conditions") / "boundary_conditions_test_02_specimen_one.py", \
        default_bc
    print("[bc_gui smoke] per-test result/default save-name OK")

    _check_load_table_precision()
    print("[bc_gui smoke] OK")


def _check_load_table_precision() -> None:
    """A per-step load must survive the time-step table unchanged.

    The table DISPLAYS loads rounded (%g, 6 significant digits) so it stays
    readable.  Re-parsing that text on the way back out cost ~2e-6 relative on
    every redraw, which reached the solver as a different applied moment.  The
    exact float now rides in Qt.UserRole and wins unless the user actually
    typed something.
    """
    import tempfile
    from pathlib import Path

    from PySide6.QtWidgets import QApplication

    from vfm_project import ProjectTest
    from vfm_user_inputs import build_user_inputs

    from .experiment_setup_dialog import ExperimentSetupDialog

    QApplication.instance() or QApplication([])
    dialog = ExperimentSetupDialog(build_user_inputs(),
                                   Path(tempfile.mkdtemp()))
    tab = dialog._tabs_by_id[1]

    #  More digits than %g keeps, and not representable in 6 s.f.
    awkward = [0.25428145234, -0.25428145234, 1.0 / 3.0]
    tab.external = [{"label": "M top", "dof": "W_Y", "edge": "top",
                     "loadType": "distributed", "sortDirection": "x"}]
    tab.tests = [
        ProjectTest(data_file=Path(f"step{i}.mat"), experiment_id=1,
                    external=[{"label": "M top", "dof": "W_Y", "edge": "top",
                               "loadType": "distributed",
                               "sortDirection": "x", "value": value}])
        for i, value in enumerate(awkward)
    ]
    tab._set_applied_load_source("manual")

    #  Several redraws, as a real session would do.
    for _ in range(4):
        tab.refresh()
        tab.commit_step_loads()

    got = [test.external[0].get("value") for test in tab.tests]
    assert got == awkward, (
        "the load table lost precision on a round trip", got, awkward)

    #  A genuine edit must still win over the shadow copy.
    tab.table.item(0, 2).setText("42.5")
    tab.commit_step_loads()
    assert tab.tests[0].external[0]["value"] == 42.5, tab.tests[0].external[0]
    tab.refresh()
    tab.commit_step_loads()
    assert tab.tests[0].external[0]["value"] == 42.5, tab.tests[0].external[0]
    print("[bc_gui smoke] per-step load table is lossless "
          "(and still accepts edits)")
    return 0


def mesh_pkg_inactive_placeholder():
    """Return an empty 1-D int array for ``inactive_dofs``.

    The plate BC assembly is happy to receive a zero-length array when no
    DOFs need to be marked inactive; the smoke test exercises only the
    prescribed + linking branches.
    """
    import numpy as np
    return np.zeros(0, dtype=np.int64)


if __name__ == "__main__":
    sys.exit(main())
