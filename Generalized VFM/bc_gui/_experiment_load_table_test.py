"""Headless regression test for the experiment load-step table.

Run with::

    python -m bc_gui._experiment_load_table_test

Qt uses its offscreen platform automatically here, so no display is needed.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace


def main() -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from bc_gui.bc_dialogs import AddBcDialog
    from bc_gui.bc_model import CATEGORY_EXTERNAL
    from bc_gui.experiment_setup_dialog import ExperimentSetupDialog
    from vfm_mesh import build_mesh
    from vfm_project import (
        ProjectTest,
        VfmProject,
        apply_experiment_definition,
        load_project,
        save_project,
    )
    from vfm_user_inputs import Bounds, MeshSettings

    app = QApplication.instance() or QApplication([])
    defaults = SimpleNamespace(
        material_type="orthotropic",
        analysis_type="out_of_plane",
    )

    with tempfile.TemporaryDirectory() as tmp:
        dialog = ExperimentSetupDialog(defaults, Path(tmp))
        tab = dialog._tabs_by_id[1]
        tab.tests = [
            ProjectTest(Path(tmp) / f"step_{step:02d}.mat")
            for step in range(1, 4)
        ]
        tab.external = [{
            "label": "Applied bending moment",
            "dof": "W_X",
            "edge": "right",
            "loadType": "concentrated",
        }]
        apply_experiment_definition(
            VfmProject(tests=tab.tests),
            1,
            prescribed=[],
            linking=[],
            external=tab.external,
            mesh_settings=None,
            sample_area_total=None,
            applied_load_source="manual",
        )
        tab._set_applied_load_source("manual")
        tab.refresh()

        assert tab.table.rowCount() == 3
        assert tab.table.columnCount() == 3
        assert "Applied bending moment" in tab.table.horizontalHeaderItem(2).text()
        for row, value in enumerate((10.0, 20.0, 35.0)):
            item = tab.table.item(row, 2)
            assert bool(item.flags() & Qt.ItemIsEditable)
            item.setText(str(value))

        assert [test.external[0]["value"] for test in tab.tests] == [
            10.0, 20.0, 35.0]

        # The old defect: any redraw discarded values still living only in
        # the widget.  Verify a source toggle and explicit refresh preserve
        # every step's independent magnitude.
        mat_index = tab.load_source_combo.findData("mat")
        manual_index = tab.load_source_combo.findData("manual")
        tab.load_source_combo.setCurrentIndex(mat_index)
        tab.load_source_combo.setCurrentIndex(manual_index)
        tab.refresh()
        assert [tab.table.item(row, 2).text() for row in range(3)] == [
            "10", "20", "35"]
        assert [test.external[0]["value"] for test in tab.tests] == [
            10.0, 20.0, 35.0]

        # Non-finite text must never reach a solver case.
        tab.table.item(1, 2).setText("not-a-load")
        assert "value" not in tab.tests[1].external[0]
        tab.table.item(1, 2).setText("22.5")
        assert tab.tests[1].external[0]["value"] == 22.5

        project_path = Path(tmp) / "per_step_loads.vfmproj.json"
        save_project(VfmProject(tests=tab.tests), project_path)
        reloaded = load_project(project_path)
        assert [test.external[0]["value"] for test in reloaded.tests] == [
            10.0, 22.5, 35.0]
        assert all(test.applied_load_source == "manual"
                   for test in reloaded.tests)

        # In experiment mode, the topology editor must not offer the obsolete
        # single scalar that used to imply one load for the entire ramp.
        mesh = build_mesh(
            MeshSettings(
                number_of_elements_x=1,
                number_of_elements_y=1,
                bounds=Bounds(x_min=0.0, x_max=1.0,
                              y_min=0.0, y_max=1.0),
            ),
            analysis_type="out_of_plane",
        )
        topology_dialog = AddBcDialog(
            category=CATEGORY_EXTERNAL,
            selected_node_ids=[0],
            mesh_pkg=mesh,
            analysis_type="out_of_plane",
            external_values_managed_by_parent=True,
        )
        assert topology_dialog._external_value_edit is None
        assert topology_dialog.result_entry().value is None
        topology_dialog.close()
        topology_dialog.deleteLater()

        dialog.close()
        dialog.deleteLater()

    app.processEvents()
    print("[experiment load table] per-step editing and redraw persistence OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
