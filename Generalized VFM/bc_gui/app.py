# =============================================================================
#  bc_gui / app.py  (plate)
# -----------------------------------------------------------------------------
#  Entry point used by plate/run_vfm.py.  Boots a QApplication, shows the
#  BcEditorWindow, and returns a ``BcEditorResult`` describing what the
#  user did:
#
#      * ``saved_bc_path``  -- the .py file written by the GUI's Save
#                              button (or ``None`` on cancel)
#      * ``mesh_settings``  -- the ``MeshSettings`` after any live edits
#                              in the new mesh-editor panel (``None`` if
#                              the user did not apply a new mesh)
#      * ``mesh_pkg``       -- the rebuilt ``MeshPackage`` matching the
#                              new settings (``None`` if not changed)
#
#  Cancel sets ``saved_bc_path`` to ``None``; plate ``run_vfm.py`` exits in
#  that case instead of running the solver.
#
#  The function is safe to call from inside an already-running Qt event
#  loop (e.g. from a Jupyter qtconsole): it reuses the existing
#  QApplication instance if one is found.
# =============================================================================

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from .bc_model import BcDocument
from .main_window import BcEditorWindow

DEFAULT_OUTPUT_NAME = "boundary_conditions_gui.py"


@dataclass
class BcEditorTestResult:
    """Per-test BC/mesh result returned by the GUI."""
    test_index: int
    data_file: Path
    saved_bc_path: Optional[Path] = None
    mesh_settings: Optional[Any] = None
    mesh_pkg: Optional[Any] = None
    sample_area_total: Optional[float] = None
    #  Angle from this test's measurement frame to the material 1-axis.  The
    #  fields stay put; the D matrix is rotated by it.
    material_orientation_deg: float = 0.0
    #  The edited BcDocument itself.  The experiment-centred setup dialog
    #  needs the entries in memory -- it applies one topology across every
    #  time step -- and must not depend on the user having saved a .py file.
    document: Optional[Any] = None


@dataclass
class BcEditorResult:
    """What the BC + mesh GUI gave back to the caller.

    The first three fields are ``None`` when the user cancels or never touched
    the corresponding piece of UI; callers detect "no change" by checking
    ``result.saved_bc_path is None`` / ``result.mesh_pkg is None`` and fall
    back to their defaults.

    When the user closes with **Use saved file && run solver**,
    ``saved_bc_path`` is set and ``sample_area_total`` / ``max_iterations`` /
    ``convergence_tolerance`` mirror the Solver settings tab (``None`` area
    means the MAT loader falls back to the bounding-box estimate).
    """
    saved_bc_path: Optional[Path] = None
    mesh_settings: Optional[Any] = None     # MeshSettings -- avoid hard import
    mesh_pkg: Optional[Any] = None          # MeshPackage  -- avoid hard import
    sample_area_total: Optional[float] = None
    material_orientation_deg: Optional[float] = None
    max_iterations: Optional[int] = None
    convergence_tolerance: Optional[float] = None
    test_results: List[BcEditorTestResult] = field(default_factory=list)


def _initial_document(initial_bc_file: Optional[Path]) -> BcDocument:
    """Build the starting document for the editor.

    Accepts a Python BC module (``.py``) or, as a convenience, a legacy
    Q_Matrix-style JSON file (``.json``).  Anything else -- or a missing
    path -- starts the editor empty.
    """
    if initial_bc_file is None:
        return BcDocument()
    path = Path(initial_bc_file)
    if not path.is_file():
        return BcDocument()
    try:
        return BcDocument.from_any_file(path)
    except Exception as exc:                # pragma: no cover -- diagnostic only
        print(f"[bc_gui] warning: could not pre-load {path}: {exc}")
        return BcDocument()


def launch_bc_editor(mesh_pkg=None,
                     initial_bc_file: Optional[Path] = None,
                     output_dir: Optional[Path] = None,
                     data_file: Optional[Path] = None,
                     data_files: Optional[List[Path]] = None,
                     identification_settings: Optional[Any] = None,
                     analysis_type: Optional[str] = None,
                     mesh_editor_seed_settings: Optional[Any] = None,
                     initial_document: Optional[Any] = None,
                     window_title: Optional[str] = None,
                     initial_material_orientation_deg: float = 0.0,
                     external_values_managed_by_parent: bool = False,
                     parent: Optional[Any] = None,
                     ) -> BcEditorResult:
    """Open the plate BC + mesh editor and return what the user did.

    Parameters
    ----------
    mesh_pkg
        Optional virtual mesh from ``vfm_mesh.build_mesh``.  Pass ``None``
        to start with no mesh on the canvas (measurement overlay only);
        in that case ``mesh_editor_seed_settings`` seeds the Virtual mesh
        tab until the user clicks Apply.
    initial_bc_file
        Optional path to an existing ``.py`` (or legacy ``.json``) file
        used to pre-populate the editor.  If not provided, the editor
        starts empty.
    output_dir
        Folder where the GUI's Save button writes the ``.py`` file.  If
        not provided, defaults to ``./boundary_conditions`` next to the caller.
    data_file
        Optional path to the experimental MAT file.  When provided the
        toolbar exposes a "Show measurement points" toggle that lazily
        loads the point cloud and overlays it on the virtual mesh.
    data_files
        Ordered experimental MAT files for a stacked run.  When omitted,
        ``data_file`` is used as the single-test legacy path.
    identification_settings
        Optional :class:`vfm_user_inputs.IdentificationSettings` used to seed
        the Solver settings tab (max iterations and coefficients tolerance).
        When omitted, sensible plate defaults are used.
    mesh_editor_seed_settings
        Optional :class:`vfm_user_inputs.MeshSettings` used when
        ``mesh_pkg`` is ``None`` to populate the Virtual mesh spin boxes.
        When omitted as well, the editor opens on a placeholder mesh whose
        bounds are refitted to the measurement cloud as soon as the MAT
        file is read.
    external_values_managed_by_parent
        When true, this editor defines only EXTERNAL group topology.  The
        force/moment magnitude is then entered once per load step in the
        parent experiment table.  The standalone default retains the legacy
        single-case value field.
    parent
        Widget that opened the editor.  Pass it when launching from a modal
        dialog (e.g. the run-setup dialog): the editor is then parented to
        it and made application-modal, so it sits on top of the modal stack
        and actually receives input instead of being blocked by its opener.

    Returns
    -------
    BcEditorResult
        ``saved_bc_path`` / ``mesh_settings`` / ``mesh_pkg``, each
        ``None`` when the user did not change the corresponding piece.
    """
    # PySide6 import is deferred to keep the module importable on systems
    # without Qt installed (so the smoke test can run headless).
    from PySide6.QtCore import QEvent, QEventLoop, QObject, Qt
    from PySide6.QtWidgets import QApplication

    if output_dir is None:
        output_dir = Path.cwd() / "boundary_conditions"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    document = (initial_document if initial_document is not None
                else _initial_document(initial_bc_file))
    if data_files is None:
        data_files = [Path(data_file)] if data_file is not None else []
    else:
        data_files = [Path(p) for p in data_files]
    if data_file is None and data_files:
        data_file = data_files[0]

    existing_app = QApplication.instance()
    owns_app = existing_app is None
    app = existing_app if existing_app is not None else QApplication(sys.argv)

    window = BcEditorWindow(
        mesh_pkg=mesh_pkg,
        document=document,
        output_dir=output_dir,
        data_file=data_file,
        data_files=data_files,
        identification_settings=identification_settings,
        analysis_type=analysis_type,
        mesh_editor_seed_settings=mesh_editor_seed_settings,
        external_values_managed_by_parent=(
            external_values_managed_by_parent),
        parent=parent,
    )
    window.set_material_orientation_deg(initial_material_orientation_deg)
    if window_title:
        window.setWindowTitle(window_title)
    if parent is not None:
        #  The opener is typically an application-modal dialog, which would
        #  otherwise swallow every click meant for this window.  Making the
        #  editor application-modal too pushes it onto the top of Qt's modal
        #  stack, so it takes input and the dialog waits behind it.
        window.setWindowModality(Qt.WindowModality.ApplicationModal)
    window.show()
    window.raise_()
    window.activateWindow()

    def _capture_result() -> BcEditorResult:
        return BcEditorResult(
            saved_bc_path=window.saved_path(),
            mesh_settings=window.final_mesh_settings(),
            mesh_pkg=window.final_mesh_pkg(),
            sample_area_total=window.final_sample_area_total(),
            material_orientation_deg=window.final_material_orientation_deg(),
            max_iterations=window.final_max_iterations(),
            convergence_tolerance=window.final_convergence_tolerance(),
            test_results=window.final_test_results(),
        )

    if owns_app:
        app.exec()
        return _capture_result()

    # An earlier dialog (e.g. run setup) already owns QApplication.
    # Quit a local loop on window close, capturing results while the
    # window is still alive (do not rely on destroyed/WA_DeleteOnClose).
    loop = QEventLoop()
    captured: dict = {"result": None}

    class _CloseWatcher(QObject):
        def eventFilter(self, watched, event):  # noqa: N802
            if watched is window and event.type() == QEvent.Type.Close:
                captured["result"] = _capture_result()
                loop.quit()
            return False

    watcher = _CloseWatcher(window)
    window.installEventFilter(watcher)
    loop.exec()
    window.removeEventFilter(watcher)
    result = (captured["result"] if captured["result"] is not None
              else _capture_result())
    if parent is not None:
        #  Qt owns the parented window, so drop it once the results are out;
        #  otherwise every Edit click leaves a hidden editor behind.
        window.deleteLater()
    return result


def launch_virtual_field_viewer(mesh_pkg, identification_result) -> None:
    """Open the post-run 3D viewer for optimal virtual fields (blocks)."""
    from .virtual_field_viewer import launch_virtual_field_viewer as _launch
    _launch(mesh_pkg, identification_result)


def launch_identifiability_diagnosis(identification_result) -> None:
    """Open the post-run identifiability diagnosis window."""
    from .identifiability_viewer import launch_identifiability_diagnosis as _launch
    _launch(identification_result)
