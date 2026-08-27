# =============================================================================
#  bc_gui package (plate)
# -----------------------------------------------------------------------------
#  Optional PySide6-based boundary-condition editor for the plate VFM solver.
#
#  The actual GUI submodules (``app``, ``main_window``, ``mesh_canvas``,
#  ``bc_dialogs``) import PySide6 and matplotlib eagerly.  This top-level
#  ``__init__`` does NOT, so that the headless modules (``bc_model``, the
#  smoke test) remain importable on a stripped-down Python install with
#  only numpy / scipy / scikit-fem available.
#
#  ``launch_bc_editor`` and the matching ``BcEditorResult`` dataclass are
#  exposed via ``__getattr__`` and are only imported the first time they
#  are accessed -- run_vfm.py only calls them when the user opts in.
# =============================================================================

from __future__ import annotations

from typing import Any

__all__ = [
    "launch_run_setup",
    "RunSetupResult",
    "launch_bc_editor",
    "BcEditorResult",
    "BcEditorTestResult",
    "launch_virtual_field_viewer",
    "launch_identifiability_diagnosis",
    "launch_results_window",
]


def __getattr__(name: str) -> Any:
    if name == "launch_run_setup":
        from .run_setup_dialog import launch_run_setup as _setup
        return _setup
    if name == "RunSetupResult":
        from .run_setup_dialog import RunSetupResult as _SetupResult
        return _SetupResult
    if name == "launch_bc_editor":
        from .app import launch_bc_editor as _launch
        return _launch
    if name == "BcEditorResult":
        from .app import BcEditorResult as _Result
        return _Result
    if name == "BcEditorTestResult":
        from .app import BcEditorTestResult as _TestResult
        return _TestResult
    if name == "launch_virtual_field_viewer":
        from .app import launch_virtual_field_viewer as _vf
        return _vf
    if name == "launch_identifiability_diagnosis":
        from .app import launch_identifiability_diagnosis as _diag
        return _diag
    if name == "launch_results_window":
        from .results_window import launch_results_window as _results
        return _results
    raise AttributeError(f"module 'bc_gui' has no attribute {name!r}")
