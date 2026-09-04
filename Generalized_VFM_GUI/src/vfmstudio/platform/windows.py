"""Windows-only integration (taskbar identity, focus helpers)."""

from __future__ import annotations

import sys

_APP_USER_MODEL_ID = "AfonsoMesquita.abdXtract.1"
_PREPARED = False


def prepare_windows_app_identity() -> None:
    """Give this Python process its own taskbar identity (not python.exe).

    Must run before ``QApplication`` is constructed so Windows uses the app
    icon in the taskbar instead of the Python interpreter icon.
    """
    global _PREPARED
    from vfmstudio.platform.scipy_wdac import prepare_scipy_native_imports

    prepare_scipy_native_imports()
    if _PREPARED or sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(_APP_USER_MODEL_ID)
        _PREPARED = True
    except Exception:
        pass
