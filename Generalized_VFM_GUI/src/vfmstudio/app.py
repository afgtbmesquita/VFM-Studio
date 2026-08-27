"""Qt application bootstrap. GUI imports live here, not in ``core``."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QTimer, Qt, qInstallMessageHandler
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

from vfmstudio import __version__


def _qt_message_handler(mode, context, message: str) -> None:
    # Keep VTK/Qt plugin noise out of a modal crash dialog.
    _ = (mode, context)
    if message:
        sys.stderr.write(message + "\n")


def _bring_to_front(window: QWidget) -> None:
    """Raise the main window above other apps (Windows focus-steal workaround)."""
    window.show()
    window.setWindowState(
        (window.windowState() & ~Qt.WindowState.WindowMinimized) | Qt.WindowState.WindowActive
    )
    window.raise_()
    window.activateWindow()
    app = QApplication.instance()
    if app is not None:
        app.setActiveWindow(window)
    if sys.platform != "win32":
        return
    try:
        import ctypes

        user32 = ctypes.windll.user32
        hwnd = int(window.winId())
        hwnd_topmost = -1
        hwnd_notopmost = -2
        flags = 0x0001 | 0x0002 | 0x0040  # NOSIZE | NOMOVE | SHOWWINDOW
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        user32.SetWindowPos(hwnd, hwnd_topmost, 0, 0, 0, 0, flags)
        user32.SetWindowPos(hwnd, hwnd_notopmost, 0, 0, 0, 0, flags)
        foreground = user32.GetForegroundWindow()
        this_tid = user32.GetWindowThreadProcessId(hwnd, None)
        fg_tid = user32.GetWindowThreadProcessId(foreground, None) if foreground else 0
        if fg_tid and fg_tid != this_tid:
            user32.AttachThreadInput(fg_tid, this_tid, True)
            user32.BringWindowToTop(hwnd)
            user32.SetForegroundWindow(hwnd)
            user32.AttachThreadInput(fg_tid, this_tid, False)
        else:
            user32.BringWindowToTop(hwnd)
            user32.SetForegroundWindow(hwnd)
    except Exception:
        pass


def _load_stylesheet() -> str:
    qss = Path(__file__).resolve().parent / "gui" / "style" / "app.qss"
    if qss.is_file():
        return qss.read_text(encoding="utf-8")
    return ""


def run_app(project_path: Optional[Path] = None) -> int:
    qInstallMessageHandler(_qt_message_handler)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("VFM Studio")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("VFM Studio")
    font = QFont("Arial")
    font.setPointSize(9)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(font)
    app.setStyleSheet(_load_stylesheet())

    def _excepthook(exc_type, exc, tb) -> None:
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        sys.stderr.write(text)
        try:
            QMessageBox.critical(None, "VFM Studio", f"Unhandled error:\n\n{exc}")
        except Exception:
            pass

    sys.excepthook = _excepthook

    from vfmstudio.gui.main_window import MainWindow

    window = MainWindow()
    window.show()
    _bring_to_front(window)
    QTimer.singleShot(0, lambda: _bring_to_front(window))
    QTimer.singleShot(250, lambda: _bring_to_front(window))
    if project_path is not None:
        window.open_project_path(Path(project_path))
    return app.exec()
