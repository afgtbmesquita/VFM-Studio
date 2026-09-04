"""Qt application bootstrap. GUI imports live here, not in ``core``."""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QEventLoop, QTimer, Qt, qInstallMessageHandler
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QMessageBox, QStyleFactory, QWidget

from vfmstudio import __version__
from vfmstudio.branding import APP_AUTHOR, APP_NAME
from vfmstudio.gui.splash import show_splash
from vfmstudio.resources import app_icon


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


def _pump(app: QApplication) -> None:
    app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)


SPLASH_MINIMUM_S = 1.0


def _hold_splash(
    app: QApplication,
    splash: Optional[QWidget],
    started: float,
    minimum_s: float = SPLASH_MINIMUM_S,
) -> None:
    """Keep the banner visible long enough to read, even on a fast launch."""
    if splash is None:
        return
    remaining_ms = int(max(0.0, minimum_s - (time.monotonic() - started)) * 1000)
    if remaining_ms <= 0:
        return
    splash.raise_()
    app.processEvents()
    loop = QEventLoop()
    QTimer.singleShot(remaining_ms, loop.quit)
    loop.exec()


def _close_splash(splash: Optional[QWidget], window: Optional[QWidget] = None) -> None:
    if splash is None:
        return
    finish = getattr(splash, "finish", None)
    if callable(finish) and window is not None:
        finish(window)
    else:
        splash.close()


def run_app(project_path: Optional[Path] = None) -> int:
    from vfmstudio.platform.windows import prepare_windows_app_identity

    prepare_windows_app_identity()
    qInstallMessageHandler(_qt_message_handler)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication.instance() or QApplication(sys.argv)
    fusion = QStyleFactory.create("Fusion")
    if fusion is not None:
        app.setStyle(fusion)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    app.setOrganizationName(APP_AUTHOR)
    font = QFont("Arial")
    font.setPointSize(9)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(font)
    app.setWindowIcon(app_icon())

    splash = show_splash(app)
    shown_at = time.monotonic()
    if splash is not None:
        _pump(app)

    app.setStyleSheet(_load_stylesheet())
    if splash is not None:
        splash.setStyleSheet("background: #eef2f5; border: none;")
        splash.raise_()
        _pump(app)

    def _excepthook(exc_type, exc, tb) -> None:
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        sys.stderr.write(text)
        try:
            from vfmstudio.platform.scipy_wdac import explain_native_import_error

            QMessageBox.critical(
                None, APP_NAME, f"Unhandled error:\n\n{explain_native_import_error(exc)}"
            )
        except Exception:
            pass

    sys.excepthook = _excepthook

    window = None
    try:
        from vfmstudio.gui.main_window import MainWindow

        if splash is not None:
            _pump(app)
        window = MainWindow()
        _hold_splash(app, splash, shown_at)
        window.show()
        _close_splash(splash, window)
        _bring_to_front(window)
        QTimer.singleShot(0, lambda: _bring_to_front(window))
        QTimer.singleShot(250, lambda: _bring_to_front(window))
        if project_path is not None:
            window.open_project_path(Path(project_path))
        return app.exec()
    except Exception:
        _close_splash(splash, window)
        raise
