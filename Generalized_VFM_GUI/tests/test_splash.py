"""Startup banner shown before the main window loads."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("VFMSTUDIO_NO_VISTA", "1")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QWidget

from vfmstudio.branding import APP_AUTHOR, APP_LICENSE_LINE, APP_NAME
from vfmstudio.gui.splash import BANNER_HEIGHT, BANNER_WIDTH, SplashBanner, build_splash_pixmap, show_splash
from vfmstudio.resources import app_icon


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_splash_pixmap_size(qapp):
    pixmap = build_splash_pixmap(1.0)
    assert not pixmap.isNull()
    assert pixmap.width() == BANNER_WIDTH
    assert pixmap.height() == BANNER_HEIGHT
    assert pixmap.devicePixelRatio() == 1.0


def test_splash_pixmap_uses_device_pixel_ratio(qapp):
    pixmap = build_splash_pixmap(2.0)
    assert pixmap.width() == BANNER_WIDTH * 2
    assert pixmap.height() == BANNER_HEIGHT * 2
    assert pixmap.devicePixelRatio() == 1.0


def test_show_splash_is_visible(qapp):
    os.environ.pop("VFMSTUDIO_NO_SPLASH", None)
    splash = show_splash(qapp)
    assert isinstance(splash, (SplashBanner, QWidget))
    assert splash.isVisible()
    assert splash.width() <= BANNER_WIDTH
    assert splash.height() <= BANNER_HEIGHT
    assert APP_NAME in splash.windowTitle()
    splash.close()


def test_show_splash_can_be_disabled(qapp, monkeypatch):
    monkeypatch.setenv("VFMSTUDIO_NO_SPLASH", "1")
    assert show_splash(qapp) is None


def test_app_icon_is_bundled():
    icon_obj = app_icon()
    assert not icon_obj.isNull()
    assert not icon_obj.pixmap(32, 32).isNull()


def test_windows_ico_is_bundled():
    from vfmstudio.resources import _resources_dir

    ico = _resources_dir() / "LamXtract_Icon.ico"
    assert ico.is_file()
    assert ico.stat().st_size > 1_000


def test_official_logo_is_bundled():
    from vfmstudio.gui.splash import logo_path

    path = logo_path()
    assert path is not None
    assert path.name == "LamXtract_Logo.png"
    assert path.is_file()
    assert path.stat().st_size > 300_000


def test_logo_never_upscales(qapp):
    from vfmstudio.gui.splash import _fit_logo_pixmap, _load_logo_master

    _LOGO_MASTER = None
    import vfmstudio.gui.splash as splash_mod

    splash_mod._LOGO_MASTER = None
    logo = _load_logo_master()
    assert logo is not None
    fitted = _fit_logo_pixmap(logo, 4000, 4000)
    assert fitted.width() <= logo.width()
    assert fitted.height() <= logo.height()
    small = _fit_logo_pixmap(logo, 200, 40)
    assert small.width() <= 200
    assert small.height() <= 40
    assert APP_NAME == "LamXtract"
    assert APP_AUTHOR == "Afonso Mesquita"
    assert "GPL" in APP_LICENSE_LINE
    assert "warranty" in APP_LICENSE_LINE.lower()
