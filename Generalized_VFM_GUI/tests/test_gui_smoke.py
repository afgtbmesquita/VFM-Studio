"""GUI smoke test (offscreen)."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("VFMSTUDIO_NO_VISTA", "1")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from vfmstudio.gui.main_window import MainWindow


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_main_window_constructs(qapp):
    window = MainWindow()
    assert "VFM Studio" in window.windowTitle()
    assert hasattr(window.props, "request_visualize_steps")
    assert window.viewport.load_plot_visible() is False
    assert hasattr(window.viewport, "_load_plot")
    overlay = window.viewport._load_plot
    assert overlay._plot_canvas is not None
    assert overlay._spec_canvas is not None
    from PySide6.QtGui import QPalette

    fill = overlay._force.palette().color(QPalette.ColorRole.Base).name().lower()
    assert fill == "#e6e6e6"
    xy_axes = window.actions.get("xy_axes")
    assert xy_axes.isCheckable()
    assert xy_axes.isChecked() is False
    assert hasattr(window.props, "_exp_fit_axes")
    assert hasattr(window.props, "view_changed")
    assert hasattr(window.props, "_mesh_show_counts")
    assert window.props._mesh_show_counts.isChecked() is False
    assert "Add load steps" in window.actions.get("add_step").text()
    window.close()
