"""Publication export of post-processing plots (PDF vector / PNG 600 dpi)."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from matplotlib.figure import Figure

from vfmstudio.gui.results.plot_export import (
    PNG_DPI,
    has_exportable_plot,
    resolve_export_path,
    save_figure,
)


def _png_dpi(path: Path) -> float:
    data = path.read_bytes()
    marker = data.find(b"pHYs")
    assert marker >= 0, "PNG is missing a pHYs dpi chunk"
    x_ppm, y_ppm, unit = struct.unpack(">IIB", data[marker + 4 : marker + 13])
    assert unit == 1
    assert x_ppm == y_ppm
    return x_ppm * 0.0254


def _line_figure() -> Figure:
    figure = Figure(figsize=(4.0, 3.0), facecolor="#FFFFFF")
    axis = figure.add_subplot(111)
    axis.plot([0.0, 1.0, 2.0], [0.0, 1.0, 0.5], color="#0072B2")
    axis.set_xlabel(r"$D_{11}$")
    axis.set_ylabel("value")
    axis.set_title("Coefficient")
    return figure


def test_resolve_export_path_follows_suffix_or_filter():
    assert resolve_export_path(r"C:\out\plot.pdf", "PNG (*.png)").suffix == ".pdf"
    assert resolve_export_path(r"C:\out\plot.png", "PDF (*.pdf)").suffix == ".png"
    assert resolve_export_path(r"C:\out\plot", "PNG (*.png)").name == "plot.png"
    assert resolve_export_path(r"C:\out\plot", "PDF (*.pdf)").name == "plot.pdf"


def test_has_exportable_plot_covers_plot_pages_only():
    assert has_exportable_plot("post_plots")
    assert has_exportable_plot("post_virtual_fields")
    assert has_exportable_plot("post_stress_space")
    assert has_exportable_plot("post_uncertainty_cov")
    assert not has_exportable_plot("post_identifiability")
    assert not has_exportable_plot("post_uncertainty")
    assert not has_exportable_plot("material")


def test_save_figure_pdf_is_vector_and_png_is_600_dpi(tmp_path):
    figure = _line_figure()
    pdf = tmp_path / "plot.pdf"
    png = tmp_path / "plot.png"
    save_figure(figure, pdf)
    save_figure(figure, png)
    assert pdf.read_bytes()[:4] == b"%PDF"
    assert pdf.stat().st_size > 500
    # Type-42 embedding writes a TrueType stream (FontFile2), not Type 3 bitmaps.
    assert b"/FontFile2" in pdf.read_bytes() or b"/FontFile3" in pdf.read_bytes()
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert abs(_png_dpi(png) - PNG_DPI) < 1.0


def test_save_figure_rejects_unknown_suffix(tmp_path):
    figure = _line_figure()
    with pytest.raises(ValueError, match=r"\.pdf or \.png"):
        save_figure(figure, tmp_path / "plot.jpg")


def test_plot_style_uses_abaqus_sans_serif():
    from vfmstudio.gui.results.plot_style import MPL_RCPARAMS, heading_stylesheet

    assert MPL_RCPARAMS["font.family"] == "sans-serif"
    assert "Arial" in MPL_RCPARAMS["font.sans-serif"]
    assert "Times New Roman" not in str(MPL_RCPARAMS)
    assert "Arial" in heading_stylesheet()
    assert "Times" not in heading_stylesheet()
