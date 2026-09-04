"""Publication export of post-processing matplotlib figures.

PDF is written as a vector file with Type-42 (embedded TrueType) fonts —
the same recipe as the campaign ``make_figures`` scripts.  Any rasterized
artists (heatmaps, mplot3d surfaces) are sampled at PNG_DPI so they stay
sharp when the PDF is zoomed.  PNG is always PNG_DPI.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

PNG_DPI = 600

PLOT_EXPORT_STEMS = {
    "post_processing": "coefficient_evolution",
    "post_plots": "coefficient_evolution",
    "post_parameter_space": "parameter_space",
    "post_stress_space": "stress_resultant_space",
    "post_experiment_check": "experiment_check",
    "post_identifiability_strength": "identifiability_strength",
    "post_virtual_fields": "virtual_fields",
    "post_uncertainty_cov": "covariance_theta",
    "post_uncertainty_experiments": "experiment_uncertainty",
    "post_uncertainty_ab": "matrices_ab",
    "post_uncertainty_info": "information_geometry",
    "post_uncertainty_loo": "leave_one_out",
}

FILE_FILTER = "PDF (*.pdf);;PNG (*.png)"


def default_export_stem(kind: str) -> str:
    return PLOT_EXPORT_STEMS.get(str(kind), "plot")


def resolve_export_path(chosen: str, selected_filter: str = "") -> Path:
    """Honor an explicit suffix; otherwise follow the dialog filter."""
    path = Path(chosen)
    suffix = path.suffix.lower()
    if suffix in {".pdf", ".png"}:
        return path
    if "png" in str(selected_filter).lower():
        return path.with_suffix(".png")
    return path.with_suffix(".pdf")


def _savefig(figure, path: Path, suffix: str) -> None:
    import matplotlib as mpl

    kwargs = dict(
        format=suffix,
        facecolor="white",
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.04,
        transparent=False,
        dpi=PNG_DPI,
    )
    if suffix == "pdf":
        with mpl.rc_context(
            {
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "pdf.compression": 0,
            }
        ):
            figure.savefig(path, **kwargs)
        return
    figure.savefig(path, **kwargs)


def save_figure(figure, path: str | Path) -> Path:
    """Write ``figure`` to ``path`` (.pdf vector or .png at 600 dpi).

    Saves through a local temp file first.  Direct ``savefig`` into a
    Google Drive / OneDrive folder can land a tiny cloud stub instead of
    the real file.
    """
    dest = Path(path)
    suffix = dest.suffix.lower().lstrip(".")
    if suffix not in {"pdf", "png"}:
        raise ValueError(f"Export path must end in .pdf or .png, got {dest.name!r}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / dest.name
        _savefig(figure, src, suffix)
        shutil.copy2(src, dest)
    if not dest.is_file() or dest.stat().st_size < 200:
        raise OSError(f"Figure did not land on disk: {dest}")
    return dest


def has_exportable_plot(kind: str) -> bool:
    return str(kind) in PLOT_EXPORT_STEMS
