# =============================================================================
#  bc_gui / identifiability_viewer.py
# -----------------------------------------------------------------------------
#  Post-run GUI for Fisher-like constitutive identifiability diagnostics.
# =============================================================================

from __future__ import annotations

import math
from typing import List

import numpy as np
import matplotlib.colors as mcolors
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.tri import Triangulation
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from vfm_identifiability import (
    IdentifiabilityCaseDiagnostics,
    IdentifiabilityDiagnostics,
    compute_identifiability_diagnostics,
)


class IdentifiabilityDiagnosisWindow(QMainWindow):
    """Window showing H, singular values, correlation, and CID fields."""

    def __init__(self, diagnostics: IdentifiabilityDiagnostics) -> None:
        super().__init__()
        self.setWindowTitle("Plate VFM - Identifiability Diagnosis")
        self.resize(1280, 860)
        self._diagnostics = diagnostics

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        header = QLabel(
            f"Material: {diagnostics.material_type}    "
            f"Analysis: {diagnostics.analysis_type}    "
            f"Parameters: {', '.join(diagnostics.parameter_labels)}"
        )
        header.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(header)

        tabs = QTabWidget()
        tabs.addTab(self._build_matrix_tab("Aggregate", diagnostics.aggregate),
                    "Aggregate")
        for idx, case in enumerate(diagnostics.cases, start=1):
            tabs.addTab(self._build_case_tab(case), f"Test {idx}")
        layout.addWidget(tabs, 1)

    def _build_matrix_tab(self, title: str, matrix_diag) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(_summary_box(title, matrix_diag))
        left_layout.addWidget(_matrix_table("Information matrix H",
                                            matrix_diag.H,
                                            self._diagnostics.parameter_labels))
        left_layout.addWidget(_singular_table(matrix_diag.singular_values))
        split.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        fig = Figure(figsize=(5.5, 4.5), constrained_layout=True)
        canvas = FigureCanvasQTAgg(fig)
        _plot_correlation_heatmap(fig, matrix_diag.correlation,
                                  self._diagnostics.parameter_labels)
        right_layout.addWidget(canvas, 1)
        right_layout.addWidget(NavigationToolbar2QT(canvas, right))
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)

        layout.addWidget(split, 1)
        return widget

    def _build_case_tab(self, case: IdentifiabilityCaseDiagnostics) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        title = case.label
        if case.data_file is not None:
            title += f"    ({case.data_file.name})"
        heading = QLabel(title)
        heading.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(heading)

        split = QSplitter(Qt.Vertical)
        matrix_host = self._build_matrix_tab(case.label, case.matrix)
        split.addWidget(matrix_host)

        cid_host = QWidget()
        cid_layout = QVBoxLayout(cid_host)
        fig = Figure(figsize=(10, 7), constrained_layout=True)
        canvas = FigureCanvasQTAgg(fig)
        _plot_cid_fields(
            fig, case, self._diagnostics.parameter_labels, case.mesh_pkg)
        cid_layout.addWidget(canvas, 1)
        cid_layout.addWidget(NavigationToolbar2QT(canvas, cid_host))
        split.addWidget(cid_host)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 2)

        layout.addWidget(split, 1)
        return widget


def launch_identifiability_diagnosis(identification_result) -> None:
    """Open the identifiability diagnosis window modally."""
    import sys
    from PySide6.QtCore import QEventLoop
    from PySide6.QtWidgets import QApplication

    cases = getattr(identification_result, "diagnostic_cases", None)
    if not cases:
        raise ValueError(
            "Identification result does not include diagnostic cases. "
            "Re-run the solver so measurement fields are retained."
        )
    constitutive_model = getattr(identification_result, "constitutive_model", None)
    if constitutive_model is None:
        raise ValueError(
            "Identification result does not include a constitutive model."
        )
    diagnostics = compute_identifiability_diagnostics(cases, constitutive_model)

    existing_app = QApplication.instance()
    owns_app = existing_app is None
    app = existing_app if existing_app is not None else QApplication(sys.argv)
    if owns_app:
        app.setQuitOnLastWindowClosed(True)

    window = IdentifiabilityDiagnosisWindow(diagnostics)
    window.setAttribute(Qt.WA_DeleteOnClose, True)
    window.show()
    window.raise_()
    window.activateWindow()

    if owns_app:
        window.destroyed.connect(app.quit)
        app.exec()
    else:
        loop = QEventLoop()
        window.destroyed.connect(loop.quit)
        loop.exec()


def _summary_box(title: str, matrix_diag) -> QGroupBox:
    box = QGroupBox("Summary")
    layout = QVBoxLayout(box)
    det = matrix_diag.determinant
    sign = matrix_diag.slogdet_sign
    logabs = matrix_diag.slogdet
    text = QTextEdit()
    text.setReadOnly(True)
    text.setMaximumHeight(150)
    text.setPlainText(
        "\n".join([
            title,
            f"det(H) = {_format_number(det)}",
            f"sign(det(H)) = {_format_number(sign)}",
            f"log(abs(det(H))) = {_format_number(logabs)}",
            f"cond(H) = {_format_number(matrix_diag.condition_number)}",
            f"covariance = {'pinv(H)' if matrix_diag.used_pseudoinverse else 'inv(H)'}",
        ])
    )
    layout.addWidget(text)
    return box


def _matrix_table(title: str,
                  matrix: np.ndarray,
                  labels: List[str]) -> QGroupBox:
    box = QGroupBox(title)
    layout = QVBoxLayout(box)
    table = QTableWidget(matrix.shape[0], matrix.shape[1])
    table.setHorizontalHeaderLabels(labels)
    table.setVerticalHeaderLabels(labels)
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.setSelectionMode(QAbstractItemView.ExtendedSelection)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            table.setItem(i, j, QTableWidgetItem(_format_number(matrix[i, j])))
    table.resizeColumnsToContents()
    layout.addWidget(table)
    return box


def _singular_table(singular_values: np.ndarray) -> QGroupBox:
    box = QGroupBox("Singular values")
    layout = QVBoxLayout(box)
    table = QTableWidget(singular_values.size, 2)
    table.setHorizontalHeaderLabels(["index", "value"])
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    for idx, value in enumerate(singular_values):
        table.setItem(idx, 0, QTableWidgetItem(str(idx + 1)))
        table.setItem(idx, 1, QTableWidgetItem(_format_number(value)))
    table.resizeColumnsToContents()
    layout.addWidget(table)
    return box


def _plot_correlation_heatmap(fig: Figure,
                              correlation: np.ndarray,
                              labels: List[str]) -> None:
    fig.clear()
    ax = fig.add_subplot(111)
    data = np.asarray(correlation, dtype=float)
    im = ax.imshow(data, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_title("Correlation index")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            label = "nan" if not np.isfinite(value) else f"{value:.2f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.85)


def _cid_log_norm(values: np.ndarray) -> mcolors.LogNorm:
    """Colour normalisation for CID fields (strictly non-negative)."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    positive = finite[finite > 0.0]
    if positive.size == 0:
        vmin = 1.0e-30
        vmax = 1.0
    else:
        vmin = max(float(positive.min()), 1.0e-30)
        vmax = max(float(positive.max()), vmin * 10.0)
    return mcolors.LogNorm(vmin=vmin, vmax=vmax)


def _plot_mesh_element_outlines(ax, mesh_pkg) -> None:
    """Draw surviving virtual-element edges on a 2-D axes."""
    coords = np.asarray(mesh_pkg.node_coords, dtype=float)
    elems = np.asarray(mesh_pkg.element_nodes, dtype=int)
    for nodes in elems:
        ring = coords[nodes]
        closed = np.vstack([ring, ring[0]])
        ax.plot(closed[:, 0], closed[:, 1],
                color="#555555", linewidth=0.6, alpha=0.85, zorder=2)


def _plot_cid_fields(fig: Figure,
                     case: IdentifiabilityCaseDiagnostics,
                     labels: List[str],
                     mesh_pkg=None) -> None:
    fig.clear()
    n_param = len(labels)
    ncols = int(math.ceil(math.sqrt(n_param)))
    nrows = int(math.ceil(n_param / ncols))
    points = np.asarray(case.points, dtype=float)
    x = points[:, 0]
    y = points[:, 1]
    CID = np.asarray(case.CID, dtype=float)

    use_mesh = mesh_pkg is not None
    if use_mesh:
        from bc_gui.mesh_canvas import _filtered_triangulation

    title = "Constitutive information density (log scale)"
    if case.n_points_total > case.n_points_used > 0:
        title += (
            f"  ({case.n_points_used}/{case.n_points_total} points in mesh)"
        )

    for idx, label in enumerate(labels):
        ax = fig.add_subplot(nrows, ncols, idx + 1)
        values = CID[idx, :]
        norm = _cid_log_norm(values)
        plotted = False
        if points.shape[0] >= 3:
            try:
                if use_mesh:
                    triang = _filtered_triangulation(x, y)
                else:
                    triang = Triangulation(x, y)
                contour = ax.tricontourf(
                    triang, values, cmap="viridis", norm=norm)
                fig.colorbar(contour, ax=ax, shrink=0.78)
                plotted = True
            except Exception:
                plotted = False
        if not plotted:
            scatter = ax.scatter(
                x, y, c=values, s=18.0, cmap="viridis", norm=norm, zorder=3)
            fig.colorbar(scatter, ax=ax, shrink=0.78)
        if use_mesh:
            _plot_mesh_element_outlines(ax, mesh_pkg)
        ax.set_title(label)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    for idx in range(n_param, nrows * ncols):
        ax = fig.add_subplot(nrows, ncols, idx + 1)
        ax.set_visible(False)
    fig.suptitle(title)


def _format_number(value) -> str:
    value = float(value)
    if not np.isfinite(value):
        return str(value)
    return f"{value:.6e}"
