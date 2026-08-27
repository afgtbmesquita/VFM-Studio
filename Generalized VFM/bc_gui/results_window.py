# =============================================================================
#  bc_gui / results_window.py
# -----------------------------------------------------------------------------
#  Post-run results window: the converged coefficients, how they got there, and
#  what the noise model does and does not say about them -- shown in the GUI
#  instead of scrolling past in a terminal that has already closed.
#
#  THE ONE THING THIS WINDOW MUST NOT DO
#  ---------------------------------------------------------------------------
#  It must not present eq. (20) as "the uncertainty".  Eq. (20) is the variance
#  of theta UNDER the noise model, and on this campaign the data reject that
#  model at chi2/dof ~ 830.  So the parameter table shows BOTH columns, the
#  evolution plot carries no error band, and the header states the reduced
#  chi2 next to the numbers.  A reader who takes only the headline away should
#  still not come away with a false precision.
# =============================================================================

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

#  Same fixed hue order as the paper figures, so a coefficient is the same
#  colour on screen and in print.  Validated for colour-vision deficiency.
SERIES_COLOUR = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")
SERIES_MARKER = ("o", "s", "^", "D")


def _reduced_chi_square(result) -> Optional[dict]:
    """chi2/dof of the noise model, or None if it cannot be formed."""
    try:
        design, rhs = result.K_matrix, result.U_vector
        covariance = result.global_covariance
        if design is None or rhs is None or covariance is None:
            return None
        residual = design @ result.final_parameters - rhs
        dof = design.shape[0] - design.shape[1]
        if dof <= 0:
            return None
        total = float(residual @ np.linalg.solve(covariance, residual))
        return {"chi2": total, "dof": dof, "reduced": total / dof,
                "birge": float(np.sqrt(total / dof))}
    except (np.linalg.LinAlgError, ValueError, AttributeError):
        return None


class ResultsWindow(QMainWindow):
    """Identification results, with the caveats attached."""

    def __init__(self, identification_result, mesh_pkg=None,
                 project_name: str = "") -> None:
        super().__init__()
        self._result = identification_result
        self._mesh_pkg = mesh_pkg
        self.setWindowTitle("VFM results"
                            + (f" — {project_name}" if project_name else ""))
        self.resize(1060, 760)

        self._labels = list(
            identification_result.constitutive_model.parameter_labels)
        self._goodness = _reduced_chi_square(identification_result)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        layout.addWidget(self._build_header())

        tabs = QTabWidget()
        tabs.addTab(self._build_parameter_tab(), "Coefficients")
        tabs.addTab(self._build_evolution_tab(), "Evolution")
        tabs.addTab(self._build_convergence_tab(), "Convergence")
        tabs.addTab(self._build_noise_tab(), "Noise model")
        layout.addWidget(tabs, 1)

        layout.addLayout(self._build_buttons())

    # ------------------------------------------------------------------
    #  Header
    # ------------------------------------------------------------------
    def _build_header(self) -> QWidget:
        result = self._result
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        headline = "   ".join(
            f"{label} = {value:.4g}"
            for label, value in zip(self._labels, result.final_parameters))
        title = QLabel(headline)
        font = QFont()
        font.setPointSize(13)
        font.setBold(True)
        title.setFont(font)
        title.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(title)

        status = ("converged" if result.converged
                  else "DID NOT CONVERGE — max iterations reached")
        pieces = [f"{status} in {result.iterations_completed} iterations",
                  f"weighting = {getattr(result, 'weighting', 'gls')}"]
        if self._goodness is not None:
            pieces.append(f"chi2/dof = {self._goodness['reduced']:.0f}")
        subtitle = QLabel("N.mm   |   " + "   |   ".join(pieces))
        subtitle.setStyleSheet("color: #52514e;")
        layout.addWidget(subtitle)

        if self._goodness is not None and self._goodness["reduced"] > 2.0:
            warning = QLabel(
                f"The noise model is rejected by the data "
                f"(chi2/dof = {self._goodness['reduced']:.0f}, so it explains "
                f"{100 / self._goodness['reduced']:.2f} % of the residual "
                f"variance). Treat the eq. (20) column as a variance UNDER "
                f"THE MODEL, not as the accuracy of the coefficients.")
            warning.setWordWrap(True)
            warning.setStyleSheet(
                "color: #8a2a17; background: #fdf1ec; border: 1px solid "
                "#e8663c; border-radius: 4px; padding: 6px;")
            layout.addWidget(warning)
        return box

    # ------------------------------------------------------------------
    #  Coefficients
    # ------------------------------------------------------------------
    def _build_parameter_tab(self) -> QWidget:
        result = self._result
        std = result.parameter_std
        birge = self._goodness["birge"] if self._goodness else None

        page = QWidget()
        layout = QVBoxLayout(page)

        headers = ["coefficient", "value (N.mm)", "eq. (20) sigma",
                   "sigma / |value|"]
        if birge is not None:
            headers.append(f"x sqrt(chi2/dof) = {birge:.1f}")
        table = QTableWidget(len(self._labels), len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setEditTriggers(QTableWidget.NoEditTriggers)

        for row, label in enumerate(self._labels):
            value = float(result.final_parameters[row])
            sigma = float(std[row]) if std is not None else float("nan")
            cells = [label, f"{value:.4f}", f"{sigma:.4f}",
                     f"{100.0 * sigma / abs(value):.2f} %" if value else "-"]
            if birge is not None:
                cells.append(f"{sigma * birge:.4f}")
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if column == 0:
                    item.setForeground(
                        _colour(SERIES_COLOUR[row % len(SERIES_COLOUR)]))
                table.setItem(row, column, item)
        layout.addWidget(table)

        note = QLabel(
            "The eq. (20) column answers \"if the ONLY error were spatially "
            "white DIC noise with covariance Gamma-hat, how far would theta "
            "move?\".  It does not answer \"how well is this coefficient "
            "known?\".  The last column rescales it so the fit is consistent "
            "(the Birge correction); treat that as a floor, not a budget.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #52514e;")
        layout.addWidget(note)
        return page

    # ------------------------------------------------------------------
    #  Evolution
    # ------------------------------------------------------------------
    def _build_evolution_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        figure = Figure(figsize=(9, 5.4), constrained_layout=True)
        canvas = FigureCanvas(figure)
        layout.addWidget(NavigationToolbar(canvas, page))
        layout.addWidget(canvas, 1)

        history = self._result.history
        if not history:
            layout.addWidget(QLabel("No iteration history was recorded."))
            return page

        iterations = np.array([0] + [record.iteration for record in history])
        values = np.vstack([history[0].parameters_before]
                           + [record.parameters_after for record in history])

        #  One gridspec, built once: adding a spanning axes after calling
        #  subplots() leaves constrained_layout with two competing layouts and
        #  the right-hand panel overlaps the middle column.
        grid = figure.add_gridspec(2, 3)
        panels = [figure.add_subplot(grid[i // 2, i % 2]) for i in range(4)]
        rate = figure.add_subplot(grid[:, 2])
        for index, axis in enumerate(panels):
            colour = SERIES_COLOUR[index % len(SERIES_COLOUR)]
            axis.axhline(values[-1, index], color="#8f8e88", linewidth=0.8,
                         linestyle=(0, (4, 3)))
            axis.plot(iterations, values[:, index], color=colour,
                      marker=SERIES_MARKER[index % len(SERIES_MARKER)],
                      markersize=4.5, markeredgecolor="white",
                      markeredgewidth=0.6, linewidth=1.5)
            axis.set_title(self._labels[index], fontsize=10)
            axis.set_ylabel("N.mm", fontsize=8)
            axis.set_xticks(iterations)
            axis.tick_params(labelsize=8)
            axis.grid(True, axis="y", color="#dcdbd5", linewidth=0.5)
            axis.set_axisbelow(True)
            for side in ("top", "right"):
                axis.spines[side].set_visible(False)
            if index >= 2:
                axis.set_xlabel("iteration", fontsize=8)
            axis.annotate(f"{values[-1, index]:.4g}",
                          xy=(iterations[-1], values[-1, index]),
                          xytext=(-3, 6), textcoords="offset points",
                          ha="right", fontsize=8, color=colour)

        #  Right-hand column spans both rows: relative distance to the fixed
        #  point, which is the only way to see the last three iterations.
        for index, label in enumerate(self._labels):
            final = values[-1, index]
            distance = np.abs(values[:-1, index] - final) / max(abs(final), 1e-30)
            rate.plot(iterations[:-1], np.maximum(distance, 1e-16),
                      color=SERIES_COLOUR[index % len(SERIES_COLOUR)],
                      marker=SERIES_MARKER[index % len(SERIES_MARKER)],
                      markersize=4.5, markeredgecolor="white",
                      markeredgewidth=0.6, linewidth=1.5, label=label)
        rate.set_yscale("log")
        rate.set_title("Convergence", fontsize=10)
        rate.set_xlabel("iteration", fontsize=8)
        rate.set_ylabel("|theta_k - theta_final| / |theta_final|", fontsize=8)
        rate.set_xticks(iterations[:-1])
        rate.tick_params(labelsize=8)
        rate.grid(True, axis="y", color="#dcdbd5", linewidth=0.5)
        rate.set_axisbelow(True)
        for side in ("top", "right"):
            rate.spines[side].set_visible(False)
        rate.legend(fontsize=8, frameon=False, ncol=2)

        figure.suptitle(
            "Coefficient evolution (iteration 0 is the initial guess). "
            "No error band is drawn: see the Coefficients tab.", fontsize=9)
        canvas.draw_idle()
        return page

    # ------------------------------------------------------------------
    #  Convergence
    # ------------------------------------------------------------------
    def _build_convergence_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        history = self._result.history
        diagnostics = list(getattr(self._result, "iteration_diagnostics", []))

        headers = ["iteration"] + list(self._labels) + [
            "deltaQ", "deltaC", "cond(whitened H)", "||H - I||_F"]
        table = QTableWidget(len(history), len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setEditTriggers(QTableWidget.NoEditTriggers)

        for row, record in enumerate(history):
            delta_c = (diagnostics[row].covariance_delta
                       if row < len(diagnostics) else float("nan"))
            cells = [str(record.iteration)]
            cells += [f"{v:.5g}" for v in record.parameters_after]
            cells += [f"{record.delta_q:.3e}",
                      "-" if not np.isfinite(delta_c) else f"{delta_c:.3e}",
                      f"{record.condition_number:.3e}",
                      f"{record.identity_deviation:.3e}"]
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                table.setItem(row, column, item)
        layout.addWidget(table)

        note = QLabel(
            "deltaQ is the parameter criterion (eq. 18); deltaC is the "
            "covariance criterion (eq. 19). Both must pass. ||H - I||_F stays "
            "at round-off because Pierron's speciality condition pins H to "
            "the identity — which is why the rank test in section 1 cannot "
            "detect an unidentifiable coefficient.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #52514e;")
        layout.addWidget(note)
        return page

    # ------------------------------------------------------------------
    #  Noise model
    # ------------------------------------------------------------------
    def _build_noise_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setFont(QFont("Consolas" if _has_font("Consolas") else "Courier New", 9))

        lines = []
        goodness = self._goodness
        if goodness is not None:
            lines += [
                "GOODNESS OF FIT",
                f"  chi2 = {goodness['chi2']:.1f}   dof = {goodness['dof']}   "
                f"chi2/dof = {goodness['reduced']:.1f}",
                f"  the noise model explains "
                f"{100 / goodness['reduced']:.2f} % of the residual variance",
                f"  scale eq. (20) by at least {goodness['birge']:.1f} before "
                f"quoting an interval",
                "",
            ]

        for report in getattr(self._result, "experiments", []):
            noise = report.noise
            correlation = noise.correlation
            lines.append(f"EXPERIMENT {report.experiment_id}  ({report.label})")
            lines.append(f"  {report.n_steps} load steps, {report.n_points} "
                         f"points, dA = {report.area_per_point:.6f} mm^2")
            lines.append(f"  Gamma is {noise.gamma.shape[0]}x"
                         f"{noise.gamma.shape[0]} from a degree-"
                         f"{noise.basis_degree} spatial basis")

            primary = int(np.argmax(noise.residual_scale[0]))
            temporal = noise.temporal_correlation(primary)
            upper = temporal[np.triu_indices(noise.n_steps, k=1)]
            if upper.size:
                lines.append(
                    f"  mean cross-step correlation = {upper.mean():.3f}   "
                    f"(0.0 would mean the load steps are independent)")
            lines.append(
                f"  spatial rho at the nearest neighbour = "
                f"{correlation.nearest_rho:.3f}   kappa = "
                f"{correlation.kappa:.1f}  [measured, NOT applied]")
            diagnostics = report.covariance_diagnostics or {}
            if diagnostics:
                lines.append(
                    f"  C: positive definite = "
                    f"{diagnostics.get('positive_definite')}, cond = "
                    f"{diagnostics.get('condition_number', float('nan')):.3e}")
            lines.append("  sigma per load step (x1e3):")
            for step in range(noise.n_steps):
                row = "  ".join(f"{1e3 * v:7.4f}"
                                for v in noise.residual_scale[step])
                lines.append(f"    step {step:>2}   {row}")
            lines.append("")

        lines += [
            "READING THIS",
            "  Gamma is estimated as whatever a smooth spatial basis cannot",
            "  represent, so it is EFFECTIVE error, not a pure DIC noise",
            "  measurement.  kappa is measured and reported but never folded",
            "  into C: eq. (3) is used verbatim, and the resulting misfit is",
            "  reported through chi2/dof rather than absorbed by inflating",
            "  the covariance.",
        ]
        text.setPlainText("\n".join(lines))
        layout.addWidget(text)
        return page

    # ------------------------------------------------------------------
    #  Buttons
    # ------------------------------------------------------------------
    def _build_buttons(self) -> QHBoxLayout:
        row = QHBoxLayout()

        virtual = QPushButton("Virtual fields (3D)…")
        virtual.setToolTip("Open the optimal virtual fields for the converged "
                           "solution.")
        virtual.clicked.connect(self._open_virtual_fields)
        virtual.setEnabled(
            self._mesh_pkg is not None
            and bool(getattr(self._result, "optimal_nodal_history", [])))
        row.addWidget(virtual)

        identifiability = QPushButton("Identifiability…")
        identifiability.clicked.connect(self._open_identifiability)
        identifiability.setEnabled(
            bool(getattr(self._result, "diagnostic_cases", [])))
        row.addWidget(identifiability)

        row.addStretch(1)
        close = QPushButton("Close")
        close.setDefault(True)
        close.clicked.connect(self.close)
        row.addWidget(close)
        return row

    def _open_virtual_fields(self) -> None:
        from .virtual_field_viewer import VirtualFieldViewerWindow
        #  Held on self so Qt does not garbage-collect the window on return.
        self._virtual_window = VirtualFieldViewerWindow(
            self._mesh_pkg, self._result)
        self._virtual_window.show()
        self._virtual_window.raise_()

    def _open_identifiability(self) -> None:
        #  The window takes COMPUTED diagnostics, not the identification
        #  result -- the module-level launcher does that step and then blocks,
        #  which is not what we want from a non-modal button.
        from vfm_identifiability import compute_identifiability_diagnostics

        from .identifiability_viewer import IdentifiabilityDiagnosisWindow

        try:
            diagnostics = compute_identifiability_diagnostics(
                self._result.diagnostic_cases,
                self._result.constitutive_model)
        except (ValueError, AttributeError, np.linalg.LinAlgError) as error:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Identifiability unavailable",
                                str(error))
            return
        self._identifiability_window = IdentifiabilityDiagnosisWindow(
            diagnostics)
        self._identifiability_window.show()
        self._identifiability_window.raise_()


def _colour(hex_value: str):
    from PySide6.QtGui import QColor
    return QColor(hex_value)


def _has_font(name: str) -> bool:
    from PySide6.QtGui import QFontDatabase
    try:
        return name in QFontDatabase.families()
    except (AttributeError, TypeError):
        return False


def launch_results_window(identification_result, mesh_pkg=None,
                          project_name: str = "",
                          show_virtual_fields: bool = False) -> None:
    """Show the results window, creating a QApplication if needed (blocks).

    show_virtual_fields opens the 3D viewer immediately alongside, for the
    "jump straight in" path.
    """
    owns_app = QApplication.instance() is None
    app = QApplication.instance() or QApplication([])

    window = ResultsWindow(identification_result, mesh_pkg=mesh_pkg,
                           project_name=project_name)
    window.show()
    window.raise_()
    if show_virtual_fields and mesh_pkg is not None:
        window._open_virtual_fields()

    if owns_app:
        app.exec()
    else:
        from PySide6.QtCore import QEventLoop, QEvent, QObject

        loop = QEventLoop()

        class _CloseWatcher(QObject):
            def eventFilter(self, obj, event):
                if obj is window and event.type() == QEvent.Type.Close:
                    loop.quit()
                return False

        watcher = _CloseWatcher()
        window.installEventFilter(watcher)
        loop.exec()
