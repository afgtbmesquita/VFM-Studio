# =============================================================================
#  bc_gui / solver_settings_panel.py  (plate)
# -----------------------------------------------------------------------------
#  Right-dock "Solver settings" tab: sample-area override, identification
#  loop limits, and a toggle to visualise how many measurement points fall
#  inside each virtual-mesh element (same mapping as the VFM assembly).
# =============================================================================

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class SolverSettingsPanel(QWidget):
    """Identification / integration options surfaced next to the mesh editor.

    Signals
    -------
    measurement_counts_toggled : Signal[bool]
        Emitted with ``True`` when the user wants per-element counts drawn
        on the mesh, ``False`` when the overlay should be cleared.
    """

    measurement_counts_toggled = Signal(bool)
    material_orientation_changed = Signal(float)

    def __init__(self,
                 max_iterations: int,
                 convergence_tolerance: float,
                 *,
                 data_file_available: bool = True,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._user_sample_area_override = False
        self._area_programmatic_update = False

        root = QVBoxLayout(self)

        form = QFormLayout()

        self._sample_area_spin = QDoubleSpinBox()
        self._sample_area_spin.setRange(1.0e-18, 1.0e18)
        self._sample_area_spin.setDecimals(8)
        self._sample_area_spin.setSingleStep(1.0)
        self._sample_area_spin.setValue(0.0)
        self._sample_area_spin.setEnabled(data_file_available)
        self._sample_area_spin.setToolTip(
            "<p>This field represents the total area of the sample corresponding "
            "to the measurement points. It is used to calculate "
            r"<i>ΔS</i> = (area of sample) / <i>N</i> in the measurement-point "
            "integration, where <i>N</i> is the number of measurement points "
            "(equal weight per point).</p>"
        )
        self._sample_area_spin.setToolTipDuration(60000)
        self._sample_area_spin.valueChanged.connect(self._on_sample_area_changed)
        form.addRow("Area of sample", self._sample_area_spin)

        self._orientation_spin = QDoubleSpinBox()
        self._orientation_spin.setRange(-180.0, 180.0)
        self._orientation_spin.setDecimals(3)
        self._orientation_spin.setSingleStep(5.0)
        self._orientation_spin.setSuffix(" deg")
        self._orientation_spin.setValue(0.0)
        self._orientation_spin.setToolTip(
            "<p>Angle from this experiment's measurement frame (+<i>x</i>) to "
            "the material 1-axis, counter-clockwise positive. The arrows drawn "
            "on the mesh show it.</p>"
            "<p>The measured fields, the virtual mesh, the boundary conditions "
            "and the applied load all stay in the measurement frame; it is the "
            "<b>D matrix</b> that is rotated by this angle. Do <b>not</b> also "
            "rotate the fields — that would rotate twice.</p>"
            "<p>0 means this coupon was measured in the material frame.</p>"
        )
        self._orientation_spin.setToolTipDuration(60000)
        self._orientation_spin.valueChanged.connect(
            self._on_orientation_changed)
        form.addRow("Material frame", self._orientation_spin)

        self._max_iter_spin = QSpinBox()
        self._max_iter_spin.setRange(1, 100_000)
        self._max_iter_spin.setValue(int(max_iterations))
        form.addRow("Max iterations", self._max_iter_spin)

        self._coef_tol_spin = QDoubleSpinBox()
        self._coef_tol_spin.setRange(1.0e-16, 1.0)
        self._coef_tol_spin.setDecimals(10)
        self._coef_tol_spin.setSingleStep(1.0e-5)
        self._coef_tol_spin.setValue(float(convergence_tolerance))
        form.addRow("Coefficients tolerance", self._coef_tol_spin)

        root.addLayout(form)

        self._counts_btn = QPushButton("Show Measurement Points per Element")
        self._counts_btn.setCheckable(True)
        self._counts_btn.setEnabled(data_file_available)
        self._counts_btn.setToolTip(
            "Annotate each element with how many measurement points map "
            "into it (same containment rule as the identification assembly)."
        )
        self._counts_btn.toggled.connect(self._on_counts_toggled)
        row = QHBoxLayout()
        row.addWidget(self._counts_btn)
        row.addStretch(1)
        root.addLayout(row)

        if not data_file_available:
            hint = QLabel(
                "No measurement file was passed to the editor — sample area "
                "and per-element counts are disabled."
            )
            hint.setWordWrap(True)
            root.addWidget(hint)

        root.addStretch(1)

    # ------------------------------------------------------------------
    def set_automatic_sample_area(self, area: float) -> None:
        """Push a bbox-derived estimate into the spin box unless the user edited it."""
        if self._user_sample_area_override:
            return
        self._area_programmatic_update = True
        self._sample_area_spin.setValue(float(area))
        self._area_programmatic_update = False

    def apply_solver_settings(self,
                              *,
                              sample_area_total: Optional[float] = None,
                              max_iterations: Optional[int] = None,
                              convergence_tolerance: Optional[float] = None) -> None:
        """Fill the widgets from settings persisted in a BC file."""
        if sample_area_total is not None:
            self._area_programmatic_update = True
            self._sample_area_spin.setValue(float(sample_area_total))
            self._area_programmatic_update = False
            self._user_sample_area_override = True
        if max_iterations is not None:
            self._max_iter_spin.setValue(int(max_iterations))
        if convergence_tolerance is not None:
            self._coef_tol_spin.setValue(float(convergence_tolerance))

    def set_sample_area_total(self, sample_area_total: Optional[float]) -> None:
        """Restore the per-test sample area state."""
        self._area_programmatic_update = True
        self._sample_area_spin.setValue(
            0.0 if sample_area_total is None else float(sample_area_total)
        )
        self._area_programmatic_update = False
        self._user_sample_area_override = sample_area_total is not None

    def set_data_file_available(self, available: bool) -> None:
        """Enable/disable widgets tied to the selected test's data file."""
        available = bool(available)
        self._sample_area_spin.setEnabled(available)
        self._counts_btn.setEnabled(available)
        if not available:
            self.clear_measurement_counts()

    def sample_area_total(self) -> Optional[float]:
        """Return the spin-box area, or ``None`` if unset / invalid (treat as auto)."""
        v = float(self._sample_area_spin.value())
        if v <= 0.0:
            return None
        return v

    def material_orientation_deg(self) -> float:
        """Angle from the measurement frame +x to the material 1-axis."""
        return float(self._orientation_spin.value())

    def set_material_orientation_deg(self, angle_deg: Optional[float]) -> None:
        """Restore the per-test orientation without re-emitting the signal."""
        self._orientation_spin.blockSignals(True)
        self._orientation_spin.setValue(
            0.0 if angle_deg is None else float(angle_deg))
        self._orientation_spin.blockSignals(False)

    def max_iterations(self) -> int:
        return int(self._max_iter_spin.value())

    def convergence_tolerance(self) -> float:
        return float(self._coef_tol_spin.value())

    def measurement_counts_enabled(self) -> bool:
        return bool(self._counts_btn.isEnabled() and self._counts_btn.isChecked())

    def clear_measurement_counts(self) -> None:
        """Uncheck the counts button without emitting ``measurement_counts_toggled``."""
        self._counts_btn.blockSignals(True)
        self._counts_btn.setChecked(False)
        self._counts_btn.blockSignals(False)

    # ------------------------------------------------------------------
    def _on_sample_area_changed(self, _value: float) -> None:
        if self._area_programmatic_update:
            return
        self._user_sample_area_override = True

    def _on_counts_toggled(self, checked: bool) -> None:
        self.measurement_counts_toggled.emit(bool(checked))

    def _on_orientation_changed(self, value: float) -> None:
        self.material_orientation_changed.emit(float(value))
