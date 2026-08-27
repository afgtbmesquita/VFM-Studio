# =============================================================================
#  bc_gui / mesh_editor.py  (plate)
# -----------------------------------------------------------------------------
#  Compact ``QGroupBox`` panel that exposes the virtual-mesh settings to
#  the user (number of elements + rectangular bounds) and emits a
#  ``mesh_settings_changed`` Qt signal when the user clicks Apply.
#
#  The panel is owned by ``BcEditorWindow`` and lives in the right dock,
#  just above the "Use saved file & run solver" / "Cancel" finish row, so
#  the user can iterate on the mesh while the BCs are also visible.
#
#  Why a separate dataclass on top of ``MeshSettings``?
#  ----------------------------------------------------
#  ``vfm_user_inputs.MeshSettings`` already carries everything we need
#  (``number_of_elements_x``, ``number_of_elements_y`` and a ``Bounds``
#  sub-object).  This module simply provides the Qt widgets that read and
#  write that data structure; no new state schema is introduced.
# =============================================================================

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

# We import these at runtime (and not via ``TYPE_CHECKING``) because Qt
# signals carry the type as an actual runtime parameter.  Both lookups
# are cheap.
from vfm_user_inputs import Bounds, MeshSettings


# Practical clamps on the spin boxes.  The lower bound is the smallest
# value that still produces a non-degenerate mesh; the upper bound is
# generous enough that any realistic specimen fits comfortably and that
# accidental huge inputs cannot crash the GUI.
_MIN_ELEMENTS = 1
_MAX_ELEMENTS = 1000
_MIN_COORD = -1.0e6
_MAX_COORD = 1.0e6
_COORD_DECIMALS = 4

# Seed used when the editor opens with neither a mesh on the canvas nor
# MESH_SETTINGS from a BC file (a brand-new experiment).  Small enough to
# stay legible on the canvas; the user is expected to edit it before Apply.
DEFAULT_ELEMENTS_X = 4
DEFAULT_ELEMENTS_Y = 4


class ManualNodesDialog(QDialog):
    """Dialog for typing explicit structured-grid node coordinates."""

    preview_nodes_changed = Signal(object)        # list[(x, y)]
    preview_nodes_cleared = Signal()
    manual_mesh_applied = Signal(object)          # MeshSettings instance

    def __init__(self,
                 seed_settings: MeshSettings,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Manual Nodes")
        self.resize(420, 520)
        self._updating = False

        layout = QVBoxLayout(self)

        self._table = QTableWidget(0, 2, self)
        self._table.setHorizontalHeaderLabels(["x", "y"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._table.itemChanged.connect(self._refresh_preview)
        layout.addWidget(self._table, 1)

        edit_row = QHBoxLayout()
        self._add_btn = QPushButton("Add row")
        self._add_btn.clicked.connect(self._add_empty_row)
        edit_row.addWidget(self._add_btn)

        self._remove_btn = QPushButton("Remove selected")
        self._remove_btn.clicked.connect(self._remove_selected_rows)
        edit_row.addWidget(self._remove_btn)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self._clear_rows)
        edit_row.addWidget(self._clear_btn)
        layout.addLayout(edit_row)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        action_row = QHBoxLayout()
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.clicked.connect(self._on_apply)
        action_row.addWidget(self._apply_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self.reject)
        action_row.addWidget(self._cancel_btn)
        layout.addLayout(action_row)

        self._populate_from_settings(seed_settings)
        self._refresh_preview()

    def reject(self) -> None:  # noqa: D401
        self.preview_nodes_cleared.emit()
        super().reject()

    def accept(self) -> None:  # noqa: D401
        self.preview_nodes_cleared.emit()
        super().accept()

    def _populate_from_settings(self, settings: MeshSettings) -> None:
        x_nodes = getattr(settings, "x_nodes", None)
        y_nodes = getattr(settings, "y_nodes", None)
        if x_nodes is not None and y_nodes is not None:
            xs = [float(x) for x in x_nodes]
            ys = [float(y) for y in y_nodes]
        else:
            xs = _uniform_axis(settings.bounds.x_min, settings.bounds.x_max,
                               int(settings.number_of_elements_x))
            ys = _uniform_axis(settings.bounds.y_min, settings.bounds.y_max,
                               int(settings.number_of_elements_y))

        rows = [(x, y) for y in ys for x in xs]
        self._updating = True
        self._table.setRowCount(0)
        for x, y in rows:
            self._append_row(x, y)
        self._updating = False

    def _append_row(self,
                    x: Optional[float] = None,
                    y: Optional[float] = None) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._table.setItem(row, 0, QTableWidgetItem("" if x is None else f"{x:g}"))
        self._table.setItem(row, 1, QTableWidgetItem("" if y is None else f"{y:g}"))

    def _add_empty_row(self) -> None:
        self._append_row()
        self._refresh_preview()

    def _remove_selected_rows(self) -> None:
        rows = sorted(
            {index.row() for index in self._table.selectedIndexes()},
            reverse=True,
        )
        if not rows:
            return
        self._updating = True
        for row in rows:
            self._table.removeRow(row)
        self._updating = False
        self._refresh_preview()

    def _clear_rows(self) -> None:
        self._updating = True
        self._table.setRowCount(0)
        self._updating = False
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        if self._updating:
            return
        points = self._read_preview_points()
        self.preview_nodes_changed.emit(points)
        try:
            settings, _points = self._validated_settings()
        except ValueError as exc:
            if points:
                self._status_label.setText(
                    f"Previewing {len(points)} complete node(s). {exc}"
                )
            else:
                self._status_label.setText(str(exc))
            self._apply_btn.setEnabled(False)
            return
        self._status_label.setText(
            f"Valid structured grid: {settings.number_of_elements_x} x "
            f"{settings.number_of_elements_y} elements, {len(points)} nodes."
        )
        self._apply_btn.setEnabled(True)

    def _read_preview_points(self) -> List[Tuple[float, float]]:
        points: List[Tuple[float, float]] = []
        for row in range(self._table.rowCount()):
            x = _parse_table_float(self._table.item(row, 0))
            y = _parse_table_float(self._table.item(row, 1))
            if x is not None and y is not None:
                points.append((x, y))
        return points

    def _read_strict_points(self) -> List[Tuple[float, float]]:
        points: List[Tuple[float, float]] = []
        for row in range(self._table.rowCount()):
            x_text = _table_text(self._table.item(row, 0))
            y_text = _table_text(self._table.item(row, 1))
            if not x_text and not y_text:
                continue
            if not x_text or not y_text:
                raise ValueError(
                    f"Row {row + 1} is incomplete; enter both x and y."
                )
            try:
                x = float(x_text)
                y = float(y_text)
            except ValueError as exc:
                raise ValueError(
                    f"Row {row + 1} contains a non-numeric coordinate."
                ) from exc
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError(
                    f"Row {row + 1} must contain finite coordinates."
                )
            points.append((x, y))
        return points

    def _validated_settings(self) -> Tuple[MeshSettings, List[Tuple[float, float]]]:
        points = self._read_strict_points()
        from vfm_mesh import mesh_settings_from_node_points
        settings = mesh_settings_from_node_points(points)
        return settings, points

    def _on_apply(self) -> None:
        try:
            settings, _points = self._validated_settings()
        except ValueError as exc:
            self._status_label.setText(str(exc))
            return
        self.manual_mesh_applied.emit(settings)
        self.accept()


class MeshEditorPanel(QWidget):
    """Live editor for the rectangular virtual mesh.

    Signals
    -------
    mesh_settings_changed : Signal[MeshSettings]
        Emitted on Apply with a fully-built ``MeshSettings``.  The parent
        window is responsible for rebuilding the actual mesh package and
        for any side effects (e.g. dropping BCs that referenced the old
        node IDs).
    """

    mesh_settings_changed = Signal(object)         # MeshSettings instance
    manual_nodes_preview_changed = Signal(object)  # list[(x, y)]
    manual_nodes_preview_cleared = Signal()

    def __init__(self,
                 initial_settings: MeshSettings,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._initial_settings = _clone_settings(initial_settings)

        layout = QVBoxLayout(self)

        # ---- Element-count + bounds form -----------------------------
        form = QFormLayout()

        self._nx_spin = QSpinBox()
        self._nx_spin.setRange(_MIN_ELEMENTS, _MAX_ELEMENTS)
        self._nx_spin.setToolTip(
            "Number of elements along the x-axis.  Each element is a "
            "rectangular Q1 (membrane) + BFS (bending) composite quad."
        )
        form.addRow("Elements in x", self._nx_spin)

        self._ny_spin = QSpinBox()
        self._ny_spin.setRange(_MIN_ELEMENTS, _MAX_ELEMENTS)
        self._ny_spin.setToolTip(
            "Number of elements along the y-axis."
        )
        form.addRow("Elements in y", self._ny_spin)

        self._x_min_spin = self._make_coord_spin("Lower x bound (x_min).")
        form.addRow("x min", self._x_min_spin)

        self._x_max_spin = self._make_coord_spin("Upper x bound (x_max).")
        form.addRow("x max", self._x_max_spin)

        self._y_min_spin = self._make_coord_spin("Lower y bound (y_min).")
        form.addRow("y min", self._y_min_spin)

        self._y_max_spin = self._make_coord_spin("Upper y bound (y_max).")
        form.addRow("y max", self._y_max_spin)

        layout.addLayout(form)

        # ---- Apply / Reset buttons -----------------------------------
        button_row = QHBoxLayout()
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.setToolTip(
            "Rebuild the virtual mesh with the values above.  Any "
            "existing BCs that referenced the old mesh are dropped "
            "(you will be asked to confirm first)."
        )
        self._apply_btn.clicked.connect(self._on_apply)
        button_row.addWidget(self._apply_btn)

        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setToolTip(
            "Restore the mesh settings the GUI was launched with.  This "
            "only resets the form -- click Apply to actually rebuild "
            "the mesh."
        )
        self._reset_btn.clicked.connect(self._on_reset)
        button_row.addWidget(self._reset_btn)
        layout.addLayout(button_row)

        manual_row = QHBoxLayout()
        self._manual_nodes_btn = QPushButton("Manual Nodes")
        self._manual_nodes_btn.setToolTip(
            "Type explicit x/y node coordinates for a nonuniform structured "
            "quadrilateral mesh."
        )
        self._manual_nodes_btn.clicked.connect(self._on_manual_nodes)
        manual_row.addWidget(self._manual_nodes_btn)
        layout.addLayout(manual_row)

        # ---- Status line ---------------------------------------------
        # Shows the resulting element / DOF count after each Apply so
        # the user can sanity-check that the new mesh is what they
        # asked for.
        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        # Populate the form with the initial values.
        self._populate_from_settings(self._initial_settings)

    # ------------------------------------------------------------------
    #  Public helpers
    # ------------------------------------------------------------------
    def current_settings(self) -> MeshSettings:
        """Read the widget state and return a fresh ``MeshSettings``."""
        return MeshSettings(
            number_of_elements_x=int(self._nx_spin.value()),
            number_of_elements_y=int(self._ny_spin.value()),
            bounds=Bounds(
                x_min=float(self._x_min_spin.value()),
                x_max=float(self._x_max_spin.value()),
                y_min=float(self._y_min_spin.value()),
                y_max=float(self._y_max_spin.value()),
            ),
        )

    def set_status(self, text: str) -> None:
        """Set the bottom-of-panel status line (called by the parent)."""
        self._status_label.setText(text)

    def sync_with_settings(self, settings: MeshSettings) -> None:
        """Force the widgets to mirror an externally-applied mesh."""
        self._populate_from_settings(settings)

    # ------------------------------------------------------------------
    #  Internals
    # ------------------------------------------------------------------
    def _make_coord_spin(self, tooltip: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(_MIN_COORD, _MAX_COORD)
        spin.setDecimals(_COORD_DECIMALS)
        spin.setSingleStep(0.1)
        spin.setToolTip(tooltip)
        return spin

    def _populate_from_settings(self, settings: MeshSettings) -> None:
        self._nx_spin.setValue(int(settings.number_of_elements_x))
        self._ny_spin.setValue(int(settings.number_of_elements_y))
        self._x_min_spin.setValue(float(settings.bounds.x_min))
        self._x_max_spin.setValue(float(settings.bounds.x_max))
        self._y_min_spin.setValue(float(settings.bounds.y_min))
        self._y_max_spin.setValue(float(settings.bounds.y_max))

    def _on_apply(self) -> None:
        """Validate the form, then emit ``mesh_settings_changed``.

        The actual mesh rebuild happens in the parent window's slot.
        Validation here mirrors what ``vfm_mesh.build_mesh`` enforces so
        the user gets immediate feedback instead of a Python exception.
        """
        settings = self.current_settings()
        x_min = settings.bounds.x_min
        x_max = settings.bounds.x_max
        y_min = settings.bounds.y_min
        y_max = settings.bounds.y_max
        if x_max <= x_min or y_max <= y_min:
            self._status_label.setText(
                "Invalid bounds: need x_max > x_min and y_max > y_min."
            )
            return
        if (settings.number_of_elements_x < 1
                or settings.number_of_elements_y < 1):
            self._status_label.setText(
                "Element counts must be >= 1 in each direction."
            )
            return

        self.mesh_settings_changed.emit(settings)

    def _on_reset(self) -> None:
        """Restore the initial values (the ones the GUI was launched with)."""
        self._populate_from_settings(self._initial_settings)
        self._status_label.setText("Reverted to the initial mesh settings.")

    def _on_manual_nodes(self) -> None:
        dialog = ManualNodesDialog(self.current_settings(), self)
        dialog.preview_nodes_changed.connect(
            self.manual_nodes_preview_changed.emit
        )
        dialog.preview_nodes_cleared.connect(
            self.manual_nodes_preview_cleared.emit
        )
        dialog.manual_mesh_applied.connect(self._on_manual_mesh_applied)
        dialog.exec()
        self.manual_nodes_preview_cleared.emit()

    def _on_manual_mesh_applied(self, settings: MeshSettings) -> None:
        self._populate_from_settings(settings)
        self.mesh_settings_changed.emit(settings)


def _clone_settings(settings: MeshSettings) -> MeshSettings:
    """Return a shallow copy that does not share the ``Bounds`` instance."""
    return MeshSettings(
        number_of_elements_x=int(settings.number_of_elements_x),
        number_of_elements_y=int(settings.number_of_elements_y),
        bounds=Bounds(
            x_min=float(settings.bounds.x_min),
            x_max=float(settings.bounds.x_max),
            y_min=float(settings.bounds.y_min),
            y_max=float(settings.bounds.y_max),
        ),
        x_nodes=([float(x) for x in getattr(settings, "x_nodes", None)]
                 if getattr(settings, "x_nodes", None) is not None else None),
        y_nodes=([float(y) for y in getattr(settings, "y_nodes", None)]
                 if getattr(settings, "y_nodes", None) is not None else None),
    )


def mesh_settings_from_package(mesh_pkg) -> MeshSettings:
    """Build a ``MeshSettings`` from an existing ``MeshPackage``.

    Used by the parent window to seed the editor panel with the values
    that match the mesh actually being displayed.
    """
    x_nodes = getattr(mesh_pkg, "x_nodes", None)
    y_nodes = getattr(mesh_pkg, "y_nodes", None)
    return MeshSettings(
        number_of_elements_x=int(mesh_pkg.nx),
        number_of_elements_y=int(mesh_pkg.ny),
        bounds=Bounds(
            x_min=float(mesh_pkg.x_min),
            x_max=float(mesh_pkg.x_max),
            y_min=float(mesh_pkg.y_min),
            y_max=float(mesh_pkg.y_max),
        ),
        x_nodes=([float(x) for x in x_nodes]
                 if x_nodes is not None else None),
        y_nodes=([float(y) for y in y_nodes]
                 if y_nodes is not None else None),
    )


def default_mesh_settings() -> MeshSettings:
    """Return a neutral placeholder seed for the mesh-editor panel.

    Used when the editor is opened with no mesh and no saved
    ``MESH_SETTINGS``: the spin boxes need *some* valid values before the
    measurement cloud has been read.  Callers should replace it with
    :func:`mesh_settings_from_points` as soon as the MAT file is loaded.
    """
    return MeshSettings(
        number_of_elements_x=DEFAULT_ELEMENTS_X,
        number_of_elements_y=DEFAULT_ELEMENTS_Y,
        bounds=Bounds(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0),
    )


def mesh_settings_from_points(points) -> Optional[MeshSettings]:
    """Fit default ``MeshSettings`` to a measurement cloud bounding box.

    ``points`` is an ``(N, 2)`` array of measurement coordinates.  Returns
    ``None`` when the cloud is missing, malformed, or degenerate in either
    direction, so the caller can keep whatever seed it already has instead
    of building a zero-width mesh.
    """
    if points is None:
        return None
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 2:
        return None
    finite = pts[np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])]
    if finite.shape[0] == 0:
        return None
    x_min, x_max = float(finite[:, 0].min()), float(finite[:, 0].max())
    y_min, y_max = float(finite[:, 1].min()), float(finite[:, 1].max())
    if not (x_max > x_min and y_max > y_min):
        return None
    return MeshSettings(
        number_of_elements_x=DEFAULT_ELEMENTS_X,
        number_of_elements_y=DEFAULT_ELEMENTS_Y,
        bounds=Bounds(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max),
    )


def _uniform_axis(start: float, stop: float, n_elements: int) -> List[float]:
    n = max(1, int(n_elements))
    step = (float(stop) - float(start)) / n
    return [float(start) + i * step for i in range(n + 1)]


def _table_text(item: Optional[QTableWidgetItem]) -> str:
    if item is None:
        return ""
    return item.text().strip()


def _parse_table_float(item: Optional[QTableWidgetItem]) -> Optional[float]:
    text = _table_text(item)
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value
