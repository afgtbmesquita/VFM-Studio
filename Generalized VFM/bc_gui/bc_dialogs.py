# =============================================================================
#  bc_gui / bc_dialogs.py  (plate)
# -----------------------------------------------------------------------------
#  Modal Qt dialogs for adding and editing plate boundary-condition entries.
#
#  A single ``AddBcDialog`` class handles all three categories
#  (Prescribed / Linking / External); fields that do not apply to the
#  active category are hidden so the dialog stays compact.
#
#  ``detect_named_edge`` is a small helper used by the dialog (and by the
#  main window) to offer the Abaqus-style shortcut of replacing a long
#  ``"nodes": [...]`` selection by the canonical ``"edge": "left"`` form
#  when the user picked every node on a named edge.
#
#  The only meaningful difference from the Q_Matrix bc_dialogs is the DOF
#  combo box: the plate solver supports six DOF labels (U, V, W, W_X, W_Y,
#  W_XY) instead of just two.
# =============================================================================

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
)

from .bc_model import (
    CATEGORY_EXTERNAL,
    CATEGORY_LINKING,
    CATEGORY_PRESCRIBED,
    ExternalEntry,
    LinkingEntry,
    PrescribedEntry,
    SUPPORTED_DOFS,
    SUPPORTED_EDGES,
    SUPPORTED_LOAD_TYPES,
    _BaseEntry,
    allowed_dofs_for,
)


# =============================================================================
#  Named-edge detection
# -----------------------------------------------------------------------------
#  Mirrors the tolerance logic in ``plate/vfm_mesh._select_edge_nodes`` so
#  the GUI agrees with the solver on what "all nodes on the left edge"
#  means.
# =============================================================================

def _edge_node_indices(mesh_pkg, edge_name: str) -> np.ndarray:
    coords = np.asarray(mesh_pkg.node_coords)
    tolerance = max(
        1.0e-12,
        1.0e-10 * max(
            abs(float(mesh_pkg.x_min)), abs(float(mesh_pkg.x_max)),
            abs(float(mesh_pkg.y_min)), abs(float(mesh_pkg.y_max)),
            1.0,
        ),
    )
    edge = str(edge_name).lower()
    if edge == "left":
        mask = np.abs(coords[:, 0] - float(mesh_pkg.x_min)) <= tolerance
    elif edge == "right":
        mask = np.abs(coords[:, 0] - float(mesh_pkg.x_max)) <= tolerance
    elif edge == "bottom":
        mask = np.abs(coords[:, 1] - float(mesh_pkg.y_min)) <= tolerance
    elif edge == "top":
        mask = np.abs(coords[:, 1] - float(mesh_pkg.y_max)) <= tolerance
    else:
        raise ValueError(f"Unknown edge {edge_name!r}.")
    return np.where(mask)[0]


def _format_float_for_edit(value: float) -> str:
    """Render a float as a clean text-edit string for the dialog."""
    f = float(value)
    if f == 0.0:
        return "0.0"
    return repr(f)


def detect_named_edge(mesh_pkg, node_ids: Sequence[int]) -> Optional[str]:
    """Return the edge label whose node set equals ``node_ids`` exactly.

    Returns ``None`` if the selection does not match any of the four edges
    or if it is empty.
    """
    if not node_ids:
        return None
    picked = set(int(n) for n in node_ids)
    for edge_name in SUPPORTED_EDGES:
        edge_set = set(_edge_node_indices(mesh_pkg, edge_name).tolist())
        if edge_set == picked:
            return edge_name
    return None


# =============================================================================
#  AddBcDialog
# =============================================================================

class AddBcDialog(QDialog):
    """Form to create / edit one plate boundary-condition entry."""

    def __init__(self,
                 category: str,
                 selected_node_ids: Sequence[int],
                 mesh_pkg,
                 existing_entry: Optional[_BaseEntry] = None,
                 analysis_type: Optional[str] = None,
                 external_values_managed_by_parent: bool = False,
                 parent=None) -> None:
        super().__init__(parent)
        self._category = category
        self._mesh = mesh_pkg
        self._analysis_type = analysis_type
        self._external_values_managed_by_parent = bool(
            external_values_managed_by_parent)
        self._selected_node_ids: List[int] = sorted(int(n)
                                                     for n in selected_node_ids)
        self._existing_entry = existing_entry

        # Pre-compute the auto-detected edge (None if not a perfect match).
        if existing_entry is None:
            self._auto_edge = detect_named_edge(mesh_pkg,
                                                self._selected_node_ids)
        else:
            self._auto_edge = (existing_entry.edge
                               if existing_entry.selection_mode == "edge"
                               else detect_named_edge(
                                   mesh_pkg,
                                   existing_entry.nodes or []))

        self.setWindowTitle(self._window_title())
        self.setModal(True)

        self._build_ui()
        self._populate_from_existing()
        self._update_edge_shortcut_visibility()

    # ------------------------------------------------------------------
    #  Public result
    # ------------------------------------------------------------------
    def result_entry(self) -> _BaseEntry:
        return self._build_entry()

    # ------------------------------------------------------------------
    #  Layout
    # ------------------------------------------------------------------
    def _window_title(self) -> str:
        title_map = {
            CATEGORY_PRESCRIBED: "prescribed boundary condition",
            CATEGORY_LINKING: "linking constraint",
            CATEGORY_EXTERNAL: "external-work group",
        }
        verb = "Edit" if self._existing_entry is not None else "Add"
        return f"{verb} {title_map[self._category]}"

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ---------------- Form group -------------------------------------
        form_group = QGroupBox("Entry fields")
        form = QFormLayout(form_group)

        self._label_edit = QLineEdit(self._suggested_label())
        form.addRow("Label", self._label_edit)

        self._dof_combo = QComboBox()
        allowed = allowed_dofs_for(self._analysis_type)
        self._dof_combo.addItems(list(allowed))
        tip_lines = ["Plate DOF labels"]
        if self._analysis_type is not None:
            tip_lines.append(f"(filtered for analysis_type='{self._analysis_type}'):")
        else:
            tip_lines.append("(no active analysis type -- showing all):")
        descriptions = {
            "U":    "U     - in-plane horizontal displacement",
            "V":    "V     - in-plane vertical displacement",
            "W":    "W     - transverse (out-of-plane) deflection",
            "W_X":  "W_X   - rotation dw/dx (moment about y-axis)",
            "W_Y":  "W_Y   - rotation dw/dy (moment about x-axis)",
            "W_XY": "W_XY  - twist d2w/(dx dy)",
        }
        for dof in allowed:
            tip_lines.append("  " + descriptions[dof])
        self._dof_combo.setToolTip("\n".join(tip_lines))
        form.addRow("DOF", self._dof_combo)

        # Linking-only field
        if self._category == CATEGORY_LINKING:
            self._sort_combo_linking = QComboBox()
            self._sort_combo_linking.addItems(["(none)", "x", "y"])
            form.addRow("Sort direction", self._sort_combo_linking)
        else:
            self._sort_combo_linking = None  # type: ignore[assignment]

        # External-only fields
        if self._category == CATEGORY_EXTERNAL:
            self._load_combo = QComboBox()
            self._load_combo.addItems(list(SUPPORTED_LOAD_TYPES))
            form.addRow("Load type", self._load_combo)

            self._sort_combo_external = QComboBox()
            self._sort_combo_external.addItems(["(none)", "x", "y"])
            form.addRow("Sort direction", self._sort_combo_external)

            if self._external_values_managed_by_parent:
                # This editor defines experiment-level topology.  A scalar
                # here would be misleading because load is a step property.
                per_step_note = QLabel(
                    "Set one value for every load step in the experiment "
                    "table after closing this editor.")
                per_step_note.setWordWrap(True)
                per_step_note.setStyleSheet("color: #52514e;")
                per_step_note.setToolTip(
                    "The node selection, DOF and load type are shared by all "
                    "steps. Only the force or moment magnitude varies from "
                    "one step to the next.")
                form.addRow("Applied force / moment", per_step_note)
                self._external_value_edit = None  # type: ignore[assignment]
            else:
                # Standalone editor: one BC file represents one load case, so
                # retaining a single fallback value is appropriate.
                self._external_value_edit = QLineEdit("")
                self._external_value_edit.setPlaceholderText(
                    "blank -> use F from MAT file"
                )
                self._external_value_edit.setToolTip(
                    "Applied force or moment magnitude for this external-work "
                    "group, in the same units as the measurement file.  Leave "
                    "blank to use the 'F' field from the MAT file (legacy "
                    "behaviour).  For the new Loading_Data/FromAbaqustoVFM.m "
                    "workflow the MAT file does not contain F, so a numeric "
                    "value here is required."
                )
                form.addRow("Applied force / moment",
                            self._external_value_edit)
        else:
            self._load_combo = None              # type: ignore[assignment]
            self._sort_combo_external = None     # type: ignore[assignment]
            self._external_value_edit = None     # type: ignore[assignment]

        # Value field -- always read-only for prescribed / linking because
        # the solver only accepts homogeneous constraints (value = 0).
        if self._category in (CATEGORY_PRESCRIBED, CATEGORY_LINKING):
            self._value_edit = QLineEdit("0.0")
            self._value_edit.setReadOnly(True)
            self._value_edit.setToolTip(
                "The plate solver only supports homogeneous "
                "constraints (value = 0).  This field is fixed for now."
            )
            form.addRow("Value", self._value_edit)
        else:
            self._value_edit = None  # type: ignore[assignment]

        layout.addWidget(form_group)

        # ---------------- Selection group --------------------------------
        selection_group = QGroupBox("Selection")
        selection_layout = QVBoxLayout(selection_group)

        self._selection_label = QLabel(self._format_selection_text())
        self._selection_label.setWordWrap(True)
        selection_layout.addWidget(self._selection_label)

        self._edge_shortcut_check = QCheckBox(
            "Save as named edge selector instead of explicit node list"
        )
        self._edge_shortcut_check.setToolTip(
            "When the picked nodes exactly match one of the four mesh "
            "edges (left / right / bottom / top), tick this to store the "
            "entry as 'edge: <name>' in the generated .py file instead "
            "of a long 'nodes: [...]' array.  The file stays readable "
            "and the solver gets the same node set either way."
        )
        selection_layout.addWidget(self._edge_shortcut_check)

        layout.addWidget(selection_group)

        # ---------------- Buttons ----------------------------------------
        button_box = QDialogButtonBox(QDialogButtonBox.Ok
                                      | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    def _suggested_label(self) -> str:
        if self._existing_entry is not None:
            return self._existing_entry.label
        verb_map = {
            CATEGORY_PRESCRIBED: "Clamp",
            CATEGORY_LINKING: "Rigid-link",
            CATEGORY_EXTERNAL: "Measure",
        }
        verb = verb_map[self._category]
        if self._auto_edge is not None:
            return f"{verb} {self._auto_edge} edge"
        if self._selected_node_ids:
            return f"{verb} {len(self._selected_node_ids)} selected node(s)"
        return f"{verb} <select nodes first>"

    def _format_selection_text(self) -> str:
        if self._existing_entry is not None and self._auto_edge is not None \
                and self._existing_entry.selection_mode == "edge":
            return f"Edge selector: '{self._auto_edge}'"
        if not self._selected_node_ids and self._existing_entry is None:
            return ("No nodes selected.  Cancel this dialog, pick nodes on "
                    "the mesh, and try again.")
        node_ids = self._selected_node_ids
        if self._existing_entry is not None and not node_ids:
            node_ids = self._existing_entry.nodes or []
        preview = ", ".join(str(n) for n in node_ids[:24])
        if len(node_ids) > 24:
            preview += ", ..."
        line1 = f"{len(node_ids)} node(s) selected:"
        line2 = f"   [{preview}]"
        if self._auto_edge is not None:
            line3 = (f"   Selection matches the '{self._auto_edge}' edge "
                     "exactly -- you can tick the box below to save the "
                     "entry as an edge selector.")
            return "\n".join([line1, line2, line3])
        return "\n".join([line1, line2])

    def _populate_from_existing(self) -> None:
        entry = self._existing_entry
        if entry is None:
            if self._auto_edge is not None:
                self._edge_shortcut_check.setChecked(True)
            return

        self._label_edit.setText(entry.label)
        idx = self._dof_combo.findText(entry.dof)
        if idx >= 0:
            self._dof_combo.setCurrentIndex(idx)

        if self._category == CATEGORY_LINKING and isinstance(entry, LinkingEntry):
            self._set_sort_combo(self._sort_combo_linking,
                                 entry.sort_direction)
        if self._category == CATEGORY_EXTERNAL and isinstance(entry,
                                                              ExternalEntry):
            idx = self._load_combo.findText(entry.load_type)
            if idx >= 0:
                self._load_combo.setCurrentIndex(idx)
            self._set_sort_combo(self._sort_combo_external,
                                 entry.sort_direction)
            if self._external_value_edit is not None:
                if entry.value is None:
                    self._external_value_edit.setText("")
                else:
                    self._external_value_edit.setText(
                        _format_float_for_edit(entry.value)
                    )

        if entry.selection_mode == "edge" and entry.edge is not None:
            self._selected_node_ids = sorted(
                _edge_node_indices(self._mesh, entry.edge).tolist()
            )
            self._edge_shortcut_check.setChecked(True)
            self._selection_label.setText(self._format_selection_text())
        else:
            if entry.nodes is not None and not self._selected_node_ids:
                self._selected_node_ids = list(entry.nodes)
                self._selection_label.setText(self._format_selection_text())

    def _update_edge_shortcut_visibility(self) -> None:
        match = self._auto_edge is not None
        self._edge_shortcut_check.setEnabled(match)
        if not match:
            self._edge_shortcut_check.setChecked(False)
            self._edge_shortcut_check.setToolTip(
                "Available only when the selection matches one of the four "
                "named mesh edges exactly."
            )

    @staticmethod
    def _set_sort_combo(combo: QComboBox,
                        sort_direction: Optional[str]) -> None:
        if combo is None:
            return
        if not sort_direction:
            combo.setCurrentIndex(0)
            return
        idx = combo.findText(sort_direction)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    # ------------------------------------------------------------------
    #  Validation + build
    # ------------------------------------------------------------------
    def _on_accept(self) -> None:
        try:
            self._build_entry()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid boundary condition", str(exc))
            return
        self.accept()

    def _build_entry(self) -> _BaseEntry:
        label = self._label_edit.text().strip() or "<unlabeled>"
        dof = self._dof_combo.currentText()

        use_edge = (self._edge_shortcut_check.isChecked()
                    and self._auto_edge is not None)
        edge = self._auto_edge if use_edge else None
        nodes = None if use_edge else list(self._selected_node_ids)
        selection_mode = "edge" if use_edge else "nodes"

        if not use_edge and not nodes:
            raise ValueError(
                "Cannot save: there is no node selection and no matching "
                "named edge.  Pick nodes on the mesh first."
            )

        if self._category == CATEGORY_PRESCRIBED:
            return PrescribedEntry(
                label=label, dof=dof, edge=edge, nodes=nodes,
                selection_mode=selection_mode, value=0.0,
            )
        if self._category == CATEGORY_LINKING:
            sort_text = (self._sort_combo_linking.currentText()
                         if self._sort_combo_linking is not None else "(none)")
            sort_direction = None if sort_text == "(none)" else sort_text
            return LinkingEntry(
                label=label, dof=dof, edge=edge, nodes=nodes,
                selection_mode=selection_mode, value=0.0,
                sort_direction=sort_direction,
            )
        if self._category == CATEGORY_EXTERNAL:
            load_type = (self._load_combo.currentText()
                         if self._load_combo is not None else "concentrated")
            sort_text = (self._sort_combo_external.currentText()
                         if self._sort_combo_external is not None
                         else "(none)")
            sort_direction = None if sort_text == "(none)" else sort_text
            value_text = (self._external_value_edit.text()
                          if self._external_value_edit is not None
                          else "")
            return ExternalEntry(
                label=label, dof=dof, edge=edge, nodes=nodes,
                selection_mode=selection_mode, load_type=load_type,
                sort_direction=sort_direction,
                value=value_text if value_text.strip() else None,
            )
        raise ValueError(f"Unknown category {self._category!r}.")
