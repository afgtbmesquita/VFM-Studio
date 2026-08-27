# =============================================================================
#  bc_gui / main_window.py  (plate)
# -----------------------------------------------------------------------------
#  QMainWindow that glues the mesh canvas, the BC document model, and the
#  Add/Edit dialogs together for the plate solver.
#
#  Layout (Abaqus-inspired):
#
#      +------------------------------------------------------+
#      | toolbar: mode radios + ID toggles + selection actions |
#      +------------------------------+-----------------------+
#      |                              |  BC tree (Prescribed,  |
#      |                              |  Linking, External)    |
#      |       MeshCanvas             |  + Add / Edit / Delete |
#      |                              |  + Save / Load buttons |
#      |                              |                       |
#      +------------------------------+-----------------------+
#      | status bar: selection summary + last save path        |
#      +------------------------------------------------------+
#
#  Save / Load uses Python module files (.py).  The default output name
#  is ``boundary_conditions_gui.py``; legacy ``.json`` files saved by the
#  Q_Matrix editor can also be loaded as a seed (they are immediately
#  re-written to .py on the next Save).
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QToolBar,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QTabWidget,
    QComboBox,
)

from .bc_dialogs import AddBcDialog
from .bc_model import (
    ALL_CATEGORIES,
    CATEGORY_EXTERNAL,
    CATEGORY_LINKING,
    CATEGORY_PRESCRIBED,
    BcDocument,
    ExternalEntry,
    LinkingEntry,
    PrescribedEntry,
    _BaseEntry,
)
from .mesh_canvas import (
    OVERLAY_EPS_X,
    OVERLAY_EPS_XY,
    OVERLAY_EPS_Y,
    OVERLAY_K_X,
    OVERLAY_K_XY,
    OVERLAY_K_Y,
    OVERLAY_NONE,
    OVERLAY_POINTS,
    MatplotlibToolbar,
    MeshCanvas,
)
from .mesh_editor import (
    MeshEditorPanel,
    default_mesh_settings,
    mesh_settings_from_package,
    mesh_settings_from_points,
)
from .solver_settings_panel import SolverSettingsPanel

# Dropdown spec: (overlay key, menu label, dynamic button label).  Order is
# the order shown to the user; the first entry is the "off" state so the
# QActionGroup always has a selection.
_OVERLAY_MENU_SPECS = (
    (OVERLAY_NONE,   "(no overlay)",           "Show: (none)"),
    (OVERLAY_POINTS, "Show measurement points", "Show: measurement points"),
    (OVERLAY_EPS_X,  "Show Eps X  (\u03b5_xx)", "Show: \u03b5_xx"),
    (OVERLAY_EPS_Y,  "Show Eps Y  (\u03b5_yy)", "Show: \u03b5_yy"),
    (OVERLAY_EPS_XY, "Show Eps XY (\u03b3_xy)", "Show: \u03b3_xy"),
    (OVERLAY_K_X,    "Show Kx     (\u03ba_xx)", "Show: \u03ba_xx"),
    (OVERLAY_K_Y,    "Show Ky     (\u03ba_yy)", "Show: \u03ba_yy"),
    (OVERLAY_K_XY,   "Show Kxy    (\u03ba_xy)", "Show: \u03ba_xy"),
)

_FIELD_OVERLAYS = frozenset({
    OVERLAY_EPS_X,
    OVERLAY_EPS_Y,
    OVERLAY_EPS_XY,
    OVERLAY_K_X,
    OVERLAY_K_Y,
    OVERLAY_K_XY,
})

DEFAULT_OUTPUT_NAME = "boundary_conditions_gui.py"


def default_bc_path_for_test(output_dir: Path,
                             test_index: int,
                             data_file: Path) -> Path:
    """Return the deterministic default BC save path for one test."""
    stem = Path(data_file).stem
    safe = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_"
        for ch in stem
    ).strip("_") or "data"
    return Path(output_dir) / (
        f"boundary_conditions_test_{test_index + 1:02d}_{safe}.py"
    )


@dataclass
class _TestEditorState:
    test_index: int
    data_file: Path
    document: BcDocument
    saved_path: Optional[Path]
    mesh_pkg: Optional[object]
    user_mesh_pkg: Optional[object]
    user_mesh_settings: Optional[object]
    cached_measurement_points: Optional[object] = None
    cached_measurement_strains: Optional[object] = None
    sample_area_total: Optional[float] = None
    material_orientation_deg: float = 0.0


class BcEditorWindow(QMainWindow):
    """Mesh viewer + boundary-condition editor for the plate solver."""

    def __init__(self,
                 mesh_pkg,
                 document: BcDocument,
                 output_dir: Path,
                 data_file: Optional[Path] = None,
                 data_files: Optional[list[Path]] = None,
                 identification_settings=None,
                 analysis_type: Optional[str] = None,
                 mesh_editor_seed_settings=None,
                 external_values_managed_by_parent: bool = False,
                 parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("VFM boundary-condition editor (plate)")
        self.resize(1180, 760)

        self._analysis_type = (str(analysis_type).strip().lower()
                               if analysis_type is not None
                               else (getattr(mesh_pkg, "analysis_type", "both")
                                     if mesh_pkg is not None else "both"))
        self._external_values_managed_by_parent = bool(
            external_values_managed_by_parent)
        # The document persists this so the saved .py file records intent.
        document.analysis_type = self._analysis_type

        restored_mesh_settings = getattr(document, "mesh_settings", None)
        restored_mesh_pkg = None
        if restored_mesh_settings is not None:
            from vfm_mesh import build_mesh
            try:
                restored_mesh_pkg = build_mesh(
                    restored_mesh_settings,
                    analysis_type=self._analysis_type,
                )
                mesh_pkg = restored_mesh_pkg
            except Exception as exc:
                print(
                    "[bc_gui] warning: could not restore MESH_SETTINGS "
                    f"from the BC file: {exc}"
                )

        self._mesh = mesh_pkg

        #  True while the Virtual mesh tab is showing the neutral placeholder
        #  seed, i.e. nobody has told us what the mesh should look like yet.
        #  Cleared as soon as the measurement cloud gives us real bounds.
        self._placeholder_mesh_seed = False
        if self._mesh is not None:
            self._initial_mesh_settings = mesh_settings_from_package(self._mesh)
        elif mesh_editor_seed_settings is not None:
            self._initial_mesh_settings = mesh_editor_seed_settings
        else:
            #  Brand-new experiment: no mesh, no MESH_SETTINGS in the BC
            #  file, nothing saved by the caller.  Open on a placeholder and
            #  refit it to the measurement bounding box once the MAT file
            #  has been read (see ``_seed_mesh_settings_from_measurements``).
            self._initial_mesh_settings = default_mesh_settings()
            self._placeholder_mesh_seed = True
        self._user_mesh_settings = (restored_mesh_settings
                                    if restored_mesh_pkg is not None
                                    else None)
        self._user_mesh_pkg = restored_mesh_pkg
        self._mesh_editor: Optional[MeshEditorPanel] = None
        self._dock_tab_widget: Optional[QTabWidget] = None
        self._solver_panel: Optional[SolverSettingsPanel] = None

        self._document = document
        self._output_dir = Path(output_dir)
        self._saved_path: Optional[Path] = None
        if data_files is None:
            data_files = [Path(data_file)] if data_file is not None else []
        self._data_files = [Path(p) for p in data_files]
        if not self._data_files and data_file is not None:
            self._data_files = [Path(data_file)]
        self._data_file: Optional[Path] = (
            self._data_files[0]
            if self._data_files
            else (Path(data_file) if data_file is not None else None)
        )
        self._active_test_index = 0
        self._test_combo: Optional[QComboBox] = None
        self._test_states = self._initial_test_states(document)
        self._cached_measurement_points = None
        self._cached_measurement_strains = None

        from vfm_user_inputs import IdentificationSettings

        if identification_settings is None:
            identification_settings = IdentificationSettings()
        self._identification_seed = replace(identification_settings)

        self._overlay_button: Optional[QToolButton] = None
        self._overlay_menu: Optional[QMenu] = None
        self._overlay_action_group: Optional[QActionGroup] = None
        self._overlay_actions: Dict[str, QAction] = {}
        self._probe_btn: Optional[QPushButton] = None
        self._overlay_button_labels: Dict[str, str] = {
            key: btn_label
            for key, _menu_label, btn_label in _OVERLAY_MENU_SPECS
        }

        self._bc_overlay_visible: bool = True
        self._bc_overlay_btn: Optional[QPushButton] = None

        # ---- Mesh canvas (central widget) -------------------------------
        self._canvas = MeshCanvas(
            mesh_pkg,
            parent=self,
            analysis_type=self._analysis_type,
        )
        self._canvas.selection_changed.connect(self._on_selection_changed)
        # Push the initial document so every entry is highlighted on launch.
        self._canvas.set_bc_document(self._document)
        self.setCentralWidget(self._canvas)

        # ---- Toolbar -----------------------------------------------------
        self._build_toolbar()

        # ---- Right dock: BC tree + buttons ------------------------------
        self._build_bc_dock()

        # ---- Status bar --------------------------------------------------
        self._status_selection = QLabel("0 nodes selected, 0 elements selected")
        self._status_saved = QLabel("not saved yet")
        self.statusBar().addWidget(self._status_selection, 1)
        self.statusBar().addPermanentWidget(self._status_saved)

        # ---- Sync visuals ----------------------------------------------
        self._refresh_bc_tree()
        self._on_selection_changed()
        self._apply_startup_measurement_overlay()

    def _apply_startup_measurement_overlay(self) -> None:
        """If a MAT path exists, load it once and default the canvas to points."""
        if self._overlay_button is None or not self._overlay_button.isEnabled():
            return
        if OVERLAY_POINTS not in self._overlay_actions:
            return
        try:
            self._load_measurement_data()
        except Exception as exc:
            print(f"[bc_gui] could not preload measurement points: {exc}")
            self._revert_overlay_selection_to_none()
            return
        pts_act = self._overlay_actions[OVERLAY_POINTS]
        none_act = self._overlay_actions.get(OVERLAY_NONE)
        pts_act.blockSignals(True)
        if none_act is not None:
            none_act.blockSignals(True)
            none_act.setChecked(False)
            none_act.blockSignals(False)
        pts_act.setChecked(True)
        pts_act.blockSignals(False)
        self._set_overlay_button_label(OVERLAY_POINTS)
        self._canvas.set_active_overlay(OVERLAY_POINTS)
        self._refresh_probe_button_state()

    # ==================================================================
    #  Public result
    # ==================================================================
    def saved_path(self) -> Optional[Path]:
        return self._saved_path

    def _initial_test_states(self,
                             first_document: BcDocument) -> list[_TestEditorState]:
        """Create ordered per-test GUI state containers."""
        if not self._data_files:
            return []
        states: list[_TestEditorState] = []
        for idx, path in enumerate(self._data_files):
            if idx == 0:
                doc = first_document
                mesh_pkg = self._mesh
                user_mesh_pkg = self._user_mesh_pkg
                user_mesh_settings = self._user_mesh_settings
            else:
                doc = BcDocument(analysis_type=self._analysis_type)
                mesh_pkg = None
                user_mesh_pkg = None
                user_mesh_settings = None
            states.append(_TestEditorState(
                test_index=idx,
                data_file=Path(path),
                document=doc,
                saved_path=None,
                mesh_pkg=mesh_pkg,
                user_mesh_pkg=user_mesh_pkg,
                user_mesh_settings=user_mesh_settings,
            ))
        return states

    def _default_bc_path_for_test(self, test_index: int) -> Path:
        """Deterministic default save path for a test index."""
        if self._data_files and 0 <= test_index < len(self._data_files):
            data_file = self._data_files[test_index]
        else:
            data_file = Path("data.mat")
        return default_bc_path_for_test(self._output_dir, test_index, data_file)

    def _store_current_test_state(self) -> None:
        """Copy current widgets/document state into the active test slot."""
        if not self._test_states:
            return
        state = self._test_states[self._active_test_index]
        state.document = self._document
        state.saved_path = self._saved_path
        state.mesh_pkg = self._mesh
        state.user_mesh_pkg = self._user_mesh_pkg
        state.user_mesh_settings = self._user_mesh_settings
        state.cached_measurement_points = self._cached_measurement_points
        state.cached_measurement_strains = self._cached_measurement_strains
        state.sample_area_total = (
            self._solver_panel.sample_area_total()
            if self._solver_panel is not None else state.sample_area_total
        )
        state.material_orientation_deg = (
            self._solver_panel.material_orientation_deg()
            if self._solver_panel is not None
            else state.material_orientation_deg
        )

    def _restore_test_state(self, test_index: int) -> None:
        """Swap the window to an already stored test state."""
        state = self._test_states[test_index]
        self._active_test_index = test_index
        self._document = state.document
        self._saved_path = state.saved_path
        self._mesh = state.mesh_pkg
        self._user_mesh_pkg = state.user_mesh_pkg
        self._user_mesh_settings = state.user_mesh_settings
        self._data_file = state.data_file
        self._cached_measurement_points = state.cached_measurement_points
        self._cached_measurement_strains = state.cached_measurement_strains

        self._canvas.set_mesh(self._mesh)
        self._canvas.set_bc_document(self._document)
        self._canvas.set_measurement_points(self._cached_measurement_points)
        self._canvas.set_measurement_strains(self._cached_measurement_strains)
        self._revert_overlay_selection_to_none()

        if self._mesh_editor is not None:
            if self._user_mesh_settings is not None:
                self._mesh_editor.sync_with_settings(self._user_mesh_settings)
            else:
                self._mesh_editor.sync_with_settings(self._initial_mesh_settings)
            label = f"Test {test_index + 1}: {state.data_file.name}"
            self._mesh_editor.set_status(label)

        if self._solver_panel is not None:
            data_ok = state.data_file.is_file()
            self._solver_panel.set_data_file_available(data_ok)
            self._solver_panel.set_sample_area_total(state.sample_area_total)
            self._solver_panel.set_material_orientation_deg(
                state.material_orientation_deg)
        self._canvas.set_material_frame_angle(state.material_orientation_deg)

        self._refresh_bc_tree()
        self._on_selection_changed()
        self._status_saved.setText(
            f"loaded {self._saved_path.name}"
            if self._saved_path is not None else "not saved yet"
        )
        self._refresh_measurement_count_overlay()

    def _on_select_test(self, index: int) -> None:
        if index == self._active_test_index or not self._test_states:
            return
        self._store_current_test_state()
        self._restore_test_state(int(index))

    # ==================================================================
    #  Toolbar
    # ==================================================================
    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Selection & view")
        toolbar.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        if len(self._data_files) > 1:
            toolbar.addWidget(QLabel("Test", toolbar))
            self._test_combo = QComboBox(toolbar)
            for idx, path in enumerate(self._data_files, start=1):
                self._test_combo.addItem(f"Test {idx}: {Path(path).name}")
            self._test_combo.currentIndexChanged.connect(self._on_select_test)
            toolbar.addWidget(self._test_combo)
            toolbar.addSeparator()

        mode_group = QActionGroup(toolbar)
        mode_group.setExclusive(True)

        self._mode_navigate = QAction("Pan / Zoom", toolbar, checkable=True)
        self._mode_nodes = QAction("Pick nodes", toolbar, checkable=True)
        self._mode_elements = QAction("Pick elements", toolbar, checkable=True)
        self._mode_nodes.setChecked(True)
        for action, mode_name in (
            (self._mode_navigate, "navigate"),
            (self._mode_nodes, "nodes"),
            (self._mode_elements, "elements"),
        ):
            mode_group.addAction(action)
            toolbar.addAction(action)
            action.triggered.connect(
                lambda _checked=False, mode=mode_name: self._canvas.set_mode(mode)
            )

        toolbar.addSeparator()

        clear_action = QAction("Clear selection", toolbar)
        clear_action.setShortcut(QKeySequence("Esc"))
        clear_action.triggered.connect(self._canvas.clear_selection)
        toolbar.addAction(clear_action)

        elem2node_action = QAction("Elements -> nodes", toolbar)
        elem2node_action.setToolTip(
            "Add the four corner nodes of every selected element to the "
            "node selection.  Useful when picking BCs in element mode."
        )
        elem2node_action.triggered.connect(self._canvas.expand_elements_to_nodes)
        toolbar.addAction(elem2node_action)

        toolbar.addSeparator()

        node_id_action = QAction("Node IDs", toolbar, checkable=True)
        node_id_action.toggled.connect(self._canvas.set_show_node_ids)
        toolbar.addAction(node_id_action)

        elem_id_action = QAction("Element IDs", toolbar, checkable=True)
        elem_id_action.toggled.connect(self._canvas.set_show_element_ids)
        toolbar.addAction(elem_id_action)

        #  On by default: which way the plies run is part of reading the BCs,
        #  not an extra.  The angle itself is set in Solver Settings.
        frame_action = QAction("Material frame", toolbar, checkable=True)
        frame_action.setChecked(True)
        frame_action.setToolTip(
            "Draw the material 1- and 2-axes over the specimen.\n"
            "This is the frame the identified D is expressed in; the mesh, "
            "the measured fields and these boundary conditions are all in the "
            "canvas frame.\n"
            "Set the angle in the Solver Settings tab."
        )
        frame_action.toggled.connect(self._canvas.set_show_material_frame)
        toolbar.addAction(frame_action)
        self._material_frame_action = frame_action

        toolbar.addSeparator()

        # ---- Data-overlay dropdown ------------------------------------
        # Single toolbar button whose menu offers:
        #   * Show measurement points          (dot cloud)
        #   * Show Eps X / Y / XY              (membrane strain contours)
        #   * Show Kx / Ky / Kxy               (curvature contours)
        # plus an explicit "(no overlay)" entry so the user can always
        # turn everything off.  Actions are exclusive (radio-button style)
        # because the canvas can show at most one data overlay at a time.
        self._overlay_button = QToolButton(toolbar)
        self._overlay_button.setText(
            self._overlay_button_labels[OVERLAY_NONE]
        )
        self._overlay_button.setPopupMode(QToolButton.InstantPopup)
        self._overlay_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self._overlay_button.setToolTip(
            "Pick what to overlay on the virtual mesh:\n"
            "  - Show measurement points: experimental point cloud "
            "from the MAT file\n"
            "  - Show Eps X / Y / XY:     filled contour of the "
            "membrane-strain measurements\n"
            "  - Show Kx / Ky / Kxy:      filled contour of the "
            "curvature measurements\n\n"
            "The data is loaded from the MAT file the first time any "
            "option is picked.  A colorbar appears south of the plot "
            "whenever a strain or curvature is shown.\n\n"
            "Use the Probe button (to the right) for left-click read-outs on "
            "contours; right-click removes the nearest probe marker.  "
            "Changing overlay clears probes; Clear selection also removes "
            "probe markers."
        )

        self._overlay_menu = QMenu(self._overlay_button)
        self._overlay_action_group = QActionGroup(self._overlay_menu)
        self._overlay_action_group.setExclusive(True)
        self._overlay_actions = {}

        # Per-analysis filter: in_plane has no curvature data, out_of_plane
        # has no in-plane strain data.  Hide irrelevant entries entirely.
        _membrane_overlays = {OVERLAY_EPS_X, OVERLAY_EPS_Y, OVERLAY_EPS_XY}
        _curvature_overlays = {OVERLAY_K_X, OVERLAY_K_Y, OVERLAY_K_XY}
        if self._analysis_type == "in_plane":
            _hidden_overlays = _curvature_overlays
        elif self._analysis_type == "out_of_plane":
            _hidden_overlays = _membrane_overlays
        else:
            _hidden_overlays = set()

        data_path_ok = (
            self._data_file is not None and Path(self._data_file).is_file()
        )
        default_overlay_key = OVERLAY_POINTS if data_path_ok else OVERLAY_NONE

        for key, menu_label, _btn_label in _OVERLAY_MENU_SPECS:
            if key in _hidden_overlays:
                continue
            action = QAction(menu_label, self._overlay_menu)
            action.setCheckable(True)
            action.setChecked(key == default_overlay_key)
            self._overlay_actions[key] = action
            self._overlay_action_group.addAction(action)
            self._overlay_menu.addAction(action)
            action.triggered.connect(
                lambda _checked=False, k=key: self._on_select_overlay(k)
            )
        self._overlay_button.setMenu(self._overlay_menu)

        if self._data_file is None:
            # No MAT file -> the dropdown is decorative.  Leave the "(no
            # overlay)" entry pre-selected and disable the button.
            self._overlay_button.setEnabled(False)
            self._overlay_button.setToolTip(
                "No experimental MAT file was provided to the GUI; the "
                "data-overlay options are unavailable."
            )

        toolbar.addWidget(self._overlay_button)

        self._probe_btn = QPushButton("Probe", toolbar)
        self._probe_btn.setCheckable(True)
        self._probe_btn.setEnabled(False)
        self._probe_btn.setToolTip(
            "When a strain or curvature contour is shown, turn Probe on to "
            "left-click the plot and show the field value at the nearest "
            "measurement node.  Right-click removes the nearest probe marker.  "
            "Turn Probe off to use normal node/element picking again."
        )
        self._probe_btn.toggled.connect(self._on_toggle_probe)
        toolbar.addWidget(self._probe_btn)
        self._refresh_probe_button_state()

        # Matplotlib's built-in navigation bar (Home / Pan / Zoom / Save).
        # Adding it as a separate toolbar gives the user a "Home" button to
        # reset the view after the measurement-point overlay expands it.
        nav_toolbar = MatplotlibToolbar(self._canvas, self)
        nav_toolbar.setObjectName("MatplotlibNavigationToolbar")
        self.addToolBar(Qt.TopToolBarArea, nav_toolbar)

    # ==================================================================
    #  Right dock
    # ==================================================================
    def _build_bc_dock(self) -> None:
        dock = QDockWidget("Boundary conditions", self)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)

        container = QWidget(dock)
        layout = QVBoxLayout(container)

        # ---- BC tree ----------------------------------------------
        self._tree = QTreeWidget()
        self._tree.setColumnCount(2)
        self._tree.setHeaderLabels(["Entry", "Selection"])
        self._tree.setRootIsDecorated(True)
        self._tree.setUniformRowHeights(True)
        self._tree.itemDoubleClicked.connect(self._on_tree_double_click)
        self._tree.itemSelectionChanged.connect(self._on_tree_selection_changed)
        layout.addWidget(self._tree, 1)

        # ---- Add buttons (one per category) ------------------------
        add_row = QHBoxLayout()
        self._add_prescribed_btn = QPushButton("+ Prescribed")
        self._add_linking_btn = QPushButton("+ Linking")
        self._add_external_btn = QPushButton("+ External")
        for btn, category in (
            (self._add_prescribed_btn, CATEGORY_PRESCRIBED),
            (self._add_linking_btn, CATEGORY_LINKING),
            (self._add_external_btn, CATEGORY_EXTERNAL),
        ):
            btn.clicked.connect(
                lambda _checked=False, cat=category: self._on_add_entry(cat)
            )
            add_row.addWidget(btn)
        layout.addLayout(add_row)

        # ---- Edit / delete ----------------------------------------
        modify_row = QHBoxLayout()
        self._edit_btn = QPushButton("Edit")
        self._edit_btn.clicked.connect(self._on_edit_entry)
        self._delete_btn = QPushButton("Delete")
        self._delete_btn.clicked.connect(self._on_delete_entry)
        self._highlight_btn = QPushButton("Highlight on mesh")
        self._highlight_btn.clicked.connect(self._on_highlight_entry)
        for btn in (self._edit_btn, self._delete_btn, self._highlight_btn):
            btn.setEnabled(False)
            modify_row.addWidget(btn)
        layout.addLayout(modify_row)

        # ---- BC overlay master toggle ----------------------------
        # One dynamic button that flips between "Hide all" (when the BC
        # overlay layer is visible) and "Highlight all" (when it has been
        # turned off).  Operates on the orange / red / purple decoration
        # layer drawn on top of the mesh.
        overlay_row = QHBoxLayout()
        self._bc_overlay_btn = QPushButton()
        self._bc_overlay_btn.clicked.connect(self._on_toggle_bc_overlay)
        overlay_row.addWidget(self._bc_overlay_btn)
        layout.addLayout(overlay_row)
        self._refresh_bc_overlay_button()

        # ---- Save / Load -----------------------------------------
        io_row = QHBoxLayout()
        self._save_btn = QPushButton("Save")
        self._save_btn.setShortcut(QKeySequence.Save)
        self._save_btn.clicked.connect(self._on_save)
        self._save_as_btn = QPushButton("Save as...")
        self._save_as_btn.clicked.connect(self._on_save_as)
        self._load_btn = QPushButton("Load...")
        self._load_btn.clicked.connect(self._on_load)
        io_row.addWidget(self._save_btn)
        io_row.addWidget(self._save_as_btn)
        io_row.addWidget(self._load_btn)
        layout.addLayout(io_row)

        # ---- Live virtual-mesh editor + solver settings ----------
        # Tabbed panel: mesh counts / bounds, then identification options.
        self._dock_tab_widget = QTabWidget()
        self._mesh_editor = MeshEditorPanel(
            initial_settings=self._initial_mesh_settings,
            parent=self,
        )
        if self._user_mesh_settings is not None:
            self._mesh_editor.set_status(
                "Restored virtual mesh settings from the loaded BC file."
            )
        elif self._mesh is None:
            self._mesh_editor.set_status(
                "No virtual mesh on the plot yet — click Apply to build one "
                "from the settings above, or load a BC file that includes "
                "MESH_SETTINGS."
            )
        self._mesh_editor.mesh_settings_changed.connect(
            self._on_apply_new_mesh
        )
        self._mesh_editor.manual_nodes_preview_changed.connect(
            self._canvas.set_manual_node_preview
        )
        self._mesh_editor.manual_nodes_preview_cleared.connect(
            self._canvas.clear_manual_node_preview
        )
        data_ok = (
            self._data_file is not None
            and Path(self._data_file).is_file()
        )
        saved_solver_settings = getattr(self._document, "solver_settings", None) or {}
        max_iterations = saved_solver_settings.get(
            "max_iterations",
            int(self._identification_seed.max_iterations),
        )
        convergence_tolerance = saved_solver_settings.get(
            "convergence_tolerance",
            float(self._identification_seed.convergence_tolerance),
        )
        self._solver_panel = SolverSettingsPanel(
            max_iterations=int(max_iterations),
            convergence_tolerance=float(convergence_tolerance),
            data_file_available=data_ok,
            parent=self,
        )
        self._apply_document_solver_settings(saved_solver_settings)
        self._solver_panel.measurement_counts_toggled.connect(
            self._on_measurement_counts_toggled
        )
        #  Live: the arrows on the canvas track the spin box while the user
        #  is picking nodes, so the frame theta is expressed in is never a
        #  number they have to hold in their head.
        self._solver_panel.material_orientation_changed.connect(
            self._canvas.set_material_frame_angle
        )
        self._canvas.set_material_frame_angle(
            self._solver_panel.material_orientation_deg()
        )
        self._dock_tab_widget.addTab(self._mesh_editor, "Virtual mesh")
        self._dock_tab_widget.addTab(self._solver_panel, "Solver Settings")
        self._dock_tab_widget.currentChanged.connect(self._on_dock_tab_changed)
        layout.addWidget(self._dock_tab_widget)

        # ---- Accept / cancel buttons -----------------------------
        finish_row = QHBoxLayout()
        self._use_btn = QPushButton("Use saved file && run solver")
        self._use_btn.setToolTip(
            "Closes the editor.  run_vfm.py uses the current saved or "
            "loaded .py file as the boundary-conditions module for the run."
        )
        self._use_btn.clicked.connect(self._on_use_and_close)
        self._cancel_btn = QPushButton("Cancel (exit without running)")
        self._cancel_btn.setToolTip(
            "Closes the editor and stops run_vfm.py without loading measurements "
            "or running the identification."
        )
        self._cancel_btn.clicked.connect(self._on_cancel_and_close)
        finish_row.addWidget(self._use_btn)
        finish_row.addWidget(self._cancel_btn)
        layout.addLayout(finish_row)

        container.setLayout(layout)
        dock.setWidget(container)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    # ==================================================================
    #  Right-dock solver tab + measurement-count overlay
    # ==================================================================
    def _on_dock_tab_changed(self, index: int) -> None:
        if index == 1:
            self._ensure_solver_tab_primed()

    def _ensure_solver_tab_primed(self) -> None:
        """Lazy-load measurements once so the sample-area spin box matches the file."""
        if self._data_file is None or not Path(self._data_file).is_file():
            return
        if self._solver_panel is None:
            return
        try:
            if (self._cached_measurement_points is None
                    or self._cached_measurement_strains is None):
                self._load_measurement_data()
            else:
                area = self._bbox_area_from_points(self._cached_measurement_points)
                self._solver_panel.set_automatic_sample_area(area)
        except Exception:
            # Errors are surfaced when the user actually requests an overlay;
            # here we avoid duplicate modal spam on tab switches.
            pass

    @staticmethod
    def _bbox_area_from_points(points) -> float:
        pts = np.asarray(points, dtype=float)
        dx = float(pts[:, 0].max() - pts[:, 0].min())
        dy = float(pts[:, 1].max() - pts[:, 1].min())
        return dx * dy

    def _on_measurement_counts_toggled(self, checked: bool) -> None:
        if not checked:
            self._canvas.set_element_measurement_counts(None)
            return
        if self._data_file is None or not Path(self._data_file).is_file():
            QMessageBox.information(
                self, "No measurement file",
                "Cannot show per-element counts without an experimental MAT file."
            )
            if self._solver_panel is not None:
                self._solver_panel.clear_measurement_counts()
            return
        try:
            if (self._cached_measurement_points is None
                    or self._cached_measurement_strains is None):
                self._load_measurement_data()
        except Exception as exc:
            QMessageBox.critical(
                self, "Could not load measurement data",
                f"Failed to read measurement data from\n"
                f"  {self._data_file}\n\n{exc}"
            )
            if self._solver_panel is not None:
                self._solver_panel.clear_measurement_counts()
            return
        self._refresh_measurement_count_overlay()

    def _refresh_measurement_count_overlay(self) -> None:
        if self._mesh is None:
            self._canvas.set_element_measurement_counts(None)
            return
        if (self._solver_panel is None
                or not self._solver_panel.measurement_counts_enabled()):
            self._canvas.set_element_measurement_counts(None)
            return
        if self._cached_measurement_points is None:
            return
        from vfm_mesh import count_measurements_per_element

        counts = count_measurements_per_element(
            self._cached_measurement_points,
            self._mesh,
        )
        self._canvas.set_element_measurement_counts(counts)

    def _current_solver_settings_payload(self) -> Dict[str, object]:
        """Return the solver-tab state in the BC-file metadata schema."""
        if self._solver_panel is None:
            return dict(getattr(self._document, "solver_settings", None) or {})
        return {
            "sample_area_total": self._solver_panel.sample_area_total(),
            "max_iterations": self._solver_panel.max_iterations(),
            "convergence_tolerance": self._solver_panel.convergence_tolerance(),
        }

    def _apply_document_solver_settings(self,
                                        solver_settings: Optional[Dict[str, object]]
                                        ) -> None:
        """Restore solver-tab widgets from a loaded BC document."""
        if self._solver_panel is None or not solver_settings:
            return
        self._solver_panel.apply_solver_settings(
            sample_area_total=solver_settings.get("sample_area_total"),
            max_iterations=solver_settings.get("max_iterations"),
            convergence_tolerance=solver_settings.get("convergence_tolerance"),
        )

    # ==================================================================
    #  Selection / tree synchronisation
    # ==================================================================
    def _on_selection_changed(self) -> None:
        n_nodes = len(self._canvas.selected_nodes())
        n_elements = len(self._canvas.selected_elements())
        self._status_selection.setText(
            f"{n_nodes} node(s) selected, {n_elements} element(s) selected"
        )

    def _on_tree_selection_changed(self) -> None:
        item = self._current_entry_item()
        has_entry = item is not None
        self._edit_btn.setEnabled(has_entry)
        self._delete_btn.setEnabled(has_entry)
        self._highlight_btn.setEnabled(has_entry)

    def _current_entry_item(self) -> Optional[QTreeWidgetItem]:
        items = self._tree.selectedItems()
        if not items:
            return None
        item = items[0]
        if item.parent() is None:           # category header
            return None
        return item

    def _category_and_index(self,
                            item: QTreeWidgetItem) -> tuple[str, int]:
        parent = item.parent()
        category = parent.data(0, Qt.UserRole)
        index = parent.indexOfChild(item)
        return category, index

    def _refresh_bc_tree(self) -> None:
        previously_selected: Optional[tuple[str, int]] = None
        current = self._current_entry_item()
        if current is not None:
            previously_selected = self._category_and_index(current)

        self._tree.clear()

        for category in ALL_CATEGORIES:
            entries = self._document.entries(category)
            header = QTreeWidgetItem(
                [self._category_header(category, len(entries)), ""]
            )
            header.setData(0, Qt.UserRole, category)
            header.setFirstColumnSpanned(False)
            font = header.font(0)
            font.setBold(True)
            header.setFont(0, font)
            self._tree.addTopLevelItem(header)
            for entry in entries:
                child = QTreeWidgetItem([
                    self._entry_label(entry),
                    self._entry_selection_text(entry),
                ])
                child.setToolTip(0, self._entry_tooltip(entry))
                header.addChild(child)
            header.setExpanded(True)

        self._tree.resizeColumnToContents(0)

        if previously_selected is not None:
            category, index = previously_selected
            self._select_tree_entry(category, index)

        self._on_tree_selection_changed()

        if hasattr(self, "_canvas") and self._canvas is not None:
            self._canvas.set_bc_document(self._document)

    def _select_tree_entry(self, category: str, index: int) -> None:
        for i in range(self._tree.topLevelItemCount()):
            header = self._tree.topLevelItem(i)
            if header.data(0, Qt.UserRole) != category:
                continue
            if 0 <= index < header.childCount():
                child = header.child(index)
                child.setSelected(True)
                self._tree.setCurrentItem(child)
            return

    @staticmethod
    def _category_header(category: str, count: int) -> str:
        label_map = {
            CATEGORY_PRESCRIBED: "Prescribed (Dirichlet)",
            CATEGORY_LINKING: "Linking (tie)",
            CATEGORY_EXTERNAL: "External work",
        }
        return f"{label_map[category]}  -- {count}"

    @staticmethod
    def _entry_label(entry: _BaseEntry) -> str:
        kind = ""
        if isinstance(entry, PrescribedEntry):
            kind = "[P]"
        elif isinstance(entry, LinkingEntry):
            kind = "[L]"
        elif isinstance(entry, ExternalEntry):
            kind = f"[E,{entry.load_type}]"
            if entry.value is not None:
                return (f"{kind} {entry.dof}  {entry.label}  "
                        f"(F={entry.value:g})")
        return f"{kind} {entry.dof}  {entry.label}"

    @staticmethod
    def _entry_selection_text(entry: _BaseEntry) -> str:
        if entry.selection_mode == "edge" and entry.edge is not None:
            return f"edge: {entry.edge}"
        nodes = entry.nodes or []
        if not nodes:
            return "(no nodes)"
        if len(nodes) <= 6:
            return "nodes: " + ", ".join(str(n) for n in nodes)
        head = ", ".join(str(n) for n in nodes[:5])
        return f"nodes: [{head}, ... ({len(nodes)} total)]"

    @staticmethod
    def _entry_tooltip(entry: _BaseEntry) -> str:
        lines = [
            f"label: {entry.label}",
            f"DOF:   {entry.dof}",
        ]
        if entry.selection_mode == "edge" and entry.edge is not None:
            lines.append(f"edge:  {entry.edge}")
        else:
            nodes = entry.nodes or []
            lines.append(f"nodes: {nodes}")
        if isinstance(entry, LinkingEntry) and entry.sort_direction:
            lines.append(f"sortDirection: {entry.sort_direction}")
        if isinstance(entry, ExternalEntry):
            lines.append(f"loadType: {entry.load_type}")
            if entry.sort_direction:
                lines.append(f"sortDirection: {entry.sort_direction}")
            if entry.value is not None:
                lines.append(f"value: {entry.value:g}")
            else:
                lines.append("value: <use F from MAT file>")
        return "\n".join(lines)

    # ==================================================================
    #  BC actions
    # ==================================================================
    def _on_add_entry(self, category: str) -> None:
        if self._mesh is None:
            QMessageBox.information(
                self, "No virtual mesh",
                "Apply a virtual mesh in the Virtual mesh tab (or load a BC "
                "file that includes mesh settings) before adding boundary "
                "conditions."
            )
            return
        node_ids = self._canvas.selected_nodes()
        if not node_ids and self._canvas.selected_elements():
            self._canvas.expand_elements_to_nodes()
            node_ids = self._canvas.selected_nodes()
        if not node_ids:
            QMessageBox.information(
                self, "No selection",
                "Pick at least one node (or element) on the mesh first, "
                "then click '+ {0}'.".format(category.capitalize())
            )
            return

        dialog = AddBcDialog(
            category=category,
            selected_node_ids=node_ids,
            mesh_pkg=self._mesh,
            analysis_type=self._analysis_type,
            external_values_managed_by_parent=(
                self._external_values_managed_by_parent),
            parent=self,
        )
        if dialog.exec() != AddBcDialog.Accepted:
            return

        entry = dialog.result_entry()
        self._document.add(category, entry)
        self._refresh_bc_tree()
        new_index = self._document.count(category) - 1
        self._select_tree_entry(category, new_index)

    def _on_edit_entry(self) -> None:
        item = self._current_entry_item()
        if item is None:
            return
        if self._mesh is None:
            QMessageBox.information(
                self, "No virtual mesh",
                "Apply a virtual mesh (or load a BC file with mesh settings) "
                "before editing entries that reference the mesh."
            )
            return
        category, index = self._category_and_index(item)
        existing = self._document.entries(category)[index]

        seed_nodes = self._canvas.selected_nodes() or (existing.nodes or [])
        dialog = AddBcDialog(
            category=category,
            selected_node_ids=seed_nodes,
            mesh_pkg=self._mesh,
            existing_entry=existing,
            analysis_type=self._analysis_type,
            external_values_managed_by_parent=(
                self._external_values_managed_by_parent),
            parent=self,
        )
        if dialog.exec() != AddBcDialog.Accepted:
            return

        new_entry = dialog.result_entry()
        self._document.replace(category, index, new_entry)
        self._refresh_bc_tree()
        self._select_tree_entry(category, index)

    def _on_delete_entry(self) -> None:
        item = self._current_entry_item()
        if item is None:
            return
        category, index = self._category_and_index(item)
        entry = self._document.entries(category)[index]
        confirm = QMessageBox.question(
            self, "Delete entry",
            f"Delete '{entry.label}' from {category}?",
        )
        if confirm != QMessageBox.Yes:
            return
        self._document.remove(category, index)
        self._refresh_bc_tree()

    def _on_toggle_bc_overlay(self) -> None:
        """Flip the BC-overlay visibility and refresh the dynamic button."""
        self._bc_overlay_visible = not self._bc_overlay_visible
        self._canvas.set_show_bc_decorations(self._bc_overlay_visible)
        self._refresh_bc_overlay_button()

    def _refresh_bc_overlay_button(self) -> None:
        """Synchronise the toggle button's label/tooltip to current state."""
        if self._bc_overlay_btn is None:
            return
        if self._bc_overlay_visible:
            self._bc_overlay_btn.setText("Hide all")
            self._bc_overlay_btn.setToolTip(
                "Hide every BC decoration (orange Prescribed arrows / "
                "red External-load arrows / purple Linking markers) on "
                "the mesh.  Click again to bring them back."
            )
        else:
            self._bc_overlay_btn.setText("Highlight all")
            self._bc_overlay_btn.setToolTip(
                "Show every BC decoration (orange Prescribed arrows / "
                "red External-load arrows / purple Linking markers) on "
                "the mesh."
            )

    def _on_highlight_entry(self) -> None:
        item = self._current_entry_item()
        if item is None:
            return
        if self._mesh is None:
            QMessageBox.information(
                self, "No virtual mesh",
                "Apply a virtual mesh (or load a BC file with mesh settings) "
                "before highlighting BC entries on the plot."
            )
            return
        category, index = self._category_and_index(item)
        entry = self._document.entries(category)[index]
        node_ids = self._entry_node_ids(entry)
        if not node_ids:
            return
        self._canvas.set_selected_nodes(node_ids)

    def _entry_node_ids(self, entry: _BaseEntry) -> list[int]:
        if entry.selection_mode == "edge" and entry.edge is not None:
            if self._mesh is None:
                return []
            from .bc_dialogs import _edge_node_indices
            return sorted(_edge_node_indices(self._mesh, entry.edge).tolist())
        return list(entry.nodes or [])

    def _on_tree_double_click(self, item: QTreeWidgetItem, _column: int) -> None:
        if item.parent() is None:
            return
        self._on_edit_entry()

    # ==================================================================
    #  Save / Load
    # ==================================================================
    def _on_save(self) -> None:
        target = self._default_bc_path_for_test(self._active_test_index)
        self._save_to_path(target)

    def _on_save_as(self) -> None:
        default_name = (self._saved_path.name if self._saved_path is not None
                        else self._default_bc_path_for_test(
                            self._active_test_index).name)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save boundary conditions",
            str(self._output_dir / default_name),
            "Boundary-condition Python (*.py);;All files (*.*)",
        )
        if not path:
            return
        target = Path(path)
        # Force a .py extension -- the plate solver rejects anything else.
        if target.suffix.lower() != ".py":
            target = target.with_suffix(".py")
        self._save_to_path(target)

    def _save_to_path(self, target: Path) -> None:
        try:
            if self._mesh is None:
                mesh_settings = self._mesh_editor.current_settings()
            else:
                mesh_settings = mesh_settings_from_package(self._mesh)
            self._document.mesh_settings = mesh_settings
            self._document.solver_settings = self._current_solver_settings_payload()
            target = self._document.write_py(target, keep_backup=True)
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self._saved_path = target
        self._store_current_test_state()
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._status_saved.setText(f"saved {target.name} at {timestamp}")

    def _on_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load boundary conditions",
            str(self._output_dir),
            "Boundary-condition Python (*.py);;"
            "Legacy JSON (*.json);;All files (*.*)",
        )
        if not path:
            return
        try:
            new_doc = BcDocument.from_any_file(Path(path))
        except Exception as exc:
            QMessageBox.critical(self, "Load failed",
                                  f"Could not read {path}:\n{exc}")
            return
        loaded_path = Path(path)
        if new_doc.mesh_settings is not None:
            if not self._restore_mesh_for_loaded_document(new_doc.mesh_settings):
                return

        # Warn on analysis-type mismatch and on entries with DOFs that are
        # not allowed in the active analysis type.
        loaded_analysis = new_doc.analysis_type
        if loaded_analysis is not None and loaded_analysis != self._analysis_type:
            QMessageBox.warning(
                self, "Analysis-type mismatch",
                f"The BC file declares ANALYSIS_TYPE='{loaded_analysis}' "
                f"but this run is configured for "
                f"'{self._analysis_type}'.  Incompatible DOFs will be "
                "rejected by the solver; please review the entries."
            )
        incompatibles = new_doc.incompatible_entries(self._analysis_type)
        if incompatibles:
            entry_lines = []
            for category, idx, entry in incompatibles[:10]:
                entry_lines.append(
                    f"  - {category}[{idx}] {entry.label!r}: DOF '{entry.dof}'"
                )
            more = ("" if len(incompatibles) <= 10
                    else f"\n  ... and {len(incompatibles) - 10} more")
            QMessageBox.warning(
                self, "Incompatible BC entries",
                "The following entries reference DOFs that are not "
                f"allowed in analysis_type='{self._analysis_type}':\n"
                + "\n".join(entry_lines) + more
                + "\n\nEdit or delete them before saving."
            )

        # Pin the document to the active analysis type so the next save
        # records the run intent rather than the (possibly older) loaded
        # value.
        new_doc.analysis_type = self._analysis_type
        self._document = new_doc
        self._apply_document_solver_settings(new_doc.solver_settings)
        self._refresh_bc_tree()
        self._saved_path = (loaded_path
                            if loaded_path.suffix.lower() == ".py"
                            else None)
        self._store_current_test_state()
        self._status_saved.setText(f"loaded {loaded_path.name}")

    # ==================================================================
    #  Window close handling
    # ==================================================================
    def _on_use_and_close(self) -> None:
        self._store_current_test_state()
        if self._test_states:
            for idx in range(len(self._test_states)):
                if idx != self._active_test_index:
                    self._restore_test_state(idx)
                if self._saved_path is None:
                    self._save_to_path(self._default_bc_path_for_test(idx))
                    if self._saved_path is None:
                        return
                self._store_current_test_state()
            self._restore_test_state(0)
            if self._test_combo is not None:
                self._test_combo.blockSignals(True)
                self._test_combo.setCurrentIndex(0)
                self._test_combo.blockSignals(False)
        elif self._saved_path is None:
            self._save_to_path(self._output_dir / DEFAULT_OUTPUT_NAME)
            if self._saved_path is None:
                return
        self.close()

    def _on_cancel_and_close(self) -> None:
        self._saved_path = None
        # Cancel also discards any live mesh edits the user applied.  The
        # caller relies on ``final_mesh_pkg() is None`` to detect this.
        self._user_mesh_settings = None
        self._user_mesh_pkg = None
        for state in self._test_states:
            state.saved_path = None
            state.user_mesh_settings = None
            state.user_mesh_pkg = None
        self.close()

    # ==================================================================
    #  Final-state accessors (consumed by ``launch_bc_editor``)
    # ==================================================================
    def final_mesh_pkg(self):
        """Return the rebuilt ``MeshPackage`` or ``None`` if untouched.

        ``None`` means the user never applied a new mesh (or pressed
        Cancel); callers should fall back to the mesh that was passed in
        at construction time.
        """
        return self._user_mesh_pkg

    def final_mesh_settings(self):
        """Return the ``MeshSettings`` matching ``final_mesh_pkg()``."""
        return self._user_mesh_settings

    def final_sample_area_total(self) -> Optional[float]:
        """Total sample area for ``area_per_point``, or ``None`` if dialog aborted."""
        if self._saved_path is None or self._solver_panel is None:
            return None
        return self._solver_panel.sample_area_total()

    def set_material_orientation_deg(self, angle_deg: float) -> None:
        """Seed the material-frame angle from the caller (the setup dialog).

        Both the panel spin box and the canvas arrows are updated, and the
        value is written into every stored per-test slot so switching tests
        does not silently revert it.
        """
        angle = float(angle_deg or 0.0)
        if self._solver_panel is not None:
            self._solver_panel.set_material_orientation_deg(angle)
        for state in self._test_states:
            state.material_orientation_deg = angle
        self._canvas.set_material_frame_angle(angle)

    def final_material_orientation_deg(self) -> Optional[float]:
        """Material-frame angle, or ``None`` if the dialog was aborted."""
        if self._saved_path is None or self._solver_panel is None:
            return None
        return self._solver_panel.material_orientation_deg()

    def final_max_iterations(self) -> Optional[int]:
        if self._saved_path is None or self._solver_panel is None:
            return None
        return self._solver_panel.max_iterations()

    def final_convergence_tolerance(self) -> Optional[float]:
        if self._saved_path is None or self._solver_panel is None:
            return None
        return self._solver_panel.convergence_tolerance()

    def final_test_results(self):
        """Return ordered per-test result objects for the launcher."""
        self._store_current_test_state()
        if not self._test_states:
            return []
        from .app import BcEditorTestResult

        return [
            BcEditorTestResult(
                test_index=state.test_index,
                data_file=state.data_file,
                saved_bc_path=state.saved_path,
                mesh_settings=state.user_mesh_settings,
                mesh_pkg=state.user_mesh_pkg,
                sample_area_total=state.sample_area_total,
                material_orientation_deg=state.material_orientation_deg,
                #  The live document, so a caller that never asked the user to
                #  save a .py file can still read the entries back.
                document=state.document,
            )
            for state in self._test_states
        ]

    # ==================================================================
    #  Live virtual-mesh editor
    # ==================================================================
    def _active_mesh_settings(self):
        """Return the mesh settings that should drive the next rebuild."""
        if self._user_mesh_settings is not None:
            return self._user_mesh_settings
        if self._mesh is not None:
            from bc_gui.mesh_editor import mesh_settings_from_package
            return mesh_settings_from_package(self._mesh)
        return self._initial_mesh_settings

    @staticmethod
    def _mesh_topology_changed(old_pkg, new_pkg) -> bool:
        if old_pkg is None or new_pkg is None:
            return old_pkg is not new_pkg
        if (old_pkg.n_nodes != new_pkg.n_nodes
                or old_pkg.n_elements != new_pkg.n_elements):
            return True
        return not np.allclose(old_pkg.node_coords, new_pkg.node_coords)

    def _clear_all_bcs(self) -> None:
        for category in (CATEGORY_PRESCRIBED, CATEGORY_LINKING,
                         CATEGORY_EXTERNAL):
            while self._document.count(category) > 0:
                self._document.remove(category, 0)
        self._refresh_bc_tree()

    def _build_mesh_pkg_from_settings(self, settings):
        """Build the structured mesh and prune empty elements when possible."""
        from vfm_mesh import build_mesh, prune_mesh_by_measurement_points

        mesh_pkg = build_mesh(settings, analysis_type=self._analysis_type)
        prune_report = None
        points = self._cached_measurement_points
        if points is not None and np.asarray(points).size > 0:
            mesh_pkg, prune_report = prune_mesh_by_measurement_points(
                mesh_pkg, points)
        return mesh_pkg, prune_report

    def _apply_mesh_package(self,
                            new_pkg,
                            settings,
                            *,
                            prune_report=None,
                            clear_bcs: bool = False,
                            status_prefix: str = "") -> None:
        """Swap the canvas mesh and record the solver-facing package."""
        topology_changed = self._mesh_topology_changed(self._mesh, new_pkg)
        if clear_bcs or topology_changed:
            self._clear_all_bcs()

        self._canvas.set_mesh(new_pkg)
        self._mesh = new_pkg
        self._user_mesh_pkg = new_pkg
        self._user_mesh_settings = settings

        status = (
            f"{status_prefix}{new_pkg.n_elements} elements, "
            f"{new_pkg.n_nodes} nodes, {new_pkg.n_dofs} DOFs."
        )
        if prune_report is not None and prune_report.n_elements_removed > 0:
            status += (
                f"  Removed {prune_report.n_elements_removed} empty "
                f"element(s) with no measurement points."
            )
        if topology_changed and not clear_bcs:
            status += "  Node/element IDs changed -- BCs were cleared."
        if clear_bcs:
            status += "  All previous BCs were dropped."

        if self._mesh_editor is not None:
            self._mesh_editor.sync_with_settings(settings)
            self._mesh_editor.set_status(status)

        self._on_selection_changed()
        self._refresh_measurement_count_overlay()

    def _reconcile_mesh_with_measurements(self) -> bool:
        """Rebuild and prune the mesh after measurement data is available."""
        settings = self._active_mesh_settings()
        if settings is None:
            return False
        try:
            new_pkg, prune_report = self._build_mesh_pkg_from_settings(settings)
        except ValueError as exc:
            QMessageBox.critical(
                self, "Mesh prune failed",
                f"Could not drop empty virtual elements:\n\n{exc}",
            )
            return False
        except Exception as exc:
            QMessageBox.critical(
                self, "Mesh rebuild failed",
                f"Could not rebuild the virtual mesh:\n\n{exc}",
            )
            return False

        self._apply_mesh_package(
            new_pkg,
            settings,
            prune_report=prune_report,
            status_prefix="Mesh updated: ",
        )
        return True

    def _restore_mesh_for_loaded_document(self, settings) -> bool:
        """Rebuild the displayed mesh from a saved BC file without clearing BCs."""
        try:
            new_pkg, prune_report = self._build_mesh_pkg_from_settings(settings)
        except Exception as exc:
            QMessageBox.critical(
                self, "Mesh restore failed",
                "The BC file was read, but its saved MESH_SETTINGS could "
                f"not rebuild the virtual mesh:\n\n{exc}"
            )
            return False

        self._apply_mesh_package(
            new_pkg,
            settings,
            prune_report=prune_report,
            status_prefix="Restored saved mesh: ",
        )
        return True

    def _on_apply_new_mesh(self, new_settings) -> None:
        """Slot for ``MeshEditorPanel.mesh_settings_changed``.

        Confirms the destructive BC reset with the user (the answered
        question selected the "drop all BCs on mesh change" policy),
        rebuilds the mesh via :func:`vfm_mesh.build_mesh`, swaps it into
        the canvas, and clears the BC document so the tree and the
        overlay agree with the new mesh.
        """
        has_bcs = (self._document.count(CATEGORY_PRESCRIBED) > 0
                   or self._document.count(CATEGORY_LINKING) > 0
                   or self._document.count(CATEGORY_EXTERNAL) > 0)
        if has_bcs:
            confirm = QMessageBox.question(
                self, "Rebuild virtual mesh?",
                "Rebuilding the mesh will drop every existing boundary "
                "condition because node and element IDs are not stable "
                "across a mesh rebuild.\n\n"
                "Edge-based BCs would be safe but, per the chosen "
                "policy, ALL entries are cleared so prescribed / "
                "linking / external lists stay consistent.\n\n"
                "Continue?",
            )
            if confirm != QMessageBox.Yes:
                if self._mesh_editor is not None:
                    self._mesh_editor.set_status(
                        "Rebuild cancelled -- mesh and BCs unchanged."
                    )
                return

        # Import here to avoid a hard dependency at module import time
        # (the bc_gui smoke test imports this module from a context that
        # may not have scikit-fem loaded yet).
        try:
            new_pkg, prune_report = self._build_mesh_pkg_from_settings(
                new_settings)
        except Exception as exc:
            QMessageBox.critical(
                self, "Mesh rebuild failed",
                f"Could not rebuild the mesh with the new settings:\n\n{exc}"
            )
            if self._mesh_editor is not None:
                self._mesh_editor.set_status(
                    f"Mesh rebuild failed: {exc}"
                )
            return

        self._apply_mesh_package(
            new_pkg,
            new_settings,
            prune_report=prune_report,
            clear_bcs=True,
            status_prefix="Mesh rebuilt: ",
        )

    # ==================================================================
    #  Data-overlay dropdown
    # ==================================================================
    def _on_select_overlay(self, key: str) -> None:
        """Slot for the toolbar dropdown.

        Handles three cases:
          * ``OVERLAY_NONE`` -> hide every data overlay (no disk I/O).
          * Any other key, when measurement data is not yet cached
            -> load the MAT file once, push it into the canvas, and
               then activate the requested overlay.
          * Any other key with data already cached -> just activate it.

        On a load error the dropdown is reverted to ``OVERLAY_NONE`` and
        the canvas is told the same, so the GUI never ends up with a
        checked menu item that has no data backing it.
        """
        if key == OVERLAY_NONE:
            self._set_overlay_button_label(OVERLAY_NONE)
            self._canvas.set_active_overlay(OVERLAY_NONE)
            self._refresh_probe_button_state()
            return

        if (self._cached_measurement_points is None
                or self._cached_measurement_strains is None):
            try:
                self._load_measurement_data()
            except Exception as exc:
                QMessageBox.critical(
                    self, "Could not load measurement data",
                    f"Failed to read measurement data from\n"
                    f"  {self._data_file}\n\n{exc}"
                )
                self._revert_overlay_selection_to_none()
                return

        self._set_overlay_button_label(key)
        self._canvas.set_active_overlay(key)
        self._refresh_probe_button_state()

    def _on_toggle_probe(self, checked: bool) -> None:
        """Toolbar ``Probe`` button toggles contour read-out mode on the canvas."""
        self._canvas.set_probe_mode(bool(checked))

    def _refresh_probe_button_state(self) -> None:
        """Enable Probe only for strain/curvature overlays; sync canvas mode."""
        if self._probe_btn is None:
            return
        key = self._canvas.active_overlay()
        field = key in _FIELD_OVERLAYS
        if not field:
            self._probe_btn.blockSignals(True)
            self._probe_btn.setChecked(False)
            self._probe_btn.setEnabled(False)
            self._probe_btn.blockSignals(False)
            self._canvas.set_probe_mode(False)
        else:
            self._probe_btn.setEnabled(True)

    def _revert_overlay_selection_to_none(self) -> None:
        """Force the dropdown + canvas back to the off state."""
        action = self._overlay_actions.get(OVERLAY_NONE)
        if action is not None:
            blocker = action.blockSignals(True)
            action.setChecked(True)
            action.blockSignals(blocker)
        self._set_overlay_button_label(OVERLAY_NONE)
        self._canvas.set_active_overlay(OVERLAY_NONE)
        self._refresh_probe_button_state()

    def _set_overlay_button_label(self, key: str) -> None:
        """Update the toolbar button text to mirror the active overlay."""
        if self._overlay_button is None:
            return
        label = self._overlay_button_labels.get(
            key, self._overlay_button_labels[OVERLAY_NONE]
        )
        self._overlay_button.setText(label)

    def _load_measurement_data(self) -> None:
        """Read the MAT file once and cache both points and strains.

        Uses ``vfm_data_io.load_point_measurements`` so the GUI shares the
        same file-format and validation logic as the solver.  Imports it
        lazily so the GUI module stays importable when ``scipy`` is absent.
        Pushes the parsed arrays straight into the canvas so subsequent
        overlay switches happen in memory.
        """
        if self._data_file is None:
            raise FileNotFoundError(
                "No experimental MAT file path was passed to the GUI."
            )

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            from vfm_data_io import load_point_measurements
            measurements = load_point_measurements(
                self._data_file,
                analysis_type=self._analysis_type,
            )
        finally:
            QApplication.restoreOverrideCursor()

        self._cached_measurement_points = measurements.points
        self._cached_measurement_strains = measurements.strains
        # Push to the canvas in order: points first (so the strain setter
        # can validate its column count against the cached point count).
        self._canvas.set_measurement_points(measurements.points)
        self._canvas.set_measurement_strains(measurements.strains)
        if self._solver_panel is not None:
            area = self._bbox_area_from_points(measurements.points)
            self._solver_panel.set_automatic_sample_area(area)

        self._seed_mesh_settings_from_measurements()

        if self._user_mesh_settings is not None or self._mesh is not None:
            self._reconcile_mesh_with_measurements()

    def _seed_mesh_settings_from_measurements(self) -> None:
        """Fit the placeholder mesh seed to the measurement bounding box.

        Only runs while the Virtual mesh tab still holds the placeholder:
        once a mesh exists -- applied by the user or restored from the BC
        file -- those values are authoritative and must not be overwritten.
        """
        if not self._placeholder_mesh_seed:
            return
        if self._mesh is not None or self._user_mesh_settings is not None:
            return
        fitted = mesh_settings_from_points(self._cached_measurement_points)
        if fitted is None:
            return
        self._initial_mesh_settings = fitted
        self._placeholder_mesh_seed = False
        if self._mesh_editor is not None:
            self._mesh_editor.sync_with_settings(fitted)
            self._mesh_editor.set_status(
                "Bounds seeded from the measurement cloud — adjust the "
                "element counts and click Apply to build the mesh."
            )
