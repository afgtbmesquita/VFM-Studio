"""Property editor dock — edits the session document in place."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from vfmstudio.core import mesh_convert
from vfmstudio.core.fields import (
    estimate_experiment_sample_area,
    mat_has_deformed_configuration,
    sync_mesh_bounds_from_file,
)
from vfmstudio.core.jobs import reduced_chi_square
from vfmstudio.core.document import BoundsModel, parameter_unit_plain
from vfmstudio.core.free_mesh import (
    CONFORM_MESSAGE,
    ConformError,
    construct_free_quad_mesh,
    inspect_free_points,
)
from vfmstudio.core.drag_mesh import DRAG_HINT
from vfmstudio.core.parameter_space import (
    default_axis_labels,
    group_statistics,
    space_from_document,
)
from vfmstudio.core.step_series import resolve_step_data_file, step_display_label
from vfmstudio.core.stress_space import (
    ALL_STEPS,
    available_resultants,
    color_choices,
    default_plot_axes,
    list_stress_space_targets,
    resolve_axis_key,
    resolve_color_key,
    resolve_stress_space_target,
)
from vfmstudio.core.uncertainty import default_baseline_flags
from vfmstudio.gui.results.experiment_check import collect_experiment_steps
from vfmstudio.gui.results.plot_export import has_exportable_plot
from vfmstudio.gui.results.plot_style import experiment_colour
from vfmstudio.gui.results.table_style import format_number
from vfmstudio.gui.session import AppSession, Selection, is_post_kind, is_uncertainty_kind
from vfmstudio.gui.virtual_fields_view import (
    QUANTITY_ITEMS,
    normalize_quantity,
    normalize_view_mode,
)
from vfmstudio.gui.widgets.color_swatch import ColorSwatchButton, LINE_PALETTE
from vfmstudio.gui.widgets.load_table import AppliedLoadEditor, fill_blank_external_values

_BC_ADD_LABELS = (
    ("prescribed", "Prescribed"),
    ("linking", "Linking"),
    ("external", "External"),
)
_BC_ADD_TIPS = {
    "prescribed": "Add a prescribed (essential) boundary condition.",
    "linking": "Add a linking condition that ties a DOF along a cut.",
    "external": "Add an external-work group (applied force or moment).",
}


def _block_set_text(widget: QLineEdit, text: str) -> None:
    widget.blockSignals(True)
    widget.setText(text)
    widget.blockSignals(False)


def _block_set_combo(widget: QComboBox, text: str) -> None:
    widget.blockSignals(True)
    widget.setCurrentText(text)
    widget.blockSignals(False)


def _block_set_combo_data(widget: QComboBox, data) -> None:
    index = widget.findData(data)
    if index < 0:
        index = 0
    widget.blockSignals(True)
    widget.setCurrentIndex(index)
    widget.blockSignals(False)


_DEFORM_SCALE_MAX = 50.0
_DEFORM_SCALE_TICKS = 10

_UQ_PLOT_MODES = {
    "post_uncertainty_cov": [
        "Identified θ ± σ",
        "Uncertainty ladder",
        "Robust covariance",
        "Working covariance",
        "Correlation",
        "Misfit spectrum",
    ],
    "post_uncertainty_experiments": [
        "Score norms",
        "Score components",
        "Sandwich anatomy",
    ],
    "post_uncertainty_ab": ["A and B", "A (sensitivity)", "B (variability)"],
    "post_uncertainty_info": [
        "cond(A − Aᵢ)",
        "Share of A",
        "Eigenvalues of A",
        "Constrained combinations",
    ],
    "post_uncertainty_loo": [
        "θ without each experiment",
        "Precision contribution U",
        "Influence I",
        "U versus I",
    ],
}


def _block_set_spin(widget: QSpinBox | QDoubleSpinBox, value: float) -> None:
    widget.blockSignals(True)
    widget.setValue(value)
    widget.blockSignals(False)


def _block_set_slider(widget: QSlider, ticks: int) -> None:
    widget.blockSignals(True)
    widget.setValue(int(ticks))
    widget.blockSignals(False)


def _block_set_checked(widget: QCheckBox, checked: bool) -> None:
    widget.blockSignals(True)
    widget.setChecked(checked)
    widget.blockSignals(False)


class _ContentPage(QWidget):
    """Inner page whose hint is the form's full size, so the scroll area can overflow."""

    def sizeHint(self):
        layout = self.layout()
        if layout is not None:
            return layout.sizeHint()
        return super().sizeHint()

    def minimumSizeHint(self):
        layout = self.layout()
        if layout is not None:
            return layout.minimumSize()
        return super().minimumSizeHint()


class PropertyEditorDock(QWidget):
    document_changed = Signal()
    view_changed = Signal()
    deformation_scale_changed = Signal(float, bool)
    request_add_bc = Signal(int, str)
    request_delete_bc = Signal(int, str, int)
    request_add_step = Signal(int)
    request_visualize_steps = Signal()
    request_uncertainty = Signal()
    request_loo_refit = Signal()
    request_export_plot = Signal()
    request_pick_nodes = Signal(bool)
    request_clear_bc_nodes = Signal()
    request_free_mesh_mode = Signal(bool)
    request_drag_mesh_mode = Signal(bool)
    request_clear_free_mesh = Signal()
    request_clear_mesh = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._session: Optional[AppSession] = None
        self._shown_key = None
        self._theta_spins: list[QDoubleSpinBox] = []
        self._theta_labels: list[QLabel] = []
        self._bound_spins: dict[str, QDoubleSpinBox] = {}
        self._deform_sliders: list[QSlider] = []
        self._deform_spins: list[QDoubleSpinBox] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self._title = QLabel("Properties")
        self._title.setStyleSheet("font-weight: 600; background: none;")
        layout.addWidget(self._title)
        self._stack = QStackedWidget()
        layout.addWidget(self._stack, 1)
        self._pages: dict[str, QWidget] = {}
        self._export_buttons: list[QPushButton] = []
        self._syncing = False
        self._build_pages()
        for label in self.findChildren(QLabel):
            label.setAutoFillBackground(False)

    def _add_page(self, name: str) -> QVBoxLayout:
        page = _ContentPage()
        page.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        column = QVBoxLayout(page)
        column.setContentsMargins(0, 0, 4, 8)
        column.setSpacing(8)
        column.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll = QScrollArea()
        scroll.setObjectName("propertyScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        scroll.setWidget(page)
        self._pages[name] = scroll
        self._stack.addWidget(scroll)
        return column

    def _section(self, column: QVBoxLayout, title: str) -> QFormLayout:
        box = QGroupBox(title)
        box.setObjectName("propertySection")
        form = QFormLayout(box)
        form.setContentsMargins(8, 14, 8, 8)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(6)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setLabelAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        column.addWidget(box)
        return form

    def _hint(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setProperty("role", "hint")
        label.setAutoFillBackground(False)
        return label

    def _compact(self, button: QPushButton) -> QPushButton:
        button.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        button.setProperty("role", "compact")
        return button

    def _button_row(self, *buttons: QPushButton) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        for button in buttons:
            self._compact(button)
            layout.addWidget(button, 0)
        layout.addStretch(1)
        return row

    def _combo_action_row(self, combo: QWidget, button: QPushButton) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(combo, 1)
        self._compact(button)
        layout.addWidget(button, 0)
        return row

    def _make_export_button(self) -> QPushButton:
        button = QPushButton("Export")
        button.setToolTip(
            "Save the current plot as PDF (vector, Type-42 fonts) or PNG (600 dpi)."
        )
        button.clicked.connect(self.request_export_plot.emit)
        self._compact(button)
        self._export_buttons.append(button)
        return button

    def _add_export_button(self, column: QVBoxLayout) -> QPushButton:
        button = self._make_export_button()
        column.addWidget(self._button_row(button))
        return button

    def _finish_page(self, column: QVBoxLayout) -> None:
        column.addStretch(1)

    def _build_pages(self) -> None:
        project = self._add_page("project")
        identity = self._section(project, "Project")
        self._proj_desc = QLineEdit()
        self._proj_desc.editingFinished.connect(self._on_project_desc)
        identity.addRow("Description", self._proj_desc)
        self._finish_page(project)

        material = self._add_page("material")
        model = self._section(material, "Constitutive model")
        self._mat_type = QComboBox()
        self._mat_type.addItems(["iso", "orthotropic", "anisotropic"])
        self._mat_type.currentTextChanged.connect(self._on_material_type)
        self._mat_analysis = QComboBox()
        self._mat_analysis.addItems(["in_plane", "out_of_plane", "both"])
        self._mat_analysis.currentTextChanged.connect(self._on_analysis_type)
        model.addRow("Material", self._mat_type)
        model.addRow("Analysis", self._mat_analysis)
        guess = self._section(material, "Initial parameters")
        for _ in range(8):
            spin = QDoubleSpinBox()
            spin.setDecimals(6)
            spin.setRange(-1e12, 1e12)
            spin.editingFinished.connect(self._on_theta_finished)
            label = QLabel()
            self._theta_spins.append(spin)
            self._theta_labels.append(label)
            guess.addRow(label, spin)
        self._finish_page(material)

        solver = self._add_page("solver")
        fgls = self._section(solver, "FGLS")
        self._sol_iters = QSpinBox()
        self._sol_iters.setRange(1, 500)
        self._sol_iters.valueChanged.connect(self._on_solver_iters)
        self._sol_tol = QDoubleSpinBox()
        self._sol_tol.setDecimals(12)
        self._sol_tol.setRange(0.0, 1.0)
        self._sol_tol.editingFinished.connect(self._on_solver_tol)
        self._sol_degree = QSpinBox()
        self._sol_degree.setRange(0, 20)
        self._sol_degree.valueChanged.connect(self._on_solver_degree)
        self._sol_weight = QComboBox()
        self._sol_weight.addItems(["gls", "block_diagonal", "diagonal", "ols"])
        self._sol_weight.currentTextChanged.connect(self._on_solver_weight)
        self._sol_spatial = QCheckBox()
        self._sol_spatial.setMinimumSize(18, 18)
        self._sol_spatial.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        self._sol_spatial.setToolTip("Apply a spatial correction to the Γ covariance.")
        self._sol_spatial.toggled.connect(self._on_solver_spatial)
        self._sol_cov = QDoubleSpinBox()
        self._sol_cov.setDecimals(12)
        self._sol_cov.setRange(0.0, 1.0)
        self._sol_cov.editingFinished.connect(self._on_solver_cov)
        fgls.addRow("Max iterations", self._sol_iters)
        fgls.addRow("Tolerance", self._sol_tol)
        fgls.addRow("Γ basis degree", self._sol_degree)
        fgls.addRow("Weighting", self._sol_weight)
        fgls.addRow("Covariance tolerance", self._sol_cov)
        fgls.addRow("Spatial correction of Γ", self._sol_spatial)
        preflight = self._section(solver, "Pre-flight")
        self._sol_check = QLabel("—")
        self._sol_check.setWordWrap(True)
        self._sol_check.setProperty("role", "hint")
        preflight.addRow(self._sol_check)
        self._finish_page(solver)

        experiment = self._add_page("experiment")
        specimen = self._section(experiment, "Specimen")
        self._exp_name = QLineEdit()
        self._exp_name.editingFinished.connect(self._on_exp_name)
        self._exp_angle = QDoubleSpinBox()
        self._exp_angle.setRange(-180.0, 180.0)
        self._exp_angle.setDecimals(2)
        self._exp_angle.editingFinished.connect(self._on_exp_angle)
        self._exp_area = QDoubleSpinBox()
        self._exp_area.setDecimals(6)
        self._exp_area.setRange(0.0, 1e12)
        self._exp_area.setToolTip(
            "Total specimen area used as ΔS = A / N in the measurement-point "
            "integration. Prefill is n × (typical Delaunay cell area) from the "
            "DIC cloud, so concavities and holes are not filled by the bounding "
            "box. Edit to override; From DIC recomputes from the load-step file."
        )
        self._exp_area.editingFinished.connect(self._on_exp_area)
        self._exp_area_estimate = QPushButton("From DIC")
        self._exp_area_estimate.setToolTip(
            "Recompute sample area from the DIC point cloud "
            "(filtered Delaunay, n × 2 × median triangle area)."
        )
        self._exp_area_estimate.clicked.connect(self._on_exp_area_estimate)
        specimen.addRow("Name", self._exp_name)
        specimen.addRow("Material φ (deg)", self._exp_angle)
        specimen.addRow(
            "Sample area",
            self._combo_action_row(self._exp_area, self._exp_area_estimate),
        )
        self._exp_fit_axes = QCheckBox()
        self._exp_fit_axes.setMinimumSize(18, 18)
        self._exp_fit_axes.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        self._exp_fit_axes.setToolTip(
            "Rotate the DIC cloud by the smallest angle that lines the specimen "
            "up with the X/Y axes (typical 2–3° camera tilt)."
        )
        self._exp_fit_axes.toggled.connect(self._on_exp_fit_axes)
        specimen.addRow("Fit data to system axes", self._exp_fit_axes)
        self._exp_include = QCheckBox()
        self._exp_include.setMinimumSize(18, 18)
        self._exp_include.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        self._exp_include.setToolTip(
            "When ticked, this coupon enters FGLS Identify. "
            "The same flag is used under Coefficient Evolution."
        )
        self._exp_include.toggled.connect(self._on_exp_include)
        specimen.addRow("Include in identification", self._exp_include)
        display = self._section(experiment, "Display")
        self._exp_show_deformed = QCheckBox()
        self._exp_show_deformed.setMinimumSize(18, 18)
        self._exp_show_deformed.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        self._exp_show_deformed.setToolTip(
            "Off: draw a flat plate (X, Y, 0). On: warp to X+U, Y+V, Z+W "
            "when the load-step MAT file contains matching displacement fields."
        )
        self._exp_show_deformed.toggled.connect(self._on_exp_show_deformed)
        display.addRow("Show deformed configuration", self._exp_show_deformed)
        deform_row, self._exp_deform_slider, self._exp_deform_spin = (
            self._make_deform_scale_row()
        )
        display.addRow("Deformation scale", deform_row)
        experiment.addWidget(
            self._hint(
                "Mesh, boundary conditions, load steps, and the analysis window "
                "each have their own tree entry."
            )
        )
        self._finish_page(experiment)

        mesh = self._add_page("mesh")
        structured = self._section(mesh, "Structured mesh")
        self._mesh_nx = QSpinBox()
        self._mesh_nx.setRange(1, 200)
        self._mesh_nx.editingFinished.connect(self._on_mesh_nx)
        self._mesh_ny = QSpinBox()
        self._mesh_ny.setRange(1, 200)
        self._mesh_ny.editingFinished.connect(self._on_mesh_ny)
        structured.addRow("Elements Nx", self._mesh_nx)
        structured.addRow("Elements Ny", self._mesh_ny)
        for attr, title in (
            ("x_min", "x min"),
            ("x_max", "x max"),
            ("y_min", "y min"),
            ("y_max", "y max"),
        ):
            spin = QDoubleSpinBox()
            spin.setDecimals(6)
            spin.setRange(-1e6, 1e6)
            spin.editingFinished.connect(lambda a=attr: self._on_mesh_bound(a))
            self._bound_spins[attr] = spin
            structured.addRow(title, spin)
        self._mesh_fit = QPushButton("Fit to DIC")
        self._mesh_fit.setToolTip(
            "Set the mesh bounding box from the first load-step MAT file."
        )
        self._mesh_fit.clicked.connect(self._on_mesh_fit)
        self._mesh_clear = QPushButton("Clear")
        self._mesh_clear.setToolTip(
            "Delete this experiment's mesh and every boundary condition "
            "that depends on it."
        )
        self._mesh_clear.clicked.connect(self.request_clear_mesh.emit)
        structured.addRow(self._button_row(self._mesh_fit, self._mesh_clear))

        occupancy = self._section(mesh, "DIC occupancy")
        self._mesh_show_counts = QCheckBox()
        self._mesh_show_counts.setMinimumSize(18, 18)
        self._mesh_show_counts.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        self._mesh_show_counts.setToolTip(
            "Annotate each element with how many DIC points map into it "
            "(same containment rule as the identification assembly). "
            "Empty elements are shown in red."
        )
        self._mesh_show_counts.toggled.connect(self._on_mesh_show_counts)
        occupancy.addRow("Show points per element", self._mesh_show_counts)
        self._mesh_count_summary = QLabel("—")
        self._mesh_count_summary.setWordWrap(True)
        self._mesh_count_summary.setProperty("role", "hint")
        occupancy.addRow(self._mesh_count_summary)

        tools = self._section(mesh, "Edit tools")
        self._mesh_free = QPushButton("Free mesh")
        self._mesh_free.setCheckable(True)
        self._mesh_free.setToolTip(
            "Place nodes in a locked top view, then construct an "
            "axis-aligned quadrilateral mesh."
        )
        self._mesh_free.toggled.connect(self._on_free_mesh_toggled)
        self._mesh_drag = QPushButton("Drag mesh")
        self._mesh_drag.setCheckable(True)
        self._mesh_drag.setToolTip(
            "Select a mesh node or element edge and drag the whole "
            "connected grid line. Use this to push an edge until it "
            "matches the specimen."
        )
        self._mesh_drag.toggled.connect(self._on_drag_mesh_toggled)
        tools.addRow(self._button_row(self._mesh_free, self._mesh_drag))
        self._mesh_construct = QPushButton("Construct")
        self._mesh_construct.setToolTip(
            "Build a conformal quadrilateral mesh from the placed nodes."
        )
        self._mesh_construct.clicked.connect(self._on_construct_free_mesh)
        self._mesh_clear_points = QPushButton("Clear points")
        self._mesh_clear_points.clicked.connect(self._on_clear_free_mesh_points)
        self._mesh_free_actions = self._button_row(
            self._mesh_construct, self._mesh_clear_points
        )
        tools.addRow(self._mesh_free_actions)
        self._mesh_free_hint = QLabel("Click in the viewport to add nodes.")
        self._mesh_free_hint.setWordWrap(True)
        self._mesh_free_hint.setProperty("role", "hint")
        tools.addRow(self._mesh_free_hint)
        self._mesh_drag_hint = QLabel(DRAG_HINT)
        self._mesh_drag_hint.setWordWrap(True)
        self._mesh_drag_hint.setProperty("role", "hint")
        tools.addRow(self._mesh_drag_hint)
        self._mesh_construct.hide()
        self._mesh_clear_points.hide()
        self._mesh_free_actions.hide()
        self._mesh_free_hint.hide()
        self._mesh_drag_hint.hide()
        self._finish_page(mesh)

        bcs = self._add_page("bcs")
        bcs.addWidget(
            self._hint(
                "Create conditions here. Select one in the tree to edit "
                "its DOF, region, and values — those fields live only on "
                "the individual condition."
            )
        )
        create = self._section(bcs, "Add condition")
        self._add_bc_buttons = {}
        add_buttons = []
        for kind, label in _BC_ADD_LABELS:
            button = QPushButton(label)
            button.setToolTip(_BC_ADD_TIPS[kind])
            button.clicked.connect(lambda checked=False, k=kind: self._on_add_bc(k))
            self._add_bc_buttons[kind] = button
            add_buttons.append(button)
        create.addRow(self._button_row(*add_buttons))
        copy = self._section(bcs, "Copy from experiment")
        self._bc_copy_from = QComboBox()
        self._bc_copy = QPushButton("Copy")
        self._bc_copy.setToolTip("Replace this experiment's boundary conditions.")
        self._bc_copy.clicked.connect(self._on_copy_bcs)
        copy.addRow(self._combo_action_row(self._bc_copy_from, self._bc_copy))
        self._bc_summary = self._hint("—")
        bcs.addWidget(self._bc_summary)
        self._finish_page(bcs)

        bc = self._add_page("bc")
        definition = self._section(bc, "Definition")
        self._bc_label = QLineEdit()
        self._bc_label.editingFinished.connect(self._on_bc_label)
        self._bc_dof = QComboBox()
        self._bc_dof.addItems(["W", "W_X", "W_Y", "W_XY", "U", "V"])
        self._bc_dof.currentTextChanged.connect(self._on_bc_dof)
        definition.addRow("Label", self._bc_label)
        definition.addRow("DOF", self._bc_dof)
        region = self._section(bc, "Region")
        self._bc_edge = QComboBox()
        self._bc_edge.addItems(["", "left", "right", "top", "bottom"])
        self._bc_edge.currentTextChanged.connect(self._on_bc_edge)
        region.addRow("Named edge", self._bc_edge)
        self._bc_nodes_hint = QLabel("Nodes: (named edge)")
        self._bc_nodes_hint.setWordWrap(True)
        self._bc_nodes_hint.setProperty("role", "hint")
        region.addRow(self._bc_nodes_hint)
        self._bc_pick_nodes = QPushButton("Pick nodes")
        self._bc_pick_nodes.setCheckable(True)
        self._bc_pick_nodes.toggled.connect(self.request_pick_nodes.emit)
        self._bc_clear_nodes = QPushButton("Clear")
        self._bc_clear_nodes.clicked.connect(self.request_clear_bc_nodes.emit)
        region.addRow(self._button_row(self._bc_pick_nodes, self._bc_clear_nodes))
        self._bc_load_form = self._section(bc, "External work")
        self._bc_load_box = self._bc_load_form.parentWidget()
        self._bc_load = QComboBox()
        self._bc_load.addItems(["distributed", "concentrated"])
        self._bc_load.currentTextChanged.connect(self._on_bc_load)
        self._bc_load_form.addRow("Load type", self._bc_load)
        self._bc_load_form.addRow(
            self._hint(
                "Which .mat F component drives this group is set under "
                "Load steps → Applied loads."
            )
        )
        self._bc_value_form = self._section(bc, "Prescribed value")
        self._bc_value_box = self._bc_value_form.parentWidget()
        self._bc_value = QDoubleSpinBox()
        self._bc_value.setDecimals(6)
        self._bc_value.setRange(-1e12, 1e12)
        self._bc_value.editingFinished.connect(self._on_bc_value)
        self._bc_value_form.addRow("Value", self._bc_value)
        self._bc_delete = QPushButton("Delete")
        self._bc_delete.setToolTip("Remove this boundary condition.")
        self._bc_delete.clicked.connect(self._on_delete_bc)
        bc.addWidget(self._button_row(self._bc_delete))
        self._finish_page(bc)

        step = self._add_page("step")
        self._step_fields = self._section(step, "This step").parentWidget()
        step_form = self._step_fields.layout()
        self._step_label = QLineEdit()
        self._step_label.editingFinished.connect(self._on_step_label)
        self._step_path_row = QWidget()
        path_layout = QHBoxLayout(self._step_path_row)
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.setSpacing(6)
        self._step_path = QLineEdit()
        self._step_path.editingFinished.connect(self._on_step_path)
        browse = QPushButton("…")
        browse.setFixedWidth(32)
        browse.setToolTip("Browse for a measurement MAT file.")
        browse.clicked.connect(self._on_step_browse)
        self._compact(browse)
        path_layout.addWidget(self._step_path, 1)
        path_layout.addWidget(browse, 0)
        self._step_label_caption = QLabel("Label")
        self._step_path_caption = QLabel("Data file")
        step_form.addRow(self._step_label_caption, self._step_label)
        step_form.addRow(self._step_path_caption, self._step_path_row)

        analysis = self._section(step, "Analysis window")
        self._analysis_start = QComboBox()
        self._analysis_start.currentIndexChanged.connect(self._on_analysis_start)
        self._analysis_end = QComboBox()
        self._analysis_end.currentIndexChanged.connect(self._on_analysis_end)
        self._analysis_reference = QComboBox()
        self._analysis_reference.setToolTip(
            "Kinematics (strain, curvature, displacement) and applied loads in "
            "the 3D view are shown relative to this step: current − reference. "
            "Identify uses the same increment. Choose None to display the "
            "stored files as-is."
        )
        self._analysis_reference.currentIndexChanged.connect(self._on_analysis_reference)
        analysis.addRow("Start step", self._analysis_start)
        analysis.addRow("End step", self._analysis_end)
        analysis.addRow("Reference step", self._analysis_reference)
        self._visualize_steps = QPushButton("Visualize")
        self._visualize_steps.setToolTip("Plot applied load against load-step data.")
        self._visualize_steps.clicked.connect(self.request_visualize_steps.emit)
        self._add_step_button = QPushButton("Add steps…")
        self._add_step_button.setToolTip(
            "Import MAT files or Vic-3D DIC CSVs plus synchronized machine data "
            "as load steps on this experiment."
        )
        self._add_step_button.clicked.connect(self._on_add_step)
        analysis.addRow(
            self._button_row(self._add_step_button, self._visualize_steps)
        )

        loads = self._section(step, "Applied loads")
        self._exp_source = QComboBox()
        self._exp_source.addItem(".mat F field", "mat")
        self._exp_source.addItem("this table", "manual")
        self._exp_source.setToolTip(
            "Where Identify reads the applied force/moment.\n"
            "• .mat F field — each step's own measurement file.\n"
            "• this table — the force/moment grid below (one cell per step "
            "and EXTERNAL group). Use this when F is missing or you need to "
            "override it."
        )
        self._exp_source.currentIndexChanged.connect(self._on_exp_source)
        self._exp_load_editor = AppliedLoadEditor()
        self._exp_load_editor.loads_changed.connect(self._on_load_table_changed)
        self._exp_load_editor.mapping_changed.connect(self._notify)
        self._step_load_editor = self._exp_load_editor
        loads.addRow("Source", self._exp_source)
        loads.addRow(self._exp_load_editor)
        self._finish_page(step)

        empty = self._add_page("empty")
        empty.addWidget(QLabel("No experiment selected."))
        self._finish_page(empty)

        post = self._add_page("post_processing")
        status = self._section(post, "Status")
        self._post_status = QLabel("No identification result.")
        self._post_status.setWordWrap(True)
        status.addRow(self._post_status)
        self._post_export = self._add_export_button(post)
        self._finish_page(post)

        experiment_check = self._add_page("post_experiment_check")
        self._exp_check_status = QLabel("No identification result.")
        self._exp_check_status.setWordWrap(True)
        experiment_check.addWidget(self._exp_check_status)
        tests = self._section(experiment_check, "Experiments")
        self._exp_check_scroll = QScrollArea()
        self._exp_check_scroll.setWidgetResizable(True)
        self._exp_check_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._exp_check_scroll.setMinimumHeight(160)
        self._exp_check_host = QWidget()
        self._exp_check_layout = QVBoxLayout(self._exp_check_host)
        self._exp_check_layout.setContentsMargins(0, 0, 0, 0)
        self._exp_check_layout.setSpacing(4)
        self._exp_check_scroll.setWidget(self._exp_check_host)
        tests.addRow(self._exp_check_scroll)
        view = self._section(experiment_check, "View")
        self._exp_check_mode = QComboBox()
        self._exp_check_mode.addItems(
            [
                "Coefficients vs load step",
                "One panel per experiment",
                "Deviation from θ̂",
            ]
        )
        self._exp_check_mode.currentIndexChanged.connect(self._on_post_plot_mode)
        view.addRow("Plot", self._exp_check_mode)
        self._exp_check_export = self._add_export_button(experiment_check)
        self._finish_page(experiment_check)

        ident_strength = self._add_page("post_identifiability_strength")
        self._ident_strength_status = self._hint(
            "The left plot always shows eq. (63) for every test. Untick tests "
            "to measure the information they add on the right (G_k)."
        )
        ident_strength.addWidget(self._ident_strength_status)
        ident_view = self._section(ident_strength, "View")
        self._ident_shift_scale = QComboBox()
        self._ident_shift_scale.addItem("Parameter units", "units")
        self._ident_shift_scale.addItem("Relative to |θ̂|", "relative")
        self._ident_shift_scale.currentIndexChanged.connect(self._on_ident_shift_scale)
        ident_view.addRow("Eq. (63) values", self._ident_shift_scale)
        baseline = self._section(ident_strength, "Baseline tests")
        self._ident_base_scroll = QScrollArea()
        self._ident_base_scroll.setWidgetResizable(True)
        self._ident_base_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._ident_base_scroll.setMinimumHeight(160)
        self._ident_base_host = QWidget()
        self._ident_base_layout = QVBoxLayout(self._ident_base_host)
        self._ident_base_layout.setContentsMargins(0, 0, 0, 0)
        self._ident_base_layout.setSpacing(4)
        self._ident_base_scroll.setWidget(self._ident_base_host)
        baseline.addRow(self._ident_base_scroll)
        self._ident_strength_export = self._add_export_button(ident_strength)
        self._finish_page(ident_strength)

        virtual_fields = self._add_page("post_virtual_fields")
        self._vf_status = QLabel(
            "Each tile is the optimal virtual field for one constitutive "
            "unknown. FGLS recomputes those fields at every load step of "
            "every experiment. Choose which experiment and step to plot."
        )
        self._vf_status.setWordWrap(True)
        virtual_fields.addWidget(self._vf_status)
        vf = self._section(virtual_fields, "View")
        self._vf_experiment = QComboBox()
        self._vf_experiment.currentIndexChanged.connect(self._on_vf_experiment)
        vf.addRow("Experiment", self._vf_experiment)
        self._vf_step = QComboBox()
        self._vf_step.currentIndexChanged.connect(self._on_vf_step)
        vf.addRow("Load step", self._vf_step)
        self._vf_mode = QComboBox()
        self._vf_mode.addItems(["3D surface", "2D contour"])
        self._vf_mode.currentIndexChanged.connect(self._on_vf_view)
        vf.addRow("View", self._vf_mode)
        self._vf_quantity = QComboBox()
        for text, _math in QUANTITY_ITEMS:
            self._vf_quantity.addItem(text, text)
        self._vf_quantity.currentIndexChanged.connect(self._on_vf_quantity)
        vf.addRow("Quantity", self._vf_quantity)
        self._vf_unknown = QComboBox()
        self._vf_unknown.currentIndexChanged.connect(self._on_vf_unknown)
        vf.addRow("3D unknown", self._vf_unknown)
        self._vf_export = self._add_export_button(virtual_fields)
        self._finish_page(virtual_fields)

        plots = self._add_page("post_plots")
        self._fit_tests_status = QLabel(
            "Tick the tests in the combination. After Identify, every "
            "non-empty subset is already available; ticking applies it instantly."
        )
        self._fit_tests_status.setWordWrap(True)
        plots.addWidget(self._fit_tests_status)
        fit = self._section(plots, "Tests")
        self._fit_tests_scroll = QScrollArea()
        self._fit_tests_scroll.setWidgetResizable(True)
        self._fit_tests_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._fit_tests_scroll.setMinimumHeight(160)
        self._fit_tests_host = QWidget()
        self._fit_tests_layout = QVBoxLayout(self._fit_tests_host)
        self._fit_tests_layout.setContentsMargins(0, 0, 0, 0)
        self._fit_tests_layout.setSpacing(4)
        self._fit_test_ids: list[int] = []
        self._fit_tests_scroll.setWidget(self._fit_tests_host)
        fit.addRow(self._fit_tests_scroll)
        self._fit_tests_all = QPushButton("All")
        self._fit_tests_all.clicked.connect(lambda: self._set_all_fit_tests(True))
        self._fit_tests_none = QPushButton("None")
        self._fit_tests_none.clicked.connect(lambda: self._set_all_fit_tests(False))
        self._plots_export = self._make_export_button()
        plots.addWidget(
            self._button_row(
                self._fit_tests_all, self._fit_tests_none, self._plots_export
            )
        )
        self._finish_page(plots)

        parameter_space = self._add_page("post_parameter_space")
        self._ps_status = QLabel(
            "Each point is one non-empty combination of the identified tests, "
            "using the virtual fields from Identify. The star is all tests."
        )
        self._ps_status.setWordWrap(True)
        parameter_space.addWidget(self._ps_status)
        axes = self._section(parameter_space, "Axes")
        self._ps_left_x = QComboBox()
        self._ps_left_y = QComboBox()
        self._ps_right_x = QComboBox()
        self._ps_right_y = QComboBox()
        self._ps_combos = (
            self._ps_left_x,
            self._ps_left_y,
            self._ps_right_x,
            self._ps_right_y,
        )
        for combo in self._ps_combos:
            combo.currentIndexChanged.connect(self._on_parameter_space_axis)
        axes.addRow("Left plot X", self._ps_left_x)
        axes.addRow("Left plot Y", self._ps_left_y)
        axes.addRow("Right plot X", self._ps_right_x)
        axes.addRow("Right plot Y", self._ps_right_y)
        groups = self._section(parameter_space, "Highlight groups")
        self._ps_groups_hint = self._hint(
            "Colour every combination that includes all ticked tests. "
            "Ticking more tests keeps the intersection, so fewer points "
            "stay highlighted. If several groups match, the stricter one wins."
        )
        groups.addRow(self._ps_groups_hint)
        self._ps_groups_scroll = QScrollArea()
        self._ps_groups_scroll.setWidgetResizable(True)
        self._ps_groups_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._ps_groups_scroll.setMinimumHeight(200)
        self._ps_groups_host = QWidget()
        self._ps_groups_layout = QVBoxLayout(self._ps_groups_host)
        self._ps_groups_layout.setContentsMargins(0, 0, 0, 0)
        self._ps_groups_layout.setSpacing(4)
        self._ps_groups_scroll.setWidget(self._ps_groups_host)
        groups.addRow(self._ps_groups_scroll)
        self._ps_add_group = QPushButton("Add group")
        self._ps_add_group.clicked.connect(self._on_ps_add_group)
        self._ps_export = self._make_export_button()
        parameter_space.addWidget(
            self._button_row(self._ps_add_group, self._ps_export)
        )
        self._ps_group_stat_labels: list[QLabel] = []
        self._finish_page(parameter_space)

        stress_space = self._add_page("post_stress_space")
        self._ss_status = QLabel(
            "Each point is one DIC subset. Resultants are reconstructed from "
            "the identified stiffness (N = A ε, M = D κ). Choose the test, "
            "load step, and which resultants appear on the 2D and 3D plots."
        )
        self._ss_status.setWordWrap(True)
        stress_space.addWidget(self._ss_status)
        ss = self._section(stress_space, "View")
        self._ss_experiment = QComboBox()
        self._ss_experiment.currentIndexChanged.connect(self._on_ss_experiment)
        ss.addRow("Experiment", self._ss_experiment)
        self._ss_step = QComboBox()
        self._ss_step.currentIndexChanged.connect(self._on_ss_step)
        ss.addRow("Load step", self._ss_step)
        self._ss_2d_x = QComboBox()
        self._ss_2d_y = QComboBox()
        self._ss_3d_x = QComboBox()
        self._ss_3d_y = QComboBox()
        self._ss_3d_z = QComboBox()
        self._ss_color = QComboBox()
        self._ss_axis_combos = (
            self._ss_2d_x,
            self._ss_2d_y,
            self._ss_3d_x,
            self._ss_3d_y,
            self._ss_3d_z,
        )
        for combo in self._ss_axis_combos:
            combo.currentIndexChanged.connect(self._on_ss_axis)
        self._ss_color.currentIndexChanged.connect(self._on_ss_axis)
        ss.addRow("Left plot X", self._ss_2d_x)
        ss.addRow("Left plot Y", self._ss_2d_y)
        ss.addRow("Right plot X", self._ss_3d_x)
        ss.addRow("Right plot Y", self._ss_3d_y)
        ss.addRow("Right plot Z", self._ss_3d_z)
        ss.addRow("Colour", self._ss_color)
        self._ss_export = self._add_export_button(stress_space)
        self._finish_page(stress_space)

        uq = self._add_page("uncertainty")
        self._uq_status = QLabel("No uncertainty analysis.")
        self._uq_status.setWordWrap(True)
        uq.addWidget(self._uq_status)
        uq_view = self._section(uq, "View")
        self._uq_kind = QComboBox()
        self._uq_kind.addItems(["CR2", "CR3", "CR0"])
        self._uq_kind.currentTextChanged.connect(self._on_uq_kind)
        uq_view.addRow("Sandwich correction", self._uq_kind)
        self._uq_plot_mode = QComboBox()
        self._uq_plot_mode.currentIndexChanged.connect(self._on_post_plot_mode)
        uq_view.addRow("View", self._uq_plot_mode)
        self._uq_recompute = QPushButton("Recompute")
        self._uq_recompute.clicked.connect(self.request_uncertainty.emit)
        self._uq_refit = QPushButton("Refit FGLS")
        self._uq_refit.setToolTip("Refit FGLS after leaving each experiment out.")
        self._uq_refit.clicked.connect(self.request_loo_refit.emit)
        self._uq_export = self._make_export_button()
        uq.addWidget(
            self._button_row(self._uq_recompute, self._uq_refit, self._uq_export)
        )
        self._finish_page(uq)


    def _page_name(self, kind: str) -> str:
        if is_uncertainty_kind(kind):
            return "uncertainty"
        if kind == "post_plots":
            return "post_plots"
        if kind == "post_parameter_space":
            return "post_parameter_space"
        if kind == "post_stress_space":
            return "post_stress_space"
        if kind == "post_experiment_check":
            return "post_experiment_check"
        if kind == "post_identifiability_strength":
            return "post_identifiability_strength"
        if kind == "post_virtual_fields":
            return "post_virtual_fields"
        if is_post_kind(kind):
            return "post_processing"
        if kind in {"project", "experiments"}:
            return "project"
        if kind == "experiment":
            return "experiment"
        if kind in {"bcs", "bc_group"}:
            return "bcs"
        if kind in {"step", "steps"}:
            return "step"
        if kind in self._pages:
            return kind
        return "project"

    def _notify(self) -> None:
        if self._session is not None:
            self._session.mark_dirty()
        self.document_changed.emit()

    def show_selection(self, session: AppSession, selection: Optional[Selection] = None) -> None:
        selection = selection or session.selection
        key = (
            selection.kind,
            selection.experiment_id,
            selection.step_index,
            selection.bc_kind,
            selection.bc_index,
        )
        page_name = self._page_name(selection.kind)
        self._session = session
        titles = {
            "project": "Project",
            "material": "Material",
            "solver": "Solver",
            "experiment": "Experiment",
            "bcs": "Boundary conditions",
            "bc_group": "Boundary conditions",
            "mesh": "Mesh",
            "bc": "Boundary condition",
            "step": "Load step",
            "steps": "Load steps",
            "empty": "Properties",
            "post_processing": "Post Processing",
            "post_plots": "Coefficient Evolution",
            "post_parameter_space": "Parameter space",
            "post_stress_space": "Stress resultant space",
            "post_experiment_check": "Individual Experiment check",
            "post_identifiability": "Identifiability",
            "post_identifiability_strength": "Identifiability strength",
            "post_virtual_fields": "Virtual fields",
            "post_uncertainty": "Uncertainty quantification",
            "post_uncertainty_cov": "Cov(θ)",
            "post_uncertainty_experiments": "Experiment uncertainty",
            "post_uncertainty_ab": "Matrices A and B",
            "post_uncertainty_info": "Information geometry",
            "post_uncertainty_loo": "Leave-one-out",
            "uncertainty": "Uncertainty quantification",
        }
        if page_name in {"experiment", "mesh", "bc", "bcs", "step"} and session.document.experiment_by_id(selection.experiment_id or 0) is None:
            page_name = "empty"
        title_key = selection.kind if selection.kind in titles else page_name
        title = titles.get(title_key, "Properties")
        if selection.kind == "bc_group" and selection.bc_kind:
            title = f"{str(selection.bc_kind).title()} conditions"
        elif selection.kind == "bc" and selection.bc_kind:
            title = f"{str(selection.bc_kind).title()} condition"
        self._title.setText(title)
        self._syncing = True
        try:
            self._fill_page(page_name, session, selection)
            self._sync_export_buttons(selection.kind)
            current = self._pages[page_name]
            previous = self._stack.currentWidget()
            self._stack.setCurrentWidget(current)
            if page_name == "material":
                self._fill_material(session)
        finally:
            self._syncing = False
        if current is not previous and isinstance(current, QScrollArea):
            current.verticalScrollBar().setValue(0)
        self._shown_key = key

    def _sync_export_buttons(self, kind: str) -> None:
        show = has_exportable_plot(kind)
        if hasattr(self, "_post_export"):
            self._post_export.setVisible(show)
        if hasattr(self, "_vf_export"):
            self._vf_export.setVisible(kind == "post_virtual_fields")
        if hasattr(self, "_uq_export"):
            self._uq_export.setVisible(show and is_uncertainty_kind(kind))

    def _fill_page(self, page_name: str, session: AppSession, selection: Selection) -> None:
        if page_name == "project":
            _block_set_text(self._proj_desc, session.document.description)
        elif page_name == "material":
            self._fill_material(session)
        elif page_name == "solver":
            self._fill_solver(session)
        elif page_name == "experiment":
            self._fill_experiment(session, selection.experiment_id)
        elif page_name == "mesh":
            self._fill_mesh(session, selection.experiment_id)
        elif page_name == "bcs":
            self._fill_bcs(session, selection)
        elif page_name == "bc":
            self._fill_bc(session, selection)
        elif page_name == "step":
            self._fill_step(session, selection)
        elif page_name == "post_processing":
            self._fill_post_processing(session)
        elif page_name == "post_plots":
            self._fill_fit_tests(session)
        elif page_name == "post_parameter_space":
            self._fill_parameter_space(session)
        elif page_name == "post_stress_space":
            self._fill_stress_space(session)
        elif page_name == "post_experiment_check":
            self._fill_experiment_check(session)
        elif page_name == "post_identifiability_strength":
            self._fill_identifiability_strength(session)
        elif page_name == "post_virtual_fields":
            self._fill_virtual_fields(session)
        elif page_name == "uncertainty":
            self._fill_uncertainty(session, selection)

    def _fill_post_processing(self, session: AppSession) -> None:
        if session.result is None:
            self._post_status.setText("No identification result.")
            return
        chi = reduced_chi_square(session.result)
        if chi:
            self._post_status.setText(
                f"χ²/dof = {chi['reduced']:.2f}  (Birge {chi['birge']:.2f})"
            )
        else:
            self._post_status.setText("Identification complete.")

    def _fill_virtual_fields(self, session: AppSession) -> None:
        from vfmstudio.core.virtual_fields import (
            list_virtual_field_targets,
            resolve_virtual_field_target,
        )

        targets = list_virtual_field_targets(session.document, session.result)
        if session.result is None:
            self._vf_status.setText("Run Identify. Virtual fields are computed with FGLS.")
        elif not targets:
            self._vf_status.setText(
                "This result has no per-experiment virtual fields. "
                "Each tile is one constitutive unknown from the stored Identify fields."
            )
        else:
            n_unknowns = len(getattr(session.result, "optimal_field_results", None) or [])
            extra = (
                f" This identification has {n_unknowns} unknown"
                f"{'s' if n_unknowns != 1 else ''}."
                if n_unknowns
                else ""
            )
            self._vf_status.setText(
                "Each tile is the optimal virtual field for one constitutive "
                "unknown. FGLS recomputes those fields at every load step of "
                "every experiment. Choose which experiment and step to plot."
                + extra
            )
        target = resolve_virtual_field_target(
            session.document,
            session.result,
            session.vf_experiment_id,
            session.vf_step_index,
        )
        if target is not None:
            session.vf_experiment_id = target.experiment_id
            session.vf_step_index = target.step_index
        experiments = []
        seen = set()
        for item in targets:
            if item.experiment_id in seen:
                continue
            seen.add(item.experiment_id)
            experiments.append((item.experiment_id, item.experiment_name))
        self._vf_experiment.blockSignals(True)
        self._vf_experiment.clear()
        for experiment_id, name in experiments:
            self._vf_experiment.addItem(name, experiment_id)
        if target is not None:
            index = self._vf_experiment.findData(target.experiment_id)
            if index >= 0:
                self._vf_experiment.setCurrentIndex(index)
        self._vf_experiment.blockSignals(False)
        self._vf_experiment.setEnabled(bool(experiments))
        session.vf_view_mode = normalize_view_mode(session.vf_view_mode)
        session.vf_quantity = normalize_quantity(session.vf_quantity)
        _block_set_combo(self._vf_mode, session.vf_view_mode)
        _block_set_combo(self._vf_quantity, session.vf_quantity)
        self._fill_vf_unknowns(session)
        self._refill_vf_steps(session, targets, emit=False)

    def _refill_vf_steps(self, session: AppSession, targets=None, emit: bool = True) -> None:
        from vfmstudio.core.virtual_fields import list_virtual_field_targets

        if targets is None:
            targets = list_virtual_field_targets(session.document, session.result)
        experiment_id = session.vf_experiment_id
        steps = [
            item
            for item in targets
            if experiment_id is None or item.experiment_id == experiment_id
        ]
        self._vf_step.blockSignals(True)
        self._vf_step.clear()
        for item in steps:
            self._vf_step.addItem(item.step_label, item.step_index)
        wanted = session.vf_step_index
        index = self._vf_step.findData(wanted)
        if index < 0:
            index = 0
        if self._vf_step.count():
            self._vf_step.setCurrentIndex(index)
            session.vf_step_index = int(self._vf_step.currentData())
        else:
            session.vf_step_index = None
        self._vf_step.blockSignals(False)
        self._vf_step.setEnabled(self._vf_step.count() > 0)
        if emit:
            self.view_changed.emit()

    def _on_vf_experiment(self, _index: int = 0) -> None:
        if self._session is None:
            return
        data = self._vf_experiment.currentData()
        self._session.vf_experiment_id = None if data is None else int(data)
        self._session.vf_step_index = 0
        self._refill_vf_steps(self._session, emit=True)

    def _on_vf_step(self, _index: int = 0) -> None:
        if self._session is None:
            return
        data = self._vf_step.currentData()
        self._session.vf_step_index = None if data is None else int(data)
        self.view_changed.emit()

    def _on_vf_view(self, _index: int = 0) -> None:
        if self._session is None:
            return
        self._session.vf_view_mode = normalize_view_mode(self._vf_mode.currentText())
        self.view_changed.emit()

    def _on_vf_quantity(self, _index: int = 0) -> None:
        if self._session is None:
            return
        data = self._vf_quantity.currentData()
        self._session.vf_quantity = normalize_quantity(
            str(data or self._vf_quantity.currentText() or "w*")
        )
        self.view_changed.emit()

    def _fill_vf_unknowns(self, session: AppSession) -> None:
        from vfmstudio.core.virtual_fields import parameter_labels_for

        labels = parameter_labels_for(session.result, session.document) if session.result else []
        self._vf_unknown.blockSignals(True)
        self._vf_unknown.clear()
        for index, name in enumerate(labels):
            self._vf_unknown.addItem(name, index)
        wanted = int(session.vf_parameter_index or 0)
        if wanted >= self._vf_unknown.count():
            wanted = 0
        if self._vf_unknown.count():
            self._vf_unknown.setCurrentIndex(wanted)
            session.vf_parameter_index = int(self._vf_unknown.currentData())
        self._vf_unknown.blockSignals(False)
        self._vf_unknown.setEnabled(self._vf_unknown.count() > 0)

    def _on_vf_unknown(self, _index: int = 0) -> None:
        if self._session is None:
            return
        data = self._vf_unknown.currentData()
        self._session.vf_parameter_index = 0 if data is None else int(data)
        self._session.show_virtual_field = True
        self.view_changed.emit()

    def _on_post_plot_mode(self, _index: int = 0) -> None:
        if self._session is None:
            return
        kind = self._session.selection.kind
        if kind == "post_experiment_check":
            text = self._exp_check_mode.currentText()
        else:
            text = self._uq_plot_mode.currentText()
        if text:
            self._session.plot_modes[kind] = text
        self.view_changed.emit()

    def _clear_fit_test_rows(self) -> None:
        while self._fit_tests_layout.count():
            item = self._fit_tests_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _fill_fit_tests(self, session: AppSession) -> None:
        experiments = session.document.experiments
        if not experiments:
            self._clear_fit_test_rows()
            self._fit_test_ids = []
            self._fit_tests_status.setText(
                "Add experiments first. Tick the tests in the combination "
                "after Identify."
            )
            self._fit_tests_all.setEnabled(False)
            self._fit_tests_none.setEnabled(False)
            return
        self._fit_tests_all.setEnabled(True)
        self._fit_tests_none.setEnabled(True)
        ids = [int(experiment.experiment_id) for experiment in experiments]
        boxes = [
            self._fit_tests_layout.itemAt(index).widget()
            for index in range(self._fit_tests_layout.count())
            if isinstance(self._fit_tests_layout.itemAt(index).widget(), QCheckBox)
        ]
        if ids == getattr(self, "_fit_test_ids", None) and len(boxes) == len(experiments):
            for box, experiment in zip(boxes, experiments):
                name = experiment.name or f"Experiment {experiment.experiment_id}"
                n_steps = len(experiment.included_step_indices())
                step_text = "1 load step" if n_steps == 1 else f"{n_steps} load steps"
                box.blockSignals(True)
                box.setText(f"{name}  ({step_text})")
                box.setChecked(bool(experiment.include_in_identification))
                box.blockSignals(False)
            self._refresh_fit_tests_status()
            return
        self._clear_fit_test_rows()
        self._fit_test_ids = ids
        for experiment in experiments:
            name = experiment.name or f"Experiment {experiment.experiment_id}"
            n_steps = len(experiment.included_step_indices())
            step_text = "1 load step" if n_steps == 1 else f"{n_steps} load steps"
            box = QCheckBox(f"{name}  ({step_text})")
            box.setChecked(bool(experiment.include_in_identification))
            box.toggled.connect(
                lambda checked, eid=int(experiment.experiment_id): self._on_fit_test(
                    eid, checked
                )
            )
            self._fit_tests_layout.addWidget(box)
        self._fit_tests_layout.addStretch(1)
        self._refresh_fit_tests_status()

    def _refresh_fit_tests_status(self) -> None:
        if self._session is None:
            return
        experiments = self._session.document.experiments
        n_all = len(experiments)
        n_on = len(self._session.document.experiments_in_identification())
        if n_all == 0:
            return
        if n_on == 0:
            self._fit_tests_status.setText(
                "No tests selected. Tick at least one test to show a combination."
            )
            return
        if n_on == n_all:
            scope = "all tests"
        elif n_on == 1:
            scope = "1 test"
        else:
            scope = f"{n_on} of {n_all} tests"
        prefix = (
            f"Selected combination: {scope}. Each ticked test contributes all "
            "of its load steps (analysis window)."
        )
        from vfmstudio.core.parameter_space import estimate_combination

        estimate = estimate_combination(self._session.result, self._session.document)
        if self._session.result is None or not estimate.has_blocks:
            self._fit_tests_status.setText(
                prefix + " Identify to solve; every combination is then "
                "available by ticking tests."
            )
            return
        if estimate.theta is None:
            self._fit_tests_status.setText(
                prefix + " Those tests were not in the last Identify."
            )
            return
        if estimate.is_full:
            self._fit_tests_status.setText(
                prefix + " Showing the last Identify. Tick a subset to apply "
                "that combination instantly."
            )
            return
        self._fit_tests_status.setText(
            prefix + " Showing that combination instantly from the last Identify."
        )

    def _on_fit_test(self, experiment_id: int, checked: bool) -> None:
        if self._session is None:
            return
        experiment = self._session.document.experiment_by_id(int(experiment_id))
        if experiment is None:
            return
        experiment.include_in_identification = bool(checked)
        self._session.mark_dirty()
        try:
            self._refresh_fit_tests_status()
        except Exception:
            pass
        self.view_changed.emit()

    def _set_all_fit_tests(self, included: bool) -> None:
        if self._session is None:
            return
        for experiment in self._session.document.experiments:
            experiment.include_in_identification = bool(included)
        for index in range(self._fit_tests_layout.count()):
            widget = self._fit_tests_layout.itemAt(index).widget()
            if isinstance(widget, QCheckBox):
                widget.blockSignals(True)
                widget.setChecked(bool(included))
                widget.blockSignals(False)
        self._session.mark_dirty()
        self._refresh_fit_tests_status()
        self.view_changed.emit()

    def _clear_exp_check_rows(self) -> None:
        while self._exp_check_layout.count():
            item = self._exp_check_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _fill_experiment_check(self, session: AppSession) -> None:
        current = session.plot_modes.get("post_experiment_check") or self._exp_check_mode.currentText()
        _block_set_combo(self._exp_check_mode, current)
        session.plot_modes["post_experiment_check"] = self._exp_check_mode.currentText()
        self._clear_exp_check_rows()
        if session.result is None:
            self._exp_check_status.setText("Run Identify first.")
            return
        series, _, _ = collect_experiment_steps(session.result, session.document)
        if not series:
            self._exp_check_status.setText("This result has no per-load-step estimates.")
            return
        self._exp_check_status.setText(
            "Choose which tests are drawn, and the colour of each line."
        )
        for rec in series:
            experiment_id = int(rec.experiment_id)
            session.experiment_check_visible.setdefault(experiment_id, True)
            session.experiment_check_colors.setdefault(
                experiment_id, experiment_colour(rec.series_index)
            )
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)
            box = QCheckBox(rec.name)
            box.setChecked(bool(session.experiment_check_visible[experiment_id]))
            box.toggled.connect(
                lambda checked, eid=experiment_id: self._on_exp_check_visible(eid, checked)
            )
            swatch = ColorSwatchButton(session.experiment_check_colors[experiment_id])
            swatch.colorChanged.connect(
                lambda color, eid=experiment_id: self._on_exp_check_color(eid, color)
            )
            row_layout.addWidget(box, 1)
            row_layout.addWidget(swatch)
            self._exp_check_layout.addWidget(row)
        self._exp_check_layout.addStretch(1)

    def _clear_ident_base_rows(self) -> None:
        while self._ident_base_layout.count():
            item = self._ident_base_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _fill_identifiability_strength(self, session: AppSession) -> None:
        self._clear_ident_base_rows()
        if session.uncertainty is None:
            self._ident_strength_status.setText(
                "Information gain and deletion influence need Identify. "
                "Run Identify first."
            )
            return
        names = session.uncertainty.cluster_names
        ids = session.uncertainty.cluster_ids
        defaults = default_baseline_flags(names, ids)
        self._ident_strength_status.setText(
            "Left: eq. (63) for every test. Choose parameter units or "
            "(θ̂₍₋ᵢ₎ − θ̂)/|θ̂|. Untick tests below to measure the information "
            "they add on the right (G_k)."
        )
        scale = str(getattr(session, "identifiability_shift_scale", "units") or "units")
        if scale not in {"units", "relative"}:
            scale = "units"
        _block_set_combo_data(self._ident_shift_scale, scale)
        for name, experiment_id in zip(names, ids):
            eid = int(experiment_id)
            session.identifiability_baseline.setdefault(eid, defaults.get(eid, True))
            box = QCheckBox(str(name))
            box.setChecked(bool(session.identifiability_baseline[eid]))
            box.toggled.connect(
                lambda checked, experiment=eid: self._on_ident_baseline(experiment, checked)
            )
            self._ident_base_layout.addWidget(box)
        self._ident_base_layout.addStretch(1)

    def _on_ident_baseline(self, experiment_id: int, checked: bool) -> None:
        if self._session is None:
            return
        self._session.identifiability_baseline[int(experiment_id)] = bool(checked)
        self.view_changed.emit()

    def _on_ident_shift_scale(self) -> None:
        if self._session is None:
            return
        data = self._ident_shift_scale.currentData()
        self._session.identifiability_shift_scale = str(data or "units")
        self.view_changed.emit()

    def _parameter_space_labels(self, session: AppSession) -> list[str]:
        labels = list(session.document.material.labels())
        if labels:
            return labels
        result = session.result
        if result is None:
            return []
        model = getattr(result, "constitutive_model", None)
        return list(getattr(model, "parameter_labels", ()) or ())

    def _rebuild_ps_combo(self, combo: QComboBox, labels: list[str], current: str) -> None:
        combo.blockSignals(True)
        combo.clear()
        combo.setEnabled(True)
        for name in labels:
            combo.addItem(name, name)
        index = combo.findData(current)
        if index < 0:
            index = combo.findText(current)
        combo.setCurrentIndex(max(index, 0))
        combo.blockSignals(False)

    def _fill_parameter_space(self, session: AppSession) -> None:
        labels = self._parameter_space_labels(session)
        if not labels:
            self._ps_status.setText(
                "Set a material model, then run Identify. Combinations use "
                "the per-test GLS blocks from that solve."
            )
            for combo in self._ps_combos:
                combo.blockSignals(True)
                combo.clear()
                combo.setEnabled(False)
                combo.blockSignals(False)
            self._fill_ps_groups(session)
            return
        defaults = default_axis_labels(labels)
        if not session.parameter_space_left_x:
            (
                session.parameter_space_left_x,
                session.parameter_space_left_y,
                session.parameter_space_right_x,
                session.parameter_space_right_y,
            ) = defaults
        reports = list(getattr(session.result, "experiments", None) or [])
        n_tests = len(reports)
        if n_tests:
            n_subsets = (1 << n_tests) - 1
            self._ps_status.setText(
                f"{n_subsets} non-empty combination"
                f"{'s' if n_subsets != 1 else ''} of {n_tests} identified "
                "test"
                f"{'s' if n_tests != 1 else ''}. Choose which stiffnesses "
                "appear on each plot. The star is all tests together."
            )
        else:
            self._ps_status.setText(
                "Choose which stiffnesses appear on each plot. Run Identify "
                "to fill every non-empty combination of tests."
            )
        current = (
            session.parameter_space_left_x,
            session.parameter_space_left_y,
            session.parameter_space_right_x,
            session.parameter_space_right_y,
        )
        for combo, value in zip(self._ps_combos, current):
            self._rebuild_ps_combo(combo, labels, value)
        session.parameter_space_left_x = str(
            self._ps_left_x.currentData() or self._ps_left_x.currentText() or ""
        )
        session.parameter_space_left_y = str(
            self._ps_left_y.currentData() or self._ps_left_y.currentText() or ""
        )
        session.parameter_space_right_x = str(
            self._ps_right_x.currentData() or self._ps_right_x.currentText() or ""
        )
        session.parameter_space_right_y = str(
            self._ps_right_y.currentData() or self._ps_right_y.currentText() or ""
        )
        self._fill_ps_groups(session)

    def _parameter_space_experiments(self, session: AppSession) -> list[tuple[int, str]]:
        rows: list[tuple[int, str]] = []
        seen: set[int] = set()
        for experiment in session.document.experiments:
            experiment_id = int(experiment.experiment_id)
            seen.add(experiment_id)
            name = experiment.name or f"Experiment {experiment_id}"
            rows.append((experiment_id, name))
        for report in list(getattr(session.result, "experiments", None) or []):
            experiment_id = int(getattr(report, "experiment_id", 0) or 0)
            if experiment_id <= 0 or experiment_id in seen:
                continue
            seen.add(experiment_id)
            name = str(getattr(report, "label", "") or "")
            experiment = session.document.experiment_by_id(experiment_id)
            if experiment is not None and experiment.name:
                name = experiment.name
            rows.append((experiment_id, name or f"Experiment {experiment_id}"))
        return rows

    def _clear_ps_groups(self) -> None:
        self._ps_group_stat_labels: list[QLabel] = []
        while self._ps_groups_layout.count():
            item = self._ps_groups_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _fill_ps_groups(self, session: AppSession) -> None:
        self._clear_ps_groups()
        experiments = self._parameter_space_experiments(session)
        self._ps_add_group.setEnabled(bool(experiments))
        if not experiments:
            empty = QLabel(
                "Add experiments in the solver, then add a group to colour "
                "their combinations (A tests, twist only, A+B bending, …)."
            )
            empty.setWordWrap(True)
            self._ps_groups_layout.addWidget(empty)
            self._ps_groups_layout.addStretch(1)
            return
        for index, group in enumerate(session.parameter_space_groups):
            self._ps_groups_layout.addWidget(
                self._make_ps_group_card(index, group, experiments)
            )
        self._ps_groups_layout.addStretch(1)

    def _make_ps_group_card(
        self,
        index: int,
        group: dict,
        experiments: list[tuple[int, str]],
    ) -> QWidget:
        card = QWidget()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(0, 4, 0, 8)
        layout.setSpacing(4)
        header = QWidget()
        row = QHBoxLayout(header)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        color = str(group.get("color") or LINE_PALETTE[index % len(LINE_PALETTE)])
        swatch = ColorSwatchButton(color)
        swatch.setToolTip("Group colour")
        swatch.colorChanged.connect(
            lambda value, i=index: self._on_ps_group_color(i, value)
        )
        name = QLineEdit(str(group.get("name") or f"Group {index + 1}"))
        name.setPlaceholderText("Group name")
        name.editingFinished.connect(
            lambda i=index, edit=name: self._on_ps_group_name(i, edit.text())
        )
        remove = QPushButton("Remove")
        self._compact(remove)
        remove.clicked.connect(lambda _checked=False, i=index: self._on_ps_remove_group(i))
        row.addWidget(swatch)
        row.addWidget(name, 1)
        row.addWidget(remove)
        layout.addWidget(header)
        selected = {int(experiment_id) for experiment_id in (group.get("experiment_ids") or [])}
        for experiment_id, label in experiments:
            box = QCheckBox(label)
            box.setChecked(experiment_id in selected)
            box.toggled.connect(
                lambda checked, i=index, eid=experiment_id: self._on_ps_group_experiment(
                    i, eid, checked
                )
            )
            layout.addWidget(box)
        stats = QLabel()
        stats.setWordWrap(True)
        stats.setProperty("role", "hint")
        layout.addWidget(stats)
        self._ps_group_stat_labels.append(stats)
        stats.setText(self._ps_group_stats_text(self._ps_space(), group))
        return card

    def _ps_space(self):
        if self._session is None:
            return None
        return space_from_document(self._session.result, self._session.document)

    def _ps_group_stats_text(self, space, group: dict) -> str:
        ids = list(group.get("experiment_ids") or [])
        if not ids:
            return "Tick tests to compute mean and σ of their combinations."
        if space is None:
            return "Run Identify to compute mean and σ."
        stats = group_statistics(space, ids)
        n = int(stats["n"])
        if n == 0:
            return "No combinations include all ticked tests."
        noun = "combination" if n == 1 else "combinations"
        lines = [f"{n} {noun}"]
        mean = stats["mean"]
        std = stats["std"]
        for name, mu, sigma in zip(stats["labels"], mean, std):
            unit = parameter_unit_plain(name)
            unit_bit = f" {unit}" if unit else ""
            mu_text = format_number(mu)
            if n < 2 or sigma != sigma:
                lines.append(f"{name}: {mu_text}{unit_bit}")
            else:
                lines.append(
                    f"{name}: {mu_text} ± {format_number(sigma)}{unit_bit}"
                )
        if n < 2:
            lines.append("σ needs at least 2 combinations.")
        return "\n".join(lines)

    def _refresh_ps_group_stats(self) -> None:
        session = self._session
        groups = list(session.parameter_space_groups) if session is not None else []
        space = self._ps_space()
        for index, label in enumerate(self._ps_group_stat_labels):
            group = groups[index] if index < len(groups) else {}
            label.setText(self._ps_group_stats_text(space, group))

    def _on_ps_add_group(self) -> None:
        if self._session is None:
            return
        n = len(self._session.parameter_space_groups)
        self._session.parameter_space_groups.append(
            {
                "name": f"Group {n + 1}",
                "color": LINE_PALETTE[n % len(LINE_PALETTE)],
                "experiment_ids": [],
            }
        )
        self._fill_ps_groups(self._session)
        self.view_changed.emit()

    def _on_ps_remove_group(self, index: int) -> None:
        if self._session is None:
            return
        groups = self._session.parameter_space_groups
        if 0 <= index < len(groups):
            groups.pop(index)
        self._fill_ps_groups(self._session)
        self.view_changed.emit()

    def _on_ps_group_name(self, index: int, text: str) -> None:
        if self._session is None:
            return
        groups = self._session.parameter_space_groups
        if not (0 <= index < len(groups)):
            return
        groups[index]["name"] = str(text).strip() or f"Group {index + 1}"
        self.view_changed.emit()

    def _on_ps_group_color(self, index: int, color: str) -> None:
        if self._session is None:
            return
        groups = self._session.parameter_space_groups
        if not (0 <= index < len(groups)):
            return
        groups[index]["color"] = str(color)
        self.view_changed.emit()

    def _on_ps_group_experiment(
        self, index: int, experiment_id: int, checked: bool
    ) -> None:
        if self._session is None:
            return
        groups = self._session.parameter_space_groups
        if not (0 <= index < len(groups)):
            return
        ids = [int(item) for item in (groups[index].get("experiment_ids") or [])]
        key = int(experiment_id)
        if checked and key not in ids:
            ids.append(key)
        if not checked:
            ids = [item for item in ids if item != key]
        groups[index]["experiment_ids"] = ids
        self._refresh_ps_group_stats()
        self.view_changed.emit()

    def _on_parameter_space_axis(self) -> None:
        if self._session is None:
            return
        self._session.parameter_space_left_x = str(
            self._ps_left_x.currentData() or self._ps_left_x.currentText() or ""
        )
        self._session.parameter_space_left_y = str(
            self._ps_left_y.currentData() or self._ps_left_y.currentText() or ""
        )
        self._session.parameter_space_right_x = str(
            self._ps_right_x.currentData() or self._ps_right_x.currentText() or ""
        )
        self._session.parameter_space_right_y = str(
            self._ps_right_y.currentData() or self._ps_right_y.currentText() or ""
        )
        self.view_changed.emit()

    def _fill_stress_space(self, session: AppSession) -> None:
        analysis = session.document.material.analysis_type
        defaults = default_plot_axes(analysis)
        keys = [spec.key for spec in available_resultants(analysis)]
        session.stress_space_2d_x = resolve_axis_key(
            analysis, session.stress_space_2d_x, defaults["x2d"]
        )
        session.stress_space_2d_y = resolve_axis_key(
            analysis, session.stress_space_2d_y, defaults["y2d"]
        )
        session.stress_space_3d_x = resolve_axis_key(
            analysis, session.stress_space_3d_x, defaults["x3d"]
        )
        session.stress_space_3d_y = resolve_axis_key(
            analysis, session.stress_space_3d_y, defaults["y3d"]
        )
        session.stress_space_3d_z = resolve_axis_key(
            analysis, session.stress_space_3d_z, defaults["z3d"]
        )
        session.stress_space_color = resolve_color_key(
            analysis, session.stress_space_color, defaults["color"]
        )
        axis_values = (
            session.stress_space_2d_x,
            session.stress_space_2d_y,
            session.stress_space_3d_x,
            session.stress_space_3d_y,
            session.stress_space_3d_z,
        )
        for combo, value in zip(self._ss_axis_combos, axis_values):
            self._rebuild_ps_combo(combo, keys, value)
        colors = color_choices(analysis)
        self._rebuild_ps_combo(self._ss_color, colors, session.stress_space_color)
        session.stress_space_2d_x = str(
            self._ss_2d_x.currentData() or self._ss_2d_x.currentText() or ""
        )
        session.stress_space_2d_y = str(
            self._ss_2d_y.currentData() or self._ss_2d_y.currentText() or ""
        )
        session.stress_space_3d_x = str(
            self._ss_3d_x.currentData() or self._ss_3d_x.currentText() or ""
        )
        session.stress_space_3d_y = str(
            self._ss_3d_y.currentData() or self._ss_3d_y.currentText() or ""
        )
        session.stress_space_3d_z = str(
            self._ss_3d_z.currentData() or self._ss_3d_z.currentText() or ""
        )
        session.stress_space_color = str(
            self._ss_color.currentData() or self._ss_color.currentText() or ""
        )
        targets = list_stress_space_targets(session.document)
        target, step_index = resolve_stress_space_target(
            session.document,
            session.stress_space_experiment_id,
            session.stress_space_step_index,
        )
        if session.result is None:
            self._ss_status.setText(
                "Run Identify first. Stress resultants are reconstructed from "
                "the identified stiffness at every DIC point."
            )
        elif target is None:
            self._ss_status.setText(
                "Add an experiment with load steps. Each point is one DIC subset."
            )
        else:
            n_steps = len(target.step_indices)
            extra = (
                " All load steps overlay the loading path."
                if step_index == ALL_STEPS
                else ""
            )
            self._ss_status.setText(
                f"{target.experiment_name}: {n_steps} load step"
                f"{'s' if n_steps != 1 else ''} in the analysis window. "
                "Choose which resultants appear on the 2D (left) and 3D (right) "
                "plots."
                + extra
            )
        self._ss_experiment.blockSignals(True)
        self._ss_experiment.clear()
        for item in targets:
            self._ss_experiment.addItem(item.experiment_name, item.experiment_id)
        if target is not None:
            index = self._ss_experiment.findData(target.experiment_id)
            if index >= 0:
                self._ss_experiment.setCurrentIndex(index)
            session.stress_space_experiment_id = target.experiment_id
        self._ss_experiment.blockSignals(False)
        self._ss_experiment.setEnabled(bool(targets))
        self._refill_ss_steps(session, targets, emit=False)

    def _refill_ss_steps(self, session: AppSession, targets=None, emit: bool = True) -> None:
        if targets is None:
            targets = list_stress_space_targets(session.document)
        experiment_id = session.stress_space_experiment_id
        target = None
        for item in targets:
            if experiment_id is None or item.experiment_id == experiment_id:
                target = item
                if experiment_id is not None:
                    break
        self._ss_step.blockSignals(True)
        self._ss_step.clear()
        self._ss_step.addItem("All load steps", ALL_STEPS)
        if target is not None:
            for index, label in zip(target.step_indices, target.step_labels):
                self._ss_step.addItem(label, int(index))
        wanted = session.stress_space_step_index
        if wanted is None:
            wanted = ALL_STEPS
        index = self._ss_step.findData(int(wanted))
        if index < 0:
            index = 0
        if self._ss_step.count():
            self._ss_step.setCurrentIndex(index)
            data = self._ss_step.currentData()
            session.stress_space_step_index = ALL_STEPS if data is None else int(data)
        else:
            session.stress_space_step_index = ALL_STEPS
        self._ss_step.blockSignals(False)
        self._ss_step.setEnabled(self._ss_step.count() > 0)
        if emit:
            self.view_changed.emit()

    def _on_ss_experiment(self, _index: int = 0) -> None:
        if self._session is None:
            return
        data = self._ss_experiment.currentData()
        self._session.stress_space_experiment_id = None if data is None else int(data)
        self._session.stress_space_step_index = ALL_STEPS
        self._refill_ss_steps(self._session, emit=True)

    def _on_ss_step(self, _index: int = 0) -> None:
        if self._session is None:
            return
        data = self._ss_step.currentData()
        self._session.stress_space_step_index = ALL_STEPS if data is None else int(data)
        self.view_changed.emit()

    def _on_ss_axis(self) -> None:
        if self._session is None:
            return
        self._session.stress_space_2d_x = str(
            self._ss_2d_x.currentData() or self._ss_2d_x.currentText() or ""
        )
        self._session.stress_space_2d_y = str(
            self._ss_2d_y.currentData() or self._ss_2d_y.currentText() or ""
        )
        self._session.stress_space_3d_x = str(
            self._ss_3d_x.currentData() or self._ss_3d_x.currentText() or ""
        )
        self._session.stress_space_3d_y = str(
            self._ss_3d_y.currentData() or self._ss_3d_y.currentText() or ""
        )
        self._session.stress_space_3d_z = str(
            self._ss_3d_z.currentData() or self._ss_3d_z.currentText() or ""
        )
        self._session.stress_space_color = str(
            self._ss_color.currentData() or self._ss_color.currentText() or ""
        )
        self.view_changed.emit()

    def _on_exp_check_visible(self, experiment_id: int, checked: bool) -> None:
        if self._session is None:
            return
        self._session.experiment_check_visible[int(experiment_id)] = bool(checked)
        self.view_changed.emit()

    def _on_exp_check_color(self, experiment_id: int, color: str) -> None:
        if self._session is None:
            return
        self._session.experiment_check_colors[int(experiment_id)] = str(color)
        self.view_changed.emit()

    def _fill_uncertainty(self, session: AppSession, selection: Selection) -> None:
        _block_set_combo(self._uq_kind, session.uncertainty_kind)
        labels = _UQ_PLOT_MODES.get(selection.kind, [])
        self._uq_plot_mode.blockSignals(True)
        self._uq_plot_mode.clear()
        self._uq_plot_mode.addItems(labels)
        wanted = session.plot_modes.get(selection.kind)
        if wanted:
            index = self._uq_plot_mode.findText(wanted)
            if index >= 0:
                self._uq_plot_mode.setCurrentIndex(index)
        self._uq_plot_mode.blockSignals(False)
        self._uq_plot_mode.setVisible(bool(labels))
        if labels:
            session.plot_modes[selection.kind] = self._uq_plot_mode.currentText()
        self._uq_refit.setEnabled(selection.kind == "post_uncertainty_loo")
        uq = session.uncertainty
        if session.result is None:
            self._uq_status.setText("Run Identify first.")
            return
        if uq is None:
            self._uq_status.setText("No uncertainty analysis yet. Click Recompute.")
            return
        chi = uq.chi2.get("reduced")
        chi_text = f"{chi:.1f}" if chi is not None and chi == chi else "—"
        need = uq.n_parameters + 1
        posed = (
            "yes — quote sandwich σ"
            if uq.well_posed
            else f"no — M < {need}, read Information geometry"
        )
        cond = uq.information.condition
        cond_text = f"{cond:.3g}" if cond == cond and cond != float("inf") else "∞"
        self._uq_status.setText(
            f"M = {uq.n_clusters} experiments\n"
            f"p = {uq.n_parameters} parameters (need M ≥ {need})\n"
            f"rank(B) ≤ {uq.rank_bound}\n"
            f"cond(A) = {cond_text}\n"
            f"χ²/dof = {chi_text}\n"
            f"Robust well-posed: {posed}"
        )

    def _on_uq_kind(self, text: str) -> None:
        if self._session is None:
            return
        kind = str(text or "CR2")
        if kind == self._session.uncertainty_kind:
            return
        self._session.uncertainty_kind = kind
        self.request_uncertainty.emit()

    def _fill_material(self, session: AppSession) -> None:
        material = session.document.material
        material.sync_parameters()
        _block_set_combo(self._mat_type, material.material_type)
        _block_set_combo(self._mat_analysis, material.analysis_type)
        labels = material.labels()
        values = list(material.initial_parameters)
        for index, spin in enumerate(self._theta_spins):
            visible = index < len(labels)
            spin.setVisible(visible)
            self._theta_labels[index].setVisible(visible)
            if visible:
                self._theta_labels[index].setText(f"θ  {labels[index]}")
                _block_set_spin(spin, float(values[index]))

    def _fill_solver(self, session: AppSession) -> None:
        solver = session.document.solver
        _block_set_spin(self._sol_iters, solver.max_iterations)
        _block_set_spin(self._sol_tol, solver.convergence_tolerance)
        _block_set_spin(self._sol_degree, solver.basis_degree)
        _block_set_combo(self._sol_weight, solver.weighting)
        _block_set_checked(self._sol_spatial, solver.spatial_correction)
        _block_set_spin(self._sol_cov, solver.covariance_tolerance)
        self._update_solver_check(session)

    def _update_solver_check(self, session: AppSession) -> None:
        from vfmstudio.core.jobs import validate_document

        report = validate_document(session.document)
        if report.errors:
            text = "Cannot identify:\n" + "\n".join(report.errors[:8])
        elif report.warnings:
            text = "Warnings:\n" + "\n".join(report.warnings[:6])
        else:
            included = session.document.experiments_in_identification()
            names = ", ".join(
                item.name or f"Experiment {item.experiment_id}" for item in included
            ) or "none"
            text = f"Ready to identify ({names})."
        coverage = []
        for experiment in session.document.experiments_in_identification() or session.document.experiments:
            name = experiment.name or f"Experiment {experiment.experiment_id}"
            n_p = len(experiment.prescribed)
            n_l = len(experiment.linking)
            n_e = len(experiment.external)
            if n_p + n_l + n_e == 0:
                coverage.append(f"{name}: no BCs.")
            else:
                coverage.append(
                    f"{name}: {n_p} prescribed, {n_l} linking, {n_e} external."
                )
        if coverage:
            text = text + "\nBC coverage:\n" + "\n".join(coverage)
        self._sol_check.setText(text)

    def _fill_experiment(self, session: AppSession, experiment_id: Optional[int]) -> None:
        experiment = session.document.experiment_by_id(experiment_id or 0)
        if experiment is None:
            return
        _block_set_text(self._exp_name, experiment.name)
        _block_set_spin(self._exp_angle, experiment.material_orientation_deg)
        _block_set_spin(self._exp_area, float(experiment.sample_area_total or 0.0))
        _block_set_checked(self._exp_fit_axes, bool(experiment.fit_data_to_system_axes))
        self._exp_fit_axes.setVisible(True)
        _block_set_checked(
            self._exp_show_deformed, bool(experiment.show_deformed_configuration)
        )
        self._exp_show_deformed.setVisible(True)
        self._fill_deform_scale(experiment)
        _block_set_checked(
            self._exp_include, bool(experiment.include_in_identification)
        )

    def _fill_bcs(self, session: AppSession, selection: Selection) -> None:
        experiment = session.document.experiment_by_id(selection.experiment_id or 0)
        if experiment is None:
            return
        self._fill_bc_copy_combo(session, experiment.experiment_id)
        n_p = len(experiment.prescribed)
        n_l = len(experiment.linking)
        n_e = len(experiment.external)
        self._bc_summary.setText(
            f"{n_p} prescribed · {n_l} linking · {n_e} external"
        )

    def _fill_mesh(self, session: AppSession, experiment_id: Optional[int]) -> None:
        experiment = session.document.experiment_by_id(experiment_id or 0)
        if experiment is None:
            return
        mesh = experiment.mesh
        _block_set_spin(self._mesh_nx, mesh.n_elements_x)
        _block_set_spin(self._mesh_ny, mesh.n_elements_y)
        for attr, spin in self._bound_spins.items():
            _block_set_spin(spin, float(getattr(mesh.bounds, attr)))
        _block_set_checked(
            self._mesh_show_counts, bool(session.show_element_point_counts)
        )
        self._mesh_show_counts.setVisible(True)
        self._sync_free_mesh_controls(session)
        self._refresh_mesh_count_summary()

    def _fill_bc(self, session: AppSession, selection: Selection) -> None:
        entry = self._current_bc()
        if entry is None:
            return
        _block_set_text(self._bc_label, entry.label)
        _block_set_combo(self._bc_dof, entry.dof or "W")
        _block_set_combo(self._bc_edge, entry.edge or "")
        is_external = selection.bc_kind == "external"
        self._bc_load_box.setVisible(is_external)
        self._bc_value_box.setVisible(not is_external)
        if is_external:
            _block_set_combo(self._bc_load, entry.load_type or "distributed")
        else:
            _block_set_spin(self._bc_value, float(entry.value or 0.0))
        nodes = list(entry.nodes or [])
        if nodes:
            shown = ", ".join(str(item) for item in nodes[:12])
            extra = "" if len(nodes) <= 12 else f" … (+{len(nodes) - 12})"
            self._bc_nodes_hint.setText(f"Nodes: {shown}{extra}")
        else:
            self._bc_nodes_hint.setText("Nodes: (named edge)")
        picking = bool(getattr(session, "pick_nodes", False))
        self._bc_pick_nodes.blockSignals(True)
        self._bc_pick_nodes.setChecked(picking)
        self._bc_pick_nodes.blockSignals(False)
        self._bc_pick_nodes.setText("Picking…" if picking else "Pick nodes")

    def _fill_step(self, session: AppSession, selection: Selection) -> None:
        step = self._current_step()
        experiment = self._current_experiment()
        show_step_fields = selection.kind == "step" and step is not None
        self._step_fields.setVisible(show_step_fields)
        self._step_label.setVisible(show_step_fields)
        self._step_label_caption.setVisible(show_step_fields)
        self._step_path.setVisible(show_step_fields)
        self._step_path_row.setVisible(show_step_fields)
        self._step_path_caption.setVisible(show_step_fields)
        if experiment is None:
            self._populate_load_editor(self._exp_load_editor, None)
            self._analysis_start.setEnabled(False)
            self._analysis_end.setEnabled(False)
            self._analysis_reference.setEnabled(False)
            self._visualize_steps.setEnabled(False)
            self._add_step_button.setEnabled(False)
            return
        _block_set_combo_data(self._exp_source, experiment.applied_load_source)
        highlight = (
            selection.step_index
            if selection.kind == "step" and selection.step_index is not None
            else None
        )
        if show_step_fields and step is not None:
            _block_set_text(self._step_label, step.label)
            _block_set_text(self._step_path, str(step.data_file))
        self._populate_load_editor(
            self._exp_load_editor, experiment, highlight_row=highlight if show_step_fields else None
        )
        n_steps = len(experiment.steps)
        has_steps = n_steps > 0
        self._analysis_start.setEnabled(has_steps)
        self._analysis_end.setEnabled(has_steps)
        self._analysis_reference.setEnabled(has_steps)
        self._visualize_steps.setEnabled(has_steps)
        self._add_step_button.setEnabled(True)
        self._fill_analysis_combos(
            experiment,
            self._analysis_start,
            self._analysis_end,
            self._analysis_reference,
        )

    def _fill_analysis_combos(
        self, experiment, start_combo: QComboBox, end_combo: QComboBox, ref_combo: QComboBox
    ) -> None:
        n_steps = len(experiment.steps)
        enabled = n_steps > 0
        start_combo.setEnabled(enabled)
        end_combo.setEnabled(enabled)
        ref_combo.setEnabled(enabled)
        if not enabled:
            for combo in (start_combo, end_combo, ref_combo):
                combo.blockSignals(True)
                combo.clear()
                combo.blockSignals(False)
            return
        start_value = experiment.resolved_analysis_start()
        end_value = experiment.resolved_analysis_end()
        self._fill_step_index_combo(start_combo, experiment.steps, start_value)
        self._fill_step_index_combo(end_combo, experiment.steps, end_value)
        ref_combo.blockSignals(True)
        ref_combo.clear()
        ref_combo.addItem("None", None)
        for index, item in enumerate(experiment.steps):
            ref_combo.addItem(self._step_choice_label(index, item), index + 1)
        reference = experiment.reference_step
        if reference is None:
            ref_combo.setCurrentIndex(0)
        else:
            match = ref_combo.findData(int(reference))
            ref_combo.setCurrentIndex(match if match >= 0 else 0)
        ref_combo.blockSignals(False)

    @staticmethod
    def _step_choice_label(index: int, step) -> str:
        return f"{index + 1}: {step_display_label(step)}"

    def _fill_step_index_combo(self, combo: QComboBox, steps, selected: int) -> None:
        combo.blockSignals(True)
        combo.clear()
        for index, item in enumerate(steps):
            combo.addItem(self._step_choice_label(index, item), index + 1)
        match = combo.findData(int(selected))
        combo.setCurrentIndex(match if match >= 0 else 0)
        combo.blockSignals(False)

    def _populate_load_editor(
        self,
        editor: AppliedLoadEditor,
        experiment,
        *,
        highlight_row: Optional[int] = None,
    ) -> None:
        source_path = None
        if self._session is not None:
            source_path = self._session.document.source_path
        editor.populate(
            experiment,
            source_path=source_path,
            highlight_row=highlight_row,
        )

    def _on_load_table_changed(self) -> None:
        experiment = self._current_experiment()
        if experiment is not None and experiment.applied_load_source != "manual":
            experiment.applied_load_source = "manual"
        self._notify()

    def _current_experiment(self):
        if self._session is None:
            return None
        selection = self._session.selection
        return self._session.document.experiment_by_id(selection.experiment_id or 0)

    def _current_bc(self):
        experiment = self._current_experiment()
        if self._session is None or experiment is None:
            return None
        selection = self._session.selection
        if selection.bc_kind is None or selection.bc_index is None:
            return None
        entries = getattr(experiment, selection.bc_kind)
        if selection.bc_index >= len(entries):
            return None
        return entries[selection.bc_index]

    def _current_step(self):
        experiment = self._current_experiment()
        if self._session is None or experiment is None or not experiment.steps:
            return None
        index = self._session.selection.step_index
        index = 0 if index is None else index
        index = max(0, min(index, len(experiment.steps) - 1))
        return experiment.steps[index]

    def _on_project_desc(self) -> None:
        if self._session is None:
            return
        self._session.document.description = self._proj_desc.text()
        self._notify()

    def _on_material_type(self, text: str) -> None:
        if self._syncing or self._session is None:
            return
        material = self._session.document.material
        if material.material_type == text:
            return
        material.material_type = text
        material.sync_parameters()
        self._notify()
        self._fill_material(self._session)

    def _on_analysis_type(self, text: str) -> None:
        if self._syncing or self._session is None:
            return
        material = self._session.document.material
        if material.analysis_type == text:
            return
        material.analysis_type = text
        material.sync_parameters()
        self._notify()
        self._fill_material(self._session)

    def _on_theta_finished(self) -> None:
        if self._syncing or self._session is None:
            return
        material = self._session.document.material
        labels = material.labels()
        material.initial_parameters = [
            float(self._theta_spins[index].value()) for index in range(len(labels))
        ]
        self._notify()

    def _on_solver_iters(self, value: int) -> None:
        if self._session is None:
            return
        self._session.document.solver.max_iterations = int(value)
        self._notify()

    def _on_solver_tol(self) -> None:
        if self._session is None:
            return
        self._session.document.solver.convergence_tolerance = float(self._sol_tol.value())
        self._notify()

    def _on_solver_degree(self, value: int) -> None:
        if self._session is None:
            return
        self._session.document.solver.basis_degree = int(value)
        self._notify()

    def _on_solver_weight(self, text: str) -> None:
        if self._session is None:
            return
        self._session.document.solver.weighting = text
        self._notify()

    def _on_solver_spatial(self, checked: bool) -> None:
        if self._session is None:
            return
        self._session.document.solver.spatial_correction = bool(checked)
        self._notify()

    def _on_solver_cov(self) -> None:
        if self._session is None:
            return
        self._session.document.solver.covariance_tolerance = float(self._sol_cov.value())
        self._notify()

    def _on_exp_include(self, checked: bool) -> None:
        experiment = self._current_experiment()
        if experiment is None:
            return
        experiment.include_in_identification = bool(checked)
        self._notify()
        self.view_changed.emit()

    def _on_exp_name(self) -> None:
        experiment = self._current_experiment()
        if experiment is None:
            return
        experiment.name = self._exp_name.text()
        self._notify()

    def _on_exp_angle(self) -> None:
        experiment = self._current_experiment()
        if experiment is None:
            return
        experiment.material_orientation_deg = float(self._exp_angle.value())
        self._notify()

    def _on_exp_area(self) -> None:
        experiment = self._current_experiment()
        if experiment is None:
            return
        experiment.sample_area_total = float(self._exp_area.value()) or None
        self._notify()

    def _on_exp_area_estimate(self) -> None:
        experiment = self._current_experiment()
        if experiment is None or self._session is None:
            return
        if not experiment.steps:
            QMessageBox.information(
                self,
                "Sample area",
                "Import a MAT load step first so the area can be estimated "
                "from the DIC cloud.",
            )
            return
        area = estimate_experiment_sample_area(
            experiment,
            experiment.steps[0].data_file,
            analysis_type=self._session.document.material.analysis_type,
            source_path=self._session.document.source_path,
            replace=True,
        )
        if area is None:
            QMessageBox.information(
                self,
                "Sample area",
                "Could not estimate the specimen area from the DIC points.",
            )
            return
        _block_set_spin(self._exp_area, float(area))
        self._notify()

    def _on_exp_source(self, _index: int = 0) -> None:
        experiment = self._current_experiment()
        if experiment is None:
            return
        source = str(self._exp_source.currentData() or "mat")
        if source == "manual":
            source_path = (
                self._session.document.source_path if self._session is not None else None
            )
            fill_blank_external_values(experiment, source_path)
        experiment.applied_load_source = source
        self._populate_load_editor(self._exp_load_editor, experiment)
        self._notify()

    def _on_exp_fit_axes(self, checked: bool) -> None:
        experiment = self._current_experiment()
        if experiment is None or self._session is None:
            return
        experiment.fit_data_to_system_axes = bool(checked)
        if experiment.steps:
            sync_mesh_bounds_from_file(
                experiment,
                experiment.steps[0].data_file,
                source_path=self._session.document.source_path,
                analysis_type=self._session.document.material.analysis_type,
                replace=True,
            )
        self._notify()

    def _on_exp_show_deformed(self, checked: bool) -> None:
        experiment = self._current_experiment()
        if experiment is None or self._session is None:
            return
        experiment.show_deformed_configuration = bool(checked)
        self._set_deform_scale_enabled(bool(checked))
        if checked:
            step = self._session.selected_step() or self._current_step()
            path = None
            if step is not None:
                path = resolve_step_data_file(
                    step.data_file, self._session.document.source_path
                )
            if not mat_has_deformed_configuration(path):
                QMessageBox.information(
                    self,
                    "Deformed configuration",
                    "Deformed configuration is not available for this load step. "
                    "The MAT file does not contain U, V, and W (and matching "
                    "X, Y, Z) for each point.",
                )
        self._notify()

    def _make_deform_scale_row(self) -> tuple[QWidget, QSlider, QDoubleSpinBox]:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, int(_DEFORM_SCALE_MAX * _DEFORM_SCALE_TICKS))
        slider.setSingleStep(1)
        slider.setPageStep(_DEFORM_SCALE_TICKS)
        slider.setValue(_DEFORM_SCALE_TICKS)
        slider.setToolTip(
            "Scale DIC displacements in the 3D view. 1× is true scale; "
            "increase to exaggerate out-of-plane motion."
        )
        spin = QDoubleSpinBox()
        spin.setRange(0.0, _DEFORM_SCALE_MAX)
        spin.setDecimals(1)
        spin.setSingleStep(0.5)
        spin.setSuffix(" ×")
        spin.setValue(1.0)
        spin.setMaximumWidth(78)
        spin.setToolTip(slider.toolTip())
        layout.addWidget(slider, 1)
        layout.addWidget(spin, 0)
        slider.valueChanged.connect(self._on_deform_slider)
        slider.sliderReleased.connect(self._on_deform_slider_released)
        spin.valueChanged.connect(self._on_deform_spin)
        self._deform_sliders.append(slider)
        self._deform_spins.append(spin)
        return row, slider, spin

    def _fill_deform_scale(self, experiment) -> None:
        scale = float(getattr(experiment, "deformation_scale", 1.0) or 1.0)
        ticks = int(round(scale * _DEFORM_SCALE_TICKS))
        for slider in self._deform_sliders:
            _block_set_slider(slider, ticks)
        for spin in self._deform_spins:
            _block_set_spin(spin, scale)
        self._set_deform_scale_enabled(bool(experiment.show_deformed_configuration))

    def _set_deform_scale_enabled(self, enabled: bool) -> None:
        for slider in self._deform_sliders:
            slider.setEnabled(enabled)
        for spin in self._deform_spins:
            spin.setEnabled(enabled)

    def _on_deform_slider(self, ticks: int) -> None:
        self._commit_deformation_scale(float(ticks) / _DEFORM_SCALE_TICKS, live=True)

    def _on_deform_spin(self, value: float) -> None:
        self._commit_deformation_scale(float(value), live=False)

    def _on_deform_slider_released(self) -> None:
        experiment = self._current_experiment()
        if experiment is None:
            return
        self._commit_deformation_scale(
            float(experiment.deformation_scale), live=False
        )

    def _commit_deformation_scale(self, value: float, *, live: bool = True) -> None:
        value = min(_DEFORM_SCALE_MAX, max(0.0, float(value)))
        ticks = int(round(value * _DEFORM_SCALE_TICKS))
        for slider in self._deform_sliders:
            _block_set_slider(slider, ticks)
        for spin in self._deform_spins:
            _block_set_spin(spin, value)
        experiment = self._current_experiment()
        if experiment is None:
            return
        experiment.deformation_scale = value
        if not live and self._session is not None:
            self._session.mark_dirty()
        self.deformation_scale_changed.emit(value, live)

    def _on_add_bc(self, kind: str = "prescribed") -> None:
        experiment = self._current_experiment()
        if experiment is None:
            return
        if kind not in {"prescribed", "linking", "external"}:
            return
        self.request_add_bc.emit(experiment.experiment_id, kind)

    def _on_delete_bc(self) -> None:
        experiment = self._current_experiment()
        if experiment is None or self._session is None:
            return
        selection = self._session.selection
        if selection.bc_kind is None or selection.bc_index is None:
            return
        self.request_delete_bc.emit(
            experiment.experiment_id, str(selection.bc_kind), int(selection.bc_index)
        )

    def _on_add_step(self) -> None:
        experiment = self._current_experiment()
        if experiment is None:
            return
        self.request_add_step.emit(experiment.experiment_id)

    def _on_mesh_nx(self) -> None:
        experiment = self._current_experiment()
        if experiment is None:
            return
        experiment.mesh.clear_free_and_manual()
        experiment.mesh.n_elements_x = int(self._mesh_nx.value())
        self._notify()
        self._refresh_mesh_count_summary()

    def _on_mesh_ny(self) -> None:
        experiment = self._current_experiment()
        if experiment is None:
            return
        experiment.mesh.clear_free_and_manual()
        experiment.mesh.n_elements_y = int(self._mesh_ny.value())
        self._notify()
        self._refresh_mesh_count_summary()

    def _on_mesh_bound(self, attr: str) -> None:
        experiment = self._current_experiment()
        if experiment is None:
            return
        experiment.mesh.clear_free_and_manual()
        setattr(experiment.mesh.bounds, attr, float(self._bound_spins[attr].value()))
        self._notify()
        self._refresh_mesh_count_summary()

    def _on_mesh_fit(self) -> None:
        experiment = self._current_experiment()
        if experiment is None or self._session is None:
            return
        if not experiment.steps:
            QMessageBox.information(self, "Fit mesh", "Import a MAT load step first.")
            return
        experiment.mesh.clear_free_and_manual()
        ok = sync_mesh_bounds_from_file(
            experiment,
            experiment.steps[0].data_file,
            source_path=self._session.document.source_path,
            analysis_type=self._session.document.material.analysis_type,
            replace=True,
        )
        if not ok:
            QMessageBox.information(
                self, "Fit mesh", "Could not read specimen bounds from the MAT file."
            )
            return
        self._fill_mesh(self._session, experiment.experiment_id)
        self._notify()

    def _sync_free_mesh_controls(self, session: AppSession) -> None:
        in_free = bool(getattr(session, "free_mesh_mode", False))
        in_drag = bool(getattr(session, "drag_mesh_mode", False))
        in_tool = in_free or in_drag
        if hasattr(self, "_mesh_free"):
            self._mesh_free.blockSignals(True)
            self._mesh_free.setChecked(in_free)
            self._mesh_free.blockSignals(False)
        if hasattr(self, "_mesh_drag"):
            self._mesh_drag.blockSignals(True)
            self._mesh_drag.setChecked(in_drag)
            self._mesh_drag.blockSignals(False)
        if hasattr(self, "_mesh_free_actions"):
            self._mesh_free_actions.setVisible(in_free)
        if hasattr(self, "_mesh_construct"):
            self._mesh_construct.setVisible(in_free)
            self._mesh_construct.setEnabled(in_free)
        if hasattr(self, "_mesh_clear_points"):
            self._mesh_clear_points.setVisible(in_free)
            self._mesh_clear_points.setEnabled(in_free)
        if hasattr(self, "_mesh_free_hint"):
            self._mesh_free_hint.setVisible(in_free)
        if hasattr(self, "_mesh_drag_hint"):
            self._mesh_drag_hint.setVisible(in_drag)
        enabled = not in_tool
        self._mesh_nx.setEnabled(enabled)
        self._mesh_ny.setEnabled(enabled)
        for spin in self._bound_spins.values():
            spin.setEnabled(enabled)
        self._mesh_fit.setEnabled(enabled)
        if hasattr(self, "_mesh_clear"):
            self._mesh_clear.setEnabled(enabled)
        self._refresh_free_mesh_hint()

    def _refresh_free_mesh_hint(self) -> None:
        if not hasattr(self, "_mesh_free_hint"):
            return
        session = self._session
        if session is None or not bool(getattr(session, "free_mesh_mode", False)):
            self._mesh_free_hint.setText("Click in the viewport to add nodes.")
            return
        status = inspect_free_points(session.free_mesh_points)
        self._mesh_free_hint.setText(status.hint)

    def _on_free_mesh_toggled(self, checked: bool) -> None:
        self.request_free_mesh_mode.emit(bool(checked))

    def _on_drag_mesh_toggled(self, checked: bool) -> None:
        self.request_drag_mesh_mode.emit(bool(checked))

    def _on_construct_free_mesh(self) -> None:
        experiment = self._current_experiment()
        if experiment is None or self._session is None:
            return
        try:
            status = construct_free_quad_mesh(self._session.free_mesh_points)
        except ConformError:
            QMessageBox.warning(self, "Construct Mesh", CONFORM_MESSAGE)
            return
        mesh = experiment.mesh
        mesh.n_elements_x = max(int(status.n_elements_x), 1)
        mesh.n_elements_y = max(int(status.n_elements_y), 1)
        mesh.bounds = BoundsModel(
            x_min=float(status.x_nodes[0]),
            x_max=float(status.x_nodes[-1]),
            y_min=float(status.y_nodes[0]),
            y_max=float(status.y_nodes[-1]),
        )
        mesh.x_nodes = list(status.x_nodes)
        mesh.y_nodes = list(status.y_nodes)
        mesh.free_nodes = [[float(x), float(y)] for x, y in status.snapped_points]
        mesh.keep_cells = [[int(ix), int(iy)] for ix, iy in status.keep_cells]
        self._session.free_mesh_points = list(status.snapped_points)
        self._fill_mesh(self._session, experiment.experiment_id)
        self._notify()

    def _on_clear_free_mesh_points(self) -> None:
        self.request_clear_free_mesh.emit()

    def _fill_bc_copy_combo(self, session: AppSession, experiment_id: int) -> None:
        self._bc_copy_from.blockSignals(True)
        self._bc_copy_from.clear()
        for item in session.document.experiments:
            if int(item.experiment_id) == int(experiment_id):
                continue
            name = item.name or f"Experiment {item.experiment_id}"
            self._bc_copy_from.addItem(name, item.experiment_id)
        self._bc_copy_from.blockSignals(False)
        enabled = self._bc_copy_from.count() > 0
        self._bc_copy_from.setEnabled(enabled)
        self._bc_copy.setEnabled(enabled)

    def _on_copy_bcs(self) -> None:
        if self._session is None:
            return
        target = self._current_experiment()
        source_id = self._bc_copy_from.currentData()
        if target is None or source_id is None:
            return
        copied = self._session.document.copy_boundary_conditions(
            int(source_id), int(target.experiment_id)
        )
        if copied is None:
            return
        self._notify()

    def _on_mesh_show_counts(self, checked: bool) -> None:
        if self._session is None:
            return
        self._session.show_element_point_counts = bool(checked)
        self._refresh_mesh_count_summary()
        self.view_changed.emit()

    def _refresh_mesh_count_summary(self) -> None:
        if not hasattr(self, "_mesh_count_summary"):
            return
        if self._session is None:
            self._mesh_count_summary.setText("—")
            return
        experiment = self._current_experiment()
        if experiment is None:
            self._mesh_count_summary.setText("—")
            return
        try:
            mesh_pkg = mesh_convert.build_experiment_mesh(
                self._session.document, experiment
            )
            xy = mesh_convert.dic_xy_for_counts(
                experiment,
                self._session.document.material.analysis_type,
                self._session.document.source_path,
                step=self._session.selected_step(),
            )
        except Exception:
            self._mesh_count_summary.setText(
                "Could not count DIC points for this mesh."
            )
            return
        if xy is None or xy.shape[0] == 0:
            self._mesh_count_summary.setText(
                "Load a step with a DIC file to count points per element."
            )
            return
        try:
            counts = mesh_convert.count_dic_points_per_element(mesh_pkg, xy)
        except Exception:
            self._mesh_count_summary.setText(
                "Could not count DIC points for this mesh."
            )
            return
        n_el = int(counts.size)
        text = mesh_convert.summarize_element_point_counts(counts)
        if n_el > mesh_convert.MAX_ELEMENT_COUNT_LABELS:
            text += " Too many elements to label on the mesh; counts are shown here."
        elif self._session.show_element_point_counts:
            text += " Occupied elements are green; empty elements are red."
        self._mesh_count_summary.setText(text)

    def _on_bc_label(self) -> None:
        entry = self._current_bc()
        if entry is None:
            return
        entry.label = self._bc_label.text()
        self._notify()

    def _on_bc_dof(self, text: str) -> None:
        entry = self._current_bc()
        if entry is None:
            return
        entry.dof = text
        self._notify()

    def _on_bc_edge(self, text: str) -> None:
        entry = self._current_bc()
        if entry is None:
            return
        entry.edge = text or None
        self._notify()

    def _on_bc_load(self, text: str) -> None:
        entry = self._current_bc()
        if entry is None:
            return
        entry.load_type = text
        self._notify()

    def _on_bc_value(self) -> None:
        entry = self._current_bc()
        if entry is None:
            return
        entry.value = float(self._bc_value.value())
        self._notify()

    def _on_step_label(self) -> None:
        step = self._current_step()
        if step is None:
            return
        step.label = self._step_label.text()
        self._notify()

    def _on_step_path(self) -> None:
        step = self._current_step()
        if step is None:
            return
        step.data_file = Path(self._step_path.text())
        self._notify()

    def _on_step_browse(self) -> None:
        step = self._current_step()
        if step is None:
            return
        chosen, _ = QFileDialog.getOpenFileName(
            self, "Measurement file", str(step.data_file), "MAT (*.mat)"
        )
        if chosen:
            step.data_file = Path(chosen)
            _block_set_text(self._step_path, chosen)
            self._apply_step_bounds(step)
            self._notify()

    def _apply_step_bounds(self, step) -> None:
        if self._session is None:
            return
        experiment = self._current_experiment()
        if experiment is None:
            return
        sync_mesh_bounds_from_file(
            experiment,
            step.data_file,
            source_path=self._session.document.source_path,
            analysis_type=self._session.document.material.analysis_type,
        )

    def _on_analysis_start(self, _index: int = 0) -> None:
        experiment = self._current_experiment()
        if experiment is None:
            return
        value = self._analysis_start.currentData()
        if value is None:
            return
        experiment.analysis_start = int(value)
        self._notify()

    def _on_analysis_end(self, _index: int = 0) -> None:
        experiment = self._current_experiment()
        if experiment is None:
            return
        value = self._analysis_end.currentData()
        if value is None:
            return
        experiment.analysis_end = int(value)
        self._notify()

    def _on_analysis_reference(self, _index: int = 0) -> None:
        experiment = self._current_experiment()
        if experiment is None:
            return
        value = self._analysis_reference.currentData()
        experiment.reference_step = None if value is None else int(value)
        self._notify()
