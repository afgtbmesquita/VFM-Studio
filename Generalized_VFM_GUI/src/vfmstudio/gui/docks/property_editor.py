"""Property editor dock — edits the session document in place."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from vfmstudio.core import mesh_convert
from vfmstudio.core.fields import sync_mesh_bounds_from_file
from vfmstudio.core.jobs import reduced_chi_square
from vfmstudio.core.document import parameter_unit_plain
from vfmstudio.core.parameter_space import (
    default_axis_labels,
    group_statistics,
    space_from_document,
)
from vfmstudio.core.step_series import step_display_label
from vfmstudio.core.uncertainty import default_baseline_flags
from vfmstudio.gui.results.experiment_check import collect_experiment_steps
from vfmstudio.gui.results.plot_style import experiment_colour
from vfmstudio.gui.results.table_style import format_number
from vfmstudio.gui.session import AppSession, Selection, is_post_kind, is_uncertainty_kind
from vfmstudio.gui.widgets.color_swatch import ColorSwatchButton, LINE_PALETTE


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


def _block_set_spin(widget: QSpinBox | QDoubleSpinBox, value: float) -> None:
    widget.blockSignals(True)
    widget.setValue(value)
    widget.blockSignals(False)


def _block_set_checked(widget: QCheckBox, checked: bool) -> None:
    widget.blockSignals(True)
    widget.setChecked(checked)
    widget.blockSignals(False)


class PropertyEditorDock(QWidget):
    document_changed = Signal()
    view_changed = Signal()
    request_add_bc = Signal(int, str)
    request_add_step = Signal(int)
    request_visualize_steps = Signal()
    request_uncertainty = Signal()
    request_loo_refit = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._session: Optional[AppSession] = None
        self._shown_key = None
        self._theta_spins: list[QDoubleSpinBox] = []
        self._theta_labels: list[QLabel] = []
        self._bound_spins: dict[str, QDoubleSpinBox] = {}
        self._step_load_spins: list[QDoubleSpinBox] = []
        self._step_load_labels: list[QLabel] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self._title = QLabel("Properties")
        self._title.setStyleSheet("font-weight: 600;")
        layout.addWidget(self._title)
        self._stack = QStackedWidget()
        layout.addWidget(self._stack)
        layout.addStretch(1)
        self._pages: dict[str, QWidget] = {}
        self._build_pages()

    def _add_page(self, name: str) -> QFormLayout:
        page = QWidget()
        form = QFormLayout(page)
        self._pages[name] = page
        self._stack.addWidget(page)
        return form

    def _build_pages(self) -> None:
        project = self._add_page("project")
        self._proj_desc = QLineEdit()
        self._proj_desc.editingFinished.connect(self._on_project_desc)
        project.addRow("Description", self._proj_desc)

        material = self._add_page("material")
        self._mat_type = QComboBox()
        self._mat_type.addItems(["iso", "orthotropic", "anisotropic"])
        self._mat_type.currentTextChanged.connect(self._on_material_type)
        self._mat_analysis = QComboBox()
        self._mat_analysis.addItems(["in_plane", "out_of_plane", "both"])
        self._mat_analysis.currentTextChanged.connect(self._on_analysis_type)
        material.addRow("Material", self._mat_type)
        material.addRow("Analysis", self._mat_analysis)
        for _ in range(8):
            spin = QDoubleSpinBox()
            spin.setDecimals(6)
            spin.setRange(-1e12, 1e12)
            spin.editingFinished.connect(self._on_theta_finished)
            label = QLabel()
            self._theta_spins.append(spin)
            self._theta_labels.append(label)
            material.addRow(label, spin)

        solver = self._add_page("solver")
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
        self._sol_spatial = QCheckBox("Spatial correction of Γ")
        self._sol_spatial.toggled.connect(self._on_solver_spatial)
        solver.addRow("Max iterations", self._sol_iters)
        solver.addRow("Tolerance", self._sol_tol)
        solver.addRow("Γ basis degree", self._sol_degree)
        solver.addRow("Weighting", self._sol_weight)
        solver.addRow(self._sol_spatial)

        experiment = self._add_page("experiment")
        self._exp_name = QLineEdit()
        self._exp_name.editingFinished.connect(self._on_exp_name)
        self._exp_angle = QDoubleSpinBox()
        self._exp_angle.setRange(-180.0, 180.0)
        self._exp_angle.setDecimals(2)
        self._exp_angle.editingFinished.connect(self._on_exp_angle)
        self._exp_area = QDoubleSpinBox()
        self._exp_area.setDecimals(6)
        self._exp_area.setRange(0.0, 1e12)
        self._exp_area.editingFinished.connect(self._on_exp_area)
        self._exp_source = QComboBox()
        self._exp_source.addItems(["mat", "manual"])
        self._exp_source.currentTextChanged.connect(self._on_exp_source)
        add_bc = QPushButton("Add EXTERNAL group…")
        add_bc.clicked.connect(self._on_add_bc)
        add_step = QPushButton("Add load steps…")
        add_step.clicked.connect(self._on_add_step)
        add_step.setToolTip(
            "Import one or more measurement MAT files as load steps on this experiment."
        )
        experiment.addRow("Name", self._exp_name)
        experiment.addRow("Material φ (deg)", self._exp_angle)
        experiment.addRow("Sample area", self._exp_area)
        experiment.addRow("Applied load from", self._exp_source)
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
        experiment.addRow("Fit data to system axes", self._exp_fit_axes)
        experiment.addRow(add_bc)
        experiment.addRow(add_step)

        mesh = self._add_page("mesh")
        self._mesh_nx = QSpinBox()
        self._mesh_nx.setRange(1, 200)
        self._mesh_nx.valueChanged.connect(self._on_mesh_nx)
        self._mesh_ny = QSpinBox()
        self._mesh_ny.setRange(1, 200)
        self._mesh_ny.valueChanged.connect(self._on_mesh_ny)
        mesh.addRow("Elements Nx", self._mesh_nx)
        mesh.addRow("Elements Ny", self._mesh_ny)
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
            mesh.addRow(title, spin)
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
        mesh.addRow("Show DIC points per element", self._mesh_show_counts)
        self._mesh_count_summary = QLabel("—")
        self._mesh_count_summary.setWordWrap(True)
        self._mesh_count_summary.setProperty("role", "hint")
        mesh.addRow(self._mesh_count_summary)

        bc = self._add_page("bc")
        self._bc_label = QLineEdit()
        self._bc_label.editingFinished.connect(self._on_bc_label)
        self._bc_dof = QComboBox()
        self._bc_dof.addItems(["W", "W_X", "W_Y", "W_XY", "U", "V"])
        self._bc_dof.currentTextChanged.connect(self._on_bc_dof)
        self._bc_edge = QComboBox()
        self._bc_edge.addItems(["", "left", "right", "top", "bottom"])
        self._bc_edge.currentTextChanged.connect(self._on_bc_edge)
        self._bc_load = QComboBox()
        self._bc_load.addItems(["distributed", "concentrated"])
        self._bc_load.currentTextChanged.connect(self._on_bc_load)
        self._bc_load_label = QLabel("Load type")
        bc.addRow("Label", self._bc_label)
        bc.addRow("DOF", self._bc_dof)
        bc.addRow("Edge", self._bc_edge)
        bc.addRow(self._bc_load_label, self._bc_load)

        step = self._add_page("step")
        self._step_label = QLineEdit()
        self._step_label.editingFinished.connect(self._on_step_label)
        self._step_path_row = QWidget()
        path_layout = QHBoxLayout(self._step_path_row)
        path_layout.setContentsMargins(0, 0, 0, 0)
        self._step_path = QLineEdit()
        self._step_path.editingFinished.connect(self._on_step_path)
        browse = QPushButton("…")
        browse.clicked.connect(self._on_step_browse)
        path_layout.addWidget(self._step_path, 1)
        path_layout.addWidget(browse)
        self._step_label_caption = QLabel("Label")
        self._step_path_caption = QLabel("Data file")
        step.addRow(self._step_label_caption, self._step_label)
        step.addRow(self._step_path_caption, self._step_path_row)
        for _ in range(8):
            spin = QDoubleSpinBox()
            spin.setDecimals(6)
            spin.setRange(-1e12, 1e12)
            spin.editingFinished.connect(self._on_step_load)
            label = QLabel()
            self._step_load_spins.append(spin)
            self._step_load_labels.append(label)
            step.addRow(label, spin)

        self._analysis_header = QLabel("Analysis")
        self._analysis_header.setStyleSheet("font-weight: 600; padding-top: 8px;")
        step.addRow(self._analysis_header)
        self._analysis_start = QComboBox()
        self._analysis_start.currentIndexChanged.connect(self._on_analysis_start)
        self._analysis_end = QComboBox()
        self._analysis_end.currentIndexChanged.connect(self._on_analysis_end)
        self._analysis_reference = QComboBox()
        self._analysis_reference.currentIndexChanged.connect(self._on_analysis_reference)
        step.addRow("Start step", self._analysis_start)
        step.addRow("End step", self._analysis_end)
        step.addRow("Reference step", self._analysis_reference)
        self._visualize_steps = QPushButton("Visualize Step Load Data")
        self._visualize_steps.clicked.connect(self.request_visualize_steps.emit)
        step.addRow(self._visualize_steps)

        empty = self._add_page("empty")
        empty.addRow(QLabel("No experiment selected."))

        post = self._add_page("post_processing")
        self._post_status = QLabel("No identification result.")
        self._post_status.setWordWrap(True)
        post.addRow(self._post_status)

        experiment_check = self._add_page("post_experiment_check")
        self._exp_check_status = QLabel("No identification result.")
        self._exp_check_status.setWordWrap(True)
        experiment_check.addRow(self._exp_check_status)
        self._exp_check_scroll = QScrollArea()
        self._exp_check_scroll.setWidgetResizable(True)
        self._exp_check_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._exp_check_scroll.setMinimumHeight(160)
        self._exp_check_host = QWidget()
        self._exp_check_layout = QVBoxLayout(self._exp_check_host)
        self._exp_check_layout.setContentsMargins(0, 0, 0, 0)
        self._exp_check_layout.setSpacing(4)
        self._exp_check_scroll.setWidget(self._exp_check_host)
        experiment_check.addRow(self._exp_check_scroll)

        ident_strength = self._add_page("post_identifiability_strength")
        self._ident_strength_status = QLabel(
            "The left plot always shows eq. (63) for every test. Untick tests "
            "to measure the information they add on the right (G_k)."
        )
        self._ident_strength_status.setWordWrap(True)
        ident_strength.addRow(self._ident_strength_status)
        self._ident_shift_scale = QComboBox()
        self._ident_shift_scale.addItem("Parameter units", "units")
        self._ident_shift_scale.addItem("Relative to |θ̂|", "relative")
        self._ident_shift_scale.currentIndexChanged.connect(self._on_ident_shift_scale)
        ident_strength.addRow("Eq. (63) values", self._ident_shift_scale)
        self._ident_base_scroll = QScrollArea()
        self._ident_base_scroll.setWidgetResizable(True)
        self._ident_base_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._ident_base_scroll.setMinimumHeight(160)
        self._ident_base_host = QWidget()
        self._ident_base_layout = QVBoxLayout(self._ident_base_host)
        self._ident_base_layout.setContentsMargins(0, 0, 0, 0)
        self._ident_base_layout.setSpacing(4)
        self._ident_base_scroll.setWidget(self._ident_base_host)
        ident_strength.addRow(self._ident_base_scroll)

        plots = self._add_page("post_plots")
        self._fit_tests_status = QLabel(
            "Tick the tests in the combination. After Identify, every "
            "non-empty subset is already available; ticking applies it instantly."
        )
        self._fit_tests_status.setWordWrap(True)
        plots.addRow(self._fit_tests_status)
        self._fit_tests_scroll = QScrollArea()
        self._fit_tests_scroll.setWidgetResizable(True)
        self._fit_tests_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._fit_tests_scroll.setMinimumHeight(160)
        self._fit_tests_host = QWidget()
        self._fit_tests_layout = QVBoxLayout(self._fit_tests_host)
        self._fit_tests_layout.setContentsMargins(0, 0, 0, 0)
        self._fit_tests_layout.setSpacing(4)
        self._fit_tests_scroll.setWidget(self._fit_tests_host)
        plots.addRow(self._fit_tests_scroll)
        fit_buttons = QWidget()
        fit_row = QHBoxLayout(fit_buttons)
        fit_row.setContentsMargins(0, 0, 0, 0)
        self._fit_tests_all = QPushButton("All")
        self._fit_tests_all.clicked.connect(lambda: self._set_all_fit_tests(True))
        self._fit_tests_none = QPushButton("None")
        self._fit_tests_none.clicked.connect(lambda: self._set_all_fit_tests(False))
        fit_row.addWidget(self._fit_tests_all)
        fit_row.addWidget(self._fit_tests_none)
        plots.addRow(fit_buttons)

        parameter_space = self._add_page("post_parameter_space")
        self._ps_status = QLabel(
            "Each point is one non-empty combination of the identified tests, "
            "using the virtual fields from Identify. The star is all tests."
        )
        self._ps_status.setWordWrap(True)
        parameter_space.addRow(self._ps_status)
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
        parameter_space.addRow("Left plot X", self._ps_left_x)
        parameter_space.addRow("Left plot Y", self._ps_left_y)
        parameter_space.addRow("Right plot X", self._ps_right_x)
        parameter_space.addRow("Right plot Y", self._ps_right_y)
        self._ps_groups_hint = QLabel(
            "Colour every combination that includes all ticked tests. "
            "Ticking more tests keeps the intersection, so fewer points "
            "stay highlighted. If several groups match, the stricter one wins."
        )
        self._ps_groups_hint.setWordWrap(True)
        self._ps_groups_hint.setProperty("role", "hint")
        parameter_space.addRow(self._ps_groups_hint)
        self._ps_groups_scroll = QScrollArea()
        self._ps_groups_scroll.setWidgetResizable(True)
        self._ps_groups_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._ps_groups_scroll.setMinimumHeight(200)
        self._ps_groups_host = QWidget()
        self._ps_groups_layout = QVBoxLayout(self._ps_groups_host)
        self._ps_groups_layout.setContentsMargins(0, 0, 0, 0)
        self._ps_groups_layout.setSpacing(4)
        self._ps_groups_scroll.setWidget(self._ps_groups_host)
        parameter_space.addRow(self._ps_groups_scroll)
        self._ps_add_group = QPushButton("Add group")
        self._ps_add_group.clicked.connect(self._on_ps_add_group)
        parameter_space.addRow(self._ps_add_group)
        self._ps_group_stat_labels: list[QLabel] = []

        uq = self._add_page("uncertainty")
        self._uq_status = QLabel("No uncertainty analysis.")
        self._uq_status.setWordWrap(True)
        uq.addRow(self._uq_status)
        self._uq_kind = QComboBox()
        self._uq_kind.addItems(["CR2", "CR3", "CR0"])
        self._uq_kind.currentTextChanged.connect(self._on_uq_kind)
        uq.addRow("Sandwich correction", self._uq_kind)
        self._uq_recompute = QPushButton("Recompute")
        self._uq_recompute.clicked.connect(self.request_uncertainty.emit)
        uq.addRow(self._uq_recompute)
        self._uq_refit = QPushButton("Refit FGLS on deletion")
        self._uq_refit.clicked.connect(self.request_loo_refit.emit)
        uq.addRow(self._uq_refit)

    def _page_name(self, kind: str) -> str:
        if is_uncertainty_kind(kind):
            return "uncertainty"
        if kind == "post_plots":
            return "post_plots"
        if kind == "post_parameter_space":
            return "post_parameter_space"
        if kind == "post_experiment_check":
            return "post_experiment_check"
        if kind == "post_identifiability_strength":
            return "post_identifiability_strength"
        if is_post_kind(kind):
            return "post_processing"
        if kind in {"project", "experiments"}:
            return "project"
        if kind in {"experiment", "bcs", "bc_group"}:
            return "experiment"
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
            "mesh": "Mesh",
            "bc": "Boundary condition",
            "step": "Load step",
            "steps": "Load steps",
            "empty": "Properties",
            "post_processing": "Post Processing",
            "post_plots": "Coefficient Evolution",
            "post_parameter_space": "Parameter space",
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
        if page_name == "experiment" and session.document.experiment_by_id(selection.experiment_id or 0) is None:
            page_name = "empty"
        title_key = selection.kind if selection.kind in titles else page_name
        self._title.setText(titles.get(title_key, "Properties"))
        self._fill_page(page_name, session, selection)
        self._stack.setCurrentWidget(self._pages[page_name])
        self._shown_key = key

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
        elif page_name == "post_experiment_check":
            self._fill_experiment_check(session)
        elif page_name == "post_identifiability_strength":
            self._fill_identifiability_strength(session)
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

    def _clear_fit_test_rows(self) -> None:
        while self._fit_tests_layout.count():
            item = self._fit_tests_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _fill_fit_tests(self, session: AppSession) -> None:
        self._clear_fit_test_rows()
        experiments = session.document.experiments
        if not experiments:
            self._fit_tests_status.setText(
                "Add experiments first. Tick the tests in the combination "
                "after Identify."
            )
            self._fit_tests_all.setEnabled(False)
            self._fit_tests_none.setEnabled(False)
            return
        self._fit_tests_all.setEnabled(True)
        self._fit_tests_none.setEnabled(True)
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
        from vfmstudio.core.parameter_space import (
            is_full_combination,
            selected_experiment_ids,
            space_from_document,
            theta_for_experiment_ids,
        )

        space = space_from_document(self._session.result, self._session.document)
        ids = selected_experiment_ids(self._session.document)
        theta = theta_for_experiment_ids(space, ids) if space is not None else None
        if self._session.result is None or space is None:
            self._fit_tests_status.setText(
                prefix + " Identify to solve; every combination is then "
                "available by ticking tests."
            )
            return
        if theta is None:
            self._fit_tests_status.setText(
                prefix + " Those tests were not in the last Identify."
            )
            return
        if is_full_combination(space, ids):
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
        self._refresh_fit_tests_status()
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
        _block_set_combo(self._mat_type, material.material_type)
        _block_set_combo(self._mat_analysis, material.analysis_type)
        labels = material.labels()
        values = list(material.initial_parameters)
        while len(values) < len(labels):
            values.append(1.0)
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

    def _fill_experiment(self, session: AppSession, experiment_id: Optional[int]) -> None:
        experiment = session.document.experiment_by_id(experiment_id or 0)
        if experiment is None:
            return
        _block_set_text(self._exp_name, experiment.name)
        _block_set_spin(self._exp_angle, experiment.material_orientation_deg)
        _block_set_spin(self._exp_area, float(experiment.sample_area_total or 0.0))
        _block_set_combo(self._exp_source, experiment.applied_load_source)
        _block_set_checked(self._exp_fit_axes, bool(experiment.fit_data_to_system_axes))
        self._exp_fit_axes.setVisible(True)

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
        self._refresh_mesh_count_summary()

    def _fill_bc(self, session: AppSession, selection: Selection) -> None:
        entry = self._current_bc()
        if entry is None:
            return
        _block_set_text(self._bc_label, entry.label)
        _block_set_combo(self._bc_dof, entry.dof or "W")
        _block_set_combo(self._bc_edge, entry.edge or "")
        is_external = selection.bc_kind == "external"
        self._bc_load.setVisible(is_external)
        self._bc_load_label.setVisible(is_external)
        if is_external:
            _block_set_combo(self._bc_load, entry.load_type or "distributed")

    def _fill_step(self, session: AppSession, selection: Selection) -> None:
        step = self._current_step()
        experiment = self._current_experiment()
        show_step_fields = selection.kind == "step" and step is not None
        self._step_label.setVisible(show_step_fields)
        self._step_label_caption.setVisible(show_step_fields)
        self._step_path.setVisible(show_step_fields)
        self._step_path_row.setVisible(show_step_fields)
        self._step_path_caption.setVisible(show_step_fields)
        if step is None or experiment is None:
            for spin, label in zip(self._step_load_spins, self._step_load_labels):
                spin.setVisible(False)
                label.setVisible(False)
            if experiment is None:
                self._analysis_start.setEnabled(False)
                self._analysis_end.setEnabled(False)
                self._analysis_reference.setEnabled(False)
                self._visualize_steps.setEnabled(False)
                return
        else:
            _block_set_text(self._step_label, step.label)
            _block_set_text(self._step_path, str(step.data_file))
            show_loads = show_step_fields and experiment.applied_load_source == "manual"
            for index, (spin, label) in enumerate(zip(self._step_load_spins, self._step_load_labels)):
                visible = show_loads and index < len(experiment.external)
                spin.setVisible(visible)
                label.setVisible(visible)
                if visible:
                    group = experiment.external[index]
                    label.setText(group.label or f"EXTERNAL {index + 1}")
                    value = 0.0
                    if index < len(step.external_values) and step.external_values[index] is not None:
                        value = float(step.external_values[index])
                    _block_set_spin(spin, value)
        n_steps = len(experiment.steps)
        has_steps = n_steps > 0
        self._analysis_start.setEnabled(has_steps)
        self._analysis_end.setEnabled(has_steps)
        self._analysis_reference.setEnabled(has_steps)
        self._visualize_steps.setEnabled(has_steps)
        if not has_steps:
            for combo in (
                self._analysis_start,
                self._analysis_end,
                self._analysis_reference,
            ):
                combo.blockSignals(True)
                combo.clear()
                combo.blockSignals(False)
            return
        start_value = experiment.resolved_analysis_start()
        end_value = experiment.resolved_analysis_end()
        self._fill_step_index_combo(self._analysis_start, experiment.steps, start_value)
        self._fill_step_index_combo(self._analysis_end, experiment.steps, end_value)
        self._analysis_reference.blockSignals(True)
        self._analysis_reference.clear()
        self._analysis_reference.addItem("None", None)
        for index, item in enumerate(experiment.steps):
            self._analysis_reference.addItem(
                self._step_choice_label(index, item),
                index + 1,
            )
        reference = experiment.reference_step
        if reference is None:
            self._analysis_reference.setCurrentIndex(0)
        else:
            match = self._analysis_reference.findData(int(reference))
            self._analysis_reference.setCurrentIndex(match if match >= 0 else 0)
        self._analysis_reference.blockSignals(False)

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
        if self._session is None:
            return
        self._session.document.material.material_type = text
        self._notify()
        self._fill_material(self._session)

    def _on_analysis_type(self, text: str) -> None:
        if self._session is None:
            return
        self._session.document.material.analysis_type = text
        self._notify()

    def _on_theta_finished(self) -> None:
        if self._session is None:
            return
        material = self._session.document.material
        labels = material.labels()
        current = list(material.initial_parameters)
        while len(current) < len(labels):
            current.append(1.0)
        for index in range(len(labels)):
            current[index] = self._theta_spins[index].value()
        material.initial_parameters = current
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

    def _on_exp_source(self, text: str) -> None:
        experiment = self._current_experiment()
        if experiment is None:
            return
        experiment.applied_load_source = text
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

    def _on_add_bc(self) -> None:
        experiment = self._current_experiment()
        if experiment is None:
            return
        self.request_add_bc.emit(experiment.experiment_id, "external")

    def _on_add_step(self) -> None:
        experiment = self._current_experiment()
        if experiment is None:
            return
        self.request_add_step.emit(experiment.experiment_id)

    def _on_mesh_nx(self, value: int) -> None:
        experiment = self._current_experiment()
        if experiment is None:
            return
        experiment.mesh.n_elements_x = int(value)
        self._notify()
        self._refresh_mesh_count_summary()

    def _on_mesh_ny(self, value: int) -> None:
        experiment = self._current_experiment()
        if experiment is None:
            return
        experiment.mesh.n_elements_y = int(value)
        self._notify()
        self._refresh_mesh_count_summary()

    def _on_mesh_bound(self, attr: str) -> None:
        experiment = self._current_experiment()
        if experiment is None:
            return
        setattr(experiment.mesh.bounds, attr, float(self._bound_spins[attr].value()))
        self._notify()
        self._refresh_mesh_count_summary()

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

    def _on_step_load(self) -> None:
        step = self._current_step()
        experiment = self._current_experiment()
        if step is None or experiment is None:
            return
        values = list(step.external_values)
        for index in range(len(experiment.external)):
            while len(values) <= index:
                values.append(None)
            if index < len(self._step_load_spins) and self._step_load_spins[index].isVisible():
                values[index] = float(self._step_load_spins[index].value())
        step.external_values = values
        self._notify()

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
