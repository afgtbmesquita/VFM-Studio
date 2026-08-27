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
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from vfmstudio.core import mesh_convert
from vfmstudio.core.fields import sync_mesh_bounds_from_file
from vfmstudio.core.step_series import step_display_label
from vfmstudio.gui.session import AppSession, Selection


def _block_set_text(widget: QLineEdit, text: str) -> None:
    widget.blockSignals(True)
    widget.setText(text)
    widget.blockSignals(False)


def _block_set_combo(widget: QComboBox, text: str) -> None:
    widget.blockSignals(True)
    widget.setCurrentText(text)
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
        self._analysis_start = QSpinBox()
        self._analysis_start.setRange(1, 1)
        self._analysis_start.valueChanged.connect(self._on_analysis_start)
        self._analysis_end = QSpinBox()
        self._analysis_end.setRange(1, 1)
        self._analysis_end.valueChanged.connect(self._on_analysis_end)
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

    def _page_name(self, kind: str) -> str:
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
        }
        if page_name == "experiment" and session.document.experiment_by_id(selection.experiment_id or 0) is None:
            page_name = "empty"
        title_key = selection.kind if selection.kind == "steps" else page_name
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
            return
        self._analysis_start.blockSignals(True)
        self._analysis_end.blockSignals(True)
        self._analysis_start.setRange(1, n_steps)
        self._analysis_end.setRange(1, n_steps)
        self._analysis_start.setValue(experiment.resolved_analysis_start())
        self._analysis_end.setValue(experiment.resolved_analysis_end())
        self._analysis_start.blockSignals(False)
        self._analysis_end.blockSignals(False)
        self._analysis_reference.blockSignals(True)
        self._analysis_reference.clear()
        self._analysis_reference.addItem("None", None)
        for index, item in enumerate(experiment.steps):
            self._analysis_reference.addItem(
                f"{index + 1}: {step_display_label(item)}",
                index + 1,
            )
        reference = experiment.reference_step
        if reference is None:
            self._analysis_reference.setCurrentIndex(0)
        else:
            match = self._analysis_reference.findData(int(reference))
            self._analysis_reference.setCurrentIndex(match if match >= 0 else 0)
        self._analysis_reference.blockSignals(False)

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

    def _on_analysis_start(self, value: int) -> None:
        experiment = self._current_experiment()
        if experiment is None:
            return
        experiment.analysis_start = int(value)
        self._notify()

    def _on_analysis_end(self, value: int) -> None:
        experiment = self._current_experiment()
        if experiment is None:
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
