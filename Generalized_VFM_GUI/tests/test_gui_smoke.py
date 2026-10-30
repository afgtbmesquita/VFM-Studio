"""GUI smoke test (offscreen)."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("VFMSTUDIO_NO_VISTA", "1")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from vfmstudio.gui.main_window import MainWindow
from vfmstudio.gui.session import Selection


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _top_level_labels(window) -> list[str]:
    tree = window.tree.tree
    return [tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())]


def _child_labels(item) -> list[str]:
    return [item.child(i).text(0) for i in range(item.childCount())]


def test_main_window_constructs(qapp):
    from vfmstudio.resources import app_icon

    window = MainWindow()
    assert "abdXtract" in window.windowTitle()
    assert not window.windowIcon().isNull()
    assert not app_icon().isNull()
    assert hasattr(window.props, "request_visualize_steps")
    assert window.viewport.load_plot_visible() is False
    assert hasattr(window.viewport, "_load_plot")
    overlay = window.viewport._load_plot
    assert overlay._plot_canvas is not None
    assert overlay._spec_canvas is not None
    from PySide6.QtGui import QPalette

    fill = overlay._force.palette().color(QPalette.ColorRole.Base).name().lower()
    assert fill == "#e6e6e6"
    xy_axes = window.actions.get("xy_axes")
    assert xy_axes.isCheckable()
    assert xy_axes.isChecked() is False
    assert hasattr(window.props, "_exp_fit_axes")
    assert hasattr(window.props, "_exp_area_estimate")
    assert window.props._exp_area_estimate.text() == "From DIC"
    assert hasattr(window.props, "_exp_show_deformed")
    assert hasattr(window.props, "_exp_deform_slider")
    assert hasattr(window.props, "_exp_deform_spin")
    assert hasattr(window.props, "deformation_scale_changed")
    assert hasattr(window.props, "view_changed")
    assert hasattr(window.props, "_mesh_show_counts")
    assert window.props._mesh_show_counts.isChecked() is False
    assert "Add load steps" in window.actions.get("add_step").text()
    assert hasattr(window.viewport, "_empty_page")
    assert hasattr(window.props, "_exp_include")
    assert hasattr(window.props, "_sol_cov")
    assert hasattr(window.props, "_bc_value")
    assert hasattr(window.props, "_exp_load_editor")
    assert hasattr(window.props, "_step_load_editor")
    assert window.props._exp_source.itemData(0) == "mat"
    assert window.props._exp_source.itemData(1) == "manual"
    from PySide6.QtWidgets import QScrollArea

    assert isinstance(window.props._pages["step"], QScrollArea)
    assert window.props._pages["step"].widgetResizable() is True
    assert isinstance(window.props._pages["experiment"], QScrollArea)
    assert hasattr(window, "module_bar")
    assert hasattr(window, "dock_jobs")
    assert hasattr(window, "jobs")
    assert hasattr(window.props, "_bc_pick_nodes")
    assert window.actions.get("plots_fill").isCheckable()
    assert window.actions.get("plots_fill").isChecked() is False
    window.close()


def test_toolbar_actions_look_like_buttons(qapp):
    from PySide6.QtWidgets import QLabel, QToolButton

    window = MainWindow()
    labels = [
        widget.text().strip()
        for widget in window._main_bar.findChildren(QLabel)
    ]
    assert "min" in labels
    assert "max" in labels
    for label in window._main_bar.findChildren(QLabel):
        if label.text().strip() in {"min", "max"}:
            assert label.objectName() == "toolLabel"
    main_buttons = [
        widget
        for widget in window._main_bar.findChildren(QToolButton)
        if widget.text() == "Reset range"
    ]
    assert main_buttons
    assert main_buttons[0].autoRaise() is False
    window.set_module("model")
    context_buttons = [
        widget
        for widget in window._context.findChildren(QToolButton)
        if "experiment" in widget.text().lower()
    ]
    assert context_buttons
    assert context_buttons[0].autoRaise() is False
    window.set_module("mesh")
    mesh_texts = [widget.text() for widget in window._context.findChildren(QToolButton)]
    assert "Previous step" in mesh_texts
    assert "Next step" in mesh_texts
    window.session.dirty = False
    window.close()


def test_post_processing_replaces_bottom_result_docks(qapp):
    window = MainWindow()
    assert hasattr(window, "dock_log")
    assert not hasattr(window, "dock_plots")
    assert not hasattr(window, "dock_results")
    labels = _top_level_labels(window)
    assert labels[-2:] == ["Solver", "Post Processing"]
    post = window.tree.tree.topLevelItem(labels.index("Post Processing"))
    assert _child_labels(post) == [
        "Coefficient Evolution",
        "Parameter space",
        "Stress resultant space",
        "Individual Experiment check",
        "Identifiability",
        "Identifiability strength",
        "Virtual fields",
        "Uncertainty quantification",
    ]
    uq = post.child(7)
    assert _child_labels(uq) == [
        "Cov(θ)",
        "Experiment uncertainty",
        "Matrices A and B",
        "Information geometry",
        "Leave-one-out",
    ]
    assert window._splitter.widget(0) is window.viewport
    assert window._splitter.widget(1) is window.post
    assert not window.viewport.isHidden()
    window._on_tree_selection(Selection(kind="post_plots"))
    assert not window.post.isHidden()
    assert not window.viewport.isHidden()
    assert window.session.module == "results"
    assert window.post._stack.currentWidget() is window.post.plots
    assert window.props._title.text() == "Tests"
    assert hasattr(window.props, "_fit_tests_scroll")
    assert not hasattr(window.props, "_fit_tests_run")
    assert not hasattr(window.props, "request_identify")
    assert hasattr(window.post.plots, "apply_combination")
    assert not hasattr(window.post, "coeff")
    assert window.props._plots_export.text() == "Export"
    assert not window.props._plots_export.isHidden()
    assert window.post.figure_for_kind("post_plots") is window.post.plots._figure
    window._on_tree_selection(Selection(kind="post_parameter_space"))
    assert window.post._stack.currentWidget() is window.post.parameter_space
    assert window.props._title.text() == "Axes"
    assert hasattr(window.props, "_ps_left_x")
    assert hasattr(window.props, "_ps_right_y")
    assert window.props._ps_export.text() == "Export"
    window._on_tree_selection(Selection(kind="post_stress_space"))
    assert window.post._stack.currentWidget() is window.post.stress_space
    assert window.props._title.text() == "View"
    assert hasattr(window.props, "_ss_2d_x")
    assert hasattr(window.props, "_ss_3d_z")
    assert hasattr(window.props, "_ss_experiment")
    assert hasattr(window.props, "_ss_step")
    assert hasattr(window.props, "_ss_color")
    assert window.props._ss_export.text() == "Export"
    assert window.post.figure_for_kind("post_stress_space") is window.post.stress_space._figure
    assert [window.props._ss_2d_x.itemText(i) for i in range(window.props._ss_2d_x.count())] == [
        "Mx",
        "My",
        "Mxy",
    ]
    assert window.props._ss_2d_x.currentText() == "Mx"
    assert window.props._ss_2d_y.currentText() == "My"
    assert window.props._ss_3d_z.currentText() == "Mxy"
    assert window.props._ss_step.itemText(0) == "All load steps"
    window._on_tree_selection(Selection(kind="post_experiment_check"))
    assert window.post._stack.currentWidget() is window.post.experiment_check
    assert window.props._title.text() == "Display"
    assert hasattr(window.props, "_exp_check_status")
    assert window.props._exp_check_export.text() == "Export"
    window._on_tree_selection(Selection(kind="post_identifiability"))
    assert window.props._title.text() == "Status"
    assert window.props._post_export.isHidden()
    assert window.post.figure_for_kind("post_identifiability") is None
    window._on_tree_selection(Selection(kind="post_identifiability_strength"))
    assert window.post._stack.currentWidget() is window.post.ident_strength
    assert window.props._title.text() == "Baseline"
    assert hasattr(window.props, "_ident_base_scroll")
    assert hasattr(window.props, "_ident_shift_scale")
    assert window.props._ident_strength_export.text() == "Export"
    window._on_tree_selection(Selection(kind="post_virtual_fields"))
    assert not window.post.isHidden()
    assert not window.viewport.isHidden()
    assert window.post._stack.currentWidget() is window.post.vf_panel
    assert window.props._title.text() == "View"
    assert hasattr(window.props, "_vf_experiment")
    assert hasattr(window.props, "_vf_step")
    assert hasattr(window.props, "_vf_mode")
    assert hasattr(window.props, "_vf_quantity")
    assert not window.props._vf_export.isHidden()
    assert window.post.figure_for_kind("post_virtual_fields") is window.post.vf_panel._figure
    window._on_tree_selection(Selection(kind="post_uncertainty"))
    assert not window.post.isHidden()
    assert window.post._stack.currentWidget() is window.post.uq_landing
    assert window.props._title.text() == "Overview"
    assert "M ≥ p + 1" in window.post.uq_landing._verdict.text()
    assert window.props._uq_export.isHidden()
    assert [window.post.uq_info._mode.itemText(i) for i in range(window.post.uq_info._mode.count())] == [
        "cond(A − Aᵢ)",
        "Share of A",
        "Eigenvalues of A",
        "Constrained combinations",
    ]
    window._on_tree_selection(Selection(kind="post_uncertainty_cov"))
    assert window.post._stack.currentWidget() is window.post.uq_cov
    assert window.props._title.text() == "View"
    assert not window.props._uq_export.isHidden()
    window._on_tree_selection(Selection(kind="post_uncertainty_experiments"))
    assert window.post._stack.currentWidget() is window.post.uq_experiments
    assert window.props._title.text() == "View"
    window._on_tree_selection(Selection(kind="post_uncertainty_ab"))
    assert window.post._stack.currentWidget() is window.post.uq_ab
    assert window.props._title.text() == "View"
    window._on_tree_selection(Selection(kind="post_uncertainty_info"))
    assert window.post._stack.currentWidget() is window.post.uq_info
    assert window.props._title.text() == "View"
    window._on_tree_selection(Selection(kind="post_uncertainty_loo"))
    assert window.post._stack.currentWidget() is window.post.uq_loo
    assert window.props._title.text() == "View"
    assert window.props._uq_export.text() == "Export"
    assert hasattr(window.props, "_uq_kind")
    assert [window.props._uq_kind.itemText(i) for i in range(window.props._uq_kind.count())] == [
        "CR2",
        "CR3",
        "CR0",
    ]
    from vfmstudio.gui.results.plot_style import SERIES_COLOUR
    from vfmstudio.gui.results.uncertainty_plots import CR3, ROBUST, WORK

    assert "#F0E442" not in SERIES_COLOUR
    assert "#2a78d6" not in SERIES_COLOUR
    assert WORK.startswith("#")
    assert ROBUST.startswith("#")
    assert CR3.startswith("#")
    assert [window.props._vf_mode.itemText(i) for i in range(window.props._vf_mode.count())] == [
        "3D surface",
        "2D contour",
    ]
    assert [window.props._vf_quantity.itemText(i) for i in range(window.props._vf_quantity.count())] == [
        "w*",
        "kx*",
        "ky*",
        "kxy*",
    ]
    assert hasattr(window.props, "_vf_unknown")
    assert hasattr(window.props, "_exp_check_mode")
    assert hasattr(window.props, "_uq_plot_mode")
    assert hasattr(window.props, "_mesh_fit")
    assert hasattr(window.props, "_mesh_clear")
    assert hasattr(window.props, "_mesh_free")
    assert hasattr(window.props, "_mesh_drag")
    assert hasattr(window.props, "_mesh_construct")
    assert hasattr(window.props, "_bc_copy")
    window._on_tree_selection(Selection(kind="post_experiment_check"))
    assert [window.props._exp_check_mode.itemText(i) for i in range(window.props._exp_check_mode.count())] == [
        "Coefficients vs load step",
        "One panel per experiment",
        "Deviation from θ̂",
    ]
    assert window.post.experiment_check._mode.isHidden()
    window._on_tree_selection(Selection(kind="post_uncertainty_info"))
    assert window.props._uq_plot_mode.count() == 4
    assert window.post.uq_info._mode.isHidden()
    assert not hasattr(window.post.vf_panel, "_mode")
    assert window.field_combo.isEnabled() is True
    assert window.display_combo.isEnabled() is True
    window._on_tree_selection(Selection(kind="mesh"))
    assert not window.viewport.isHidden()
    assert window.session.module == "results"
    assert not window.post.isHidden()
    window.set_module("model")
    assert window.post.isHidden()
    assert window.field_combo.isEnabled() is True
    plots = window.post.plots
    assert not hasattr(plots, "_mode")
    assert hasattr(plots, "_canvas")
    assert "Coefficient evolution" in plots._title.text()
    from vfmstudio.gui.virtual_fields_view import (
        _3d_gridspec_margins,
        _tile_shape,
        collect_virtual_fields,
    )

    assert _tile_shape(1) == (1, 1)
    assert _tile_shape(2) == (1, 2)
    assert _tile_shape(4) == (2, 2)
    assert _tile_shape(6) == (2, 3)
    assert _3d_gridspec_margins(6)["bottom"] >= 0.14
    assert _3d_gridspec_margins(1)["bottom"] >= 0.12

    class _Field:
        def __init__(self, values):
            self.nodal_variables_k = values

    class _Model:
        parameter_labels = ["D11", "D22", "D12", "D66"]

    class _Result:
        optimal_field_results = [
            _Field([0.0, 1.0]),
            _Field([2.0, 3.0]),
            _Field([4.0, 5.0]),
            _Field([6.0, 7.0]),
        ]
        constitutive_model = _Model()

    fields, names = collect_virtual_fields(_Result(), None)
    assert len(fields) == 4
    assert names == ["D11", "D22", "D12", "D66"]
    window.close()


def test_refresh_all_does_not_fill_hidden_result_pages(qapp, monkeypatch):
    window = MainWindow()
    ps, ss, vf = [], [], []
    monkeypatch.setattr(
        window.post.parameter_space,
        "update_from",
        lambda *args, **kwargs: ps.append(1),
    )
    monkeypatch.setattr(
        window.post.stress_space,
        "update_from",
        lambda *args, **kwargs: ss.append(1),
    )
    monkeypatch.setattr(
        window.post.vf_panel,
        "update_from",
        lambda *args, **kwargs: vf.append(1),
    )
    window._refresh_all()
    assert ps == []
    assert ss == []
    assert vf == []
    window._on_tree_selection(Selection(kind="post_parameter_space"))
    assert ps == [1]
    assert ss == []
    assert vf == []
    window._on_tree_selection(Selection(kind="post_stress_space"))
    assert ss == [1]
    assert vf == []
    window._on_tree_selection(Selection(kind="post_virtual_fields"))
    assert vf == [1]
    window.close()


def test_tree_rebuild_skips_when_structure_unchanged(qapp):
    window = MainWindow()
    first = window.tree.tree.topLevelItem(0)
    window.tree.rebuild(window.session)
    assert window.tree.tree.topLevelItem(0) is first
    window.close()


def test_virtual_fields_tiled_draw_all_parameters(qapp):
    from types import SimpleNamespace

    import numpy as np

    from vfmstudio.core import mesh_convert
    from vfmstudio.core.document import ExperimentModel, MeshModel, empty_document
    from vfmstudio.gui.virtual_fields_view import VirtualFieldTiledView

    document = empty_document()
    document.material.analysis_type = "out_of_plane"
    document.experiments.append(
        ExperimentModel(experiment_id=1, mesh=MeshModel(n_elements_x=3, n_elements_y=3))
    )
    mesh_pkg = mesh_convert.build_experiment_mesh(document, document.experiments[0])
    n_dof = int(mesh_pkg.n_dofs)
    labels = ["D11", "D12", "D22", "D66", "D16", "D26"]
    rng = np.random.default_rng(0)
    result = SimpleNamespace(
        optimal_field_results=[
            SimpleNamespace(nodal_variables_k=rng.normal(size=n_dof))
            for _ in labels
        ],
        constitutive_model=SimpleNamespace(parameter_labels=labels),
    )
    view = VirtualFieldTiledView()
    view.update_from(document, result, warp_scale=1.0)
    axes_3d = [ax for ax in view._figure.axes if ax.name == "3d"]
    assert len(axes_3d) == 6
    title_axes = [
        ax for ax in view._figure.axes
        if ax.name != "3d" and ax.texts and not ax.axison
    ]
    assert len(title_axes) == 6
    aspect = axes_3d[0].get_box_aspect()
    assert aspect is not None
    ax_r, ay_r, az_r = (float(v) for v in aspect)
    assert az_r / max(ax_r, ay_r) < 1.2
    zlabels = [str(ax.get_zlabel() or "") for ax in axes_3d]
    assert all(label.strip() for label in zlabels)
    # Bottom-row x-axis labels sit inside the figure, not under the Jobs dock.
    assert min(ax.get_position().y0 for ax in axes_3d) >= 0.12
    assert view._canvas.minimumHeight() >= 560
    # Colorbar axes sit to the right of their 3D plot and stay on-canvas.
    for axis in axes_3d:
        plot = axis.get_position()
        siblings = [
            cax for cax in view._figure.axes
            if cax.name != "3d"
            and cax.axison
            and abs(cax.get_position().y0 + 0.5 * cax.get_position().height
                    - (plot.y0 + 0.5 * plot.height)) < 0.08
        ]
        assert siblings
        cax = min(siblings, key=lambda a: abs(a.get_position().x0 - plot.x1))
        cpos = cax.get_position()
        assert cpos.x0 >= plot.x1 - 1.0e-6
        assert cpos.x1 <= 0.99
    view.update_from(document, result, warp_scale=1.0, view_mode="2D contour")
    titled_2d = [
        ax for ax in view._figure.axes
        if ax.name != "3d" and ax.get_title()
    ]
    assert len(titled_2d) == 6
    plot_pos = titled_2d[0].get_position()
    colorbars = [
        ax for ax in view._figure.axes
        if ax.name != "3d" and not ax.get_title() and ax.axison
    ]
    assert colorbars
    gaps = [
        cax.get_position().x0 - plot_pos.x1
        for cax in colorbars
        if abs(cax.get_position().y0 - plot_pos.y0) < 0.05
    ]
    assert gaps and min(gaps) < 0.025
    view.update_from(document, result, warp_scale=1.0, view_mode="2D contour", quantity="kx*")
    titled_kx = [
        ax for ax in view._figure.axes
        if ax.name != "3d" and ax.get_title()
    ]
    assert len(titled_kx) == 6
    view.close()


def test_coefficient_evolution_selects_tests_for_identification(qapp):
    from PySide6.QtWidgets import QCheckBox

    from vfmstudio.core.document import ExperimentModel
    from vfmstudio.gui.docks.property_editor import PropertyEditorDock
    from vfmstudio.gui.session import AppSession

    session = AppSession()
    session.document.experiments = [
        ExperimentModel(experiment_id=1, name="A1_PB"),
        ExperimentModel(experiment_id=2, name="B1_PB"),
        ExperimentModel(experiment_id=3, name="A1_Twist"),
    ]
    dock = PropertyEditorDock()
    dock.show_selection(session, Selection(kind="post_plots"))
    checks = [
        dock._fit_tests_layout.itemAt(index).widget()
        for index in range(dock._fit_tests_layout.count())
        if isinstance(dock._fit_tests_layout.itemAt(index).widget(), QCheckBox)
    ]
    assert [box.text().split()[0] for box in checks] == ["A1_PB", "B1_PB", "A1_Twist"]
    assert all(box.isChecked() for box in checks)
    assert "all tests" in dock._fit_tests_status.text()
    checks[1].setChecked(False)
    checks[2].setChecked(False)
    included = session.document.experiments_in_identification()
    assert [item.experiment_id for item in included] == [1]
    assert "1 test" in dock._fit_tests_status.text()
    dock._set_all_fit_tests(True)
    assert all(
        experiment.include_in_identification
        for experiment in session.document.experiments
    )
    dock.close()


def test_coefficient_evolution_applies_frozen_combination(qapp):
    import numpy as np

    from vfmstudio.core.document import ExperimentModel, empty_document
    from vfmstudio.core.uncertainty import system_from_blocks
    from vfmstudio.gui.docks.plot_dock import PlotDock

    rng = np.random.default_rng(8)
    theta = rng.normal(size=4)
    designs, observations, covariances, names = [], [], [], []
    for index in range(3):
        design = np.eye(4) + 0.05 * rng.normal(size=(4, 4))
        covariance = np.eye(4)
        observation = design @ theta + 0.1 * rng.normal(size=4)
        designs.append(design)
        observations.append(observation)
        covariances.append(covariance)
        names.append(f"E{index + 1}")
    system = system_from_blocks(
        designs, observations, covariances, theta, ["D11", "D12", "D22", "D66"], names
    )
    document = empty_document()
    document.experiments = [
        ExperimentModel(experiment_id=1, name="A1_PB"),
        ExperimentModel(experiment_id=2, name="B1_PB"),
        ExperimentModel(experiment_id=3, name="A1_Twist"),
    ]
    document.experiments[2].include_in_identification = False
    dock = PlotDock()
    dock.set_results(system, document=document)
    assert "Frozen GLS" in dock._subtitle.text()
    assert "A1_PB" in dock._subtitle.text()
    assert "B1_PB" in dock._subtitle.text()
    document.experiments[2].include_in_identification = True
    dock.apply_combination(document)
    assert "Frozen GLS" not in dock._subtitle.text()
    dock.close()


def test_unchecking_tests_updates_coefficient_evolution_plot(qapp):
    import numpy as np
    from PySide6.QtWidgets import QCheckBox

    from vfmstudio.core.document import ExperimentModel
    from vfmstudio.core.uncertainty import system_from_blocks
    from vfmstudio.gui.docks.plot_dock import PlotDock
    from vfmstudio.gui.docks.property_editor import PropertyEditorDock
    from vfmstudio.gui.session import AppSession

    rng = np.random.default_rng(8)
    theta = rng.normal(size=4)
    designs, observations, covariances, names = [], [], [], []
    for index in range(3):
        design = np.eye(4) + 0.05 * rng.normal(size=(4, 4))
        covariance = np.eye(4)
        observation = design @ theta + 0.1 * rng.normal(size=4)
        designs.append(design)
        observations.append(observation)
        covariances.append(covariance)
        names.append(f"E{index + 1}")
    system = system_from_blocks(
        designs, observations, covariances, theta, ["D11", "D12", "D22", "D66"], names
    )
    session = AppSession()
    session.document.experiments = [
        ExperimentModel(experiment_id=1, name="A1_PB"),
        ExperimentModel(experiment_id=2, name="B1_PB"),
        ExperimentModel(experiment_id=3, name="A1_Twist"),
    ]
    session.result = system
    plots = PlotDock()
    plots.set_results(system, document=session.document)
    props = PropertyEditorDock()
    props.view_changed.connect(lambda: plots.apply_combination(session.document))
    props.show_selection(session, Selection(kind="post_plots"))
    assert "Frozen GLS" not in plots._subtitle.text()
    checks = [
        props._fit_tests_layout.itemAt(index).widget()
        for index in range(props._fit_tests_layout.count())
        if isinstance(props._fit_tests_layout.itemAt(index).widget(), QCheckBox)
    ]
    assert len(checks) == 3
    checks[-1].setChecked(False)
    assert session.document.experiments[2].include_in_identification is False
    assert "Frozen GLS" in plots._subtitle.text()
    assert "Selected combination" in plots._title.text()
    assert any(
        "Shift from Identify" in axis.get_title() for axis in plots._figure.axes
    )
    assert any(
        list(axis.get_xticklabels())
        and "Identify" in "".join(tick.get_text() for tick in axis.get_xticklabels())
        for axis in plots._figure.axes
    )
    plots.close()
    props.close()


def test_coefficient_evolution_twelve_tests_toggle(qapp):
    import numpy as np
    from PySide6.QtWidgets import QCheckBox

    from vfmstudio.core.document import ExperimentModel
    from vfmstudio.core.uncertainty import system_from_blocks
    from vfmstudio.gui.docks.plot_dock import PlotDock
    from vfmstudio.gui.docks.property_editor import PropertyEditorDock
    from vfmstudio.gui.session import AppSession

    rng = np.random.default_rng(21)
    n_tests, n_par = 12, 4
    theta = rng.normal(size=n_par)
    designs, observations, covariances, names = [], [], [], []
    for index in range(n_tests):
        design = np.eye(n_par) + 0.04 * rng.normal(size=(n_par, n_par))
        covariance = np.eye(n_par)
        observation = design @ theta + 0.05 * rng.normal(size=n_par)
        designs.append(design)
        observations.append(observation)
        covariances.append(covariance)
        names.append(f"T{index + 1}")
    system = system_from_blocks(
        designs, observations, covariances, theta, ["D11", "D12", "D22", "D66"], names
    )
    session = AppSession()
    session.document.experiments = [
        ExperimentModel(experiment_id=index + 1, name=f"T{index + 1}")
        for index in range(n_tests)
    ]
    session.result = system
    plots = PlotDock()
    plots.set_results(system, document=session.document)
    props = PropertyEditorDock()
    props.view_changed.connect(lambda: plots.apply_combination(session.document))
    props.show_selection(session, Selection(kind="post_plots"))
    checks = [
        props._fit_tests_layout.itemAt(index).widget()
        for index in range(props._fit_tests_layout.count())
        if isinstance(props._fit_tests_layout.itemAt(index).widget(), QCheckBox)
    ]
    assert len(checks) == 12
    assert "Frozen GLS" not in plots._subtitle.text()
    checks[-1].setChecked(False)
    checks[-2].setChecked(False)
    assert session.document.experiments[-1].include_in_identification is False
    assert "Frozen GLS" in plots._subtitle.text()
    assert "Selected combination" in plots._title.text()
    assert any("Shift from Identify" in axis.get_title() for axis in plots._figure.axes)
    props._set_all_fit_tests(True)
    assert "Frozen GLS" not in plots._subtitle.text()
    assert "FGLS loop" in plots._title.text()
    plots.close()
    props.close()


def test_parameter_space_properties_choose_axes(qapp):
    from vfmstudio.gui.docks.property_editor import PropertyEditorDock
    from vfmstudio.gui.session import AppSession

    session = AppSession()
    dock = PropertyEditorDock()
    dock.show_selection(session, Selection(kind="post_parameter_space"))
    assert dock._title.text() == "Axes"
    assert dock._ps_left_x.count() >= 2
    assert dock._ps_left_x.currentText()
    assert dock._ps_left_y.currentText()
    assert session.parameter_space_left_x
    assert session.parameter_space_left_y
    first = dock._ps_left_x.itemText(0)
    other = dock._ps_left_x.itemText(1)
    dock._ps_left_x.setCurrentText(other if session.parameter_space_left_x == first else first)
    assert session.parameter_space_left_x in {first, other}
    dock.close()


def test_parameter_space_properties_highlight_groups(qapp):
    import numpy as np
    from PySide6.QtWidgets import QCheckBox

    from vfmstudio.core.document import ExperimentModel
    from vfmstudio.core.uncertainty import system_from_blocks
    from vfmstudio.gui.docks.property_editor import PropertyEditorDock
    from vfmstudio.gui.session import AppSession

    session = AppSession()
    session.document.experiments = [
        ExperimentModel(experiment_id=1, name="A1_PB"),
        ExperimentModel(experiment_id=2, name="A2_PB"),
        ExperimentModel(experiment_id=3, name="B1_Twist"),
    ]
    dock = PropertyEditorDock()
    dock.show_selection(session, Selection(kind="post_parameter_space"))
    assert dock._ps_add_group.isEnabled()
    assert session.parameter_space_groups == []
    dock._ps_add_group.click()
    assert len(session.parameter_space_groups) == 1
    assert session.parameter_space_groups[0]["experiment_ids"] == []
    card = dock._ps_groups_layout.itemAt(0).widget()
    boxes = card.findChildren(QCheckBox)
    assert [box.text() for box in boxes] == ["A1_PB", "A2_PB", "B1_Twist"]
    boxes[0].setChecked(True)
    boxes[1].setChecked(True)
    assert set(session.parameter_space_groups[0]["experiment_ids"]) == {1, 2}
    assert "Identify" in dock._ps_group_stat_labels[0].text()
    rng = np.random.default_rng(0)
    theta = np.array([500.0, 20.0, 400.0, 30.0])
    designs, observations, covariances, names = [], [], [], []
    for index in range(3):
        design = np.eye(4) + 0.05 * rng.normal(size=(4, 4))
        covariance = np.eye(4)
        observation = design @ theta + 0.1 * rng.normal(size=4)
        designs.append(design)
        observations.append(observation)
        covariances.append(covariance)
        names.append(["A1_PB", "A2_PB", "B1_Twist"][index])
    session.result = system_from_blocks(
        designs, observations, covariances, theta, ["D11", "D12", "D22", "D66"], names
    )
    dock.show_selection(session, Selection(kind="post_parameter_space"))
    stats = dock._ps_group_stat_labels[0].text()
    assert "2 combinations" in stats
    assert "±" in stats
    assert "D11" in stats
    dock.close()


def test_parameter_space_panel_draws_two_plots(qapp):
    import numpy as np

    from vfmstudio.core.parameter_space import from_system
    from vfmstudio.core.uncertainty import system_from_blocks
    from vfmstudio.gui.results.parameter_space import ParameterSpacePanel

    rng = np.random.default_rng(0)
    theta = np.array([500.0, 20.0, 400.0, 30.0])
    designs, observations, covariances, names = [], [], [], []
    for index in range(3):
        design = np.eye(4) + 0.05 * rng.normal(size=(4, 4))
        covariance = np.eye(4)
        observation = design @ theta + 0.1 * rng.normal(size=4)
        designs.append(design)
        observations.append(observation)
        covariances.append(covariance)
        names.append(f"E{index + 1}")
    system = system_from_blocks(
        designs, observations, covariances, theta, ["D11", "D12", "D22", "D66"], names
    )
    panel = ParameterSpacePanel()
    panel._space = from_system(system)
    panel.refresh()
    assert panel._space.n_subsets == 7
    assert len(panel._axes) == 2
    assert all(xy is not None and len(xy) == 7 for xy in panel._xy)
    panel.set_groups(
        [{"name": "A tests", "color": "#D55E00", "experiment_ids": [1, 2]}]
    )
    panel.refresh()
    assert panel._assigned is not None
    assert int(np.sum(panel._assigned == 0)) == 2
    panel.set_groups(
        [{"name": "A1", "color": "#D55E00", "experiment_ids": [1]}]
    )
    panel.refresh()
    assert int(np.sum(panel._assigned == 0)) == 4
    panel.close()


def test_parameter_space_panel_uses_space_from_document_cache(qapp):
    import numpy as np

    from vfmstudio.core.document import ExperimentModel, empty_document
    from vfmstudio.core.parameter_space import space_from_document
    from vfmstudio.core.uncertainty import system_from_blocks
    from vfmstudio.gui.results.parameter_space import ParameterSpacePanel

    rng = np.random.default_rng(3)
    theta = np.array([1.0, 2.0, 3.0, 4.0])
    designs, observations, covariances, names = [], [], [], []
    for index in range(3):
        design = np.eye(4) + 0.02 * rng.normal(size=(4, 4))
        designs.append(design)
        observations.append(design @ theta)
        covariances.append(np.eye(4))
        names.append(f"E{index + 1}")
    system = system_from_blocks(
        designs, observations, covariances, theta, ["D11", "D12", "D22", "D66"], names
    )
    document = empty_document()
    document.experiments = [
        ExperimentModel(experiment_id=index + 1, name=name)
        for index, name in enumerate(names)
    ]
    panel = ParameterSpacePanel()
    panel.update_from(document, system)
    cached = space_from_document(system, document)
    assert panel._space is cached
    panel.close()


def test_delete_bc_from_tree(qapp, monkeypatch):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QMessageBox

    from vfmstudio.core.document import BoundaryEntry, ExperimentModel, LoadStepModel

    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
    )

    window = MainWindow()
    experiment = ExperimentModel(
        experiment_id=1,
        name="exp",
        prescribed=[
            BoundaryEntry(label="fix W", dof="W", edge="bottom"),
            BoundaryEntry(label="fix RX", dof="W_X", edge="bottom"),
        ],
        linking=[BoundaryEntry(label="link", dof="W", edge="top")],
        external=[
            BoundaryEntry.model_validate(
                {
                    "label": "M1",
                    "dof": "W_Y",
                    "edge": "top",
                    "loadType": "distributed",
                }
            ),
            BoundaryEntry.model_validate(
                {
                    "label": "M2",
                    "dof": "W_Y",
                    "edge": "bottom",
                    "loadType": "distributed",
                }
            ),
        ],
        steps=[LoadStepModel(data_file="a.mat", label="s1", external_values=[10.0, 20.0])],
    )
    window.session.document.experiments = [experiment]
    window._refresh_all()
    assert window.tree.tree.contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu
    assert hasattr(window.tree, "request_delete_bc")
    assert hasattr(window.props, "request_delete_bc")
    assert window.actions.get("delete_bc").isEnabled() is False

    window.session.selection = Selection(
        kind="bc", experiment_id=1, bc_kind="prescribed", bc_index=0
    )
    window._apply_selection()
    assert window.actions.get("delete_bc").isEnabled() is True
    assert window.props._bc_delete.text() == "Delete"

    window.tree.request_delete_bc.emit(1, "prescribed", 0)
    assert [entry.label for entry in experiment.prescribed] == ["fix RX"]
    node = window.tree._find_item(
        Selection(kind="bc", experiment_id=1, bc_kind="linking", bc_index=0)
    )
    assert node is not None
    window.tree._emit_delete(node)
    assert experiment.linking == []
    window._delete_bc(1, "external", 0)
    assert [entry.label for entry in experiment.external] == ["M2"]
    assert experiment.steps[0].external_values == [20.0]
    assert window.session.selection.kind == "bc_group"
    assert window.session.selection.bc_kind == "external"
    assert window.actions.get("delete_bc").isEnabled() is False
    window.session.dirty = False
    window.close()


def test_add_prescribed_linking_and_external_bcs(qapp, monkeypatch):
    from vfmstudio.core.document import BoundaryEntry, ExperimentModel

    window = MainWindow()
    experiment = ExperimentModel(experiment_id=1, name="exp")
    window.session.document.experiments = [experiment]
    window._refresh_all()
    assert list(window.props._add_bc_buttons) == ["prescribed", "linking", "external"]
    assert [button.text() for button in window.props._add_bc_buttons.values()] == [
        "Prescribed",
        "Linking",
        "External",
    ]
    assert hasattr(window.tree, "request_add_bc")
    window._on_tree_selection(Selection(kind="bcs", experiment_id=1))
    assert window.props._title.text() == "Boundary conditions"

    class FakeDialog:
        def __init__(self, parent=None, kind="prescribed", **kwargs):
            self.kind = kind

        def entry(self):
            return BoundaryEntry(label=f"{self.kind} bc", dof="W", edge="top")

    monkeypatch.setattr("vfmstudio.gui.main_window.AddBcDialog", FakeDialog)
    monkeypatch.setattr(window, "_exec_dialog", lambda dialog, name: 1)

    window.props.request_add_bc.emit(1, "prescribed")
    assert [entry.label for entry in experiment.prescribed] == ["prescribed bc"]
    bcs_node = window.tree._find_item(Selection(kind="bcs", experiment_id=1))
    assert bcs_node is not None
    window.tree._emit_add(bcs_node, "linking")
    assert [entry.label for entry in experiment.linking] == ["linking bc"]
    group = window.tree._find_item(
        Selection(kind="bc_group", experiment_id=1, bc_kind="external")
    )
    assert group is not None
    window.tree._emit_add(group, "external")
    assert [entry.label for entry in experiment.external] == ["external bc"]
    assert window.session.selection.kind == "bc_group"
    assert window.session.selection.bc_kind == "external"
    window.session.dirty = False
    window.close()


def test_add_multiple_dofs_appends_each_and_pads_external(qapp, monkeypatch):
    from vfmstudio.core.bc_entries import make_boundary_entries
    from vfmstudio.core.document import ExperimentModel, LoadStepModel

    window = MainWindow()
    experiment = ExperimentModel(experiment_id=1, name="exp")
    window.session.document.experiments = [experiment]
    window._refresh_all()

    class MultiDialog:
        def __init__(self, parent=None, kind="prescribed", **kwargs):
            self.kind = kind

        def entries(self):
            if self.kind == "prescribed":
                return make_boundary_entries(
                    "prescribed", ["W", "W_X", "W_Y", "W_XY"], nodes=[16, 24]
                )
            return make_boundary_entries("external", ["W_Y", "W_X"], edge="top")

    monkeypatch.setattr("vfmstudio.gui.main_window.AddBcDialog", MultiDialog)
    monkeypatch.setattr(window, "_exec_dialog", lambda dialog, name: 1)
    monkeypatch.setattr(window, "_refresh_all", lambda: None)

    window._add_bc(1, "prescribed")
    assert [entry.dof for entry in experiment.prescribed] == ["W", "W_X", "W_Y", "W_XY"]
    assert all(entry.edge is None for entry in experiment.prescribed)
    assert all(entry.nodes == [16, 24] for entry in experiment.prescribed)
    assert window.session.selection.bc_index == 0

    experiment.steps = [LoadStepModel(data_file="unused.mat", external_values=[])]
    window._add_bc(1, "external")
    assert [entry.dof for entry in experiment.external] == ["W_X", "W_Y"]
    assert experiment.steps[0].external_values == [None, None]
    window.session.dirty = False
    window.close()


def test_export_post_plot_writes_pdf_and_png(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._on_tree_selection(Selection(kind="post_plots"))
    pdf = tmp_path / "coefficient_evolution.pdf"
    png = tmp_path / "coefficient_evolution.png"

    monkeypatch.setattr(
        "vfmstudio.gui.main_window.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(pdf), "PDF (*.pdf)"),
    )
    window.export_post_plot()
    assert pdf.is_file()
    assert pdf.read_bytes()[:4] == b"%PDF"

    monkeypatch.setattr(
        "vfmstudio.gui.main_window.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(png), "PNG (*.png)"),
    )
    window.export_post_plot()
    assert png.is_file()
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    window.close()


def test_applied_load_table_edits_every_step(qapp, tmp_path):
    import numpy as np
    from PySide6.QtCore import Qt
    from scipy.io import savemat

    from vfmstudio.core.document import BoundaryEntry, ExperimentModel, LoadStepModel

    forces = [10.0, 20.0, 30.0]
    paths = []
    for index, force in enumerate(forces):
        path = tmp_path / f"step_{index}.mat"
        savemat(path, {"F": np.array([force])})
        paths.append(path)

    window = MainWindow()
    experiment = ExperimentModel(
        experiment_id=1,
        name="exp",
        applied_load_source="mat",
        external=[
            BoundaryEntry(label="M_yy top", dof="W_Y", edge="top", load_type="concentrated")
        ],
        steps=[
            LoadStepModel(data_file=paths[0], label="s1"),
            LoadStepModel(data_file=paths[1], label="s2"),
            LoadStepModel(data_file=paths[2], label="s3"),
        ],
    )
    window.session.document.experiments = [experiment]
    window.session.selection = Selection(kind="steps", experiment_id=1)
    window._refresh_all()

    table = window.props._exp_load_editor.table
    assert table.rowCount() == 3
    assert table.columnCount() == 3
    assert table.horizontalHeaderItem(2).text() == "M_yy top"
    assert table.item(0, 2).text() == "10"
    assert table.item(1, 2).text() == "20"
    assert not bool(table.item(0, 2).flags() & Qt.ItemFlag.ItemIsEditable)

    window.props._exp_source.setCurrentIndex(1)
    assert experiment.applied_load_source == "manual"
    assert [step.external_values[0] for step in experiment.steps] == forces
    table = window.props._exp_load_editor.table
    assert bool(table.item(1, 2).flags() & Qt.ItemFlag.ItemIsEditable)

    table.item(1, 2).setText("99.5")
    assert experiment.steps[1].external_values[0] == 99.5
    assert experiment.steps[0].external_values[0] == 10.0
    assert experiment.steps[2].external_values[0] == 30.0
    window.session.dirty = False
    window.close()


def test_load_table_has_a_column_for_each_external_group(qapp):
    from vfmstudio.core.document import BoundaryEntry, ExperimentModel, LoadStepModel
    from vfmstudio.gui.widgets.load_table import AppliedLoadEditor

    experiment = ExperimentModel(
        experiment_id=1,
        applied_load_source="manual",
        external=[
            BoundaryEntry(label="Sys2_External W_X", dof="W_X", edge="right"),
            BoundaryEntry(label="W_Y @10, 15", dof="W_Y", nodes=[10, 15]),
        ],
        steps=[
            LoadStepModel(
                data_file="a.mat",
                label="s1",
                external_values=[5.55524, 5.39068],
            ),
            LoadStepModel(
                data_file="b.mat",
                label="s2",
                external_values=[11.1811, 10.1237],
            ),
        ],
    )
    editor = AppliedLoadEditor()
    editor.populate(experiment)
    table = editor.table
    assert table.columnCount() == 4
    assert table.horizontalHeaderItem(2).text() == "Sys2_External W_X"
    assert table.horizontalHeaderItem(3).text() == "W_Y @10, 15"
    assert table.item(0, 2).text() == "5.55524"
    assert table.item(0, 3).text() == "5.39068"
    header = table.horizontalHeader()
    from PySide6.QtWidgets import QHeaderView

    assert header.sectionResizeMode(2) == QHeaderView.ResizeMode.Stretch
    assert header.sectionResizeMode(3) == QHeaderView.ResizeMode.Stretch
    editor.close()


def test_load_table_assigns_mat_f_component_to_group(qapp, tmp_path):
    import numpy as np
    from scipy.io import savemat

    from vfmstudio.core.document import BoundaryEntry, ExperimentModel, LoadStepModel
    from vfmstudio.gui.widgets.load_table import AppliedLoadEditor

    path = tmp_path / "biax.mat"
    savemat(path, {"F": np.array([-12.5, 40.0])})
    experiment = ExperimentModel(
        experiment_id=1,
        applied_load_source="mat",
        external=[
            BoundaryEntry(label="group A", dof="W_Y", edge="top"),
            BoundaryEntry(label="group B", dof="W_X", edge="left"),
        ],
        steps=[LoadStepModel(data_file=path, label="s1")],
    )
    editor = AppliedLoadEditor()
    editor.populate(experiment)
    assert editor.table.item(0, 2).text() == "-12.5"
    assert editor.table.item(0, 3).text() == "40"
    assert len(editor._map_combos) == 2
    editor._map_combos[0].setCurrentIndex(1)
    assert experiment.external[0].force_index == 1
    assert editor.table.item(0, 2).text() == "40"
    assert editor.table.item(0, 3).text() == "40"
    editor._map_combos[1].setCurrentIndex(0)
    assert experiment.external[1].force_index == 0
    assert editor.table.item(0, 2).text() == "40"
    assert editor.table.item(0, 3).text() == "-12.5"
    editor.close()


def test_load_table_mat_f_is_relative_to_reference_step(qapp, tmp_path):
    import numpy as np
    from scipy.io import savemat

    from vfmstudio.core.document import BoundaryEntry, ExperimentModel, LoadStepModel
    from vfmstudio.gui.widgets.load_table import AppliedLoadEditor

    ref = tmp_path / "ref.mat"
    later = tmp_path / "later.mat"
    savemat(ref, {"F": np.array([10.0])})
    savemat(later, {"F": np.array([40.0])})
    experiment = ExperimentModel(
        experiment_id=1,
        applied_load_source="mat",
        reference_step=1,
        external=[BoundaryEntry(label="Myy", dof="W_Y", edge="top")],
        steps=[
            LoadStepModel(data_file=ref, label="ref"),
            LoadStepModel(data_file=later, label="later"),
        ],
    )
    editor = AppliedLoadEditor()
    editor.populate(experiment)
    assert editor.table.item(0, 2).text() == "0"
    assert editor.table.item(1, 2).text() == "30"
    editor.close()


def test_load_table_shows_named_f_labels(qapp, tmp_path):
    import numpy as np
    from scipy.io import savemat

    from vfmstudio.core.document import BoundaryEntry, ExperimentModel, LoadStepModel
    from vfmstudio.gui.widgets.load_table import AppliedLoadEditor

    path = tmp_path / "named.mat"
    savemat(
        path,
        {
            "F": np.array([-12.5, 40.0]),
            "F_labels": np.array(["Ty_sys1", "Ty_sys2"], dtype=object),
        },
    )
    experiment = ExperimentModel(
        experiment_id=1,
        applied_load_source="mat",
        external=[
            BoundaryEntry(label="group A", dof="W_Y", edge="top"),
            BoundaryEntry(label="group B", dof="W_X", edge="left"),
        ],
        steps=[LoadStepModel(data_file=path, label="s1")],
    )
    editor = AppliedLoadEditor()
    editor.populate(experiment)
    assert "Ty_sys1" in editor._map_combos[0].itemText(0)
    assert "Ty_sys2" in editor._map_combos[0].itemText(1)
    editor.close()


def test_fill_blanks_from_mat_f(qapp, tmp_path):
    import numpy as np
    from scipy.io import savemat

    from vfmstudio.core.document import BoundaryEntry, ExperimentModel, LoadStepModel

    path = tmp_path / "loaded.mat"
    savemat(path, {"F": np.array([-1250.0])})
    missing = tmp_path / "no_force.mat"
    savemat(missing, {"x": np.array([0.0, 1.0])})

    window = MainWindow()
    experiment = ExperimentModel(
        experiment_id=1,
        name="exp",
        applied_load_source="manual",
        external=[BoundaryEntry(label="M_yy top", dof="W_Y", edge="top")],
        steps=[
            LoadStepModel(data_file=path, label="has F"),
            LoadStepModel(data_file=missing, label="no F"),
        ],
    )
    window.session.document.experiments = [experiment]
    window.session.selection = Selection(kind="steps", experiment_id=1)
    window._refresh_all()

    table = window.props._exp_load_editor.table
    assert table.item(0, 2).text() == ""
    assert table.item(1, 2).text() == ""
    window.props._exp_load_editor.fill_button.click()
    assert experiment.applied_load_source == "manual"
    assert experiment.steps[0].external_values[0] == -1250.0
    assert experiment.steps[1].external_values[0] is None
    table = window.props._exp_load_editor.table
    assert table.item(0, 2).text() == "-1250"
    assert table.item(1, 2).text() == ""
    window.session.dirty = False
    window.close()


def test_properties_sidebar_scrolls_when_content_overflows(qapp):
    from PySide6.QtWidgets import QScrollArea

    from vfmstudio.core.document import BoundaryEntry, ExperimentModel, LoadStepModel

    window = MainWindow()
    experiment = ExperimentModel(
        experiment_id=1,
        name="exp",
        applied_load_source="manual",
        external=[BoundaryEntry(label="M_yy top", dof="W_Y", edge="top")],
        steps=[
            LoadStepModel(data_file=f"s{index}.mat", label=f"s{index}")
            for index in range(9)
        ],
    )
    window.session.document.experiments = [experiment]
    window.session.selection = Selection(kind="steps", experiment_id=1)
    window.show()
    window.resize(1100, 420)
    window.dock_props.resize(280, 220)
    window._refresh_all()
    qapp.processEvents()

    scroll = window.props._pages["step"]
    assert isinstance(scroll, QScrollArea)
    inner = scroll.widget()
    assert inner is not None
    assert inner.minimumSizeHint().height() > scroll.viewport().height()
    assert scroll.verticalScrollBar().maximum() > 0
    window.session.dirty = False
    window.close()


def test_free_mesh_generation_toggle(qapp):
    from vfmstudio.core.document import ExperimentModel, MeshModel

    window = MainWindow()
    window.session.document.experiments = [
        ExperimentModel(experiment_id=1, name="coupon", mesh=MeshModel())
    ]
    window.set_module("mesh")
    assert hasattr(window.props, "_mesh_free")
    assert window.props._mesh_construct.isHidden()
    window.props._mesh_free.setChecked(True)
    assert window.session.free_mesh_mode is True
    assert not window.props._mesh_construct.isHidden()
    assert not window.props._mesh_clear_points.isHidden()
    assert not window.props._mesh_nx.isEnabled()
    window.props._mesh_free.setChecked(False)
    assert window.session.free_mesh_mode is False
    assert window.props._mesh_construct.isHidden()
    assert window.props._mesh_nx.isEnabled()
    window.session.dirty = False
    window.close()


def test_drag_mesh_toggle_exclusive_with_free_mesh(qapp):
    from vfmstudio.core.document import ExperimentModel, MeshModel

    window = MainWindow()
    window.session.document.experiments = [
        ExperimentModel(experiment_id=1, name="coupon", mesh=MeshModel())
    ]
    window.set_module("mesh")
    assert hasattr(window.props, "_mesh_drag")
    assert window.props._mesh_drag_hint.isHidden()
    window.props._mesh_drag.setChecked(True)
    assert window.session.drag_mesh_mode is True
    assert window.session.free_mesh_mode is False
    assert not window.props._mesh_drag_hint.isHidden()
    assert not window.props._mesh_nx.isEnabled()
    window.props._mesh_free.setChecked(True)
    assert window.session.free_mesh_mode is True
    assert window.session.drag_mesh_mode is False
    assert window.props._mesh_drag_hint.isHidden()
    window.props._mesh_free.setChecked(False)
    assert window.session.free_mesh_mode is False
    assert window.props._mesh_nx.isEnabled()
    window.session.dirty = False
    window.close()


def test_clear_mesh_confirms_and_drops_bcs(qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from vfmstudio.core.document import BoundaryEntry, ExperimentModel, LoadStepModel, MeshModel

    prompts = []

    def _question(*args, **kwargs):
        prompts.append(args[2] if len(args) > 2 else kwargs.get("text", ""))
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", _question)

    window = MainWindow()
    experiment = ExperimentModel(
        experiment_id=1,
        name="coupon",
        mesh=MeshModel(n_elements_x=6, n_elements_y=5, x_nodes=[0.0, 1.0]),
        prescribed=[BoundaryEntry(label="fix W", dof="W", edge="bottom")],
        external=[
            BoundaryEntry.model_validate(
                {"label": "M1", "dof": "W_Y", "edge": "top", "loadType": "distributed"}
            )
        ],
        steps=[LoadStepModel(data_file="a.mat", label="s1", external_values=[12.0])],
    )
    window.session.document.experiments = [experiment]
    window.session.selection = Selection(kind="mesh", experiment_id=1)
    window.session.result = object()
    window._refresh_all()
    window.props._mesh_clear.click()
    assert prompts
    assert "boundary conditions" in prompts[0]
    assert "identification results" in prompts[0].lower()
    assert experiment.mesh.n_elements_x == 2
    assert experiment.prescribed == []
    assert experiment.external == []
    assert experiment.steps[0].external_values == []
    assert experiment.mesh.x_nodes is None
    window.session.dirty = False
    window.close()


def test_material_page_keeps_anisotropic_six_parameters(qapp):
    from vfmstudio.core.document import ExperimentModel, MaterialModel
    from vfmstudio.gui.session import Selection

    window = MainWindow()
    theta = [400.0, 50.0, 400.0, 100.0, 50.0, 50.0]
    window.session.document.material = MaterialModel(
        material_type="anisotropic",
        analysis_type="out_of_plane",
        initial_parameters=list(theta),
    )
    window.session.document.experiments = [
        ExperimentModel(experiment_id=1, name="biax")
    ]
    window.session.selection = Selection(kind="experiment", experiment_id=1)
    window._refresh_all()
    assert window.session.document.material.initial_parameters == theta
    window._on_tree_selection(Selection(kind="material"))
    material = window.session.document.material
    assert material.material_type == "anisotropic"
    assert material.analysis_type == "out_of_plane"
    assert material.initial_parameters == theta
    assert window.props._mat_type.currentText() == "anisotropic"
    visible = [
        spin for spin in window.props._theta_spins if not spin.isHidden()
    ]
    assert len(visible) == 6
    assert [spin.value() for spin in visible] == theta
    window.session.dirty = False
    window.close()


