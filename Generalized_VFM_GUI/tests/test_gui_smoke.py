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
    window = MainWindow()
    assert "VFM Studio" in window.windowTitle()
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
    assert hasattr(window.props, "view_changed")
    assert hasattr(window.props, "_mesh_show_counts")
    assert window.props._mesh_show_counts.isChecked() is False
    assert "Add load steps" in window.actions.get("add_step").text()
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
        "Individual Experiment check",
        "Identifiability",
        "Virtual fields",
        "Uncertainty quantification",
    ]
    uq = post.child(4)
    assert _child_labels(uq) == [
        "Cov(θ)",
        "Experiment uncertainty",
        "Matrices A and B",
        "Information geometry",
        "Leave-one-out",
    ]
    assert window._central.currentWidget() is window.viewport
    window._on_tree_selection(Selection(kind="post_plots"))
    assert window._central.currentWidget() is window.post
    assert window.post._stack.currentWidget() is window.post.plots
    assert window.props._title.text() == "Coefficient Evolution"
    assert not hasattr(window.post, "coeff")
    window._on_tree_selection(Selection(kind="post_experiment_check"))
    assert window.post._stack.currentWidget() is window.post.experiment_check
    assert window.props._title.text() == "Individual Experiment check"
    window._on_tree_selection(Selection(kind="post_virtual_fields"))
    assert window._central.currentWidget() is window.post
    assert window.post._stack.currentWidget() is window.post.vf_panel
    assert window.props._title.text() == "Virtual fields"
    window._on_tree_selection(Selection(kind="post_uncertainty"))
    assert window._central.currentWidget() is window.post
    assert window.post._stack.currentWidget() is window.post.uq_landing
    assert window.props._title.text() == "Uncertainty quantification"
    assert "M ≥ p + 1" in window.post.uq_landing._verdict.text()
    assert [window.post.uq_info._mode.itemText(i) for i in range(window.post.uq_info._mode.count())] == [
        "cond(A − Aᵢ)",
        "Share of A",
        "Eigenvalues of A",
        "Constrained combinations",
    ]
    window._on_tree_selection(Selection(kind="post_uncertainty_cov"))
    assert window.post._stack.currentWidget() is window.post.uq_cov
    assert window.props._title.text() == "Cov(θ)"
    window._on_tree_selection(Selection(kind="post_uncertainty_experiments"))
    assert window.post._stack.currentWidget() is window.post.uq_experiments
    assert window.props._title.text() == "Experiment uncertainty"
    window._on_tree_selection(Selection(kind="post_uncertainty_ab"))
    assert window.post._stack.currentWidget() is window.post.uq_ab
    assert window.props._title.text() == "Matrices A and B"
    window._on_tree_selection(Selection(kind="post_uncertainty_info"))
    assert window.post._stack.currentWidget() is window.post.uq_info
    assert window.props._title.text() == "Information geometry"
    window._on_tree_selection(Selection(kind="post_uncertainty_loo"))
    assert window.post._stack.currentWidget() is window.post.uq_loo
    assert window.props._title.text() == "Leave-one-out"
    assert hasattr(window.props, "_uq_kind")
    assert [window.props._uq_kind.itemText(i) for i in range(window.props._uq_kind.count())] == [
        "CR2",
        "CR3",
        "CR0",
    ]
    from vfmstudio.gui.results.uncertainty_plots import CR3, ROBUST, WORK

    assert WORK.startswith("#")
    assert ROBUST.startswith("#")
    assert CR3.startswith("#")
    assert [window.post.vf_panel._mode.itemText(i) for i in range(window.post.vf_panel._mode.count())] == [
        "3D surface",
        "2D contour",
    ]
    assert [window.post.vf_panel._quantity.itemText(i) for i in range(window.post.vf_panel._quantity.count())] == [
        "w*",
        "kx*",
        "ky*",
        "kxy*",
    ]
    assert window.field_combo.isEnabled() is False
    assert window.display_combo.isEnabled() is False
    window._on_tree_selection(Selection(kind="mesh"))
    assert window._central.currentWidget() is window.viewport
    assert window.field_combo.isEnabled() is True
    plots = window.post.plots
    assert not hasattr(plots, "_mode")
    assert hasattr(plots, "_canvas")
    assert "Coefficient evolution" in plots._title.text()
    from vfmstudio.gui.virtual_fields_view import _tile_shape, collect_virtual_fields

    assert _tile_shape(1) == (1, 1)
    assert _tile_shape(2) == (1, 2)
    assert _tile_shape(4) == (2, 2)

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
    labels = ["D11", "D22", "D12", "D66"]
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
    assert len(axes_3d) == 4
    title_axes = [
        ax for ax in view._figure.axes
        if ax.name != "3d" and ax.texts and not ax.axison
    ]
    assert len(title_axes) == 4
    aspect = axes_3d[0].get_box_aspect()
    assert aspect is not None
    ax_r, ay_r, az_r = (float(v) for v in aspect)
    assert az_r / max(ax_r, ay_r) < 1.2
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
    view._mode.setCurrentText("2D contour")
    titled_2d = [
        ax for ax in view._figure.axes
        if ax.name != "3d" and ax.get_title()
    ]
    assert len(titled_2d) == 4
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
    view._quantity.setCurrentText("kx*")
    titled_kx = [
        ax for ax in view._figure.axes
        if ax.name != "3d" and ax.get_title()
    ]
    assert len(titled_kx) == 4
    view.close()
