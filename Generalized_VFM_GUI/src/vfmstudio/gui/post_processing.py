"""Full-size post-processing workspace shown in the main area."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QStackedWidget, QVBoxLayout, QWidget

from vfmstudio.core.document import SessionDocument
from vfmstudio.gui.docks.plot_dock import PlotDock
from vfmstudio.gui.results import IdentifiabilityPanel
from vfmstudio.gui.results.experiment_check import ExperimentCheckPanel
from vfmstudio.gui.results.identifiability_strength import IdentifiabilityStrengthPanel
from vfmstudio.gui.results.parameter_space import ParameterSpacePanel
from vfmstudio.gui.results.uncertainty import (
    CovariancePanel,
    ExperimentUncertaintyPanel,
    InformationGeometryPanel,
    LeaveOneOutPanel,
    MatricesPanel,
    UncertaintyLanding,
)
from vfmstudio.gui.virtual_fields_view import VirtualFieldTiledView

_KIND_FIGURE_WIDGET = {
    "post_processing": "plots",
    "post_plots": "plots",
    "post_parameter_space": "parameter_space",
    "post_experiment_check": "experiment_check",
    "post_identifiability_strength": "ident_strength",
    "post_virtual_fields": "vf_panel",
    "post_uncertainty_cov": "uq_cov",
    "post_uncertainty_experiments": "uq_experiments",
    "post_uncertainty_ab": "uq_ab",
    "post_uncertainty_info": "uq_info",
    "post_uncertainty_loo": "uq_loo",
}


class PostProcessingWorkspace(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.plots = PlotDock()
        self.parameter_space = ParameterSpacePanel()
        self.experiment_check = ExperimentCheckPanel()
        self.ident_panel = IdentifiabilityPanel()
        self.ident_strength = IdentifiabilityStrengthPanel()
        self.vf_panel = VirtualFieldTiledView()
        self.uq_landing = UncertaintyLanding()
        self.uq_cov = CovariancePanel()
        self.uq_experiments = ExperimentUncertaintyPanel()
        self.uq_ab = MatricesPanel()
        self.uq_info = InformationGeometryPanel()
        self.uq_loo = LeaveOneOutPanel()

        self._stack = QStackedWidget()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._stack)

        self._pages: dict[str, int] = {}
        for widget, kinds in (
            (self.plots, ("post_processing", "post_plots")),
            (self.parameter_space, ("post_parameter_space",)),
            (self.experiment_check, ("post_experiment_check",)),
            (self.ident_panel, ("post_identifiability",)),
            (self.ident_strength, ("post_identifiability_strength",)),
            (self.vf_panel, ("post_virtual_fields",)),
            (self.uq_landing, ("post_uncertainty",)),
            (self.uq_cov, ("post_uncertainty_cov",)),
            (self.uq_experiments, ("post_uncertainty_experiments",)),
            (self.uq_ab, ("post_uncertainty_ab",)),
            (self.uq_info, ("post_uncertainty_info",)),
            (self.uq_loo, ("post_uncertainty_loo",)),
        ):
            self._stack.addWidget(widget)
            index = self._stack.indexOf(widget)
            for kind in kinds:
                self._pages[kind] = index

    def show_kind(self, kind: str) -> None:
        self._stack.setCurrentIndex(self._pages.get(kind, 0))

    def figure_for_kind(self, kind: str):
        """Matplotlib figure for the post-processing page, or None."""
        attr = _KIND_FIGURE_WIDGET.get(str(kind))
        if attr is None:
            return None
        widget = getattr(self, attr, None)
        return getattr(widget, "_figure", None)

    def set_results(
        self,
        document: SessionDocument,
        result: Any,
        identifiability: Any = None,
        warp_scale: float = 1.0,
        uncertainty: Any = None,
    ) -> None:
        self.plots.set_results(result, identifiability, document)
        self.parameter_space.update_from(document, result)
        self.experiment_check.update_from(document, result)
        self.ident_panel.update_from(identifiability)
        self.ident_strength.update_from(document, uncertainty)
        self.vf_panel.update_from(document, result, warp_scale=warp_scale)
        self.uq_cov.update_from(document, uncertainty)
        self.uq_experiments.update_from(document, uncertainty)
        self.uq_ab.update_from(document, uncertainty)
        self.uq_info.update_from(document, uncertainty)
        self.uq_loo.update_from(document, uncertainty)
        self.uq_landing.update_from(uncertainty)
