"""Full-size post-processing workspace shown in the main area."""

from __future__ import annotations

from typing import Any, Optional

from PySide6.QtWidgets import QStackedWidget, QVBoxLayout, QWidget

from vfmstudio.core.document import SessionDocument
from vfmstudio.gui.docks.plot_dock import PlotDock
from vfmstudio.gui.results import IdentifiabilityPanel
from vfmstudio.gui.results.experiment_check import ExperimentCheckPanel
from vfmstudio.gui.results.identifiability_strength import IdentifiabilityStrengthPanel
from vfmstudio.gui.results.parameter_space import ParameterSpacePanel
from vfmstudio.gui.results.stress_space import StressSpacePanel
from vfmstudio.gui.results.uncertainty import (
    CovariancePanel,
    ExperimentUncertaintyPanel,
    InformationGeometryPanel,
    LeaveOneOutPanel,
    MatricesPanel,
    UncertaintyLanding,
)
from vfmstudio.gui.session import is_post_kind
from vfmstudio.gui.virtual_fields_view import VirtualFieldTiledView

_KIND_FIGURE_WIDGET = {
    "post_processing": "plots",
    "post_plots": "plots",
    "post_parameter_space": "parameter_space",
    "post_stress_space": "stress_space",
    "post_experiment_check": "experiment_check",
    "post_identifiability_strength": "ident_strength",
    "post_virtual_fields": "vf_panel",
    "post_uncertainty_cov": "uq_cov",
    "post_uncertainty_experiments": "uq_experiments",
    "post_uncertainty_ab": "uq_ab",
    "post_uncertainty_info": "uq_info",
    "post_uncertainty_loo": "uq_loo",
}

_KIND_PAGE = {
    "post_processing": "plots",
    "post_plots": "plots",
    "post_parameter_space": "parameter_space",
    "post_stress_space": "stress_space",
    "post_experiment_check": "experiment_check",
    "post_identifiability": "ident_panel",
    "post_identifiability_strength": "ident_strength",
    "post_virtual_fields": "vf_panel",
    "post_uncertainty": "uq_landing",
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
        self.stress_space = StressSpacePanel()
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
            (self.stress_space, ("post_stress_space",)),
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

        self._document: Optional[SessionDocument] = None
        self._result: Any = None
        self._identifiability: Any = None
        self._uncertainty: Any = None
        self._warp_scale: float = 1.0
        self._vf_experiment_id: int | None = None
        self._vf_step_index: int | None = None
        self._vf_view_mode: str | None = None
        self._vf_quantity: str | None = None
        self._filled: set[str] = set()
        self._current_kind: str = "post_plots"

    def show_kind(self, kind: str) -> None:
        self._current_kind = str(kind)
        self._stack.setCurrentIndex(self._pages.get(kind, 0))
        page = _KIND_PAGE.get(str(kind))
        already = page in self._filled
        self._fill_page(page)
        if already and page == "plots" and self._document is not None:
            self.plots.apply_combination(self._document)

    def figure_for_kind(self, kind: str):
        """Matplotlib figure for the post-processing page, or None."""
        page = _KIND_PAGE.get(str(kind))
        if page is not None:
            self._fill_page(page)
        attr = _KIND_FIGURE_WIDGET.get(str(kind))
        if attr is None:
            return None
        widget = getattr(self, attr, None)
        if widget is None:
            return None
        exporter = getattr(widget, "figure_for_export", None)
        if callable(exporter):
            return exporter()
        return getattr(widget, "_figure", None)

    def set_results(
        self,
        document: SessionDocument,
        result: Any,
        identifiability: Any = None,
        warp_scale: float = 1.0,
        uncertainty: Any = None,
        vf_experiment_id: int | None = None,
        vf_step_index: int | None = None,
        vf_view_mode: str | None = None,
        vf_quantity: str | None = None,
        kind: str | None = None,
    ) -> None:
        self._document = document
        self._result = result
        self._identifiability = identifiability
        self._uncertainty = uncertainty
        self._warp_scale = float(warp_scale)
        self._vf_experiment_id = vf_experiment_id
        self._vf_step_index = vf_step_index
        self._vf_view_mode = vf_view_mode
        self._vf_quantity = vf_quantity
        self._filled.clear()
        target = kind if kind is not None else self._current_kind
        if is_post_kind(str(target or "")):
            self.show_kind(str(target))

    def _fill_page(self, page: Optional[str]) -> None:
        if not page or page in self._filled:
            return
        document = self._document
        result = self._result
        self._filled.add(page)
        if page == "plots":
            self.plots.set_results(result, self._identifiability, document)
        elif page == "parameter_space":
            self.parameter_space.update_from(document, result)
        elif page == "stress_space":
            self.stress_space.update_from(document, result, paint=False)
        elif page == "experiment_check":
            if document is not None:
                self.experiment_check.update_from(document, result)
        elif page == "ident_panel":
            self.ident_panel.update_from(self._identifiability)
        elif page == "ident_strength":
            if document is not None:
                self.ident_strength.update_from(document, self._uncertainty)
        elif page == "vf_panel":
            if document is not None:
                self.vf_panel.update_from(
                    document,
                    result,
                    warp_scale=self._warp_scale,
                    experiment_id=self._vf_experiment_id,
                    step_index=self._vf_step_index,
                    view_mode=self._vf_view_mode,
                    quantity=self._vf_quantity,
                )
        elif page == "uq_landing":
            self.uq_landing.update_from(self._uncertainty)
        elif page == "uq_cov":
            if document is not None:
                self.uq_cov.update_from(document, self._uncertainty)
        elif page == "uq_experiments":
            if document is not None:
                self.uq_experiments.update_from(document, self._uncertainty)
        elif page == "uq_ab":
            if document is not None:
                self.uq_ab.update_from(document, self._uncertainty)
        elif page == "uq_info":
            if document is not None:
                self.uq_info.update_from(document, self._uncertainty)
        elif page == "uq_loo":
            if document is not None:
                self.uq_loo.update_from(document, self._uncertainty)
        else:
            self._filled.discard(page)
