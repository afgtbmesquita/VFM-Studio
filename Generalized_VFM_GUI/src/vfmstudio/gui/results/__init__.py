"""Results docks."""

from vfmstudio.gui.results.coefficients import CoefficientPanel
from vfmstudio.gui.results.convergence import ConvergencePanel
from vfmstudio.gui.results.experiment_check import ExperimentCheckPanel
from vfmstudio.gui.results.identifiability import IdentifiabilityPanel
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
from vfmstudio.gui.results.virtual_fields import VirtualFieldPanel

__all__ = [
    "CoefficientPanel",
    "ConvergencePanel",
    "CovariancePanel",
    "ExperimentCheckPanel",
    "ExperimentUncertaintyPanel",
    "IdentifiabilityPanel",
    "IdentifiabilityStrengthPanel",
    "InformationGeometryPanel",
    "LeaveOneOutPanel",
    "MatricesPanel",
    "ParameterSpacePanel",
    "UncertaintyLanding",
    "VirtualFieldPanel",
]
