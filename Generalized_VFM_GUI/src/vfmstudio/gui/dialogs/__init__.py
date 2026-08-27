"""GUI dialogs."""

from vfmstudio.gui.dialogs.bc_editor import AddBcDialog
from vfmstudio.gui.dialogs.experiment_editor import ExperimentEditorDialog, ensure_experiment
from vfmstudio.gui.dialogs.import_mat import AddStepDialog, ImportMatDialog
from vfmstudio.gui.dialogs.new_project import NewProjectDialog
from vfmstudio.gui.dialogs.solver_settings import SolverSettingsDialog

__all__ = [
    "AddBcDialog",
    "AddStepDialog",
    "ExperimentEditorDialog",
    "ImportMatDialog",
    "NewProjectDialog",
    "SolverSettingsDialog",
    "ensure_experiment",
]
