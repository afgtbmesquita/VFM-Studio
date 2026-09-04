"""GUI dialogs."""

from vfmstudio.gui.dialogs.bc_editor import AddBcDialog
from vfmstudio.gui.dialogs.experiment_editor import ExperimentEditorDialog, ensure_experiment
from vfmstudio.gui.dialogs.import_mat import ImportMatDialog
from vfmstudio.gui.dialogs.import_steps import AddLoadStepsDialog
from vfmstudio.gui.dialogs.new_project import NewProjectDialog
from vfmstudio.gui.dialogs.solver_settings import SolverSettingsDialog
from vfmstudio.gui.dialogs.workflow import WorkflowDialog

__all__ = [
    "AddBcDialog",
    "AddLoadStepsDialog",
    "AddStepDialog",
    "ExperimentEditorDialog",
    "ImportMatDialog",
    "NewProjectDialog",
    "SolverSettingsDialog",
    "WorkflowDialog",
    "ensure_experiment",
]

AddStepDialog = AddLoadStepsDialog

