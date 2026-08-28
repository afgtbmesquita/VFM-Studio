"""Model tree dock — experiments / mesh / BCs / load steps / post processing."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from vfmstudio.gui.session import AppSession, Selection
from vfmstudio.resources import icon

ROLE_KIND = Qt.ItemDataRole.UserRole
ROLE_EXP = Qt.ItemDataRole.UserRole + 1
ROLE_STEP = Qt.ItemDataRole.UserRole + 2
ROLE_BC_KIND = Qt.ItemDataRole.UserRole + 3
ROLE_BC_INDEX = Qt.ItemDataRole.UserRole + 4

_CATEGORY_KINDS = {
    "project",
    "material",
    "experiments",
    "experiment",
    "mesh",
    "bcs",
    "bc_group",
    "steps",
    "solver",
    "post_processing",
}

_KIND_ICONS = {
    "project": "project",
    "material": "material",
    "experiments": "experiments",
    "experiment": "experiment",
    "mesh": "mesh",
    "bcs": "bcs",
    "bc_group": "bcs",
    "bc": "bc",
    "steps": "step",
    "step": "step",
    "solver": "solver",
    "post_processing": "post",
    "post_plots": "post",
    "post_parameter_space": "post",
    "post_experiment_check": "post",
    "post_identifiability": "post",
    "post_identifiability_strength": "post",
    "post_virtual_fields": "post",
    "post_uncertainty": "post",
    "post_uncertainty_cov": "post",
    "post_uncertainty_experiments": "post",
    "post_uncertainty_ab": "post",
    "post_uncertainty_info": "post",
    "post_uncertainty_loo": "post",
}

_POST_CHILDREN = (
    ("Coefficient Evolution", "post_plots", ()),
    ("Parameter space", "post_parameter_space", ()),
    ("Individual Experiment check", "post_experiment_check", ()),
    ("Identifiability", "post_identifiability", ()),
    ("Identifiability strength", "post_identifiability_strength", ()),
    ("Virtual fields", "post_virtual_fields", ()),
    (
        "Uncertainty quantification",
        "post_uncertainty",
        (
            ("Cov(θ)", "post_uncertainty_cov"),
            ("Experiment uncertainty", "post_uncertainty_experiments"),
            ("Matrices A and B", "post_uncertainty_ab"),
            ("Information geometry", "post_uncertainty_info"),
            ("Leave-one-out", "post_uncertainty_loo"),
        ),
    ),
)


def _font(bold: bool) -> QFont:
    font = QFont("Arial", 9)
    font.setBold(bold)
    return font


def _step_label(step) -> str:
    raw = str(step.label or getattr(step.data_file, "name", "") or "step")
    if ":" in raw:
        raw = raw.split(":")[-1]
    raw = raw.replace("\\", "/").split("/")[-1]
    if raw.lower().endswith(".mat"):
        raw = raw[:-4]
    return raw


def _bc_label(entry) -> str:
    label = entry.label or f"{entry.dof} @{entry.edge or 'nodes'}"
    return " ".join(str(label).split())


def _item(parent, text, kind, exp=None, step=None, bc_kind=None, bc_index=None, icon_name=None):
    node = QTreeWidgetItem(parent, [text])
    node.setData(0, ROLE_KIND, kind)
    node.setData(0, ROLE_EXP, exp)
    node.setData(0, ROLE_STEP, step)
    node.setData(0, ROLE_BC_KIND, bc_kind)
    node.setData(0, ROLE_BC_INDEX, bc_index)
    node.setFont(0, _font(kind in _CATEGORY_KINDS))
    glyph = icon_name or _KIND_ICONS.get(kind)
    if glyph:
        node.setIcon(0, icon(glyph))
    return node


def _selection_from_node(node: QTreeWidgetItem) -> Selection:
    return Selection(
        kind=str(node.data(0, ROLE_KIND) or "project"),
        experiment_id=node.data(0, ROLE_EXP),
        step_index=node.data(0, ROLE_STEP),
        bc_kind=node.data(0, ROLE_BC_KIND),
        bc_index=node.data(0, ROLE_BC_INDEX),
    )


def _node_matches(node: QTreeWidgetItem, selection: Selection) -> bool:
    current = _selection_from_node(node)
    if current.kind != selection.kind:
        return False
    if current.kind in {"bc", "bc_group"}:
        return (
            current.experiment_id == selection.experiment_id
            and current.bc_kind == selection.bc_kind
            and current.bc_index == selection.bc_index
        )
    if current.kind in {"step"}:
        return (
            current.experiment_id == selection.experiment_id
            and current.step_index == selection.step_index
        )
    if current.kind in {"experiment", "mesh", "bcs", "steps"}:
        return current.experiment_id == selection.experiment_id
    return True


class ModelTreeDock(QWidget):
    selection_changed = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(8)
        self.tree.setIconSize(QSize(14, 14))
        self.tree.setUniformRowHeights(True)
        self.tree.setAnimated(False)
        self.tree.setRootIsDecorated(True)
        self.tree.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.tree.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.tree.setStyleSheet("QTreeView { padding: 0px; margin: 0px; }")
        self.tree.itemSelectionChanged.connect(self._emit_selection)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tree)

    def rebuild(self, session: AppSession) -> None:
        self.tree.blockSignals(True)
        self.tree.clear()
        document = session.document
        path = session.path
        if path is not None:
            title = path.name
            if title.endswith(".vfmproj.json"):
                title = title[: -len(".vfmproj.json")]
            elif title.endswith(".json"):
                title = path.stem
        else:
            title = "Project"
        _item(self.tree, title, "project")
        _item(self.tree, "Material", "material")
        exp_root = _item(self.tree, "Experiments", "experiments")
        for experiment in document.experiments:
            exp_id = experiment.experiment_id
            title = experiment.name or f"Experiment {exp_id}"
            exp_item = _item(exp_root, title, "experiment", exp=exp_id)
            _item(exp_item, "Mesh", "mesh", exp=exp_id)
            bc_root = _item(exp_item, "Boundary conditions", "bcs", exp=exp_id)
            for kind, entries in (
                ("prescribed", experiment.prescribed),
                ("linking", experiment.linking),
                ("external", experiment.external),
            ):
                group_icon = {
                    "prescribed": "bc",
                    "linking": "bc-linking",
                    "external": "bc-external",
                }[kind]
                group = _item(
                    bc_root, kind.title(), "bc_group",
                    exp=exp_id, bc_kind=kind, icon_name=group_icon,
                )
                for index, entry in enumerate(entries):
                    _item(
                        group, _bc_label(entry), "bc",
                        exp=exp_id, bc_kind=kind, bc_index=index,
                        icon_name=group_icon,
                    )
            steps_root = _item(exp_item, "Load steps", "steps", exp=exp_id)
            for index, step in enumerate(experiment.steps):
                _item(steps_root, _step_label(step), "step", exp=exp_id, step=index)
            exp_item.setExpanded(True)
        exp_root.setExpanded(True)
        _item(self.tree, "Solver", "solver")
        post_root = _item(self.tree, "Post Processing", "post_processing")
        for label, kind, children in _POST_CHILDREN:
            node = _item(post_root, label, kind)
            for child_label, child_kind in children:
                _item(node, child_label, child_kind)
            if children:
                node.setExpanded(True)
        post_root.setExpanded(True)
        self.tree.blockSignals(False)
        self.select(session.selection)

    def select(self, selection: Selection) -> None:
        match = self._find_item(selection)
        self.tree.blockSignals(True)
        if match is not None:
            self.tree.setCurrentItem(match)
        else:
            self.tree.clearSelection()
        self.tree.blockSignals(False)

    def _find_item(self, selection: Selection):
        def walk(node: QTreeWidgetItem):
            if _node_matches(node, selection):
                return node
            for index in range(node.childCount()):
                found = walk(node.child(index))
                if found is not None:
                    return found
            return None

        for index in range(self.tree.topLevelItemCount()):
            found = walk(self.tree.topLevelItem(index))
            if found is not None:
                return found
        return None

    def _emit_selection(self) -> None:
        items = self.tree.selectedItems()
        if not items:
            return
        self.selection_changed.emit(_selection_from_node(items[0]))
