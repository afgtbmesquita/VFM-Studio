"""Model tree dock — experiments / mesh / BCs / load steps / post processing."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import QMenu, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

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
    "post_stress_space": "post",
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

_BC_ADD_LABELS = (
    ("prescribed", "Add prescribed BC…"),
    ("linking", "Add linking BC…"),
    ("external", "Add external BC…"),
)

_POST_CHILDREN = (
    ("Coefficient Evolution", "post_plots", ()),
    ("Parameter space", "post_parameter_space", ()),
    ("Stress resultant space", "post_stress_space", ()),
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


def _tree_signature(session: AppSession) -> tuple:
    path = session.path
    document = session.document
    return (
        str(path) if path is not None else "",
        tuple(
            (
                int(experiment.experiment_id),
                experiment.name or "",
                tuple(_step_label(step) for step in experiment.steps),
                tuple(_bc_label(entry) for entry in experiment.prescribed),
                tuple(_bc_label(entry) for entry in experiment.linking),
                tuple(_bc_label(entry) for entry in experiment.external),
            )
            for experiment in document.experiments
        ),
    )


class ModelTreeDock(QWidget):
    selection_changed = Signal(object)
    request_add_experiment = Signal()
    request_add_bc = Signal(int, str)
    request_delete_bc = Signal(int, str, int)
    request_add_step = Signal(int)
    request_delete_step = Signal(int, int)
    request_move_step = Signal(int, int, int)
    request_delete_experiment = Signal(int)
    request_duplicate_experiment = Signal(int)

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
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_context_menu)
        self.tree.itemSelectionChanged.connect(self._emit_selection)
        self._delete_bc_shortcut = QShortcut(QKeySequence.StandardKey.Delete, self.tree)
        self._delete_bc_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._delete_bc_shortcut.activated.connect(self._delete_current_bc)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tree)

    def rebuild(self, session: AppSession) -> None:
        signature = _tree_signature(session)
        if (
            signature == getattr(self, "_signature", None)
            and self.tree.topLevelItemCount() > 0
        ):
            self.select(session.selection)
            return
        self._signature = signature
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
            parent = match.parent()
            while parent is not None:
                parent.setExpanded(True)
                parent = parent.parent()
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

    def _on_context_menu(self, pos) -> None:
        node = self.tree.itemAt(pos)
        if node is None:
            return
        self.tree.setCurrentItem(node)
        kind = str(node.data(0, ROLE_KIND) or "")
        menu = QMenu(self)
        if kind in {"experiments", "project"}:
            menu.addAction("Add experiment").setData(("add_experiment", None))
        elif kind == "experiment":
            menu.addAction("Add load steps…").setData(("add_step", None))
            menu.addAction("Duplicate experiment").setData(("duplicate_experiment", None))
            menu.addSeparator()
            label = node.text(0) or "experiment"
            menu.addAction(f'Delete "{label}"').setData(("delete_experiment", None))
        elif kind == "steps":
            menu.addAction("Add load steps…").setData(("add_step", None))
        elif kind == "step":
            menu.addAction("Move up").setData(("move_step", -1))
            menu.addAction("Move down").setData(("move_step", 1))
            menu.addSeparator()
            label = node.text(0) or "load step"
            menu.addAction(f'Delete "{label}"').setData(("delete_step", None))
        elif kind == "bcs":
            for bc_kind, label in _BC_ADD_LABELS:
                menu.addAction(label).setData(("add_bc", bc_kind))
        elif kind == "bc_group":
            bc_kind = str(node.data(0, ROLE_BC_KIND) or "")
            if bc_kind in {"prescribed", "linking", "external"}:
                menu.addAction(f"Add {bc_kind} BC…").setData(("add_bc", bc_kind))
        elif kind == "bc":
            label = node.text(0) or "boundary condition"
            menu.addAction(f'Delete "{label}"').setData(("delete_bc", None))
        else:
            return
        if not menu.actions():
            return
        chosen = menu.exec(self.tree.viewport().mapToGlobal(pos))
        if chosen is None:
            return
        action, extra = chosen.data()
        self._dispatch_context(node, str(action), extra)

    def _dispatch_context(self, node: QTreeWidgetItem, action: str, extra) -> None:
        experiment_id = node.data(0, ROLE_EXP)
        if action == "add_experiment":
            self.request_add_experiment.emit()
            return
        if experiment_id is None:
            return
        exp_id = int(experiment_id)
        if action == "add_step":
            self.request_add_step.emit(exp_id)
        elif action == "duplicate_experiment":
            self.request_duplicate_experiment.emit(exp_id)
        elif action == "delete_experiment":
            self.request_delete_experiment.emit(exp_id)
        elif action == "delete_step":
            index = node.data(0, ROLE_STEP)
            if index is not None:
                self.request_delete_step.emit(exp_id, int(index))
        elif action == "move_step":
            index = node.data(0, ROLE_STEP)
            if index is not None:
                self.request_move_step.emit(exp_id, int(index), int(extra))
        elif action == "add_bc":
            self._emit_add(node, str(extra))
        elif action == "delete_bc":
            self._emit_delete(node)

    def _delete_current_bc(self) -> None:
        items = self.tree.selectedItems()
        if not items:
            return
        node = items[0]
        if str(node.data(0, ROLE_KIND) or "") != "bc":
            return
        self._emit_delete(node)

    def _emit_add(self, node: QTreeWidgetItem, kind: str) -> None:
        experiment_id = node.data(0, ROLE_EXP)
        if experiment_id is None or kind not in {"prescribed", "linking", "external"}:
            return
        self.request_add_bc.emit(int(experiment_id), kind)

    def _emit_delete(self, node: QTreeWidgetItem) -> None:
        experiment_id = node.data(0, ROLE_EXP)
        kind = node.data(0, ROLE_BC_KIND)
        index = node.data(0, ROLE_BC_INDEX)
        if experiment_id is None or not kind or index is None:
            return
        self.request_delete_bc.emit(int(experiment_id), str(kind), int(index))
