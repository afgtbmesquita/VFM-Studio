"""DIC data agent window: arrange raw Vic-3D exports into tests, convert, measure, hand off to the chat.

Raw files are only ever read. Results go to a scratch folder; ``.mat`` export is
an explicit user action into a folder that cannot overlap the raw data.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Optional

from PySide6.QtCore import QObject, QRunnable, QSize, Qt, QThreadPool, QUrl, Signal, Slot
from PySide6.QtGui import (
    QBrush,
    QColor,
    QDesktopServices,
    QDragEnterEvent,
    QDropEvent,
    QKeySequence,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from vfmstudio.agents.dic_data.analysis import DataRefused
from vfmstudio.agents.dic_data.intake import (
    SYNC_NAME,
    FolderCompanions,
    TestGroup,
    assign_companions,
    classify_submission,
    clear_machine,
    clear_step_map,
    format_load,
    frame_match,
    group_by_folder,
    inspect_raw_file,
    order_key,
    read_machine_file,
    read_step_map_file,
)
from vfmstudio.agents.dic_data.raw_format import RawFileInfo, describe_expected_format
from vfmstudio.agents.dic_data.session import (
    DataAgentRun,
    describe_payload,
    export_solver_mats,
    run_analysis,
    validate_groups,
)

FILE_COLUMNS = ("Test / file", "Step", "DIC frame", "Campaign step", "Time [s]", "Machine load", "Facets", "Format")
REQ_COLUMNS = ("Requirement", "Submitted", "Status")
STRUCTURE_COLUMNS = ("Array", "Shape", "Type", "Range", "Meaning")
ROLE = Qt.ItemDataRole.UserRole
COMPANIONS = Qt.ItemDataRole.UserRole + 1
REQ_KIND = Qt.ItemDataRole.UserRole
REQ_INDEX = Qt.ItemDataRole.UserRole + 1
GROUP = "group"
BAD = QColor("#c0392b")
WARN = QColor("#b9770e")
INFO = QColor("#2f6fb3")
OK = QColor("#2e7d32")

# (key, sidebar title, sidebar subtitle, page hint)
PAGES = (
    ("files", "Files", "Add the DIC exports and the machine data",
     "Each test needs two submissions: the raw Vic-3D exports (exp-<frame>_<camera>.csv) and the machine "
     f"data ({SYNC_NAME}). They are used only when both are here and the DIC frame numbers are the same "
     "set. Nothing is read from beside the files you add. A step map (step_DIC_file_map.csv) is optional "
     "and is used only if you add that file too. Raw files are only read, never modified."),
    ("report", "Report", "Measured facts and automatic checks",
     "Every number measured from the raw data, with the automatic threshold checks. The Assistant's "
     "interpretation of these facts appears in the chat panel of the main window."),
    ("deformation", "Deformation", "Increment fields and load response",
     "How the specimen deformed between step 1 and the last step, and how the interior mean curvatures "
     "follow the machine load. Double-click a figure to open it at full resolution."),
    ("noise", "Noise vs model", "Measured scatter against the model",
     "Per-facet scatter of the measured curvatures compared with the noise model, in the interior and "
     "towards the free edges; then the DIC parameters inferred from the export and how noise, signal "
     "and smoothing bias change with the curvature filter size."),
    ("quality", "Read quality", "Dropouts and correlation sigma",
     "Where Vic-3D lost facets or correlated poorly, step by step."),
    ("solver", "Solver data", "Layout A .mat arrays and export",
     "The arrays the solver reads, one Layout A .mat per step, exactly as the VFM Studio importer builds "
     "them (κ = −2cxx, −2cyy, +2cxy; ε = exx, eyy, 2exy). Export writes copies into a folder you choose."),
)
FILES, REPORT, DEFORMATION, NOISE, QUALITY, SOLVER = range(len(PAGES))


class _Signals(QObject):
    finished = Signal(object)
    failed = Signal(object)
    progress = Signal(str)


class _Task(QRunnable):
    def __init__(self, work: Callable[[Callable[[str], None]], Any]) -> None:
        super().__init__()
        self.work = work
        self.signals = _Signals()
        self.setAutoDelete(False)

    @Slot()
    def run(self) -> None:
        try:
            self.signals.finished.emit(self.work(self.signals.progress.emit))
        except Exception as exc:
            self.signals.failed.emit(exc)


class FileTree(QTreeWidget):
    """Tests as top-level rows, their files as children. Drag to reorder or move between tests."""

    paths_dropped = Signal(list)
    rearranged = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setColumnCount(len(FILE_COLUMNS))
        self.setHeaderLabels(FILE_COLUMNS)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setDropIndicatorShown(True)
        self.setAcceptDrops(True)
        self.setAlternatingRowColors(True)
        self.setUniformRowHeights(True)
        header = self.header()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        if event.mimeData().hasUrls():
            paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
            event.acceptProposedAction()
            if paths:
                self.paths_dropped.emit(paths)
            return
        target = self.itemAt(event.position().toPoint())
        where = self.dropIndicatorPosition()
        event.setDropAction(Qt.DropAction.IgnoreAction)
        event.accept()
        dragged = self.selectedItems()
        if not dragged or target is None or target in dragged:
            return
        below = where in (
            QAbstractItemView.DropIndicatorPosition.BelowItem,
            QAbstractItemView.DropIndicatorPosition.OnItem,
        )
        if all(is_group(item) for item in dragged):
            anchor = target if is_group(target) else target.parent()
            if anchor in dragged:
                return
            for item in dragged:
                self.takeTopLevelItem(self.indexOfTopLevelItem(item))
            index = self.indexOfTopLevelItem(anchor) + (1 if below else 0)
            for offset, item in enumerate(dragged):
                self.insertTopLevelItem(index + offset, item)
                item.setExpanded(True)
        elif not any(is_group(item) for item in dragged):
            if is_group(target):
                group, index = target, (target.childCount() if where != QAbstractItemView.DropIndicatorPosition.AboveItem else 0)
                anchor = None
            else:
                group, anchor = target.parent(), target
                index = 0
            for item in dragged:
                item.parent().removeChild(item)
            if anchor is not None:
                index = group.indexOfChild(anchor) + (1 if below else 0)
            for offset, item in enumerate(dragged):
                group.insertChild(index + offset, item)
            group.setExpanded(True)
        else:
            return
        self.clearSelection()
        for item in dragged:
            item.setSelected(True)
        self.rearranged.emit()


def _read_submission(paths: list) -> tuple[list[RawFileInfo], list[FolderCompanions], list[FolderCompanions]]:
    """Classify a user submission and read it. Companion files are only those in the submission."""
    submission = classify_submission(paths)
    infos = [inspect_raw_file(path) for path in submission.dic_files]
    machines = [read_machine_file(path) for path in submission.machine_files]
    step_maps = [read_step_map_file(path) for path in submission.step_maps]
    return infos, machines, step_maps


def _dedupe_companions(items: list[FolderCompanions], attr: str) -> list[FolderCompanions]:
    seen: set[str] = set()
    chosen: list[FolderCompanions] = []
    for item in items:
        path = getattr(item, attr)
        key = str(path.resolve()).lower() if path is not None else str(id(item))
        if key in seen:
            continue
        seen.add(key)
        chosen.append(item)
    return chosen


def is_group(item: Optional[QTreeWidgetItem]) -> bool:
    return item is not None and item.data(0, ROLE) == GROUP


class ScaledImage(QLabel):
    """A figure that follows the available width at the screen's pixel density, never upscaled."""

    def __init__(self, path: str, parent=None) -> None:
        super().__init__(parent)
        self._path = path
        self._pixmap = QPixmap(path)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setMinimumWidth(200)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Double-click to open the full-resolution image.")
        self._rescale()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap.isNull():
            return
        ratio = max(1.0, self.devicePixelRatioF())
        target = min(int(self.width() * ratio), self._pixmap.width())
        scaled = self._pixmap.scaledToWidth(max(1, target), Qt.TransformationMode.SmoothTransformation)
        scaled.setDevicePixelRatio(ratio)
        self.setPixmap(scaled)
        self.setFixedHeight(math.ceil(scaled.height() / ratio))

    def mouseDoubleClickEvent(self, event) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(self._path))


class StepCard(QWidget):
    """One sidebar entry: numbered badge, title, and a one-line state."""

    def __init__(self, number: int, title: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("stepCard")
        self.number = number
        self.badge = QLabel(str(number))
        self.badge.setObjectName("stepNumber")
        self.badge.setFixedSize(22, 22)
        self.badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title = QLabel(title)
        self.title.setObjectName("stepTitle")
        self.detail = QLabel()
        self.detail.setObjectName("stepDetail")
        self.detail.setWordWrap(True)
        text = QVBoxLayout()
        text.setContentsMargins(0, 0, 0, 0)
        text.setSpacing(1)
        text.addWidget(self.title)
        text.addWidget(self.detail)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 9, 8, 9)
        layout.setSpacing(8)
        layout.addWidget(self.badge, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(text, 1)

    def set_state(self, state: str, detail: str) -> None:
        """state: plain, new, done, stale or locked."""
        self.badge.setText("✓" if state == "done" else str(self.number))
        self.detail.setText(detail)
        for label in (self.badge, self.title, self.detail):
            if label.property("state") != state:
                label.setProperty("state", state)
                label.style().unpolish(label)
                label.style().polish(label)


def _scroll_page() -> tuple[QScrollArea, QVBoxLayout]:
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(0, 4, 8, 10)
    layout.setSpacing(8)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.Shape.NoFrame)
    scroll.setWidget(page)
    return scroll, layout


def _heading(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "sectionTitle")
    label.setStyleSheet("font-weight: 600; font-size: 11pt; margin-top: 6px;")
    return label


def _note(text: str, color: Optional[QColor] = None) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    if color is not None:
        label.setStyleSheet(f"color: {color.name()};")
    return label


def _page(index: int, body: QWidget, *extra: QWidget) -> QWidget:
    _key, title, _subtitle, hint = PAGES[index]
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)
    heading = QLabel(title)
    heading.setObjectName("dataAgentPageTitle")
    note = _note(hint)
    note.setObjectName("dataAgentPageHint")
    layout.addWidget(heading)
    layout.addWidget(note)
    for widget in extra:
        layout.addWidget(widget)
    layout.addWidget(body, 1)
    return page


class DataAgentWindow(QMainWindow):
    """Non-modal window feeding the DIC data agent.

    A guided pipeline: Files → (Analyse) → Report → Deformation → Noise → Read quality → Solver data.
    The sidebar shows where the user is and what is left to look at; one primary button moves forward.
    """

    analysis_ready = Signal(object)
    refused = Signal(str)
    log_message = Signal(str)
    exported = Signal(object, object)

    def __init__(self, parent=None, pool: Optional[QThreadPool] = None) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.setObjectName("dataAgentWindow")
        self.setWindowTitle("DIC data agent")
        self.resize(1180, 780)
        self.setAcceptDrops(True)
        self._pool = pool or QThreadPool.globalInstance()
        self._tasks: list[_Task] = []
        self._run: Optional[DataAgentRun] = None
        self._run_signature: Optional[tuple] = None
        self._pending_signature: Optional[tuple] = None
        self._visited: set[int] = set()
        self._test_counter = 0
        self._loose_machines: list[FolderCompanions] = []
        self._loose_steps: list[FolderCompanions] = []
        self._block_reason = "Add the DIC exports and the machine data."
        self._busy = False
        self._refreshing = False
        self._banner_info = False

        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_files_page())
        self.report_view = QTextBrowser()
        self.report_view.setObjectName("dataAgentReport")
        self.report_view.setOpenLinks(False)
        self.pages.addWidget(_page(REPORT, self.report_view))
        self._fields_scroll, self._fields_layout = _scroll_page()
        self._noise_scroll, self._noise_layout = _scroll_page()
        self._quality_scroll, self._quality_layout = _scroll_page()
        self.pages.addWidget(_page(DEFORMATION, self._fields_scroll))
        self.pages.addWidget(_page(NOISE, self._noise_scroll))
        self.pages.addWidget(_page(QUALITY, self._quality_scroll))
        self.pages.addWidget(self._build_solver_page())

        self.steps = QListWidget()
        self.steps.setObjectName("dataAgentSteps")
        self.steps.setFixedWidth(250)
        self.steps.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.steps.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._cards: list[StepCard] = []
        for index, (_key, title, _subtitle, _hint) in enumerate(PAGES):
            card = StepCard(index + 1, title)
            item = QListWidgetItem()
            self.steps.addItem(item)
            self.steps.setItemWidget(item, card)
            self._cards.append(card)
        self.steps.currentRowChanged.connect(self._on_step_changed)

        self._banner = QLabel()
        self._banner.setObjectName("dataAgentBanner")
        self._banner.setWordWrap(True)
        self._banner.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._banner.hide()

        self._back = QPushButton("◀ Back")
        self._back.clicked.connect(lambda: self.go_to(self.pages.currentIndex() - 1))
        self._primary = QPushButton("Analyse ▶")
        self._primary.setObjectName("dataAgentAnalyse")
        self._primary.setMinimumWidth(170)
        self._primary.clicked.connect(self._on_primary)
        self._progress_text = QLabel()
        self._progress_text.setWordWrap(True)
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setMaximumWidth(160)
        self._progress.hide()
        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.addWidget(self._back)
        footer.addSpacing(8)
        footer.addWidget(self._progress_text, 1)
        footer.addWidget(self._progress)
        footer.addSpacing(8)
        footer.addWidget(self._primary)

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(8)
        right.addWidget(self._banner)
        right.addWidget(self.pages, 1)
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(12)
        body.addWidget(self.steps)
        body.addLayout(right, 1)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 8)
        layout.setSpacing(8)
        layout.addLayout(body, 1)
        layout.addLayout(footer)
        self.setCentralWidget(central)
        self._clear_results()
        self._refresh()
        self.steps.setCurrentRow(FILES)

    def _build_files_page(self) -> QWidget:
        self._add_files = QPushButton("Add DIC files…")
        self._add_files.clicked.connect(self._choose_files)
        self._add_machine = QPushButton("Add machine data…")
        self._add_machine.clicked.connect(self._choose_machine)
        self._add_folder = QPushButton("Add folder…")
        self._add_folder.clicked.connect(self._choose_folder)
        self._new_test = QPushButton("New test")
        self._new_test.setToolTip("Create a test; selected files move into it.")
        self._new_test.clicked.connect(self._new_group)
        self._up = QPushButton("▲")
        self._up.setToolTip("Move the selection up")
        self._up.clicked.connect(lambda: self._move_selection(-1))
        self._down = QPushButton("▼")
        self._down.setToolTip("Move the selection down")
        self._down.clicked.connect(lambda: self._move_selection(1))
        self._auto = QPushButton("Auto-order")
        self._auto.setToolTip(
            "Regroup DIC files by folder and sort by DIC frame, or by campaign step when a step map was submitted."
        )
        self._auto.clicked.connect(self._auto_order)
        self._remove = QPushButton("Remove")
        self._remove.clicked.connect(self._remove_selection)
        self._clear = QPushButton("Clear")
        self._clear.clicked.connect(self._clear_all)
        for button in (self._up, self._down):
            button.setFixedWidth(32)
        tools_host = QWidget()
        tools = QHBoxLayout(tools_host)
        tools.setContentsMargins(0, 0, 0, 0)
        for button in (self._add_files, self._add_machine, self._add_folder, self._new_test, self._up, self._down,
                       self._auto, self._remove, self._clear):
            tools.addWidget(button)
        tools.addStretch(1)

        self.requirements = QTreeWidget()
        self.requirements.setObjectName("dataAgentRequirements")
        self.requirements.setColumnCount(len(REQ_COLUMNS))
        self.requirements.setHeaderLabels(REQ_COLUMNS)
        self.requirements.setRootIsDecorated(True)
        self.requirements.setMinimumHeight(132)
        self.requirements.setMaximumHeight(210)
        self.requirements.setAcceptDrops(False)
        self.requirements.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        req_header = self.requirements.header()
        req_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        req_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        req_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.requirements.itemSelectionChanged.connect(self._update_buttons)

        self.tree = FileTree()
        self.tree.paths_dropped.connect(self.add_paths)
        self.tree.rearranged.connect(self._refresh)
        self.tree.itemSelectionChanged.connect(self._update_buttons)
        self.tree.itemChanged.connect(self._on_tree_item_changed)
        QShortcut(QKeySequence.StandardKey.Delete, self.tree, activated=self._remove_selection)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.requirements)
        layout.addWidget(self.tree, 1)
        self._summary = _note("")
        self._status = _note("")
        layout.addWidget(self._summary)
        layout.addWidget(self._status)
        return _page(FILES, body, tools_host)

    def _build_solver_page(self) -> QWidget:
        self.structure = QTreeWidget()
        self.structure.setColumnCount(len(STRUCTURE_COLUMNS))
        self.structure.setHeaderLabels(STRUCTURE_COLUMNS)
        self.structure.setAlternatingRowColors(True)
        self.structure.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.structure.header().setStretchLastSection(True)
        self._structure_note = _note("")
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.structure, 1)
        layout.addWidget(self._structure_note)
        return _page(SOLVER, body)

    # -- public API --------------------------------------------------------

    def groups(self) -> list[TestGroup]:
        out = []
        for index in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(index)
            files = [item.child(row).data(0, ROLE) for row in range(item.childCount())]
            out.append(TestGroup(name=item.text(0), files=files, companions=self._companions_of(item)))
        return out

    def add_paths(self, paths: list) -> None:
        if not paths:
            return
        self._set_busy(True, f"Checking {len(paths)} item(s)…")
        self._start(lambda _progress: _read_submission(paths), self._on_inspected, self._on_inspect_failed)

    def analyse(self) -> None:
        """Analyse the listed tests; does nothing while the current results match the file list."""
        if self._busy or self.results_current():
            return
        if self._loose_machines:
            names = ", ".join(item.sync_file.name for item in self._loose_machines if item.sync_file)
            self._refuse([f"{names} does not match one test. The DIC frames and the machine frames have to be the same set."])
            return
        groups = [group for group in self.groups() if group.files]
        reasons = validate_groups(groups)
        if reasons:
            self._refuse(reasons)
            return
        self._hide_banner()
        self._pending_signature = self._signature()
        self._set_busy(True, "Building solver data…")
        self._start(lambda progress: run_analysis(groups, progress), self._on_analysed, self._on_analyse_failed)

    def results_current(self) -> bool:
        """True when results exist and the file list has not changed since they were computed."""
        return self._run is not None and self._signature() == self._run_signature

    def go_to(self, index: int) -> None:
        if 0 <= index < len(PAGES) and self._page_enabled(index):
            self.steps.setCurrentRow(index)

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def last_run(self) -> Optional[DataAgentRun]:
        return self._run

    # -- intake ------------------------------------------------------------

    def _known_paths(self) -> set[str]:
        return {str(info.path.resolve()).lower() for group in self.groups() for info in group.files}

    def _signature(self) -> tuple:
        return tuple(
            (
                group.name,
                tuple(str(info.path) for info in group.files),
                str(group.companions.sync_file) if group.companions.sync_file else "",
                str(group.companions.step_map_file) if group.companions.step_map_file else "",
            )
            for group in self.groups()
            if group.files
        )

    def _on_inspected(self, payload: tuple) -> None:
        infos, machines, step_maps = payload
        self._set_busy(False)
        known = self._known_paths()
        fresh = [info for info in infos if str(info.path.resolve()).lower() not in known]
        self._refreshing = True
        for group in group_by_folder(fresh):
            target = self._group_for_folder(group.folders[0]) if group.folders and len(group.folders) == 1 else None
            if target is None:
                target = self._add_group(group.name)
            for info in group.files:
                target.addChild(self._file_item(info))
            target.setExpanded(True)
        self._assign_incoming(machines, step_maps)
        for index in range(self.tree.topLevelItemCount()):
            self._sort_group(self.tree.topLevelItem(index))
        self._refreshing = False
        self._refresh()
        if not fresh and not machines and not step_maps:
            self._progress_text.setText("No new files found." if not infos else "Those files are already listed.")
        refused = [info for info in fresh if not info.ok]
        if refused:
            self._refuse([f"{info.path.name}: {'; '.join(info.problems)}" for info in refused], notify=False)
        self.go_to(FILES)

    def _on_inspect_failed(self, exc: Exception) -> None:
        self._set_busy(False)
        self._show_banner(f"Could not read the inputs: {exc}", BAD)

    def _group_for_folder(self, folder: Path) -> Optional[QTreeWidgetItem]:
        for index in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(index)
            parents = {item.child(row).data(0, ROLE).path.parent for row in range(item.childCount())}
            if parents == {folder}:
                return item
        return None

    def _companions_of(self, item: QTreeWidgetItem) -> FolderCompanions:
        companions = item.data(0, COMPANIONS)
        if not isinstance(companions, FolderCompanions):
            companions = FolderCompanions(folder=Path())
            item.setData(0, COMPANIONS, companions)
        return companions

    def _add_group(self, name: str) -> QTreeWidgetItem:
        item = QTreeWidgetItem([name])
        item.setData(0, ROLE, GROUP)
        item.setData(0, COMPANIONS, FolderCompanions(folder=Path()))
        item.setFlags(
            Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEditable
            | Qt.ItemFlag.ItemIsDragEnabled | Qt.ItemFlag.ItemIsDropEnabled
        )
        font = item.font(0)
        font.setBold(True)
        item.setFont(0, font)
        item.setToolTip(0, "Double-click to rename this test.")
        self.tree.addTopLevelItem(item)
        return item

    def _file_item(self, info: RawFileInfo) -> QTreeWidgetItem:
        values = [
            info.path.name,
            "",
            "" if info.dic_number is None else str(info.dic_number),
            "",
            "",
            "",
            f"{info.n_facets:,}" if info.n_facets else "",
            "✓ raw Vic-3D" if info.ok else "✗ " + "; ".join(info.problems),
        ]
        item = QTreeWidgetItem(values)
        item.setData(0, ROLE, info)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsDragEnabled)
        item.setToolTip(0, str(info.path))
        if not info.ok:
            for column in range(len(FILE_COLUMNS)):
                item.setForeground(column, QBrush(BAD))
            item.setToolTip(len(FILE_COLUMNS) - 1, "\n".join(info.problems) + "\n\nExpected:\n" + describe_expected_format())
        elif info.warnings:
            item.setForeground(len(FILE_COLUMNS) - 1, QBrush(WARN))
            item.setText(len(FILE_COLUMNS) - 1, "✓ raw Vic-3D (see note)")
            item.setToolTip(len(FILE_COLUMNS) - 1, "\n".join(info.warnings))
        return item

    def _sort_group(self, group: QTreeWidgetItem) -> None:
        items = [group.takeChild(0) for _ in range(group.childCount())]
        step_of_dic = self._companions_of(group).step_of_dic
        items.sort(key=lambda item: order_key(item.data(0, ROLE), step_of_dic))
        for item in items:
            group.addChild(item)

    def _all_file_items(self) -> list[QTreeWidgetItem]:
        items = []
        for index in range(self.tree.topLevelItemCount()):
            group = self.tree.topLevelItem(index)
            items.extend(group.child(row) for row in range(group.childCount()))
        return items

    # -- arranging ---------------------------------------------------------

    def _new_group(self) -> None:
        self._test_counter += 1
        selected = [item for item in self.tree.selectedItems() if not is_group(item)]
        self._refreshing = True
        group = self._add_group(f"Test {self._test_counter}")
        for item in selected:
            item.parent().removeChild(item)
            group.addChild(item)
        group.setExpanded(True)
        self._refreshing = False
        self._refresh()
        self.tree.editItem(group, 0)

    def _move_selection(self, delta: int) -> None:
        selected = self.tree.selectedItems()
        if not selected:
            return
        if all(is_group(item) for item in selected):
            rows = sorted((self.tree.indexOfTopLevelItem(item) for item in selected), reverse=delta > 0)
            for row in rows:
                target = row + delta
                if 0 <= target < self.tree.topLevelItemCount():
                    item = self.tree.takeTopLevelItem(row)
                    self.tree.insertTopLevelItem(target, item)
                    item.setExpanded(True)
                    item.setSelected(True)
        elif not any(is_group(item) for item in selected):
            parents = {id(item.parent()) for item in selected}
            if len(parents) != 1:
                return
            group = selected[0].parent()
            rows = sorted((group.indexOfChild(item) for item in selected), reverse=delta > 0)
            if (delta < 0 and rows[0] == 0) or (delta > 0 and rows[0] == group.childCount() - 1):
                return
            for row in rows:
                item = group.takeChild(row)
                group.insertChild(row + delta, item)
                item.setSelected(True)
        self._refresh()

    def _remove_selection(self) -> None:
        selected = self.requirements.selectedItems()
        removed_requirement = False
        for item in selected:
            kind = item.data(0, REQ_KIND)
            index = item.data(0, REQ_INDEX)
            if kind == "machine" and isinstance(index, int):
                clear_machine(self._companions_of(self.tree.topLevelItem(index)))
                removed_requirement = True
            elif kind == "step" and isinstance(index, int):
                clear_step_map(self._companions_of(self.tree.topLevelItem(index)))
                removed_requirement = True
            elif kind == "loose-machine" and isinstance(index, int) and 0 <= index < len(self._loose_machines):
                del self._loose_machines[index]
                removed_requirement = True
            elif kind == "loose-step" and isinstance(index, int) and 0 <= index < len(self._loose_steps):
                del self._loose_steps[index]
                removed_requirement = True
        if removed_requirement:
            self.requirements.clearSelection()
            self._refresh()
            return
        for item in self.tree.selectedItems():
            if is_group(item):
                self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(item))
            elif item.parent() is not None:
                item.parent().removeChild(item)
        self._refresh()

    def _clear_all(self) -> None:
        self.tree.clear()
        self._loose_machines.clear()
        self._loose_steps.clear()
        self._test_counter = 0
        self._hide_banner()
        self._clear_results()
        self._refresh()

    def _auto_order(self) -> None:
        infos = [item.data(0, ROLE) for item in self._all_file_items()]
        saved_machines = [
            group.companions for group in self.groups() if group.companions.sync_file is not None
        ]
        saved_steps = [
            group.companions for group in self.groups() if group.companions.step_map_file is not None
        ]
        self._refreshing = True
        self.tree.clear()
        for group in group_by_folder(infos):
            item = self._add_group(group.name)
            for info in group.files:
                item.addChild(self._file_item(info))
            item.setExpanded(True)
        self._assign_incoming(saved_machines, saved_steps)
        for index in range(self.tree.topLevelItemCount()):
            self._sort_group(self.tree.topLevelItem(index))
        self._refreshing = False
        self._refresh()

    def _on_tree_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if not self._refreshing and column == 0 and is_group(item):
            self._update_buttons()

    def _assign_incoming(self, machines: list[FolderCompanions], step_maps: list[FolderCompanions]) -> None:
        self._loose_machines, self._loose_steps = assign_companions(
            self.groups(),
            _dedupe_companions([*machines, *self._loose_machines], "sync_file"),
            _dedupe_companions([*step_maps, *self._loose_steps], "step_map_file"),
        )

    def _release_empty_companions(self) -> None:
        """A test with no DIC files cannot hold machine data; put it back for another test."""
        for index in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(index)
            if item.childCount():
                continue
            companions = self._companions_of(item)
            if companions.sync_file is None and companions.step_map_file is None:
                continue
            if companions.sync_file is not None:
                self._loose_machines.append(companions)
            if companions.step_map_file is not None:
                self._loose_steps.append(companions)
            item.setData(0, COMPANIONS, FolderCompanions(folder=Path()))

    def _requirement_row(
        self,
        parent: Optional[QTreeWidgetItem],
        requirement: str,
        submitted: str,
        status: str,
        color: QColor,
        kind: str,
        index: Optional[int],
        tip: str = "",
    ) -> None:
        item = QTreeWidgetItem([requirement, submitted, status])
        item.setData(0, REQ_KIND, kind)
        item.setData(0, REQ_INDEX, index)
        item.setForeground(2, QBrush(color))
        if status != "Submitted" and status != "Matched":
            item.setForeground(1, QBrush(color))
        if tip:
            item.setToolTip(1, tip)
        if parent is None:
            self.requirements.addTopLevelItem(item)
        else:
            parent.addChild(item)

    def _fill_requirements(self) -> None:
        """Checklist of the two required submissions and whether their frames match."""
        self.requirements.clear()
        groups = self.groups()
        populated = [group for group in groups if group.files]
        if not populated and not self._loose_machines and not self._loose_steps:
            self._requirement_row(None, "DIC exports", "exp-<frame>_<camera>.csv", "Required", WARN, "dic", None)
            self._requirement_row(None, "Machine data", SYNC_NAME, "Required", WARN, "machine", None)
            self._requirement_row(
                None, "Frame match", "The same DIC frame numbers in both files", "Waiting", WARN, "match", None
            )
            self._block_reason = "Add the DIC exports and the machine data."
            return
        reasons: list[str] = []
        for index, group in enumerate(groups):
            if not group.files:
                continue
            parent = QTreeWidgetItem([group.name, "", ""])
            font = parent.font(0)
            font.setBold(True)
            parent.setFont(0, font)
            parent.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.requirements.addTopLevelItem(parent)
            self._requirement_row(
                parent,
                "DIC exports",
                f"{len(group.files)} file{'s' if len(group.files) != 1 else ''}",
                "Submitted",
                OK,
                "dic",
                index,
            )
            machine = group.companions.sync_file
            self._requirement_row(
                parent,
                "Machine data",
                machine.name if machine is not None else SYNC_NAME,
                "Submitted" if machine is not None else "Required",
                OK if machine is not None else WARN,
                "machine",
                index,
                "Select this row and press Remove to drop the machine file." if machine is not None else "",
            )
            match = frame_match(group)
            if match.ready:
                detail, status, color = f"{match.dic_count} DIC frames", "Matched", OK
            else:
                detail, status, color = match.problems[0], "Not matched", BAD
                reasons.append(f"{group.name}: {match.problems[0]}")
            self._requirement_row(parent, "Frame match", detail, status, color, "match", index)
            step_map = group.companions.step_map_file
            if step_map is not None:
                self._requirement_row(
                    parent,
                    "Step map (optional)",
                    step_map.name,
                    "Submitted",
                    INFO,
                    "step",
                    index,
                    "Optional. Used only for campaign step and time. Select this row and press Remove to drop it.",
                )
            parent.setExpanded(True)
        for index, machine in enumerate(self._loose_machines):
            name = machine.sync_file.name if machine.sync_file is not None else "machine data"
            detail = machine.notes[0] if machine.notes else "These frames do not match one test."
            self._requirement_row(None, "Machine data", name, "Not matched to a test", BAD, "loose-machine", index, detail)
            reasons.append(f"{name} does not match one test.")
        for index, step in enumerate(self._loose_steps):
            name = step.step_map_file.name if step.step_map_file is not None else "step map"
            self._requirement_row(None, "Step map (optional)", name, "Not matched to a test", WARN, "loose-step", index)
        if not populated:
            reasons.insert(0, "Add the DIC exports.")
        self._block_reason = reasons[0] if reasons else ""

    def _refresh(self) -> None:
        self._release_empty_companions()
        self._refreshing = True
        self._assign_incoming([], [])
        tests = 0
        files = 0
        bad = 0
        for index in range(self.tree.topLevelItemCount()):
            group = self.tree.topLevelItem(index)
            companions = self._companions_of(group)
            count = group.childCount()
            group_bad = 0
            for row in range(count):
                child = group.child(row)
                info = child.data(0, ROLE)
                child.setText(1, str(row + 1))
                step = time = None
                if info.dic_number is not None:
                    step = companions.step_of_dic.get(info.dic_number)
                    time = companions.time_of_dic.get(info.dic_number)
                child.setText(3, "" if step is None else str(step))
                child.setText(4, "" if time is None else f"{time:.1f}")
                child.setText(5, format_load(info, companions) if info.ok else "")
                group_bad += 0 if info.ok else 1
            folders = sorted({str(group.child(row).data(0, ROLE).path.parent) for row in range(count)})
            group.setText(1, f"{count} step{'s' if count != 1 else ''}")
            group.setText(7, f"✗ {group_bad} refused" if group_bad else ("✓" if count else "empty"))
            group.setForeground(7, QBrush(BAD) if group_bad else QBrush())
            group.setToolTip(1, "\n".join(folders))
            if count:
                tests += 1
            files += count
            bad += group_bad
        self._fill_requirements()
        self._refreshing = False
        if files:
            text = f"{tests} test{'s' if tests != 1 else ''}, {files} DIC file{'s' if files != 1 else ''}."
            if bad:
                text += f"  {bad} file{'s' if bad != 1 else ''} do not match the raw format and must be removed."
            elif self._block_reason:
                text += "  " + self._block_reason
            else:
                text += "  DIC frames match the machine data. Step 1 of each test is the reference for increments."
        else:
            text = (
                f"No files yet. Add DIC exports (exp-<frame>_<camera>.csv) and machine data ({SYNC_NAME}). "
                "Both are required, and they are used only when the frame numbers match."
            )
        self._summary.setText(text)
        self._file_counts = (tests, files, bad)
        self._update_buttons()

    # -- navigation --------------------------------------------------------

    def _page_enabled(self, index: int) -> bool:
        return index == FILES or self._run is not None

    def _on_step_changed(self, row: int) -> None:
        if row < 0:
            return
        if not self._page_enabled(row):
            self.steps.setCurrentRow(self.pages.currentIndex())
            return
        self.pages.setCurrentIndex(row)
        if row != FILES and self._run is not None:
            self._visited.add(row)
        self._update_buttons()

    def _on_primary(self) -> None:
        current = self.pages.currentIndex()
        if current == FILES and not self.results_current():
            self.analyse()
        elif current < SOLVER:
            self.go_to(current + 1)
        else:
            self._export_mats()

    def _update_steps(self) -> None:
        has_run = self._run is not None
        current = self.results_current()
        tests, files, bad = getattr(self, "_file_counts", (0, 0, 0))
        for index, card in enumerate(self._cards):
            item = self.steps.item(index)
            enabled = self._page_enabled(index)
            flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable if enabled else Qt.ItemFlag.NoItemFlags
            item.setFlags(flags)
            subtitle = PAGES[index][2]
            if index == FILES:
                if not files:
                    card.set_state("new", subtitle)
                elif bad:
                    card.set_state("stale", f"{bad} file{'s' if bad != 1 else ''} refused; remove them")
                elif self._block_reason:
                    card.set_state("stale", self._block_reason)
                elif current:
                    card.set_state("done", f"{tests} test{'s' if tests != 1 else ''}, {files} files · analysed")
                elif has_run:
                    card.set_state("stale", "Changed since the analysis")
                else:
                    card.set_state("plain", f"{tests} test{'s' if tests != 1 else ''}, {files} files · ready")
            elif not has_run:
                card.set_state("locked", "Available after Analyse")
            elif not current:
                card.set_state("stale", "Outdated: analyse again")
            elif index in self._visited:
                card.set_state("done", subtitle)
            else:
                card.set_state("new", subtitle + " · not opened yet")
        self._fit_step_cards()

    def _fit_step_cards(self) -> None:
        """Size each card for its wrapped text at the sidebar's real width and font."""
        width = self.steps.viewport().width()
        if width <= 50:
            width = self.steps.width() - 2 * self.steps.frameWidth()
        for index, card in enumerate(self._cards):
            card.ensurePolished()
            card.layout().activate()
            height = card.heightForWidth(width)
            if height <= 0:
                height = card.sizeHint().height()
            self.steps.item(index).setSizeHint(QSize(width, height))

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._fit_step_cards()

    def _update_nav(self) -> None:
        busy = self._busy
        page = self.pages.currentIndex()
        current = self.results_current()
        has_files = bool(self._all_file_items())
        blocked = bool(self._block_reason)
        self._back.setEnabled(not busy and page > FILES)
        if page == FILES and not current:
            if busy and self._pending_signature is not None:
                text, enabled, tip = "Analysing…", False, ""
            else:
                text = "Analyse again ▶" if self._run is not None else "Analyse ▶"
                enabled = not busy and has_files and not blocked
                tip = self._block_reason or "Convert the listed files, measure them, and send the facts to the Assistant."
        elif page < SOLVER:
            text, enabled = f"Next: {PAGES[page + 1][1]} ▶", not busy
            tip = "The listed files are already analysed; change the files to analyse again." if page == FILES else ""
        else:
            text, enabled = "Export .mat files…", not busy and self._run is not None
            tip = "Write one Layout A .mat per step into a new folder (never into the raw-data folders)."
        self._primary.setText(text)
        self._primary.setEnabled(enabled)
        self._primary.setToolTip(tip)
        if self._banner_info and not current:
            self._hide_banner()
        if self._run is None:
            self._status.setText("")
        elif current:
            self._status.setText(
                "✓ These files are analysed. Review the results with Next ▶ or the steps on the left; "
                "the Assistant's interpretation is in the chat panel."
            )
            self._status.setStyleSheet("color: #2e7d32;")
        else:
            self._status.setText(
                "The files changed since the last analysis, so the results on the left are outdated. "
                "Press Analyse again to update them (or undo the change)."
            )
            self._status.setStyleSheet(f"color: {WARN.name()};")

    def _update_buttons(self) -> None:
        busy = self._busy
        has_files = bool(self._all_file_items())
        selected = bool(self.tree.selectedItems())
        for button in (self._add_files, self._add_machine, self._add_folder, self._new_test):
            button.setEnabled(not busy)
        for button in (self._up, self._down, self._remove):
            button.setEnabled(not busy and selected)
        self._auto.setEnabled(not busy and has_files)
        self._clear.setEnabled(not busy and has_files)
        self._update_steps()
        self._update_nav()

    # -- analysis ----------------------------------------------------------

    def _on_analysed(self, run: DataAgentRun) -> None:
        self._run_signature = self._pending_signature
        self._pending_signature = None
        self._set_busy(False, f"Done. Results in {run.workdir}")
        self._run = run
        self._show_results(run)
        self._show_banner(
            "Analysis done. The Assistant is now interpreting these facts in the chat panel. Work through the "
            "steps on the left (Next ▶ moves forward); accepting or rejecting the test stays your call.",
            INFO,
        )
        self.log_message.emit(f"DIC data agent: analysed {len(run.analyses)} test(s); facts in {run.facts_path}")
        self.analysis_ready.emit(run)

    def _on_analyse_failed(self, exc: Exception) -> None:
        self._pending_signature = None
        self._set_busy(False)
        if isinstance(exc, DataRefused):
            self._refuse(exc.reasons)
        else:
            self._show_banner(f"Analysis failed: {exc}", BAD)
            self.log_message.emit(f"DIC data agent: analysis failed: {exc!r}")

    def _refuse(self, reasons: list[str], notify: bool = True) -> None:
        shown = reasons[:8] + ([f"… and {len(reasons) - 8} more"] if len(reasons) > 8 else [])
        text = (
            "Refused: the data does not match the raw Vic-3D export format the agent treats. "
            "Nothing was analysed.\n• " + "\n• ".join(shown)
        )
        self._show_banner(text, BAD)
        if notify:
            self.refused.emit(text)

    def _clear_results(self) -> None:
        self._run = None
        self._run_signature = None
        self._visited = set()
        self.report_view.clear()
        self.structure.clear()
        self._structure_note.setText("")
        for layout in (self._fields_layout, self._noise_layout, self._quality_layout):
            while layout.count():
                child = layout.takeAt(0)
                if child.widget() is not None:
                    child.widget().deleteLater()
            layout.addStretch(1)
        if self.pages.currentIndex() != FILES:
            self.steps.setCurrentRow(FILES)
        self._update_buttons()

    def _show_results(self, run: DataAgentRun) -> None:
        self._visited = set()
        for layout in (self._fields_layout, self._noise_layout, self._quality_layout):
            while layout.count():
                child = layout.takeAt(0)
                if child.widget() is not None:
                    child.widget().deleteLater()
        self.report_view.setMarkdown(run.report_markdown)
        self.structure.clear()
        self._structure_note.setText(f"Working copy of the analysis (scratch folder): {run.workdir}")
        for analysis in run.analyses:
            facts = analysis.facts
            frame = facts["deformation"].get("curvature_frame_check", {})
            inconsistent = str(frame.get("verdict", "")).startswith("INCONSISTENT")
            test_item = QTreeWidgetItem([analysis.name, f"{len(analysis.payloads)} steps", "", "",
                                         frame.get("verdict", "")])
            font = test_item.font(0)
            font.setBold(True)
            test_item.setFont(0, font)
            if inconsistent:
                test_item.setForeground(4, QBrush(BAD))
            self.structure.addTopLevelItem(test_item)
            for index, (info, payload) in enumerate(zip(analysis.files, analysis.payloads)):
                step_item = QTreeWidgetItem([
                    f"step {index + 1} · {info.path.name}", "", "", "", format_load(info, analysis.companions)
                ])
                test_item.addChild(step_item)
                for row in describe_payload(payload):
                    step_item.addChild(QTreeWidgetItem(list(row)))
                if index == 0:
                    step_item.setExpanded(True)
            test_item.setExpanded(True)
            self._fill_fields(analysis, run, inconsistent, frame)
            self._fill_noise(analysis, run)
            self._fill_quality(analysis, run)
        for layout in (self._fields_layout, self._noise_layout, self._quality_layout):
            layout.addStretch(1)
        self._update_buttons()
        self.go_to(REPORT)

    def _figure(self, layout: QVBoxLayout, run: DataAgentRun, name: str, key: str) -> None:
        relative = run.figures.get(name, {}).get(key)
        if relative:
            layout.addWidget(ScaledImage(str(run.workdir / relative)))

    def _fill_fields(self, analysis, run, inconsistent: bool, frame: dict) -> None:
        facts = analysis.facts
        deformation = facts["deformation"]
        specimen = facts["specimen"]
        layout = self._fields_layout
        layout.addWidget(_heading(analysis.name))
        means = deformation["interior_mean_dk_at_final_step_inv_mm"]
        lines = [
            f"Measured region {specimen['measured_span_mm']['x']} × {specimen['measured_span_mm']['y']} mm, "
            f"{specimen['n_steps']} steps, camera {specimen.get('camera_image_axes', '')}.",
            f"Dominant machine moment {deformation['dominant_machine_moment'] or '—'}, "
            f"dominant curvature {deformation['dominant_curvature_component']}; interior mean Δκ at step "
            f"{deformation['final_step_analysed']}: "
            + ", ".join(f"{key} {value:+.3g}" for key, value in means.items() if value is not None)
            + " 1/mm.",
        ]
        layout.addWidget(_note("  ".join(lines)))
        if frame:
            layout.addWidget(_note("Curvature frame check: " + frame.get("verdict", ""), BAD if inconsistent else None))
        self._figure(layout, run, analysis.name, "fields")
        self._figure(layout, run, analysis.name, "response")

    def _fill_noise(self, analysis, run) -> None:
        noise = analysis.facts["noise"]
        layout = self._noise_layout
        layout.addWidget(_heading(analysis.name))
        if not noise.get("available"):
            layout.addWidget(_note(f"Noise not measured: {noise.get('reason', 'not enough steps')}.", WARN))
            self._fill_dic_parameters(analysis, run, layout)
            return
        rows = [
            "<tr><th align=left>component</th><th>measured interior</th><th>model interior</th>"
            "<th>measured / model (all facets)</th><th>edge amplification meas. / model</th>"
            "<th>legacy envelope / measured</th><th>map correlation</th><th>SNR at max load</th></tr>"
        ]
        snr = noise.get("signal_to_noise_at_max_load", {})
        for comp, stats in noise["components"].items():
            rows.append(
                f"<tr><td>{comp}</td><td align=right>{stats['measured_interior_median_inv_mm']:.3g}</td>"
                f"<td align=right>{stats['model_interior_median_inv_mm']:.3g}</td>"
                f"<td align=right>{stats['measured_to_model_ratio']:.2f}</td>"
                f"<td align=right>{stats['measured_edge_amplification']:.2f} / {stats['model_edge_amplification']:.2f}</td>"
                f"<td align=right>{stats['legacy_to_measured_ratio']:.2f}</td>"
                f"<td align=right>{stats['pointwise_map_correlation']:.2f}</td>"
                f"<td align=right>{snr.get(comp, '—')}</td></tr>"
            )
        table = QLabel("<table cellspacing=0 cellpadding=4>" + "".join(rows) + "</table>")
        table.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(table)
        layout.addWidget(_note(
            f"σ in 1/mm. Measured: per-facet scatter about a quadratic in {noise.get('load_variable', 'load')} "
            f"({noise.get('residual_dof_per_facet')} residual dof per facet). Model: interior floor × local "
            "stencil amplification, on the same facets."
        ))
        self._figure(layout, run, analysis.name, "noise_maps")
        self._figure(layout, run, analysis.name, "noise_profile")
        self._fill_dic_parameters(analysis, run, layout)

    def _fill_dic_parameters(self, analysis, run, layout) -> None:
        params = analysis.facts.get("dic_parameters")
        if not params:
            return
        layout.addWidget(_heading(f"{analysis.name}: DIC parameters and curvature filter"))
        rec = params["recorded_in_export"]
        lines = [
            f"Recorded in the export: facet step {rec['facet_step_px']} px, pitch "
            f"{rec['facet_pitch_mm']['x']:.3g} × {rec['facet_pitch_mm']['y']:.3g} mm, image scale "
            f"{rec['image_scale_mm_per_px']['x']:.3g} × {rec['image_scale_mm_per_px']['y']:.3g} mm/px."
        ]
        subset = params.get("subset") or {}
        if subset.get("subset_size_estimate_px"):
            lines.append(f"Subset size (estimated from the noise correlation length): "
                         f"≈ {subset['subset_size_estimate_px']:g} px.")
        else:
            lines.append(f"Subset size not estimated: {subset.get('reason', 'noise correlation too long')}.")
        filt = params.get("curvature_filter") or {}
        if filt.get("available"):
            best = filt["best"]
            model = filt.get("noise_model_assumption") or {}
            lines.append(
                f"Curvature filter (inferred): {best['filter_size_facets']} facets, {best['weighting']}; "
                f"mismatch to the exported Δκ {best['mismatch_mean']:.2f} ({filt['verdict'].split(':')[0]})."
                + (f" Noise-model stencil (uniform {model['filter_size_facets']} facets): "
                   f"mismatch {model['mismatch_mean']:.2f}." if model.get("mismatch_mean") is not None else "")
            )
        else:
            lines.append(f"Curvature filter not identified: {filt.get('reason', 'unknown reason')}.")
        layout.addWidget(_note("  ".join(lines)))
        sweep = params.get("curvature_filter_sweep") or {}
        if sweep.get("rows"):
            dominant = sweep["dominant_component"]

            def cell(value, fmt=".3g"):
                return "—" if value is None else format(value, fmt)

            rows = [
                "<tr><th align=left>filter [facets]</th><th>window x [mm]</th>"
                f"<th>interior noise {dominant} [1/mm]</th><th>SNR</th><th>smoothing bias index</th>"
                "<th>full-window facets</th><th>mismatch to export</th></tr>"
            ]
            for row in sweep["rows"]:
                size = row["filter_size_facets"]
                mark = " ◀ inferred" if size == sweep["identified_filter_size_facets"] else ""
                rows.append(
                    f"<tr><td>{size}{mark}</td><td align=right>{cell(row['window_mm']['x'])}</td>"
                    f"<td align=right>{cell((row.get('noise_interior_median_inv_mm') or {}).get(dominant))}</td>"
                    f"<td align=right>{cell(row.get('snr_interior'))}</td>"
                    f"<td align=right>{cell(row.get('smoothing_bias_index'))}</td>"
                    f"<td align=right>{cell(row['full_window_facet_fraction'], '.0%')}</td>"
                    f"<td align=right>{cell(row.get('mismatch_to_export'))}</td></tr>"
                )
            table = QLabel("<table cellspacing=0 cellpadding=4>" + "".join(rows) + "</table>")
            table.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            layout.addWidget(table)
            layout.addWidget(_note(
                f"Each row re-fits ΔW with that filter size ({sweep['weighting']}). Signal is the interior mean "
                f"Δ{dominant} at step {sweep['final_step_analysed']}; noise uses the same temporal-residual "
                "estimator as above. Smoothing bias index 1 means the window follows ΔW as well as the smallest one."
            ))
        self._figure(layout, run, analysis.name, "filter_sweep")

    def _fill_quality(self, analysis, run) -> None:
        quality = analysis.facts["read_quality"]
        layout = self._quality_layout
        layout.addWidget(_heading(analysis.name))
        layout.addWidget(_note(
            f"{quality['facets_union']:,} facets read in at least one step, {quality['facets_in_every_step']:,} in "
            f"every step. Vic-3D correlation sigma median {quality['vic_sigma_specimen_median_px']} px. "
            f"Zones: {quality['zone_layout']}."
        ))
        flagged = [zone for zone in quality["zones"] if zone["flags"]]
        if flagged:
            text = "\n".join(f"• {zone['zone']}: {'; '.join(zone['flags'])}" for zone in flagged)
            layout.addWidget(_note("Poorly read zones:\n" + text, WARN))
        else:
            layout.addWidget(_note("No zone crosses the dropout, correlation-sigma or noise-ratio thresholds."))
        self._figure(layout, run, analysis.name, "quality")

    def _export_mats(self) -> None:
        if self._run is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Export solver .mat files into…")
        if not folder:
            return
        try:
            written = export_solver_mats(self._run, Path(folder))
        except DataRefused as exc:
            QMessageBox.warning(self, "Export refused", "\n".join(exc.reasons))
            return
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.log_message.emit(f"DIC data agent: exported {len(written)} .mat file(s) to {folder}")
        self.exported.emit(self._run, written)
        QMessageBox.information(self, "Export done", f"Wrote {len(written)} .mat file(s) under\n{folder}")

    # -- plumbing ----------------------------------------------------------

    def _choose_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "DIC exports", "", "Vic-3D CSV (*.csv);;All files (*)"
        )
        self.add_paths(paths)

    def _choose_machine(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Machine data", "", "Machine data (*.csv);;All files (*)"
        )
        self.add_paths(paths)

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Test folder with DIC exports and machine data")
        if folder:
            self.add_paths([folder])

    def _start(self, work, on_done, on_fail) -> None:
        task = _Task(work)
        task.signals.finished.connect(on_done)
        task.signals.failed.connect(on_fail)
        task.signals.progress.connect(self._progress_text.setText)
        task.signals.finished.connect(lambda _result, task=task: self._forget(task))
        task.signals.failed.connect(lambda _exc, task=task: self._forget(task))
        self._tasks.append(task)
        self._pool.start(task)

    def _forget(self, task: _Task) -> None:
        if task in self._tasks:
            self._tasks.remove(task)

    def _set_busy(self, busy: bool, text: str = "") -> None:
        self._busy = busy
        self._progress.setVisible(busy)
        if text or not busy:
            self._progress_text.setText(text)
        self.tree.setEnabled(not busy)
        self.requirements.setEnabled(not busy)
        self.steps.setEnabled(not busy)
        self._update_buttons()

    def _show_banner(self, text: str, color: QColor) -> None:
        self._banner_info = color == INFO
        self._banner.setText(text)
        self._banner.setStyleSheet(
            f"background: rgba({color.red()}, {color.green()}, {color.blue()}, 30); "
            f"border: 1px solid {color.name()}; border-radius: 4px; "
            f"padding: 6px; color: {color.darker(130).name()};"
        )
        self._banner.show()

    def _hide_banner(self) -> None:
        self._banner_info = False
        self._banner.hide()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            event.acceptProposedAction()
            self.add_paths(paths)

    def wait_idle(self, timeout_ms: int = 600000) -> bool:
        """Block until background work finishes (tests and scripted use)."""
        return self._pool.waitForDone(timeout_ms)
