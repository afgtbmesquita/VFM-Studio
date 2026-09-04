"""Import Vic-3D / measurement ``.mat`` files as load steps."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
)

from vfmstudio.core.document import LoadStepModel


def natural_path_key(path: Path | str):
    """Sort key so ``step2`` / ``step09`` / ``step10`` order numerically."""
    name = Path(path).name
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", name)]


def _path_identity(path: Path) -> str:
    try:
        if path.exists():
            return str(path.resolve())
    except OSError:
        pass
    return str(path)


def load_steps_from_paths(
    paths: Iterable[Path | str],
    *,
    skip: Optional[Sequence[Path | str]] = None,
) -> List[LoadStepModel]:
    """Build load steps from MAT paths, naturally sorted and de-duplicated."""
    skip_ids = {_path_identity(Path(item)) for item in (skip or [])}
    steps: List[LoadStepModel] = []
    seen: set[str] = set()
    for path in sorted((Path(item) for item in paths), key=natural_path_key):
        if not str(path).strip():
            continue
        identity = _path_identity(path)
        if identity in seen or identity in skip_ids:
            continue
        seen.add(identity)
        steps.append(LoadStepModel(data_file=path, label=path.stem))
    return steps


class ImportMatDialog(QDialog):
    """Pick one or many measurement ``.mat`` files for the same experiment."""

    def __init__(self, parent=None, *, start_directory: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("Import MAT files")
        self.resize(520, 280)
        self._start_directory = start_directory or ""
        self._paths: List[Path] = []

        hint = QLabel(
            "Select one or more measurement MAT files. They are added as "
            "separate load steps on the current experiment (natural filename order)."
        )
        hint.setWordWrap(True)
        hint.setProperty("role", "hint")

        self._list = QListWidget()
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        clear = QPushButton("Clear")
        clear.clicked.connect(self._clear)
        row = QHBoxLayout()
        row.addWidget(browse)
        row.addWidget(clear)
        row.addStretch(1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(hint)
        layout.addWidget(self._list, 1)
        layout.addLayout(row)
        layout.addWidget(buttons)

    def _browse(self) -> None:
        chosen, _ = QFileDialog.getOpenFileNames(
            self,
            "Measurement files",
            self._start_directory,
            "MAT (*.mat);;All files (*)",
        )
        if not chosen:
            return
        self._start_directory = str(Path(chosen[0]).parent)
        existing = {_path_identity(path) for path in self._paths}
        for text in chosen:
            path = Path(text)
            identity = _path_identity(path)
            if identity in existing:
                continue
            existing.add(identity)
            self._paths.append(path)
        self._paths.sort(key=natural_path_key)
        self._rebuild_list()

    def _clear(self) -> None:
        self._paths = []
        self._rebuild_list()

    def _rebuild_list(self) -> None:
        self._list.clear()
        for path in self._paths:
            self._list.addItem(str(path))

    def steps(self, skip: Optional[Sequence[Path | str]] = None) -> List[LoadStepModel]:
        return load_steps_from_paths(self._paths, skip=skip)

    def step(self) -> Optional[LoadStepModel]:
        steps = self.steps()
        return steps[0] if steps else None


AddStepDialog = ImportMatDialog
