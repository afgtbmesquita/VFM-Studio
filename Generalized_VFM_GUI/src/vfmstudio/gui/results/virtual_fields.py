"""Note panel for optimal virtual fields (viewport owns the warp overlay)."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class VirtualFieldPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._label = QLabel(
            "After identification, use Results → Show virtual field in viewport "
            "to warp the mesh by w*."
        )
        self._label.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self._label)

    def update_from(self, result: Any) -> None:
        fields = getattr(result, "optimal_field_results", None) or []
        self._label.setText(
            f"{len(fields)} optimal virtual field(s) stored on the result. "
            "Viewport → Virtual field overlays w* on the mesh."
        )
