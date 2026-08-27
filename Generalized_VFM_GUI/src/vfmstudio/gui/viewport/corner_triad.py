"""Screen-fixed XYZ triad overlay (Abaqus-style view triad)."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

_AXIS = (
    ("X", QColor("#C0392B")),
    ("Y", QColor("#1B8F5A")),
    ("Z", QColor("#2A78D6")),
)


class CornerTriad(QWidget):
    """Draws a camera-aligned RGB triad in the lower-left of the 3D view."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(96, 96)
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(255, 255, 255))
        self.setPalette(palette)
        self._axes = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]

    def set_from_camera(self, plotter) -> None:
        try:
            cam = plotter.camera
            look = np.asarray(cam.GetDirectionOfProjection(), dtype=float)
            up = np.asarray(cam.GetViewUp(), dtype=float)
        except Exception:
            return
        right = np.cross(look, up)
        rn = float(np.linalg.norm(right))
        un = float(np.linalg.norm(up))
        if rn < 1e-9 or un < 1e-9:
            return
        right = right / rn
        up = up / un
        look = look / (float(np.linalg.norm(look)) + 1e-12)
        axes = []
        for basis in np.eye(3):
            sx = float(np.dot(basis, right))
            sy = float(np.dot(basis, up))
            depth = float(np.dot(basis, -look))
            axes.append((sx, sy, depth))
        self._axes = axes
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ARG002
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(QColor("#A8A8A8"), 1))
        painter.setBrush(QColor(248, 248, 248, 230))
        painter.drawRoundedRect(1, 1, self.width() - 2, self.height() - 2, 6, 6)
        origin = QPointF(self.width() * 0.46, self.height() * 0.58)
        length = 34.0
        order = sorted(range(3), key=lambda i: self._axes[i][2])
        font = QFont("Arial", 9)
        font.setBold(True)
        painter.setFont(font)
        for index in order:
            sx, sy, _depth = self._axes[index]
            label, color = _AXIS[index]
            tip = QPointF(origin.x() + sx * length, origin.y() - sy * length)
            painter.setPen(QPen(color, 3))
            painter.drawLine(origin, tip)
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(_arrow_head(origin, tip))
            painter.setPen(color)
            painter.drawText(tip + QPointF(4, -4), label)
        painter.end()


def _arrow_head(origin: QPointF, tip: QPointF) -> QPolygonF:
    dx = tip.x() - origin.x()
    dy = tip.y() - origin.y()
    norm = (dx * dx + dy * dy) ** 0.5 or 1.0
    ux, uy = dx / norm, dy / norm
    px, py = -uy, ux
    return QPolygonF(
        [
            tip,
            QPointF(tip.x() - 9 * ux + 4 * px, tip.y() - 9 * uy + 4 * py),
            QPointF(tip.x() - 9 * ux - 4 * px, tip.y() - 9 * uy - 4 * py),
        ]
    )
