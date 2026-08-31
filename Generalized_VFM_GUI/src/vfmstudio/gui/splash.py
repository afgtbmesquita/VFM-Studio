"""Startup banner shown while the GUI and visualization stack load.

Depends only on Qt (and ``branding``) so it can appear before VTK, PyVista,
and the main window are imported. The official high-resolution logo is the
lockup; supporting copy is painted around it.
"""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, QSize, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetricsF,
    QImage,
    QLinearGradient,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PySide6.QtWidgets import QApplication, QWidget

from vfmstudio.branding import APP_AUTHOR, APP_LICENSE_LINE, APP_NAME
from vfmstudio.resources import app_icon

BANNER_WIDTH = 960
BANNER_HEIGHT = 540

_BG_CENTER = QColor("#fafcfe")
_BG_EDGE = QColor("#e8eef4")
_NAVY = QColor("#1A2F4A")
_CREDIT = QColor("#6B7784")
_FOOTER = QColor("#8A96A2")
_TEAL = QColor("#2BB8C8")

_LOGO_MASTER: QPixmap | None = None


def _splash_enabled() -> bool:
    return os.environ.get("VFMSTUDIO_NO_SPLASH", "").strip().lower() not in {
        "1",
        "true",
        "yes",
    }


def logo_path() -> Path | None:
    folder = Path(__file__).resolve().parents[1] / "resources"
    for name in ("LamXtract_Logo.png", "LamXtract_Logo.jpg"):
        path = folder / name
        if path.is_file():
            return path
    return None


def _font(pixel_size: int, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    font = QFont()
    font.setFamilies(["Segoe UI", "Arial", "Helvetica", "sans-serif"])
    font.setPixelSize(max(11, int(pixel_size)))
    font.setWeight(weight)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    return font


def _opaque_bounds(image: QImage, alpha_min: int = 18) -> QRect:
    x0, y0, x1, y1 = image.width(), image.height(), -1, -1
    for y in range(image.height()):
        for x in range(image.width()):
            if image.pixelColor(x, y).alpha() < alpha_min:
                continue
            x0 = min(x0, x)
            y0 = min(y0, y)
            x1 = max(x1, x)
            y1 = max(y1, y)
    if x1 < 0:
        return QRect(0, 0, image.width(), image.height())
    pad = 4
    return QRect(
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(image.width(), x1 - x0 + 1 + 2 * pad),
        min(image.height(), y1 - y0 + 1 + 2 * pad),
    )


def _prepare_logo_image(image: QImage) -> QImage:
    """Remove the studio-white matte so the lockup blends with the splash."""
    img = image.convertToFormat(QImage.Format.Format_ARGB32)
    for y in range(img.height()):
        for x in range(img.width()):
            color = img.pixelColor(x, y)
            floor = min(color.red(), color.green(), color.blue())
            if floor >= 250:
                color.setAlpha(0)
            elif floor >= 236:
                color.setAlpha(int(color.alpha() * (250 - floor) / 14))
            img.setPixelColor(x, y, color)
    return img.copy(_opaque_bounds(img))


def _image_has_transparency(image: QImage) -> bool:
    if not image.hasAlphaChannel():
        return False
    for x, y in (
        (0, 0),
        (image.width() - 1, 0),
        (0, image.height() - 1),
        (image.width() - 1, image.height() - 1),
    ):
        if image.pixelColor(x, y).alpha() < 16:
            return True
    return False


def _load_logo_master() -> QPixmap | None:
    global _LOGO_MASTER
    if _LOGO_MASTER is not None and not _LOGO_MASTER.isNull():
        return _LOGO_MASTER
    path = logo_path()
    if path is None:
        return None
    image = QImage(str(path))
    if image.isNull():
        return None
    if not _image_has_transparency(image):
        image = _prepare_logo_image(image)
    _LOGO_MASTER = QPixmap.fromImage(image)
    return _LOGO_MASTER


def _draw_faded_background(painter: QPainter, width: float, height: float) -> None:
    """Soft washes that echo the logo navy and cyan — no grid."""
    base = QLinearGradient(0, 0, width, height)
    base.setColorAt(0.0, _BG_CENTER)
    base.setColorAt(0.55, QColor("#f3f7fb"))
    base.setColorAt(1.0, _BG_EDGE)
    painter.fillRect(QRectF(0, 0, width, height), QBrush(base))

    center_glow = QRadialGradient(width * 0.50, height * 0.42, width * 0.55)
    center_glow.setColorAt(0.0, QColor(255, 255, 255, 120))
    center_glow.setColorAt(0.55, QColor(255, 255, 255, 24))
    center_glow.setColorAt(1.0, QColor(255, 255, 255, 0))
    painter.fillRect(QRectF(0, 0, width, height), QBrush(center_glow))

    cyan_corner = QRadialGradient(width * 0.92, height * 0.08, width * 0.42)
    cyan_corner.setColorAt(0.0, QColor(78, 212, 230, 52))
    cyan_corner.setColorAt(0.45, QColor(43, 184, 200, 18))
    cyan_corner.setColorAt(1.0, QColor(43, 184, 200, 0))
    painter.fillRect(QRectF(0, 0, width, height), QBrush(cyan_corner))

    navy_corner = QRadialGradient(width * 0.06, height * 0.94, width * 0.40)
    navy_corner.setColorAt(0.0, QColor(26, 47, 74, 34))
    navy_corner.setColorAt(0.50, QColor(26, 47, 74, 10))
    navy_corner.setColorAt(1.0, QColor(26, 47, 74, 0))
    painter.fillRect(QRectF(0, 0, width, height), QBrush(navy_corner))


def _centered_text(
    painter: QPainter,
    text: str,
    y: float,
    width: float,
    color: QColor,
    pixel_size: int,
    weight: QFont.Weight = QFont.Weight.Normal,
) -> float:
    painter.setFont(_font(pixel_size, weight))
    painter.setPen(color)
    metrics = QFontMetricsF(painter.font())
    painter.drawText(
        QRectF(24, y - metrics.ascent(), width - 48, metrics.height()),
        int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter),
        text,
    )
    return metrics.height()


def _fit_logo_pixmap(logo: QPixmap, max_w: float, max_h: float) -> QPixmap:
    """Scale down only — never upscale above the source asset."""
    if logo.isNull():
        return logo
    lw = float(logo.width())
    lh = float(logo.height())
    if lw <= 0 or lh <= 0:
        return logo
    scale = min(1.0, max_w / lw, max_h / lh)
    if scale >= 0.999:
        return logo
    new_w = max(1, int(round(lw * scale)))
    new_h = max(1, int(round(lh * scale)))
    return logo.scaled(
        new_w,
        new_h,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def _draw_logo(painter: QPainter, width: float, height: float) -> tuple[float, float]:
    """Place the lockup at mid-height. Returns (bottom y, logo height)."""
    logo = _load_logo_master()
    if logo is None or logo.isNull():
        fallback_h = height * 0.10
        _centered_text(
            painter,
            f"{APP_NAME}®",
            height * 0.5,
            width,
            _NAVY,
            int(fallback_h),
            QFont.Weight.DemiBold,
        )
        return height * 0.5 + fallback_h * 0.5, fallback_h

    fitted = _fit_logo_pixmap(logo, width * 0.88, height * 0.32)
    x = (width - fitted.width()) / 2.0
    y = height * 0.5 - fitted.height() / 2.0
    painter.drawPixmap(int(round(x)), int(round(y)), fitted)
    return y + fitted.height(), float(fitted.height())


def _draw_copy(
    painter: QPainter,
    width: float,
    height: float,
    text_top: float,
) -> None:
    scale = height / BANNER_HEIGHT
    credit_size = max(14, int(17 * scale))
    loading_size = max(16, int(19 * scale))
    credit_gap = max(40.0, 52.0 * scale)
    loading_gap = max(12.0, 16.0 * scale)

    credit_y = text_top + credit_gap
    _centered_text(
        painter,
        f"© 2026  •  Developed by {APP_AUTHOR}",
        credit_y,
        width,
        _CREDIT,
        credit_size,
    )
    _centered_text(
        painter,
        "Loading…",
        credit_y + loading_gap + credit_size,
        width,
        _TEAL,
        loading_size,
        QFont.Weight.DemiBold,
    )

    rule_y = height - max(52.0, 58.0 * scale)
    painter.setPen(QPen(QColor(26, 47, 74, 22), 1.0))
    painter.drawLine(QPointF(width * 0.20, rule_y), QPointF(width * 0.80, rule_y))
    _centered_text(
        painter,
        f"{APP_NAME}® is a trademark of {APP_AUTHOR}.  {APP_LICENSE_LINE}",
        height - max(26.0, 30.0 * scale),
        width,
        _FOOTER,
        max(11, int(12 * scale)),
    )


def build_splash_pixmap(
    device_pixel_ratio: float = 1.0,
    width: int = BANNER_WIDTH,
    height: int = BANNER_HEIGHT,
) -> QPixmap:
    """Paint at physical pixel resolution; the window scales down once for display."""
    dpr = max(1.0, float(device_pixel_ratio))
    physical_w = max(1, int(round(width * dpr)))
    physical_h = max(1, int(round(height * dpr)))
    pixmap = QPixmap(physical_w, physical_h)
    pixmap.setDevicePixelRatio(1.0)
    pixmap.fill(_BG_CENTER)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    painter.scale(dpr, dpr)
    _draw_faded_background(painter, float(width), float(height))

    logo_bottom, _ = _draw_logo(painter, float(width), float(height))
    _draw_copy(painter, float(width), float(height), logo_bottom)
    painter.end()
    return pixmap


def _fit_banner_size(app: QApplication) -> QSize:
    """Keep the banner inside the usable desktop so nothing is clipped."""
    screen = app.primaryScreen()
    if screen is None:
        return QSize(BANNER_WIDTH, BANNER_HEIGHT)
    geo = screen.availableGeometry()
    margin = 48
    max_w = max(320, geo.width() - margin)
    max_h = max(200, geo.height() - margin)
    width = min(int(geo.width() * 0.72), BANNER_WIDTH, max_w)
    height = int(round(width * BANNER_HEIGHT / BANNER_WIDTH))
    if height > min(int(geo.height() * 0.70), max_h):
        height = min(int(geo.height() * 0.70), max_h)
        width = int(round(height * BANNER_WIDTH / BANNER_HEIGHT))
    width = min(width, max_w)
    height = min(height, max_h)
    return QSize(max(1, width), max(1, height))


class SplashBanner(QWidget):
    """Frameless banner; pixmap is painted once at native device resolution."""

    def __init__(self, pixmap: QPixmap, logical: QSize) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool,
        )
        self._pixmap = pixmap
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(app_icon())
        self.setFixedSize(logical)
        self.setStyleSheet(f"background: {_BG_CENTER.name()};")

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: ARG002
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), _BG_CENTER)
        if self._pixmap.size() == self.size():
            painter.drawPixmap(0, 0, self._pixmap)
        else:
            painter.drawPixmap(self.rect(), self._pixmap)

    def finish(self, widget: QWidget | None = None) -> None:
        _ = widget
        self.close()


def show_splash(app: QApplication) -> SplashBanner | None:
    if not _splash_enabled():
        return None
    logical = _fit_banner_size(app)
    screen = app.primaryScreen()
    dpr = screen.devicePixelRatio() if screen is not None else 1.0
    pixmap = build_splash_pixmap(dpr, logical.width(), logical.height())
    splash = SplashBanner(pixmap, logical)
    if screen is not None:
        geo = screen.availableGeometry()
        origin = geo.center() - QPoint(logical.width() // 2, logical.height() // 2)
        splash.move(origin)
    splash.show()
    splash.raise_()
    splash.activateWindow()
    app.processEvents()
    return splash
