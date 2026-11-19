from __future__ import annotations

import json
import shutil
import sys
import ctypes
import time
import warnings
from math import ceil, floor
from pathlib import Path

from PIL import Image
import numpy as np
from PySide6.QtCore import QEvent, QEventLoop, QPointF, QRectF, QThread, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QFont, QImage, QKeyEvent, QPainter, QPainterPath, QPainterPathStroker, QPen, QPixmap, QTransform
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .processing import process_image
from .exporting import export_image, output_path, read_dpi, read_text_image, LAYERED_FORMATS


ROOT = Path(__file__).resolve().parent.parent
UPLOADS = ROOT / "uploads"
OUTPUTS = ROOT / "outputs"
TEMPLATES = ROOT / "templates"
PREVIEW_CACHE = ROOT / ".preview_cache"
SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

DEFAULT_TEMPLATES = {
    "拓印保守去背景": {"image_type": "rubbing", "mode": "conservative", "keep_faint": True, "background_radius": 31, "contrast": 1.28, "threshold_bias": 5, "denoise_radius": 0.65, "dpi": 300, "use_output_dpi": False},
    "古籍平衡去背景": {"image_type": "book", "mode": "balanced", "keep_faint": True, "background_radius": 41, "contrast": 1.18, "threshold_bias": 3, "denoise_radius": 0.8, "dpi": 300, "use_output_dpi": False},
    "手稿保守去背景": {"image_type": "manuscript", "mode": "conservative", "keep_faint": True, "background_radius": 25, "contrast": 1.10, "threshold_bias": -2, "denoise_radius": 0.35, "dpi": 300, "use_output_dpi": False},
}


class NoWheelSpinBox(QSpinBox):
    """Keep parameter values stable while the user scrolls the form."""

    def wheelEvent(self, event) -> None:  # noqa: N802
        event.accept()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    """Keep decimal parameter values stable while the user scrolls the form."""

    def wheelEvent(self, event) -> None:  # noqa: N802
        event.accept()


def _read_dpi(path: Path, fallback: float = 300.0) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            value = read_dpi(path, fallback)
        if isinstance(value, (tuple, list)):
            value = value[0] if value else fallback
        value = float(value)
        return value if value > 0 else fallback
    except (OSError, TypeError, ValueError):
        return fallback


def _preview_on_white(image: Image.Image) -> Image.Image:
    """Use a white display canvas without adding a background to the file."""
    if "A" in image.getbands() or "transparency" in image.info:
        rgba = image.convert("RGBA")
        display = Image.new("RGB", rgba.size, "white")
        display.paste(rgba, mask=rgba.getchannel("A"))
        rgba.close()
        return display
    return image.convert("RGB")


class ImageView(QWidget):
    high_res_status = Signal(str)
    view_changed = Signal(float, QPointF)
    erase_stroke = Signal(QPointF, QPointF)
    erase_started = Signal()
    erase_finished = Signal()
    eraser_size_changed = Signal(float)

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title = title
        self.pixmap = QPixmap()
        self.source_size = (1, 1)
        self.image_dpi = 300.0
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self._panning = False
        self._erasing = False
        self.eraser_target = False
        self._ctrl_down = False
        self.eraser_size = 24.0
        self._eraser_cursor = self._make_eraser_cursor()
        self._last_mouse = QPointF(0, 0)
        self._space_down = False
        self._high_res_source: Path | None = None
        self._high_res_loader: ImageLoadWorker | None = None
        self._high_res_requested = False
        self._high_res_level = 0
        self._high_res_max_dimension = 0
        self.erase_paths: list[QPainterPath] = []
        self._cursor_position = QPointF(-1000, -1000)
        self._generation = 0
        self._loaders: list[QThread] = []
        self.setMinimumSize(280, 360)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_image(self, path: Path, high_res_source: Path | None = None) -> None:
        self._generation += 1
        self.erase_paths.clear()
        image = QImage(str(path))
        self.pixmap = QPixmap.fromImage(image) if not image.isNull() else QPixmap()
        if not self.pixmap.isNull():
            self.source_size = (self.pixmap.width(), self.pixmap.height())
        self._high_res_source = high_res_source
        self._high_res_requested = False
        self._high_res_loader = None
        self._high_res_level = 0
        self._high_res_max_dimension = max(self.source_size)
        self.reset_view()
        self.update()

    def clear_image(self) -> None:
        self._generation += 1
        self.erase_paths.clear()
        self.pixmap = QPixmap()
        self._high_res_source = None
        self._high_res_loader = None
        self.source_size = (1, 1)
        self.reset_view()
        self.update()

    def set_coordinate_size(self, width: int, height: int, dpi: float | None = None) -> None:
        self.source_size = (max(1, width), max(1, height))
        if dpi is not None and dpi > 0:
            self.image_dpi = float(dpi)
        self.update()

    def fit_scale(self) -> float:
        width = max(1e-6, self.source_size[0] / self.image_dpi)
        height = max(1e-6, self.source_size[1] / self.image_dpi)
        margin = 18
        return min(max(1, self.width() - margin * 2) / width,
                   max(1, self.height() - margin * 2) / height)

    def request_high_res(self) -> None:
        if self._high_res_source is None or self._high_res_loader is not None:
            return
        if not self._high_res_source.exists():
            return
        source_max = max(self.source_size)
        target_max = min(source_max, max(8192, int(max(self.width(), self.height()) * self.zoom * 2.0)))
        if self._high_res_requested and target_max <= self._high_res_max_dimension:
            return
        if target_max <= self._high_res_max_dimension:
            return
        self._high_res_requested = True
        self.high_res_status.emit("正在异步载入高清原图…")
        self._high_res_loader = ImageLoadWorker(self._high_res_source, target_max)
        generation = self._generation
        loader = self._high_res_loader
        self._loaders.append(loader)
        loader.loaded.connect(lambda image: self._high_res_loaded(image) if generation == self._generation else None)
        loader.failed.connect(lambda error: self._high_res_failed(error) if generation == self._generation else None)
        loader.finished.connect(lambda: self._loaders.remove(loader) if loader in self._loaders else None)
        self._high_res_loader.start()

    def _high_res_loaded(self, image: QImage) -> None:
        pixmap = QPixmap.fromImage(image)
        if pixmap.isNull():
            self._high_res_failed("高清原图无法创建显示缓存")
            return
        self.pixmap = pixmap
        self._high_res_max_dimension = max(pixmap.width(), pixmap.height())
        self.high_res_status.emit(f"高清原图已载入（{pixmap.width()}×{pixmap.height()}）")
        self.update()
        self._high_res_loader = None

    def _high_res_failed(self, message: str) -> None:
        self._high_res_loader = None
        self.high_res_status.emit(f"高清原图载入失败，继续使用预览图：{message}")

    def reset_view(self, scale: float | None = None) -> None:
        self.zoom = float(scale if scale is not None else self.fit_scale())
        self.pan = QPointF(0, 0)
        self._space_down = False
        self._erasing = False
        self._set_hand_cursor(False)
        self.update()
        self.view_changed.emit(self.zoom, QPointF(self.pan))

    def set_view_state(self, zoom: float, pan: QPointF) -> None:
        self.zoom = max(0.08, float(zoom))
        self.pan = QPointF(pan)
        self.update()

    def set_space_pan_active(self, active: bool) -> None:
        self._space_down = active
        self._set_hand_cursor(self._panning)

    def set_eraser_target(self, enabled: bool) -> None:
        self.eraser_target = enabled
        if not enabled and self._erasing:
            self._erasing = False
        self._set_hand_cursor(self._panning)

    def set_eraser_modifier_active(self, active: bool) -> None:
        self._ctrl_down = active
        self._set_hand_cursor(self._panning)
        self._cursor_position = QPointF(self.mapFromGlobal(QCursor.pos()))
        self.update()

    def adjust_eraser_size(self, delta: int) -> None:
        if not delta:
            return
        step = max(2.0, self.eraser_size * 0.12)
        self.eraser_size = max(4.0, min(256.0, self.eraser_size + (step if delta > 0 else -step)))
        self._eraser_cursor = self._make_eraser_cursor(self.eraser_size)
        self._set_hand_cursor(False)
        self.eraser_size_changed.emit(self.eraser_size)
        self.update()

    @staticmethod
    def _make_eraser_cursor(size: float = 24.0) -> QCursor:
        diameter = max(16, int(round(size)) + 8)
        pixmap = QPixmap(diameter, diameter)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        inset = 4
        diameter_inside = diameter - inset * 2
        painter.setPen(QPen(QColor("#111111"), 3))
        painter.drawEllipse(inset, inset, diameter_inside, diameter_inside)
        painter.setPen(QPen(QColor("#ffffff"), 1))
        painter.drawEllipse(inset, inset, diameter_inside, diameter_inside)
        painter.end()
        return QCursor(pixmap, diameter // 2, diameter // 2)

    def _set_hand_cursor(self, dragging: bool) -> None:
        if self._erasing or (self.eraser_target and self._ctrl_down):
            # Draw the ring in logical widget pixels. Native Windows cursors
            # can resize large bitmaps or offset them on high-DPI displays.
            self.setCursor(Qt.CursorShape.BlankCursor)
        elif dragging or self._space_down:
            self.setCursor(Qt.CursorShape.ClosedHandCursor if dragging else Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def _image_rect(self) -> QRectF:
        if self.pixmap.isNull():
            return QRectF()
        margin = 18
        # Compute the fitted geometry without rescaling the full-resolution
        # pixmap on every pan, zoom or paint event.
        width = self.source_size[0] / self.image_dpi
        height = self.source_size[1] / self.image_dpi
        size = QPointF(width * self.zoom, height * self.zoom)
        center = QPointF(self.width() / 2, self.height() / 2) + self.pan
        return QRectF(center.x() - size.x() / 2, center.y() - size.y() / 2, size.x(), size.y())

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#1c2229"))
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if self.pixmap.isNull():
            painter.setPen(QColor("#aab3bd"))
            empty_message = "暂无处理结果" if self.title.startswith("去背景") else "尚未导入图片"
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, empty_message)
            return
        rect = self._image_rect()
        painter.drawPixmap(rect, self.pixmap, QRectF(self.pixmap.rect()))
        if self.erase_paths:
            painter.save()
            painter.setClipRect(rect)
            painter.translate(rect.topLeft())
            painter.scale(rect.width() / self.source_size[0], rect.height() / self.source_size[1])
            for path in self.erase_paths:
                painter.fillPath(path, QColor("white"))
            painter.restore()
        if self.eraser_target and (self._ctrl_down or self._erasing) and self.rect().contains(self._cursor_position.toPoint()):
            radius = self.eraser_size / 2.0
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#111111"), 3))
            painter.drawEllipse(self._cursor_position, radius, radius)
            painter.setPen(QPen(QColor("white"), 1))
            painter.drawEllipse(self._cursor_position, radius, radius)
        painter.setPen(QColor("#aab3bd"))
        painter.drawText(12, self.height() - 12, f"缩放 {round(self.zoom / max(1e-6, self.fit_scale()) * 100)}% · Alt+滚轮缩放 · 空格+拖动平移")

    def wheelEvent(self, event) -> None:  # noqa: N802
        # Windows may omit Alt from the native wheel event while it is still held.
        modifiers = event.modifiers() | QApplication.keyboardModifiers()
        if sys.platform == "win32" and ctypes.windll.user32.GetAsyncKeyState(0x12) & 0x8000:
            modifiers |= Qt.KeyboardModifier.AltModifier
        if sys.platform == "win32" and ctypes.windll.user32.GetAsyncKeyState(0x11) & 0x8000:
            modifiers |= Qt.KeyboardModifier.ControlModifier
        delta = self.wheel_delta(event)
        if self.eraser_target and modifiers & Qt.KeyboardModifier.ControlModifier:
            self.adjust_eraser_size(delta)
            event.accept()
            return
        if self.pixmap.isNull():
            event.ignore()
            return
        if modifiers & Qt.KeyboardModifier.AltModifier:
            self.zoom_at(event.position(), delta)
            event.accept()
            return
        angle = event.angleDelta()
        pixel = event.pixelDelta()
        horizontal = angle.x() or pixel.x()
        vertical = angle.y() or pixel.y()
        if horizontal or vertical:
            self.pan += QPointF(float(-horizontal) * 0.8, float(vertical) * 0.8)
            self.update()
            self.view_changed.emit(self.zoom, QPointF(self.pan))
            event.accept()
            return
        event.ignore()
        event.accept()

    @staticmethod
    def wheel_delta(event) -> int:
        angle = event.angleDelta()
        delta = angle.y() or angle.x()
        if delta == 0:
            pixel = event.pixelDelta()
            delta = pixel.y() or pixel.x()
        return delta

    def zoom_at(self, cursor: QPointF, delta: int) -> None:
        if self.pixmap.isNull() or delta == 0:
            return
        if delta > 0:
            self.request_high_res()
        old_rect = self._image_rect()
        if old_rect.width() <= 0 or old_rect.height() <= 0:
            return
        ux = (cursor.x() - old_rect.left()) / old_rect.width()
        uy = (cursor.y() - old_rect.top()) / old_rect.height()
        factor = 1.15 if delta > 0 else 1 / 1.15
        self.zoom = max(0.08, self.zoom * factor)
        new_rect = self._image_rect()
        desired_top_left = QPointF(cursor.x() - ux * new_rect.width(), cursor.y() - uy * new_rect.height())
        self.pan += desired_top_left - new_rect.topLeft()
        self.update()
        self.view_changed.emit(self.zoom, QPointF(self.pan))

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self._cursor_position = event.position()
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        modifiers = event.modifiers() | QApplication.keyboardModifiers()
        if sys.platform == "win32" and ctypes.windll.user32.GetAsyncKeyState(0x11) & 0x8000:
            modifiers |= Qt.KeyboardModifier.ControlModifier
        if self.eraser_target and event.button() == Qt.MouseButton.LeftButton and modifiers & Qt.KeyboardModifier.ControlModifier:
            self._erasing = True
            self._last_mouse = event.position()
            self._set_hand_cursor(False)
            self.erase_started.emit()
            self.erase_stroke.emit(event.position(), event.position())
            event.accept()
            return
        if event.button() == Qt.MouseButton.MiddleButton or (event.button() == Qt.MouseButton.LeftButton and self._space_down):
            self._panning = True
            self._last_mouse = event.position()
            self._set_hand_cursor(True)
            event.accept()
            return
        event.ignore()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        self._cursor_position = event.position()
        if self._ctrl_down:
            self.update()
        if self._erasing:
            self.erase_stroke.emit(self._last_mouse, event.position())
            self._last_mouse = event.position()
            event.accept()
            return
        if self._panning:
            delta = event.position() - self._last_mouse
            self.pan += delta
            self._last_mouse = event.position()
            self.update()
            self.view_changed.emit(self.zoom, QPointF(self.pan))
            event.accept()
            return
        event.ignore()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._erasing and event.button() == Qt.MouseButton.LeftButton:
            self._erasing = False
            self._set_hand_cursor(False)
            self.erase_finished.emit()
            event.accept()
            return
        if self._panning and event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._panning = False
            self._set_hand_cursor(False)
            event.accept()
            return
        event.ignore()

    def paint_eraser_preview(self, start: QPointF, end: QPointF) -> None:
        if self.pixmap.isNull():
            return
        rect = self._image_rect()
        if rect.width() <= 0 or rect.height() <= 0:
            return
        if not self.erase_paths:
            path = QPainterPath()
            path.setFillRule(Qt.FillRule.WindingFill)
            self.erase_paths.append(path)
        centerline = QPainterPath(start)
        centerline.lineTo(end)
        stroker = QPainterPathStroker()
        stroker.setWidth(self.eraser_size)
        stroker.setCapStyle(Qt.PenCapStyle.RoundCap)
        shape = stroker.createStroke(centerline)
        if start == end:
            shape.addEllipse(start, self.eraser_size / 2, self.eraser_size / 2)
        transform = QTransform(self.source_size[0] / rect.width(), 0, 0,
                               self.source_size[1] / rect.height(),
                               -rect.left() * self.source_size[0] / rect.width(),
                               -rect.top() * self.source_size[1] / rect.height())
        self.erase_paths[-1].addPath(transform.map(shape))
        self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._cursor_position = QPointF(-1000, -1000)
        self.update()
        super().leaveEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_down = True
            self._set_hand_cursor(False)
            event.accept()
            return
        event.ignore()

    def keyReleaseEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_down = False
            self._set_hand_cursor(False)
            event.accept()
            return
        event.ignore()


class ImageLoadWorker(QThread):
    loaded = Signal(QImage)
    failed = Signal(str)

    def __init__(self, source: Path, max_dimension: int = 8192) -> None:
        super().__init__()
        self.source = source
        self.max_dimension = max(1024, max_dimension)

    def run(self) -> None:
        try:
            # Decode a bounded RGB level using Pillow, preserving ink and paper
            # colors while bypassing Qt image-file allocation limits.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", Image.DecompressionBombWarning)
                opened = read_text_image(self.source) if self.source.suffix.lower() == ".psd" else Image.open(self.source)
                source_max = max(opened.size)
                reduce_factor = max(1, ceil(source_max / self.max_dimension))
                level = opened.reduce(reduce_factor) if reduce_factor > 1 else opened.copy()
                display = _preview_on_white(level)
                width, height = display.size
                raw = display.tobytes()
                image = QImage(raw, width, height, width * 3, QImage.Format.Format_RGB888).copy()
                level.close()
                display.close()
                opened.close()
            if image.isNull():
                raise RuntimeError("高清原图生成空图像")
            self.loaded.emit(image)
        except Exception as exc:  # pragma: no cover - surfaced in the UI
            self.failed.emit(str(exc))


class ProcessWorker(QThread):
    completed = Signal(dict)
    failed = Signal(str)
    progress = Signal(int, str)

    def __init__(self, source: Path, output_dir: Path, image_type: str, mode: str, keep_faint: bool, params: dict) -> None:
        super().__init__()
        self.source = source
        self.output_dir = output_dir
        self.image_type = image_type
        self.mode = mode
        self.keep_faint = keep_faint
        self.params = params

    def run(self) -> None:
        started = time.perf_counter()
        try:
            job_id = self.source.stem
            result = process_image(
                self.source,
                self.output_dir,
                self.image_type,
                self.mode,
                self.keep_faint,
                params=self.params,
                preview_dir=PREVIEW_CACHE / "processed" / self.source.stem,
                progress_callback=lambda value, message: self.progress.emit(value, message),
            )
            result["job_id"] = job_id
            result["source"] = str(self.source)
            result["output_dir"] = str(self.output_dir)
            result["processing_seconds"] = round(time.perf_counter() - started, 2)
            self.completed.emit(result)
        except Exception as exc:  # pragma: no cover - surfaced in the UI
            self.failed.emit(str(exc))


class EditImageLoadWorker(QThread):
    loaded = Signal(object)
    failed = Signal(str)

    def __init__(self, source: Path) -> None:
        super().__init__()
        self.source = source

    def run(self) -> None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", Image.DecompressionBombWarning)
                self.loaded.emit(read_text_image(self.source))
        except Exception as exc:
            self.failed.emit(str(exc))


class ExportWorker(QThread):
    completed = Signal(str)
    failed = Signal(str)
    progress = Signal(int, str)

    def __init__(self, source: Path | Image.Image, destination: Path, params: dict, dpi: tuple[float, float]) -> None:
        super().__init__()
        self.source, self.destination, self.params, self.dpi = source, destination, params, dpi

    def run(self) -> None:
        image = None
        try:
            if isinstance(self.source, Image.Image):
                image = self.source
            else:
                self.progress.emit(5, "读取文字图层")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", Image.DecompressionBombWarning)
                    image = read_text_image(self.source)
            export_image(image, self.destination, self.params.get("output_format", "tif"),
                         bool(self.params.get("white_background", False)), self.dpi,
                         progress_callback=self.progress.emit)
            self.completed.emit(str(self.destination))
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            if image is not None and image is not self.source:
                image.close()


class ImportWorker(QThread):
    completed = Signal(str, str, str, int, int)
    failed = Signal(str)
    progress = Signal(int, str)

    def __init__(self, source: Path, destination: Path) -> None:
        super().__init__()
        self.source = source
        self.destination = destination

    def run(self) -> None:
        try:
            self.progress.emit(10, "读取图片")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", Image.DecompressionBombWarning)
                with (read_text_image(self.source) if self.source.suffix.lower() == ".psd" else Image.open(self.source)) as opened:
                    width, height = opened.size
                    self.progress.emit(25, "保存原始文件")
                    shutil.copy2(self.source, self.destination)
                    self.progress.emit(45, "生成预览")
                    reduce_factor = max(1, ceil(max(width, height) / 2400))
                    preview = opened.reduce(reduce_factor) if reduce_factor > 1 else opened.copy()
                    preview = _preview_on_white(preview)
                    self.progress.emit(75, "准备预览")
                    self.destination.parent.mkdir(exist_ok=True)
                    preview_path = self.destination.with_name(f"{self.destination.stem}_preview.jpg")
                    preview.save(preview_path, format="JPEG", quality=88, optimize=True)
                    preview.close()
            self.progress.emit(100, "导入完成")
            self.completed.emit(str(self.destination), str(preview_path), self.source.name, width, height)
        except Exception as exc:  # pragma: no cover - surfaced in the UI
            self.failed.emit(str(exc))


class PreviewWorker(QThread):
    completed = Signal(str, int, int)
    failed = Signal(str)
    progress = Signal(int, str)

    def __init__(self, source: Path) -> None:
        super().__init__()
        self.source = source

    def run(self) -> None:
        try:
            PREVIEW_CACHE.mkdir(parents=True, exist_ok=True)
            cache_name = f"{self.source.stem}_{self.source.stat().st_size}_{self.source.stat().st_mtime_ns}_alpha_preview.jpg"
            preview_path = PREVIEW_CACHE / cache_name
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", Image.DecompressionBombWarning)
                with (read_text_image(self.source) if self.source.suffix.lower() == ".psd" else Image.open(self.source)) as opened:
                    width, height = opened.size
                    if not preview_path.exists():
                        self.progress.emit(30, "生成当前图片预览")
                        factor = max(1, ceil(max(width, height) / 2400))
                        preview = opened.reduce(factor) if factor > 1 else opened.copy()
                        preview = _preview_on_white(preview)
                        preview.save(preview_path, format="JPEG", quality=92, optimize=True)
                        preview.close()
            self.completed.emit(str(preview_path), width, height)
        except Exception as exc:  # pragma: no cover - surfaced in the UI
            self.failed.emit(str(exc))


class BatchWorker(QThread):
    item_progress = Signal(int, str)
    overall_progress = Signal(int, int, str)
    completed = Signal(dict)
    failed = Signal(str)

    def __init__(self, files: list[Path], source_root: Path, output_root: Path, image_type: str, mode: str, keep_faint: bool, params: dict) -> None:
        super().__init__()
        self.files = files
        self.source_root = source_root
        self.output_root = output_root
        self.image_type = image_type
        self.mode = mode
        self.keep_faint = keep_faint
        self.params = params
        self.cancel_requested = False

    def request_cancel(self) -> None:
        self.cancel_requested = True

    def run(self) -> None:
        succeeded = 0
        skipped = 0
        failed: list[str] = []
        output_names: set[str] = set()
        total = len(self.files)
        batch_started = time.perf_counter()
        try:
            for index, source in enumerate(self.files):
                if self.cancel_requested:
                    break
                item_started = time.perf_counter()
                self.item_progress.emit(0, f"正在处理 {source.name}")
                try:
                    output_file = output_path(self.output_root, source.stem, self.params.get("output_format", "tif"))
                    output_name = output_file.name.casefold()
                    if output_file.is_file() or output_name in output_names:
                        skipped += 1
                        elapsed = time.perf_counter() - item_started
                        self.item_progress.emit(100, f"{source.name}：输出文件已存在或同名，已跳过（耗时 {elapsed:.2f} 秒）")
                    else:
                        output_names.add(output_name)
                        process_image(
                            source,
                            self.output_root,
                            self.image_type,
                            self.mode,
                            self.keep_faint,
                            params=self.params,
                            preview_dir=PREVIEW_CACHE / "processed" / source.stem,
                            progress_callback=lambda value, message, name=source.name: self.item_progress.emit(value, f"{name}：{message}"),
                            # Batch results are previewed on demand when opened.
                            generate_preview=False,
                            output_format=self.params.get("output_format", "tif"),
                            white_background=bool(self.params.get("white_background", False)),
                            overwrite_output=False,
                        )
                        succeeded += 1
                        elapsed = time.perf_counter() - item_started
                        self.item_progress.emit(100, f"{source.name}：处理完成，耗时 {elapsed:.2f} 秒")
                except FileExistsError:
                    skipped += 1
                    self.item_progress.emit(100, f"{source.name}：目标文件已存在，已跳过")
                except Exception as exc:
                    elapsed = time.perf_counter() - item_started
                    failed.append(f"{source.name}: {exc}")
                    self.item_progress.emit(100, f"{source.name}：处理失败，耗时 {elapsed:.2f} 秒")
                self.overall_progress.emit(index + 1, total, source.name)
            self.completed.emit({
                "total": total,
                "succeeded": succeeded,
                "skipped": skipped,
                "failed": failed,
                "cancelled": self.cancel_requested,
                "elapsed_seconds": round(time.perf_counter() - batch_started, 2),
            })
        except Exception as exc:  # pragma: no cover - surfaced in the UI
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ImageLab · 文献图片去背景工作台")
        self.resize(1360, 820)
        self.source_root: Path | None = None
        self.output_root: Path | None = None
        self.files: list[Path] = []
        self.current_index = -1
        self.source: Path | None = None
        self.preview_source: Path | None = None
        self.result: dict | None = None
        self.current_result_dir: Path | None = None
        self.current_saved = False
        self.worker: ProcessWorker | None = None
        self.preview_worker: PreviewWorker | None = None
        self.batch_worker: BatchWorker | None = None
        self._alt_zoom_held = False
        self._syncing_views = False
        self._templates: dict[str, dict] = {}
        self.export_worker: ExportWorker | None = None
        self._edit_loaders: list[EditImageLoadWorker] = []
        self._edit_generation = 0
        self._edited_image: Image.Image | None = None
        self._edited_result_path: Path | None = None
        self._edited_preview_path: Path | None = None
        self._eraser_undo: list[list[tuple[tuple[int, int, int, int], Image.Image]]] = []
        self._edited_dirty = False
        self._build_ui()
        self._load_templates()
        QApplication.instance().installEventFilter(self)

    def _build_ui(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget { background:#171b20; color:#e6e9ed; font-family:'Microsoft YaHei'; }
            QFrame#panel { background:#222831; border:1px solid #39424d; border-radius:7px; }
            QLabel#title { font-size:20px; font-weight:600; color:#f3f5f7; }
            QLabel#muted { color:#aab3bd; font-size:12px; }
            QPushButton { min-height:34px; padding:0 13px; border:1px solid #46515d; border-radius:6px; background:#2b333d; color:#e6e9ed; }
            QPushButton:hover { border-color:#d07855; color:#f0b59d; background:#343e49; }
            QPushButton:disabled { color:#6f7a86; border-color:#303943; background:#20262d; }
            QPushButton#primary { background:#a94f36; color:white; border-color:#c46648; font-weight:600; }
            QPushButton#primary:hover { background:#bd6044; }
            QPushButton#save { background:#287b68; color:white; border-color:#3d9a82; font-weight:600; }
            QPushButton#save:hover { background:#35947b; }
            QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox { min-height:34px; border:1px solid #46515d; border-radius:5px; padding:0 8px; background:#1c2229; color:#e6e9ed; }
            QComboBox QAbstractItemView { background:#252d36; color:#e6e9ed; selection-background-color:#a94f36; }
            QCheckBox { spacing:8px; color:#dce1e6; }
            QScrollArea { border:0; background:#222831; }
            QProgressBar { height:7px; border:0; border-radius:3px; background:#303943; color:#e6e9ed; }
            QProgressBar::chunk { border-radius:3px; background:#3d896f; }
        """)
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(22, 18, 22, 18)
        outer.setSpacing(14)

        header = QHBoxLayout()
        title = QLabel("ImageLab · 文献图片去背景")
        title.setObjectName("title")
        header.addWidget(title)
        subtitle = QLabel("保守去背景 · 笔画优先 · 可回溯")
        subtitle.setObjectName("muted")
        header.addWidget(subtitle)
        header.addStretch()
        outer.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_preview_panel())
        self.settings_panel = self._build_right_panel()
        splitter.addWidget(self.settings_panel)
        splitter.setSizes([960, 400])
        outer.addWidget(splitter, 1)
        self.setCentralWidget(root)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if isinstance(event, QKeyEvent) and event.key() == Qt.Key.Key_Alt and not event.isAutoRepeat():
            self._alt_zoom_held = event.type() == QEvent.Type.KeyPress
            return False
        if isinstance(event, QKeyEvent) and event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            active = event.type() == QEvent.Type.KeyPress
            self.original_view.set_space_pan_active(active)
            self.cleaned_view.set_space_pan_active(active)
            return False
        if isinstance(event, QKeyEvent) and event.key() == Qt.Key.Key_Z and not event.isAutoRepeat():
            modifiers = event.modifiers() | QApplication.keyboardModifiers()
            if sys.platform == "win32" and ctypes.windll.user32.GetAsyncKeyState(0x11) & 0x8000:
                modifiers |= Qt.KeyboardModifier.ControlModifier
            if event.type() == QEvent.Type.KeyPress and modifiers & Qt.KeyboardModifier.ControlModifier and self._eraser_undo:
                self.undo_erase()
                return True
        if isinstance(event, QKeyEvent) and event.key() in (Qt.Key.Key_Control, Qt.Key.Key_Meta) and not event.isAutoRepeat():
            active = event.type() == QEvent.Type.KeyPress
            self.cleaned_view.set_eraser_modifier_active(active)
            return False
        if event.type() == QEvent.Type.Wheel and hasattr(event, "globalPosition"):
            alt_down = bool(QApplication.queryKeyboardModifiers() & Qt.KeyboardModifier.AltModifier)
            alt_down = alt_down or self._alt_zoom_held
            if sys.platform == "win32":
                alt_down = alt_down or bool(ctypes.windll.user32.GetAsyncKeyState(0x12) & 0x8000)
            if alt_down:
                global_pos = event.globalPosition().toPoint()
                for view in (self.original_view, self.cleaned_view):
                    local_pos = view.mapFromGlobal(global_pos)
                    if view.rect().contains(local_pos) and not view.pixmap.isNull():
                        delta = ImageView.wheel_delta(event)
                        view.zoom_at(QPointF(local_pos), delta)
                        if delta:
                            return True
        return super().eventFilter(watched, event)

    def _build_left_panel(self) -> QWidget:
        panel = QFrame(objectName="panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 16, 14, 16)
        label = QLabel("当前资料")
        label.setObjectName("muted")
        layout.addWidget(label)
        self.file_label = QLabel("尚未导入图片")
        self.file_label.setWordWrap(True)
        layout.addWidget(self.file_label)
        layout.addSpacing(20)
        label = QLabel("工作步骤")
        label.setObjectName("muted")
        layout.addWidget(label)
        self.step_process = QPushButton("2 处理")
        self.step_process.setEnabled(False)
        self.step_review = QPushButton("3 复核")
        self.step_review.setEnabled(False)
        self.step_export = QPushButton("4 导出")
        self.step_export.setEnabled(False)
        layout.addWidget(self.step_process)
        layout.addWidget(self.step_review)
        layout.addWidget(self.step_export)
        layout.addStretch()
        note = QLabel("原图只读\n处理结果按版本保存")
        note.setObjectName("muted")
        layout.addWidget(note)
        return panel

    def _build_preview_panel(self) -> QWidget:
        panel = QFrame(objectName="panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        head = QHBoxLayout()
        head.setSpacing(2)
        self.preview_title = QLabel("原图与去背景结果对比")
        self.preview_title.setStyleSheet("font-size:15px;font-weight:600")
        head.addWidget(self.preview_title)
        self.reset_view_button = QPushButton("重置视图")
        self.reset_view_button.clicked.connect(self.reset_views)
        self.reset_view_button.setFixedWidth(78)
        head.addWidget(self.reset_view_button)
        head.addStretch()
        self.previous_button = QPushButton("上一张")
        self.previous_button.clicked.connect(lambda: self.change_file(-1))
        self.previous_button.setEnabled(False)
        self.previous_button.setFixedWidth(76)
        head.addWidget(self.previous_button)
        self.next_button = QPushButton("下一张")
        self.next_button.clicked.connect(lambda: self.change_file(1))
        self.next_button.setEnabled(False)
        self.next_button.setFixedWidth(76)
        head.addWidget(self.next_button)
        head.setSpacing(8)
        self.page_label = QLabel("0 / 0")
        self.page_label.setObjectName("muted")
        self.page_label.setMinimumWidth(82)
        self.page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        head.addWidget(self.page_label)
        jump_group = QWidget()
        jump_group.setFixedWidth(132)
        jump_group.setFixedHeight(36)
        jump_layout = QHBoxLayout(jump_group)
        jump_layout.setContentsMargins(0, 0, 0, 0)
        jump_layout.setSpacing(4)
        self.jump_label = QLabel("跳至")
        self.jump_label.setFixedWidth(38)
        self.jump_label.setFixedHeight(36)
        self.jump_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        jump_layout.addWidget(self.jump_label)
        self.jump_spin = NoWheelSpinBox()
        self.jump_spin.setRange(1, 1)
        self.jump_spin.setEnabled(False)
        self.jump_spin.setFixedWidth(86)
        self.jump_spin.setFixedHeight(36)
        self.jump_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.jump_spin.setToolTip("输入图片序号后按 Enter 跳转")
        self.jump_spin.editingFinished.connect(self.jump_to_file)
        jump_layout.addWidget(self.jump_spin)
        jump_layout.activate()
        head.addWidget(jump_group)
        layout.addLayout(head)
        self.preview_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.original_view = ImageView("原图")
        self.cleaned_view = ImageView("去背景结果（Ctrl+左键擦除）")
        self.cleaned_view.set_eraser_target(True)
        self.original_view.high_res_status.connect(self.status_message)
        self.cleaned_view.high_res_status.connect(self.status_message)
        self.cleaned_view.erase_stroke.connect(self.erase_processed_stroke)
        self.cleaned_view.erase_started.connect(self.begin_erase)
        self.cleaned_view.erase_finished.connect(self.finish_erase)
        self.cleaned_view.eraser_size_changed.connect(self.eraser_size_changed)
        self.original_view.view_changed.connect(lambda zoom, pan: self._sync_view(self.cleaned_view, zoom, pan))
        self.cleaned_view.view_changed.connect(lambda zoom, pan: self._sync_view(self.original_view, zoom, pan))
        self.preview_splitter.addWidget(self._view_panel("原图", self.original_view))
        self.preview_splitter.addWidget(self._view_panel("去背景结果（透明区域以白色预览）", self.cleaned_view))
        layout.addWidget(self.preview_splitter, 1)
        self.status = QLabel("请导入图片开始处理")
        self.status.setObjectName("muted")
        layout.addWidget(self.status)
        return panel

    @staticmethod
    def _view_panel(title: str, view: ImageView) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        label = QLabel(title)
        label.setObjectName("previewLabel")
        label.setStyleSheet("font-size:13px;font-weight:600;color:#e6e9ed;padding-left:3px;")
        layout.addWidget(label)
        layout.addWidget(view, 1)
        return panel

    def _sync_view(self, target: ImageView, zoom: float, pan: QPointF) -> None:
        if self._syncing_views:
            return
        self._syncing_views = True
        try:
            target.set_view_state(zoom, pan)
        finally:
            self._syncing_views = False

    def _build_right_panel(self) -> QWidget:
        panel = QFrame(objectName="panel")
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(12, 12, 12, 12)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.addWidget(QLabel("原图路径"))
        self.source_dir_edit = QLineEdit()
        self.source_dir_edit.setReadOnly(True)
        source_row = QHBoxLayout()
        source_row.addWidget(self.source_dir_edit, 1)
        source_btn = QPushButton("选择")
        source_btn.clicked.connect(self.select_source_directory)
        source_row.addWidget(source_btn)
        layout.addLayout(source_row)
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setReadOnly(True)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_dir_edit, 1)
        output_btn = QPushButton("选择")
        output_btn.clicked.connect(self.select_output_directory)
        output_row.addWidget(output_btn)
        layout.addWidget(QLabel("导出路径"))
        layout.addLayout(output_row)
        layout.addWidget(QLabel("处理模板"))
        self.template_combo = QComboBox()
        self.template_combo.currentTextChanged.connect(self.apply_selected_template)
        layout.addWidget(self.template_combo)
        template_buttons = QHBoxLayout()
        self.save_template_button = QPushButton("保存模板")
        self.save_template_button.clicked.connect(self.save_template)
        template_buttons.addWidget(self.save_template_button)
        self.save_as_template_button = QPushButton("另存为")
        self.save_as_template_button.clicked.connect(self.save_template_as)
        template_buttons.addWidget(self.save_as_template_button)
        layout.addLayout(template_buttons)

        form = QFormLayout()
        self.type_combo = QComboBox()
        self.type_combo.addItem("古代碑文拓印", "rubbing")
        self.type_combo.addItem("古籍扫描", "book")
        self.type_combo.addItem("古代手稿扫描", "manuscript")
        self.type_combo.addItem("其他文献图像", "other")
        form.addRow("资料类型", self.type_combo)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("保守去背景", "conservative")
        self.mode_combo.addItem("平衡去背景", "balanced")
        self.mode_combo.addItem("强力去背景", "strong")
        form.addRow("去背景策略", self.mode_combo)
        self.background_radius = NoWheelSpinBox()
        self.background_radius.setRange(3, 301)
        self.background_radius.setSingleStep(2)
        form.addRow("背景判定半径", self.background_radius)
        self.threshold_bias = NoWheelSpinBox()
        self.threshold_bias.setRange(-80, 80)
        form.addRow("文字判定偏移", self.threshold_bias)
        self.contrast = NoWheelDoubleSpinBox()
        self.contrast.setRange(0.5, 2.0)
        self.contrast.setSingleStep(0.05)
        self.contrast.setDecimals(2)
        form.addRow("分析对比度", self.contrast)
        self.denoise_radius = NoWheelDoubleSpinBox()
        self.denoise_radius.setRange(0.0, 4.0)
        self.denoise_radius.setSingleStep(0.05)
        self.denoise_radius.setDecimals(2)
        form.addRow("分析降噪半径", self.denoise_radius)
        self.output_dpi = NoWheelSpinBox()
        self.output_dpi.setRange(72, 2400)
        self.output_dpi.setSingleStep(1)
        self.output_dpi.setSuffix(" dpi")
        self.output_dpi.setToolTip("勾选统一输出分辨率后，按此 DPI 重新采样并保存图片")
        form.addRow("输出图片分辨率", self.output_dpi)
        self.use_output_dpi = QCheckBox("启用统一输出分辨率")
        self.use_output_dpi.setChecked(False)
        self.use_output_dpi.setToolTip("未勾选时保留原图像素尺寸和原图 DPI；勾选后按设定 DPI 重新采样")
        self.use_output_dpi.toggled.connect(self.output_dpi.setEnabled)
        form.addRow("分辨率规则", self.use_output_dpi)
        self.output_dpi.setEnabled(False)
        self.output_format = QComboBox()
        for label, value in (("TIF", "tif"), ("PNG", "png"), ("JPG", "jpg"), ("PSD", "psd")):
            self.output_format.addItem(label, value)
        form.addRow("输出格式", self.output_format)
        self.white_background = QCheckBox("添加白底图层（最底层）")
        form.addRow("图层设置", self.white_background)
        self.format_note = QLabel()
        self.format_note.setWordWrap(True)
        self.format_note.setObjectName("muted")
        form.addRow(self.format_note)
        self.output_format.currentIndexChanged.connect(self.output_format_changed)
        self.output_format_changed()
        layout.addLayout(form)
        self.keep_faint = QCheckBox("保留淡色笔画")
        self.keep_faint.setChecked(True)
        layout.addWidget(self.keep_faint)
        note = QLabel("翻页仅查看原图和已有结果，点击“处理当前图片”才去背景。只保留原有墨色；Ctrl + 左键擦除，Ctrl + 滚轮调整大小，Ctrl + Z 撤销。未统一分辨率时保留原图尺寸与 DPI。")
        note.setWordWrap(True)
        note.setObjectName("muted")
        layout.addWidget(note)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        self.process_button = QPushButton("处理当前图片")
        self.process_button.setObjectName("primary")
        self.process_button.setFixedWidth(112)
        self.process_button.setEnabled(False)
        self.process_button.clicked.connect(self.process)
        action_row.addWidget(self.process_button)
        self.export_button = QPushButton("保存当前图片")
        self.export_button.setObjectName("save")
        self.export_button.setFixedWidth(124)
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.save_current)
        action_row.addWidget(self.export_button)
        action_row.addStretch()
        layout.addLayout(action_row)

        batch_row = QHBoxLayout()
        batch_row.setSpacing(8)
        self.batch_button = QPushButton("自动批量处理")
        self.batch_button.setFixedWidth(124)
        self.batch_button.setEnabled(False)
        self.batch_button.clicked.connect(self.process_batch)
        batch_row.addWidget(self.batch_button)
        self.cancel_batch_button = QPushButton("取消批量处理")
        self.cancel_batch_button.setFixedWidth(124)
        self.cancel_batch_button.setEnabled(False)
        self.cancel_batch_button.clicked.connect(self.cancel_batch)
        batch_row.addWidget(self.cancel_batch_button)
        batch_row.addStretch()
        layout.addLayout(batch_row)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setRange(0, 100)
        self.batch_progress_bar.setValue(0)
        layout.addWidget(self.batch_progress_bar)
        self.progress_label = QLabel("等待任务")
        self.progress_label.setObjectName("muted")
        layout.addWidget(self.progress_label)

        self.file_label = QLabel("尚未选择图片")
        self.file_label.setWordWrap(True)
        layout.addWidget(self.file_label)

        layout.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll)
        return panel

    def _load_templates(self) -> None:
        TEMPLATES.mkdir(parents=True, exist_ok=True)
        self._templates = {name: dict(values) for name, values in DEFAULT_TEMPLATES.items()}
        for path in sorted(TEMPLATES.glob("*.json")):
            try:
                values = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(values, dict):
                    self._templates[path.stem] = values
            except (OSError, ValueError):
                continue
        self.template_combo.blockSignals(True)
        self.template_combo.clear()
        self.template_combo.addItems(list(self._templates))
        self.template_combo.blockSignals(False)
        if self.template_combo.count():
            self.apply_selected_template(self.template_combo.currentText())

    def _current_params(self) -> dict:
        return {
            "image_type": self.type_combo.currentData(),
            "mode": self.mode_combo.currentData(),
            "keep_faint": self.keep_faint.isChecked(),
            "background_radius": self.background_radius.value(),
            "contrast": self.contrast.value(),
            "threshold_bias": self.threshold_bias.value(),
            "denoise_radius": self.denoise_radius.value(),
            "dpi": self.output_dpi.value(),
            "use_output_dpi": self.use_output_dpi.isChecked(),
            "output_format": self.output_format.currentData(),
            "white_background": self.white_background.isChecked(),
        }

    def apply_selected_template(self, name: str) -> None:
        values = self._templates.get(name)
        if not values:
            return
        self.type_combo.setCurrentIndex(max(0, self.type_combo.findData(values.get("image_type", "other"))))
        self.mode_combo.setCurrentIndex(max(0, self.mode_combo.findData(values.get("mode", "balanced"))))
        self.keep_faint.setChecked(bool(values.get("keep_faint", True)))
        self.background_radius.setValue(int(values.get("background_radius", 31)))
        self.contrast.setValue(float(values.get("contrast", 1.15)))
        self.threshold_bias.setValue(int(values.get("threshold_bias", 0)))
        self.denoise_radius.setValue(float(values.get("denoise_radius", 0.55)))
        self.output_dpi.setValue(int(values.get("dpi", 300)))
        # Old user templates predate the explicit switch. They must preserve
        # source resolution until the user opts into unified resampling.
        self.use_output_dpi.setChecked(bool(values.get("use_output_dpi", False)))
        self.output_dpi.setEnabled(self.use_output_dpi.isChecked())
        self.output_format.setCurrentIndex(max(0, self.output_format.findData(values.get("output_format", "tif"))))
        self.white_background.setChecked(bool(values.get("white_background", False)))

    def output_format_changed(self) -> None:
        fmt = self.output_format.currentData()
        self.white_background.setEnabled(fmt in LAYERED_FORMATS)
        self.format_note.setText({
            "jpg": "JPG 不支持透明或图层，将合成为白底图片（有损压缩）。",
            "png": "PNG 无损保存透明文字，不支持独立白底图层。",
            "tif": "TIF 无损保存；勾选白底时保存可编辑文字层和最底部白底层。",
            "psd": "PSD 保留可编辑文字层；可选最底部纯白图层。",
        }[fmt])

    def save_template(self) -> None:
        name = self.template_combo.currentText().strip()
        if not name or name in DEFAULT_TEMPLATES:
            self.save_template_as()
            return
        self._templates[name] = self._current_params()
        TEMPLATES.mkdir(parents=True, exist_ok=True)
        (TEMPLATES / f"{name}.json").write_text(json.dumps(self._templates[name], ensure_ascii=False, indent=2), encoding="utf-8")
        self.status.setText(f"模板已保存：{name}")

    def save_template_as(self) -> None:
        name, accepted = QInputDialog.getText(self, "模板另存为", "模板名称")
        name = name.strip()
        if not accepted or not name:
            return
        self._templates[name] = self._current_params()
        TEMPLATES.mkdir(parents=True, exist_ok=True)
        (TEMPLATES / f"{name}.json").write_text(json.dumps(self._templates[name], ensure_ascii=False, indent=2), encoding="utf-8")
        self.template_combo.addItem(name)
        self.template_combo.setCurrentText(name)
        self.status.setText(f"已另存模板：{name}")

    def select_source_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "选择原图目录", str(self.source_root or ROOT))
        if not selected:
            return
        self.source_root = Path(selected)
        self.source_dir_edit.setText(str(self.source_root))
        if self.output_root is None:
            self.output_root = self.source_root / "processed"
            self.output_dir_edit.setText(str(self.output_root))
        self.refresh_file_list()

    def select_output_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "选择处理结果目录", str(self.output_root or ROOT))
        if selected:
            self.output_root = Path(selected)
            self.output_dir_edit.setText(str(self.output_root))
            self.refresh_file_list()
            self.batch_button.setEnabled(bool(self.files))

    def refresh_file_list(self) -> None:
        if not self.source_root:
            return
        # Only process images directly inside the selected source directory.
        # Files in child directories are intentionally excluded.
        iterator = self.source_root.glob("*")
        self.files = sorted(
            (
                path
                for path in iterator
                if path.is_file()
                and path.suffix.lower() in SUPPORTED_SUFFIXES
                and not (self.output_root and path.is_relative_to(self.output_root))
            ),
            key=lambda path: str(path).lower(),
        )
        self.current_index = 0 if self.files else -1
        self._update_page_controls()
        if self.current_index >= 0:
            self.load_current_file()
        else:
            self.source = None
            self.file_label.setText("目录中没有支持的图片")
            self.process_button.setEnabled(False)

    def _update_page_controls(self) -> None:
        total = len(self.files)
        current = self.current_index + 1 if self.current_index >= 0 else 0
        self.page_label.setText(f"{current} / {total}")
        self.jump_spin.blockSignals(True)
        self.jump_spin.setRange(1, max(1, total))
        self.jump_spin.setValue(max(1, current))
        self.jump_spin.setEnabled(total > 0)
        self.jump_spin.blockSignals(False)
        self.previous_button.setEnabled(self.current_index > 0)
        self.next_button.setEnabled(0 <= self.current_index < total - 1)
        self.batch_button.setEnabled(bool(self.files and self.output_root))

    def change_file(self, offset: int) -> None:
        if not self.files or self.preview_worker and self.preview_worker.isRunning() or self.worker and self.worker.isRunning() or self.batch_worker and self.batch_worker.isRunning() or self.export_worker and self.export_worker.isRunning():
            return
        target = self.current_index + offset
        if 0 <= target < len(self.files):
            self.current_index = target
            self._update_page_controls()
            self.load_current_file()

    def jump_to_file(self) -> None:
        if not self.files or self.preview_worker and self.preview_worker.isRunning() or self.worker and self.worker.isRunning() or self.batch_worker and self.batch_worker.isRunning() or self.export_worker and self.export_worker.isRunning():
            return
        target = self.jump_spin.value() - 1
        if target == self.current_index:
            return
        if 0 <= target < len(self.files):
            self.current_index = target
            self._update_page_controls()
            self.load_current_file()

    def load_current_file(self) -> None:
        if self.current_index < 0 or self.current_index >= len(self.files):
            return
        self.source = self.files[self.current_index]
        self.result = None
        self.current_result_dir = None
        self.current_saved = False
        self._edited_image = None
        self._edited_result_path = None
        self._edited_preview_path = None
        self._eraser_undo.clear()
        self._edited_dirty = False
        self._edit_generation += 1
        self.cleaned_view.set_eraser_target(False)
        self.file_label.setText(f"当前图片：{self.source.name}\n{self.source}")
        self.preview_source = None
        self.original_view.clear_image()
        self.cleaned_view.clear_image()
        self.export_button.setEnabled(False)
        self.process_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("正在准备预览…")
        self.status.setText("正在读取图片预览，请稍候…")
        self.preview_worker = PreviewWorker(self.source)
        self.preview_worker.progress.connect(lambda value, message: self._set_progress(value, message))
        self.preview_worker.completed.connect(self.preview_ready)
        self.preview_worker.failed.connect(self.preview_failed)
        self.preview_worker.start()

    def preview_ready(self, preview: str, width: int, height: int) -> None:
        self.preview_source = Path(preview)
        # Clear the result pane before fitting the original image. Clearing a
        # view emits its own (1x1 placeholder) view state; doing it afterwards
        # would overwrite the original pane with an enormous zoom value.
        self.cleaned_view.clear_image()
        self.original_view.set_image(self.preview_source, high_res_source=self.source)
        self.original_view.set_coordinate_size(width, height, _read_dpi(self.source))
        self.reset_views()
        self.process_button.setEnabled(True)
        self.batch_button.setEnabled(bool(self.files and self.output_root))
        self.progress_bar.setValue(100)
        self.progress_label.setText(f"预览就绪：{width}×{height}")
        existing = None
        if self.output_root:
            extensions = dict.fromkeys([self.output_format.currentData(), "tif", "png", "psd", "jpg", "tiff", "jpeg"])
            existing = next((path for ext in extensions
                             if (path := self.output_root / f"{self.source.stem}.{ext}").is_file()
                             and path.resolve() != self.source.resolve()), None)
        if existing and existing.is_file() and existing.resolve() != self.source.resolve():
            self._existing_result_source = existing
            self.process_button.setEnabled(False)
            self.status.setText("发现已存在的处理结果，正在载入已有结果预览…")
            self.preview_worker = PreviewWorker(existing)
            self.preview_worker.progress.connect(lambda value, message: self._set_progress(value, message))
            self.preview_worker.completed.connect(self.existing_result_preview_ready)
            self.preview_worker.failed.connect(self.preview_failed)
            self.preview_worker.start()
            return
        self.progress_label.setText("原图预览就绪，尚无处理结果")
        self.status.setText("可以处理当前图片，或使用自动批量处理")

    def existing_result_preview_ready(self, preview: str, width: int, height: int) -> None:
        result_path = getattr(self, "_existing_result_source", None)
        if not result_path:
            return
        dpi_x, dpi_y = read_dpi(result_path)
        result = {
            "enhanced": result_path.name,
            "layered_tif": result_path.name,
            "transparent": result_path.name,
            "preview_path": preview,
            "enhanced_preview": preview,
            "output_dir": str(result_path.parent),
            "width": width,
            "height": height,
            "original_width": self.original_view.source_size[0],
            "original_height": self.original_view.source_size[1],
            "dpi": dpi_x,
            "dpi_x": dpi_x,
            "dpi_y": dpi_y,
        }
        self.processing_done(result, existing=True)
        self.status.setText("当前显示的是已存在的处理后图片；点击“处理当前图片”可重新处理并刷新预览")

    def preview_failed(self, message: str) -> None:
        self.process_button.setEnabled(bool(self.source))
        self.progress_label.setText("预览失败")
        self.status.setText("图片预览失败")
        QMessageBox.warning(self, "无法读取图片", message)

    def _set_progress(self, value: int, message: str) -> None:
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)
        self.status.setText(f"{message}（{value}%）")

    def status_message(self, message: str) -> None:
        self.status.setText(message)

    def process(self) -> None:
        if not self.source or not self.output_root:
            return
        self.process_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status.setText("正在处理当前图片，请稍候…")
        PREVIEW_CACHE.mkdir(parents=True, exist_ok=True)
        output_dir = PREVIEW_CACHE / "current_results" / f"{self.source.stem}_{self.source.stat().st_size}"
        self.current_result_dir = output_dir
        self.current_saved = False
        self.worker = ProcessWorker(self.source, output_dir, self.type_combo.currentData(), self.mode_combo.currentData(), self.keep_faint.isChecked(), self._current_params())
        self.worker.progress.connect(self.processing_progress)
        self.worker.completed.connect(self.processing_done)
        self.worker.failed.connect(self.processing_failed)
        self.worker.start()

    def processing_done(self, result: dict, existing: bool = False) -> None:
        self.result = result
        self.current_result_dir = Path(result["output_dir"])
        self.current_saved = False
        self._edited_image = None
        self._edited_result_path = None
        self._edited_preview_path = None
        self._eraser_undo.clear()
        self._edited_dirty = False
        self._edit_generation += 1
        self.progress_bar.setValue(100)
        elapsed = result.get("processing_seconds") if not existing else None
        if elapsed is not None:
            elapsed_text = f"当前图片处理完成，耗时{float(elapsed):.2f}秒"
            self.progress_label.setText(elapsed_text)
        else:
            self.progress_label.setText("当前图片处理完成")
        self.process_button.setEnabled(True)
        self.export_button.setEnabled(True)
        output_dir = Path(result["output_dir"])
        cleaned_preview = Path(result.get("preview_path", result.get("enhanced_preview", output_dir / result["enhanced"])))
        cleaned_full = output_dir / result["enhanced"]
        self.cleaned_view.set_image(cleaned_preview, high_res_source=cleaned_full)
        self.cleaned_view.set_coordinate_size(result["width"], result["height"], result.get("dpi", 300))
        self.original_view.set_coordinate_size(result["original_width"], result["original_height"], _read_dpi(self.source))
        self.reset_views()
        self._prepare_edit_image(cleaned_full)
        if not existing:
            if elapsed is not None:
                self.status.setText(elapsed_text)
            else:
                self.status.setText(f"处理完成：{result['width']}×{result['height']}，文字墨色保持原图像素")

    def _ensure_edited_image(self) -> Image.Image | None:
        if self._edited_image is not None:
            return self._edited_image
        if not self.result:
            return None
        source_path = Path(self.result["output_dir"]) / self.result["enhanced"]
        if not source_path.exists():
            return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", Image.DecompressionBombWarning)
                self._edited_image = read_text_image(source_path)
            self.cleaned_view.set_eraser_target(True)
            return self._edited_image
        except (OSError, ValueError):
            return None

    def _prepare_edit_image(self, source: Path) -> None:
        """Decode editable pixels in the background before the first click."""
        self.cleaned_view.set_eraser_target(False)
        generation = self._edit_generation
        worker = EditImageLoadWorker(source)
        self._edit_loaders.append(worker)
        def ready(image):
            if generation != self._edit_generation:
                image.close()
                return
            if self._edited_image is None:
                self._edited_image = image
            else:
                image.close()
            self.cleaned_view.set_eraser_target(True)
        worker.loaded.connect(ready)
        worker.failed.connect(lambda message: self.status.setText(f"编辑准备失败：{message}") if generation == self._edit_generation else None)
        worker.finished.connect(lambda: self._edit_loaders.remove(worker) if worker in self._edit_loaders else None)
        worker.start()

    def _write_edited_file(self) -> None:
        if self._edited_image is None or not self.result:
            return
        root = PREVIEW_CACHE / "edits"
        root.mkdir(parents=True, exist_ok=True)
        stem = self.source.stem if self.source else "image"
        self._edited_result_path = root / f"{stem}_{self._edited_image.width}x{self._edited_image.height}.tif"
        dpi_x = float(self.result.get("dpi_x", self.result.get("dpi", 300)))
        dpi_y = float(self.result.get("dpi_y", dpi_x))
        self._edited_image.save(self._edited_result_path, format="TIFF", compression="tiff_lzw", dpi=(dpi_x, dpi_y))

    def _refresh_edited_preview(self) -> None:
        if self._edited_image is None or not self.result:
            return
        self.cleaned_view.update()

    def begin_erase(self) -> None:
        if self._edited_image is None:
            return
        # Store only touched image patches, never a full-page alpha snapshot.
        self._eraser_undo.append([])
        path = QPainterPath()
        path.setFillRule(Qt.FillRule.WindingFill)
        self.cleaned_view.erase_paths.append(path)
        if len(self._eraser_undo) > 20:
            for _, patch in self._eraser_undo.pop(0):
                patch.close()
            # Keep older overlays visible; the last 20 strokes are undoable.

    def eraser_size_changed(self, size: float) -> None:
        self.status.setText(f"橡皮擦大小：{size:.0f}px（Ctrl+滚轮调整，Ctrl+Z 撤销）")

    def undo_erase(self) -> None:
        if not self._eraser_undo or self._edited_image is None:
            return
        for box, patch in reversed(self._eraser_undo.pop()):
            self._edited_image.paste(patch, box[:2])
            patch.close()
        if self.cleaned_view.erase_paths:
            self.cleaned_view.erase_paths.pop()
        self._edited_dirty = True
        self.current_saved = False
        self.cleaned_view.update()
        self.status.setText("已撤销最近一次橡皮擦修改")

    def finish_erase(self) -> None:
        if self._edited_image is not None:
            self.status.setText("橡皮擦修改已保留在内存中，点击“保存当前图片”导出；Ctrl + Z 撤销")

    def erase_processed_stroke(self, start: QPointF, end: QPointF) -> None:
        image = self._edited_image
        if image is None:
            return
        rect = self.cleaned_view._image_rect()
        if rect.width() <= 0 or rect.height() <= 0:
            return
        if not self._eraser_undo:
            self.begin_erase()
        radius = self.cleaned_view.eraser_size / 2.0
        sx, sy = rect.width() / image.width, rect.height() / image.height
        left = max(0, floor((min(start.x(), end.x()) - radius - rect.left()) / sx))
        top = max(0, floor((min(start.y(), end.y()) - radius - rect.top()) / sy))
        right = min(image.width, ceil((max(start.x(), end.x()) + radius - rect.left()) / sx))
        bottom = min(image.height, ceil((max(start.y(), end.y()) + radius - rect.top()) / sy))
        if right <= left or bottom <= top:
            return
        # Test source pixel centers against the exact on-screen circle/capsule.
        # This uses the same floating canvas rectangle as drawing and works
        # with pan, zoom, resampled results and different preview resolutions.
        xs = rect.left() + (np.arange(left, right)[None, :] + .5) * sx
        ys = rect.top() + (np.arange(top, bottom)[:, None] + .5) * sy
        dx, dy = end.x() - start.x(), end.y() - start.y()
        length2 = dx * dx + dy * dy
        fraction = np.clip(((xs - start.x()) * dx + (ys - start.y()) * dy) / length2, 0, 1) if length2 else 0
        hit = (xs - start.x() - fraction * dx) ** 2 + (ys - start.y() - fraction * dy) ** 2 <= radius * radius
        box = (left, top, right, bottom)
        before = image.crop(box)
        pixels = np.array(before)
        if not np.any(hit & (pixels[:, :, 3] > 0)):
            before.close()
            return
        self._eraser_undo[-1].append((box, before))
        pixels[:, :, 3][hit] = 0
        image.paste(Image.fromarray(pixels, "RGBA"), (left, top))
        self._edited_dirty = True
        self.current_saved = False
        self.cleaned_view.paint_eraser_preview(start, end)
        self.status.setText("橡皮擦已修改处理结果；按住 Ctrl + 鼠标左键拖动可继续擦除")

    def processing_failed(self, message: str) -> None:
        self.process_button.setEnabled(True)
        self.status.setText("处理失败")
        QMessageBox.critical(self, "处理失败", message)

    def processing_progress(self, value: int, message: str) -> None:
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)
        self.status.setText(f"{message}（{value}%）")

    def process_batch(self) -> None:
        if not self.files or not self.source_root or not self.output_root or self.batch_worker and self.batch_worker.isRunning():
            return
        self.process_button.setEnabled(False)
        self.batch_button.setEnabled(False)
        self.cancel_batch_button.setEnabled(True)
        self.batch_progress_bar.setValue(0)
        self.progress_bar.setValue(0)
        self.batch_worker = BatchWorker(self.files, self.source_root, self.output_root, self.type_combo.currentData(), self.mode_combo.currentData(), self.keep_faint.isChecked(), self._current_params())
        self.batch_worker.item_progress.connect(self.processing_progress)
        self.batch_worker.overall_progress.connect(self.batch_progress)
        self.batch_worker.completed.connect(self.batch_done)
        self.batch_worker.failed.connect(self.batch_failed)
        self.batch_worker.start()
        self.status.setText(f"开始批量处理，共 {len(self.files)} 张图片")

    def batch_progress(self, current: int, total: int, name: str) -> None:
        value = int(current * 100 / max(1, total))
        self.batch_progress_bar.setValue(value)

    def batch_done(self, summary: dict) -> None:
        self.batch_button.setEnabled(True)
        self.cancel_batch_button.setEnabled(False)
        self.process_button.setEnabled(bool(self.source))
        failed = summary.get("failed", [])
        state = "已取消" if summary.get("cancelled") else "已完成"
        elapsed = summary.get("elapsed_seconds")
        timing = f"，全部耗时{float(elapsed):.2f}秒" if elapsed is not None else ""
        self.status.setText(f"批量处理{state}：成功 {summary.get('succeeded', 0)} 张，跳过 {summary.get('skipped', 0)} 张，失败 {len(failed)} 张{timing}")
        self.progress_label.setText(f"批量任务结束{timing}")
        if failed:
            QMessageBox.warning(self, "批量处理完成", "以下文件处理失败：\n" + "\n".join(failed[:12]))

    def batch_failed(self, message: str) -> None:
        self.batch_button.setEnabled(True)
        self.cancel_batch_button.setEnabled(False)
        self.process_button.setEnabled(bool(self.source))
        self.status.setText("批量处理失败")
        QMessageBox.critical(self, "批量处理失败", message)

    def cancel_batch(self) -> None:
        if self.batch_worker:
            self.batch_worker.request_cancel()
            self.status.setText("正在取消批量任务，当前文件完成后停止…")

    def reset_views(self) -> None:
        self._syncing_views = True
        try:
            candidates = [view.fit_scale() for view in (self.original_view, self.cleaned_view) if not view.pixmap.isNull()]
            scale = min(candidates) if candidates else None
            self.original_view.reset_view(scale)
            self.cleaned_view.reset_view(scale)
        finally:
            self._syncing_views = False

    def save_current(self) -> None:
        if not self.result:
            self.status.setText("请先点击“处理当前图片”，或打开已有处理结果")
            return
        if not self.output_root or not self.source:
            QMessageBox.warning(self, "未设置输出目录", "请先选择处理结果保存目录。")
            return
        destination = self.output_root
        destination.mkdir(parents=True, exist_ok=True)
        source_dir = Path(self.result["output_dir"])
        name = self.result.get("layered_tif", self.result["enhanced"])
        source_file = source_dir / name
        if not source_file.exists():
            QMessageBox.warning(self, "保存失败", "没有找到处理后的图片文件。")
            return
        params = self._current_params()
        destination_file = output_path(destination, self.source.stem, params["output_format"])
        if destination_file.resolve() == self.source.resolve():
            QMessageBox.warning(self, "无法覆盖原图", "保存位置与原图相同，请选择其他输出目录。")
            return
        if destination_file.exists():
            answer = QMessageBox.question(
                self,
                "文件已存在",
                f"目标文件已存在：\n{destination_file}\n\n是否覆盖？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                self.status.setText("已取消保存，原有文件未被覆盖")
                return
        dpi = (float(self.result.get("dpi_x", self.result.get("dpi", 300))),
               float(self.result.get("dpi_y", self.result.get("dpi", 300))))
        source = self._edited_image if self._edited_dirty and self._edited_image is not None else source_file
        worker = ExportWorker(source, destination_file, params, dpi)
        self.export_worker = worker
        loop = QEventLoop()
        errors = []
        worker.progress.connect(self._set_progress)
        worker.failed.connect(errors.append)
        worker.finished.connect(loop.quit)
        controls = [self.settings_panel, self.previous_button, self.next_button, self.jump_spin, self.cleaned_view]
        states = [control.isEnabled() for control in controls]
        for control in controls:
            control.setEnabled(False)
        worker.start()
        loop.exec()
        worker.wait()
        for control, state in zip(controls, states):
            control.setEnabled(state)
        if errors:
            self.progress_label.setText("保存失败")
            QMessageBox.warning(self, "保存失败", errors[0])
            return
        self.current_saved = True
        self.status.setText(f"当前图片已保存到：{destination_file}")

    def closeEvent(self, event) -> None:  # noqa: N802
        self._edit_generation += 1
        if self.batch_worker:
            self.batch_worker.request_cancel()
        workers = [self.worker, self.preview_worker, self.batch_worker, self.export_worker,
                   *self._edit_loaders, *self.original_view._loaders, *self.cleaned_view._loaders]
        for worker in workers:
            if worker and worker.isRunning():
                worker.requestInterruption()
                worker.wait()
        QApplication.instance().removeEventFilter(self)
        super().closeEvent(event)

    def export_result(self) -> None:
        self.save_current()


def main() -> int:
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))
    window = MainWindow()
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
