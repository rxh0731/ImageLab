from __future__ import annotations

import json
import shutil
import sys
import ctypes
import warnings
from math import ceil
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QEvent, QPointF, QRectF, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QKeyEvent, QPainter, QPen, QPixmap, QPolygonF
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
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .processing import process_image


ROOT = Path(__file__).resolve().parent.parent
UPLOADS = ROOT / "uploads"
OUTPUTS = ROOT / "outputs"
TEMPLATES = ROOT / "templates"
PREVIEW_CACHE = ROOT / ".preview_cache"
SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

DEFAULT_TEMPLATES = {
    "拓印保守去背景": {"image_type": "rubbing", "mode": "conservative", "keep_faint": True, "background_radius": 31, "contrast": 1.28, "threshold_bias": 5, "denoise_radius": 0.65},
    "古籍平衡去背景": {"image_type": "book", "mode": "balanced", "keep_faint": True, "background_radius": 41, "contrast": 1.18, "threshold_bias": 3, "denoise_radius": 0.8},
    "手稿保守去背景": {"image_type": "manuscript", "mode": "conservative", "keep_faint": True, "background_radius": 25, "contrast": 1.10, "threshold_bias": -2, "denoise_radius": 0.35},
}


class ImageView(QWidget):
    high_res_status = Signal(str)
    view_changed = Signal(float, QPointF)

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title = title
        self.pixmap = QPixmap()
        self.regions: list[dict] = []
        self.source_size = (1, 1)
        self.region_size = (1, 1)
        self.show_regions = False
        self.selected_region: int | None = None
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self._panning = False
        self._last_mouse = QPointF(0, 0)
        self._space_down = False
        self._high_res_source: Path | None = None
        self._high_res_loader: ImageLoadWorker | None = None
        self._high_res_requested = False
        self._high_res_level = 0
        self._high_res_max_dimension = 0
        self.setMinimumSize(280, 360)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_image(self, path: Path, high_res_source: Path | None = None) -> None:
        image = QImage(str(path))
        self.pixmap = QPixmap.fromImage(image) if not image.isNull() else QPixmap()
        if not self.pixmap.isNull():
            self.source_size = (self.pixmap.width(), self.pixmap.height())
            self.region_size = self.source_size
        self._high_res_source = high_res_source
        self._high_res_requested = False
        self._high_res_loader = None
        self._high_res_level = 0
        self._high_res_max_dimension = max(self.source_size)
        self.reset_view()
        self.update()

    def clear_image(self) -> None:
        self.pixmap = QPixmap()
        self._high_res_source = None
        self._high_res_loader = None
        self.source_size = (1, 1)
        self.region_size = (1, 1)
        self.reset_view()
        self.update()

    def set_regions(self, regions: list[dict]) -> None:
        self.regions = regions
        self.update()

    def set_coordinate_size(self, width: int, height: int) -> None:
        self.region_size = (max(1, width), max(1, height))
        self.update()

    def request_high_res(self) -> None:
        if self._high_res_source is None or self._high_res_loader is not None:
            return
        if not self._high_res_source.exists():
            return
        source_max = max(self.region_size)
        target_max = min(source_max, max(8192, int(max(self.width(), self.height()) * self.zoom * 2.0)))
        if self._high_res_requested and target_max <= self._high_res_max_dimension:
            return
        if target_max <= self._high_res_max_dimension:
            return
        self._high_res_requested = True
        self.high_res_status.emit("正在异步载入高清原图…")
        self._high_res_loader = ImageLoadWorker(self._high_res_source, target_max)
        self._high_res_loader.loaded.connect(self._high_res_loaded)
        self._high_res_loader.failed.connect(self._high_res_failed)
        self._high_res_loader.start()

    def _high_res_loaded(self, image: QImage) -> None:
        pixmap = QPixmap.fromImage(image)
        if pixmap.isNull():
            self._high_res_failed("高清原图无法创建显示缓存")
            return
        self.pixmap = pixmap
        self.source_size = (pixmap.width(), pixmap.height())
        self._high_res_max_dimension = max(self.source_size)
        self.high_res_status.emit(f"高清原图已载入（{pixmap.width()}×{pixmap.height()}）")
        self.update()
        self._high_res_loader = None

    def _high_res_failed(self, message: str) -> None:
        self._high_res_loader = None
        self.high_res_status.emit(f"高清原图载入失败，继续使用预览图：{message}")

    def reset_view(self) -> None:
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self._space_down = False
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

    def _set_hand_cursor(self, dragging: bool) -> None:
        if dragging or self._space_down:
            self.setCursor(Qt.CursorShape.ClosedHandCursor if dragging else Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def _image_rect(self) -> QRectF:
        if self.pixmap.isNull():
            return QRectF()
        margin = 18
        fitted = self.pixmap.scaled(self.width() - margin * 2, self.height() - margin * 2, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        size = QPointF(fitted.width() * self.zoom, fitted.height() * self.zoom)
        center = QPointF(self.width() / 2, self.height() / 2) + self.pan
        return QRectF(center.x() - size.x() / 2, center.y() - size.y() / 2, size.x(), size.y())

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#1c2229"))
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if self.pixmap.isNull():
            painter.setPen(QColor("#aab3bd"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "尚未导入图片")
            return
        rect = self._image_rect()
        painter.drawPixmap(rect.toRect(), self.pixmap)
        x, y = rect.left(), rect.top()
        painter.setPen(QColor("#e6e9ed"))
        painter.drawText(int(x + 9), int(y + 18), self.title)
        if not self.show_regions:
            painter.setPen(QColor("#aab3bd"))
            painter.drawText(12, self.height() - 12, f"缩放 {round(self.zoom * 100)}% · Alt+滚轮缩放 · 空格+拖动平移")
            return
        scale_x = rect.width() / self.region_size[0]
        scale_y = rect.height() / self.region_size[1]
        for region in self.regions[:500]:
            points = [QPolygonF([QPointF(x + px * scale_x, y + py * scale_y) for px, py in region["polygon"]])]
            if not points[0]:
                continue
            confidence = region.get("confidence", 100)
            color = QColor("#b8793e" if confidence < 70 else "#a44a32")
            if region["id"] == self.selected_region:
                color = QColor("#2f6f5b")
            pen = QPen(color, 3 if region["id"] == self.selected_region else 1.5)
            painter.setPen(pen)
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), 25))
            painter.drawPolygon(points[0])
        painter.setPen(QColor("#aab3bd"))
        painter.drawText(12, self.height() - 12, f"缩放 {round(self.zoom * 100)}% · Alt+滚轮缩放 · 空格+拖动平移")

    def wheelEvent(self, event) -> None:  # noqa: N802
        # Windows may omit Alt from the native wheel event while it is still held.
        modifiers = event.modifiers() | QApplication.keyboardModifiers()
        if sys.platform == "win32" and ctypes.windll.user32.GetAsyncKeyState(0x12) & 0x8000:
            modifiers |= Qt.KeyboardModifier.AltModifier
        if not (modifiers & Qt.KeyboardModifier.AltModifier) or self.pixmap.isNull():
            event.ignore()
            return
        self.zoom_at(event.position(), self.wheel_delta(event))
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
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        if event.button() == Qt.MouseButton.MiddleButton or (event.button() == Qt.MouseButton.LeftButton and self._space_down):
            self._panning = True
            self._last_mouse = event.position()
            self._set_hand_cursor(True)
            event.accept()
            return
        event.ignore()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
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
        if self._panning and event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._panning = False
            self._set_hand_cursor(False)
            event.accept()
            return
        event.ignore()

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
            # Decode a bounded grayscale level. Loading an entire RGB TIFF can
            # exceed Qt's 256 MB image allocation limit even on a worker thread.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", Image.DecompressionBombWarning)
                opened = Image.open(self.source)
                source_max = max(opened.size)
                reduce_factor = max(1, ceil(source_max / self.max_dimension))
                gray = opened.reduce(reduce_factor) if reduce_factor > 1 else opened.copy()
                gray = gray.convert("L")
                width, height = gray.size
                raw = gray.tobytes()
                image = QImage(raw, width, height, width, QImage.Format.Format_Grayscale8).copy()
                gray.close()
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
        try:
            job_id = self.source.stem
            result = process_image(
                self.source,
                self.output_dir,
                self.image_type,
                self.mode,
                self.keep_faint,
                params=self.params,
                progress_callback=lambda value, message: self.progress.emit(value, message),
            )
            result["job_id"] = job_id
            result["source"] = str(self.source)
            result["output_dir"] = str(self.output_dir)
            self.completed.emit(result)
        except Exception as exc:  # pragma: no cover - surfaced in the UI
            self.failed.emit(str(exc))


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
                with Image.open(self.source) as opened:
                    width, height = opened.size
                    self.progress.emit(25, "保存原始文件")
                    shutil.copy2(self.source, self.destination)
                    self.progress.emit(45, "生成预览")
                    reduce_factor = max(1, ceil(max(width, height) / 2400))
                    preview = opened.reduce(reduce_factor) if reduce_factor > 1 else opened.copy()
                    preview = preview.convert("RGB")
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
            cache_name = f"{self.source.stem}_{self.source.stat().st_size}_{self.source.stat().st_mtime_ns}_preview.jpg"
            preview_path = PREVIEW_CACHE / cache_name
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", Image.DecompressionBombWarning)
                with Image.open(self.source) as opened:
                    width, height = opened.size
                    if not preview_path.exists():
                        self.progress.emit(30, "生成当前图片预览")
                        factor = max(1, ceil(max(width, height) / 2400))
                        preview = opened.reduce(factor) if factor > 1 else opened.copy()
                        preview = preview.convert("RGB")
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
        failed: list[str] = []
        total = len(self.files)
        try:
            for index, source in enumerate(self.files):
                if self.cancel_requested:
                    break
                self.item_progress.emit(0, f"正在处理 {source.name}")
                try:
                    relative_parent = source.relative_to(self.source_root).parent
                    output_dir = self.output_root / relative_parent / source.stem
                    result = process_image(
                        source,
                        output_dir,
                        self.image_type,
                        self.mode,
                        self.keep_faint,
                        params=self.params,
                        progress_callback=lambda value, message, name=source.name: self.item_progress.emit(value, f"{name}：{message}"),
                    )
                    succeeded += 1
                except Exception as exc:
                    failed.append(f"{source.name}: {exc}")
                self.overall_progress.emit(index + 1, total, source.name)
            self.completed.emit({"total": total, "succeeded": succeeded, "failed": failed, "cancelled": self.cancel_requested})
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
        self._auto_process_after_preview = False
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
            QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox { min-height:34px; border:1px solid #46515d; border-radius:5px; padding:0 8px; background:#1c2229; color:#e6e9ed; }
            QComboBox QAbstractItemView { background:#252d36; color:#e6e9ed; selection-background-color:#a94f36; }
            QCheckBox { spacing:8px; color:#dce1e6; }
            QTableWidget { background:#1c2229; color:#e6e9ed; border:1px solid #46515d; gridline-color:#303943; }
            QHeaderView::section { padding:5px; background:#2b333d; border:0; color:#c4ccd5; font-size:11px; }
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
        self.import_button = QPushButton("选择原图目录")
        self.import_button.clicked.connect(self.select_source_directory)
        header.addWidget(self.import_button)
        self.export_button = QPushButton("保存当前图片")
        self.export_button.setObjectName("primary")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.save_current)
        header.addWidget(self.export_button)
        outer.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_preview_panel())
        splitter.addWidget(self._build_right_panel())
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
        self.preview_title = QLabel("原图与去背景结果对比")
        self.preview_title.setStyleSheet("font-size:15px;font-weight:600")
        head.addWidget(self.preview_title)
        head.addStretch()
        self.previous_button = QPushButton("上一张")
        self.previous_button.clicked.connect(lambda: self.change_file(-1))
        self.previous_button.setEnabled(False)
        head.addWidget(self.previous_button)
        self.next_button = QPushButton("下一张")
        self.next_button.clicked.connect(lambda: self.change_file(1))
        self.next_button.setEnabled(False)
        head.addWidget(self.next_button)
        self.page_label = QLabel("0 / 0")
        self.page_label.setObjectName("muted")
        head.addWidget(self.page_label)
        self.reset_view_button = QPushButton("重置视图")
        self.reset_view_button.clicked.connect(self.reset_views)
        head.addWidget(self.reset_view_button)
        self.regions_check = QCheckBox("显示文字区域")
        self.regions_check.toggled.connect(self.toggle_regions)
        self.regions_check.setEnabled(False)
        head.addWidget(self.regions_check)
        layout.addLayout(head)
        self.preview_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.original_view = ImageView("原图")
        self.cleaned_view = ImageView("白底净化结果")
        self.original_view.high_res_status.connect(self.status_message)
        self.cleaned_view.high_res_status.connect(self.status_message)
        self.original_view.view_changed.connect(lambda zoom, pan: self._sync_view(self.cleaned_view, zoom, pan))
        self.cleaned_view.view_changed.connect(lambda zoom, pan: self._sync_view(self.original_view, zoom, pan))
        self.preview_splitter.addWidget(self.original_view)
        self.preview_splitter.addWidget(self.cleaned_view)
        layout.addWidget(self.preview_splitter, 1)
        self.status = QLabel("请导入图片开始处理")
        self.status.setObjectName("muted")
        layout.addWidget(self.status)
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
        heading = QLabel("工作设置")
        heading.setStyleSheet("font-size:15px;font-weight:600")
        layout.addWidget(heading)

        layout.addWidget(QLabel("目录"))
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
        layout.addWidget(QLabel("输出目录"))
        layout.addLayout(output_row)
        self.recursive_check = QCheckBox("包含子目录")
        self.recursive_check.setChecked(True)
        self.recursive_check.toggled.connect(self.refresh_file_list)
        layout.addWidget(self.recursive_check)

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
        self.background_radius = QSpinBox()
        self.background_radius.setRange(3, 301)
        self.background_radius.setSingleStep(2)
        form.addRow("背景判定半径", self.background_radius)
        self.threshold_bias = QSpinBox()
        self.threshold_bias.setRange(-80, 80)
        form.addRow("文字判定偏移", self.threshold_bias)
        self.contrast = QDoubleSpinBox()
        self.contrast.setRange(0.5, 2.0)
        self.contrast.setSingleStep(0.05)
        self.contrast.setDecimals(2)
        form.addRow("分析对比度", self.contrast)
        self.denoise_radius = QDoubleSpinBox()
        self.denoise_radius.setRange(0.0, 4.0)
        self.denoise_radius.setSingleStep(0.05)
        self.denoise_radius.setDecimals(2)
        form.addRow("分析降噪半径", self.denoise_radius)
        layout.addLayout(form)
        self.keep_faint = QCheckBox("保留淡色笔画")
        self.keep_faint.setChecked(True)
        layout.addWidget(self.keep_faint)
        note = QLabel("参数只用于判断文字区域。文字区域内的原图墨色和像素不会改变。")
        note.setWordWrap(True)
        note.setObjectName("muted")
        layout.addWidget(note)

        self.process_button = QPushButton("处理当前图片")
        self.process_button.setObjectName("primary")
        self.process_button.setEnabled(False)
        self.process_button.clicked.connect(self.process)
        layout.addWidget(self.process_button)
        self.batch_button = QPushButton("自动批量处理目录")
        self.batch_button.setEnabled(False)
        self.batch_button.clicked.connect(self.process_batch)
        layout.addWidget(self.batch_button)
        self.cancel_batch_button = QPushButton("取消批量处理")
        self.cancel_batch_button.setEnabled(False)
        self.cancel_batch_button.clicked.connect(self.cancel_batch)
        layout.addWidget(self.cancel_batch_button)
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
        self.region_table = QTableWidget(0, 2)
        self.region_table.setHorizontalHeaderLabels(["区域", "置信度"])
        self.region_table.setMaximumHeight(170)
        self.region_table.cellClicked.connect(self.select_region)
        layout.addWidget(QLabel("低置信度区域"))
        layout.addWidget(self.region_table)
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
            self.batch_button.setEnabled(bool(self.files))

    def refresh_file_list(self) -> None:
        if not self.source_root:
            return
        iterator = self.source_root.rglob("*") if self.recursive_check.isChecked() else self.source_root.glob("*")
        self.files = sorted((path for path in iterator if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES), key=lambda path: str(path).lower())
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
        self.previous_button.setEnabled(self.current_index > 0)
        self.next_button.setEnabled(0 <= self.current_index < total - 1)
        self.batch_button.setEnabled(bool(self.files and self.output_root))

    def change_file(self, offset: int) -> None:
        if not self.files or self.preview_worker and self.preview_worker.isRunning() or self.worker and self.worker.isRunning() or self.batch_worker and self.batch_worker.isRunning():
            return
        target = self.current_index + offset
        if 0 <= target < len(self.files):
            self.current_index = target
            self._auto_process_after_preview = True
            self._update_page_controls()
            self.load_current_file()

    def load_current_file(self) -> None:
        if self.current_index < 0 or self.current_index >= len(self.files):
            return
        self.source = self.files[self.current_index]
        self.result = None
        self.current_result_dir = None
        self.current_saved = False
        self.file_label.setText(f"当前图片：{self.source.name}\n{self.source}")
        self.preview_source = None
        self.original_view.clear_image()
        self.cleaned_view.clear_image()
        self.region_table.setRowCount(0)
        self.regions_check.setChecked(False)
        self.regions_check.setEnabled(False)
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
        self.original_view.set_image(self.preview_source, high_res_source=self.source)
        self.original_view.set_coordinate_size(width, height)
        self.cleaned_view.clear_image()
        self.process_button.setEnabled(True)
        self.batch_button.setEnabled(bool(self.files and self.output_root))
        self.progress_bar.setValue(100)
        self.progress_label.setText(f"预览就绪：{width}×{height}")
        if self._auto_process_after_preview:
            self._auto_process_after_preview = False
            if self.output_root:
                self.status.setText("翻页完成，正在按当前模板自动处理…")
                QTimer.singleShot(0, self.process)
                return
        self.status.setText("可以处理当前图片，或使用自动批量处理目录")

    def preview_failed(self, message: str) -> None:
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
        relative_parent = self.source.relative_to(self.source_root).parent if self.source_root else Path()
        PREVIEW_CACHE.mkdir(parents=True, exist_ok=True)
        output_dir = PREVIEW_CACHE / "current_results" / f"{self.source.stem}_{self.source.stat().st_size}"
        self.current_result_dir = output_dir
        self.current_saved = False
        self.worker = ProcessWorker(self.source, output_dir, self.type_combo.currentData(), self.mode_combo.currentData(), self.keep_faint.isChecked(), self._current_params())
        self.worker.progress.connect(self.processing_progress)
        self.worker.completed.connect(self.processing_done)
        self.worker.failed.connect(self.processing_failed)
        self.worker.start()

    def processing_done(self, result: dict) -> None:
        self.result = result
        self.current_result_dir = Path(result["output_dir"])
        self.current_saved = False
        self.progress_bar.setValue(100)
        self.progress_label.setText("当前图片处理完成")
        self.process_button.setEnabled(True)
        self.export_button.setEnabled(True)
        output_dir = Path(result["output_dir"])
        cleaned_preview = output_dir / result.get("enhanced_preview", result["enhanced"])
        cleaned_full = output_dir / result["enhanced"]
        self.cleaned_view.set_image(cleaned_preview, high_res_source=cleaned_full)
        self.cleaned_view.set_coordinate_size(result["width"], result["height"])
        self.cleaned_view.set_regions(result["regions"])
        self.original_view.set_coordinate_size(result["width"], result["height"])
        self.original_view.set_regions(result["regions"])
        self.regions_check.setEnabled(bool(result["regions"]))
        self.status.setText(f"处理完成：{result['width']}×{result['height']}，文字墨色保持原图像素")
        self.populate_regions(result["regions"])

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
        self.progress_label.setText(f"批量进度：{current}/{total} · {name}")

    def batch_done(self, summary: dict) -> None:
        self.batch_button.setEnabled(True)
        self.cancel_batch_button.setEnabled(False)
        self.process_button.setEnabled(bool(self.source))
        failed = summary.get("failed", [])
        state = "已取消" if summary.get("cancelled") else "已完成"
        self.status.setText(f"批量处理{state}：成功 {summary.get('succeeded', 0)} 张，失败 {len(failed)} 张")
        self.progress_label.setText("批量任务结束")
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

    def populate_regions(self, regions: list[dict]) -> None:
        low = [r for r in regions if r.get("confidence", 100) < 70]
        self.region_table.setRowCount(len(low))
        for row, region in enumerate(low):
            self.region_table.setItem(row, 0, QTableWidgetItem(f"区域 {region['id']}"))
            self.region_table.setItem(row, 1, QTableWidgetItem(f"{region['confidence']}%"))

    def toggle_regions(self, checked: bool) -> None:
        self.original_view.show_regions = checked
        self.cleaned_view.show_regions = checked
        self.original_view.update()
        self.cleaned_view.update()

    def reset_views(self) -> None:
        self._syncing_views = True
        try:
            self.original_view.reset_view()
            self.cleaned_view.set_view_state(self.original_view.zoom, self.original_view.pan)
        finally:
            self._syncing_views = False

    def select_region(self, row: int, column: int) -> None:
        if not self.result:
            return
        low = [r for r in self.result["regions"] if r.get("confidence", 100) < 70]
        if row >= len(low):
            return
        region_id = low[row]["id"]
        self.original_view.selected_region = region_id
        self.cleaned_view.selected_region = region_id
        self.regions_check.setChecked(True)
        self.status.setText(f"已选中区域 {region_id}，请对照原图确认是否保留")
        self.original_view.update()
        self.cleaned_view.update()

    def save_current(self) -> None:
        if not self.result:
            self.process()
            return
        if not self.output_root or not self.source:
            QMessageBox.warning(self, "未设置输出目录", "请先选择处理结果保存目录。")
            return
        relative_parent = self.source.relative_to(self.source_root).parent if self.source_root else Path()
        destination = self.output_root / relative_parent / self.source.stem
        destination.mkdir(parents=True, exist_ok=True)
        source_dir = Path(self.result["output_dir"])
        names = [self.result.get("white_tif", self.result["enhanced"]), self.result.get("text_tif", self.result["transparent"]), self.result["text_mask"], self.result["enhanced_preview"]]
        if self.result.get("white_preview"):
            names.append(self.result["white_preview"])
        for name in names:
            source_file = source_dir / name
            if source_file.exists():
                shutil.copy2(source_file, destination / name)
        self.current_saved = True
        self.status.setText(f"当前图片已保存到：{destination}")

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
