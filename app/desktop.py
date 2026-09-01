from __future__ import annotations

import shutil
import sys
import ctypes
from math import ceil
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QEvent, QPointF, QRectF, QThread, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QImageReader, QKeyEvent, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
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


class ImageView(QWidget):
    high_res_status = Signal(str)

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
        self.reset_view()
        self.update()

    def set_regions(self, regions: list[dict]) -> None:
        self.regions = regions
        self.update()

    def set_coordinate_size(self, width: int, height: int) -> None:
        self.region_size = (max(1, width), max(1, height))
        self.update()

    def request_high_res(self) -> None:
        if self._high_res_source is None or self._high_res_requested or self._high_res_loader is not None:
            return
        if not self._high_res_source.exists():
            return
        self._high_res_requested = True
        self.high_res_status.emit("正在异步载入高清原图…")
        self._high_res_loader = ImageLoadWorker(self._high_res_source)
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
        self.high_res_status.emit(f"高清原图已载入（{pixmap.width()}×{pixmap.height()}）")
        self.update()
        self._high_res_loader = None

    def _high_res_failed(self, message: str) -> None:
        self._high_res_loader = None
        self.high_res_status.emit(f"高清原图载入失败，继续使用预览图：{message}")

    def reset_view(self) -> None:
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self._set_hand_cursor(False)
        self.update()

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
        painter.fillRect(self.rect(), QColor("#ebe7dc"))
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if self.pixmap.isNull():
            painter.setPen(QColor("#756f62"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "尚未导入图片")
            return
        rect = self._image_rect()
        painter.drawPixmap(rect.toRect(), self.pixmap)
        x, y = rect.left(), rect.top()
        painter.setPen(QColor("#37342d"))
        painter.drawText(int(x + 9), int(y + 18), self.title)
        if not self.show_regions:
            painter.setPen(QColor("#756f62"))
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
        painter.setPen(QColor("#756f62"))
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

    def __init__(self, source: Path) -> None:
        super().__init__()
        self.source = source

    def run(self) -> None:
        qt_error = ""
        try:
            reader = QImageReader(str(self.source))
            reader.setAutoTransform(True)
            image = reader.read()
            if not image.isNull():
                self.loaded.emit(image)
                return
            qt_error = reader.errorString() or "Qt 无法读取图像数据"
        except Exception as exc:  # pragma: no cover - surfaced in the UI
            qt_error = str(exc)
        # Some large or compressed TIFFs pass canRead() but fail during Qt decode.
        # Pillow is used as a background fallback and the copied bytes are owned
        # by QImage, so the signal remains valid after this worker exits.
        try:
            with Image.open(self.source) as opened:
                opened.load()
                rgb = opened.convert("RGB")
                width, height = rgb.size
                raw = rgb.tobytes()
                image = QImage(raw, width, height, width * 3, QImage.Format.Format_RGB888).copy()
                rgb.close()
            if image.isNull():
                raise RuntimeError("备用解码器生成空图像")
            self.loaded.emit(image)
        except Exception as exc:  # pragma: no cover - surfaced in the UI
            self.failed.emit(f"Qt: {qt_error}; Pillow: {exc}")


class ProcessWorker(QThread):
    completed = Signal(dict)
    failed = Signal(str)
    progress = Signal(int, str)

    def __init__(self, source: Path, image_type: str, mode: str, keep_faint: bool) -> None:
        super().__init__()
        self.source = source
        self.image_type = image_type
        self.mode = mode
        self.keep_faint = keep_faint

    def run(self) -> None:
        try:
            job_id = self.source.stem
            result = process_image(
                self.source,
                OUTPUTS / job_id,
                self.image_type,
                self.mode,
                self.keep_faint,
                progress_callback=lambda value, message: self.progress.emit(value, message),
            )
            result["job_id"] = job_id
            result["source"] = str(self.source)
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


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ImageLab · 古文图像净化工作台")
        self.resize(1360, 820)
        self.source: Path | None = None
        self.preview_source: Path | None = None
        self.result: dict | None = None
        self.worker: ProcessWorker | None = None
        self.import_worker: ImportWorker | None = None
        self._alt_zoom_held = False
        self._build_ui()
        QApplication.instance().installEventFilter(self)

    def _build_ui(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget { background:#f6f4ed; color:#29271f; font-family:'Microsoft YaHei'; }
            QFrame#panel { background:#fffefa; border:1px solid #dedbd1; border-radius:7px; }
            QLabel#title { font-size:20px; font-weight:600; }
            QLabel#muted { color:#716d61; font-size:12px; }
            QPushButton { min-height:34px; padding:0 13px; border:1px solid #dedbd1; border-radius:6px; background:#fffefa; }
            QPushButton:hover { border-color:#a44a32; color:#a44a32; }
            QPushButton#primary { background:#a44a32; color:white; border-color:#a44a32; font-weight:600; }
            QComboBox { min-height:34px; border:1px solid #dedbd1; border-radius:5px; padding:0 8px; background:#fffefa; }
            QCheckBox { spacing:8px; }
            QTableWidget { background:#fffefa; border:1px solid #dedbd1; gridline-color:#eeeae0; }
            QHeaderView::section { padding:5px; background:#f0eee6; border:0; color:#716d61; font-size:11px; }
            QProgressBar { height:7px; border:0; border-radius:3px; background:#e9e6dd; }
            QProgressBar::chunk { border-radius:3px; background:#3e7662; }
        """)
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(22, 18, 22, 18)
        outer.setSpacing(14)

        header = QHBoxLayout()
        title = QLabel("碑文净化工作台")
        title.setObjectName("title")
        header.addWidget(title)
        subtitle = QLabel("保守去背景 · 笔画优先 · 可回溯")
        subtitle.setObjectName("muted")
        header.addWidget(subtitle)
        header.addStretch()
        self.import_button = QPushButton("＋ 导入图片")
        self.import_button.clicked.connect(self.import_image)
        header.addWidget(self.import_button)
        self.export_button = QPushButton("导出结果")
        self.export_button.setObjectName("primary")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_result)
        header.addWidget(self.export_button)
        outer.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_preview_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([220, 800, 280])
        outer.addWidget(splitter, 1)
        self.setCentralWidget(root)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if isinstance(event, QKeyEvent) and event.key() == Qt.Key.Key_Alt and not event.isAutoRepeat():
            self._alt_zoom_held = event.type() == QEvent.Type.KeyPress
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
        self.preview_title = QLabel("处理预览")
        self.preview_title.setStyleSheet("font-size:15px;font-weight:600")
        head.addWidget(self.preview_title)
        head.addStretch()
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
        self.cleaned_view = ImageView("净化结果")
        self.original_view.high_res_status.connect(self.status_message)
        self.preview_splitter.addWidget(self.original_view)
        self.preview_splitter.addWidget(self.cleaned_view)
        layout.addWidget(self.preview_splitter, 1)
        self.status = QLabel("请导入图片开始处理")
        self.status.setObjectName("muted")
        layout.addWidget(self.status)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QFrame(objectName="panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        heading = QLabel("处理参数")
        heading.setStyleSheet("font-size:15px;font-weight:600")
        layout.addWidget(heading)
        form = QFormLayout()
        self.type_combo = QComboBox()
        self.type_combo.addItem("古代碑文拓印", "rubbing")
        self.type_combo.addItem("古籍扫描", "book")
        self.type_combo.addItem("古代手稿扫描", "manuscript")
        self.type_combo.addItem("其他文献图像", "other")
        form.addRow("资料类型", self.type_combo)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("保守保真（推荐）", "conservative")
        self.mode_combo.addItem("平衡增强", "balanced")
        self.mode_combo.addItem("强力去杂", "strong")
        form.addRow("处理模式", self.mode_combo)
        layout.addLayout(form)
        self.keep_faint = QCheckBox("保留淡色笔画")
        self.keep_faint.setChecked(True)
        layout.addWidget(self.keep_faint)
        self.detect_cracks = QCheckBox("识别裂纹和污渍")
        self.detect_cracks.setChecked(True)
        layout.addWidget(self.detect_cracks)
        self.ocr_check = QCheckBox("OCR 辅助复核（预留）")
        self.ocr_check.setEnabled(False)
        layout.addWidget(self.ocr_check)
        layout.addSpacing(10)
        self.process_button = QPushButton("开始处理")
        self.process_button.setObjectName("primary")
        self.process_button.setEnabled(False)
        self.process_button.clicked.connect(self.process)
        layout.addWidget(self.process_button)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        layout.addSpacing(14)
        note = QLabel("多边形只用于候选区域复核，不会硬切原始笔画。")
        note.setWordWrap(True)
        note.setObjectName("muted")
        layout.addWidget(note)
        layout.addStretch()
        self.region_table = QTableWidget(0, 2)
        self.region_table.setHorizontalHeaderLabels(["区域", "置信度"])
        self.region_table.setMaximumHeight(210)
        self.region_table.cellClicked.connect(self.select_region)
        layout.addWidget(QLabel("低置信度区域"))
        layout.addWidget(self.region_table)
        return panel

    def import_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择文献图片", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp)")
        if not path:
            return
        source = Path(path)
        UPLOADS.mkdir(exist_ok=True)
        destination = UPLOADS / f"{source.stem}_{source.stat().st_size}{source.suffix.lower()}"
        self.import_button.setEnabled(False)
        self.process_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.status.setText("正在读取图片，请稍候…")
        self.import_worker = ImportWorker(source, destination)
        self.import_worker.progress.connect(self.import_progress)
        self.import_worker.completed.connect(self.import_done)
        self.import_worker.failed.connect(self.import_failed)
        self.import_worker.start()

    def import_done(self, destination: str, preview: str, original_name: str, width: int, height: int) -> None:
        self.import_button.setEnabled(True)
        self.progress_bar.setValue(100)
        self.progress_bar.hide()
        self.source = Path(destination)
        self.preview_source = Path(preview)
        self.result = None
        self.file_label.setText(original_name)
        self.original_view.set_image(self.preview_source, high_res_source=self.source)
        self.cleaned_view.pixmap = QPixmap()
        self.cleaned_view.update()
        self.process_button.setEnabled(True)
        self.export_button.setEnabled(False)
        self.regions_check.setEnabled(False)
        self.region_table.setRowCount(0)
        self.status.setText("图片已导入，请选择参数后开始处理")
        self.step_process.setEnabled(True)

    def import_progress(self, value: int, message: str) -> None:
        self.progress_bar.setValue(value)
        self.status.setText(f"{message}（{value}%）")

    def status_message(self, message: str) -> None:
        self.status.setText(message)

    def import_failed(self, message: str) -> None:
        self.import_button.setEnabled(True)
        self.progress_bar.hide()
        self.status.setText("图片导入失败")
        QMessageBox.warning(self, "无法导入", message)

    def process(self) -> None:
        if not self.source:
            return
        self.process_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.status.setText("正在处理，请稍候…")
        self.worker = ProcessWorker(self.source, self.type_combo.currentData(), self.mode_combo.currentData(), self.keep_faint.isChecked())
        self.worker.progress.connect(self.processing_progress)
        self.worker.completed.connect(self.processing_done)
        self.worker.failed.connect(self.processing_failed)
        self.worker.start()

    def processing_done(self, result: dict) -> None:
        self.result = result
        self.progress_bar.setValue(100)
        self.progress_bar.hide()
        self.process_button.setEnabled(True)
        self.export_button.setEnabled(True)
        self.cleaned_view.set_image(OUTPUTS / result["job_id"] / result["enhanced"])
        self.cleaned_view.set_regions(result["regions"])
        self.original_view.set_coordinate_size(result["width"], result["height"])
        self.original_view.set_regions(result["regions"])
        self.regions_check.setEnabled(bool(result["regions"]))
        self.status.setText(f"处理完成：{result['width']}×{result['height']}，文字保留置信度 {result['confidence']}%")
        self.step_review.setEnabled(True)
        self.populate_regions(result["regions"])

    def processing_failed(self, message: str) -> None:
        self.progress_bar.hide()
        self.process_button.setEnabled(True)
        self.status.setText("处理失败")
        QMessageBox.critical(self, "处理失败", message)

    def processing_progress(self, value: int, message: str) -> None:
        self.progress_bar.setValue(value)
        self.status.setText(f"{message}（{value}%）")

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
        self.original_view.reset_view()
        self.cleaned_view.reset_view()

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
        self.step_review.setEnabled(True)

    def export_result(self) -> None:
        if not self.result:
            return
        files = [
            ("增强灰度图", self.result["enhanced"]),
            ("文字候选图", self.result["text_mask"]),
            ("透明背景 PNG", self.result["transparent"]),
        ]
        target_dir = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not target_dir:
            return
        source_dir = OUTPUTS / self.result["job_id"]
        for label, name in files:
            try:
                shutil.copy2(source_dir / name, Path(target_dir) / name)
            except OSError as exc:
                QMessageBox.critical(self, "导出失败", f"{label}导出失败：{exc}")
                return
        self.step_export.setEnabled(True)
        self.status.setText(f"已导出 {len(files)} 个文件到 {target_dir}")


def main() -> int:
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
