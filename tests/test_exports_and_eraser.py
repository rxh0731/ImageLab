import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication
from psd_tools import PSDImage

import app.desktop as desktop
from app.exporting import export_image, read_text_image, read_layered_tiff, read_dpi, on_white, TIFF_LAYER_TAG


class ExportAndInteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = TemporaryDirectory(prefix="imagelab_exports_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.addCleanup(patch.stopall)
        patch.object(desktop, "TEMPLATES", self.root / "templates").start()
        patch.object(desktop, "PREVIEW_CACHE", self.root / "cache").start()
        self.image = Image.new("RGBA", (160, 120), (220, 160, 80, 0))
        self.image.paste((24, 36, 48, 255), (30, 20, 120, 100))
        self.image.putpixel((45, 44), (102, 99, 80, 255))  # Real lighter ink texture.
        self.image.putpixel((47, 44), (102, 99, 80, 128))  # Resampled fringe.

    def window(self):
        window = desktop.MainWindow()
        self.addCleanup(window.close)
        return window

    def test_formats_preserve_transparency_pixels_resolution_and_white_layers(self):
        for fmt in ("png", "tif", "psd", "jpg"):
            for white in ((False, True) if fmt in ("tif", "psd") else (False,)):
                with self.subTest(fmt=fmt, white=white):
                    path = self.root / f"字帖.003_{white}.{fmt}"
                    export_image(self.image, path, fmt, white, dpi=(600, 300))
                    xdpi, ydpi = read_dpi(path)
                    self.assertAlmostEqual(xdpi, 600, delta=.01)
                    self.assertAlmostEqual(ydpi, 300, delta=.01)
                    with Image.open(path) as opened:
                        self.assertEqual(opened.size, self.image.size)
                        if fmt == "tif":
                            self.assertEqual(getattr(opened, "n_frames", 1), 1)
                        if fmt == "jpg":
                            self.assertEqual(opened.mode, "RGB")
                            self.assertEqual(opened.getpixel((0, 0)), (255, 255, 255))
                        elif white:
                            np.testing.assert_array_equal(np.array(opened.convert("RGB")), np.array(on_white(self.image)))
                        elif fmt != "psd":
                            np.testing.assert_array_equal(np.array(opened), np.array(self.image))
                    if fmt in ("tif", "psd"):
                        np.testing.assert_array_equal(np.array(read_text_image(path)), np.array(self.image))
                        document = PSDImage.open(path) if fmt == "psd" else read_layered_tiff(path)
                        if white or fmt == "psd":
                            self.assertEqual([layer.name for layer in document], ["白底", path.stem] if white else [path.stem])
                            np.testing.assert_array_equal(np.array(document[-1].topil(apply_icc=False)), np.array(self.image))
                            if white:
                                self.assertTrue(np.all(np.array(document[0].topil(apply_icc=False)) == 255))
                        if fmt == "tif" and white:
                            # Photoshop requires the TIFF and layer records
                            # to share their byte order, unlike our own reader.
                            self.assertEqual(path.read_bytes()[:4], b"MM\x00*")
                            with Image.open(path) as tif:
                                blob = tif.tag_v2[TIFF_LAYER_TAG]
                                self.assertIn(b"8BIMLayr", blob)

    def test_batch_uses_selected_extension_white_layer_and_skips_existing(self):
        source_root = self.root / "input"
        source_root.mkdir()
        source = source_root / "sample.png"
        self.image.convert("RGB").save(source, dpi=(300, 300))
        for fmt in ("jpg", "png", "psd", "tif"):
            with self.subTest(fmt=fmt):
                dest = self.root / fmt
                params = {"output_format": fmt, "white_background": True}
                worker = desktop.BatchWorker([source], source_root, dest, "book", "balanced", True, params)
                results = []
                worker.completed.connect(results.append)
                worker.run()
                self.assertEqual(results[-1]["succeeded"], 1)
                self.assertTrue((dest / f"sample.{fmt}").is_file())
                self.assertEqual([p.name for p in dest.iterdir()], [f"sample.{fmt}"])
                with patch.object(desktop, "process_image") as process:
                    worker.run()
                    process.assert_not_called()
                self.assertEqual(results[-1]["skipped"], 1)

    def test_navigation_preview_and_save_without_result_never_trigger_processing(self):
        window = self.window()
        source = self.root / "page.png"
        self.image.convert("RGB").save(source)
        window.source = source
        window.output_root = self.root / "output"
        with patch.object(window, "process") as process:
            window.preview_ready(str(source), 160, 120)
            self.app.processEvents()
            self.assertTrue(window.cleaned_view.pixmap.isNull())
            self.assertIsNone(window.result)
            self.assertTrue(window.process_button.isEnabled())
            self.assertFalse(window.export_button.isEnabled())
            window.save_current()
            process.assert_not_called()

    def test_existing_other_format_is_shown_after_navigation_without_processing(self):
        window = self.window()
        source = self.root / "page.png"
        self.image.convert("RGB").save(source)
        dest = self.root / "output"
        dest.mkdir()
        export_image(self.image, dest / "page.psd", "psd", True, dpi=(600, 600))
        window.source, window.output_root = source, dest
        with patch.object(window, "process") as process:
            window.preview_ready(str(source), 160, 120)
            window.preview_worker.wait()
            self.app.processEvents()
            self.assertIsNotNone(window.result)
            self.assertIn("已存在", window.status.text())
            self.assertEqual(window.result["enhanced"], "page.psd")
            self.assertEqual(window.cleaned_view.image_dpi, 600)
            process.assert_not_called()

    def test_format_and_white_layer_options_round_trip_in_templates(self):
        window = self.window()
        window._templates["test"] = {"output_format": "psd", "white_background": True, "dpi": 600, "use_output_dpi": True}
        window.apply_selected_template("test")
        params = window._current_params()
        self.assertEqual(params["output_format"], "psd")
        self.assertTrue(params["white_background"])
        self.assertTrue(window.white_background.isEnabled())
        window.output_format.setCurrentIndex(window.output_format.findData("jpg"))
        self.assertFalse(window.white_background.isEnabled())
        self.assertIn("有损", window.format_note.text())

    def test_eraser_is_confined_to_circle_at_all_zoom_pan_and_preview_scales(self):
        window = self.window()
        path = self.root / "preview.png"
        Image.new("RGB", (100, 75), (23, 45, 67)).save(path)
        view = window.cleaned_view
        view.resize(600, 450)
        for zoom in (.5, 1.0, 8.0):
            for diameter in (12, 40, 96):
                with self.subTest(zoom=zoom, diameter=diameter):
                    original = Image.new("RGBA", (400, 300), (23, 45, 67, 255))
                    window._edited_image = original.copy()
                    window._eraser_undo.clear()
                    view.set_image(path)
                    view.set_coordinate_size(400, 300, 600)
                    view.set_view_state(view.fit_scale() * zoom, QPointF(27.3, -18.7))
                    view.eraser_size = diameter
                    rect = view._image_rect()
                    point = rect.center() + QPointF(3.7, 2.9)
                    xs = rect.left() + (np.arange(400)[None, :] + .5) * rect.width() / 400
                    ys = rect.top() + (np.arange(300)[:, None] + .5) * rect.height() / 300
                    inside = (xs - point.x()) ** 2 + (ys - point.y()) ** 2 <= (diameter / 2) ** 2
                    with patch.object(Image.Image, "save", side_effect=AssertionError("file I/O on eraser click")):
                        window.begin_erase()
                        window.erase_processed_stroke(point, point)
                        window.finish_erase()
                        result = np.array(window._edited_image)
                        np.testing.assert_array_equal(result[:, :, 3] == 0, inside)
                        np.testing.assert_array_equal(result[:, :, :3], np.array(original)[:, :, :3])
                        window.undo_erase()
                        np.testing.assert_array_equal(np.array(window._edited_image), np.array(original))

    def test_drag_is_confined_to_swept_circle_and_undo_preserves_overlaps(self):
        window = self.window()
        path = self.root / "preview.png"
        self.image.convert("RGB").save(path)
        image = Image.new("RGBA", (160, 120), (40, 50, 60, 255))
        window._edited_image = image
        view = window.cleaned_view
        view.set_image(path)
        view.set_coordinate_size(160, 120, 300)
        view.set_view_state(view.fit_scale() * 2, QPointF(-21.5, 30.2))
        view.eraser_size = 28
        rect = view._image_rect()
        start, end = rect.center() - QPointF(40, 18), rect.center() + QPointF(31, 22)
        window.begin_erase()
        window.erase_processed_stroke(start, end)
        window.finish_erase()
        after_first = np.array(image)
        window.begin_erase()
        window.erase_processed_stroke(end, end + QPointF(18, -30))
        window.finish_erase()
        window.undo_erase()
        np.testing.assert_array_equal(np.array(image), after_first)
        xs = rect.left() + (np.arange(160)[None, :] + .5) * rect.width() / 160
        ys = rect.top() + (np.arange(120)[:, None] + .5) * rect.height() / 120
        dx, dy = end.x()-start.x(), end.y()-start.y()
        f = np.clip(((xs-start.x())*dx+(ys-start.y())*dy)/(dx*dx+dy*dy),0,1)
        expected = (xs-start.x()-f*dx)**2+(ys-start.y()-f*dy)**2 <= 14**2
        np.testing.assert_array_equal(after_first[:, :, 3] == 0, expected)

    def test_atomic_export_failure_leaves_existing_file_intact(self):
        destination = self.root / "existing.png"
        destination.write_bytes(b"existing user result")
        with patch.object(Image.Image, "save", side_effect=OSError("simulated disk error")):
            with self.assertRaises(OSError):
                export_image(self.image, destination, "png")
        self.assertEqual(destination.read_bytes(), b"existing user result")
        self.assertEqual([p.name for p in self.root.iterdir()], ["existing.png"])

    def test_manual_save_exports_edited_text_in_each_format(self):
        window = self.window()
        source = self.root / "original.png"
        self.image.convert("RGB").save(source)
        work = self.root / "work"
        export_image(self.image, work / "page.tif", dpi=(600, 300))
        window.source = source
        window.result = {"output_dir": str(work), "enhanced": "page.tif", "dpi_x": 600, "dpi_y": 300}
        edited = self.image.copy()
        edited.putpixel((45, 44), (102, 99, 80, 0))
        window._edited_image = edited
        window._edited_dirty = True
        window.output_root = self.root / "saved"
        for fmt in ("tif", "png", "jpg", "psd"):
            with self.subTest(fmt=fmt):
                window.output_format.setCurrentIndex(window.output_format.findData(fmt))
                window.white_background.setChecked(fmt in ("tif", "psd"))
                window.save_current()
                saved = window.output_root / f"original.{fmt}"
                self.assertTrue(saved.is_file())
                np.testing.assert_allclose(read_dpi(saved), (600, 300), atol=.01)
                if fmt != "jpg":
                    np.testing.assert_array_equal(np.array(read_text_image(saved)), np.array(edited))
                if fmt in ("tif", "psd"):
                    document = read_layered_tiff(saved) if fmt == "tif" else PSDImage.open(saved)
                    self.assertEqual(document[-1].name, source.stem)
                self.assertTrue(window.settings_panel.isEnabled())
                self.assertTrue(window.cleaned_view.isEnabled())


if __name__ == "__main__":
    unittest.main()
