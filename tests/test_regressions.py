"""Regression checks for preview colors, ink preservation and desktop exports."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image, ImageDraw
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPointF

import app.desktop as desktop
from app.processing import _refine_faint_stroke_edges, _restore_source_strokes, process_image


class ImageLabRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = TemporaryDirectory(prefix="imagelab_regression_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source_root = self.root / "input"
        self.source_root.mkdir()
        self.cache_patch = patch.object(desktop, "PREVIEW_CACHE", self.root / "cache")
        self.template_patch = patch.object(desktop, "TEMPLATES", self.root / "templates")
        self.cache_patch.start()
        self.template_patch.start()
        self.addCleanup(self.cache_patch.stop)
        self.addCleanup(self.template_patch.stop)

    def make_source(self, name="sample.tif"):
        path = self.source_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (241, 180), (229, 205, 164))
        draw = ImageDraw.Draw(image)
        draw.rectangle((60, 30, 180, 145), fill=(27, 45, 62))
        draw.rectangle((90, 60, 150, 115), fill=(229, 205, 164))
        image.save(path, dpi=(300, 300))
        image.close()
        return path

    def test_high_resolution_loader_retains_color_and_uses_white_only_for_display(self):
        source = self.make_source()
        received, errors = [], []
        worker = desktop.ImageLoadWorker(source)
        worker.loaded.connect(received.append)
        worker.failed.connect(errors.append)
        worker.run()
        self.assertEqual(errors, [])
        self.assertEqual(received[0].pixelColor(70, 40).getRgb()[:3], (27, 45, 62))
        self.assertEqual(received[0].pixelColor(0, 0).getRgb()[:3], (229, 205, 164))
        # Check the downsampled path too, including odd image row widths.
        large = self.root / "large.tif"
        with Image.new("RGB", (1025, 1001), (25, 70, 130)) as image:
            image.save(large)
        worker = desktop.ImageLoadWorker(large, max_dimension=1024)
        worker.loaded.connect(received.append)
        worker.failed.connect(errors.append)
        worker.run()
        self.assertEqual(errors, [])
        self.assertEqual(received[-1].pixelColor(0, 0).getRgb()[:3], (25, 70, 130))
        transparent = self.root / "transparent.tif"
        with Image.new("RGBA", (13, 11), (200, 50, 10, 0)) as image:
            image.putpixel((6, 5), (27, 45, 62, 255))
            image.save(transparent)
        worker = desktop.ImageLoadWorker(transparent)
        worker.loaded.connect(received.append)
        worker.failed.connect(errors.append)
        worker.run()
        self.assertEqual(errors, [])
        self.assertEqual(received[-1].pixelColor(0, 0).getRgb()[:3], (255, 255, 255))
        self.assertEqual(received[-1].pixelColor(6, 5).getRgb()[:3], (27, 45, 62))

    def test_reconstruction_restores_ink_holes_but_keeps_glyph_counters_and_noise_removed(self):
        source = np.full((160, 180), 220, np.uint8)
        source[20:140, 25:155] = 45
        source[55:105, 65:115] = 220  # Real enclosed counter.
        source[5:8, 5:8] = 45  # Isolated speck with no retained ink support.
        keep = np.where(source < 100, 255, 0).astype(np.uint8)
        keep[30:48, 45:60] = 0  # Artificial cut through solid source ink.
        keep[5:8, 5:8] = 0
        repaired = _restore_source_strokes(keep, source)
        self.assertTrue(np.all(repaired[30:48, 45:60] == 255))
        self.assertTrue(np.all(repaired[55:105, 65:115] == 0))
        self.assertTrue(np.all(repaired[5:8, 5:8] == 0))

    def test_colored_rubbing_reconstruction_restores_lighter_ink_texture(self):
        source_rgb = np.full((180, 220, 3), (232, 181, 91), dtype=np.uint8)
        source_rgb[35:145, 35:185] = (42, 70, 48)
        source_gray = np.asarray(Image.fromarray(source_rgb).convert("L"), dtype=np.uint8)
        keep = np.where(source_gray < 110, 255, 0).astype(np.uint8)
        # Simulate a pale textured gap inside an otherwise continuous stroke.
        keep[75:105, 80:140] = 0
        repaired = _restore_source_strokes(keep, source_gray, source_rgb)
        self.assertTrue(np.all(repaired[75:105, 80:140] == 255))

    def test_colored_paper_texture_touching_stroke_is_not_restored(self):
        source_rgb = np.full((180, 220, 3), (232, 181, 91), dtype=np.uint8)
        source_rgb[35:145, 35:185] = (42, 70, 48)
        # A dark orange paper patch touches the stroke. It resembles ink in
        # brightness but remains distinguishable by its chroma.
        source_rgb[80:110, 185:205] = (140, 100, 45)
        source_gray = np.asarray(Image.fromarray(source_rgb).convert("L"), dtype=np.uint8)
        keep = np.where(source_gray < 110, 255, 0).astype(np.uint8)
        keep[75:105, 80:140] = 0  # Real stroke gap should still be repaired.
        keep[80:110, 185:205] = 0  # Paper patch must remain transparent.
        repaired = _restore_source_strokes(keep, source_gray, source_rgb)
        self.assertTrue(np.all(repaired[75:105, 80:140] == 255))
        self.assertTrue(np.all(repaired[80:110, 185:205] == 0))

    def test_textured_ink_pinholes_are_restored_without_filling_paper_spaces(self):
        paper = (232, 181, 91)
        source_rgb = np.full((240, 320, 3), paper, dtype=np.uint8)
        source_rgb[20:160, 20:200] = (35, 34, 16)
        # Ink-heavy pages drive the green-channel threshold to 70. These
        # lighter ink samples must survive even though they exceed it.
        source_rgb[40:43, 40:43] = (104, 101, 87)
        source_rgb[50:53, 50:53] = (133, 132, 114)
        source_rgb[60:63, 60:63] = paper  # Tiny real paper-colored aperture.
        source_rgb[90:120, 90:120] = paper  # Large enclosed counter.
        source_rgb[130:133, 20:60] = paper  # Open narrow stroke gap.
        source_gray = np.asarray(Image.fromarray(source_rgb).convert("L"), dtype=np.uint8)
        keep = np.where(source_gray < 70, 255, 0).astype(np.uint8)
        before_rgb = source_rgb.copy()
        repaired = _restore_source_strokes(keep, source_gray, source_rgb)
        self.assertTrue(np.all(repaired[40:43, 40:43] == 255))
        self.assertTrue(np.all(repaired[50:53, 50:53] == 255))
        self.assertTrue(np.all(repaired[60:63, 60:63] == 0))
        self.assertTrue(np.all(repaired[90:120, 90:120] == 0))
        self.assertTrue(np.all(repaired[130:133, 20:60] == 0))
        np.testing.assert_array_equal(source_rgb, before_rgb)

    def test_brown_ink_cuts_are_restored_without_recovering_exterior_paper(self):
        paper = (232, 181, 91)
        source_rgb = np.full((240, 320, 3), paper, dtype=np.uint8)
        source_rgb[20:190, 20:220] = (65, 56, 2)
        # Brown source ink exceeds the neutral-ink chroma ceiling. Cover
        # both a substantial interior hole and a cut open to the outer edge.
        source_rgb[50:70, 60:80] = (122, 90, 20)
        source_rgb[80:90, 195:220] = (122, 90, 20)
        source_rgb[110:140, 220:240] = (140, 100, 45)  # Touching paper stain.
        source_rgb[120:155, 80:115] = paper  # Genuine character counter.
        source_rgb[160:163, 25:70] = paper  # Thin open paper gap.
        source_rgb[205:208, 260:263] = (122, 90, 20)  # Isolated grain.
        source_rgb[100:120, 140:160] = paper
        source_rgb[108:111, 148:151] = (122, 90, 20)  # Grain inside a small counter.
        source_gray = np.asarray(Image.fromarray(source_rgb).convert("L"))
        keep = np.where(source_gray < 70, 255, 0).astype(np.uint8)
        before = source_rgb.copy()
        repaired = _restore_source_strokes(keep, source_gray, source_rgb)
        self.assertTrue(np.all(repaired[50:70, 60:80] == 255))
        self.assertTrue(np.all(repaired[80:90, 195:210] == 255))
        self.assertTrue(np.all(repaired[110:140, 220:240] == 0))
        self.assertTrue(np.all(repaired[120:155, 80:115] == 0))
        self.assertTrue(np.all(repaired[160:163, 25:70] == 0))
        self.assertTrue(np.all(repaired[205:208, 260:263] == 0))
        self.assertTrue(np.all(repaired[100:120, 140:160] == 0))
        self.assertTrue(np.all(repaired[keep > 0] == 255))
        np.testing.assert_array_equal(source_rgb, before)

    def test_faint_edge_refinement_retains_source_fringe_and_rejects_paper(self):
        source = np.full((260, 320, 3), (232, 181, 91), np.uint8)
        source[50:210, 50:230] = (120, 100, 60)
        source[50:210, 228:230] = (163, 147, 100)  # Pale outer ink fringe.
        source[90:110, 230:240] = (140, 100, 45)  # Touching orange pattern.
        source[120:150, 110:140] = (232, 181, 91)  # Actual counter.
        source[170:172, 50:80] = (232, 181, 91)  # Open paper gap.
        source[60:65, 232:234] = (120, 100, 60)  # Nearby isolated dark grain.
        keep = np.zeros(source.shape[:2], np.uint8)
        keep[52:208, 52:228] = 255
        keep[120:150, 110:140] = 0
        keep[170:172, 50:80] = 0
        before = source.copy()
        repaired = _refine_faint_stroke_edges(keep.copy(), source)
        self.assertTrue(np.all(repaired[55:85, 228:230] == 255))
        self.assertTrue(np.all(repaired[90:110, 230:240] == 0))
        self.assertTrue(np.all(repaired[120:150, 110:140] == 0))
        self.assertTrue(np.all(repaired[170:172, 50:80] == 0))
        self.assertTrue(np.all(repaired[60:65, 232:234] == 0))
        self.assertTrue(np.all(repaired[keep > 0] == 255))
        self.assertEqual(set(np.unique(repaired)), {0, 255})
        np.testing.assert_array_equal(source, before)

    def test_edge_refinement_does_not_change_clear_dark_strokes_or_blank_pages(self):
        source = np.full((260, 320, 3), (232, 181, 91), np.uint8)
        source[50:210, 50:230] = (25, 20, 8)
        keep = np.zeros(source.shape[:2], np.uint8)
        keep[52:208, 52:228] = 255
        np.testing.assert_array_equal(_refine_faint_stroke_edges(keep.copy(), source), keep)
        blank = np.zeros_like(keep)
        np.testing.assert_array_equal(_refine_faint_stroke_edges(blank.copy(), source), blank)

    def test_dark_stroke_grain_repairs_holes_and_open_cuts_without_restoring_paper(self):
        source = np.full((300, 360, 3), (232, 181, 91), np.uint8)
        texture = np.random.default_rng(640).integers(0, 100, size=(220, 240))
        stroke = np.full((220, 240, 3), (35, 33, 25), np.uint8)
        stroke[texture >= 55] = (70, 67, 52)
        stroke[texture >= 85] = (125, 120, 105)
        source[30:250, 30:270] = stroke
        # Dark median ink used to skip edge repair, despite pale scan grain
        # cutting holes inside it and notches open to the outer boundary.
        source[80:94, 90:104] = (100, 96, 80)
        source[135:155, 30:44] = (100, 96, 80)
        source[170:190, 30:44] = (232, 181, 91)  # Real open paper gap.
        source[110:140, 160:190] = (232, 181, 91)  # Real counter.
        source[150:160, 270:280] = (140, 100, 45)  # Touching orange paper.
        source[55:59, 274:278] = (100, 96, 80)  # Detached dark grain.
        source[210, 270:282] = (100, 96, 80)  # Thin touching background line.
        source[220:223, 270:290] = (100, 96, 80)  # Several-pixel scan hairline.
        keep = np.zeros(source.shape[:2], np.uint8)
        keep[30:250, 30:270] = 255
        keep[80:94, 90:104] = 0
        keep[135:155, 30:44] = 0
        keep[170:190, 30:44] = 0
        keep[110:140, 160:190] = 0
        before = source.copy()
        repaired = _refine_faint_stroke_edges(keep.copy(), source)
        self.assertTrue(np.all(repaired[80:94, 90:104] == 255))
        self.assertTrue(np.all(repaired[135:155, 30:44] == 255))
        self.assertTrue(np.all(repaired[170:190, 30:44] == 0))
        self.assertTrue(np.all(repaired[110:140, 160:190] == 0))
        self.assertTrue(np.all(repaired[150:160, 270:280] == 0))
        self.assertTrue(np.all(repaired[55:59, 274:278] == 0))
        self.assertTrue(np.all(repaired[210, 270:282] == 0))
        self.assertTrue(np.all(repaired[220:223, 270:290] == 0))
        self.assertTrue(np.all(repaired[keep > 0] == 255))
        self.assertEqual(set(np.unique(repaired)), {0, 255})
        np.testing.assert_array_equal(source, before)

    def test_textured_faint_tip_repairs_deep_notch_but_preserves_paper_and_components(self):
        source = np.full((280, 340, 3), (232, 181, 91), np.uint8)
        source[::2] = (250, 235, 170)  # Alternating bare paper and orange pattern.
        texture = np.random.default_rng(331).integers(0, 100, size=(200, 220))
        stroke = np.full((200, 220, 3), (90, 75, 25), np.uint8)
        stroke[texture >= 55] = (120, 100, 60)
        stroke[texture >= 80] = (160, 145, 100)
        stroke[texture >= 98] = (130, 100, 10)
        source[30:230, 30:250] = stroke
        # A 14px-deep cut open to the exterior exceeds the 3px edge band.
        source[175:195, 236:250] = (139, 102, 32)
        source[181:183, 241:243] = (247, 230, 174)  # Near-white ink highlight inside the cut.
        source[185:187, 246:248] = (232, 181, 91)  # Real paper aperture must remain open.
        source[190:192, 239:241] = (255, 250, 230)  # Bare white paper aperture.
        source[80:110, 236:250] = (232, 181, 91)  # Genuine open paper gap.
        source[130:150, 100:130] = (232, 181, 91)  # Actual counter.
        # A separate paper/ink speck must not acquire a hull joining it to
        # the main stroke; hulls are computed independently per component.
        source[180:184, 254:258] = (139, 102, 32)
        keep = np.zeros(source.shape[:2], np.uint8)
        keep[30:230, 30:250] = 255
        keep[175:195, 236:250] = 0
        keep[80:110, 236:250] = 0
        keep[130:150, 100:130] = 0
        before = source.copy()
        repaired = _refine_faint_stroke_edges(keep.copy(), source)
        expected = np.full((20, 14), 255, np.uint8)
        expected[10:12, 10:12] = 0
        expected[15:17, 3:5] = 0
        np.testing.assert_array_equal(repaired[175:195, 236:250], expected)
        self.assertTrue(np.all(repaired[80:110, 236:250] == 0))
        self.assertTrue(np.all(repaired[130:150, 100:130] == 0))
        self.assertTrue(np.all(repaired[180:184, 254:258] == 0))
        self.assertTrue(np.all(repaired[keep > 0] == 255))
        np.testing.assert_array_equal(source, before)

    def test_faint_edge_refinement_is_bounded_and_consistent_at_tile_boundaries(self):
        source = np.full((1100, 1100, 3), (232, 181, 91), np.uint8)
        source[30:1070, 200:515] = (120, 100, 60)
        keep = np.zeros(source.shape[:2], np.uint8)
        keep[30:1070, 200:500] = 255
        repaired = _refine_faint_stroke_edges(keep.copy(), source)
        self.assertTrue(np.all(repaired[80:1020, 500:503] == 255))
        # More unretained source ink exists past the fixed inspection band;
        # it must not cause iterative outward growth or depend on tile order.
        self.assertTrue(np.all(repaired[:, 504:] == 0))
        for row in [511, 512, 513, 1023, 1024]:
            np.testing.assert_array_equal(repaired[row, 495:520], repaired[100, 495:520])

    def test_single_and_batch_exports_are_directly_inside_output_directory(self):
        source = self.make_source("subfolder/first.tif")
        result = process_image(source, self.root / "work", "book", "balanced", True,
                               params={"dpi": 300}, preview_dir=self.root / "preview")
        result["output_dir"] = str(self.root / "work")
        window = desktop.MainWindow()
        self.addCleanup(window.close)
        self.addCleanup(self.qt_app.removeEventFilter, window)
        window.source_root = self.source_root
        window.source = source
        window.output_root = self.root / "manual"
        window.output_root.mkdir()
        unrelated = window.output_root / "first_text.tif"
        unrelated.write_bytes(b"existing user file")
        window.result = result
        window.save_current()
        self.assertEqual(sorted(p.name for p in window.output_root.iterdir()),
                         ["first.tif", "first_text.tif"])
        self.assertEqual(unrelated.read_bytes(), b"existing user file")
        with Image.open(window.output_root / "first.tif") as saved, Image.open(source) as original:
            self.assertEqual(saved.n_frames, 1)
            self.assertEqual(saved.mode, "RGBA")
            self.assertEqual(saved.getpixel((0, 0))[3], 0)
            data = np.array(saved)
            keep = data[:, :, 3] > 0
            self.assertTrue(np.any(keep))
            np.testing.assert_array_equal(data[:, :, :3][keep], np.array(original)[keep])
        second = self.make_source("another/second.tif")
        (self.root / "batch").mkdir()
        unrelated = self.root / "batch" / "first_white.tif"
        unrelated.write_bytes(b"another existing user file")
        summaries = []
        worker = desktop.BatchWorker([source, second], self.source_root, self.root / "batch",
                                     "book", "balanced", True, {"dpi": 600, "use_output_dpi": True})
        worker.completed.connect(summaries.append)
        worker.run()
        self.assertEqual(summaries[0]["succeeded"], 2)
        self.assertEqual(summaries[0]["failed"], [])
        self.assertEqual(sorted(p.name for p in (self.root / "batch").iterdir()),
                         ["first.tif", "first_white.tif", "second.tif"])
        self.assertEqual(unrelated.read_bytes(), b"another existing user file")
        with Image.open(self.root / "batch" / "first.tif") as saved:
            self.assertEqual(saved.n_frames, 1)
            self.assertEqual(saved.mode, "RGBA")
            self.assertEqual(saved.size, (482, 360))
            self.assertEqual(saved.info["dpi"], (600.0, 600.0))

    def test_flat_batch_does_not_overwrite_another_input_with_same_name(self):
        source = self.make_source("first/sample.tif")
        duplicate = self.make_source("second/sample.tif")
        summaries = []
        worker = desktop.BatchWorker([source, duplicate], self.source_root, self.root / "batch",
                                     "book", "conservative", True, {"dpi": 300})
        worker.completed.connect(summaries.append)
        worker.run()
        self.assertEqual(summaries[0]["succeeded"], 1)
        self.assertEqual(summaries[0]["skipped"], 1)
        self.assertEqual(summaries[0]["failed"], [])
        self.assertEqual([p.name for p in (self.root / "batch").iterdir()], ["sample.tif"])

    def test_skipping_unused_preview_preserves_export_pixels_and_resolution(self):
        source = self.make_source()
        params = {"dpi": 600, "use_output_dpi": True}
        normal = process_image(source, self.root / "normal", "book", "balanced", True,
                               params=params, preview_dir=self.root / "shown_preview")
        with patch.object(Image.Image, "thumbnail", side_effect=AssertionError("unused preview generated")):
            batch = process_image(source, self.root / "batch", "book", "balanced", True,
                                  params=params, preview_dir=self.root / "unused_preview", generate_preview=False)
        self.assertFalse((self.root / "unused_preview").exists())
        self.assertEqual(Path(batch["preview_path"]), self.root / "batch" / "sample.tif")
        with Image.open(self.root / "normal" / normal["enhanced"]) as expected, \
                Image.open(self.root / "batch" / batch["enhanced"]) as actual:
            np.testing.assert_array_equal(np.array(actual), np.array(expected))
            self.assertEqual(actual.info["dpi"], expected.info["dpi"])
            self.assertEqual(actual.size, (482, 360))
            self.assertEqual(actual.mode, "RGBA")
            self.assertEqual(actual.n_frames, 1)

    def test_batch_skips_existing_outputs_without_processing_or_overwriting(self):
        source = self.make_source("existing.jpg")
        new_source = self.make_source("new.jpg")
        output = self.root / "batch"
        output.mkdir()
        existing = output / "existing.tif"
        existing.write_bytes(b"leave the existing file untouched")
        before = existing.read_bytes(), existing.stat().st_mtime_ns
        summaries, progress = [], []
        worker = desktop.BatchWorker([source, new_source], self.source_root, output,
                                     "book", "conservative", True, {"dpi": 300})
        worker.completed.connect(summaries.append)
        worker.overall_progress.connect(lambda current, total, name: progress.append((current, total)))
        with patch.object(desktop, "process_image", wraps=process_image) as process:
            worker.run()
        self.assertEqual(process.call_count, 1)
        self.assertEqual(process.call_args.args[0], new_source)
        self.assertFalse(process.call_args.kwargs["generate_preview"])
        self.assertEqual(summaries[0]["succeeded"], 1)
        self.assertEqual(summaries[0]["skipped"], 1)
        self.assertEqual(summaries[0]["failed"], [])
        self.assertEqual(progress, [(1, 2), (2, 2)])
        self.assertEqual((existing.read_bytes(), existing.stat().st_mtime_ns), before)
        self.assertTrue((output / "new.tif").is_file())
        # Re-running a completed batch does no image processing at all.
        worker = desktop.BatchWorker([source, new_source], self.source_root, output,
                                     "book", "conservative", True, {"dpi": 600})
        worker.completed.connect(summaries.append)
        with patch.object(desktop, "process_image") as process:
            worker.run()
        process.assert_not_called()
        self.assertEqual(summaries[-1]["succeeded"], 0)
        self.assertEqual(summaries[-1]["skipped"], 2)
        self.assertEqual(summaries[-1]["failed"], [])
        self.assertEqual((existing.read_bytes(), existing.stat().st_mtime_ns), before)

    def test_processing_refuses_to_overwrite_the_source_tiff(self):
        source = self.make_source()
        original_bytes = source.read_bytes()
        with self.assertRaises(ValueError):
            process_image(source, source.parent, "book", "balanced", True)
        self.assertEqual(source.read_bytes(), original_bytes)

    def test_ctrl_eraser_makes_processed_pixels_transparent_and_save_uses_edit(self):
        source = self.make_source()
        result = process_image(source, self.root / "work", "book", "conservative", True,
                               params={"dpi": 300}, preview_dir=self.root / "preview")
        result["output_dir"] = str(self.root / "work")
        window = desktop.MainWindow()
        self.addCleanup(window.close)
        self.addCleanup(self.qt_app.removeEventFilter, window)
        window.source_root = self.source_root
        window.source = source
        window.output_root = self.root / "save"
        window.processing_done(result)
        window.cleaned_view.resize(420, 300)
        self.qt_app.processEvents()
        edited = window._ensure_edited_image()
        self.assertIsNotNone(edited)
        alpha = np.array(edited.getchannel("A"))
        ys, xs = np.where(alpha > 0)
        self.assertGreater(len(xs), 0)
        x, y = int(xs[len(xs) // 2]), int(ys[len(ys) // 2])
        rect = window.cleaned_view._image_rect()
        point = QPointF(rect.left() + x / max(1, edited.width - 1) * rect.width(),
                        rect.top() + y / max(1, edited.height - 1) * rect.height())
        window.erase_processed_stroke(point, point)
        self.assertEqual(edited.getpixel((x, y))[3], 0)
        window.finish_erase()
        self.assertTrue(window._edited_dirty)
        self.assertIsNone(window._edited_result_path)  # No whole-file write per click.

        window.output_root.mkdir()
        destination = window.output_root / "sample.tif"
        destination.write_bytes(b"keep this file")
        before = destination.read_bytes()
        with patch.object(desktop.QMessageBox, "question", return_value=desktop.QMessageBox.StandardButton.No):
            window.save_current()
        self.assertEqual(destination.read_bytes(), before)
        with patch.object(desktop.QMessageBox, "question", return_value=desktop.QMessageBox.StandardButton.Yes):
            window.save_current()
        self.assertNotEqual(destination.read_bytes(), before)
        with Image.open(destination) as saved:
            self.assertEqual(saved.mode, "RGBA")
            self.assertEqual(saved.n_frames, 1)

    def test_eraser_size_changes_cursor_and_ctrl_z_restores_alpha_snapshot(self):
        source = self.make_source()
        result = process_image(source, self.root / "work", "book", "conservative", True,
                               params={"dpi": 300}, preview_dir=self.root / "preview")
        result["output_dir"] = str(self.root / "work")
        window = desktop.MainWindow()
        self.addCleanup(window.close)
        self.addCleanup(self.qt_app.removeEventFilter, window)
        window.source = source
        window.output_root = self.root / "save"
        window.processing_done(result)
        before = np.array(window._ensure_edited_image().getchannel("A"))
        old_size = window.cleaned_view.eraser_size
        window.cleaned_view.set_eraser_modifier_active(True)
        window.cleaned_view.adjust_eraser_size(120)
        self.assertGreater(window.cleaned_view.eraser_size, old_size)
        self.assertEqual(window.cleaned_view.cursor().shape(), desktop.Qt.CursorShape.BlankCursor)
        window.begin_erase()
        image = window._edited_image
        ys, xs = np.where(before > 0)
        x, y = int(xs[len(xs) // 2]), int(ys[len(ys) // 2])
        rect = window.cleaned_view._image_rect()
        point = QPointF(rect.left() + x / max(1, image.width - 1) * rect.width(),
                        rect.top() + y / max(1, image.height - 1) * rect.height())
        window.erase_processed_stroke(point, point)
        window.finish_erase()
        self.assertFalse(np.array_equal(np.array(image.getchannel("A")), before))
        window.undo_erase()
        np.testing.assert_array_equal(np.array(image.getchannel("A")), before)

    def test_existing_result_preview_is_loaded_and_marked_until_manual_refresh(self):
        source = self.make_source()
        existing_dir = self.root / "existing"
        result = process_image(source, existing_dir, "book", "conservative", True,
                               params={"dpi": 600, "use_output_dpi": True}, preview_dir=self.root / "existing_preview")
        window = desktop.MainWindow()
        self.addCleanup(window.close)
        self.addCleanup(self.qt_app.removeEventFilter, window)
        window.source_root = self.source_root
        window.source = source
        window.output_root = existing_dir
        window.files = [source]
        window.current_index = 0
        window._update_page_controls()
        window.original_view.set_image(source, high_res_source=source)
        window.original_view.set_coordinate_size(241, 180, 300)
        window._existing_result_source = existing_dir / "sample.tif"
        window.existing_result_preview_ready(str(self.root / "existing_preview" / "sample_preview.png"),
                                              result["width"], result["height"])
        self.assertIsNotNone(window.result)
        self.assertEqual(window.result["output_dir"], str(existing_dir))
        self.assertIn("已存在", window.status.text())
        self.assertTrue(window.process_button.isEnabled())
        self.assertAlmostEqual(window.original_view.image_dpi, 300)
        self.assertAlmostEqual(window.cleaned_view.image_dpi, 600)

    def test_output_resolution_switch_preserves_source_when_disabled(self):
        source = self.make_source()
        unchanged = process_image(
            source,
            self.root / "unchanged",
            "book",
            "conservative",
            True,
            params={"dpi": 600, "use_output_dpi": False},
        )
        with Image.open(self.root / "unchanged" / "sample.tif") as saved:
            self.assertEqual(saved.size, (241, 180))
            self.assertEqual(saved.info["dpi"], (300.0, 300.0))
        self.assertEqual(unchanged["width"], 241)
        self.assertEqual(unchanged["height"], 180)
        self.assertFalse(unchanged["used_output_dpi"])

        resampled = process_image(
            source,
            self.root / "resampled",
            "book",
            "conservative",
            True,
            params={"dpi": 600, "use_output_dpi": True},
        )
        with Image.open(self.root / "resampled" / "sample.tif") as saved:
            self.assertEqual(saved.size, (482, 360))
            self.assertEqual(saved.info["dpi"], (600.0, 600.0))
        self.assertEqual(resampled["width"], 482)
        self.assertEqual(resampled["height"], 360)
        self.assertTrue(resampled["used_output_dpi"])

    def test_jump_control_targets_requested_index(self):
        window = desktop.MainWindow()
        self.addCleanup(window.close)
        self.addCleanup(self.qt_app.removeEventFilter, window)
        window.files = [self.make_source("one.tif"), self.make_source("two.tif"), self.make_source("three.tif")]
        window.current_index = 0
        window._update_page_controls()
        window.jump_spin.setValue(3)
        with patch.object(window, "load_current_file") as load:
            window.jump_to_file()
        self.assertEqual(window.current_index, 2)
        self.assertFalse(hasattr(window, "_auto_process_after_preview"))
        load.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
