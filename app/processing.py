from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import cv2
from PIL import Image, ImageOps


@dataclass(frozen=True)
class ProcessingPreset:
    background_radius: int
    contrast: float
    threshold_bias: int
    denoise_radius: float


PRESETS = {
    "rubbing": ProcessingPreset(31, 1.28, 5, 0.65),
    "book": ProcessingPreset(41, 1.18, 3, 0.8),
    "manuscript": ProcessingPreset(25, 1.10, -2, 0.35),
    "other": ProcessingPreset(31, 1.15, 0, 0.55),
}


def _normalize(gray: Image.Image) -> Image.Image:
    arr = np.asarray(gray, dtype=np.uint8)
    low, high = np.percentile(arr, (1, 99))
    if high <= low:
        return gray.convert("L")
    out = cv2.convertScaleAbs(arr, alpha=255.0 / (high - low), beta=-low * 255.0 / (high - low))
    return Image.fromarray(out, mode="L")


def _repair_stroke_density(ink: np.ndarray) -> np.ndarray:
    """Restore small holes inside retained strokes without filling glyph counters."""
    # The raw threshold can leave narrow background-colored cuts inside a
    # solid ink stroke. A single 9x9 closing bridges those cuts while keeping
    # the larger counters of a glyph open.
    closed = cv2.morphologyEx(
        ink,
        cv2.MORPH_CLOSE,
        np.ones((9, 9), dtype=np.uint8),
        iterations=1,
    )
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return closed
    for index, contour in enumerate(contours):
        parent = int(hierarchy[0][index][3])
        if parent < 0:
            continue
        hole_area = cv2.contourArea(contour)
        parent_area = max(1.0, cv2.contourArea(contours[parent]))
        # Small texture gaps inside a stroke are filled. Large enclosed glyph
        # counters such as 口 remain open and continue to show the source.
        if hole_area <= min(1500.0, max(64.0, parent_area * 0.08)):
            cv2.drawContours(closed, [contour], -1, 255, thickness=cv2.FILLED)
    return closed


def _clean_ink_mask(mask_arr: np.ndarray, width: int, height: int, mode: str) -> np.ndarray:
    """Remove isolated salt-and-pepper texture without hard-cropping real strokes."""
    ink = np.where(mask_arr < 128, 255, 0).astype(np.uint8)
    if mode == "conservative":
        return _repair_stroke_density(ink)
    # Analyze huge masks at a bounded working size, then map the keep mask back
    # to the original grid. Final output pixels remain full resolution.
    analysis_scale = min(1.0, 3200.0 / max(width, height))
    if analysis_scale < 1.0:
        analysis_size = (max(1, int(width * analysis_scale)), max(1, int(height * analysis_scale)))
        ink = cv2.resize(ink, analysis_size, interpolation=cv2.INTER_AREA)
        ink = np.where(ink > 96, 255, 0).astype(np.uint8)
    if mode == "balanced":
        # Background texture is often connected to a character by a faint
        # line, so area-only filtering cannot separate it reliably. Keep the
        # thick core of each stroke instead: small specks do not have enough
        # distance from the background to form a core, while detached dots
        # with a real stroke core remain eligible for source-pixel passthrough.
        distance = cv2.distanceTransform((ink > 0).astype(np.uint8), cv2.DIST_L2, 5)
        positive = distance[distance > 0]
        if positive.size:
            core_width = float(np.clip(np.percentile(positive, 50), 3.0, 12.0))
        else:
            core_width = 3.0
        core = np.where(distance >= core_width, 255, 0).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
        min_core_area = max(12, int(core.shape[0] * core.shape[1] * 0.0000002))
        keep = np.zeros_like(core)
        for index in range(1, count):
            if int(stats[index, cv2.CC_STAT_AREA]) >= min_core_area:
                keep[labels == index] = 255
        # At very large source sizes, a 3200px analysis level can reduce
        # legitimate character strokes to one-pixel lines. If the thickness
        # pass would discard nearly everything, retain connected candidates
        # instead of returning an almost blank page.
        ink_area = int(np.count_nonzero(ink))
        keep_area = int(np.count_nonzero(keep))
        if ink_area and keep_area < ink_area * 0.25:
            count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
            min_candidate_area = max(20, int(ink.shape[0] * ink.shape[1] * 0.0000002))
            keep = np.zeros_like(ink)
            for index in range(1, count):
                area = int(stats[index, cv2.CC_STAT_AREA])
                component_width = int(stats[index, cv2.CC_STAT_WIDTH])
                component_height = int(stats[index, cv2.CC_STAT_HEIGHT])
                if area >= min_candidate_area or (
                    max(component_width, component_height) >= 20
                    and area >= max(3, min_candidate_area // 4)
                ):
                    keep[labels == index] = 255
        if analysis_scale < 1.0:
            keep = cv2.resize(keep, (width, height), interpolation=cv2.INTER_NEAREST)
        return _repair_stroke_density(keep)
    # The strong path is allowed to use a light median cleanup after the
    # conservative core selection above.
    if mode == "strong":
        ink = cv2.medianBlur(ink, 3)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    min_area = max(8, int(width * height * (0.00000010 if mode == "balanced" else 0.00000055)))
    keep = np.zeros_like(ink)
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        w = int(stats[index, cv2.CC_STAT_WIDTH])
        h = int(stats[index, cv2.CC_STAT_HEIGHT])
        # Long strokes are retained even when fragmented into a small area.
        if area >= min_area or (max(w, h) >= 20 and area >= max(3, min_area // 4)):
            keep[labels == index] = 255
    if analysis_scale < 1.0:
        keep = cv2.resize(keep, (width, height), interpolation=cv2.INTER_NEAREST)
    return _repair_stroke_density(keep)


def _stroke_edge_band(ink_mask: np.ndarray, enhanced_arr: np.ndarray, threshold: int, mode: str) -> np.ndarray:
    """Keep antialiased pixels immediately around retained strokes."""
    if mode == "conservative":
        return ink_mask
    # Do not let white-background cleanup cut through the antialiased fringe.
    # Balanced mode is deliberately wider because its priority is preserving
    # the exact source stroke edge, including faint pixels between samples.
    radius = 2 if mode == "balanced" else 1
    kernel_size = radius * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    band = cv2.dilate(ink_mask, kernel, iterations=1)
    edge_limit = min(235, max(threshold + (64 if mode == "balanced" else 56), 190))
    edge_pixels = enhanced_arr <= edge_limit
    return np.where((ink_mask > 0) | ((band > 0) & edge_pixels), 255, 0).astype(np.uint8)


def process_image(
    source: Path,
    output_dir: Path,
    image_type: str,
    mode: str,
    keep_faint: bool,
    params: dict | None = None,
    preview_dir: Path | None = None,
    progress_callback=None,
) -> dict:
    def progress(value: int, message: str) -> None:
        if progress_callback:
            progress_callback(value, message)

    progress(5, "读取图片")
    preset = PRESETS.get(image_type, PRESETS["other"])
    if mode == "balanced":
        preset = ProcessingPreset(preset.background_radius, preset.contrast + 0.10, preset.threshold_bias + 2, preset.denoise_radius + 0.2)
    elif mode == "strong":
        preset = ProcessingPreset(preset.background_radius + 10, preset.contrast + 0.18, preset.threshold_bias + 5, preset.denoise_radius + 0.5)
    if params:
        preset = ProcessingPreset(
            max(3, int(params.get("background_radius", preset.background_radius))),
            max(0.5, min(2.0, float(params.get("contrast", preset.contrast)))),
            int(params.get("threshold_bias", preset.threshold_bias)),
            max(0.0, min(4.0, float(params.get("denoise_radius", preset.denoise_radius)))),
        )
    output_dpi = 300
    if params and "dpi" in params:
        output_dpi = max(72, min(2400, int(params["dpi"])))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        with Image.open(source) as opened:
            original_size = opened.size
            source_dpi_info = opened.info.get("dpi")
            original_rgb = opened.convert("RGB").copy()
            original = ImageOps.grayscale(original_rgb)
    source_rgb_arr = np.asarray(original_rgb, dtype=np.uint8).copy()
    source_dpi_x = 300.0
    if isinstance(source_dpi_info, (tuple, list)) and source_dpi_info:
        try:
            source_dpi_x = float(source_dpi_info[0])
        except (TypeError, ValueError):
            source_dpi_x = 300.0
    elif source_dpi_info is not None:
        try:
            source_dpi_x = float(source_dpi_info)
        except (TypeError, ValueError):
            source_dpi_x = 300.0
    if not np.isfinite(source_dpi_x) or source_dpi_x <= 0:
        source_dpi_x = 300.0
    progress(18, "规范化亮度")
    normalized = _normalize(original)
    norm_arr = np.asarray(normalized, dtype=np.uint8)
    height, width = norm_arr.shape
    # Estimate the low-frequency background on a smaller image, then upsample it.
    # This is substantially faster than a full-resolution large-radius blur.
    shrink = max(1, int(round(min(width, height) / 900)))
    small_size = (max(1, width // shrink), max(1, height // shrink))
    small = cv2.resize(norm_arr, small_size, interpolation=cv2.INTER_AREA)
    sigma = max(1.0, preset.background_radius / shrink)
    background_small = cv2.GaussianBlur(small, (0, 0), sigmaX=sigma, sigmaY=sigma)
    bg_arr = cv2.resize(background_small.astype(np.uint8), (width, height), interpolation=cv2.INTER_LINEAR)
    progress(42, "校正背景")
    # Divide by the estimated background so uneven paper/stone tone becomes a
    # white field. A floor prevents dark stains from amplifying sensor noise.
    bg_safe = np.maximum(bg_arr, 32).astype(np.uint8)
    corrected_arr = cv2.divide(norm_arr, bg_safe, scale=255.0, dtype=cv2.CV_8U)
    # Keep the brightest background samples at white while retaining dark ink.
    background_level = float(np.percentile(corrected_arr, 98.5))
    if background_level > 1:
        corrected_arr = cv2.convertScaleAbs(corrected_arr, alpha=255.0 / background_level)
    if preset.denoise_radius > 0:
        corrected_arr = cv2.GaussianBlur(corrected_arr, (0, 0), sigmaX=preset.denoise_radius, sigmaY=preset.denoise_radius)
    if mode == "strong":
        corrected_arr = cv2.medianBlur(corrected_arr, 3)
    enhanced_arr = cv2.convertScaleAbs(corrected_arr, alpha=preset.contrast, beta=128.0 * (1.0 - preset.contrast))
    enhanced = Image.fromarray(enhanced_arr, mode="L")
    progress(62, "生成文字保留掩膜")

    # Preserve faint strokes by using a conservative threshold and keeping the grayscale source.
    threshold = 128 + preset.threshold_bias
    mask_arr = np.asarray(enhanced, dtype=np.uint8)
    if keep_faint:
        threshold -= 9
    if mode == "balanced":
        # Keep a wider candidate range so pale antialiased parts and small
        # detached dots remain source-pixel passthrough candidates.
        threshold += 24
    elif mode == "strong":
        threshold += 4
    raw_mask = np.where(mask_arr < threshold, 35, 255).astype(np.uint8)
    ink_mask = _clean_ink_mask(raw_mask, width, height, mode)
    display_mask = _stroke_edge_band(ink_mask, enhanced_arr, threshold, mode)
    # The output is a background removal operation. Every retained text pixel
    # is copied from the original RGB source; only pixels outside the mask are
    # replaced by white or transparent pixels.
    retained = display_mask > 0
    cleaned_display = np.where(retained[..., None], source_rgb_arr, 255).astype(np.uint8)
    cleaned_display = Image.fromarray(cleaned_display, mode="RGB")
    alpha = Image.fromarray(np.where(retained, 255, 0).astype(np.uint8), mode="L")
    source_rgba = Image.fromarray(np.dstack((source_rgb_arr, np.asarray(alpha, dtype=np.uint8))), mode="RGBA")
    transparent = source_rgba

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    for stale_name in (
        f"{stem}_white.tif",
        f"{stem}_text.tif",
        f"{stem}_transparent.png",
        f"{stem}_enhanced.png",
        f"{stem}_text-mask.png",
        f"{stem}_enhanced_preview.png",
    ):
        (output_dir / stale_name).unlink(missing_ok=True)
    layered_path = output_dir / f"{stem}.tif"
    preview_root = preview_dir or (output_dir.parent / ".imagelab_previews" / output_dir.name)
    preview_root.mkdir(parents=True, exist_ok=True)
    (preview_root / f"{stem}_text-mask.png").unlink(missing_ok=True)
    preview_path = preview_root / f"{stem}_preview.png"
    progress(76, "准备输出图层")
    # The requested DPI controls the actual raster size. Missing source DPI is
    # treated as 300 dpi so the default setting preserves the original pixels.
    resize_factor = output_dpi / source_dpi_x
    output_size = (
        max(1, int(round(original_size[0] * resize_factor))),
        max(1, int(round(original_size[1] * resize_factor))),
    )
    if output_size != original_size:
        cleaned_display = cleaned_display.resize(output_size, Image.Resampling.LANCZOS)
        transparent = transparent.resize(output_size, Image.Resampling.LANCZOS)
    progress(90, "保存单个 TIFF 图层文件")
    # Page 0 is an immediately visible white-background composite for ordinary
    # image viewers. Page 1 is the transparent text layer, retaining source
    # ink pixels and alpha outside the retained mask for layer-aware editors.
    cleaned_display.save(
        layered_path,
        format="TIFF",
        save_all=True,
        append_images=[transparent],
        compression="tiff_lzw",
        dpi=(output_dpi, output_dpi),
    )
    preview = cleaned_display.copy()
    preview.thumbnail((6000, 6000), Image.Resampling.LANCZOS)
    preview.save(preview_path, format="PNG", optimize=True)
    preview.close()
    return {
        "enhanced": layered_path.name,
        "layered_tif": layered_path.name,
        "transparent": layered_path.name,
        "enhanced_preview": str(preview_path),
        "preview_path": str(preview_path),
        "dpi": output_dpi,
        "width": output_size[0],
        "height": output_size[1],
        "original_width": original_size[0],
        "original_height": original_size[1],
        "source_dpi": round(source_dpi_x, 4),
        "resample_factor": round(resize_factor, 6),
        "confidence": round(min(99.0, 86.0 + (9.0 if keep_faint else 0.0)), 1),
    }
