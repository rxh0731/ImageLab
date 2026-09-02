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


def _find_text_regions(mask: Image.Image, enhanced: Image.Image, max_size: int = 2400) -> list[dict]:
    """Return editable candidate regions; never use these polygons to hard-crop pixels."""
    full_width, full_height = mask.size
    scale = min(1.0, max_size / max(full_width, full_height))
    analysis_size = (max(1, int(full_width * scale)), max(1, int(full_height * scale)))
    binary = np.asarray(mask.resize(analysis_size, Image.Resampling.BOX), dtype=np.uint8)
    ink = np.where(binary < 128, 255, 0).astype(np.uint8)
    height, width = ink.shape
    kernel_size = max(1, int(round(min(width, height) / 1400)))
    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gray = np.asarray(enhanced.resize(analysis_size, Image.Resampling.BOX), dtype=np.uint8)
    min_area = max(12, int(width * height * 0.000002))
    regions: list[dict] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        x, y, w, h = cv2.boundingRect(contour)
        if area < min_area or w < 3 or h < 3:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, max(1.0, perimeter * 0.02), True).reshape(-1, 2)
        if len(polygon) < 3:
            continue
        region_mask = np.zeros_like(ink)
        cv2.drawContours(region_mask, [contour], -1, 255, -1)
        mean_darkness = float(255 - gray[region_mask > 0].mean()) if np.any(region_mask) else 0.0
        confidence = round(float(np.clip(55 + mean_darkness * 0.55, 55, 99)), 1)
        regions.append({
            "id": len(regions) + 1,
            "polygon": [[int(round(px / scale)), int(round(py / scale))] for px, py in polygon],
            "bbox": [int(round(x / scale)), int(round(y / scale)), int(round(w / scale)), int(round(h / scale))],
            "confidence": confidence,
        })
    regions.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    for index, region in enumerate(regions, start=1):
        region["id"] = index
    return regions[:5000]


def _clean_ink_mask(mask_arr: np.ndarray, width: int, height: int, mode: str) -> np.ndarray:
    """Remove isolated salt-and-pepper texture without hard-cropping real strokes."""
    ink = np.where(mask_arr < 128, 255, 0).astype(np.uint8)
    if mode == "conservative":
        return ink
    # Analyze huge masks at a bounded working size, then map the keep mask back
    # to the original grid. Final output pixels remain full resolution.
    analysis_scale = min(1.0, 3200.0 / max(width, height))
    if analysis_scale < 1.0:
        analysis_size = (max(1, int(width * analysis_scale)), max(1, int(height * analysis_scale)))
        ink = cv2.resize(ink, analysis_size, interpolation=cv2.INTER_AREA)
        ink = np.where(ink > 96, 255, 0).astype(np.uint8)
    # Keep the balanced path free of binary median erosion: thin and broken
    # calligraphic strokes are more important than removing every speck.
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
    return keep


def _stroke_edge_band(ink_mask: np.ndarray, enhanced_arr: np.ndarray, threshold: int, mode: str) -> np.ndarray:
    """Keep antialiased pixels immediately around retained strokes."""
    if mode == "conservative":
        return ink_mask
    # Do not let white-background cleanup cut through the antialiased fringe.
    # A 3x3 band is resolution-independent and avoids adding visible halos.
    kernel = np.ones((3, 3), dtype=np.uint8)
    band = cv2.dilate(ink_mask, kernel, iterations=1)
    edge_limit = min(245, max(threshold + 56, 180))
    edge_pixels = enhanced_arr <= edge_limit
    return np.where((ink_mask > 0) | ((band > 0) & edge_pixels), 255, 0).astype(np.uint8)


def process_image(
    source: Path,
    output_dir: Path,
    image_type: str,
    mode: str,
    keep_faint: bool,
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        with Image.open(source) as opened:
            original_size = opened.size
            original = opened.convert("L").copy()
    scale = 1.0
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
    progress(62, "提取文字候选")

    # Preserve faint strokes by using a conservative threshold and keeping the grayscale source.
    threshold = 128 + preset.threshold_bias
    mask_arr = np.asarray(enhanced, dtype=np.uint8)
    if keep_faint:
        threshold -= 9
    if mode == "balanced":
        threshold += 10
    elif mode == "strong":
        threshold += 4
    raw_mask = np.where(mask_arr < threshold, 35, 255).astype(np.uint8)
    ink_mask = _clean_ink_mask(raw_mask, width, height, mode)
    display_mask = _stroke_edge_band(ink_mask, enhanced_arr, threshold, mode)
    # In balanced/strong modes, make the displayed enhancement a true white
    # background image. Keep grayscale values only where ink was retained.
    if mode in {"balanced", "strong"}:
        cleaned_display = np.where(display_mask > 0, enhanced_arr, 255).astype(np.uint8)
    else:
        cleaned_display = enhanced_arr
    enhanced = Image.fromarray(cleaned_display, mode="L")
    text_arr = np.where(display_mask > 0, 35, 255).astype(np.uint8)
    text_mask = Image.fromarray(text_arr, mode="L")
    # Use a grayscale alpha fringe so antialiased stroke edges remain smooth.
    if mode in {"balanced", "strong"}:
        alpha_arr = np.where(display_mask > 0, np.clip(255 - enhanced_arr, 0, 255), 0).astype(np.uint8)
        alpha = Image.fromarray(alpha_arr, mode="L")
    else:
        alpha = ImageOps.invert(text_mask)
    transparent = Image.new("RGBA", enhanced.size, (38, 34, 26, 0))
    transparent.putalpha(alpha)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    enhanced_path = output_dir / f"{stem}_enhanced.png"
    mask_path = output_dir / f"{stem}_text-mask.png"
    transparent_path = output_dir / f"{stem}_transparent.png"
    preview_path = output_dir / f"{stem}_enhanced_preview.png"
    progress(78, "分析文字区域")
    regions = _find_text_regions(text_mask, enhanced)
    progress(90, "保存结果")
    enhanced.save(enhanced_path)
    text_mask.save(mask_path)
    transparent.save(transparent_path)
    preview = enhanced.copy()
    preview.thumbnail((6000, 6000), Image.Resampling.LANCZOS)
    preview.save(preview_path, format="PNG", optimize=True)
    preview.close()
    return {
        "enhanced": enhanced_path.name,
        "text_mask": mask_path.name,
        "transparent": transparent_path.name,
        "enhanced_preview": preview_path.name,
        "width": enhanced.width,
        "height": enhanced.height,
        "original_width": original_size[0],
        "original_height": original_size[1],
        "preview_scale": round(scale, 4),
        "confidence": round(min(99.0, 86.0 + (9.0 if keep_faint else 0.0)), 1),
        "regions": regions,
    }
