from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


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
    arr = np.asarray(gray, dtype=np.float32)
    low, high = np.percentile(arr, (1, 99))
    if high <= low:
        return gray.convert("L")
    out = np.clip((arr - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="L")


def _find_text_regions(mask: Image.Image, enhanced: Image.Image) -> list[dict]:
    """Return editable candidate regions; never use these polygons to hard-crop pixels."""
    binary = np.asarray(mask, dtype=np.uint8)
    ink = np.where(binary < 128, 255, 0).astype(np.uint8)
    height, width = ink.shape
    kernel_size = max(1, int(round(min(width, height) / 1400)))
    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gray = np.asarray(enhanced, dtype=np.uint8)
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
            "polygon": [[int(px), int(py)] for px, py in polygon],
            "bbox": [int(x), int(y), int(w), int(h)],
            "confidence": confidence,
        })
    regions.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    for index, region in enumerate(regions, start=1):
        region["id"] = index
    return regions[:5000]


def process_image(source: Path, output_dir: Path, image_type: str, mode: str, keep_faint: bool) -> dict:
    preset = PRESETS.get(image_type, PRESETS["other"])
    if mode == "balanced":
        preset = ProcessingPreset(preset.background_radius, preset.contrast + 0.10, preset.threshold_bias + 2, preset.denoise_radius + 0.2)
    elif mode == "strong":
        preset = ProcessingPreset(preset.background_radius + 10, preset.contrast + 0.18, preset.threshold_bias + 5, preset.denoise_radius + 0.5)

    with Image.open(source) as opened:
        original = opened.convert("L").copy()
    normalized = _normalize(original)
    background = normalized.filter(ImageFilter.GaussianBlur(radius=preset.background_radius))
    norm_arr = np.asarray(normalized, dtype=np.float32)
    bg_arr = np.asarray(background, dtype=np.float32)
    corrected_arr = np.clip((norm_arr - bg_arr) + 128.0, 0, 255).astype(np.uint8)
    corrected = Image.fromarray(corrected_arr, mode="L")
    if preset.denoise_radius > 0:
        corrected = corrected.filter(ImageFilter.GaussianBlur(radius=preset.denoise_radius))
    enhanced = ImageEnhance.Contrast(corrected).enhance(preset.contrast)

    # Preserve faint strokes by using a conservative threshold and keeping the grayscale source.
    threshold = 128 + preset.threshold_bias
    mask_arr = np.asarray(enhanced, dtype=np.uint8)
    if keep_faint:
        threshold -= 9
    text_arr = np.where(mask_arr < threshold, 35, 255).astype(np.uint8)
    text_mask = Image.fromarray(text_arr, mode="L")
    alpha = ImageOps.invert(text_mask)
    transparent = Image.new("RGBA", enhanced.size, (38, 34, 26, 0))
    transparent.putalpha(alpha)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    enhanced_path = output_dir / f"{stem}_enhanced.png"
    mask_path = output_dir / f"{stem}_text-mask.png"
    transparent_path = output_dir / f"{stem}_transparent.png"
    regions = _find_text_regions(text_mask, enhanced)
    enhanced.save(enhanced_path)
    text_mask.save(mask_path)
    transparent.save(transparent_path)
    return {
        "enhanced": enhanced_path.name,
        "text_mask": mask_path.name,
        "transparent": transparent_path.name,
        "width": enhanced.width,
        "height": enhanced.height,
        "confidence": round(min(99.0, 86.0 + (9.0 if keep_faint else 0.0)), 1),
        "regions": regions,
    }
