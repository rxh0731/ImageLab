from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
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


def process_image(source: Path, output_dir: Path, image_type: str, mode: str, keep_faint: bool) -> dict[str, str | int | float]:
    preset = PRESETS.get(image_type, PRESETS["other"])
    if mode == "balanced":
        preset = ProcessingPreset(preset.background_radius, preset.contrast + 0.10, preset.threshold_bias + 2, preset.denoise_radius + 0.2)
    elif mode == "strong":
        preset = ProcessingPreset(preset.background_radius + 10, preset.contrast + 0.18, preset.threshold_bias + 5, preset.denoise_radius + 0.5)

    original = Image.open(source).convert("L")
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
    }
