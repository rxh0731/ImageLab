from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import cv2
from PIL import Image, ImageOps
from .exporting import export_image, output_path


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


def _percentile(values: np.ndarray, percentile: float) -> float:
    """Estimate a percentile without repeatedly scanning very large rasters."""
    flat = np.asarray(values).reshape(-1)
    max_samples = 4_000_000
    if flat.size > max_samples:
        flat = flat[::int(np.ceil(flat.size / max_samples))]
    return float(np.percentile(flat, percentile))


def _normalize(gray: Image.Image) -> Image.Image:
    arr = np.asarray(gray, dtype=np.uint8)
    low, high = _percentile(arr, 1), _percentile(arr, 99)
    if high <= low:
        return gray.convert("L")
    out = cv2.convertScaleAbs(arr, alpha=255.0 / (high - low), beta=-low * 255.0 / (high - low))
    return Image.fromarray(out, mode="L")


def _repair_stroke_density(ink: np.ndarray, kernel_size: int = 9, fast: bool = False) -> np.ndarray:
    """Restore small holes inside retained strokes without filling glyph counters."""
    # The raw threshold can leave narrow background-colored cuts inside a
    # solid ink stroke. A single 9x9 closing bridges those cuts while keeping
    # the larger counters of a glyph open.
    kernel_size = max(3, int(kernel_size) | 1)
    closed = cv2.morphologyEx(
        ink,
        cv2.MORPH_CLOSE,
        np.ones((kernel_size, kernel_size), dtype=np.uint8),
        iterations=1,
    )
    # Contour hierarchy over multi-million-pixel analysis masks is much more
    # expensive than the closing itself. Large images use the source-stroke
    # reconstruction afterward, so the fast close is sufficient here and
    # avoids a long full-page contour scan.
    if fast:
        return closed
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


def _clean_ink_mask(mask_arr: np.ndarray, width: int, height: int, mode: str, fast_repair: bool = False) -> np.ndarray:
    """Remove isolated salt-and-pepper texture without hard-cropping real strokes."""
    if mode == "conservative":
        # Conservative mode keeps the original full-resolution repair path to
        # avoid changing fine, faint strokes.
        ink = np.where(mask_arr < 128, 255, 0).astype(np.uint8)
        return _repair_stroke_density(ink)
    # Analyze huge masks at a bounded working size, then map the keep mask back
    # to the original grid. Final output pixels remain full resolution.
    analysis_scale = min(1.0, 3200.0 / max(width, height))
    if analysis_scale < 1.0:
        analysis_size = (max(1, int(width * analysis_scale)), max(1, int(height * analysis_scale)))
        # Resize the two-level mask before thresholding. This avoids allocating
        # and then immediately discarding a second full-resolution binary array.
        reduced = cv2.resize(mask_arr, analysis_size, interpolation=cv2.INTER_AREA)
        ink = np.where(reduced < 128, 255, 0).astype(np.uint8)
        ink = np.where(ink > 96, 255, 0).astype(np.uint8)
    else:
        ink = np.where(mask_arr < 128, 255, 0).astype(np.uint8)
    if fast_repair:
        # Large pages use source-channel reconstruction immediately afterward.
        # A small open/close removes isolated analysis noise without the costly
        # distance-transform and connected-component passes used for normal
        # sized pages.
        kernel = np.ones((3, 3), dtype=np.uint8)
        ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel)
    if mode == "balanced":
        # Background texture is often connected to a character by a faint
        # line, so area-only filtering cannot separate it reliably. Keep the
        # thick core of each stroke instead: small specks do not have enough
        # distance from the background to form a core, while detached dots
        # with a real stroke core remain eligible for source-pixel passthrough.
        distance = cv2.distanceTransform((ink > 0).astype(np.uint8), cv2.DIST_L2, 5)
        positive = distance[distance > 0]
        if positive.size:
            core_width = float(np.clip(_percentile(positive, 50), 3.0, 12.0))
        else:
            core_width = 3.0
        core = np.where(distance >= core_width, 255, 0).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
        min_core_area = max(12, int(core.shape[0] * core.shape[1] * 0.0000002))
        keep_table = np.where(stats[:, cv2.CC_STAT_AREA] >= min_core_area, np.uint8(255), np.uint8(0))
        keep_table[0] = 0
        keep = keep_table[labels]
        # At very large source sizes, a 3200px analysis level can reduce
        # legitimate character strokes to one-pixel lines. If the thickness
        # pass would discard nearly everything, retain connected candidates
        # instead of returning an almost blank page.
        ink_area = int(np.count_nonzero(ink))
        keep_area = int(np.count_nonzero(keep))
        if ink_area and keep_area < ink_area * 0.25:
            count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
            min_candidate_area = max(20, int(ink.shape[0] * ink.shape[1] * 0.0000002))
            areas = stats[:, cv2.CC_STAT_AREA]
            lengths = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
            eligible = (areas >= min_candidate_area) | ((lengths >= 20) & (areas >= max(3, min_candidate_area // 4)))
            keep_table = np.where(eligible, np.uint8(255), np.uint8(0))
            keep_table[0] = 0
            keep = keep_table[labels]
        if analysis_scale < 1.0:
            keep = _repair_stroke_density(keep, max(3, int(round(9 * analysis_scale)) | 1), fast=fast_repair)
            keep = cv2.resize(keep, (width, height), interpolation=cv2.INTER_NEAREST)
            return keep
        return _repair_stroke_density(keep, fast=fast_repair)
    # The strong path is allowed to use a light median cleanup after the
    # conservative core selection above.
    if mode == "strong":
        ink = cv2.medianBlur(ink, 3)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    min_area = max(8, int(width * height * (0.00000010 if mode == "balanced" else 0.00000055)))
    areas = stats[:, cv2.CC_STAT_AREA]
    lengths = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    # Map labels once instead of scanning the whole mask for every component.
    eligible = (areas >= min_area) | ((lengths >= 20) & (areas >= max(3, min_area // 4)))
    keep_table = np.where(eligible, np.uint8(255), np.uint8(0))
    keep_table[0] = 0
    keep = keep_table[labels]
    if analysis_scale < 1.0:
        keep = _repair_stroke_density(keep, max(3, int(round(9 * analysis_scale)) | 1), fast=fast_repair)
        keep = cv2.resize(keep, (width, height), interpolation=cv2.INTER_NEAREST)
        return keep
    return _repair_stroke_density(keep, fast=fast_repair)


def _stroke_edge_band(ink_mask: np.ndarray, enhanced_arr: np.ndarray, threshold: int, mode: str) -> np.ndarray:
    """Keep antialiased pixels immediately around retained strokes."""
    if mode == "conservative":
        return ink_mask
    # Do not let white-background cleanup cut through the antialiased fringe.
    # Balanced mode is deliberately wider because its priority is preserving
    # the exact source stroke edge, including faint pixels between samples.
    # The source-stroke reconstruction already includes the original stroke
    # fringe. Expanding a balanced mask here would reintroduce the dark paper
    # grain immediately outside the character, so keep its boundary exact.
    radius = 0 if mode == "balanced" else 1
    if radius == 0:
        return ink_mask
    kernel_size = radius * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    band = cv2.dilate(ink_mask, kernel, iterations=1)
    edge_limit = min(235, max(threshold + (64 if mode == "balanced" else 56), 190))
    edge_pixels = enhanced_arr <= edge_limit
    return np.where((ink_mask > 0) | ((band > 0) & edge_pixels), 255, 0).astype(np.uint8)


def _restore_enclosed_ink_texture(
    ink: np.ndarray,
    source_ink_pixels: np.ndarray,
    max_hole_area: int = 64,
) -> np.ndarray:
    """Restore small enclosed ink gaps, without closing open glyph spaces.

    Run on the bounded analysis mask, not on the full-resolution page. A gap
    must be enclosed, small, and contain only source pixels compatible with
    ink. Large counters and paper-colored pinholes remain transparent.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (ink == 0).astype(np.uint8), connectivity=8,
    )
    if count <= 1:
        return ink
    areas = stats[:, cv2.CC_STAT_AREA]
    ink_counts = np.bincount(labels[source_ink_pixels], minlength=count)
    restore = (areas <= max_hole_area) & (ink_counts == areas)
    restore[0] = False
    # Components reaching the page edge belong to the exterior background.
    border_labels = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    restore[border_labels] = False
    if np.any(restore):
        ink[restore[labels]] = 255
    return ink


def _restore_supported_ink_texture(
    ink: np.ndarray,
    source_ink_pixels: np.ndarray,
) -> np.ndarray:
    """Recover textured ink cuts with nearby ink on multiple opposing sides.

    Brown ink on patterned paper may exceed the neutral-ink chroma ceiling.
    Its missing parts can connect to the exterior, so enclosed-hole filling
    alone is insufficient. Require bracketing along at least two of four
    axes within 31 analysis pixels, plus dark source-color evidence. Do not
    propagate newly restored pixels: support always comes from the old mask.
    """
    span = 31
    diagonal = np.eye(span, dtype=np.uint8)
    kernels = (
        np.ones((1, span), dtype=np.uint8),
        np.ones((span, 1), dtype=np.uint8),
        diagonal,
        np.fliplr(diagonal).copy(),
    )
    votes = np.zeros(ink.shape, dtype=np.uint8)
    for kernel in kernels:
        bracketed = cv2.morphologyEx(
            ink, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0,
        )
        votes += bracketed > 0
    # Reject fine isolated paper grain even if two nearby strokes bracket it.
    texture = np.where((ink > 0) | source_ink_pixels, 255, 0).astype(np.uint8)
    texture = cv2.morphologyEx(texture, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels = cv2.connectedComponents(texture, connectivity=8)
    supported = np.zeros(count, dtype=bool)
    supported[np.unique(labels[ink > 0])] = True
    supported[0] = False
    restore = source_ink_pixels & supported[labels] & (votes >= 2)
    ink[restore] = 255
    return ink


def _restore_source_strokes(
    ink_mask: np.ndarray,
    source_gray: np.ndarray,
    source_rgb: np.ndarray | None = None,
) -> np.ndarray:
    """Recover dark source strokes without restoring patterned paper texture.

    Background correction can make the middle of a thick stroke look like
    paper. The old implementation restored the complete Otsu connected
    component, which is unsafe for rubbings: a dark paper grain touching a
    character then brought the whole grain component back. We build a stroke
    candidate from the uncorrected image, remove fine grain, bridge small
    breaks, and keep only page-sized components. For the common orange/brown
    rubbing paper, the green channel separates black ink from the paper much
    better than grayscale.
    """
    height, width = source_gray.shape
    work_scale = min(1.0, 3200.0 / max(height, width))
    work_size = (width, height)
    work_gray = source_gray
    work_rgb = source_rgb
    if work_scale < 1.0:
        work_size = (
            max(1, int(round(width * work_scale))),
            max(1, int(round(height * work_scale))),
        )
        work_gray = cv2.resize(source_gray, work_size, interpolation=cv2.INTER_AREA)
        if source_rgb is not None:
            work_rgb = cv2.resize(source_rgb, work_size, interpolation=cv2.INTER_AREA)

    if work_rgb is not None and work_rgb.shape[:2] == work_gray.shape:
        rgb = np.asarray(work_rgb, dtype=np.uint8)
        # Splitting three channels and using byte-wise SIMD min/max avoids
        # NumPy's repeated strided reductions over millions of RGB triples.
        red_channel, green_channel, blue_channel = cv2.split(rgb)
        chroma = cv2.subtract(
            cv2.max(cv2.max(red_channel, green_channel), blue_channel),
            cv2.min(cv2.min(red_channel, green_channel), blue_channel),
        )
        paper = work_gray >= _percentile(work_gray, 55)
        chromatic_paper = bool(_percentile(chroma[paper], 50) >= 28) if np.any(paper) else False
    else:
        rgb = None
        chromatic_paper = False

    candidate_color_mask: np.ndarray | None = None
    if chromatic_paper and rgb is not None:
        # The lower green-channel cluster is the dark ink. Use a wider lower
        # quantile than the old p10-8 rule: textured strokes often contain
        # lighter pixels that still belong to the original ink. Estimate a
        # chroma ceiling from the dark ink and the paper clusters instead of
        # using one broad fixed value: a fixed limit admitted dark orange paper
        # texture in the gaps of characters such as 闻.
        ink_channel = green_channel
        threshold = int(np.clip(_percentile(ink_channel, 18), 70, 135))
        dark_reference_limit = max(60.0, min(110.0, _percentile(work_gray, 10)))
        dark_reference = work_gray <= dark_reference_limit
        dark_chroma = chroma[dark_reference]
        paper_chroma = chroma[paper]
        dark_limit = (_percentile(dark_chroma, 99.5) + 8.0) if dark_chroma.size else 64.0
        paper_limit = (_percentile(paper_chroma, 25.0) - 15.0) if paper_chroma.size else 80.0
        chroma_limit = int(np.clip(min(100.0, dark_limit, paper_limit), 48.0, 80.0))
        candidate_color_mask = (ink_channel <= threshold) & (chroma <= chroma_limit)
        candidates = np.where(candidate_color_mask, 255, 0).astype(np.uint8)
    else:
        _, candidates = cv2.threshold(
            work_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
        )

    # Scale the kernels with the page size. The 4605x3241 rubbing uses 5/15px;
    # smaller scans receive proportionally smaller cleanup kernels.
    page_scale = float(np.clip(min(work_gray.shape) / 3200.0, 0.55, 1.5))

    def odd_size(value: float) -> int:
        size = max(3, int(round(value * page_scale)))
        return size if size % 2 else size + 1

    open_size = odd_size(5)
    close_size = odd_size(15)
    candidates = cv2.morphologyEx(
        candidates, cv2.MORPH_OPEN, np.ones((open_size, open_size), dtype=np.uint8),
    )
    candidates = cv2.morphologyEx(
        candidates, cv2.MORPH_CLOSE, np.ones((close_size, close_size), dtype=np.uint8),
    )
    # Closing can bridge a dark paper texture into a real stroke. Reapply the
    # source-pixel color gate after morphology so the operation may bridge
    # small breaks in a stroke but cannot expand the restored area into pixels
    # that look like the surrounding paper.
    if candidate_color_mask is not None:
        candidates = cv2.bitwise_and(candidates, np.where(candidate_color_mask, 255, 0).astype(np.uint8))
        # A strict per-pixel gate can itself punch holes through textured ink:
        # on ink-heavy pages the p18 green threshold reaches its 70 floor,
        # although genuine stroke texture may be noticeably lighter. Restore
        # only small enclosed gaps with ink chroma and a clear brightness
        # margin below the paper. Do not relax the outer stroke boundary.
        interior_limit = max(threshold, min(200.0, _percentile(ink_channel[paper], 10) - 20.0))
        interior_ink = (ink_channel <= interior_limit) & (chroma <= chroma_limit)
        candidates = _restore_enclosed_ink_texture(candidates, interior_ink)
        # Brown texture needs both local stroke support and a clear color
        # margin from the paper. The red guard rejects orange paper with a
        # deceptively low green channel, including genuine open glyph gaps.
        texture_red_limit = min(200.0, _percentile(red_channel[paper], 5) - 24.0)
        textured_ink = ((ink_channel <= interior_limit)
                        & (red_channel <= texture_red_limit)
                        & (chroma <= min(112, chroma_limit + 32)))
        candidates = _restore_supported_ink_texture(candidates, textured_ink)

    # At very large sizes the opening/closing pass has already removed fine
    # grain. Avoid a costly connected-component table for millions of pixels;
    # the candidate is also restricted to the dark ink channel above.
    if work_scale < 0.5:
        supported = candidates
    else:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, connectivity=8)
        work_height, work_width = work_gray.shape
        min_area = max(16, int(work_height * work_width * 0.00000020))
        areas = stats[:, cv2.CC_STAT_AREA]
        lengths = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
        eligible = (areas >= min_area) | (
            (lengths >= max(20, int(round(20 * page_scale)))) & (areas >= max(4, min_area // 4))
        )
        keep_table = np.where(eligible, np.uint8(255), np.uint8(0))
        keep_table[0] = 0
        supported = keep_table[labels]
    if work_scale < 1.0:
        supported = cv2.resize(supported, (width, height), interpolation=cv2.INTER_NEAREST)
    return cv2.bitwise_or(ink_mask, supported)


def _refine_faint_stroke_edges(
    ink_mask: np.ndarray,
    source_rgb: np.ndarray,
) -> np.ndarray:
    """Recover source-supported stroke texture and edges in bounded tiles.

    Never change source colors or delete existing ink. Local, eroded ink
    samples identify pale strokes or dark ink with pale grain; nearby background samples include both
    light paper and darker pattern colors. A fixed band and connection to
    the original mask prevent unbounded propagation through touching grain.
    Textured pale ink also permits a bounded concavity repair inside each
    component's original convex envelope, never an unrestricted dilation.
    """
    height, width = ink_mask.shape
    radius = int(np.clip(np.ceil(max(height, width) / 3200.0) * 2, 3, 12))
    halo = max(40, radius * 4)
    notch_radius = min(48, radius * 6, halo - 4)
    tile_size = 512
    band_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
    paper_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 4 + 1,) * 2)
    notch_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (notch_radius * 2 + 1,) * 2)
    # Read support from a snapshot, so tile order cannot affect the result.
    original = ink_mask.copy()
    for top in range(0, height, tile_size):
        for left in range(0, width, tile_size):
            bottom, right = min(height, top + tile_size), min(width, left + tile_size)
            y0, y1 = max(0, top - halo), min(height, bottom + halo)
            x0, x1 = max(0, left - halo), min(width, right + halo)
            mask = original[y0:y1, x0:x1]
            if not np.any(mask):
                continue
            band = (cv2.dilate(mask, band_kernel) > 0) & (mask == 0)
            center = np.s_[top-y0:bottom-y0, left-x0:right-x0]
            if not np.any(band[center]):
                continue
            rgb = source_rgb[y0:y1, x0:x1]
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            core = cv2.erode(mask, np.ones((7, 7), np.uint8)) > 0
            if np.count_nonzero(core) < 64:
                continue
            # Pale edges and dark strokes with pale scan/ink grain need
            # different color guards. Do not skip textured dark ink merely
            # because its median is dark (this left holes on page 640).
            core_gray = gray[core]
            ink_median = _percentile(core_gray, 50)
            ink_gray95 = _percentile(core_gray, 95)
            pale_stroke = ink_median >= 48 and _percentile(core_gray, 75) >= 70
            dark_texture = ink_gray95 - ink_median >= 35
            if not pale_stroke and not dark_texture:
                continue
            outer = cv2.dilate(mask, paper_kernel) > 0
            neighborhood = cv2.dilate(mask, np.ones((halo * 2 + 1,) * 2, np.uint8)) > 0
            paper = neighborhood & ~outer
            if np.count_nonzero(paper) < 128:
                continue
            paper_rgb = rgb[paper]
            paper_gray = gray[paper]
            ink_rgb = rgb[core]
            # Refuse ambiguous/dark-background tiles rather than treating a
            # globally relaxed darkness cutoff as a universal edge detector.
            ink_level = _percentile(core_gray, 75)
            paper_level = _percentile(paper_gray, 60)
            if paper_level - ink_level < 45:
                continue
            # The old boundary can miss a wider pale fringe. Do not learn
            # that same missing ink as a dark "paper" cluster at tile edges.
            paper_samples = paper_gray >= (paper_level + ink_level) * 0.5
            paper_rgb = paper_rgb[paper_samples]
            paper_gray = paper_gray[paper_samples]
            paper_floor = _percentile(paper_gray, 10)
            if paper_floor - ink_level < 35:
                continue
            red_floor = _percentile(paper_rgb[:, 0], 10)
            ink_chroma = ink_rgb.max(axis=1) - ink_rgb.min(axis=1)
            paper_chroma = paper_rgb.max(axis=1) - paper_rgb.min(axis=1)
            chromatic = _percentile(paper_chroma, 50) >= 28
            ink_chroma95 = _percentile(ink_chroma, 95)
            ink_chroma995 = _percentile(ink_chroma, 99.5)
            # Keep the established pale-ink route unchanged. This new route
            # handles only dark tiles that the previous code skipped.
            neutral_texture = not pale_stroke and dark_texture and chromatic and ink_chroma995 <= 40
            if not pale_stroke and not neutral_texture:
                continue
            chroma_limit = min(115.0, ink_chroma95 + 16)
            eligible = band & (gray <= paper_floor - 25) & (rgb[:, :, 0] <= red_floor - 25)
            # Deep cuts can exceed the narrow edge band, in either brown ink
            # or neutral dark ink with pale grain. Require matching variation
            # in the retained SOURCE samples. Neutral grain keeps a much
            # tighter chroma gate to exclude the colored paper. The convex
            # envelope is a search constraint, not a mask to fill blindly.
            notches = np.zeros(mask.shape, dtype=bool)
            notch_domain = None
            if neutral_texture or (ink_chroma995 - ink_chroma95 >= 8
                                   and ink_gray95 - ink_median >= 45):
                envelope = np.zeros_like(mask)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) >= 64:
                        cv2.drawContours(envelope, [cv2.convexHull(contour)], -1, 255, cv2.FILLED)
                # Permit the ordinary narrow fringe around the envelope as
                # well: its straight chord otherwise clips a rounded tip.
                envelope = cv2.dilate(envelope, band_kernel)
                if neutral_texture:
                    # Wide elliptical dilation is costly on every textured
                    # dark-ink tile. A bounded distance field does the same
                    # local search without repeatedly scanning a large kernel.
                    nearby = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 5) <= notch_radius
                else:
                    nearby = cv2.dilate(mask, notch_kernel) > 0
                notches = ((envelope > 0) & (mask == 0) & nearby)
                notch_domain = notches
                notch_chroma = (min(48.0, ink_chroma995 + 16) if neutral_texture
                                else min(115.0, ink_chroma995 + 20))
                notch_gray = min(paper_level - 20, _percentile(core_gray, 99) + 16)
                notch_pixels = (notches & (gray <= notch_gray)
                                & (rgb[:, :, 0] <= red_floor - 12))
                ny, nx = np.nonzero(notch_pixels)
                nrgb = rgb[ny, nx]
                notch_pixels[ny, nx] = (nrgb.max(axis=1) - nrgb.min(axis=1)) <= notch_chroma
                eligible |= notch_pixels
                notches = notch_pixels
            ys, xs = np.nonzero(eligible)
            if not len(xs):
                continue
            edge_rgb = rgb[ys, xs]
            keep_edge = np.ones(len(xs), dtype=bool)
            if chromatic:
                keep_edge &= (((edge_rgb.max(axis=1) - edge_rgb.min(axis=1)) <= chroma_limit)
                              | notches[ys, xs])
            # Compare with multiple local paper tones. A pale edge must be
            # distinctly darker than each nearby paper-pattern cluster.
            # Quantile bins prevent bright bare paper dominating the model.
            for low, high in ((0, 25), (25, 60), (60, 100)):
                lower, upper = np.percentile(paper_gray, [low, high])
                samples = paper_rgb[(paper_gray >= lower) & (paper_gray <= upper)]
                if samples.size:
                    tone = np.median(samples, axis=0).astype(np.int16)
                    difference = edge_rgb.astype(np.int16) - tone
                    paper_distance = np.sum(difference.astype(np.float32) ** 2, axis=1)
                    keep_edge &= paper_distance >= 28 ** 2
            eligible[ys, xs] = keep_edge
            if not np.any(eligible[center]):
                continue
            allowed = np.where((mask > 0) | eligible, 255, 0).astype(np.uint8)
            if neutral_texture:
                # Dark scan grain can include a hairline touching a letter.
                # Require a locally solid source-supported area for NEW
                # pixels, while retaining every existing fine stroke pixel.
                solid = cv2.morphologyEx(allowed, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
                eligible &= solid > 0
                allowed = np.where((mask > 0) | eligible, 255, 0).astype(np.uint8)
            # A one-sided connection suffices at a real exterior edge. Narrow
            # isolated specks cannot become new foreground components.
            count, labels = cv2.connectedComponents(allowed, connectivity=8)
            connected = np.zeros(count, dtype=bool)
            connected[np.unique(labels[mask > 0])] = True
            connected[0] = False
            restore = eligible & connected[labels]
            if notch_domain is not None and np.any(restore & notches):
                # The recovered brown tail can enclose tiny near-white ink
                # highlights. Keep their source RGB rather than punching
                # them out. This relaxation is limited to fully enclosed
                # <=32px flecks in the repaired notch, with chroma/brightness
                # guards still rejecting orange paper and bare white paper.
                repaired = np.where((mask > 0) | restore, 255, 0).astype(np.uint8)
                pale = (notch_domain
                        & (gray <= min(240, _percentile(paper_gray, 90) + 12))
                        & (rgb[:, :, 0] <= min(252, _percentile(paper_rgb[:, 0], 90) + 4)))
                py, px = np.nonzero(pale)
                pale_rgb = rgb[py, px]
                pale_chroma = notch_chroma if neutral_texture else 115
                pale[py, px] = (pale_rgb.max(axis=1) - pale_rgb.min(axis=1)) <= pale_chroma
                repaired = _restore_enclosed_ink_texture(repaired, pale, max_hole_area=32)
                restore |= repaired > mask
            target = ink_mask[top:bottom, left:right]
            target[restore[center]] = 255
    return ink_mask


def process_image(
    source: Path,
    output_dir: Path,
    image_type: str,
    mode: str,
    keep_faint: bool,
    params: dict | None = None,
    preview_dir: Path | None = None,
    progress_callback=None,
    generate_preview: bool = True,
    output_format: str = "tif",
    white_background: bool = False,
    overwrite_output: bool = True,
) -> dict:
    def progress(value: int, message: str) -> None:
        if progress_callback:
            progress_callback(value, message)

    if source.resolve() == output_path(output_dir, source.stem, output_format).resolve():
        raise ValueError("输出图片与原图路径相同，请选择其他输出目录，以免覆盖原图。")
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        with Image.open(source) as opened:
            original_size = opened.size
            source_dpi_info = opened.info.get("dpi")
            original_rgb = opened.convert("RGB")
            original = ImageOps.grayscale(original_rgb)
    source_rgb_arr = np.asarray(original_rgb, dtype=np.uint8)
    source_dpi_x = 300.0
    source_dpi_y = 300.0
    if isinstance(source_dpi_info, (tuple, list)) and source_dpi_info:
        try:
            source_dpi_x = float(source_dpi_info[0])
        except (TypeError, ValueError):
            source_dpi_x = 300.0
        try:
            source_dpi_y = float(source_dpi_info[1] if len(source_dpi_info) > 1 else source_dpi_x)
        except (TypeError, ValueError):
            source_dpi_y = source_dpi_x
    elif source_dpi_info is not None:
        try:
            source_dpi_x = float(source_dpi_info)
        except (TypeError, ValueError):
            source_dpi_x = 300.0
        source_dpi_y = source_dpi_x
    if not np.isfinite(source_dpi_x) or source_dpi_x <= 0:
        source_dpi_x = 300.0
    if not np.isfinite(source_dpi_y) or source_dpi_y <= 0:
        source_dpi_y = source_dpi_x
    use_output_dpi = bool(params.get("use_output_dpi", False)) if params else False
    if use_output_dpi and params and "dpi" in params:
        output_dpi_x = max(72.0, min(2400.0, float(params["dpi"])))
        output_dpi_y = output_dpi_x
    else:
        output_dpi_x = source_dpi_x
        output_dpi_y = source_dpi_y
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
    background_level = _percentile(corrected_arr, 98.5)
    if background_level > 1:
        corrected_arr = cv2.convertScaleAbs(corrected_arr, alpha=255.0 / background_level)
    if preset.denoise_radius > 0:
        corrected_arr = cv2.GaussianBlur(corrected_arr, (0, 0), sigmaX=preset.denoise_radius, sigmaY=preset.denoise_radius)
    if mode == "strong":
        corrected_arr = cv2.medianBlur(corrected_arr, 3)
    enhanced_arr = cv2.convertScaleAbs(corrected_arr, alpha=preset.contrast, beta=128.0 * (1.0 - preset.contrast))
    progress(62, "生成文字保留掩膜")

    # Preserve faint strokes by using a conservative threshold and keeping the grayscale source.
    threshold = 128 + preset.threshold_bias
    mask_arr = enhanced_arr
    if keep_faint:
        threshold -= 9
    if mode == "balanced":
        # Keep a wider candidate range so pale antialiased parts and small
        # detached dots remain source-pixel passthrough candidates.
        threshold += 24
    elif mode == "strong":
        threshold += 4
    if mode != "conservative" and max(width, height) > 3200:
        # Generate the cleanup mask at the bounded analysis resolution. The
        # source RGB and final alpha remain full resolution, but this avoids
        # allocating a second 100+ megapixel binary raster.
        stride = max(1, int(np.ceil(max(width, height) / 3200.0)))
        analysis_size = (
            max(1, int(np.ceil(width / stride))),
            max(1, int(np.ceil(height / stride))),
        )
        # A strided view avoids another full-raster interpolation pass. At the
        # large scan sizes where this path is used, character strokes span many
        # source pixels and remain represented in the analysis grid.
        reduced_mask = enhanced_arr[::stride, ::stride]
        reduced_raw_mask = np.where(reduced_mask < threshold, 35, 255).astype(np.uint8)
        reduced_ink = _clean_ink_mask(
            reduced_raw_mask,
            analysis_size[0],
            analysis_size[1],
            mode,
            fast_repair=True,
        )
        ink_mask = cv2.resize(reduced_ink, (width, height), interpolation=cv2.INTER_NEAREST)
    else:
        raw_mask = np.where(mask_arr < threshold, 35, 255).astype(np.uint8)
        ink_mask = _clean_ink_mask(raw_mask, width, height, mode)
    progress(70, "根据原图补全笔画掩膜")
    ink_mask = _restore_source_strokes(
        ink_mask,
        np.asarray(original, dtype=np.uint8),
        source_rgb_arr,
    )
    progress(73, "校验笔画纹理与边缘")
    ink_mask = _refine_faint_stroke_edges(ink_mask, source_rgb_arr)
    display_mask = _stroke_edge_band(ink_mask, enhanced_arr, threshold, mode)
    # The output is a background removal operation. Every retained text pixel
    # is copied from the original RGB source; pixels outside the mask become
    # fully transparent in the exported TIFF.
    # The mask already contains only 0/255 values. Use it directly as alpha,
    # retaining the original RGB and avoiding redundant full-raster copies.
    source_rgba = original_rgb.convert("RGBA")
    # Match the former fromarray export: source decoder metadata must not
    # accidentally change output color interpretation or TIFF encoding.
    source_rgba.info.clear()
    alpha = Image.fromarray(display_mask, mode="L")
    source_rgba.putalpha(alpha)
    alpha.close()
    transparent = source_rgba

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    layered_path = output_path(output_dir, stem, output_format)
    preview_path = None
    progress(76, "准备透明文字图像")
    # Without the explicit unified-resolution switch, preserve the source DPI
    # and pixel dimensions. When enabled, the requested DPI controls the
    # actual raster size relative to the source DPI.
    resize_factor_x = output_dpi_x / source_dpi_x
    resize_factor_y = output_dpi_y / source_dpi_y
    output_size = (
        max(1, int(round(original_size[0] * resize_factor_x))),
        max(1, int(round(original_size[1] * resize_factor_y))),
    )
    if output_size != original_size:
        transparent = transparent.resize(output_size, Image.Resampling.LANCZOS)
    progress(90, "保存处理结果")
    export_image(transparent, layered_path, output_format, white_background, (output_dpi_x, output_dpi_y),
                 overwrite=overwrite_output,
                 progress_callback=lambda value, message: progress(90 + int(value * .05), message))
    if generate_preview:
        progress(96, "生成预览")
        preview_root = preview_dir or (output_dir.parent / ".imagelab_previews" / output_dir.name)
        preview_root.mkdir(parents=True, exist_ok=True)
        (preview_root / f"{stem}_text-mask.png").unlink(missing_ok=True)
        preview_path = preview_root / f"{stem}_preview.png"
        preview = transparent.copy()
        preview.thumbnail((6000, 6000), Image.Resampling.LANCZOS)
        display = Image.new("RGB", preview.size, "white")
        display.paste(preview, mask=preview.getchannel("A"))
        preview.close()
        # PNG compression effort changes file size and time, not image pixels.
        display.save(preview_path, format="PNG", compress_level=3)
        display.close()
    progress(100, "处理完成")
    return {
        "enhanced": layered_path.name,
        "layered_tif": layered_path.name,
        "transparent": layered_path.name,
        "enhanced_preview": str(preview_path or layered_path),
        "preview_path": str(preview_path or layered_path),
        "dpi": round(output_dpi_x, 4),
        "dpi_x": round(output_dpi_x, 4),
        "dpi_y": round(output_dpi_y, 4),
        "width": output_size[0],
        "height": output_size[1],
        "original_width": original_size[0],
        "original_height": original_size[1],
        "source_dpi": round(source_dpi_x, 4),
        "source_dpi_x": round(source_dpi_x, 4),
        "source_dpi_y": round(source_dpi_y, 4),
        "resample_factor": round(resize_factor_x, 6) if abs(resize_factor_x - resize_factor_y) < 1e-9 else None,
        "resample_factor_x": round(resize_factor_x, 6),
        "resample_factor_y": round(resize_factor_y, 6),
        "used_output_dpi": use_output_dpi,
        "confidence": round(min(99.0, 86.0 + (9.0 if keep_faint else 0.0)), 1),
    }
