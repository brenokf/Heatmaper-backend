"""
Automatic floor plan area detection and extraction from architectural PDFs.

Pipeline:
  1. Vector analysis (PyMuPDF path objects) — primary path for vector PDFs.
  2. Raster analysis (OpenCV or PIL) — fallback for scanned PDFs.
  3. Edge refinement — scans each edge inward to remove peripheral info blocks
     (legends, title blocks, stamps) that may be adjacent to the drawing.

No positions are hard-coded. All detection is automatic and self-adapting.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:  # pragma: no cover
    HAS_FITZ = False
    logger.warning("PyMuPDF not installed — PDF crop unavailable. pip install pymupdf")

try:
    import cv2
    HAS_CV2 = True
except ImportError:  # pragma: no cover
    HAS_CV2 = False


@dataclass
class CropResult:
    image: Image.Image   # Cropped RGB PIL image
    method: str          # 'vector' | 'raster_cv2' | 'raster_pil'
    confidence: float    # 0.0–1.0 quality estimate


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def crop_floorplan(
    pdf_bytes: bytes,
    page_index: int = 0,
    out_dpi: int = 250,
) -> CropResult:
    """
    Detect and crop the floor plan from one page of a PDF.

    Works for both vector architectural drawings (CAD-exported PDFs) and
    scanned documents. Returns a PIL Image containing only the floor plan.

    Args:
        pdf_bytes:   Raw PDF content.
        page_index:  0-based page number.
        out_dpi:     Output render resolution (dots per inch).
    """
    if not HAS_FITZ:
        raise RuntimeError(
            "PyMuPDF is required: pip install pymupdf"
        )

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if page_index >= doc.page_count:
        raise ValueError(
            f"Page {page_index} out of range (document has {doc.page_count} pages)"
        )

    page = doc.load_page(page_index)

    # Render a medium-resolution grayscale image for analysis.
    # We render at 150 DPI for analysis to keep it fast, then re-render
    # the detected region at out_dpi for the final output.
    analysis_dpi = 150
    analysis_scale = analysis_dpi / 72.0
    mat_a = fitz.Matrix(analysis_scale, analysis_scale)
    pix_gray = page.get_pixmap(matrix=mat_a, colorspace=fitz.csGRAY)
    img_gray = np.frombuffer(pix_gray.samples, dtype=np.uint8).reshape(
        pix_gray.height, pix_gray.width
    )

    ih, iw = img_gray.shape

    # ── Phase 1: Try vector path analysis ─────────────────────────────────
    drawings = page.get_drawings()
    is_vector = len(drawings) >= 30  # need enough paths for reliable analysis

    clip_pdf: Optional[fitz.Rect] = None
    method = "vector"
    confidence = 0.0

    if is_vector:
        clip_pdf, confidence = _vector_bbox(drawings, page.rect)
        if clip_pdf is None or confidence < 0.4:
            is_vector = False  # fall through to raster

    # ── Phase 2: Raster analysis if vector failed ──────────────────────────
    if not is_vector:
        bbox_px = _raster_bbox(img_gray)
        if bbox_px is None:
            bbox_px = (0, 0, iw, ih)
        x, y, w, h = bbox_px
        # Convert analysis-DPI pixels → PDF points
        pt = 72.0 / analysis_dpi
        clip_pdf = fitz.Rect(x * pt, y * pt, (x + w) * pt, (y + h) * pt)
        confidence = 0.65
        method = "raster_cv2" if HAS_CV2 else "raster_pil"

    # ── Phase 3: Edge refinement ──────────────────────────────────────────
    # Scan each edge of the detected region inward (at analysis DPI) to
    # find and remove peripheral info strips: legends, title blocks, stamps.
    # Strategy: a "drawing border" is the first row/col from the outside
    # with density ≥ DENSE.  Legend text / info blocks are sparse (< 5 %).
    clip_pdf = _refine_edges(img_gray, clip_pdf, page.rect, analysis_scale)

    # ── Phase 4: Re-render at output DPI ─────────────────────────────────
    out_scale = out_dpi / 72.0
    mat_out = fitz.Matrix(out_scale, out_scale)
    pix_out = page.get_pixmap(matrix=mat_out, clip=clip_pdf, colorspace=fitz.csRGB)
    img = Image.frombytes("RGB", (pix_out.width, pix_out.height), pix_out.samples)

    # Ensure white background (eliminates any transparency artifacts)
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img)

    doc.close()
    return CropResult(image=bg, method=method, confidence=confidence)


# ─────────────────────────────────────────────────────────────────────────────
# Vector path analysis
# ─────────────────────────────────────────────────────────────────────────────

def _vector_bbox(
    drawings: list[dict],
    page_rect,
) -> tuple[Optional[fitz.Rect], float]:
    """
    Identify the floor plan bounding box from the set of PyMuPDF path objects.

    Approach:
      - Build a 2D density grid where each cell counts how many paths overlap it.
      - Find the largest connected cluster of dense cells — that is the drawing.
      - Tighten the bounding box to the actual path extents inside the cluster.

    The floor plan has the highest density of geometric paths.
    Title blocks, legends, and stamps are isolated clusters at the periphery.
    """
    pw = float(page_rect.width)
    ph = float(page_rect.height)

    # Collect bounding rects of all meaningful paths (≥ 3 pt in any dimension)
    rects: list[tuple[float, float, float, float]] = []
    for d in drawings:
        r = d.get("rect")
        if r and r.width >= 3.0 and r.height >= 3.0:
            rects.append((float(r.x0), float(r.y0), float(r.x1), float(r.y1)))

    if len(rects) < 15:
        return None, 0.0

    # 60×60 density grid
    GW = GH = 60
    density = np.zeros((GH, GW), dtype=np.float32)

    for x0, y0, x1, y1 in rects:
        gx0 = max(0, int((x0 / pw) * GW))
        gx1 = min(GW - 1, int((x1 / pw) * GW) + 1)
        gy0 = max(0, int((y0 / ph) * GH))
        gy1 = min(GH - 1, int((y1 / ph) * GH) + 1)
        density[gy0:gy1, gx0:gx1] += 1.0

    if density.max() == 0:
        return None, 0.0

    # Threshold: keep cells with at least 8 % of peak activity
    norm = density / density.max()
    binary = (norm >= 0.08).astype(np.uint8)

    # Find the bounding box of the largest connected cluster
    grid_bbox = _largest_component_bbox(binary)
    if grid_bbox is None:
        return None, 0.0

    gx0, gy0, gx1, gy1 = grid_bbox

    # Convert grid → PDF coordinates (coarse)
    bx0 = (gx0 / GW) * pw
    by0 = (gy0 / GH) * ph
    bx1 = ((gx1 + 1) / GW) * pw
    by1 = ((gy1 + 1) / GH) * ph

    # Tighten to actual path extents within the cluster
    t_x0, t_y0, t_x1, t_y1 = pw, ph, 0.0, 0.0
    for rx0, ry0, rx1, ry1 in rects:
        cx = (rx0 + rx1) / 2.0
        cy = (ry0 + ry1) / 2.0
        if bx0 <= cx <= bx1 and by0 <= cy <= by1:
            t_x0 = min(t_x0, rx0)
            t_y0 = min(t_y0, ry0)
            t_x1 = max(t_x1, rx1)
            t_y1 = max(t_y1, ry1)

    if t_x1 <= t_x0 or t_y1 <= t_y0:
        t_x0, t_y0, t_x1, t_y1 = bx0, by0, bx1, by1

    # Confidence: area of detected region relative to page, and path coverage
    det_area = (t_x1 - t_x0) * (t_y1 - t_y0)
    coverage = det_area / (pw * ph)
    inside = sum(
        1 for rx0, ry0, rx1, ry1 in rects
        if t_x0 <= (rx0 + rx1) / 2 <= t_x1 and t_y0 <= (ry0 + ry1) / 2 <= t_y1
    )
    fraction_inside = inside / len(rects)
    confidence = min(1.0, coverage * 1.2 + fraction_inside * 0.5)

    clip = fitz.Rect(
        max(0.0, t_x0),
        max(0.0, t_y0),
        min(pw, t_x1),
        min(ph, t_y1),
    )
    return clip, confidence


# ─────────────────────────────────────────────────────────────────────────────
# Raster / image analysis
# ─────────────────────────────────────────────────────────────────────────────

def _raster_bbox(img_gray: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """
    Find the floor plan bounding box in a grayscale numpy image.
    Returns (x, y, w, h) in image pixels, or None on failure.
    """
    return _cv_bbox(img_gray) if HAS_CV2 else _pil_bbox(img_gray)


def _cv_bbox(img_gray: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """
    OpenCV pipeline:
      1. Binarize (content = bright, background = dark)
      2. Connect nearby elements with H + V morphological closing
      3. Large closing to merge the floor plan into one blob
      4. Connected components → largest = floor plan
      5. Tighten to original-binary content within that blob
    """
    ih, iw = img_gray.shape

    # Content = 255, background = 0
    _, binary = cv2.threshold(img_gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Noise removal
    k2 = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k2)

    # Connect nearby elements along rows and columns
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, iw // 25), 1))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(5, ih // 25)))
    merged = cv2.bitwise_or(
        cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kh),
        cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kv),
    )

    # Large closing: merge the floor plan into one region
    kL = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(5, iw // 12), max(5, ih // 12))
    )
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kL)

    # Connected components
    num, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    if num < 2:
        return None

    # Score: prefer large area with reasonable aspect ratio
    def _score(i: int) -> float:
        a = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        ar = w / h if h else 1.0
        ar_ok = 1.0 if 0.2 <= ar <= 5.0 else 0.4
        return (a / (iw * ih)) * ar_ok

    best = max(range(1, num), key=_score)

    bx = stats[best, cv2.CC_STAT_LEFT]
    by = stats[best, cv2.CC_STAT_TOP]
    bw = stats[best, cv2.CC_STAT_WIDTH]
    bh = stats[best, cv2.CC_STAT_HEIGHT]

    # Tighten to original binary within the blob region
    roi = cleaned[by:by + bh, bx:bx + bw]
    rows = roi.any(axis=1)
    cols = roi.any(axis=0)
    if rows.any() and cols.any():
        ry0 = int(np.argmax(rows))
        ry1 = int(bh - 1 - np.argmax(rows[::-1]))
        cx0 = int(np.argmax(cols))
        cx1 = int(bw - 1 - np.argmax(cols[::-1]))
        bx, by = bx + cx0, by + ry0
        bw, bh = cx1 - cx0 + 1, ry1 - ry0 + 1

    return (bx, by, bw, bh)


def _pil_bbox(img_gray: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """
    Pure NumPy fallback when OpenCV is not available.

    Content is first binarized onto a coarse density grid (matching
    _vector_bbox's approach) and reduced to its largest CONNECTED component
    via _largest_component_bbox_numpy, instead of a plain row/column "any
    content" projection — the latter merges a scanned floor plan with any
    disconnected peripheral block (legend, title block, stamp) that happens
    to be dark enough to pass the threshold, even across a clear white gap.
    """
    ih, iw = img_gray.shape
    dark = img_gray < 240  # True where content

    cell = max(3, round(max(iw, ih) / 200))
    gw = -(-iw // cell)
    gh = -(-ih // cell)
    density = np.zeros((gh, gw), dtype=np.float32)
    for gy in range(gh):
        y0, y1 = gy * cell, min(ih, (gy + 1) * cell)
        row_slice = dark[y0:y1, :]
        for gx in range(gw):
            x0, x1 = gx * cell, min(iw, (gx + 1) * cell)
            region = row_slice[:, x0:x1]
            density[gy, gx] = region.mean() if region.size else 0.0

    if density.max() <= 0:
        return None
    binary = (density >= 0.035).astype(np.uint8)
    bbox = _largest_component_bbox_numpy(binary)
    if bbox is None:
        return None
    gx0, gy0, gx1, gy1 = bbox

    # Map grid bbox back to pixel space, then tighten to actual dark-pixel
    # extents within it (the grid is coarse, so this trims leftover margin).
    px0, py0 = gx0 * cell, gy0 * cell
    px1, py1 = min(iw, (gx1 + 1) * cell), min(ih, (gy1 + 1) * cell)
    region = dark[py0:py1, px0:px1]
    rows = np.where(region.any(axis=1))[0]
    cols = np.where(region.any(axis=0))[0]
    if len(rows) == 0:
        return (px0, py0, px1 - px0, py1 - py0)
    y0, y1 = py0 + int(rows[0]), py0 + int(rows[-1])
    x0, x1 = px0 + int(cols[0]), px0 + int(cols[-1])
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


# ─────────────────────────────────────────────────────────────────────────────
# Edge refinement — removes peripheral info blocks (legends / title blocks)
# ─────────────────────────────────────────────────────────────────────────────

def _refine_edges(
    img_gray: np.ndarray,
    clip: fitz.Rect,
    page_rect,
    analysis_scale: float,
) -> fitz.Rect:
    """
    Scan each edge of the detected floor plan region inward.

    A "drawing border" row/col is the first one from the outside with
    pixel density ≥ DENSE.  Everything outside that border (legend text,
    info blocks) is discarded.  We limit the scan to MAX_STRIP of the
    detected span to guarantee we never eat into the main drawing.

    Why density instead of span:
        Legend rows often have items at both horizontal extremes (large
        span despite sparse content).  Density is immune to that pattern.
    """
    DENSE     = 0.05   # ≥ 5 % of sampled pixels dark → "real content"
    MAX_STRIP = 0.28   # never remove more than 28 % from any single edge

    ih, iw = img_gray.shape
    pw = float(page_rect.width)
    ph = float(page_rect.height)

    # Convert PDF clip to analysis-image pixel coordinates
    def pt_to_px(v: float, dim: float, pxdim: int) -> int:
        return max(0, min(pxdim - 1, int(round((v / dim) * pxdim))))

    px_l = pt_to_px(clip.x0, pw, iw)
    px_r = pt_to_px(clip.x1, pw, iw)
    px_t = pt_to_px(clip.y0, ph, ih)
    px_b = pt_to_px(clip.y1, ph, ih)

    if px_r <= px_l or px_b <= px_t:
        return clip

    binary = img_gray < 240  # True where dark

    def row_density(y: int, x0: int, x1: int) -> float:
        if x1 <= x0:
            return 0.0
        step = max(1, (x1 - x0) // 400)
        col_range = range(x0, x1, step)
        total = len(col_range)
        if total == 0:
            return 0.0
        dark = int(binary[y, x0:x1:step].sum())
        return dark / total

    def col_density(x: int, y0: int, y1: int) -> float:
        if y1 <= y0:
            return 0.0
        step = max(1, (y1 - y0) // 400)
        row_range = range(y0, y1, step)
        total = len(row_range)
        if total == 0:
            return 0.0
        dark = int(binary[y0:y1:step, x].sum())
        return dark / total

    cH = px_b - px_t
    cW = px_r - px_l
    max_h = max(1, int(cH * MAX_STRIP))
    max_w = max(1, int(cW * MAX_STRIP))
    mid_y = (px_t + px_b) // 2
    mid_x = (px_l + px_r) // 2

    def find_row(from_y: int, limit_y: int) -> int:
        step = 1 if limit_y >= from_y else -1
        for y in range(from_y, limit_y + step, step):
            if 0 <= y < ih and row_density(y, px_l, px_r) >= DENSE:
                return y
        return from_y

    def find_col(from_x: int, limit_x: int, y0: int, y1: int) -> int:
        step = 1 if limit_x >= from_x else -1
        for x in range(from_x, limit_x + step, step):
            if 0 <= x < iw and col_density(x, y0, y1) >= DENSE:
                return x
        return from_x

    # Scan each edge inward, clamped so we never exceed 50 % of the span
    new_b = find_row(px_b, max(mid_y, px_b - max_h))
    new_t = find_row(px_t, min(mid_y, px_t + max_h))
    new_r = find_col(px_r, max(mid_x, px_r - max_w), new_t, new_b)
    new_l = find_col(px_l, min(mid_x, px_l + max_w), new_t, new_b)

    # Convert back to PDF points
    def px_to_pt(px: int, pxdim: int, ptdim: float) -> float:
        return (px / pxdim) * ptdim

    refined = fitz.Rect(
        max(0.0, px_to_pt(new_l, iw, pw)),
        max(0.0, px_to_pt(new_t, ih, ph)),
        min(pw,  px_to_pt(new_r, iw, pw)),
        min(ph,  px_to_pt(new_b, ih, ph)),
    )

    # Safety: if refinement produces an absurdly small region, fall back
    if refined.width < clip.width * 0.3 or refined.height < clip.height * 0.3:
        logger.debug("Edge refinement produced too-small clip; keeping original")
        return clip

    return refined


# ─────────────────────────────────────────────────────────────────────────────
# Image (non-PDF) crop — no PyMuPDF required
# ─────────────────────────────────────────────────────────────────────────────

def crop_floorplan_from_image(image_bytes: bytes) -> CropResult:
    """
    Detect and crop the floor plan area from a raster image (PNG, JPEG, etc.).
    Does not require PyMuPDF. Uses Pillow + NumPy (+ OpenCV when available).
    """
    try:
        img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot open image: {exc}") from exc

    img_gray = np.array(img_pil.convert("L"))
    ih, iw = img_gray.shape

    bbox = _raster_bbox(img_gray)
    if bbox is None:
        bbox = (0, 0, iw, ih)

    x, y, w, h = bbox
    clip_px = (x, y, x + w, y + h)
    refined = _refine_edges_px(img_gray, clip_px)
    rx0, ry0, rx1, ry1 = refined

    # If refinement yielded almost the full image, fall back to raw bbox trim
    if (rx1 - rx0) >= iw * 0.99 and (ry1 - ry0) >= ih * 0.99:
        rx0, ry0, rx1, ry1 = x, y, x + w, y + h

    cropped = img_pil.crop((rx0, ry0, rx1, ry1))
    bg = Image.new("RGB", cropped.size, (255, 255, 255))
    bg.paste(cropped)

    method = "raster_cv2" if HAS_CV2 else "raster_pil"
    return CropResult(image=bg, method=method, confidence=0.65)


def _refine_edges_px(
    img_gray: np.ndarray,
    clip_px: tuple,  # (x0, y0, x1, y1)
) -> tuple:
    """Pixel-space edge refinement — strips peripheral info blocks without
    needing PyMuPDF coordinate types."""
    DENSE = 0.05
    MAX_STRIP = 0.28

    ih, iw = img_gray.shape
    px_l, px_t, px_r, px_b = clip_px

    if px_r <= px_l or px_b <= px_t:
        return clip_px

    binary = img_gray < 240

    def row_density(y: int, x0: int, x1: int) -> float:
        if x1 <= x0:
            return 0.0
        step = max(1, (x1 - x0) // 400)
        total = len(range(x0, x1, step))
        return int(binary[y, x0:x1:step].sum()) / total if total > 0 else 0.0

    def col_density(x: int, y0: int, y1: int) -> float:
        if y1 <= y0:
            return 0.0
        step = max(1, (y1 - y0) // 400)
        total = len(range(y0, y1, step))
        return int(binary[y0:y1:step, x].sum()) / total if total > 0 else 0.0

    cH = px_b - px_t
    cW = px_r - px_l
    max_h = max(1, int(cH * MAX_STRIP))
    max_w = max(1, int(cW * MAX_STRIP))
    mid_y = (px_t + px_b) // 2
    mid_x = (px_l + px_r) // 2

    def find_row(from_y: int, limit_y: int) -> int:
        step = 1 if limit_y >= from_y else -1
        for y in range(from_y, limit_y + step, step):
            if 0 <= y < ih and row_density(y, px_l, px_r) >= DENSE:
                return y
        return from_y

    def find_col(from_x: int, limit_x: int, y0: int, y1: int) -> int:
        step = 1 if limit_x >= from_x else -1
        for x in range(from_x, limit_x + step, step):
            if 0 <= x < iw and col_density(x, y0, y1) >= DENSE:
                return x
        return from_x

    new_b = find_row(px_b, max(mid_y, px_b - max_h))
    new_t = find_row(px_t, min(mid_y, px_t + max_h))
    new_r = find_col(px_r, max(mid_x, px_r - max_w), new_t, new_b)
    new_l = find_col(px_l, min(mid_x, px_l + max_w), new_t, new_b)

    if (new_r - new_l) < (px_r - px_l) * 0.3 or (new_b - new_t) < (px_b - px_t) * 0.3:
        return clip_px

    return (new_l, new_t, new_r, new_b)


# ─────────────────────────────────────────────────────────────────────────────
# Shared utility
# ─────────────────────────────────────────────────────────────────────────────

def _largest_component_bbox_numpy(binary: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """
    Pure-NumPy/BFS connected-component bounding box (used when OpenCV isn't
    installed — the default in this project's devcontainer, since opencv is
    only in requirements-optional.txt). The grids passed in here are always
    small (tens to a few hundred cells per side), so a plain iterative flood
    fill is fast — no need for scipy.

    A prior version of this fallback returned the bounding box of *every*
    nonzero cell (via np.any per row/col) instead of an actual connected
    component. That silently merged a floor plan with any disconnected
    peripheral block (legend, title block) that happened to be dense enough
    to pass the threshold, defeating the whole point of "largest component"
    — the legend would ride along in the final crop even though a clear
    white gap separates it from the drawing.
    """
    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    best: Optional[tuple[int, int, int, int]] = None
    best_size = 0

    for start_y in range(height):
        for start_x in range(width):
            if not binary[start_y, start_x] or visited[start_y, start_x]:
                continue
            stack = [(start_y, start_x)]
            visited[start_y, start_x] = True
            min_x = max_x = start_x
            min_y = max_y = start_y
            size = 0
            while stack:
                y, x = stack.pop()
                size += 1
                if x < min_x: min_x = x
                if x > max_x: max_x = x
                if y < min_y: min_y = y
                if y > max_y: max_y = y
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and binary[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if size > best_size:
                best_size = size
                best = (min_x, min_y, max_x, max_y)

    return best


def _largest_component_bbox(
    binary: np.ndarray,
) -> Optional[tuple[int, int, int, int]]:
    """
    Return grid (x0, y0, x1, y1) of the largest connected component in a
    binary NumPy array.  Uses OpenCV when available, pure NumPy/BFS fallback
    otherwise (see _largest_component_bbox_numpy).
    """
    if HAS_CV2:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        if num < 2:
            return None
        i = max(range(1, num), key=lambda j: stats[j, cv2.CC_STAT_AREA])
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        return (x, y, x + w - 1, y + h - 1)

    bbox = _largest_component_bbox_numpy(binary)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    return (x0, y0, x1, y1)
