"""
Wall segment detection and material classification for floor plans.

Supports two input modes:
- PDF (vector): uses PyMuPDF to extract line geometry, widths, colors, and
  text annotations, which give accurate wall boundaries and material hints.
- Raster image: uses OpenCV Hough lines + perpendicular thickness sampling +
  hatch pattern analysis to estimate wall positions and materials.

Material classification draws on:
  - Line/stroke width (thickness proxy)
  - Stroke color (CAD layer color conventions)
  - Fill/hatch pattern between double-line walls
  - Nearby text labels (Portuguese/Brazilian architectural keywords)
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ── Material keyword table (Portuguese / Brazilian architectural terms) ───────

_MATERIAL_KEYWORDS: dict[str, list[str]] = {
    "brick": [
        "tijolo", "alvenaria", "bloco ceramico", "bloco de ceramica",
        "bloco sil", "bloco de concreto", "blocos",
    ],
    "concrete": [
        "concreto simples", "concreto magro", "beton",
        "conc.s", "concreto s/", "concreto nao",
    ],
    "reinforced_concrete": [
        "concreto armado", "conc. arm", "c.a.", "estrutural",
        "pilar", "viga", "laje", "estrutura de concreto",
    ],
    "drywall": [
        "drywall", "gesso acartonado", "steel frame",
        "gesboard", "pladur", "placa de gesso",
    ],
    "glass": [
        "vidro", "glass", "caixilho", "esquadria de vidro",
        "bloco de vidro", "janela de vidro",
    ],
    "wood": [
        "madeira", "mdf", "osb", "compensado",
        "pinus", "eucalipto", "estrutura de madeira",
    ],
    "stone": [
        "pedra", "granito", "marmore", "ardosia",
        "calcario", "stone", "revestimento de pedra",
    ],
    "metal": [
        "metal", "ferro", "aco", "chapa",
        "aluminio", "perfil metalico", "estrutura metalica",
    ],
}


@dataclass
class WallSegment:
    x1: float
    y1: float
    x2: float
    y2: float
    material: str
    confidence: float
    thickness_px: float = 1.0
    wall_id: str = ""


# ── Material detection helpers ────────────────────────────────────────────────

def _keyword_material(text: str) -> Optional[str]:
    lower = text.lower()
    for material, keywords in _MATERIAL_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return material
    return None


def _classify_from_pdf_props(
    width_pt: float,
    color: tuple,
    is_dashed: bool,
) -> tuple[str, float]:
    """Classify material from PDF path properties (width, color, dash style)."""
    r, g, b = (float(color[0]), float(color[1]), float(color[2])) if len(color) >= 3 else (0.0, 0.0, 0.0)

    # Color-based detection: blue/cyan → glass
    if b > 0.45 and b > r + 0.18 and b > g + 0.05:
        return "glass", 0.72
    if g > 0.45 and b > 0.30 and r < 0.30:
        return "glass", 0.62

    # Red/orange → brick
    if r > 0.50 and g < 0.38 and b < 0.32:
        return "brick", 0.68

    # Yellow/warm brown → wood
    if r > 0.45 and 0.30 <= g <= 0.65 and b < 0.26:
        return "wood", 0.60

    # Dashed thin lines → glass curtain wall or window annotation
    if is_dashed and width_pt < 1.2:
        return "glass", 0.44

    # Thickness classification (PDF point widths, approx):
    #   0-0.4pt  →  glass / thin annotation
    #   0.5-0.9  →  drywall
    #   1.0-1.9  →  brick / light masonry
    #   2.0-3.9  →  concrete block
    #   4.0+     →  reinforced concrete
    if width_pt < 0.45:
        return "glass", 0.44
    if width_pt < 0.90:
        return "drywall", 0.50
    if width_pt < 2.0:
        return "brick", 0.48
    if width_pt < 4.0:
        return "concrete", 0.52
    return "reinforced_concrete", 0.56


# ── PDF analysis ─────────────────────────────────────────────────────────────

def analyze_walls_from_pdf(
    pdf_bytes: bytes,
    page_index: int = 0,
    render_width: int = 1000,
) -> tuple[list[WallSegment], int, int]:
    """
    Extract wall segments from a PDF using PyMuPDF vector graphics.

    Returns (walls, output_width, output_height) where coordinates are in
    pixel space scaled to render_width.
    """
    if not HAS_FITZ:
        raise RuntimeError("PyMuPDF not installed — run: pip install pymupdf")

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if page_index >= doc.page_count:
        page_index = 0
    page = doc[page_index]

    pw, ph = page.rect.width, page.rect.height
    scale = render_width / max(pw, 1)
    render_height = int(ph * scale)

    # Build a spatial index of material keywords from text blocks
    text_materials: list[tuple[float, float, float, float, str]] = []
    for block in page.get_text("blocks"):
        x0, y0, x1, y1, text = block[0], block[1], block[2], block[3], block[4]
        mat = _keyword_material(text)
        if mat:
            text_materials.append((x0 * scale, y0 * scale, x1 * scale, y1 * scale, mat))

    def _text_material_near(mid_x: float, mid_y: float, radius: float = 90.0) -> Optional[str]:
        for bx0, by0, bx1, by1, mat in text_materials:
            cx, cy = (bx0 + bx1) * 0.5, (by0 + by1) * 0.5
            if math.hypot(mid_x - cx, mid_y - cy) <= radius:
                return mat
        return None

    min_len_sq = (max(20, render_width // 55)) ** 2
    walls: list[WallSegment] = []

    for idx, drawing in enumerate(page.get_drawings()):
        color = drawing.get("color") or drawing.get("stroke_color") or (0.0, 0.0, 0.0)
        width_pt = float(drawing.get("width") or 0.5)
        dashes = bool(drawing.get("dashes"))

        for seg_idx, item in enumerate(drawing.get("items", [])):
            if item[0] != "l":
                continue

            p1, p2 = item[1], item[2]
            x1 = float(p1.x) * scale
            y1 = float(p1.y) * scale
            x2 = float(p2.x) * scale
            y2 = float(p2.y) * scale

            dx, dy = x2 - x1, y2 - y1
            if dx * dx + dy * dy < min_len_sq:
                continue

            mid_x, mid_y = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            text_mat = _text_material_near(mid_x, mid_y)

            if text_mat:
                material, confidence = text_mat, 0.85
            else:
                material, confidence = _classify_from_pdf_props(width_pt, color, dashes)

            thickness_px = round(width_pt * scale * 0.38, 1)

            walls.append(WallSegment(
                x1=round(x1, 1), y1=round(y1, 1),
                x2=round(x2, 1), y2=round(y2, 1),
                material=material, confidence=confidence,
                thickness_px=thickness_px,
                wall_id=f"pdf-{idx}-{seg_idx}",
            ))

    doc.close()
    walls = _dedupe_segments(walls, render_width, render_height)
    return walls, render_width, render_height


# ── Raster image analysis ─────────────────────────────────────────────────────

def analyze_walls_from_image(
    image_bytes: bytes,
) -> tuple[list[WallSegment], int, int]:
    """
    Detect wall segments from a raster image using OpenCV Hough lines.

    Falls back to PIL+NumPy edge detection when OpenCV is unavailable.
    """
    if not HAS_CV2:
        raise RuntimeError("OpenCV not installed — run: pip install opencv-python-headless")

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection
    blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=60, sigmaSpace=60)
    edges = cv2.Canny(blurred, 25, 80, apertureSize=3)

    min_len = max(25, min(width, height) // 18)
    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(25, min(width, height) // 35),
        minLineLength=min_len,
        maxLineGap=max(6, min_len // 5),
    )

    if raw_lines is None:
        return [], width, height

    walls: list[WallSegment] = []
    for i, line in enumerate(raw_lines):
        x1, y1, x2, y2 = int(line[0][0]), int(line[0][1]), int(line[0][2]), int(line[0][3])
        thickness = _measure_thickness(gray, x1, y1, x2, y2)
        hatch = _analyze_hatch(gray, x1, y1, x2, y2, thickness)
        color_rgb = _sample_color(img, x1, y1, x2, y2)
        material, confidence = _classify_image_features(thickness, hatch, color_rgb, width, height)

        walls.append(WallSegment(
            x1=float(x1), y1=float(y1),
            x2=float(x2), y2=float(y2),
            material=material, confidence=confidence,
            thickness_px=round(thickness, 1),
            wall_id=f"img-{i}",
        ))

    return _dedupe_segments(walls, width, height), width, height


def _measure_thickness(gray: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    """Estimate wall thickness (px) by scanning perpendicular to the line segment."""
    h, w = gray.shape
    dx, dy = float(x2 - x1), float(y2 - y1)
    length = math.hypot(dx, dy)
    if length < 1:
        return 1.0
    # Unit perpendicular vector
    px, py = -dy / length, dx / length
    mx, my = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    dark = sum(
        1
        for d in range(-24, 25)
        if 0 <= int(mx + px * d) < w
        and 0 <= int(my + py * d) < h
        and int(gray[int(my + py * d), int(mx + px * d)]) < 140
    )
    return max(1.0, float(dark))


def _analyze_hatch(
    gray: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    thickness: float,
) -> str:
    """
    Detect fill pattern in the region around the wall line.

    Returns: "none" | "sparse" | "medium" | "dense"
    Mapping to materials:
      none   → glass / drywall (empty space between double lines)
      sparse → brick (45° hatching per NBR 6492)
      medium → concrete (cross-hatching)
      dense  → reinforced concrete / metal (solid fill)
    """
    h, w = gray.shape
    mx, my = int((x1 + x2) * 0.5), int((y1 + y2) * 0.5)
    ps = max(8, int(thickness * 1.6))
    patch = gray[
        max(0, my - ps): min(h, my + ps),
        max(0, mx - ps): min(w, mx + ps),
    ]
    if patch.size == 0:
        return "none"

    ratio = float(np.sum(patch < 140)) / patch.size

    if ratio < 0.04:
        return "none"
    if ratio < 0.18:
        return "sparse"
    if ratio < 0.40:
        return "medium"
    return "dense"


def _sample_color(img_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int]:
    """Average BGR-to-RGB color sampled along the line."""
    h, w = img_bgr.shape[:2]
    dx, dy = x2 - x1, y2 - y1
    steps = max(4, int(math.hypot(dx, dy) / 20))
    samples = []
    for i in range(steps):
        t = i / max(steps - 1, 1)
        sx, sy = int(x1 + dx * t), int(y1 + dy * t)
        if 0 <= sx < w and 0 <= sy < h:
            b, g, r = img_bgr[sy, sx]
            samples.append((int(r), int(g), int(b)))
    if not samples:
        return (0, 0, 0)
    r = sum(s[0] for s in samples) // len(samples)
    g = sum(s[1] for s in samples) // len(samples)
    b = sum(s[2] for s in samples) // len(samples)
    return (r, g, b)


def _classify_image_features(
    thickness: float,
    hatch: str,
    color_rgb: tuple[int, int, int],
    img_width: int,
    img_height: int,
) -> tuple[str, float]:
    """
    Classify wall material from visual features.

    Priority: color hint > hatch pattern > thickness ratio.
    """
    r, g, b = color_rgb
    max_dim = max(img_width, img_height)
    rel_t = thickness / max_dim

    # Strong color signals
    if b > 150 and b > r + 35:
        return "glass", 0.66
    if r > 155 and r > b + 65 and r > g + 45:
        return "brick", 0.62

    # Hatch + thickness combined
    if hatch == "none":
        if thickness <= 3:
            return "glass", 0.56
        if thickness <= 5:
            return "drywall", 0.52
        if rel_t < 0.010:
            return "brick", 0.44
        return "brick", 0.40

    if hatch == "sparse":
        # Sparse hatching = brick (45° diagonal per NBR 6492)
        if thickness <= 8:
            return "brick", 0.58
        return "concrete", 0.50

    if hatch == "medium":
        # Cross-hatching = concrete
        if rel_t > 0.018:
            return "reinforced_concrete", 0.55
        return "concrete", 0.55

    # "dense"
    if rel_t > 0.020:
        return "reinforced_concrete", 0.54
    return "metal", 0.46


# ── Shared post-processing ────────────────────────────────────────────────────

def _dedupe_segments(
    walls: list[WallSegment],
    width: int,
    height: int,
) -> list[WallSegment]:
    """Remove near-duplicate wall segments (same midpoint + similar length)."""
    accepted: list[WallSegment] = []
    tol = max(8, min(width, height) // 90)

    for wall in walls:
        mx = (wall.x1 + wall.x2) * 0.5
        my = (wall.y1 + wall.y2) * 0.5
        length = math.hypot(wall.x2 - wall.x1, wall.y2 - wall.y1)

        dup = False
        for ex in accepted:
            emx = (ex.x1 + ex.x2) * 0.5
            emy = (ex.y1 + ex.y2) * 0.5
            el = math.hypot(ex.x2 - ex.x1, ex.y2 - ex.y1)
            if math.hypot(mx - emx, my - emy) < tol and abs(length - el) < max(tol, length * 0.22):
                dup = True
                if wall.confidence > ex.confidence:
                    ex.material = wall.material
                    ex.confidence = wall.confidence
                break

        if not dup:
            accepted.append(wall)

    return accepted
