from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

try:
    import cv2
except Exception:  # pragma: no cover - OpenCV is optional at runtime.
    cv2 = None


@dataclass
class Marker:
    x: float
    y: float
    area: int


@dataclass
class MapProfile:
    key: str
    label: str
    marker_radius: float
    mask_padding_ratio: float
    foreground_threshold: float


MAP_PROFILES = {
    "clean": MapProfile("clean", "planta baixa limpa", 110, 0.016, 30),
    "screenshot": MapProfile("screenshot", "screenshot de planta", 170, 0.022, 42),
    "wide": MapProfile("wide", "mapa/planta larga", 190, 0.020, 38),
    "dense": MapProfile("dense", "planta tecnica detalhada", 135, 0.014, 48),
}


def marker_limits(width: int, height: int) -> tuple[int, float]:
    area = width * height
    max_points = min(80, max(8, round((area**0.5) / 18)))
    min_distance = min(62.0, max(18.0, max(width, height) * 0.018))
    return max_points, min_distance


def dedupe_markers(markers: list[Marker], width: int, height: int) -> list[Marker]:
    max_points, min_distance = marker_limits(width, height)
    accepted: list[Marker] = []
    for marker in sorted(markers, key=lambda item: item.area, reverse=True):
        duplicate = any(((marker.x - other.x) ** 2 + (marker.y - other.y) ** 2) ** 0.5 < min_distance for other in accepted)
        if not duplicate:
            accepted.append(marker)
        if len(accepted) >= max_points:
            break
    return sorted(accepted, key=lambda item: (item.y, item.x))


def estimate_background(rgb: np.ndarray) -> np.ndarray:
    h, w, _ = rgb.shape
    samples = np.concatenate([rgb[0, :, :], rgb[h - 1, :, :], rgb[:, 0, :], rgb[:, w - 1]], axis=0)
    return samples.mean(axis=0)


def saturation(rgb: np.ndarray) -> np.ndarray:
    max_v = rgb.max(axis=2).astype(np.float32)
    min_v = rgb.min(axis=2).astype(np.float32)
    return np.divide(max_v - min_v, np.maximum(max_v, 1), out=np.zeros_like(max_v), where=max_v > 0)


def analyze_map(image: Image.Image) -> MapProfile:
    rgb = np.asarray(image.convert("RGB"))
    h, w, _ = rgb.shape
    bg = estimate_background(rgb)
    step = max(1, max(w, h) // 360)
    sampled = rgb[::step, ::step, :].astype(np.float32)
    contrast = np.linalg.norm(sampled - bg, axis=2)
    lum = (0.2126 * sampled[:, :, 0] + 0.7152 * sampled[:, :, 1] + 0.0722 * sampled[:, :, 2]) / 255
    sat = saturation(sampled.astype(np.uint8))
    foreground_ratio = float(((contrast > 34) | (lum < 0.72)).mean())
    colored_ratio = float(((sat > 0.28) & (contrast > 38)).mean())
    dark_ratio = float((lum < 0.42).mean())
    aspect = w / max(h, 1)

    if aspect > 1.65 or aspect < 0.65:
        return MAP_PROFILES["wide"]
    if colored_ratio > 0.09:
        return MAP_PROFILES["screenshot"]
    if foreground_ratio > 0.34 or dark_ratio > 0.18:
        return MAP_PROFILES["dense"]
    return MAP_PROFILES["clean"]


def detect_markers(image: Image.Image, profile: MapProfile) -> list[Marker]:
    rgb = np.asarray(image.convert("RGB"))
    bg = estimate_background(rgb)
    contrast = np.linalg.norm(rgb.astype(np.float32) - bg, axis=2)
    sat = saturation(rgb)
    value = rgb.max(axis=2) / 255
    marker_mask = (sat > 0.34) & (value > 0.28) & (contrast > 45)

    if cv2 is None:
        return []

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(marker_mask.astype(np.uint8), 8)
    markers: list[Marker] = []
    for label in range(1, num):
        x, y, w, h, area = stats[label]
        if area < 18 or area > 2800 or w < 5 or h < 5 or w > 76 or h > 76:
            continue
        aspect = w / max(h, 1)
        density = area / max(w * h, 1)
        if 0.45 <= aspect <= 2.2 and 0.12 <= density <= 0.92:
            cx, cy = centroids[label]
            markers.append(Marker(float(cx), float(cy), int(area)))
    return markers


def dilate_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    output = mask.astype(bool)
    for _ in range(iterations):
        source = output
        expanded = source.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                src_y0 = max(0, -dy)
                src_y1 = source.shape[0] - max(0, dy)
                src_x0 = max(0, -dx)
                src_x1 = source.shape[1] - max(0, dx)
                dst_y0 = max(0, dy)
                dst_y1 = expanded.shape[0] - max(0, -dy)
                dst_x0 = max(0, dx)
                dst_x1 = expanded.shape[1] - max(0, -dx)
                expanded[dst_y0:dst_y1, dst_x0:dst_x1] |= source[src_y0:src_y1, src_x0:src_x1]
        output = expanded
    return output


def connected_components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    components: list[tuple[int, int, int, int, int]] = []
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1))

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
                continue
            stack = [(x, y)]
            seen[y, x] = True
            min_x = max_x = x
            min_y = max_y = y
            area = 0
            while stack:
                cx, cy = stack.pop()
                area += 1
                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)
                for dx, dy in neighbors:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((nx, ny))
            components.append((area, min_x, min_y, max_x - min_x + 1, max_y - min_y + 1))

    return components


def enhance_floorplan_bw(image: Image.Image, high_res: bool = False) -> Image.Image:
    """
    Convert a floor plan image to a clean black-and-white version, close to
    just the room contours: walls stay, flooring hatch/tile-grid patterns
    (thin, repetitive, disconnected from the wall network) are stripped.

    Pipeline:
    1. Grayscale
    2. CLAHE – local contrast boost so faint walls survive thresholding
    3. Gaussian blur – remove rasterisation noise
    4. Adaptive threshold (Gaussian) – handles uneven background brightness
    5. Morphological close – bridge tiny gaps inside wall segments
    6. Morphological open, kernel scaled to resolution – erase thin hatching/
       tile-grid lines and isolated specks while keeping thicker wall strokes
    7. Connected-component area filter – drop any remaining small speckle
       component not part of the large wall network
    8. Ensure black-on-white output
    """
    if cv2 is None:
        return image.convert("L").point(lambda x: 0 if x < 200 else 255, "L")

    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    max_dim = max(width, height)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # CLAHE boosts local contrast so walls with low contrast survive thresholding
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Light denoise – 3×3 keeps fine wall lines intact
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Adaptive threshold: each pixel is compared against its local neighbourhood.
    # blockSize scales with resolution so it stays ~a few real-world cm wide;
    # C=10 keeps faint elements visible.
    block_size = max(11, (round(max_dim / 90) | 1))  # odd, ~ same physical size at any DPI
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=block_size,
        C=10,
    )

    # Close tiny gaps inside wall lines. Kernel scales with resolution so a
    # ~5000px-wide render (vector-PDF crop at 250 DPI) gets a proportionally
    # larger kernel than a small PNG upload, instead of a fixed 2px no-op.
    close_size = max(2, round(max_dim / 1200))
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)

    # Opening with a resolution-scaled kernel erases thin repeated hatching
    # (flooring tile grids, material fill patterns) and isolated noise dots,
    # while thicker wall strokes survive since they're wider than the kernel.
    open_size = max(2, round(max_dim / 700))
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (open_size, open_size))
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_open)

    # Drop any remaining small speckle component (fragments of hatching that
    # survived opening) that isn't part of the large connected wall network.
    # Foreground polarity varies per source, so filter whichever value is the
    # minority (the "ink"), not hardcoded black-on-white.
    black_pixels = int((cleaned == 0).sum())
    white_pixels = int((cleaned == 255).sum())
    ink_value = 0 if black_pixels <= white_pixels else 255
    ink_mask = (cleaned == ink_value).astype(np.uint8)
    min_component_area = max(24, round((max_dim**2) * 0.000006))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(ink_mask, 8)
    filtered_ink = np.zeros_like(ink_mask)
    for label in range(1, num):
        if stats[label, cv2.CC_STAT_AREA] >= min_component_area:
            filtered_ink[labels == label] = 1
    cleaned = np.where(filtered_ink == 1, ink_value, 255 - ink_value).astype(np.uint8)

    # Ensure background is white (most plans: dark lines on light BG)
    white_pixels = int((cleaned == 255).sum())
    black_pixels = int((cleaned == 0).sum())
    if black_pixels > white_pixels:
        cleaned = cv2.bitwise_not(cleaned)

    return Image.fromarray(cleaned, mode="L")


def detect_embedded_measurement_markers(image: Image.Image) -> list[Marker]:
    rgb = np.asarray(image.convert("RGB"))
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    red_mask = (red > 170) & (green < 135) & (blue < 110) & ((red - green) > 55) & ((red - blue) > 75)
    marker_mask = dilate_mask(red_mask, iterations=4)
    markers: list[Marker] = []

    height, width = red_mask.shape
    image_area = width * height
    max_box = min(118, max(38, round(max(width, height) * 0.055)))
    min_box = min(18, max(6, round(max(width, height) * 0.004)))
    min_area = max(18, round(image_area * 0.000006))
    max_area = max(700, min(6500, round(image_area * 0.0022)))

    for area, x, y, w, h in connected_components(marker_mask):
        if not (min_area <= area <= max_area and min_box <= w <= max_box and min_box <= h <= max_box):
            continue
        aspect = w / max(h, 1)
        density = area / max(w * h, 1)
        long_axis = max(w, h)
        short_axis = min(w, h)
        long_thin_annotation = long_axis > max(width, height) * 0.08 and short_axis <= max(8, max(width, height) * 0.01)
        if 0.5 <= aspect <= 2 and 0.045 <= density <= 0.96 and not long_thin_annotation:
            markers.append(Marker(x + w / 2, y + h / 2, area))

    return dedupe_markers(markers, width, height)
