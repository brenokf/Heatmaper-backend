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


def detect_embedded_measurement_markers(image: Image.Image) -> list[Marker]:
    rgb = np.asarray(image.convert("RGB"))
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    red_mask = (red > 170) & (green < 135) & (blue < 110) & ((red - green) > 55) & ((red - blue) > 75)
    marker_mask = dilate_mask(red_mask, iterations=4)
    markers: list[Marker] = []

    for area, x, y, w, h in connected_components(marker_mask):
        if not (80 <= area <= 5200 and 15 <= w <= 92 and 15 <= h <= 92):
            continue
        aspect = w / max(h, 1)
        if 0.55 <= aspect <= 1.85:
            markers.append(Marker(x + w / 2, y + h / 2, area))

    return markers
