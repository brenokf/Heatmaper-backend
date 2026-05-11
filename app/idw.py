from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


DEFAULT_IDW_POWER = 2.2
IDW_ZERO_DISTANCE_EPSILON = 1e-9


@dataclass(frozen=True)
class IDWPoint:
    point_id: str
    x_px: float
    y_px: float
    rssi: float
    distance_m: float | None = None


def euclidean_distance_px(x: float, y: float, point: IDWPoint) -> float:
    return math.hypot(x - point.x_px, y - point.y_px)


def idw_weight(distance_px: float, power: float = DEFAULT_IDW_POWER) -> float:
    if distance_px <= IDW_ZERO_DISTANCE_EPSILON:
        return math.inf
    return 1.0 / (distance_px**power)


def idw_interpolate(x: float, y: float, points: Sequence[IDWPoint], power: float = DEFAULT_IDW_POWER) -> float:
    if not points:
        raise ValueError("IDW requires at least one measurement point.")

    weighted_sum = 0.0
    weight_sum = 0.0
    for point in points:
        distance = euclidean_distance_px(x, y, point)
        if distance <= IDW_ZERO_DISTANCE_EPSILON:
            return float(point.rssi)
        weight = idw_weight(distance, power)
        weighted_sum += weight * point.rssi
        weight_sum += weight

    if weight_sum == 0:
        raise ValueError("IDW weight sum is zero.")
    return weighted_sum / weight_sum


def validate_points_inside_image(points: Iterable[IDWPoint], width: int, height: int) -> list[str]:
    errors: list[str] = []
    for point in points:
        if not (0 <= point.x_px < width):
            errors.append(f"{point.point_id}: x_px={point.x_px} fora da largura 0..{width - 1}")
        if not (0 <= point.y_px < height):
            errors.append(f"{point.point_id}: y_px={point.y_px} fora da altura 0..{height - 1}")
    return errors


def create_idw_grid(
    width: int,
    height: int,
    points: Sequence[IDWPoint],
    step: int = 5,
    power: float = DEFAULT_IDW_POWER,
) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise ValueError("Grid width and height must be positive.")
    if step <= 0:
        raise ValueError("Grid step must be positive.")

    grid_w = math.ceil(width / step)
    grid_h = math.ceil(height / step)
    values = np.empty((grid_h, grid_w), dtype=np.float32)

    for gy in range(grid_h):
        y = min(gy * step, height - 1)
        for gx in range(grid_w):
            x = min(gx * step, width - 1)
            values[gy, gx] = idw_interpolate(x, y, points, power)

    return values


def audit_router_reference(points: Sequence[IDWPoint]) -> str | None:
    with_distance = [point for point in points if point.distance_m is not None]
    if not with_distance:
        return None

    nearest = min(with_distance, key=lambda point: point.distance_m if point.distance_m is not None else math.inf)
    if nearest.distance_m != 0:
        return f"Menor distance_m esta em {nearest.point_id} ({nearest.distance_m}m), sem ponto AP em 0m."
    return None
