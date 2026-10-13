#!/usr/bin/env python3
"""Small, deterministic road-geometry helpers for offline offense analysis.

The helpers intentionally use calibrated image polylines instead of guessing
road geometry from a vehicle trajectory.  A missing calibration must therefore
produce an unknown lane, not a guessed traffic violation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite
from typing import Iterable, Sequence


Point = tuple[float, float]


@dataclass(frozen=True)
class Boundary:
    boundary_id: str
    boundary_type: str
    points: tuple[Point, ...]


@dataclass(frozen=True)
class Crosswalk:
    crosswalk_id: str
    pedestrian_zone: tuple[Point, ...]
    vehicle_conflict_zone: tuple[Point, ...]
    vehicle_approach_zone: tuple[Point, ...]
    confidence: float
    minimum_vehicle_confidence: float
    minimum_pedestrian_confidence: float
    motion_window_seconds: float
    minimum_vehicle_motion_per_second: float
    stopped_motion_per_second: float
    stop_hold_seconds: float
    stop_lookback_seconds: float


def _point(value: Sequence[float], *, normalized: bool, width: float, height: float) -> Point:
    if len(value) != 2:
        raise ValueError("road geometry points must contain [x, y]")
    x, y = float(value[0]), float(value[1])
    if not isfinite(x) or not isfinite(y):
        raise ValueError("road geometry points must be finite")
    if normalized:
        x *= width
        y *= height
    return x, y


def parse_boundaries(
    values: Iterable[dict], *, width: int, height: int, normalized: bool = False
) -> list[Boundary]:
    """Validate and convert a list of ordered image-space boundaries."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    result: list[Boundary] = []
    for item in values:
        points = tuple(
            _point(point, normalized=normalized, width=width, height=height)
            for point in item.get("points", [])
        )
        if len(points) < 2:
            raise ValueError("each road boundary needs at least two points")
        result.append(
            Boundary(
                boundary_id=str(item["id"]),
                boundary_type=str(item.get("type", "unknown")),
                points=points,
            )
        )
    return result


def _polygon(
    values: Iterable[Sequence[float]], *, normalized: bool, width: int, height: int
) -> tuple[Point, ...]:
    points = tuple(
        _point(point, normalized=normalized, width=width, height=height)
        for point in values
    )
    if len(points) < 3:
        raise ValueError("road geometry polygons need at least three points")
    return points


def parse_crosswalks(
    values: Iterable[dict], *, width: int, height: int, normalized: bool = False
) -> list[Crosswalk]:
    """Validate calibrated pedestrian and vehicle conflict polygons."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    result: list[Crosswalk] = []
    for item in values:
        shared_zone = item.get("polygon")
        pedestrian_values = item.get("pedestrian_zone", shared_zone)
        vehicle_values = item.get("vehicle_conflict_zone", shared_zone)
        approach_values = item.get("vehicle_approach_zone", vehicle_values)
        if pedestrian_values is None or vehicle_values is None:
            raise ValueError(
                "each crosswalk needs pedestrian_zone and vehicle_conflict_zone"
            )
        confidence = float(item.get("confidence", 0.9))
        vehicle_confidence = float(item.get("minimum_vehicle_confidence", 0.25))
        pedestrian_confidence = float(item.get("minimum_pedestrian_confidence", 0.25))
        ratio_values = {
            "confidence": confidence,
            "minimum_vehicle_confidence": vehicle_confidence,
            "minimum_pedestrian_confidence": pedestrian_confidence,
        }
        for name, value in ratio_values.items():
            if not 0 <= value <= 1:
                raise ValueError(f"crosswalk {name} must be in the range [0, 1]")
        positive_values = {
            "motion_window_seconds": float(item.get("motion_window_seconds", 0.3)),
            "minimum_vehicle_motion_per_second": float(
                item.get("minimum_vehicle_motion_per_second", 0.02)
            ),
            "stopped_motion_per_second": float(
                item.get("stopped_motion_per_second", 0.005)
            ),
            "stop_hold_seconds": float(item.get("stop_hold_seconds", 0.5)),
            "stop_lookback_seconds": float(item.get("stop_lookback_seconds", 2.0)),
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"crosswalk {name} must be positive")
        if positive_values["stopped_motion_per_second"] >= positive_values[
            "minimum_vehicle_motion_per_second"
        ]:
            raise ValueError(
                "stopped_motion_per_second must be below minimum_vehicle_motion_per_second"
            )
        result.append(
            Crosswalk(
                crosswalk_id=str(item["id"]),
                pedestrian_zone=_polygon(
                    pedestrian_values,
                    normalized=normalized,
                    width=width,
                    height=height,
                ),
                vehicle_conflict_zone=_polygon(
                    vehicle_values,
                    normalized=normalized,
                    width=width,
                    height=height,
                ),
                vehicle_approach_zone=_polygon(
                    approach_values,
                    normalized=normalized,
                    width=width,
                    height=height,
                ),
                confidence=confidence,
                minimum_vehicle_confidence=vehicle_confidence,
                minimum_pedestrian_confidence=pedestrian_confidence,
                motion_window_seconds=positive_values["motion_window_seconds"],
                minimum_vehicle_motion_per_second=positive_values[
                    "minimum_vehicle_motion_per_second"
                ],
                stopped_motion_per_second=positive_values[
                    "stopped_motion_per_second"
                ],
                stop_hold_seconds=positive_values["stop_hold_seconds"],
                stop_lookback_seconds=positive_values["stop_lookback_seconds"],
            )
        )
    return result


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Return true when a point lies inside or on the edge of a polygon."""
    if len(polygon) < 3:
        raise ValueError("polygon needs at least three points")
    x, y = point
    inside = False
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if (
            abs(cross) <= 1e-7
            and min(x1, x2) - 1e-7 <= x <= max(x1, x2) + 1e-7
            and min(y1, y2) - 1e-7 <= y <= max(y1, y2) + 1e-7
        ):
            return True
        intersects = (y1 > y) != (y2 > y)
        if intersects:
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing_x:
                inside = not inside
    return inside


def x_at_y(points: Sequence[Point], y: float) -> float | None:
    """Interpolate the x coordinate where a polyline meets a horizontal row."""
    candidates: list[float] = []
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        low, high = sorted((y1, y2))
        if y < low or y > high:
            continue
        if y1 == y2:
            candidates.extend((x1, x2))
            continue
        ratio = (y - y1) / (y2 - y1)
        candidates.append(x1 + ratio * (x2 - x1))
    if not candidates:
        return None
    return sum(candidates) / len(candidates)


def lane_at_point(
    point: Point, boundaries: Sequence[Boundary], *, minimum_margin: float = 2.0
) -> tuple[int | None, float]:
    """Return the interval index and confidence for a bottom-center point.

    Boundaries must be ordered from image-left to image-right.  The outermost
    boundaries are required; points outside them are intentionally unknown.
    """
    x, y = point
    crossings = [
        (boundary, x_at_y(boundary.points, y))
        for boundary in boundaries
    ]
    crossings = [(boundary, value) for boundary, value in crossings if value is not None]
    crossings.sort(key=lambda item: item[1])
    if len(crossings) < 2:
        return None, 0.0
    for index, ((_, left), (_, right)) in enumerate(zip(crossings, crossings[1:])):
        if left + minimum_margin <= x <= right - minimum_margin:
            width = max(1.0, right - left)
            edge_distance = min(x - left, right - x)
            confidence = max(0.0, min(1.0, 2.0 * edge_distance / width))
            return index, round(confidence, 3)
    return None, 0.0


def boundary_side(point: Point, boundary: Boundary) -> float | None:
    """Return signed horizontal distance from a boundary at the point's row."""
    x_at_row = x_at_y(boundary.points, point[1])
    if x_at_row is None:
        return None
    return point[0] - x_at_row


def bottom_center(box: Sequence[float]) -> Point:
    if len(box) != 4:
        raise ValueError("vehicle boxes must contain [x1, y1, x2, y2]")
    x1, y1, x2, y2 = (float(value) for value in box)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("vehicle boxes must have positive area")
    return (round((x1 + x2) / 2.0, 3), round(y2, 3))


def trajectory_speed(a: Point, b: Point, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return hypot(b[0] - a[0], b[1] - a[1]) / seconds
