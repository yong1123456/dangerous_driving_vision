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
class Lane:
    lane_index: int
    polygon: tuple[Point, ...]
    confidence: float
    direction: Point | None
    minimum_direction_alignment: float


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


@dataclass(frozen=True)
class TrafficSignal:
    signal_id: str
    roi: tuple[Point, ...]
    applies_to_lanes: tuple[int, ...]
    stop_line: tuple[Point, Point]
    before_side: int
    stop_line_margin_pixels: float
    confidence: float


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


def parse_lanes(
    values: Iterable[dict], *, width: int, height: int, normalized: bool = False
) -> list[Lane]:
    """Validate explicit lane polygons for arbitrary camera viewpoints."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    result: list[Lane] = []
    lane_indices: set[int] = set()
    for item in values:
        lane_index = int(item["index"])
        if lane_index < 0:
            raise ValueError("lane index must be non-negative")
        if lane_index in lane_indices:
            raise ValueError(f"duplicate lane index: {lane_index}")
        lane_indices.add(lane_index)
        confidence = float(item.get("confidence", 0.9))
        if not 0 <= confidence <= 1:
            raise ValueError("lane confidence must be in the range [0, 1]")
        direction_value = item.get("direction")
        direction = None
        if direction_value is not None:
            if len(direction_value) != 2:
                raise ValueError("lane direction must contain [dx, dy]")
            direction = (float(direction_value[0]), float(direction_value[1]))
            if not all(isfinite(value) for value in direction) or direction == (0.0, 0.0):
                raise ValueError("lane direction must be a finite non-zero vector")
        minimum_alignment = float(item.get("minimum_direction_alignment", 0.2))
        if not -1 <= minimum_alignment <= 1:
            raise ValueError(
                "lane minimum_direction_alignment must be in the range [-1, 1]"
            )
        result.append(
            Lane(
                lane_index=lane_index,
                polygon=_polygon(
                    item.get("polygon", []),
                    normalized=normalized,
                    width=width,
                    height=height,
                ),
                confidence=confidence,
                direction=direction,
                minimum_direction_alignment=minimum_alignment,
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


def parse_traffic_signals(
    values: Iterable[dict], *, width: int, height: int, normalized: bool = False
) -> list[TrafficSignal]:
    """Validate signal ROIs, lane associations and oriented stop lines."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    result: list[TrafficSignal] = []
    signal_ids: set[str] = set()
    for item in values:
        signal_id = str(item["id"])
        if signal_id in signal_ids:
            raise ValueError(f"duplicate traffic signal id: {signal_id}")
        signal_ids.add(signal_id)
        lanes = tuple(int(value) for value in item.get("applies_to_lanes", []))
        if not lanes or any(value < 0 for value in lanes) or len(set(lanes)) != len(lanes):
            raise ValueError(
                "traffic signal applies_to_lanes must contain unique non-negative lanes"
            )
        line_values = item.get("stop_line", {}).get("points", [])
        line = tuple(
            _point(point, normalized=normalized, width=width, height=height)
            for point in line_values
        )
        if len(line) != 2 or line[0] == line[1]:
            raise ValueError("traffic signal stop_line needs two distinct points")
        before_name = str(item.get("stop_line", {}).get("before_side", "negative"))
        if before_name not in {"negative", "positive"}:
            raise ValueError("stop_line before_side must be negative or positive")
        margin = float(item.get("stop_line", {}).get("margin_pixels", 3.0))
        confidence = float(item.get("confidence", 0.9))
        if margin < 0:
            raise ValueError("stop_line margin_pixels must not be negative")
        if not 0 <= confidence <= 1:
            raise ValueError("traffic signal confidence must be in the range [0, 1]")
        result.append(
            TrafficSignal(
                signal_id=signal_id,
                roi=_polygon(
                    item.get("roi", []),
                    normalized=normalized,
                    width=width,
                    height=height,
                ),
                applies_to_lanes=lanes,
                stop_line=(line[0], line[1]),
                before_side=-1 if before_name == "negative" else 1,
                stop_line_margin_pixels=margin,
                confidence=confidence,
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


def signed_distance_to_line(point: Point, line: Sequence[Point]) -> float:
    """Return oriented perpendicular distance from a two-point line."""
    if len(line) != 2 or line[0] == line[1]:
        raise ValueError("line needs two distinct points")
    (x1, y1), (x2, y2) = line
    return ((x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1)) / hypot(
        x2 - x1, y2 - y1
    )


def stop_line_position(point: Point, signal: TrafficSignal) -> str:
    distance = signed_distance_to_line(point, signal.stop_line)
    if abs(distance) <= signal.stop_line_margin_pixels:
        return "unknown"
    side = 1 if distance > 0 else -1
    return "before" if side == signal.before_side else "after"


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
    point: Point,
    boundaries: Sequence[Boundary],
    *,
    lanes: Sequence[Lane] = (),
    minimum_margin: float = 2.0,
) -> tuple[int | None, float]:
    """Return the calibrated lane and confidence for a bottom-center point.

    Explicit polygons support arbitrary road directions and take precedence.
    A point covered by multiple polygons is intentionally unknown.  Without
    polygons, boundaries must be ordered image-left to image-right and the
    outermost boundaries are required.
    """
    containing_lanes = [lane for lane in lanes if point_in_polygon(point, lane.polygon)]
    if len(containing_lanes) == 1:
        lane = containing_lanes[0]
        return lane.lane_index, round(lane.confidence, 3)
    if len(containing_lanes) > 1:
        return None, 0.0

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
