#!/usr/bin/env python3
"""Convert tracked-video JSON into calibrated scene observations and offenses.

This is an offline adapter.  It does not infer lane geometry from motion: a
road calibration JSON is required so that camera movement and arbitrary
vehicle motion cannot silently become a legal accusation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from math import hypot
from pathlib import Path
from typing import Any

from offense_analyzer import OffenseConfig, analyze_scene
from road_geometry import (
    Boundary,
    Crosswalk,
    Point,
    bottom_center,
    boundary_side,
    lane_at_point,
    parse_boundaries,
    parse_crosswalks,
    parse_lanes,
    parse_traffic_signals,
    point_in_polygon,
    stop_line_position,
)


VEHICLE_LABELS = {"vehicle", "car", "motorcycle", "bus", "truck"}
PERSON_LABELS = {"person", "pedestrian"}
TrackSample = tuple[int, float, Point, float]


def _box_center(box: list[float] | tuple[float, ...]) -> Point:
    if len(box) != 4:
        raise ValueError("boxes must contain [x1, y1, x2, y2]")
    x1, y1, x2, y2 = (float(value) for value in box)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("boxes must have positive area")
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _pose(config: dict[str, Any]) -> dict[str, Any]:
    pose = dict(config.get("camera_pose", {}))
    pose.setdefault("mode", "road_context")
    pose.setdefault("pan_degrees", 0.0)
    pose.setdefault("tilt_degrees", 0.0)
    pose.setdefault("pan_rate_deg_s", 0.0)
    pose.setdefault("tilt_rate_deg_s", 0.0)
    pose.setdefault("calibrated", True)
    pose.setdefault("road_transform_valid", True)
    pose.setdefault("confidence", 0.95)
    return pose


def _boundary_events(
    records: list[dict[str, Any]],
    tracks_by_id: dict[int, list[tuple[int, float, tuple[float, float]]]],
    boundaries: list[Boundary],
    minimum_confidence: float,
    hold_frames: int,
) -> dict[int, list[dict[str, Any]]]:
    """Emit repeated evidence only after a side transition is observed."""
    output: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if hold_frames < 1:
        raise ValueError("hold_frames must be at least one")
    for boundary in boundaries:
        if boundary.boundary_type not in {"solid", "double_solid", "solid_yellow", "double_yellow"}:
            continue
        for track_id, trajectory in tracks_by_id.items():
            sides = [(frame, timestamp, boundary_side(point, boundary)) for frame, timestamp, point in trajectory]
            for index in range(1, len(sides)):
                previous, current = sides[index - 1], sides[index]
                if previous[2] is None or current[2] is None or previous[2] == 0 or current[2] == 0:
                    continue
                if previous[2] * current[2] >= 0:
                    continue
                # A one-frame sign flip is weak evidence; require the new side
                # to persist for the configured number of observations.
                held = sides[index : index + hold_frames]
                if len(held) < hold_frames or any(item[2] is None or item[2] * current[2] <= 0 for item in held):
                    continue
                for frame, timestamp, _ in held:
                    output[frame].append(
                        {
                            "track_id": track_id,
                            "boundary_id": boundary.boundary_id,
                            "boundary_type": boundary.boundary_type,
                            "confidence": minimum_confidence,
                        }
                    )
                break
    return output


def _motion_speed(
    trajectory: list[TrackSample], index: int, window_seconds: float, diagonal: float
) -> float | None:
    """Measure image motion as frame-diagonal fractions per second."""
    current = trajectory[index]
    earliest = index
    while (
        earliest > 0
        and current[1] - trajectory[earliest - 1][1] <= window_seconds
    ):
        earliest -= 1
    if earliest == index:
        return None
    previous = trajectory[earliest]
    elapsed = current[1] - previous[1]
    if elapsed <= 0:
        return None
    return hypot(current[2][0] - previous[2][0], current[2][1] - previous[2][1]) / (
        diagonal * elapsed
    )


def _trajectory_directions(
    trajectories: dict[int, list[TrackSample]],
    *,
    window_seconds: float,
    diagonal: float,
    minimum_motion: float,
) -> dict[tuple[int, int], tuple[Point | None, float]]:
    """Return recent image-space motion vectors and their confidence."""
    if window_seconds <= 0 or minimum_motion <= 0:
        raise ValueError("direction window and minimum motion must be positive")
    output: dict[tuple[int, int], tuple[Point | None, float]] = {}
    for track_id, trajectory in trajectories.items():
        for index, current in enumerate(trajectory):
            earliest = index
            while (
                earliest > 0
                and current[1] - trajectory[earliest - 1][1] <= window_seconds
            ):
                earliest -= 1
            if earliest == index:
                output[(track_id, current[0])] = (None, 0.0)
                continue
            previous = trajectory[earliest]
            motion = (current[2][0] - previous[2][0], current[2][1] - previous[2][1])
            distance = hypot(*motion)
            confidence = min(1.0, distance / (diagonal * minimum_motion))
            output[(track_id, current[0])] = (
                motion if confidence > 0 else None,
                round(confidence, 5),
            )
    return output


def _direction_alignment(motion: Point | None, expected: Point | None) -> float | None:
    if motion is None or expected is None:
        return None
    motion_length = hypot(*motion)
    expected_length = hypot(*expected)
    if motion_length == 0 or expected_length == 0:
        return None
    return (motion[0] * expected[0] + motion[1] * expected[1]) / (
        motion_length * expected_length
    )


def _stopped_before(
    trajectory: list[TrackSample],
    index: int,
    crosswalk: Crosswalk,
    diagonal: float,
) -> bool:
    current_time = trajectory[index][1]
    earliest_time = current_time - crosswalk.stop_lookback_seconds
    stop_start: float | None = None
    for sample_index in range(1, index + 1):
        sample = trajectory[sample_index]
        if sample[1] < earliest_time:
            continue
        speed = _motion_speed(
            trajectory, sample_index, crosswalk.motion_window_seconds, diagonal
        )
        in_approach = point_in_polygon(sample[2], crosswalk.vehicle_approach_zone)
        if speed is not None and in_approach and speed <= crosswalk.stopped_motion_per_second:
            if stop_start is None:
                stop_start = trajectory[sample_index - 1][1]
            if sample[1] - stop_start >= crosswalk.stop_hold_seconds:
                return True
        else:
            stop_start = None
    return False


def _crosswalk_events(
    records: list[dict[str, Any]],
    vehicle_trajectories: dict[int, list[TrackSample]],
    person_trajectories: dict[int, list[TrackSample]],
    crosswalks: list[Crosswalk],
    width: int,
    height: int,
) -> dict[int, list[dict[str, Any]]]:
    output: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if not crosswalks:
        return output
    diagonal = hypot(width, height)
    vehicles_by_frame: dict[int, list[tuple[int, TrackSample, int]]] = defaultdict(list)
    people_by_frame: dict[int, list[tuple[int, TrackSample]]] = defaultdict(list)
    for track_id, trajectory in vehicle_trajectories.items():
        for index, sample in enumerate(trajectory):
            vehicles_by_frame[sample[0]].append((track_id, sample, index))
    for track_id, trajectory in person_trajectories.items():
        for sample in trajectory:
            people_by_frame[sample[0]].append((track_id, sample))

    for record in records:
        frame = int(record["frame"])
        for crosswalk in crosswalks:
            pedestrians = [
                (track_id, sample)
                for track_id, sample in people_by_frame.get(frame, [])
                if sample[3] >= crosswalk.minimum_pedestrian_confidence
                and point_in_polygon(sample[2], crosswalk.pedestrian_zone)
            ]
            if not pedestrians:
                continue
            for vehicle_id, vehicle, trajectory_index in vehicles_by_frame.get(frame, []):
                if vehicle[3] < crosswalk.minimum_vehicle_confidence or not point_in_polygon(
                    vehicle[2], crosswalk.vehicle_conflict_zone
                ):
                    continue
                trajectory = vehicle_trajectories[vehicle_id]
                speed = _motion_speed(
                    trajectory,
                    trajectory_index,
                    crosswalk.motion_window_seconds,
                    diagonal,
                )
                if speed is None or speed < crosswalk.minimum_vehicle_motion_per_second:
                    continue
                stopped = _stopped_before(
                    trajectory, trajectory_index, crosswalk, diagonal
                )
                for pedestrian_id, pedestrian in pedestrians:
                    output[frame].append(
                        {
                            "vehicle_track_id": vehicle_id,
                            "pedestrian_track_id": pedestrian_id,
                            "crosswalk_id": crosswalk.crosswalk_id,
                            "pedestrian_has_priority": True,
                            "vehicle_crossed_crosswalk": True,
                            "vehicle_stopped_before_crosswalk": stopped,
                            "confidence": round(
                                min(crosswalk.confidence, vehicle[3], pedestrian[3]), 5
                            ),
                            "vehicle_motion_per_second": round(speed, 5),
                        }
                    )
    return output


def build_scene(tracking: dict[str, Any], geometry: dict[str, Any]) -> dict[str, Any]:
    width = int(tracking.get("frame_width", geometry.get("image_width", 0)))
    height = int(tracking.get("frame_height", geometry.get("image_height", 0)))
    if width <= 0 or height <= 0:
        raise ValueError("tracking JSON must provide frame_width/frame_height")
    normalized = bool(geometry.get("normalized", False))
    boundaries = parse_boundaries(
        geometry.get("boundaries", []), width=width, height=height, normalized=normalized
    )
    lanes = parse_lanes(
        geometry.get("lanes", []), width=width, height=height, normalized=normalized
    )
    crosswalks = parse_crosswalks(
        geometry.get("crosswalks", []), width=width, height=height, normalized=normalized
    )
    traffic_signals = parse_traffic_signals(
        geometry.get("traffic_signals", []),
        width=width,
        height=height,
        normalized=normalized,
    )
    if boundaries and len(boundaries) < 2:
        raise ValueError("road geometry needs at least two ordered boundaries")
    if traffic_signals and len(boundaries) < 2 and not lanes:
        raise ValueError(
            "traffic signal calibration needs lane polygons or ordered boundaries"
        )
    if not boundaries and not lanes and not crosswalks and not traffic_signals:
        raise ValueError(
            "road geometry needs boundaries, lanes, a crosswalk, or traffic signals"
        )
    records = tracking.get("frame_records")
    if not isinstance(records, list) or not records:
        raise ValueError("tracking JSON must contain frame_records")
    fps = float(tracking.get("source_fps", 0.0))
    if fps <= 0:
        raise ValueError("tracking JSON must provide a positive source_fps")

    vehicle_trajectories: dict[int, list[TrackSample]] = defaultdict(list)
    person_trajectories: dict[int, list[TrackSample]] = defaultdict(list)
    for record in records:
        frame = int(record["frame"])
        timestamp = float(record.get("timestamp_seconds", (frame - 1) / fps))
        for item in record.get("tracks", []):
            try:
                track_id = int(item["track_id"])
                point = bottom_center(item["box"])
                confidence = float(item.get("confidence", 1.0))
                label = str(item.get("label", "vehicle")).lower()
            except (KeyError, TypeError, ValueError):
                continue
            if track_id >= 0:
                sample = (frame, timestamp, point, confidence)
                if label in VEHICLE_LABELS:
                    vehicle_trajectories[track_id].append(sample)
                elif label in PERSON_LABELS:
                    person_trajectories[track_id].append(sample)

    boundary_events = _boundary_events(
        records,
        {
            track_id: [sample[:3] for sample in trajectory]
            for track_id, trajectory in vehicle_trajectories.items()
        },
        boundaries,
        float(geometry.get("boundary_confidence", 0.9)),
        int(geometry.get("boundary_hold_frames", 2)),
    )
    crosswalk_events = _crosswalk_events(
        records,
        vehicle_trajectories,
        person_trajectories,
        crosswalks,
        width,
        height,
    )
    diagonal = hypot(width, height)
    trajectory_directions = _trajectory_directions(
        vehicle_trajectories,
        window_seconds=float(geometry.get("direction_window_seconds", 0.5)),
        diagonal=diagonal,
        minimum_motion=float(geometry.get("minimum_direction_motion", 0.002)),
    )
    pose = _pose(geometry)
    lanes_by_index = {lane.lane_index: lane for lane in lanes}
    signals_by_lane: dict[int, list[Any]] = defaultdict(list)
    for signal in traffic_signals:
        for lane_index in signal.applies_to_lanes:
            signals_by_lane[lane_index].append(signal)
    scene_records: list[dict[str, Any]] = []
    for record in records:
        frame = int(record["frame"])
        timestamp = float(record.get("timestamp_seconds", (frame - 1) / fps))
        vehicles: list[dict[str, Any]] = []
        for item in record.get("tracks", []):
            label = str(item.get("label", "vehicle")).lower()
            if label not in VEHICLE_LABELS:
                continue
            try:
                box = item["box"]
                point = bottom_center(box)
                lane_index, lane_confidence = lane_at_point(
                    point, boundaries, lanes=lanes
                )
            except (KeyError, TypeError, ValueError):
                continue
            track_id = int(item.get("track_id", -1))
            if track_id < 0:
                continue
            lane = lanes_by_index.get(lane_index)
            motion, direction_confidence = trajectory_directions.get(
                (track_id, frame), (None, 0.0)
            )
            direction_alignment = _direction_alignment(
                motion, lane.direction if lane is not None else None
            )
            lane_signals = signals_by_lane.get(lane_index, [])
            signal = lane_signals[0] if len(lane_signals) == 1 else None
            if (
                signal is not None
                and lane is not None
                and lane.direction is not None
                and (
                    direction_alignment is None
                    or direction_alignment < lane.minimum_direction_alignment
                )
            ):
                signal = None
            vehicles.append(
                {
                    "track_id": track_id,
                    "label": str(item.get("label", "vehicle")),
                    "lane_index": lane_index,
                    "lane_confidence": lane_confidence,
                    "turn_signal": "unknown",
                    "turn_signal_confidence": 0.0,
                    "stop_line_position": (
                        stop_line_position(point, signal)
                        if signal is not None
                        else "unknown"
                    ),
                    "signal_id": signal.signal_id if signal is not None else None,
                    "direction_alignment": (
                        round(direction_alignment, 5)
                        if direction_alignment is not None
                        else None
                    ),
                    "direction_confidence": direction_confidence,
                    "bottom_center": list(point),
                }
            )
        traffic_lights: list[dict[str, Any]] = []
        for signal in traffic_signals:
            candidates: list[dict[str, Any]] = []
            for item in record.get("tracks", []):
                if str(item.get("label", "")).lower() != "traffic_light":
                    continue
                try:
                    if point_in_polygon(_box_center(item["box"]), signal.roi):
                        candidates.append(item)
                except (KeyError, TypeError, ValueError):
                    continue
            best = max(
                candidates,
                key=lambda item: float(
                    item.get("traffic_light_state_confidence", 0.0)
                ),
                default=None,
            )
            state = str(best.get("traffic_light_state", "unknown")) if best else "unknown"
            if state not in {"red", "yellow", "green"}:
                state = "unknown"
            confidence = (
                min(
                    signal.confidence,
                    float(best.get("traffic_light_state_confidence", 0.0)),
                )
                if best is not None and state != "unknown"
                else 0.0
            )
            traffic_lights.append(
                {
                    "signal_id": signal.signal_id,
                    "state": state,
                    "confidence": round(confidence, 5),
                    "applies_to_lanes": list(signal.applies_to_lanes),
                    "source_track_id": (
                        int(best["track_id"])
                        if best is not None and best.get("track_id") is not None
                        else None
                    ),
                }
            )
        scene_records.append(
            {
                "frame": frame,
                "timestamp_seconds": timestamp,
                "camera_pose": pose,
                "vehicles": vehicles,
                "traffic_lights": traffic_lights,
                "lane_boundary_crossings": boundary_events.get(frame, []),
                "crosswalk_interactions": crosswalk_events.get(frame, []),
            }
        )
    return {
        "schema_version": 2,
        "source_video": tracking.get("source_video"),
        "source_fps": fps,
        "frame_width": width,
        "frame_height": height,
        "calibration": {
            "boundary_count": len(boundaries),
            "boundary_ids": [boundary.boundary_id for boundary in boundaries],
            "lane_count": len(lanes),
            "lane_indices": [lane.lane_index for lane in lanes],
            "crosswalk_count": len(crosswalks),
            "crosswalk_ids": [crosswalk.crosswalk_id for crosswalk in crosswalks],
            "traffic_signal_count": len(traffic_signals),
            "traffic_signal_ids": [signal.signal_id for signal in traffic_signals],
            "normalized": normalized,
        },
        "frame_records": scene_records,
    }


def _select_offender(result: dict[str, Any]) -> dict[str, Any]:
    priority = {
        "solid_line_crossing": 100,
        "red_light_running": 95,
        "wrong_way_driving": 90,
        "pedestrian_failure_to_yield": 90,
        "lane_change_without_signal": 80,
        "consecutive_lane_changes": 75,
        "littering_from_vehicle": 70,
    }
    events = [event for event in result.get("events", []) if event.get("track_id") is not None]
    if not events:
        return {
            "status": "NO_CONFIDENT_OFFENDER",
            "track_id": None,
            "reason": "No rule event has sufficient calibrated evidence.",
        }
    def evidence_confidence(event: dict[str, Any]) -> float:
        evidence = event.get("evidence", {})
        return max(
            (
                float(evidence.get(name, 0.0))
                for name in (
                    "maximum_confidence",
                    "confidence",
                    "signal_observation_ratio",
                )
            ),
            default=0.0,
        )

    ranked = sorted(
        events,
        key=lambda event: (
            priority.get(str(event["type"]), 0),
            evidence_confidence(event),
            -float(event.get("start_seconds", 0.0)),
        ),
        reverse=True,
    )
    chosen = ranked[0]
    return {
        "status": "SUSPECTED_OFFENDER",
        "track_id": int(chosen["track_id"]),
        "offense_type": chosen["type"],
        "start_frame": chosen.get("start_frame"),
        "end_frame": chosen.get("end_frame"),
        "evidence_confidence": evidence_confidence(chosen),
        "evidence": chosen.get("evidence", {}),
        "notice": "Tracking lock is a review aid; verify the original video before any legal conclusion.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracking_json", type=Path)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--rules", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "offense_rules.json")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    tracking = _load(args.tracking_json)
    geometry = _load(args.geometry)
    scene = build_scene(tracking, geometry)
    config = OffenseConfig.from_mapping(_load(args.rules))
    offenses = analyze_scene(scene, config)
    offenses["selected_offender"] = _select_offender(offenses)
    output_dir = args.output_dir or args.tracking_json.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_path = output_dir / f"{args.tracking_json.stem}_scene_observations.json"
    offense_path = output_dir / f"{args.tracking_json.stem}_offenses.json"
    scene_path.write_text(json.dumps(scene, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    offense_path.write_text(json.dumps(offenses, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    selected = offenses["selected_offender"]
    print(f"scene={scene_path.resolve()}")
    print(f"offenses={offenses['event_count']}")
    print(f"selected={selected['status']} track_id={selected.get('track_id')}")
    print(f"output={offense_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
