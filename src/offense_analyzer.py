#!/usr/bin/env python3
"""Analyze enriched scene observations for suspected traffic offenses."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OffenseConfig:
    minimum_lane_dwell_seconds: float = 0.5
    maximum_observation_gap_seconds: float = 0.25
    minimum_lane_confidence: float = 0.7
    pre_signal_seconds: float = 2.0
    post_signal_seconds: float = 0.3
    minimum_signal_confidence: float = 0.7
    minimum_signal_observation_ratio: float = 0.7
    consecutive_lane_change_window_seconds: float = 8.0
    minimum_red_stable_seconds: float = 0.5
    minimum_traffic_light_confidence: float = 0.7
    minimum_litter_confidence: float = 0.7
    minimum_litter_observations: int = 3
    minimum_camera_pose_confidence: float = 0.8
    maximum_context_pan_rate_deg_s: float = 3.0
    maximum_context_tilt_rate_deg_s: float = 3.0
    maximum_context_pan_offset_deg: float = 35.0
    minimum_boundary_confidence: float = 0.75
    minimum_boundary_observations: int = 2
    minimum_wrong_way_confidence: float = 0.75
    maximum_wrong_way_alignment: float = -0.5
    minimum_wrong_way_seconds: float = 1.0
    minimum_crosswalk_confidence: float = 0.75
    minimum_crosswalk_observations: int = 3
    crosswalk_event_merge_seconds: float = 1.0
    require_valid_road_context: bool = True

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "OffenseConfig":
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown offense config keys: {sorted(unknown)}")
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        positive = {
            "minimum_lane_dwell_seconds": self.minimum_lane_dwell_seconds,
            "maximum_observation_gap_seconds": self.maximum_observation_gap_seconds,
            "pre_signal_seconds": self.pre_signal_seconds,
            "post_signal_seconds": self.post_signal_seconds,
            "consecutive_lane_change_window_seconds": (
                self.consecutive_lane_change_window_seconds
            ),
            "minimum_red_stable_seconds": self.minimum_red_stable_seconds,
            "maximum_context_pan_rate_deg_s": self.maximum_context_pan_rate_deg_s,
            "maximum_context_tilt_rate_deg_s": (
                self.maximum_context_tilt_rate_deg_s
            ),
            "maximum_context_pan_offset_deg": self.maximum_context_pan_offset_deg,
            "minimum_wrong_way_seconds": self.minimum_wrong_way_seconds,
            "crosswalk_event_merge_seconds": self.crosswalk_event_merge_seconds,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        ratios = {
            "minimum_lane_confidence": self.minimum_lane_confidence,
            "minimum_signal_confidence": self.minimum_signal_confidence,
            "minimum_signal_observation_ratio": (
                self.minimum_signal_observation_ratio
            ),
            "minimum_traffic_light_confidence": (
                self.minimum_traffic_light_confidence
            ),
            "minimum_litter_confidence": self.minimum_litter_confidence,
            "minimum_camera_pose_confidence": self.minimum_camera_pose_confidence,
            "minimum_boundary_confidence": self.minimum_boundary_confidence,
            "minimum_wrong_way_confidence": self.minimum_wrong_way_confidence,
            "minimum_crosswalk_confidence": self.minimum_crosswalk_confidence,
        }
        for name, value in ratios.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in the range [0, 1]")
        if self.minimum_litter_observations < 1:
            raise ValueError("minimum_litter_observations must be at least 1")
        if self.minimum_crosswalk_observations < 1:
            raise ValueError("minimum_crosswalk_observations must be at least 1")
        if self.minimum_boundary_observations < 1:
            raise ValueError("minimum_boundary_observations must be at least 1")
        if not -1 <= self.maximum_wrong_way_alignment <= 1:
            raise ValueError("maximum_wrong_way_alignment must be in the range [-1, 1]")


@dataclass(frozen=True)
class CameraPose:
    mode: str
    pan_degrees: float
    tilt_degrees: float
    pan_rate_deg_s: float
    tilt_rate_deg_s: float
    calibrated: bool
    road_transform_valid: bool
    confidence: float


def camera_pose_from_record(record: dict[str, Any]) -> CameraPose | None:
    value = record.get("camera_pose")
    if not isinstance(value, dict):
        return None
    return CameraPose(
        mode=str(value.get("mode", "unknown")),
        pan_degrees=float(value.get("pan_degrees", 0.0)),
        tilt_degrees=float(value.get("tilt_degrees", 0.0)),
        pan_rate_deg_s=float(value.get("pan_rate_deg_s", 0.0)),
        tilt_rate_deg_s=float(value.get("tilt_rate_deg_s", 0.0)),
        calibrated=bool(value.get("calibrated", False)),
        road_transform_valid=bool(value.get("road_transform_valid", False)),
        confidence=float(value.get("confidence", 0.0)),
    )


def road_context_is_valid(
    pose: CameraPose | None, config: OffenseConfig
) -> bool:
    if pose is None:
        return not config.require_valid_road_context
    return (
        pose.mode == "road_context"
        and pose.calibrated
        and pose.road_transform_valid
        and pose.confidence >= config.minimum_camera_pose_confidence
        and abs(pose.pan_degrees) <= config.maximum_context_pan_offset_deg
        and abs(pose.pan_rate_deg_s) <= config.maximum_context_pan_rate_deg_s
        and abs(pose.tilt_rate_deg_s) <= config.maximum_context_tilt_rate_deg_s
    )


@dataclass(frozen=True)
class VehicleObservation:
    frame: int
    timestamp: float
    track_id: int
    label: str
    lane_index: int | None
    lane_confidence: float
    turn_signal: str
    turn_signal_confidence: float
    stop_line_position: str
    signal_id: str | None
    direction_alignment: float | None
    direction_confidence: float
    road_context_valid: bool


@dataclass(frozen=True)
class LaneSegment:
    lane_index: int
    start: VehicleObservation
    end: VehicleObservation

    @property
    def duration(self) -> float:
        return self.end.timestamp - self.start.timestamp


@dataclass(frozen=True)
class LaneChange:
    track_id: int
    label: str
    frame: int
    timestamp: float
    from_lane: int
    to_lane: int

    @property
    def direction(self) -> str:
        return "right" if self.to_lane > self.from_lane else "left"


@dataclass(frozen=True)
class LightObservation:
    frame: int
    timestamp: float
    signal_id: str
    state: str
    confidence: float
    applies_to_lanes: tuple[int, ...] | None
    road_context_valid: bool


@dataclass(frozen=True)
class LitterObservation:
    frame: int
    timestamp: float
    object_track_id: int
    source_track_id: int
    confidence: float
    box: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class BoundaryCrossingObservation:
    frame: int
    timestamp: float
    track_id: int
    boundary_id: str
    boundary_type: str
    confidence: float
    road_context_valid: bool


@dataclass(frozen=True)
class CrosswalkInteractionObservation:
    frame: int
    timestamp: float
    vehicle_track_id: int
    pedestrian_track_id: int
    crosswalk_id: str
    pedestrian_has_priority: bool
    vehicle_crossed_crosswalk: bool
    vehicle_stopped_before_crosswalk: bool
    confidence: float
    road_context_valid: bool


def load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read {description}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _timestamp(record: dict[str, Any], source_fps: float | None) -> float:
    value = record.get("timestamp_seconds")
    if value is not None:
        return float(value)
    if source_fps is None or source_fps <= 0:
        raise ValueError("Each frame needs a timestamp or a positive source_fps")
    return (int(record["frame"]) - 1) / source_fps


def collect_scene_observations(
    scene: dict[str, Any], config: OffenseConfig
) -> tuple[
    dict[int, list[VehicleObservation]],
    dict[str, list[LightObservation]],
    dict[int, list[LitterObservation]],
    list[BoundaryCrossingObservation],
    list[CrosswalkInteractionObservation],
]:
    records = scene.get("frame_records")
    if not isinstance(records, list):
        raise ValueError("Scene observations must contain a frame_records list")
    fps_value = scene.get("source_fps")
    source_fps = float(fps_value) if fps_value is not None else None
    vehicles: dict[int, list[VehicleObservation]] = defaultdict(list)
    lights: dict[str, list[LightObservation]] = defaultdict(list)
    litter: dict[int, list[LitterObservation]] = defaultdict(list)
    boundary_crossings: list[BoundaryCrossingObservation] = []
    crosswalk_interactions: list[CrosswalkInteractionObservation] = []

    for record in records:
        frame = int(record["frame"])
        timestamp = _timestamp(record, source_fps)
        context_valid = road_context_is_valid(
            camera_pose_from_record(record), config
        )
        for item in record.get("vehicles", []):
            lane_value = item.get("lane_index")
            direction_value = item.get("direction_alignment")
            observation = VehicleObservation(
                frame=frame,
                timestamp=timestamp,
                track_id=int(item["track_id"]),
                label=str(item.get("label", "vehicle")),
                lane_index=int(lane_value) if lane_value is not None else None,
                lane_confidence=float(item.get("lane_confidence", 0.0)),
                turn_signal=str(item.get("turn_signal", "unknown")),
                turn_signal_confidence=float(
                    item.get("turn_signal_confidence", 0.0)
                ),
                stop_line_position=str(item.get("stop_line_position", "unknown")),
                signal_id=(
                    str(item["signal_id"]) if item.get("signal_id") is not None else None
                ),
                direction_alignment=(
                    float(direction_value) if direction_value is not None else None
                ),
                direction_confidence=float(item.get("direction_confidence", 0.0)),
                road_context_valid=context_valid,
            )
            vehicles[observation.track_id].append(observation)

        for item in record.get("traffic_lights", []):
            lanes = item.get("applies_to_lanes")
            observation = LightObservation(
                frame=frame,
                timestamp=timestamp,
                signal_id=str(item["signal_id"]),
                state=str(item.get("state", "unknown")),
                confidence=float(item.get("confidence", 0.0)),
                applies_to_lanes=(
                    tuple(int(lane) for lane in lanes) if lanes is not None else None
                ),
                road_context_valid=context_valid,
            )
            lights[observation.signal_id].append(observation)

        for item in record.get("discarded_objects", []):
            box_values = item.get("box")
            if box_values is not None and len(box_values) != 4:
                raise ValueError(f"Invalid discarded-object box in frame {frame}")
            observation = LitterObservation(
                frame=frame,
                timestamp=timestamp,
                object_track_id=int(item["object_track_id"]),
                source_track_id=int(item["source_track_id"]),
                confidence=float(item.get("confidence", 0.0)),
                box=(
                    tuple(float(value) for value in box_values)
                    if box_values is not None
                    else None
                ),
            )
            litter[observation.object_track_id].append(observation)

        for item in record.get("lane_boundary_crossings", []):
            boundary_crossings.append(
                BoundaryCrossingObservation(
                    frame=frame,
                    timestamp=timestamp,
                    track_id=int(item["track_id"]),
                    boundary_id=str(item["boundary_id"]),
                    boundary_type=str(item.get("boundary_type", "unknown")),
                    confidence=float(item.get("confidence", 0.0)),
                    road_context_valid=context_valid,
                )
            )

        for item in record.get("crosswalk_interactions", []):
            crosswalk_interactions.append(
                CrosswalkInteractionObservation(
                    frame=frame,
                    timestamp=timestamp,
                    vehicle_track_id=int(item["vehicle_track_id"]),
                    pedestrian_track_id=int(item["pedestrian_track_id"]),
                    crosswalk_id=str(item["crosswalk_id"]),
                    pedestrian_has_priority=bool(
                        item.get("pedestrian_has_priority", False)
                    ),
                    vehicle_crossed_crosswalk=bool(
                        item.get("vehicle_crossed_crosswalk", False)
                    ),
                    vehicle_stopped_before_crosswalk=bool(
                        item.get("vehicle_stopped_before_crosswalk", False)
                    ),
                    confidence=float(item.get("confidence", 0.0)),
                    road_context_valid=context_valid,
                )
            )

    for collection in (vehicles, lights, litter):
        for observations in collection.values():
            observations.sort(key=lambda item: (item.timestamp, item.frame))
    boundary_crossings.sort(key=lambda item: (item.timestamp, item.track_id))
    crosswalk_interactions.sort(
        key=lambda item: (item.timestamp, item.vehicle_track_id)
    )
    return (
        dict(vehicles),
        dict(lights),
        dict(litter),
        boundary_crossings,
        crosswalk_interactions,
    )


def _stable_lane_segments(
    observations: list[VehicleObservation], config: OffenseConfig
) -> list[LaneSegment]:
    eligible = [
        observation
        for observation in observations
        if observation.road_context_valid
        and observation.lane_index is not None
        and observation.lane_confidence >= config.minimum_lane_confidence
    ]
    if not eligible:
        return []

    segments: list[LaneSegment] = []
    start = eligible[0]
    previous = eligible[0]
    for observation in eligible[1:]:
        gap = observation.timestamp - previous.timestamp
        if (
            observation.lane_index != previous.lane_index
            or gap > config.maximum_observation_gap_seconds
        ):
            segment = LaneSegment(int(start.lane_index), start, previous)
            if segment.duration >= config.minimum_lane_dwell_seconds:
                segments.append(segment)
            start = observation
        previous = observation
    segment = LaneSegment(int(start.lane_index), start, previous)
    if segment.duration >= config.minimum_lane_dwell_seconds:
        segments.append(segment)
    return segments


def find_lane_changes(
    vehicles: dict[int, list[VehicleObservation]], config: OffenseConfig
) -> list[LaneChange]:
    changes: list[LaneChange] = []
    for track_id, observations in vehicles.items():
        segments = _stable_lane_segments(observations, config)
        for previous, current in zip(segments, segments[1:]):
            transition_gap = current.start.timestamp - previous.end.timestamp
            if (
                current.lane_index == previous.lane_index
                or transition_gap > config.maximum_observation_gap_seconds
            ):
                continue
            changes.append(
                LaneChange(
                    track_id=track_id,
                    label=current.start.label,
                    frame=current.start.frame,
                    timestamp=current.start.timestamp,
                    from_lane=previous.lane_index,
                    to_lane=current.lane_index,
                )
            )
    return sorted(changes, key=lambda item: (item.timestamp, item.track_id))


def _signal_evidence(
    change: LaneChange,
    observations: list[VehicleObservation],
    config: OffenseConfig,
) -> tuple[bool, float, list[str]]:
    window = [
        observation
        for observation in observations
        if change.timestamp - config.pre_signal_seconds
        <= observation.timestamp
        <= change.timestamp + config.post_signal_seconds
    ]
    if not window:
        return False, 0.0, []
    observed = [
        observation
        for observation in window
        if observation.turn_signal in {"off", "left", "right", "hazard"}
        and observation.turn_signal_confidence >= config.minimum_signal_confidence
    ]
    observation_ratio = len(observed) / len(window)
    expected_seen = any(
        observation.turn_signal == change.direction for observation in observed
    )
    return expected_seen, observation_ratio, sorted(
        {observation.turn_signal for observation in observed}
    )


def lane_change_events(
    vehicles: dict[int, list[VehicleObservation]],
    changes: list[LaneChange],
    config: OffenseConfig,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for change in changes:
        expected_seen, observation_ratio, states = _signal_evidence(
            change, vehicles[change.track_id], config
        )
        if (
            not expected_seen
            and observation_ratio >= config.minimum_signal_observation_ratio
        ):
            events.append(
                {
                    "type": "lane_change_without_signal",
                    "status": "suspected",
                    "track_id": change.track_id,
                    "label": change.label,
                    "start_frame": change.frame,
                    "end_frame": change.frame,
                    "start_seconds": round(change.timestamp, 3),
                    "end_seconds": round(change.timestamp, 3),
                    "evidence": {
                        "from_lane": change.from_lane,
                        "to_lane": change.to_lane,
                        "expected_signal": change.direction,
                        "observed_signal_states": states,
                        "signal_observation_ratio": round(observation_ratio, 3),
                    },
                }
            )
    return events


def consecutive_lane_change_events(
    changes: list[LaneChange], config: OffenseConfig
) -> list[dict[str, Any]]:
    by_track: dict[int, list[LaneChange]] = defaultdict(list)
    for change in changes:
        by_track[change.track_id].append(change)
    events: list[dict[str, Any]] = []
    for track_changes in by_track.values():
        start_index = 0
        while start_index < len(track_changes):
            end_index = start_index
            while (
                end_index + 1 < len(track_changes)
                and track_changes[end_index + 1].timestamp
                - track_changes[start_index].timestamp
                <= config.consecutive_lane_change_window_seconds
            ):
                end_index += 1
            group = track_changes[start_index : end_index + 1]
            if len(group) >= 2:
                first, last = group[0], group[-1]
                events.append(
                    {
                        "type": "consecutive_lane_changes",
                        "status": "suspected",
                        "track_id": first.track_id,
                        "label": first.label,
                        "start_frame": first.frame,
                        "end_frame": last.frame,
                        "start_seconds": round(first.timestamp, 3),
                        "end_seconds": round(last.timestamp, 3),
                        "evidence": {
                            "lane_change_count": len(group),
                            "lane_sequence": [group[0].from_lane]
                            + [change.to_lane for change in group],
                            "window_seconds": round(
                                last.timestamp - first.timestamp, 3
                            ),
                        },
                    }
                )
                start_index = end_index + 1
            else:
                start_index += 1
    return events


def _red_duration_at(
    observations: list[LightObservation],
    timestamp: float,
    lane_index: int | None,
    config: OffenseConfig,
) -> float | None:
    eligible_indices = [
        index
        for index, observation in enumerate(observations)
        if observation.road_context_valid
        and observation.timestamp <= timestamp
        and timestamp - observation.timestamp
        <= config.maximum_observation_gap_seconds
    ]
    if not eligible_indices:
        return None
    index = eligible_indices[-1]
    current = observations[index]
    if current.state != "red" or current.confidence < config.minimum_traffic_light_confidence:
        return None
    if (
        current.applies_to_lanes is not None
        and lane_index is not None
        and lane_index not in current.applies_to_lanes
    ):
        return None
    start = current.timestamp
    while index > 0:
        previous = observations[index - 1]
        if (
            current.timestamp - previous.timestamp
            > config.maximum_observation_gap_seconds
            or not previous.road_context_valid
            or previous.state != "red"
            or previous.confidence < config.minimum_traffic_light_confidence
        ):
            break
        start = previous.timestamp
        current = previous
        index -= 1
    return timestamp - start


def red_light_events(
    vehicles: dict[int, list[VehicleObservation]],
    lights: dict[str, list[LightObservation]],
    config: OffenseConfig,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for track_id, observations in vehicles.items():
        emitted_signals: set[str] = set()
        for previous, current in zip(observations, observations[1:]):
            if (
                not previous.road_context_valid
                or not current.road_context_valid
                or previous.stop_line_position != "before"
                or current.stop_line_position != "after"
                or current.timestamp - previous.timestamp
                > config.maximum_observation_gap_seconds
                or current.signal_id is None
                or current.signal_id in emitted_signals
            ):
                continue
            light_history = lights.get(current.signal_id, [])
            red_duration = _red_duration_at(
                light_history, current.timestamp, previous.lane_index, config
            )
            if (
                red_duration is None
                or red_duration < config.minimum_red_stable_seconds
            ):
                continue
            emitted_signals.add(current.signal_id)
            events.append(
                {
                    "type": "red_light_running",
                    "status": "suspected",
                    "track_id": track_id,
                    "label": current.label,
                    "start_frame": previous.frame,
                    "end_frame": current.frame,
                    "start_seconds": round(previous.timestamp, 3),
                    "end_seconds": round(current.timestamp, 3),
                    "evidence": {
                        "signal_id": current.signal_id,
                        "lane_index": previous.lane_index,
                        "red_stable_seconds": round(red_duration, 3),
                        "stop_line_transition": "before_to_after",
                    },
                }
            )
    return events


def solid_line_crossing_events(
    observations: list[BoundaryCrossingObservation], config: OffenseConfig
) -> list[dict[str, Any]]:
    solid_types = {"solid", "double_solid", "solid_yellow", "double_yellow"}
    grouped: dict[tuple[int, str], list[BoundaryCrossingObservation]] = defaultdict(
        list
    )
    for observation in observations:
        if (
            observation.road_context_valid
            and observation.boundary_type in solid_types
            and observation.confidence >= config.minimum_boundary_confidence
        ):
            grouped[(observation.track_id, observation.boundary_id)].append(
                observation
            )

    events: list[dict[str, Any]] = []
    for (track_id, boundary_id), group in grouped.items():
        runs: list[list[BoundaryCrossingObservation]] = []
        current_run: list[BoundaryCrossingObservation] = []
        for observation in group:
            if (
                current_run
                and observation.timestamp - current_run[-1].timestamp
                > config.maximum_observation_gap_seconds
            ):
                runs.append(current_run)
                current_run = []
            current_run.append(observation)
        if current_run:
            runs.append(current_run)
        for run in runs:
            if len(run) < config.minimum_boundary_observations:
                continue
            first, last = run[0], run[-1]
            events.append(
                {
                    "type": "solid_line_crossing",
                    "status": "suspected",
                    "track_id": track_id,
                    "label": "vehicle",
                    "start_frame": first.frame,
                    "end_frame": last.frame,
                    "start_seconds": round(first.timestamp, 3),
                    "end_seconds": round(last.timestamp, 3),
                    "evidence": {
                        "boundary_id": boundary_id,
                        "boundary_type": first.boundary_type,
                        "observation_count": len(run),
                        "maximum_confidence": round(
                            max(item.confidence for item in run), 3
                        ),
                    },
                }
            )
    return events


def wrong_way_events(
    vehicles: dict[int, list[VehicleObservation]], config: OffenseConfig
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for track_id, observations in vehicles.items():
        runs: list[list[VehicleObservation]] = []
        current_run: list[VehicleObservation] = []
        for observation in observations:
            eligible = (
                observation.road_context_valid
                and observation.direction_alignment is not None
                and observation.direction_alignment
                <= config.maximum_wrong_way_alignment
                and observation.direction_confidence
                >= config.minimum_wrong_way_confidence
            )
            if not eligible:
                if current_run:
                    runs.append(current_run)
                    current_run = []
                continue
            if (
                current_run
                and observation.timestamp - current_run[-1].timestamp
                > config.maximum_observation_gap_seconds
            ):
                runs.append(current_run)
                current_run = []
            current_run.append(observation)
        if current_run:
            runs.append(current_run)

        for run in runs:
            first, last = run[0], run[-1]
            if last.timestamp - first.timestamp < config.minimum_wrong_way_seconds:
                continue
            events.append(
                {
                    "type": "wrong_way_driving",
                    "status": "suspected",
                    "track_id": track_id,
                    "label": first.label,
                    "start_frame": first.frame,
                    "end_frame": last.frame,
                    "start_seconds": round(first.timestamp, 3),
                    "end_seconds": round(last.timestamp, 3),
                    "evidence": {
                        "minimum_direction_alignment": round(
                            min(
                                item.direction_alignment
                                for item in run
                                if item.direction_alignment is not None
                            ),
                            3,
                        ),
                        "duration_seconds": round(
                            last.timestamp - first.timestamp, 3
                        ),
                        "observation_count": len(run),
                    },
                }
            )
    return events


def pedestrian_failure_to_yield_events(
    observations: list[CrosswalkInteractionObservation], config: OffenseConfig
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[int, int, str], list[CrosswalkInteractionObservation]
    ] = defaultdict(list)
    for observation in observations:
        if (
            observation.road_context_valid
            and observation.pedestrian_has_priority
            and observation.confidence >= config.minimum_crosswalk_confidence
        ):
            grouped[
                (
                    observation.vehicle_track_id,
                    observation.pedestrian_track_id,
                    observation.crosswalk_id,
                )
            ].append(observation)

    candidates: list[dict[str, Any]] = []
    for (vehicle_id, pedestrian_id, crosswalk_id), group in grouped.items():
        runs: list[list[CrosswalkInteractionObservation]] = []
        current_run: list[CrosswalkInteractionObservation] = []
        for observation in group:
            if (
                current_run
                and observation.timestamp - current_run[-1].timestamp
                > config.maximum_observation_gap_seconds
            ):
                runs.append(current_run)
                current_run = []
            current_run.append(observation)
        if current_run:
            runs.append(current_run)
        for run in runs:
            crossed = [item for item in run if item.vehicle_crossed_crosswalk]
            if (
                len(run) < config.minimum_crosswalk_observations
                or not crossed
                or any(item.vehicle_stopped_before_crosswalk for item in run)
            ):
                continue
            first, last = run[0], run[-1]
            confidence_sum = sum(item.confidence for item in run)
            candidates.append(
                {
                    "type": "pedestrian_failure_to_yield",
                    "status": "suspected",
                    "track_id": vehicle_id,
                    "label": "vehicle",
                    "start_frame": first.frame,
                    "end_frame": last.frame,
                    "start_seconds": round(first.timestamp, 3),
                    "end_seconds": round(last.timestamp, 3),
                    "evidence": {
                        "pedestrian_track_id": pedestrian_id,
                        "pedestrian_track_ids": [pedestrian_id],
                        "crosswalk_id": crosswalk_id,
                        "merged_candidate_count": 1,
                        "observation_count": len(run),
                        "crossing_frame": crossed[0].frame,
                        "vehicle_stopped_before_crosswalk": False,
                        "maximum_confidence": round(
                            max(item.confidence for item in run), 3
                        ),
                        "mean_confidence": round(
                            confidence_sum / len(run), 3
                        ),
                    },
                    "_confidence_sum": confidence_sum,
                }
            )

    grouped_candidates: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped_candidates[
            (candidate["track_id"], candidate["evidence"]["crosswalk_id"])
        ].append(candidate)

    events: list[dict[str, Any]] = []
    for group in grouped_candidates.values():
        group.sort(key=lambda event: (event["start_seconds"], event["start_frame"]))
        current: dict[str, Any] | None = None
        for candidate in group:
            if current is None:
                current = candidate
                continue
            gap_seconds = candidate["start_seconds"] - current["end_seconds"]
            if gap_seconds > config.crosswalk_event_merge_seconds:
                events.append(current)
                current = candidate
                continue

            current["end_frame"] = max(current["end_frame"], candidate["end_frame"])
            current["end_seconds"] = max(
                current["end_seconds"], candidate["end_seconds"]
            )
            evidence = current["evidence"]
            candidate_evidence = candidate["evidence"]
            evidence["pedestrian_track_ids"] = list(
                dict.fromkeys(
                    evidence["pedestrian_track_ids"]
                    + candidate_evidence["pedestrian_track_ids"]
                )
            )
            evidence["merged_candidate_count"] += candidate_evidence[
                "merged_candidate_count"
            ]
            evidence["observation_count"] += candidate_evidence[
                "observation_count"
            ]
            evidence["crossing_frame"] = min(
                evidence["crossing_frame"], candidate_evidence["crossing_frame"]
            )
            evidence["maximum_confidence"] = max(
                evidence["maximum_confidence"],
                candidate_evidence["maximum_confidence"],
            )
            current["_confidence_sum"] += candidate["_confidence_sum"]
            evidence["mean_confidence"] = round(
                current["_confidence_sum"] / evidence["observation_count"], 3
            )
        if current is not None:
            events.append(current)

    for event in events:
        event.pop("_confidence_sum", None)
    return events


def littering_events(
    litter: dict[int, list[LitterObservation]], config: OffenseConfig
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for object_track_id, observations in litter.items():
        run: list[LitterObservation] = []
        runs: list[list[LitterObservation]] = []
        for observation in observations:
            if observation.confidence < config.minimum_litter_confidence:
                continue
            if (
                run
                and observation.timestamp - run[-1].timestamp
                > config.maximum_observation_gap_seconds
            ):
                runs.append(run)
                run = []
            run.append(observation)
        if run:
            runs.append(run)
        for current_run in runs:
            if len(current_run) < config.minimum_litter_observations:
                continue
            first, last = current_run[0], current_run[-1]
            events.append(
                {
                    "type": "littering_from_vehicle",
                    "status": "suspected",
                    "track_id": first.source_track_id,
                    "label": "vehicle",
                    "start_frame": first.frame,
                    "end_frame": last.frame,
                    "start_seconds": round(first.timestamp, 3),
                    "end_seconds": round(last.timestamp, 3),
                    "evidence": {
                        "object_track_id": object_track_id,
                        "observation_count": len(current_run),
                        "maximum_confidence": round(
                            max(item.confidence for item in current_run), 3
                        ),
                        "last_box": (
                            list(last.box) if last.box is not None else None
                        ),
                    },
                }
            )
    return events


def analyze_scene(
    scene: dict[str, Any], config: OffenseConfig | None = None
) -> dict[str, Any]:
    active_config = config or OffenseConfig()
    active_config.validate()
    (
        vehicles,
        lights,
        litter,
        boundary_crossings,
        crosswalk_interactions,
    ) = collect_scene_observations(scene, active_config)
    changes = find_lane_changes(vehicles, active_config)
    events = lane_change_events(vehicles, changes, active_config)
    events.extend(consecutive_lane_change_events(changes, active_config))
    events.extend(red_light_events(vehicles, lights, active_config))
    events.extend(solid_line_crossing_events(boundary_crossings, active_config))
    events.extend(wrong_way_events(vehicles, active_config))
    events.extend(
        pedestrian_failure_to_yield_events(
            crosswalk_interactions, active_config
        )
    )
    events.extend(littering_events(litter, active_config))
    events.sort(key=lambda event: (event["start_seconds"], event["type"]))
    counts = Counter(event["type"] for event in events)
    return {
        "schema_version": 2,
        "source_video": scene.get("source_video"),
        "event_count": len(events),
        "event_counts": dict(sorted(counts.items())),
        "config": asdict(active_config),
        "events": events,
        "notice": (
            "Events are suspected behaviors for human review, not legal findings. "
            "Signal, lane, stop-line, and object associations must be visually verified."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_observations", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "offense_rules.json",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene = load_json(args.scene_observations, "scene observations")
    config = OffenseConfig.from_mapping(load_json(args.config, "offense config"))
    result = analyze_scene(scene, config)
    output_path = args.output or args.scene_observations.with_name(
        f"{args.scene_observations.stem}_offenses.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"events={result['event_count']}")
    for event_type, count in result["event_counts"].items():
        print(f"{event_type}={count}")
    print(f"output={output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
