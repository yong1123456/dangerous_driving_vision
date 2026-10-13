#!/usr/bin/env python3
"""Find reviewable risk events in vehicle-tracking JSON output."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class AnalysisConfig:
    motion_window_seconds: float = 0.5
    max_observation_gap_seconds: float = 0.25
    merge_gap_seconds: float = 0.6
    minimum_box_area_ratio: float = 0.005
    minimum_scale_growth: float = 1.15
    maximum_ttc_seconds: float = 3.0
    minimum_lateral_speed: float = 0.20

    def validate(self) -> None:
        positive = {
            "motion_window_seconds": self.motion_window_seconds,
            "max_observation_gap_seconds": self.max_observation_gap_seconds,
            "merge_gap_seconds": self.merge_gap_seconds,
            "minimum_scale_growth": self.minimum_scale_growth,
            "maximum_ttc_seconds": self.maximum_ttc_seconds,
            "minimum_lateral_speed": self.minimum_lateral_speed,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.minimum_box_area_ratio <= 1:
            raise ValueError("minimum_box_area_ratio must be in the range [0, 1]")
        if self.minimum_scale_growth <= 1:
            raise ValueError("minimum_scale_growth must be greater than 1")


@dataclass(frozen=True)
class TrackObservation:
    frame: int
    timestamp: float
    track_id: int
    label: str
    confidence: float
    box: tuple[float, float, float, float]

    @property
    def center_x(self) -> float:
        return (self.box[0] + self.box[2]) / 2

    @property
    def area(self) -> float:
        return max(0.0, self.box[2] - self.box[0]) * max(
            0.0, self.box[3] - self.box[1]
        )


@dataclass(frozen=True)
class Candidate:
    event_type: str
    track_id: int
    label: str
    frame: int
    timestamp: float
    ttc_seconds: float | None = None
    scale_growth: float | None = None
    lateral_speed: float | None = None
    lateral_direction: str | None = None


def load_tracking_summary(path: Path) -> dict[str, Any]:
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read tracking JSON: {path}") from error
    if not isinstance(summary, dict) or not isinstance(summary.get("frame_records"), list):
        raise ValueError("Tracking JSON must contain a frame_records list")
    return summary


def resolve_frame_size(
    summary: dict[str, Any],
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> tuple[int, int]:
    width = frame_width if frame_width is not None else summary.get("frame_width")
    height = frame_height if frame_height is not None else summary.get("frame_height")
    if not isinstance(width, int) or not isinstance(height, int) or width < 1 or height < 1:
        raise ValueError(
            "Frame size is missing; provide --frame-width and --frame-height"
        )
    return width, height


def collect_observations(
    summary: dict[str, Any],
    *,
    target_lock_only: bool = False,
) -> dict[int, list[TrackObservation]]:
    source_fps = summary.get("source_fps")
    tracks: dict[int, list[TrackObservation]] = defaultdict(list)
    canonical_target_id: int | None = None
    if target_lock_only:
        requested_id = (summary.get("target_lock_config") or {}).get("target_track_id")
        if requested_id is not None:
            canonical_target_id = int(requested_id)
    for frame_record in summary["frame_records"]:
        frame = int(frame_record["frame"])
        timestamp_value = frame_record.get("timestamp_seconds")
        if timestamp_value is None:
            if not isinstance(source_fps, (int, float)) or source_fps <= 0:
                raise ValueError("Each frame needs a timestamp or a positive source_fps")
            timestamp = (frame - 1) / float(source_fps)
        else:
            timestamp = float(timestamp_value)
        frame_tracks = frame_record.get("tracks", [])
        if target_lock_only:
            lock = frame_record.get("target_lock")
            if not isinstance(lock, dict) or lock.get("state") != "locked":
                continue
            active_id = lock.get("target_track_id")
            if active_id is None:
                continue
            active_id = int(active_id)
            if canonical_target_id is None:
                canonical_target_id = active_id
            frame_tracks = [
                track for track in frame_tracks if int(track.get("track_id", -1)) == active_id
            ]
        for track in frame_tracks:
            box_values = track.get("box", [])
            if len(box_values) != 4:
                raise ValueError(f"Invalid box in frame {frame}")
            observation = TrackObservation(
                frame=frame,
                timestamp=timestamp,
                track_id=(
                    canonical_target_id
                    if target_lock_only and canonical_target_id is not None
                    else int(track["track_id"])
                ),
                label=str(track.get("label", "vehicle")),
                confidence=float(track.get("confidence", 0.0)),
                box=tuple(float(value) for value in box_values),
            )
            if observation.area > 0:
                tracks[observation.track_id].append(observation)
    for observations in tracks.values():
        observations.sort(key=lambda item: (item.timestamp, item.frame))
    if target_lock_only and canonical_target_id is None:
        raise ValueError("Tracking JSON does not contain a locked target")
    return dict(tracks)


def _continuous_window(
    observations: list[TrackObservation], start: int, end: int, max_gap: float
) -> bool:
    return all(
        0 < observations[index].timestamp - observations[index - 1].timestamp <= max_gap
        for index in range(start + 1, end + 1)
    )


def find_candidates(
    tracks: dict[int, list[TrackObservation]],
    frame_width: int,
    frame_height: int,
    config: AnalysisConfig,
) -> list[Candidate]:
    config.validate()
    frame_area = frame_width * frame_height
    candidates: list[Candidate] = []
    for observations in tracks.values():
        for current_index, current in enumerate(observations):
            previous_index = current_index - 1
            while previous_index >= 0:
                elapsed = current.timestamp - observations[previous_index].timestamp
                if elapsed >= config.motion_window_seconds:
                    break
                previous_index -= 1
            if previous_index < 0:
                continue
            previous = observations[previous_index]
            elapsed = current.timestamp - previous.timestamp
            if not _continuous_window(
                observations,
                previous_index,
                current_index,
                config.max_observation_gap_seconds,
            ):
                continue
            if current.area / frame_area < config.minimum_box_area_ratio:
                continue

            scale_growth = math.sqrt(current.area / previous.area)
            if scale_growth >= config.minimum_scale_growth:
                ttc_seconds = elapsed / (scale_growth - 1)
                if ttc_seconds <= config.maximum_ttc_seconds:
                    candidates.append(
                        Candidate(
                            event_type="rapid_approach",
                            track_id=current.track_id,
                            label=current.label,
                            frame=current.frame,
                            timestamp=current.timestamp,
                            ttc_seconds=ttc_seconds,
                            scale_growth=scale_growth,
                        )
                    )

            lateral_delta = (current.center_x - previous.center_x) / frame_width
            lateral_speed = abs(lateral_delta) / elapsed
            if lateral_speed >= config.minimum_lateral_speed:
                candidates.append(
                    Candidate(
                        event_type="rapid_lateral_motion",
                        track_id=current.track_id,
                        label=current.label,
                        frame=current.frame,
                        timestamp=current.timestamp,
                        lateral_speed=lateral_speed,
                        lateral_direction="right" if lateral_delta > 0 else "left",
                    )
                )
    return sorted(candidates, key=lambda item: (item.timestamp, item.track_id))


def _event_from_group(group: list[Candidate]) -> dict[str, Any]:
    first, last = group[0], group[-1]
    event: dict[str, Any] = {
        "type": first.event_type,
        "track_id": first.track_id,
        "label": first.label,
        "start_frame": first.frame,
        "end_frame": last.frame,
        "start_seconds": round(first.timestamp, 3),
        "end_seconds": round(last.timestamp, 3),
    }
    if first.event_type == "rapid_approach":
        minimum_ttc = min(
            candidate.ttc_seconds
            for candidate in group
            if candidate.ttc_seconds is not None
        )
        maximum_growth = max(
            candidate.scale_growth
            for candidate in group
            if candidate.scale_growth is not None
        )
        event["severity"] = "high" if minimum_ttc <= 1.5 else "medium"
        event["evidence"] = {
            "minimum_ttc_seconds": round(minimum_ttc, 3),
            "maximum_scale_growth": round(maximum_growth, 3),
        }
    else:
        maximum_speed = max(
            candidate.lateral_speed
            for candidate in group
            if candidate.lateral_speed is not None
        )
        directions = {candidate.lateral_direction for candidate in group}
        event["severity"] = "high" if maximum_speed >= 0.5 else "medium"
        event["evidence"] = {
            "maximum_frame_widths_per_second": round(maximum_speed, 3),
            "direction": directions.pop() if len(directions) == 1 else "mixed",
        }
    return event


def merge_candidates(
    candidates: Iterable[Candidate], merge_gap_seconds: float
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[(candidate.event_type, candidate.track_id)].append(candidate)

    events: list[dict[str, Any]] = []
    for group in grouped.values():
        group.sort(key=lambda item: item.timestamp)
        current_group = [group[0]]
        for candidate in group[1:]:
            if candidate.timestamp - current_group[-1].timestamp <= merge_gap_seconds:
                current_group.append(candidate)
            else:
                events.append(_event_from_group(current_group))
                current_group = [candidate]
        events.append(_event_from_group(current_group))
    return sorted(events, key=lambda event: (event["start_seconds"], event["track_id"]))


def analyze_tracking(
    summary: dict[str, Any],
    config: AnalysisConfig | None = None,
    frame_width: int | None = None,
    frame_height: int | None = None,
    target_lock_only: bool = False,
) -> dict[str, Any]:
    active_config = config or AnalysisConfig()
    active_config.validate()
    width, height = resolve_frame_size(summary, frame_width, frame_height)
    tracks = collect_observations(summary, target_lock_only=target_lock_only)
    candidates = find_candidates(tracks, width, height, active_config)
    events = merge_candidates(candidates, active_config.merge_gap_seconds)
    counts = Counter(event["type"] for event in events)
    return {
        "schema_version": 1,
        "frame_width": width,
        "frame_height": height,
        "track_count": len(tracks),
        "observation_count": sum(len(items) for items in tracks.values()),
        "event_count": len(events),
        "event_counts": dict(sorted(counts.items())),
        "analysis_scope": "target_lock" if target_lock_only else "all_tracks",
        "config": asdict(active_config),
        "events": events,
        "notice": (
            "Events are motion-based review candidates, not final dangerous-driving "
            "or legal determinations."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracking_json", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frame-width", type=int)
    parser.add_argument("--frame-height", type=int)
    parser.add_argument("--maximum-ttc", type=float, default=3.0)
    parser.add_argument("--minimum-lateral-speed", type=float, default=0.20)
    parser.add_argument(
        "--target-lock-only",
        action="store_true",
        help="Analyze only frames where the requested target is locked",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (args.frame_width is None) != (args.frame_height is None):
        raise ValueError("--frame-width and --frame-height must be provided together")
    config = AnalysisConfig(
        maximum_ttc_seconds=args.maximum_ttc,
        minimum_lateral_speed=args.minimum_lateral_speed,
    )
    summary = load_tracking_summary(args.tracking_json)
    result = analyze_tracking(
        summary,
        config=config,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        target_lock_only=args.target_lock_only,
    )
    output_path = args.output or args.tracking_json.with_name(
        f"{args.tracking_json.stem}_risk.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"tracks={result['track_count']}")
    print(f"events={result['event_count']}")
    for event_type, count in result["event_counts"].items():
        print(f"{event_type}={count}")
    print(f"output={output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
