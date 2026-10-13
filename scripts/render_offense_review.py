#!/usr/bin/env python3
"""Render calibrated offense evidence on top of a source video for review."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from road_geometry import parse_boundaries, parse_crosswalks


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def polygon_pixels(points: tuple[tuple[float, float], ...]) -> np.ndarray:
    return np.asarray([[round(x), round(y)] for x, y in points], dtype=np.int32)


def render(
    source_video: Path,
    tracking_json: Path,
    offense_json: Path,
    geometry_json: Path,
    output_path: Path,
    highlight_before_seconds: float = 1.0,
    highlight_after_seconds: float = 1.0,
) -> None:
    tracking = load_object(tracking_json)
    offenses = load_object(offense_json)
    geometry = load_object(geometry_json)
    width = int(tracking["frame_width"])
    height = int(tracking["frame_height"])
    fps = float(tracking["source_fps"])
    normalized = bool(geometry.get("normalized", False))
    crosswalks = parse_crosswalks(
        geometry.get("crosswalks", []),
        width=width,
        height=height,
        normalized=normalized,
    )
    boundaries = parse_boundaries(
        geometry.get("boundaries", []),
        width=width,
        height=height,
        normalized=normalized,
    )
    records = {int(record["frame"]): record for record in tracking.get("frame_records", [])}
    events = offenses.get("events", [])
    event_by_frame: dict[int, list[dict[str, Any]]] = {}
    highlight_by_frame: dict[int, set[int]] = {}
    for event in events:
        for frame in range(int(event["start_frame"]), int(event["end_frame"]) + 1):
            event_by_frame.setdefault(frame, []).append(event)
        start = max(1, int(event["start_frame"]) - round(highlight_before_seconds * fps))
        end = int(event["end_frame"]) + round(highlight_after_seconds * fps)
        if event.get("track_id") is not None:
            for frame in range(start, end + 1):
                highlight_by_frame.setdefault(frame, set()).add(int(event["track_id"]))

    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source video: {source_video}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = tempfile.NamedTemporaryFile(
        prefix="offense_review_", suffix=output_path.suffix or ".mp4", delete=False
    )
    temporary_file.close()
    writer_path = Path(temporary_file.name)
    writer = cv2.VideoWriter(
        str(writer_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        writer_path.unlink(missing_ok=True)
        raise RuntimeError(f"Unable to create review video: {output_path}")
    frame_number = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1
            for crosswalk in crosswalks:
                cv2.polylines(
                    frame,
                    [polygon_pixels(crosswalk.pedestrian_zone)],
                    True,
                    (255, 180, 0),
                    3,
                )
                cv2.polylines(
                    frame,
                    [polygon_pixels(crosswalk.vehicle_conflict_zone)],
                    True,
                    (0, 0, 255),
                    3,
                )
            for boundary in boundaries:
                color = (
                    (0, 0, 255)
                    if boundary.boundary_type in {"solid", "double_solid", "solid_yellow", "double_yellow"}
                    else (180, 180, 180)
                )
                cv2.polylines(
                    frame,
                    [polygon_pixels(boundary.points)],
                    False,
                    color,
                    4 if color == (0, 0, 255) else 2,
                )
            record = records.get(frame_number, {})
            active_events = event_by_frame.get(frame_number, [])
            event_track_ids = highlight_by_frame.get(frame_number, set())
            for item in record.get("tracks", []):
                box = item.get("box")
                if not box or len(box) != 4:
                    continue
                x1, y1, x2, y2 = (int(round(float(value))) for value in box)
                label = str(item.get("label", "object"))
                track_id = int(item.get("track_id", -1))
                is_person = label in {"person", "pedestrian"}
                is_event_vehicle = track_id in event_track_ids
                color = (0, 0, 255) if is_event_vehicle else ((255, 180, 0) if is_person else (0, 220, 255))
                thickness = 5 if is_event_vehicle else 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(
                    frame,
                    f"#{track_id} {label} {float(item.get('confidence', 0.0)):.2f}",
                    (max(0, x1), max(26, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            if active_events:
                event = active_events[0]
                evidence = event.get("evidence", {})
                event_type = str(event.get("type", "suspected_offense"))
                if event_type == "pedestrian_failure_to_yield":
                    pedestrian_ids = evidence.get("pedestrian_track_ids") or [
                        evidence.get("pedestrian_track_id")
                    ]
                    pedestrian_text = "/".join(
                        str(track_id)
                        for track_id in pedestrian_ids
                        if track_id is not None
                    )
                    text = (
                        "SUSPECTED: pedestrian yield review | "
                        f"vehicle #{event.get('track_id')} / pedestrians #{pedestrian_text}"
                    )
                else:
                    text = f"SUSPECTED: {event_type} review | vehicle #{event.get('track_id')}"
                cv2.rectangle(frame, (0, 0), (width, 58), (0, 0, 140), -1)
                cv2.putText(
                    frame,
                    text,
                    (18, 39),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.putText(
                frame,
                f"frame {frame_number}  {((frame_number - 1) / fps):.2f}s  REVIEW ONLY",
                (18, height - 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        writer.release()
        capture.release()
    try:
        shutil.move(str(writer_path), str(output_path))
    finally:
        writer_path.unlink(missing_ok=True)
    print(f"output={output_path.resolve()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_video", type=Path)
    parser.add_argument("tracking_json", type=Path)
    parser.add_argument("offense_json", type=Path)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--highlight-before", type=float, default=1.0)
    parser.add_argument("--highlight-after", type=float, default=1.0)
    args = parser.parse_args()
    render(
        args.source_video,
        args.tracking_json,
        args.offense_json,
        args.geometry,
        args.output,
        highlight_before_seconds=args.highlight_before,
        highlight_after_seconds=args.highlight_after,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
