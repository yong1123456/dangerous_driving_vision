#!/usr/bin/env python3
"""Run a local Ultralytics COCO model and ByteTrack on traffic videos."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from detection_geometry import box_area_ratio  # noqa: E402
from target_lock import TargetLock, TargetLockConfig  # noqa: E402
from track_label_smoother import TrackLabelSmoother  # noqa: E402
from traffic_light_state import classify_traffic_light_state  # noqa: E402


VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
SCENE_CONTEXT_CLASSES = {0: "person", 9: "traffic_light"}


def parse_target_point(value: str | None) -> tuple[float, float] | None:
    if value is None:
        return None
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError("--target-point must use X,Y format")
    point = (float(parts[0]), float(parts[1]))
    if not all(math.isfinite(item) for item in point):
        raise ValueError("--target-point values must be finite")
    return point


def process_video(
    model: YOLO,
    video_path: Path,
    output_dir: Path,
    *,
    target_track_id: int | None = None,
    target_point: tuple[float, float] | None = None,
    lock_confirm_frames: int = 3,
    target_max_missed: int = 10,
    reassociation_confirm_frames: int = 2,
    max_reassociation_gap_frames: int = 3,
    max_box_area_ratio: float = 0.35,
    device: str = "cpu",
    include_scene_context: bool = False,
    tracker: str = "bytetrack.yaml",
    confidence_threshold: float = 0.15,
    image_size: int = 640,
) -> dict[str, Any]:
    if not 0 < max_box_area_ratio <= 1:
        raise ValueError("max_box_area_ratio must be in the range (0, 1]")
    if not 0 < confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be in the range (0, 1]")
    if image_size < 320:
        raise ValueError("image_size must be at least 320")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video metadata: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / f"{video_path.stem}_cpu_tracked.mp4"
    output_json = output_dir / f"{video_path.stem}_cpu_tracked.json"
    writer = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create output video: {output_video}")

    records: list[dict[str, Any]] = []
    unique_tracks: set[int] = set()
    unique_vehicle_tracks: set[int] = set()
    tracked_classes = dict(VEHICLE_CLASSES)
    if include_scene_context:
        tracked_classes.update(SCENE_CONTEXT_CLASSES)
    target_lock = (
        TargetLock(
            (height, width, 3),
            target_track_id=target_track_id,
            target_point=target_point,
            config=TargetLockConfig(
                confirm_frames=lock_confirm_frames,
                max_missed_frames=target_max_missed,
                reassociation_confirm_frames=reassociation_confirm_frames,
                max_reassociation_gap_frames=max_reassociation_gap_frames,
            ),
        )
        if target_track_id is not None or target_point is not None
        else None
    )
    label_smoother = TrackLabelSmoother()
    started = time.perf_counter()
    try:
        results = model.track(
            source=str(video_path),
            stream=True,
            persist=False,
            tracker=tracker,
            classes=list(tracked_classes),
            conf=confidence_threshold,
            agnostic_nms=True,
            imgsz=image_size,
            device=device,
            verbose=False,
        )
        for frame_number, result in enumerate(results, start=1):
            annotated = result.orig_img.copy()
            boxes = result.boxes
            tracks: list[dict[str, Any]] = []
            track_objects: list[Any] = []
            if boxes is not None and len(boxes):
                coordinates = boxes.xyxy.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy().astype(int)
                confidences = boxes.conf.cpu().numpy()
                ids = (
                    boxes.id.cpu().numpy().astype(int)
                    if boxes.id is not None
                    else [-1] * len(boxes)
                )
                for box, class_id, confidence, track_id in zip(
                    coordinates, class_ids, confidences, ids
                ):
                    if box_area_ratio(box, width, height) > max_box_area_ratio:
                        continue
                    if track_id < 0:
                        continue
                    unique_tracks.add(int(track_id))
                    label = tracked_classes.get(class_id, "unknown")
                    stable_label = label_smoother.observe(
                        int(track_id), class_id, label, float(confidence)
                    )
                    class_id = stable_label.class_id
                    label = stable_label.label
                    if class_id in VEHICLE_CLASSES:
                        unique_vehicle_tracks.add(int(track_id))
                    rounded_box = tuple(int(round(value)) for value in box)
                    track_record = {
                        "track_id": int(track_id),
                        "label": label,
                        "confidence": round(float(confidence), 5),
                        "box": [round(float(value), 2) for value in box],
                    }
                    state_label = ""
                    if label == "traffic_light":
                        light_state = classify_traffic_light_state(
                            result.orig_img, box
                        )
                        state_confidence = round(
                            light_state.confidence * float(confidence), 5
                        )
                        track_record.update(
                            {
                                "traffic_light_state": light_state.state,
                                "traffic_light_state_confidence": state_confidence,
                                "traffic_light_color_ratios": light_state.color_ratios,
                            }
                        )
                        state_label = (
                            f" {light_state.state} {state_confidence:.2f}"
                        )
                    tracks.append(track_record)
                    x1, y1, x2, y2 = rounded_box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 255), 2)
                    cv2.putText(
                        annotated,
                        f"#{int(track_id)} {label} {float(confidence):.2f}{state_label}",
                        (max(0, x1), max(24, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 220, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    track_objects.append(
                        SimpleNamespace(
                            track_id=int(track_id),
                            label=label,
                            box=rounded_box,
                        )
                    )
            target_state = target_lock.update(track_objects) if target_lock else None
            if target_state is not None:
                color = (0, 0, 255) if target_state.is_locked else (0, 165, 255)
                state_label = {
                    "locked": "TARGET LOCKED",
                    "temporarily_lost": "TARGET TEMPORARILY LOST",
                    "candidate": "TARGET CANDIDATE",
                    "lost": "TARGET LOST - NO AUTO SWITCH",
                    "waiting": "TARGET WAITING",
                }.get(target_state.state, "TARGET")
                if target_state.target_track_id is not None:
                    state_label += f" #{target_state.target_track_id}"
                if target_state.box is not None:
                    x1, y1, x2, y2 = target_state.box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 4)
                    text_origin = (
                        max(8, x1),
                        min(height - 12, max(28, y2 + 28)),
                    )
                else:
                    text_origin = (20, 40)
                cv2.putText(
                    annotated,
                    state_label,
                    text_origin,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            writer.write(annotated)
            record = {
                "frame": frame_number,
                "timestamp_seconds": round((frame_number - 1) / fps, 4),
                "tracks": tracks,
            }
            if target_state is not None:
                record["target_lock"] = target_state.as_dict()
            records.append(record)
    finally:
        writer.release()

    elapsed = time.perf_counter() - started
    summary = {
        "schema_version": 1,
        "source_video": str(video_path),
        "backend": f"ultralytics_{device}",
        "source_fps": fps,
        "frame_width": width,
        "frame_height": height,
        "source_frames": frame_count,
        "processed_frames": len(records),
        "processing_fps": round(len(records) / elapsed, 3) if elapsed else 0.0,
        "unique_tracks": len(unique_tracks),
        "unique_vehicle_tracks": len(unique_vehicle_tracks),
        "frame_records": records,
        "output_video": str(output_video),
        "output_json": str(output_json),
        "target_lock_requested": target_lock is not None,
        "max_box_area_ratio": max_box_area_ratio,
        "confidence_threshold": confidence_threshold,
        "image_size": image_size,
        "include_scene_context": include_scene_context,
        "class_agnostic_nms": True,
        "class_labels_smoothed": True,
        "tracker": tracker,
        "target_lock_config": (
            {
                "target_track_id": target_track_id,
                "target_point": list(target_point) if target_point else None,
                "confirm_frames": lock_confirm_frames,
                "max_missed_frames": target_max_missed,
                "reassociation_confirm_frames": reassociation_confirm_frames,
                "max_reassociation_gap_frames": max_reassociation_gap_frames,
            }
            if target_lock is not None
            else None
        ),
    }
    if target_lock is not None:
        target_states = [
            record["target_lock"]
            for record in records
            if "target_lock" in record
        ]
        state_counts: dict[str, int] = {}
        for state in target_states:
            name = str(state["state"])
            state_counts[name] = state_counts.get(name, 0) + 1
        summary["target_lock_summary"] = {
            "final_state": target_states[-1]["state"] if target_states else "waiting",
            "state_counts": state_counts,
            "track_ids_seen_by_lock": sorted(
                {
                    int(state["target_track_id"])
                    for state in target_states
                    if state.get("target_track_id") is not None
                }
            ),
            "max_missed_frames": max(
                (int(state["missed_frames"]) for state in target_states),
                default=0,
            ),
            "lock_acquired_frame": next(
                (
                    int(record["frame"])
                    for record in records
                    if record.get("target_lock", {}).get("state") == "locked"
                ),
                None,
            ),
            "reassociation_count": sum(
                1
                for state in target_states
                if state.get("match_method") == "reassociated"
            ),
            "notice": (
                "A locked target is a tracking result. Confirm any traffic offense "
                "with calibrated road observations and the original video."
            ),
        }
    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-track-id", type=int)
    parser.add_argument("--target-point", type=str)
    parser.add_argument("--lock-confirm-frames", type=int, default=3)
    parser.add_argument("--target-max-missed", type=int, default=10)
    parser.add_argument("--reassociation-confirm-frames", type=int, default=2)
    parser.add_argument("--max-reassociation-gap", type=int, default=3)
    parser.add_argument("--max-box-area-ratio", type=float, default=0.35)
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Ultralytics device, for example cpu or CUDA index 0",
    )
    parser.add_argument(
        "--scene-context",
        action="store_true",
        help="Also track persons and traffic lights for offense-scene review",
    )
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Ultralytics tracker YAML name or local YAML path",
    )
    args = parser.parse_args()

    model = YOLO(str(args.model))
    target_point = parse_target_point(args.target_point)
    for video in args.videos:
        summary = process_video(
            model,
            video,
            args.output_dir,
            target_track_id=args.target_track_id,
            target_point=target_point,
            lock_confirm_frames=args.lock_confirm_frames,
            target_max_missed=args.target_max_missed,
            reassociation_confirm_frames=args.reassociation_confirm_frames,
            max_reassociation_gap_frames=args.max_reassociation_gap,
            max_box_area_ratio=args.max_box_area_ratio,
            device=args.device,
            include_scene_context=args.scene_context,
            tracker=args.tracker,
            confidence_threshold=args.confidence,
            image_size=args.image_size,
        )
        print(
            f"video={video.name} frames={summary['processed_frames']} "
            f"tracks={summary['unique_vehicle_tracks']} "
            f"processing_fps={summary['processing_fps']:.2f} "
            f"output={summary['output_video']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
