#!/usr/bin/env python3
"""RKNN vehicle detection with Supervision ByteTrack multi-object tracking."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = PROJECT_ROOT / "third_party" / "supervision_bytetrack"
sys.path.insert(0, str(VENDOR_ROOT))

from supervision import ByteTrack  # noqa: E402
from vehicle_detector import VEHICLE_CLASSES, Detection, VehicleDetector  # noqa: E402


@dataclass(frozen=True)
class TrackedVehicle:
    track_id: int
    class_id: int
    label: str
    confidence: float
    box: tuple[int, int, int, int]


class VehicleTracker:
    def __init__(
        self,
        frame_rate: int = 10,
        track_threshold: float = 0.25,
        track_buffer: int = 30,
        match_threshold: float = 0.8,
    ) -> None:
        self.tracker = ByteTrack(
            track_thresh=track_threshold,
            track_buffer=track_buffer,
            match_thresh=match_threshold,
            frame_rate=frame_rate,
        )

    def update(self, detections: Sequence[Detection]) -> list[TrackedVehicle]:
        if detections:
            tensors = np.array(
                [
                    [
                        *detection.box,
                        detection.confidence,
                        detection.class_id,
                    ]
                    for detection in detections
                ],
                dtype=np.float32,
            )
        else:
            tensors = np.empty((0, 6), dtype=np.float32)

        tracks = self.tracker.update_with_tensors(tensors)
        return [
            TrackedVehicle(
                track_id=int(track.track_id),
                class_id=int(track.class_ids),
                label=VEHICLE_CLASSES.get(int(track.class_ids), "vehicle"),
                confidence=float(track.score),
                box=tuple(int(round(value)) for value in track.tlbr),
            )
            for track in tracks
        ]


def color_for_track(track_id: int) -> tuple[int, int, int]:
    palette = (
        (0, 220, 255),
        (80, 220, 80),
        (255, 160, 60),
        (220, 80, 220),
        (255, 100, 100),
        (100, 180, 255),
    )
    return palette[(track_id - 1) % len(palette)]


def annotate_tracks(
    frame: np.ndarray,
    tracks: Sequence[TrackedVehicle],
    histories: dict[int, deque[tuple[int, int]]],
) -> np.ndarray:
    annotated = frame.copy()
    for track in tracks:
        x1, y1, x2, y2 = track.box
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        histories[track.track_id].append(center)
        color = color_for_track(track.track_id)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            f"#{track.track_id} {track.label} {track.confidence:.2f}",
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
        points = np.array(histories[track.track_id], dtype=np.int32)
        if len(points) >= 2:
            cv2.polylines(annotated, [points], False, color, 2, cv2.LINE_AA)
    return annotated


def camera_frames(camera_index: int, limit: int) -> Iterator[np.ndarray]:
    camera = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    camera.set(cv2.CAP_PROP_FPS, 30)
    if not camera.isOpened():
        raise RuntimeError(f"Unable to open camera index {camera_index}")
    try:
        for _ in range(5):
            ok, _ = camera.read()
            if not ok:
                raise RuntimeError("Camera warmup frame read failed")
        for _ in range(limit):
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera frame read failed")
            yield frame
    finally:
        camera.release()


def image_frames(image_path: Path, limit: int) -> Iterator[np.ndarray]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Unable to read input image: {image_path}")
    for _ in range(limit):
        yield image.copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/home/orangepi/yolov8n.rknn"))
    parser.add_argument("--image", type=Path)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--frame-rate", type=int, default=10)
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "recordings" / "tracking_test.jpg",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frames < 1:
        raise ValueError("--frames must be at least 1")

    frame_source = (
        image_frames(args.image, args.frames)
        if args.image
        else camera_frames(args.camera, args.frames)
    )
    tracker = VehicleTracker(frame_rate=args.frame_rate)
    histories: dict[int, deque[tuple[int, int]]] = defaultdict(
        lambda: deque(maxlen=30)
    )
    frame_records = []
    last_annotated: np.ndarray | None = None
    started = time.perf_counter()

    with VehicleDetector(
        args.model,
        confidence_threshold=args.confidence,
    ) as detector:
        for frame_index, frame in enumerate(frame_source, start=1):
            detections = detector.detect(frame)
            tracks = tracker.update(detections)
            last_annotated = annotate_tracks(frame, tracks, histories)
            record = {
                "frame": frame_index,
                "track_ids": [track.track_id for track in tracks],
                "tracks": [
                    {
                        "track_id": track.track_id,
                        "class_id": track.class_id,
                        "label": track.label,
                        "confidence": track.confidence,
                        "box": list(track.box),
                    }
                    for track in tracks
                ],
            }
            frame_records.append(record)
            print(
                f"frame={frame_index} detections={len(detections)} "
                f"track_ids={record['track_ids']}"
            )

    if last_annotated is None:
        raise RuntimeError("No frames were processed")
    elapsed = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), last_annotated):
        raise RuntimeError(f"Unable to write output image: {args.output}")

    summary_path = args.output.with_suffix(".json")
    summary = {
        "frames": args.frames,
        "elapsed_seconds": elapsed,
        "processing_fps": args.frames / elapsed,
        "source_image": str(args.image) if args.image else None,
        "camera": None if args.image else args.camera,
        "frame_records": frame_records,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + chr(10),
        encoding="utf-8",
    )
    print(f"processing_fps={summary['processing_fps']:.2f}")
    print(f"output={args.output.resolve()}")
    print(f"summary={summary_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
