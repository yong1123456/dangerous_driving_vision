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


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    total_frames: int


def filter_detections_by_area(
    detections: Sequence[Detection],
    frame_shape: Sequence[int],
    max_area_ratio: float,
) -> list[Detection]:
    if not 0 < max_area_ratio <= 1:
        raise ValueError("max_area_ratio must be in the range (0, 1]")
    frame_area = int(frame_shape[0]) * int(frame_shape[1])
    if frame_area <= 0:
        raise ValueError("frame dimensions must be positive")
    return [
        detection
        for detection in detections
        if (
            max(0, detection.box[2] - detection.box[0])
            * max(0, detection.box[3] - detection.box[1])
            / frame_area
            <= max_area_ratio
        )
    ]


class VehicleTracker:
    def __init__(
        self,
        frame_rate: int = 10,
        track_threshold: float = 0.15,
        track_buffer: int = 90,
        match_threshold: float = 0.8,
        max_box_area_ratio: float = 0.35,
    ) -> None:
        self.max_box_area_ratio = max_box_area_ratio
        self.tracker = ByteTrack(
            track_thresh=track_threshold,
            track_buffer=track_buffer,
            match_thresh=match_threshold,
            frame_rate=frame_rate,
        )

    def update(
        self,
        detections: Sequence[Detection],
        frame_shape: Sequence[int],
    ) -> list[TrackedVehicle]:
        detections = filter_detections_by_area(
            detections, frame_shape, self.max_box_area_ratio
        )
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


def inspect_video(video_path: Path) -> VideoMetadata:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open input video: {video_path}")
    try:
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        total_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        capture.release()
    if width < 1 or height < 1:
        raise RuntimeError(f"Invalid video dimensions: {width}x{height}")
    if not np.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"Invalid video frame rate: {fps}")
    return VideoMetadata(width, height, fps, max(0, total_frames))


def video_frames(video_path: Path, limit: int | None) -> Iterator[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open input video: {video_path}")
    frame_count = 0
    try:
        while limit is None or frame_count < limit:
            ok, frame = capture.read()
            if not ok:
                break
            frame_count += 1
            yield frame
    finally:
        capture.release()
    if frame_count == 0:
        raise RuntimeError(f"No frames decoded from input video: {video_path}")


def open_video_writer(output_path: Path, metadata: VideoMetadata) -> cv2.VideoWriter:
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("Video output must use the .mp4 extension")
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        metadata.fps,
        (metadata.width, metadata.height),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Unable to open output video: {output_path}")
    return writer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/home/orangepi/yolov8n.rknn"))
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", type=Path)
    source.add_argument("--video", type=Path)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--frame-rate", type=int)
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--track-threshold", type=float, default=0.15)
    parser.add_argument("--track-buffer", type=int, default=90)
    parser.add_argument("--match-threshold", type=float, default=0.8)
    parser.add_argument("--max-box-area-ratio", type=float, default=0.35)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frames is not None and args.frames < 1:
        raise ValueError("--frames must be at least 1")

    video_metadata: VideoMetadata | None = None
    if args.video:
        video_metadata = inspect_video(args.video)
        frame_source = video_frames(args.video, args.frames)
        tracker_frame_rate = args.frame_rate or max(1, round(video_metadata.fps))
        output_path = args.output or PROJECT_ROOT / "recordings" / "tracking_test.mp4"
    else:
        frame_limit = args.frames or 10
        frame_source = (
            image_frames(args.image, frame_limit)
            if args.image
            else camera_frames(args.camera, frame_limit)
        )
        tracker_frame_rate = args.frame_rate or 10
        output_path = args.output or PROJECT_ROOT / "recordings" / "tracking_test.jpg"

    tracker = VehicleTracker(
        frame_rate=tracker_frame_rate,
        track_threshold=args.track_threshold,
        track_buffer=args.track_buffer,
        match_threshold=args.match_threshold,
        max_box_area_ratio=args.max_box_area_ratio,
    )
    histories: dict[int, deque[tuple[int, int]]] = defaultdict(
        lambda: deque(maxlen=30)
    )
    frame_records = []
    last_annotated: np.ndarray | None = None
    video_writer: cv2.VideoWriter | None = None
    processed_frames = 0
    started = time.perf_counter()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if video_metadata is not None:
        video_writer = open_video_writer(output_path, video_metadata)
    try:
        with VehicleDetector(
            args.model,
            confidence_threshold=args.confidence,
        ) as detector:
            for frame_index, frame in enumerate(frame_source, start=1):
                detections = detector.detect(frame)
                tracks = tracker.update(detections, frame.shape)
                last_annotated = annotate_tracks(frame, tracks, histories)
                if video_writer is not None:
                    video_writer.write(last_annotated)
                record = {
                    "frame": frame_index,
                    "timestamp_seconds": (
                        (frame_index - 1) / video_metadata.fps
                        if video_metadata is not None
                        else None
                    ),
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
                processed_frames = frame_index
                print(
                    f"frame={frame_index} detections={len(detections)} "
                    f"track_ids={record['track_ids']}"
                )
    finally:
        if video_writer is not None:
            video_writer.release()

    if last_annotated is None:
        raise RuntimeError("No frames were processed")
    elapsed = time.perf_counter() - started
    if video_metadata is None and not cv2.imwrite(str(output_path), last_annotated):
        raise RuntimeError(f"Unable to write output image: {output_path}")

    summary_path = output_path.with_suffix(".json")
    summary = {
        "frames": processed_frames,
        "elapsed_seconds": elapsed,
        "processing_fps": processed_frames / elapsed,
        "frame_width": int(last_annotated.shape[1]),
        "frame_height": int(last_annotated.shape[0]),
        "source_image": str(args.image) if args.image else None,
        "source_video": str(args.video) if args.video else None,
        "source_fps": video_metadata.fps if video_metadata else None,
        "source_total_frames": video_metadata.total_frames if video_metadata else None,
        "camera": None if args.image or args.video else args.camera,
        "frame_records": frame_records,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + chr(10),
        encoding="utf-8",
    )
    print(f"processing_fps={summary['processing_fps']:.2f}")
    print(f"output={output_path.resolve()}")
    print(f"summary={summary_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
