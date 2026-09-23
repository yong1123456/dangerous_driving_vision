#!/usr/bin/env python3
"""Export short review clips for events in a trajectory risk report."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps


@dataclass(frozen=True)
class ClipInterval:
    start_seconds: float
    end_seconds: float
    event_indices: tuple[int, ...]
    event_types: tuple[str, ...]
    track_ids: tuple[int, ...]


def load_risk_report(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read risk report: {path}") from error
    if not isinstance(report, dict) or not isinstance(report.get("events"), list):
        raise ValueError("Risk report must contain an events list")
    return report


def inspect_video(path: Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {path}")
    try:
        info = VideoInfo(
            width=int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
            height=int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
            frame_count=int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
        )
    finally:
        capture.release()
    if info.width < 1 or info.height < 1:
        raise RuntimeError("Video has invalid dimensions")
    if not math.isfinite(info.fps) or info.fps <= 0 or info.frame_count < 1:
        raise RuntimeError("Video has invalid frame rate or frame count")
    return info


def build_clip_intervals(
    events: Iterable[dict[str, Any]],
    video_duration: float,
    before_seconds: float = 1.0,
    after_seconds: float = 1.0,
    merge_gap_seconds: float = 0.0,
) -> list[ClipInterval]:
    if video_duration <= 0:
        raise ValueError("video_duration must be positive")
    if before_seconds < 0 or after_seconds < 0 or merge_gap_seconds < 0:
        raise ValueError("clip padding and merge gap cannot be negative")

    intervals: list[ClipInterval] = []
    for event_index, event in enumerate(events, start=1):
        start = max(0.0, float(event["start_seconds"]) - before_seconds)
        end = min(video_duration, float(event["end_seconds"]) + after_seconds)
        if end <= start:
            continue
        intervals.append(
            ClipInterval(
                start_seconds=start,
                end_seconds=end,
                event_indices=(event_index,),
                event_types=(str(event["type"]),),
                track_ids=(int(event["track_id"]),),
            )
        )
    intervals.sort(key=lambda interval: interval.start_seconds)

    merged: list[ClipInterval] = []
    for interval in intervals:
        if not merged or interval.start_seconds > (
            merged[-1].end_seconds + merge_gap_seconds
        ):
            merged.append(interval)
            continue
        previous = merged[-1]
        merged[-1] = ClipInterval(
            start_seconds=previous.start_seconds,
            end_seconds=max(previous.end_seconds, interval.end_seconds),
            event_indices=previous.event_indices + interval.event_indices,
            event_types=tuple(dict.fromkeys(previous.event_types + interval.event_types)),
            track_ids=tuple(dict.fromkeys(previous.track_ids + interval.track_ids)),
        )
    return merged


def _open_writer(path: Path, info: VideoInfo) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        info.fps,
        (info.width, info.height),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Unable to open output video: {path}")
    return writer


def export_interval(
    source_video: Path,
    output_path: Path,
    info: VideoInfo,
    interval: ClipInterval,
) -> int:
    start_frame = max(0, math.floor(interval.start_seconds * info.fps))
    end_frame = min(info.frame_count, math.ceil(interval.end_seconds * info.fps))
    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {source_video}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    writer = _open_writer(output_path, info)
    written = 0
    try:
        for _ in range(start_frame, end_frame):
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
    finally:
        writer.release()
        capture.release()
    if written == 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"No frames exported for interval: {interval}")
    return written


def export_review_clips(
    source_video: Path,
    risk_report: dict[str, Any],
    output_directory: Path,
    before_seconds: float = 1.0,
    after_seconds: float = 1.0,
    merge_gap_seconds: float = 0.0,
) -> dict[str, Any]:
    info = inspect_video(source_video)
    intervals = build_clip_intervals(
        risk_report["events"],
        info.duration_seconds,
        before_seconds,
        after_seconds,
        merge_gap_seconds,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    clips = []
    for clip_index, interval in enumerate(intervals, start=1):
        start_frame = math.floor(interval.start_seconds * info.fps)
        end_frame = math.ceil(interval.end_seconds * info.fps)
        output_path = output_directory / (
            f"review_{clip_index:03d}_{start_frame:06d}_{end_frame:06d}.mp4"
        )
        written = export_interval(source_video, output_path, info, interval)
        clips.append(
            {
                "path": str(output_path.resolve()),
                "start_seconds": round(interval.start_seconds, 3),
                "end_seconds": round(interval.end_seconds, 3),
                "frames": written,
                "event_indices": list(interval.event_indices),
                "event_types": list(interval.event_types),
                "track_ids": list(interval.track_ids),
            }
        )

    manifest = {
        "schema_version": 1,
        "source_video": str(source_video.resolve()),
        "source_width": info.width,
        "source_height": info.height,
        "source_fps": info.fps,
        "source_frames": info.frame_count,
        "before_seconds": before_seconds,
        "after_seconds": after_seconds,
        "merge_gap_seconds": merge_gap_seconds,
        "clip_count": len(clips),
        "clips": clips,
    }
    manifest_path = output_directory / "review_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_video", type=Path)
    parser.add_argument("risk_report", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--before", type=float, default=1.0)
    parser.add_argument("--after", type=float, default=1.0)
    parser.add_argument("--merge-gap", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = load_risk_report(args.risk_report)
    output_directory = args.output_dir or args.risk_report.with_name(
        f"{args.risk_report.stem}_clips"
    )
    manifest = export_review_clips(
        args.source_video,
        report,
        output_directory,
        before_seconds=args.before,
        after_seconds=args.after,
        merge_gap_seconds=args.merge_gap,
    )
    print(f"clips={manifest['clip_count']}")
    print(f"output={output_directory.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
