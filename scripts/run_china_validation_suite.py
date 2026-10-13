#!/usr/bin/env python3
"""Run the CPU detector/tracker over cataloged local Chinese road clips."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from audit_video_catalog import audit_catalog, load_catalog, resolve_local_path


def summarize_detections(summary: dict[str, Any]) -> dict[str, Any]:
    by_label: dict[str, dict[str, Any]] = {}
    track_lengths: dict[tuple[str, int], int] = {}
    labels_by_track: dict[int, set[str]] = {}
    frames_with_detections = 0
    for record in summary.get("frame_records", []):
        tracks = record.get("tracks", [])
        if tracks:
            frames_with_detections += 1
        labels_in_frame: set[str] = set()
        for track in tracks:
            label = str(track.get("label", "unknown"))
            item = by_label.setdefault(
                label,
                {"observations": 0, "track_ids": set(), "maximum_confidence": 0.0, "frames": 0},
            )
            item["observations"] += 1
            item["track_ids"].add(int(track["track_id"]))
            item["maximum_confidence"] = max(item["maximum_confidence"], float(track.get("confidence", 0.0)))
            track_key = (label, int(track["track_id"]))
            track_lengths[track_key] = track_lengths.get(track_key, 0) + 1
            labels_by_track.setdefault(int(track["track_id"]), set()).add(label)
            labels_in_frame.add(label)
        for label in labels_in_frame:
            by_label[label]["frames"] += 1
    classes: dict[str, dict[str, Any]] = {}
    for label, item in sorted(by_label.items()):
        lengths = [length for (track_label, _), length in track_lengths.items() if track_label == label]
        classes[label] = {
            "observations": item["observations"],
            "unique_tracks": len(item["track_ids"]),
            "frames": item["frames"],
            "maximum_confidence": round(item["maximum_confidence"], 5),
            "median_track_frames": float(statistics.median(lengths)) if lengths else 0.0,
            "maximum_track_frames": max(lengths, default=0),
            "stable_tracks_10_frames": sum(length >= 10 for length in lengths),
            "short_tracks_under_5_frames": sum(length < 5 for length in lengths),
        }
    return {
        "processed_frames": int(summary.get("processed_frames", 0)),
        "frames_with_detections": frames_with_detections,
        "class_switching_track_ids": sorted(
            track_id for track_id, labels in labels_by_track.items() if len(labels) > 1
        ),
        "classes": classes,
    }


def select_local_videos(
    catalog: dict[str, Any],
    *,
    root: Path | None = None,
    max_clips: int | None = None,
    ids: set[str] | None = None,
) -> list[tuple[dict[str, Any], Path]]:
    selected: list[tuple[dict[str, Any], Path]] = []
    for group in ("local_evaluation_clips", "downloaded_sources", "open_video_candidates"):
        for entry in catalog.get(group, []):
            if not isinstance(entry, dict):
                continue
            if ids is not None and str(entry.get("id")) not in ids:
                continue
            path = resolve_local_path(entry, root)
            if path is not None and path.is_file():
                selected.append((entry, path))
    if max_clips is not None:
        selected = selected[:max_clips]
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "china_video_catalog.json")
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parents[1] / "models" / "yolov8n.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--scene-context", action="store_true")
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Ultralytics tracker YAML name or local YAML path",
    )
    parser.add_argument("--id", action="append", dest="ids", help="Catalog video ID; repeatable")
    parser.add_argument("--dry-run", action="store_true", help="Only list available clips and their targets")
    args = parser.parse_args()
    catalog = load_catalog(args.catalog)
    catalog_root = args.root or args.catalog.resolve().parent.parent
    selected = select_local_videos(
        catalog,
        root=catalog_root,
        max_clips=args.max_clips,
        ids=set(args.ids) if args.ids else None,
    )
    if not selected:
        print("No local evaluation clips are available.")
        return 2
    for entry, path in selected:
        targets = ",".join(str(value) for value in entry.get("review_targets", [])) or "unlabeled"
        print(f"video={path} targets={targets}")
    if args.dry_run:
        return 0

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    sys.path.insert(0, str(project_root / "scripts"))
    from test_traffic_videos_cpu import process_video  # noqa: E402
    from ultralytics import YOLO  # noqa: E402

    model = YOLO(str(args.model))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for entry, path in selected:
        summary = process_video(
            model,
            path,
            args.output_dir,
            device=args.device,
            include_scene_context=args.scene_context,
            tracker=args.tracker,
        )
        manifest.append(
            {
                "id": entry.get("id"),
                "source": str(path),
                "review_targets": entry.get("review_targets", []),
                "split": entry.get("split"),
                "rights_status": entry.get("rights_status"),
                "tracker": summary["tracker"],
                "tracking_json": summary["output_json"],
                "tracking_video": summary["output_video"],
                "processing_fps": summary["processing_fps"],
                "unique_tracks": summary["unique_tracks"],
                "unique_vehicle_tracks": summary["unique_vehicle_tracks"],
                "detection_summary": summarize_detections(summary),
            }
        )
    output = args.output_dir / "china_validation_suite.json"
    output.write_text(json.dumps({"schema_version": 1, "clips": manifest}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"output={output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
