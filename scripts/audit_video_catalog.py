#!/usr/bin/env python3
"""Audit local video entries in the Chinese-source catalog.

This tool is deliberately conservative: a URL is not a local video, and a
public GitHub repository license is not automatically a license for its data.
The report is suitable for deciding what can enter evaluation or training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
TRAINING_LICENSE_MARKERS = (
    "cc-by",
    "cc0",
    "apache-2.0",
    "mit",
    "bsd",
    "dataset_license_allows_training",
)


def load_catalog(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("catalog root must be an object")
    return value


def iter_video_entries(catalog: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield entries that may point at a local video, without deduplicating."""
    for group_name in (
        "local_evaluation_clips",
        "downloaded_sources",
        "official_video_candidates",
        "open_video_candidates",
    ):
        for entry in catalog.get(group_name, []):
            if isinstance(entry, dict):
                yield {"catalog_group": group_name, **entry}


def resolve_local_path(entry: dict[str, Any], root: Path | None = None) -> Path | None:
    raw = entry.get("local_path") or entry.get("downloaded_path")
    if raw:
        path = Path(str(raw)).expanduser()
    else:
        filename = entry.get("filename")
        source_root = entry.get("source_root")
        if not filename or not source_root:
            return None
        path = Path(str(source_root)).expanduser() / str(filename)
    if not path.is_absolute() and root is not None:
        path = root / path
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_video(path: Path) -> dict[str, Any] | None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size:stream=codec_name,width,height,avg_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    value: dict[str, Any] | None = None
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode == 0:
            parsed = json.loads(completed.stdout)
            if isinstance(parsed, dict):
                value = parsed
    except (OSError, json.JSONDecodeError):
        value = None
    if value is None:
        # OpenCV is already a project dependency on Windows and provides a
        # useful fallback when ffprobe is not installed.
        try:
            import cv2

            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                return None
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            result: dict[str, Any] = {
                "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
                "avg_frame_rate": f"{fps:g}",
            }
            if fps > 0 and frames > 0:
                result["duration"] = frames / fps
            capture.release()
            return result
        except (ImportError, OSError):
            return None
        except Exception:
            return None
    streams = [item for item in value.get("streams", []) if isinstance(item, dict)]
    video = next((item for item in streams if item.get("width") and item.get("height")), {})
    fmt = value.get("format", {})
    result: dict[str, Any] = {}
    for key in ("duration", "size"):
        if key in fmt:
            try:
                result[key] = float(fmt[key])
            except (TypeError, ValueError):
                result[key] = fmt[key]
    for key in ("codec_name", "width", "height", "avg_frame_rate"):
        if key in video:
            result[key] = video[key]
    return result


def training_permission(entry: dict[str, Any]) -> tuple[bool, str]:
    use = str(entry.get("use", "")).lower()
    rights = str(entry.get("rights_status", "")).lower()
    if use not in {"training_candidate", "training"}:
        return False, "entry_use_is_not_training_candidate"
    if any(marker in rights for marker in TRAINING_LICENSE_MARKERS):
        return True, "rights_marker_allows_training_candidate"
    return False, "training_license_not_explicitly_verified"


def split_partition(split: object) -> str | None:
    value = str(split or "").lower()
    if not value:
        return None
    if "train" in value:
        return "training"
    if "holdout" in value or "test" in value:
        return "holdout"
    if "validation" in value or value == "val":
        return "validation"
    return value


def find_split_leakage(videos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for video in videos:
        partition = split_partition(video.get("split"))
        if partition is None:
            continue
        identities = {
            "source_group": video.get("source_group"),
            "sha256": video.get("sha256") or video.get("catalog_sha256"),
            "direct_url": video.get("direct_url"),
            "page_url": video.get("page_url"),
        }
        for kind, raw_value in identities.items():
            value = str(raw_value or "").strip().lower()
            if value:
                buckets.setdefault((kind, value), []).append(video)

    findings: list[dict[str, Any]] = []
    for (kind, value), items in sorted(buckets.items()):
        partitions = sorted({split_partition(item.get("split")) for item in items} - {None})
        if len(partitions) <= 1:
            continue
        findings.append(
            {
                "identity_type": kind,
                "identity": value,
                "partitions": partitions,
                "video_ids": sorted({str(item.get("id")) for item in items}),
            }
        )
    return findings


def audit_catalog(
    catalog: dict[str, Any], *, root: Path | None = None, include_hash: bool = True
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for entry in iter_video_entries(catalog):
        path = resolve_local_path(entry, root)
        item: dict[str, Any] = {
            "id": entry.get("id"),
            "catalog_group": entry.get("catalog_group"),
            "use": entry.get("use"),
            "rights_status": entry.get("rights_status"),
            "review_targets": entry.get("review_targets", []),
            "split": entry.get("split"),
            "source_group": entry.get("source_group"),
            "catalog_sha256": entry.get("sha256"),
            "direct_url": entry.get("direct_url"),
            "page_url": entry.get("page_url"),
            "path": str(path) if path else None,
        }
        allowed, reason = training_permission(entry)
        item["training_allowed"] = allowed
        item["training_decision"] = reason
        if path is None:
            item["status"] = "no_local_path"
            results.append(item)
            continue
        if not path.is_file():
            item["status"] = "missing"
            results.append(item)
            continue
        if path.suffix.lower() not in VIDEO_SUFFIXES:
            item["status"] = "not_a_supported_video_suffix"
            results.append(item)
            continue
        item["status"] = "ok"
        item["bytes"] = path.stat().st_size
        if include_hash:
            item["sha256"] = sha256_file(path)
        metadata = probe_video(path)
        item["ffprobe"] = metadata
        expected_hash = str(entry.get("sha256", "")).lower()
        if expected_hash and include_hash:
            item["hash_matches_catalog"] = item["sha256"] == expected_hash
            if not item["hash_matches_catalog"]:
                item["status"] = "checksum_mismatch"
        results.append(item)

    counts: dict[str, int] = {}
    for item in results:
        status = str(item["status"])
        counts[status] = counts.get(status, 0) + 1
    split_leakage = find_split_leakage(results)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "catalog_schema_version": catalog.get("schema_version"),
        "counts": dict(sorted(counts.items())),
        "training_allowed_count": sum(1 for item in results if item["training_allowed"]),
        "split_leakage_count": len(split_leakage),
        "split_leakage": split_leakage,
        "videos": results,
        "notice": (
            "A local file can be used only for the cataloged purpose. "
            "Training permission requires an explicit dataset or copyright license."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "china_video_catalog.json")
    parser.add_argument("--root", type=Path, help="Base directory for relative catalog paths")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-hash", action="store_true", help="Skip SHA-256 for faster repeated audits")
    parser.add_argument("--strict", action="store_true", help="Return failure when a referenced local file is missing")
    args = parser.parse_args()
    catalog_root = args.root or args.catalog.resolve().parent.parent
    report = audit_catalog(load_catalog(args.catalog), root=catalog_root, include_hash=not args.no_hash)
    output = args.output or args.catalog.resolve().parent.parent / "recordings" / "china_video_catalog_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"videos={len(report['videos'])} counts={report['counts']} "
        f"training_allowed={report['training_allowed_count']} "
        f"split_leakage={report['split_leakage_count']}"
    )
    print(f"output={output.resolve()}")
    required_missing = any(
        item["status"] in {"missing", "no_local_path"}
        and item["catalog_group"] in {"local_evaluation_clips", "downloaded_sources"}
        for item in report["videos"]
    )
    if args.strict and (required_missing or report["split_leakage_count"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
