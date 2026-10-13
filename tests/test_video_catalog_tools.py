import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from audit_video_catalog import (  # noqa: E402
    audit_catalog,
    find_split_leakage,
    resolve_local_path,
    training_permission,
)
from run_china_validation_suite import select_local_videos, summarize_detections  # noqa: E402


class VideoCatalogToolsTest(unittest.TestCase):
    def test_relative_local_path_resolves_against_root(self) -> None:
        entry = {"local_path": "clips/a.mp4"}
        self.assertEqual(resolve_local_path(entry, Path("D:/data")), Path("D:/data/clips/a.mp4"))

    def test_training_requires_explicit_license_marker(self) -> None:
        allowed, _ = training_permission({"use": "training_candidate", "rights_status": "CC-BY-4.0"})
        denied, reason = training_permission({"use": "training_candidate", "rights_status": "terms_require_verification"})
        self.assertTrue(allowed)
        self.assertFalse(denied)
        self.assertEqual(reason, "training_license_not_explicitly_verified")

    def test_missing_and_no_local_entries_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = {
                "schema_version": 2,
                "local_evaluation_clips": [
                    {"id": "missing", "local_path": str(Path(directory) / "missing.mp4"), "use": "evaluation_only"},
                    {"id": "remote", "page_url": "https://example.invalid/video", "use": "manual_review_candidate"},
                ],
            }
            report = audit_catalog(catalog, include_hash=False)
        self.assertEqual(report["counts"], {"missing": 1, "no_local_path": 1})

    def test_downloaded_holdout_is_selected_and_filterable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "holdout.webm"
            video.write_bytes(b"test")
            catalog = {
                "local_evaluation_clips": [],
                "downloaded_sources": [
                    {"id": "holdout_a", "local_path": "holdout.webm", "split": "holdout"}
                ],
            }
            selected = select_local_videos(catalog, root=root, ids={"holdout_a"})
            excluded = select_local_videos(catalog, root=root, ids={"other"})
        self.assertEqual([(item[0]["id"], item[1].name) for item in selected], [("holdout_a", "holdout.webm")])
        self.assertEqual(excluded, [])

    def test_downloaded_open_candidate_is_selected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "wrong_way.webm"
            video.write_bytes(b"test")
            catalog = {
                "open_video_candidates": [
                    {
                        "id": "wrong_way_a",
                        "local_path": "wrong_way.webm",
                        "split": "holdout_candidate",
                    }
                ]
            }
            selected = select_local_videos(catalog, root=root, ids={"wrong_way_a"})
        self.assertEqual(
            [(item[0]["id"], item[1].name) for item in selected],
            [("wrong_way_a", "wrong_way.webm")],
        )

    def test_checksum_mismatch_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.mp4"
            path.write_bytes(b"not-video")
            catalog = {
                "downloaded_sources": [
                    {"id": "bad_hash", "local_path": str(path), "sha256": "0" * 64}
                ]
            }
            with patch("audit_video_catalog.probe_video", return_value=None):
                report = audit_catalog(catalog, include_hash=True)
        self.assertEqual(report["videos"][0]["status"], "checksum_mismatch")

    def test_source_group_cannot_cross_training_and_holdout(self) -> None:
        findings = find_split_leakage(
            [
                {"id": "train_clip", "split": "training", "source_group": "same_upload"},
                {"id": "test_clip", "split": "holdout_candidate", "source_group": "same_upload"},
            ]
        )
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["identity_type"], "source_group")
        self.assertEqual(findings[0]["partitions"], ["holdout", "training"])

    def test_validation_aliases_do_not_create_false_leakage(self) -> None:
        findings = find_split_leakage(
            [
                {"id": "a", "split": "validation", "source_group": "same_upload"},
                {"id": "b", "split": "external_validation", "source_group": "same_upload"},
            ]
        )
        self.assertEqual(findings, [])

    def test_detection_summary_separates_classes(self) -> None:
        summary = {
            "processed_frames": 2,
            "frame_records": [
                {"tracks": [{"track_id": 1, "label": "car", "confidence": 0.8}]},
                {
                    "tracks": [
                        {"track_id": 1, "label": "car", "confidence": 0.9},
                        {"track_id": 2, "label": "person", "confidence": 0.7},
                    ]
                },
            ],
        }
        result = summarize_detections(summary)
        self.assertEqual(result["frames_with_detections"], 2)
        self.assertEqual(result["class_switching_track_ids"], [])
        self.assertEqual(result["classes"]["car"]["unique_tracks"], 1)
        self.assertEqual(result["classes"]["car"]["median_track_frames"], 2.0)
        self.assertEqual(result["classes"]["car"]["short_tracks_under_5_frames"], 1)
        self.assertEqual(result["classes"]["person"]["frames"], 1)

    def test_detection_summary_reports_class_switching_track(self) -> None:
        result = summarize_detections(
            {
                "processed_frames": 2,
                "frame_records": [
                    {"tracks": [{"track_id": 5, "label": "car", "confidence": 0.8}]},
                    {"tracks": [{"track_id": 5, "label": "truck", "confidence": 0.8}]},
                ],
            }
        )
        self.assertEqual(result["class_switching_track_ids"], [5])


if __name__ == "__main__":
    unittest.main()
