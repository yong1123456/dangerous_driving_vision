import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from review_clip_exporter import (  # noqa: E402
    build_clip_intervals,
    export_review_clips,
)


class ReviewClipExporterTest(unittest.TestCase):
    def test_overlapping_padded_events_are_merged(self) -> None:
        events = [
            {
                "type": "rapid_approach",
                "track_id": 1,
                "start_seconds": 2.0,
                "end_seconds": 2.5,
            },
            {
                "type": "rapid_lateral_motion",
                "track_id": 2,
                "start_seconds": 3.0,
                "end_seconds": 3.2,
            },
        ]

        intervals = build_clip_intervals(
            events, video_duration=5.0, before_seconds=1.0, after_seconds=1.0
        )

        self.assertEqual(len(intervals), 1)
        self.assertEqual(intervals[0].start_seconds, 1.0)
        self.assertEqual(intervals[0].end_seconds, 4.2)
        self.assertEqual(intervals[0].event_indices, (1, 2))
        self.assertEqual(intervals[0].track_ids, (1, 2))

    def test_event_at_video_boundary_is_clamped(self) -> None:
        events = [
            {
                "type": "rapid_approach",
                "track_id": 1,
                "start_seconds": 0.1,
                "end_seconds": 0.4,
            }
        ]

        interval = build_clip_intervals(events, video_duration=1.0)[0]

        self.assertEqual(interval.start_seconds, 0.0)
        self.assertEqual(interval.end_seconds, 1.0)

    def test_exports_playable_clip_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            source_path = directory_path / "source.avi"
            writer = cv2.VideoWriter(
                str(source_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                10.0,
                (64, 48),
            )
            self.assertTrue(writer.isOpened())
            for index in range(20):
                writer.write(np.full((48, 64, 3), index * 10, dtype=np.uint8))
            writer.release()
            report = {
                "events": [
                    {
                        "type": "rapid_approach",
                        "track_id": 1,
                        "start_seconds": 0.8,
                        "end_seconds": 1.0,
                    }
                ]
            }
            output_directory = directory_path / "clips"

            manifest = export_review_clips(
                source_path,
                report,
                output_directory,
                before_seconds=0.2,
                after_seconds=0.3,
            )

            self.assertEqual(manifest["clip_count"], 1)
            clip_path = Path(manifest["clips"][0]["path"])
            self.assertTrue(clip_path.is_file())
            self.assertTrue((output_directory / "review_manifest.json").is_file())
            capture = cv2.VideoCapture(str(clip_path))
            self.assertTrue(capture.isOpened())
            self.assertGreater(capture.get(cv2.CAP_PROP_FRAME_COUNT), 0)
            capture.release()


if __name__ == "__main__":
    unittest.main()
