import sys
import tempfile
import types
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

if "rknnlite.api" not in sys.modules:
    rknnlite = types.ModuleType("rknnlite")
    rknnlite_api = types.ModuleType("rknnlite.api")
    rknnlite_api.RKNNLite = type("RKNNLite", (), {"NPU_CORE_0": 1})
    rknnlite.api = rknnlite_api
    sys.modules["rknnlite"] = rknnlite
    sys.modules["rknnlite.api"] = rknnlite_api

from vehicle_detector import Detection  # noqa: E402
from vehicle_tracker import (  # noqa: E402
    filter_detections_by_area,
    inspect_video,
    video_frames,
)


class VideoInputTest(unittest.TestCase):
    def test_video_metadata_and_frame_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.avi"
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                12.0,
                (64, 48),
            )
            self.assertTrue(writer.isOpened())
            for value in range(5):
                writer.write(np.full((48, 64, 3), value * 20, dtype=np.uint8))
            writer.release()

            metadata = inspect_video(path)
            frames = list(video_frames(path, limit=3))

            self.assertEqual((metadata.width, metadata.height), (64, 48))
            self.assertAlmostEqual(metadata.fps, 12.0, places=1)
            self.assertEqual(metadata.total_frames, 5)
            self.assertEqual(len(frames), 3)
            self.assertTrue(all(frame.shape == (48, 64, 3) for frame in frames))

    def test_oversized_detection_is_removed(self) -> None:
        detections = [
            Detection(
                class_id=2,
                label="car",
                confidence=0.8,
                box=(10, 10, 50, 40),
            ),
            Detection(
                class_id=2,
                label="car",
                confidence=0.9,
                box=(0, 0, 99, 99),
            ),
        ]

        filtered = filter_detections_by_area(
            detections, frame_shape=(100, 100, 3), max_area_ratio=0.35
        )
        self.assertEqual(filtered, [detections[0]])


if __name__ == "__main__":
    unittest.main()
