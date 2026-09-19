import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(PROJECT_ROOT / "third_party" / "supervision_bytetrack"),
)

from supervision.tracker.byte_tracker.basetrack import BaseTrack  # noqa: E402
from supervision.tracker.byte_tracker.core import ByteTrack  # noqa: E402


class ByteTrackVendorTest(unittest.TestCase):
    def setUp(self) -> None:
        BaseTrack._count = 0

    def test_track_id_survives_small_motion(self) -> None:
        tracker = ByteTrack()
        first = np.array([[0, 0, 100, 100, 0.9, 2]], dtype=np.float32)
        second = np.array([[5, 0, 105, 100, 0.9, 2]], dtype=np.float32)

        first_tracks = tracker.update_with_tensors(first)
        second_tracks = tracker.update_with_tensors(second)

        self.assertEqual([track.track_id for track in first_tracks], [1])
        self.assertEqual([track.track_id for track in second_tracks], [1])


if __name__ == "__main__":
    unittest.main()
