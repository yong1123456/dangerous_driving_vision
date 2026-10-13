import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from create_video_contact_sheet import read_near_frame  # noqa: E402


class FakeCapture:
    def __init__(self, readable_indices: set[int]) -> None:
        self.readable_indices = readable_indices
        self.index = 0

    def set(self, _property: int, value: int) -> bool:
        self.index = int(value)
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.index not in self.readable_indices:
            return False, None
        return True, np.full((2, 2, 3), self.index, dtype=np.uint8)


class ContactSheetTest(unittest.TestCase):
    def test_unreadable_reported_tail_frame_backtracks(self) -> None:
        index, frame = read_near_frame(FakeCapture({232}), 234)

        self.assertEqual(index, 232)
        self.assertEqual(int(frame[0, 0, 0]), 232)

    def test_missing_backtrack_window_raises(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Unable to read frame 101"):
            read_near_frame(FakeCapture({80}), 100, maximum_backtrack=5)


if __name__ == "__main__":
    unittest.main()
