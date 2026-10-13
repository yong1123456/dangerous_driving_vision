import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from detection_geometry import box_area_ratio  # noqa: E402


class CpuTrackingScriptTest(unittest.TestCase):
    def test_box_area_ratio_clips_to_frame(self) -> None:
        self.assertAlmostEqual(box_area_ratio((-10, -10, 50, 50), 100, 100), 0.25)

    def test_oversized_false_detection_exceeds_default_limit(self) -> None:
        ratio = box_area_ratio((25, 133, 1274, 711), 1280, 720)
        self.assertGreater(ratio, 0.35)

    def test_real_vehicle_box_stays_below_default_limit(self) -> None:
        ratio = box_area_ratio((351, 152, 839, 516), 1280, 720)
        self.assertLess(ratio, 0.35)


if __name__ == "__main__":
    unittest.main()
