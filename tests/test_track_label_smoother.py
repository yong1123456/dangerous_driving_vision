import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from track_label_smoother import TrackLabelSmoother  # noqa: E402


class TrackLabelSmootherTest(unittest.TestCase):
    def test_single_low_confidence_class_flip_is_suppressed(self) -> None:
        smoother = TrackLabelSmoother(window_size=5, minimum_observations=3)
        smoother.observe(7, 2, "car", 0.9)
        smoother.observe(7, 2, "car", 0.8)
        result = smoother.observe(7, 5, "bus", 0.3)
        self.assertEqual((result.class_id, result.label), (2, "car"))

    def test_sustained_evidence_can_change_vehicle_subclass(self) -> None:
        smoother = TrackLabelSmoother(window_size=5, minimum_observations=3)
        for _ in range(2):
            smoother.observe(3, 2, "car", 0.5)
        result = None
        for _ in range(5):
            result = smoother.observe(3, 7, "truck", 0.9)
        self.assertIsNotNone(result)
        self.assertEqual((result.class_id, result.label), (7, "truck"))

    def test_cross_family_change_resets_history(self) -> None:
        smoother = TrackLabelSmoother(window_size=5, minimum_observations=3)
        for _ in range(3):
            smoother.observe(4, 2, "car", 0.9)
        result = smoother.observe(4, 0, "person", 0.7)
        self.assertEqual((result.class_id, result.label, result.observations), (0, "person", 1))


if __name__ == "__main__":
    unittest.main()
