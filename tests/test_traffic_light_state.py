import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from traffic_light_state import classify_traffic_light_state  # noqa: E402


class TrafficLightStateTest(unittest.TestCase):
    def test_classifies_red_light(self) -> None:
        frame = np.zeros((80, 40, 3), dtype=np.uint8)
        cv2.circle(frame, (20, 15), 6, (0, 0, 255), -1)

        result = classify_traffic_light_state(frame, (5, 2, 35, 45))

        self.assertEqual(result.state, "red")
        self.assertGreaterEqual(result.confidence, 0.9)

    def test_classifies_green_light(self) -> None:
        frame = np.zeros((80, 40, 3), dtype=np.uint8)
        cv2.circle(frame, (20, 60), 6, (0, 255, 0), -1)

        result = classify_traffic_light_state(frame, (5, 40, 35, 78))

        self.assertEqual(result.state, "green")
        self.assertGreaterEqual(result.confidence, 0.9)

    def test_returns_unknown_without_illuminated_color(self) -> None:
        frame = np.full((50, 30, 3), 70, dtype=np.uint8)

        result = classify_traffic_light_state(frame, (0, 0, 30, 50))

        self.assertEqual(result.state, "unknown")
        self.assertEqual(result.confidence, 0.0)

    def test_clamps_box_to_frame(self) -> None:
        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        cv2.circle(frame, (5, 5), 4, (0, 255, 255), -1)

        result = classify_traffic_light_state(frame, (-10, -10, 20, 20))

        self.assertEqual(result.state, "yellow")


if __name__ == "__main__":
    unittest.main()
