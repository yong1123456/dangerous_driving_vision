import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from risk_analyzer import AnalysisConfig, analyze_tracking  # noqa: E402


def make_summary(boxes: list[tuple[int, int, int, int]]) -> dict:
    return {
        "frame_width": 1000,
        "frame_height": 500,
        "source_fps": 10.0,
        "frame_records": [
            {
                "frame": index + 1,
                "timestamp_seconds": index / 10,
                "tracks": [
                    {
                        "track_id": 1,
                        "label": "car",
                        "confidence": 0.9,
                        "box": list(box),
                    }
                ],
            }
            for index, box in enumerate(boxes)
        ],
    }


class RiskAnalyzerTest(unittest.TestCase):
    def test_static_vehicle_does_not_create_event(self) -> None:
        summary = make_summary([(400, 200, 500, 300)] * 10)

        result = analyze_tracking(summary)

        self.assertEqual(result["event_count"], 0)

    def test_growing_vehicle_creates_rapid_approach_event(self) -> None:
        boxes = []
        for index in range(10):
            size = 70 + index * 8
            boxes.append((500 - size // 2, 300 - size, 500 + size // 2, 300))

        result = analyze_tracking(make_summary(boxes))

        approach_events = [
            event for event in result["events"] if event["type"] == "rapid_approach"
        ]
        self.assertEqual(len(approach_events), 1)
        self.assertLessEqual(
            approach_events[0]["evidence"]["minimum_ttc_seconds"], 3.0
        )

    def test_fast_sideways_vehicle_creates_lateral_event(self) -> None:
        boxes = [
            (100 + index * 25, 250, 200 + index * 25, 350)
            for index in range(10)
        ]

        result = analyze_tracking(make_summary(boxes))

        lateral_events = [
            event
            for event in result["events"]
            if event["type"] == "rapid_lateral_motion"
        ]
        self.assertEqual(len(lateral_events), 1)
        self.assertEqual(lateral_events[0]["evidence"]["direction"], "right")

    def test_old_summary_requires_explicit_frame_size(self) -> None:
        summary = make_summary([(400, 200, 500, 300)] * 10)
        del summary["frame_width"]
        del summary["frame_height"]

        with self.assertRaisesRegex(ValueError, "Frame size is missing"):
            analyze_tracking(summary)

        result = analyze_tracking(summary, frame_width=1000, frame_height=500)
        self.assertEqual(result["event_count"], 0)

    def test_detection_gap_is_not_treated_as_continuous_motion(self) -> None:
        summary = make_summary(
            [
                (450, 220, 550, 320),
                (450, 220, 550, 320),
                (450, 220, 550, 320),
                (450, 220, 550, 320),
                (450, 220, 550, 320),
                (300, 100, 700, 500),
            ]
        )
        summary["frame_records"][-1]["timestamp_seconds"] = 2.0

        result = analyze_tracking(summary, config=AnalysisConfig())

        self.assertEqual(result["event_count"], 0)


if __name__ == "__main__":
    unittest.main()
