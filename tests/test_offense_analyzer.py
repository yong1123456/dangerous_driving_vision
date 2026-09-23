import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from offense_analyzer import OffenseConfig, analyze_scene  # noqa: E402


def vehicle(
    track_id: int,
    lane_index: int,
    signal: str = "off",
    stop_line: str = "unknown",
    signal_id: str | None = None,
) -> dict:
    return {
        "track_id": track_id,
        "label": "car",
        "lane_index": lane_index,
        "lane_confidence": 0.95,
        "turn_signal": signal,
        "turn_signal_confidence": 0.95,
        "stop_line_position": stop_line,
        "signal_id": signal_id,
    }


def scene_with_vehicle_frames(items: list[dict]) -> dict:
    return {
        "source_fps": 10.0,
        "frame_records": [
            {
                "frame": index + 1,
                "timestamp_seconds": index / 10,
                "vehicles": [item],
            }
            for index, item in enumerate(items)
        ],
    }


class OffenseAnalyzerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = OffenseConfig(
            minimum_lane_dwell_seconds=0.4,
            maximum_observation_gap_seconds=0.2,
        )

    def test_unsignaled_and_consecutive_lane_changes_are_reported(self) -> None:
        items = (
            [vehicle(1, 0)] * 6
            + [vehicle(1, 1)] * 6
            + [vehicle(1, 2)] * 6
        )

        result = analyze_scene(scene_with_vehicle_frames(items), self.config)

        self.assertEqual(result["event_counts"]["lane_change_without_signal"], 2)
        self.assertEqual(result["event_counts"]["consecutive_lane_changes"], 1)
        consecutive = next(
            event
            for event in result["events"]
            if event["type"] == "consecutive_lane_changes"
        )
        self.assertEqual(consecutive["evidence"]["lane_sequence"], [0, 1, 2])

    def test_correct_turn_signal_suppresses_unsignaled_event(self) -> None:
        items = [vehicle(1, 0)] * 6 + [vehicle(1, 1, signal="right")] * 6

        result = analyze_scene(scene_with_vehicle_frames(items), self.config)

        self.assertEqual(result["event_count"], 0)

    def test_unknown_turn_signal_does_not_accuse_driver(self) -> None:
        items = [vehicle(1, 0)] * 6 + [vehicle(1, 1)] * 6
        for item in items:
            item["turn_signal"] = "unknown"
            item["turn_signal_confidence"] = 0.0

        result = analyze_scene(scene_with_vehicle_frames(items), self.config)

        self.assertEqual(result["event_count"], 0)

    def test_red_light_stop_line_crossing_is_reported(self) -> None:
        records = []
        for index in range(10):
            records.append(
                {
                    "frame": index + 1,
                    "timestamp_seconds": index / 10,
                    "vehicles": [
                        vehicle(
                            7,
                            0,
                            stop_line="before" if index < 8 else "after",
                            signal_id="main",
                        )
                    ],
                    "traffic_lights": [
                        {
                            "signal_id": "main",
                            "state": "red",
                            "confidence": 0.95,
                            "applies_to_lanes": [0],
                        }
                    ],
                }
            )

        result = analyze_scene({"frame_records": records}, self.config)

        self.assertEqual(result["event_counts"]["red_light_running"], 1)
        event = next(
            event for event in result["events"] if event["type"] == "red_light_running"
        )
        self.assertEqual(event["track_id"], 7)
        self.assertEqual(event["evidence"]["signal_id"], "main")

    def test_green_light_crossing_is_not_reported(self) -> None:
        scene = {
            "frame_records": [
                {
                    "frame": 1,
                    "timestamp_seconds": 0.0,
                    "vehicles": [
                        vehicle(7, 0, stop_line="before", signal_id="main")
                    ],
                    "traffic_lights": [
                        {"signal_id": "main", "state": "green", "confidence": 0.95}
                    ],
                },
                {
                    "frame": 2,
                    "timestamp_seconds": 0.1,
                    "vehicles": [vehicle(7, 0, stop_line="after", signal_id="main")],
                    "traffic_lights": [
                        {"signal_id": "main", "state": "green", "confidence": 0.95}
                    ],
                },
            ]
        }

        result = analyze_scene(scene, self.config)

        self.assertEqual(result["event_count"], 0)

    def test_litter_requires_multiple_confident_observations(self) -> None:
        records = []
        for index in range(4):
            records.append(
                {
                    "frame": index + 1,
                    "timestamp_seconds": index / 10,
                    "vehicles": [vehicle(4, 0)],
                    "discarded_objects": [
                        {
                            "object_track_id": 20,
                            "source_track_id": 4,
                            "confidence": 0.85,
                            "box": [100 + index, 100, 110 + index, 110],
                        }
                    ],
                }
            )

        result = analyze_scene({"frame_records": records}, self.config)

        self.assertEqual(result["event_counts"]["littering_from_vehicle"], 1)
        event = result["events"][0]
        self.assertEqual(event["evidence"]["object_track_id"], 20)
        self.assertEqual(event["evidence"]["observation_count"], 4)


if __name__ == "__main__":
    unittest.main()
