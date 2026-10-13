import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from offense_analyzer import OffenseConfig, analyze_scene  # noqa: E402


def stable_camera_pose() -> dict:
    return {
        "mode": "road_context",
        "pan_degrees": 0.0,
        "tilt_degrees": -5.0,
        "pan_rate_deg_s": 0.0,
        "tilt_rate_deg_s": 0.0,
        "calibrated": True,
        "road_transform_valid": True,
        "confidence": 0.98,
    }


def vehicle(
    track_id: int,
    lane_index: int,
    signal: str = "off",
    stop_line: str = "unknown",
    signal_id: str | None = None,
    direction_alignment: float | None = None,
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
        "direction_alignment": direction_alignment,
        "direction_confidence": 0.95 if direction_alignment is not None else 0.0,
    }


def scene_with_vehicle_frames(items: list[dict]) -> dict:
    return {
        "source_fps": 10.0,
        "frame_records": [
            {
                "frame": index + 1,
                "timestamp_seconds": index / 10,
                "camera_pose": stable_camera_pose(),
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
                    "camera_pose": stable_camera_pose(),
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
                    "camera_pose": stable_camera_pose(),
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
                    "camera_pose": stable_camera_pose(),
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
                    "camera_pose": stable_camera_pose(),
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

    def test_fast_gimbal_motion_suppresses_road_context_rules(self) -> None:
        moving_pose = stable_camera_pose()
        moving_pose["pan_rate_deg_s"] = 20.0
        records = []
        for index, lane in enumerate([0] * 6 + [1] * 6):
            records.append(
                {
                    "frame": index + 1,
                    "timestamp_seconds": index / 10,
                    "camera_pose": moving_pose,
                    "vehicles": [vehicle(1, lane)],
                }
            )

        result = analyze_scene({"frame_records": records}, self.config)

        self.assertEqual(result["event_count"], 0)

    def test_solid_line_crossing_requires_consecutive_observations(self) -> None:
        records = [
            {
                "frame": index + 1,
                "timestamp_seconds": index / 10,
                "camera_pose": stable_camera_pose(),
                "lane_boundary_crossings": [
                    {
                        "track_id": 3,
                        "boundary_id": "centerline",
                        "boundary_type": "double_solid",
                        "confidence": 0.95,
                    }
                ],
            }
            for index in range(3)
        ]

        result = analyze_scene({"frame_records": records}, self.config)

        self.assertEqual(result["event_counts"]["solid_line_crossing"], 1)
        self.assertEqual(
            result["events"][0]["evidence"]["observation_count"], 3
        )

    def test_wrong_way_requires_a_sustained_run(self) -> None:
        records = [
            {
                "frame": index + 1,
                "timestamp_seconds": index / 10,
                "camera_pose": stable_camera_pose(),
                "vehicles": [vehicle(8, 0, direction_alignment=-0.8)],
            }
            for index in range(11)
        ]

        result = analyze_scene({"frame_records": records}, self.config)

        self.assertEqual(result["event_counts"]["wrong_way_driving"], 1)
        event = next(
            event for event in result["events"] if event["type"] == "wrong_way_driving"
        )
        self.assertEqual(event["track_id"], 8)

    def test_failure_to_yield_to_pedestrian_is_reported(self) -> None:
        records = [
            {
                "frame": index + 1,
                "timestamp_seconds": index / 10,
                "camera_pose": stable_camera_pose(),
                "crosswalk_interactions": [
                    {
                        "vehicle_track_id": 5,
                        "pedestrian_track_id": 11,
                        "crosswalk_id": "crosswalk_a",
                        "pedestrian_has_priority": True,
                        "vehicle_crossed_crosswalk": True,
                        "vehicle_stopped_before_crosswalk": False,
                        "confidence": 0.9,
                    }
                ],
            }
            for index in range(3)
        ]

        result = analyze_scene({"frame_records": records}, self.config)

        self.assertEqual(result["event_counts"]["pedestrian_failure_to_yield"], 1)

    def test_crosswalk_event_merges_pedestrian_id_changes(self) -> None:
        records = []
        for pedestrian_id, start_time in ((11, 0.0), (22, 0.8)):
            for offset in range(3):
                records.append(
                    {
                        "frame": int((start_time + offset / 10) * 10) + 1,
                        "timestamp_seconds": start_time + offset / 10,
                        "camera_pose": stable_camera_pose(),
                        "crosswalk_interactions": [
                            {
                                "vehicle_track_id": 5,
                                "pedestrian_track_id": pedestrian_id,
                                "crosswalk_id": "crosswalk_a",
                                "pedestrian_has_priority": True,
                                "vehicle_crossed_crosswalk": True,
                                "vehicle_stopped_before_crosswalk": False,
                                "confidence": 0.9,
                            }
                        ],
                    }
                )

        result = analyze_scene({"frame_records": records}, self.config)

        self.assertEqual(result["event_counts"]["pedestrian_failure_to_yield"], 1)
        evidence = result["events"][0]["evidence"]
        self.assertEqual(evidence["pedestrian_track_id"], 11)
        self.assertEqual(evidence["pedestrian_track_ids"], [11, 22])
        self.assertEqual(evidence["merged_candidate_count"], 2)
        self.assertEqual(evidence["observation_count"], 6)

    def test_crosswalk_event_does_not_merge_different_vehicles(self) -> None:
        records = []
        for vehicle_id in (5, 8):
            for offset in range(3):
                records.append(
                    {
                        "frame": offset + 1,
                        "timestamp_seconds": offset / 10,
                        "camera_pose": stable_camera_pose(),
                        "crosswalk_interactions": [
                            {
                                "vehicle_track_id": vehicle_id,
                                "pedestrian_track_id": 11,
                                "crosswalk_id": "crosswalk_a",
                                "pedestrian_has_priority": True,
                                "vehicle_crossed_crosswalk": True,
                                "vehicle_stopped_before_crosswalk": False,
                                "confidence": 0.9,
                            }
                        ],
                    }
                )

        result = analyze_scene({"frame_records": records}, self.config)

        self.assertEqual(result["event_counts"]["pedestrian_failure_to_yield"], 2)
        self.assertEqual(
            {event["track_id"] for event in result["events"]}, {5, 8}
        )

    def test_target_follow_mode_suppresses_road_geometry_rules(self) -> None:
        pose = stable_camera_pose()
        pose["mode"] = "target_follow"
        records = [
            {
                "frame": index + 1,
                "timestamp_seconds": index / 10,
                "camera_pose": pose,
                "vehicles": [vehicle(2, 0 if index < 6 else 1)],
            }
            for index in range(12)
        ]

        result = analyze_scene({"frame_records": records}, self.config)

        self.assertEqual(result["event_count"], 0)


if __name__ == "__main__":
    unittest.main()
