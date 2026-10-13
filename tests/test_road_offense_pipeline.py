import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from road_offense_pipeline import _select_offender, build_scene  # noqa: E402
from offense_analyzer import OffenseConfig, analyze_scene  # noqa: E402


def tracking() -> dict:
    records = []
    # Vehicle 7 stays in lane 0, then crosses the solid center boundary and
    # remains on the other side for two frames.
    xs = [220, 230, 250, 310, 350, 360]
    for index, x in enumerate(xs):
        records.append(
            {
                "frame": index + 1,
                "timestamp_seconds": index / 10,
                "tracks": [{"track_id": 7, "label": "car", "box": [x - 20, 300, x + 20, 400]}],
            }
        )
    return {
        "source_video": "sample.mp4",
        "source_fps": 10.0,
        "frame_width": 800,
        "frame_height": 400,
        "frame_records": records,
    }


def crosswalk_tracking(*, with_person: bool = True, stopped_first: bool = False) -> dict:
    xs = ([120] * 6 + [140, 160, 180, 200]) if stopped_first else [120, 140, 160, 180, 200, 220]
    records = []
    for index, x in enumerate(xs):
        tracks = [
            {
                "track_id": 3,
                "label": "car",
                "confidence": 0.9,
                "box": [x - 20, 300, x + 20, 380],
            }
        ]
        if with_person:
            tracks.append(
                {
                    "track_id": 22,
                    "label": "person",
                    "confidence": 0.9,
                    "box": [330, 300, 370, 380],
                }
            )
        records.append(
            {
                "frame": index + 1,
                "timestamp_seconds": index / 10,
                "tracks": tracks,
            }
        )
    return {
        "source_video": "crosswalk.mp4",
        "source_fps": 10.0,
        "frame_width": 800,
        "frame_height": 400,
        "frame_records": records,
    }


def crosswalk_geometry(*, confidence: float = 0.9) -> dict:
    return {
        "camera_pose": {
            "mode": "road_context",
            "calibrated": True,
            "road_transform_valid": True,
            "confidence": 0.95,
        },
        "crosswalks": [
            {
                "id": "crosswalk_a",
                "confidence": confidence,
                "pedestrian_zone": [[300, 280], [500, 280], [500, 400], [300, 400]],
                "vehicle_conflict_zone": [[80, 280], [280, 280], [280, 400], [80, 400]],
                "vehicle_approach_zone": [[80, 280], [280, 280], [280, 400], [80, 400]],
                "motion_window_seconds": 0.2,
                "minimum_vehicle_motion_per_second": 0.02,
                "stopped_motion_per_second": 0.005,
                "stop_hold_seconds": 0.3,
                "stop_lookback_seconds": 1.0,
            }
        ],
    }


def signal_tracking(state: str = "red", confidence: float = 0.95) -> dict:
    records = []
    vehicle_bottoms = [180, 190, 200, 210, 220, 230, 240, 265, 280]
    for index, bottom in enumerate(vehicle_bottoms):
        records.append(
            {
                "frame": index + 1,
                "timestamp_seconds": index / 10,
                "tracks": [
                    {
                        "track_id": 5,
                        "label": "car",
                        "confidence": 0.9,
                        "box": [200, bottom - 80, 260, bottom],
                    },
                    {
                        "track_id": 40,
                        "label": "traffic_light",
                        "confidence": 0.9,
                        "box": [360, 30, 400, 100],
                        "traffic_light_state": state,
                        "traffic_light_state_confidence": confidence,
                    },
                ],
            }
        )
    return {
        "source_video": "intersection.mp4",
        "source_fps": 10.0,
        "frame_width": 800,
        "frame_height": 400,
        "frame_records": records,
    }


def signal_geometry() -> dict:
    return {
        "camera_pose": {
            "mode": "road_context",
            "calibrated": True,
            "road_transform_valid": True,
            "confidence": 0.95,
        },
        "boundaries": [
            {"id": "left", "type": "dashed", "points": [[100, 400], [100, 100]]},
            {"id": "middle", "type": "dashed", "points": [[300, 400], [300, 100]]},
            {"id": "right", "type": "dashed", "points": [[600, 400], [600, 100]]},
        ],
        "traffic_signals": [
            {
                "id": "main",
                "confidence": 0.95,
                "roi": [[340, 10], [420, 10], [420, 120], [340, 120]],
                "applies_to_lanes": [0],
                "stop_line": {
                    "points": [[100, 250], [300, 250]],
                    "before_side": "negative",
                    "margin_pixels": 3,
                },
            }
        ],
    }


def polygon_signal_geometry(direction: list[int] | None = None) -> dict:
    geometry = signal_geometry()
    geometry.pop("boundaries")
    lane = {
        "index": 0,
        "confidence": 0.93,
        "polygon": [[100, 100], [300, 100], [300, 400], [100, 400]],
    }
    if direction is not None:
        lane["direction"] = direction
        lane["minimum_direction_alignment"] = 0.5
    geometry["lanes"] = [lane]
    return geometry


class RoadOffensePipelineTest(unittest.TestCase):
    def test_scene_contains_calibrated_lane_and_crossing_evidence(self) -> None:
        geometry = {
            "camera_pose": {
                "mode": "road_context",
                "calibrated": True,
                "road_transform_valid": True,
                "confidence": 0.95,
            },
            "boundaries": [
                {"id": "left", "type": "dashed", "points": [[100, 400], [100, 100]]},
                {"id": "solid", "type": "double_solid", "points": [[300, 400], [300, 100]]},
                {"id": "right", "type": "dashed", "points": [[600, 400], [600, 100]]},
            ],
            "boundary_hold_frames": 2,
        }
        scene = build_scene(tracking(), geometry)
        crossing_records = [record for record in scene["frame_records"] if record["lane_boundary_crossings"]]
        self.assertTrue(crossing_records)
        self.assertEqual(scene["frame_records"][0]["vehicles"][0]["lane_index"], 0)

    def test_no_event_means_no_confident_offender(self) -> None:
        selected = _select_offender({"events": [], "event_count": 0})
        self.assertEqual(selected["status"], "NO_CONFIDENT_OFFENDER")

    def test_solid_line_crossing_selects_the_matching_vehicle(self) -> None:
        geometry = {
            "camera_pose": {
                "mode": "road_context",
                "calibrated": True,
                "road_transform_valid": True,
                "confidence": 0.95,
            },
            "boundaries": [
                {"id": "left", "type": "dashed", "points": [[100, 400], [100, 100]]},
                {"id": "solid", "type": "double_solid", "points": [[300, 400], [300, 100]]},
                {"id": "right", "type": "dashed", "points": [[600, 400], [600, 100]]},
            ],
            "boundary_hold_frames": 2,
        }
        scene = build_scene(tracking(), geometry)
        offenses = analyze_scene(scene, OffenseConfig())
        selected = _select_offender(offenses)

        self.assertEqual(offenses["event_counts"], {"solid_line_crossing": 1})
        self.assertEqual(selected["status"], "SUSPECTED_OFFENDER")
        self.assertEqual(selected["track_id"], 7)
        self.assertEqual(selected["offense_type"], "solid_line_crossing")

    def test_crosswalk_pedestrian_and_moving_vehicle_create_candidate(self) -> None:
        scene = build_scene(crosswalk_tracking(), crosswalk_geometry())
        offenses = analyze_scene(scene, OffenseConfig())

        self.assertEqual(offenses["event_counts"], {"pedestrian_failure_to_yield": 1})
        self.assertEqual(offenses["events"][0]["track_id"], 3)
        self.assertNotIn(
            22,
            {
                item["track_id"]
                for record in scene["frame_records"]
                for item in record["vehicles"]
            },
        )

    def test_crosswalk_without_pedestrian_does_not_create_candidate(self) -> None:
        scene = build_scene(
            crosswalk_tracking(with_person=False), crosswalk_geometry()
        )
        self.assertEqual(analyze_scene(scene, OffenseConfig())["event_count"], 0)

    def test_vehicle_that_stopped_before_crosswalk_is_suppressed(self) -> None:
        scene = build_scene(
            crosswalk_tracking(stopped_first=True), crosswalk_geometry()
        )
        interactions = [
            item
            for record in scene["frame_records"]
            for item in record["crosswalk_interactions"]
        ]
        self.assertTrue(interactions)
        self.assertTrue(all(item["vehicle_stopped_before_crosswalk"] for item in interactions))
        self.assertEqual(analyze_scene(scene, OffenseConfig())["event_count"], 0)

    def test_low_confidence_crosswalk_is_unknown_not_an_accusation(self) -> None:
        scene = build_scene(crosswalk_tracking(), crosswalk_geometry(confidence=0.2))
        self.assertEqual(analyze_scene(scene, OffenseConfig())["event_count"], 0)

    def test_stable_red_light_crossing_creates_candidate(self) -> None:
        scene = build_scene(signal_tracking(), signal_geometry())
        offenses = analyze_scene(scene, OffenseConfig())

        self.assertEqual(offenses["event_counts"], {"red_light_running": 1})
        self.assertEqual(offenses["events"][0]["track_id"], 5)
        self.assertEqual(scene["frame_records"][0]["traffic_lights"][0]["state"], "red")
        self.assertEqual(scene["frame_records"][0]["vehicles"][0]["signal_id"], "main")

    def test_green_light_crossing_is_not_reported(self) -> None:
        scene = build_scene(signal_tracking(state="green"), signal_geometry())
        self.assertEqual(analyze_scene(scene, OffenseConfig())["event_count"], 0)

    def test_low_confidence_red_light_is_not_reported(self) -> None:
        scene = build_scene(
            signal_tracking(confidence=0.4), signal_geometry()
        )
        self.assertEqual(analyze_scene(scene, OffenseConfig())["event_count"], 0)

    def test_signal_can_use_explicit_lane_polygon_without_boundaries(self) -> None:
        scene = build_scene(signal_tracking(), polygon_signal_geometry())
        offenses = analyze_scene(scene, OffenseConfig())

        self.assertEqual(offenses["event_counts"], {"red_light_running": 1})
        self.assertEqual(scene["calibration"]["lane_indices"], [0])

    def test_opposite_direction_vehicle_is_not_associated_with_signal(self) -> None:
        scene = build_scene(signal_tracking(), polygon_signal_geometry([0, -1]))
        offenses = analyze_scene(scene, OffenseConfig())

        self.assertEqual(offenses["event_count"], 0)
        moving = [
            vehicle
            for record in scene["frame_records"][1:]
            for vehicle in record["vehicles"]
            if vehicle["track_id"] == 5
        ]
        self.assertTrue(moving)
        self.assertTrue(all(vehicle["signal_id"] is None for vehicle in moving))


if __name__ == "__main__":
    unittest.main()
