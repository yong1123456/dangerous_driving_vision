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


if __name__ == "__main__":
    unittest.main()
