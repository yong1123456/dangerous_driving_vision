import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from road_geometry import (  # noqa: E402
    Boundary,
    boundary_side,
    lane_at_point,
    parse_boundaries,
    parse_crosswalks,
    parse_lanes,
    parse_traffic_signals,
    point_in_polygon,
    stop_line_position,
)


class RoadGeometryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.boundaries = parse_boundaries(
            [
                {"id": "left", "type": "dashed", "points": [[100, 400], [300, 100]]},
                {"id": "solid", "type": "double_solid", "points": [[300, 400], [400, 100]]},
                {"id": "right", "type": "dashed", "points": [[600, 400], [500, 100]]},
            ],
            width=800,
            height=400,
        )

    def test_lane_index_uses_ordered_intervals(self) -> None:
        self.assertEqual(lane_at_point((220, 400), self.boundaries)[0], 0)
        self.assertEqual(lane_at_point((450, 400), self.boundaries)[0], 1)
        self.assertIsNone(lane_at_point((50, 400), self.boundaries)[0])

    def test_boundary_side_changes_across_solid_line(self) -> None:
        solid = self.boundaries[1]
        self.assertLess(boundary_side((250, 400), solid), 0)
        self.assertGreater(boundary_side((350, 400), solid), 0)

    def test_explicit_lane_polygon_supports_arbitrary_viewpoint(self) -> None:
        lanes = parse_lanes(
            [
                {
                    "index": 7,
                    "confidence": 0.92,
                    "direction": [1, -0.2],
                    "minimum_direction_alignment": 0.5,
                    "polygon": [[0.1, 0.8], [0.8, 0.2], [0.9, 0.3], [0.2, 0.9]],
                }
            ],
            width=800,
            height=400,
            normalized=True,
        )

        self.assertEqual(lane_at_point((400, 220), [], lanes=lanes), (7, 0.92))
        self.assertEqual(lane_at_point((50, 350), [], lanes=lanes), (None, 0.0))
        self.assertEqual(lanes[0].direction, (1.0, -0.2))

    def test_overlapping_lane_polygons_are_unknown(self) -> None:
        lanes = parse_lanes(
            [
                {"index": 0, "polygon": [[0, 0], [500, 0], [500, 400], [0, 400]]},
                {"index": 1, "polygon": [[300, 0], [800, 0], [800, 400], [300, 400]]},
            ],
            width=800,
            height=400,
        )

        self.assertEqual(lane_at_point((400, 200), self.boundaries, lanes=lanes), (None, 0.0))

    def test_crosswalk_polygons_are_normalized_and_include_edges(self) -> None:
        crosswalk = parse_crosswalks(
            [
                {
                    "id": "zebra_a",
                    "pedestrian_zone": [[0.25, 0.5], [0.5, 0.5], [0.5, 1.0], [0.25, 1.0]],
                    "vehicle_conflict_zone": [[0.5, 0.5], [0.75, 0.5], [0.75, 1.0], [0.5, 1.0]],
                }
            ],
            width=800,
            height=400,
            normalized=True,
        )[0]

        self.assertTrue(point_in_polygon((200, 200), crosswalk.pedestrian_zone))
        self.assertTrue(point_in_polygon((300, 300), crosswalk.pedestrian_zone))
        self.assertFalse(point_in_polygon((100, 300), crosswalk.pedestrian_zone))

    def test_normalized_signal_has_oriented_stop_line(self) -> None:
        signal = parse_traffic_signals(
            [
                {
                    "id": "main",
                    "roi": [[0.4, 0.1], [0.6, 0.1], [0.6, 0.3], [0.4, 0.3]],
                    "applies_to_lanes": [0, 1],
                    "stop_line": {
                        "points": [[0.1, 0.5], [0.9, 0.5]],
                        "before_side": "negative",
                        "margin_pixels": 2,
                    },
                }
            ],
            width=800,
            height=400,
            normalized=True,
        )[0]

        self.assertEqual(stop_line_position((400, 190), signal), "before")
        self.assertEqual(stop_line_position((400, 210), signal), "after")
        self.assertEqual(stop_line_position((400, 201), signal), "unknown")


if __name__ == "__main__":
    unittest.main()
