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
    point_in_polygon,
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


if __name__ == "__main__":
    unittest.main()
