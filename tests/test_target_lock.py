import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from target_lock import TargetLock, TargetLockConfig


@dataclass
class FakeTrack:
    track_id: int
    label: str
    box: tuple[int, int, int, int]


class TargetLockTest(unittest.TestCase):
    def test_point_selection_requires_confirmation(self) -> None:
        lock = TargetLock(
            (1000, 1600, 3),
            target_point=(120, 120),
            config=TargetLockConfig(confirm_frames=2),
        )
        first = lock.update([FakeTrack(4, "car", (80, 80, 180, 180))])
        second = lock.update([FakeTrack(4, "car", (84, 82, 184, 182))])
        self.assertEqual(first.state, "candidate")
        self.assertEqual(second.state, "locked")
        self.assertEqual(second.target_track_id, 4)

    def test_reassociates_new_track_id_after_short_gap(self) -> None:
        lock = TargetLock(
            (1000, 1600, 3),
            target_track_id=4,
            config=TargetLockConfig(
                confirm_frames=1,
                max_missed_frames=2,
                reassociation_confirm_frames=1,
            ),
        )
        self.assertEqual(
            lock.update([FakeTrack(4, "car", (100, 100, 220, 240))]).state,
            "locked",
        )
        missed = lock.update([])
        self.assertEqual(missed.state, "temporarily_lost")
        recovered = lock.update([FakeTrack(9, "car", (106, 104, 226, 244))])
        self.assertEqual(recovered.state, "locked")
        self.assertEqual(recovered.target_track_id, 9)
        self.assertEqual(recovered.match_method, "reassociated")

    def test_ambiguous_point_does_not_choose_between_overlapping_vehicles(self) -> None:
        lock = TargetLock(
            (1000, 1600, 3),
            target_point=(150, 150),
            config=TargetLockConfig(confirm_frames=1),
        )
        state = lock.update(
            [
                FakeTrack(4, "car", (80, 80, 180, 180)),
                FakeTrack(5, "car", (85, 85, 185, 185)),
            ]
        )
        self.assertEqual(state.state, "waiting")
        self.assertIsNone(state.target_track_id)

    def test_ambiguous_reassociation_does_not_switch(self) -> None:
        lock = TargetLock(
            (1000, 1600, 3),
            target_track_id=4,
            config=TargetLockConfig(confirm_frames=1),
        )
        lock.update([FakeTrack(4, "car", (100, 100, 220, 240))])
        state = lock.update(
            [
                FakeTrack(8, "car", (100, 100, 220, 240)),
                FakeTrack(9, "car", (100, 100, 220, 240)),
            ]
        )
        self.assertEqual(state.target_track_id, 4)
        self.assertEqual(state.state, "temporarily_lost")
        self.assertEqual(state.missed_frames, 1)

    def test_reassociation_requires_two_stable_frames_by_default(self) -> None:
        lock = TargetLock(
            (1000, 1600, 3),
            target_track_id=4,
            config=TargetLockConfig(confirm_frames=1, max_missed_frames=3),
        )
        lock.update([FakeTrack(4, "car", (100, 100, 220, 240))])
        first = lock.update([FakeTrack(9, "car", (106, 104, 226, 244))])
        self.assertEqual(first.state, "temporarily_lost")
        self.assertEqual(first.target_track_id, 4)
        second = lock.update([FakeTrack(9, "car", (110, 108, 230, 248))])
        self.assertEqual(second.state, "locked")
        self.assertEqual(second.target_track_id, 9)

    def test_long_gap_cannot_switch_to_a_similar_vehicle(self) -> None:
        lock = TargetLock(
            (1000, 1600, 3),
            target_track_id=4,
            config=TargetLockConfig(
                confirm_frames=1,
                max_missed_frames=4,
                max_reassociation_gap_frames=1,
            ),
        )
        lock.update([FakeTrack(4, "car", (100, 100, 220, 240))])
        lock.update([])
        lock.update([])
        state = lock.update([FakeTrack(9, "car", (106, 104, 226, 244))])
        self.assertEqual(state.state, "temporarily_lost")
        self.assertEqual(state.target_track_id, 4)
        self.assertEqual(state.match_method, "not_found")

    def test_lost_target_is_not_replaced_by_point_candidate(self) -> None:
        lock = TargetLock(
            (1000, 1600, 3),
            target_point=(150, 150),
            config=TargetLockConfig(confirm_frames=1, max_missed_frames=1),
        )
        self.assertEqual(
            lock.update([FakeTrack(4, "car", (80, 80, 180, 180))]).state,
            "locked",
        )
        self.assertEqual(lock.update([]).state, "temporarily_lost")
        lost = lock.update([])
        self.assertEqual(lost.state, "lost")
        replacement = lock.update([FakeTrack(9, "car", (80, 80, 180, 180))])
        self.assertEqual(replacement.state, "lost")
        self.assertIsNone(replacement.target_track_id)


if __name__ == "__main__":
    unittest.main()
