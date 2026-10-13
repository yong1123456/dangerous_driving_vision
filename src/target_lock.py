#!/usr/bin/env python3
"""Stable target selection and re-association for tracked vehicles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class TargetLockConfig:
    """Conservative settings for turning a track into a locked target."""

    confirm_frames: int = 3
    max_missed_frames: int = 10
    min_iou: float = 0.10
    max_center_distance_ratio: float = 0.20
    min_match_score: float = 0.35
    min_match_margin: float = 0.10
    max_area_change_ratio: float = 2.25
    reassociation_confirm_frames: int = 2
    max_reassociation_gap_frames: int = 3

    def validate(self) -> None:
        if self.confirm_frames < 1:
            raise ValueError("confirm_frames must be at least 1")
        if self.max_missed_frames < 0:
            raise ValueError("max_missed_frames must be non-negative")
        if not 0 <= self.min_iou <= 1:
            raise ValueError("min_iou must be in the range [0, 1]")
        if self.max_center_distance_ratio <= 0:
            raise ValueError("max_center_distance_ratio must be positive")
        if not 0 <= self.min_match_score <= 1:
            raise ValueError("min_match_score must be in the range [0, 1]")
        if not 0 <= self.min_match_margin <= 1:
            raise ValueError("min_match_margin must be in the range [0, 1]")
        if self.max_area_change_ratio < 1:
            raise ValueError("max_area_change_ratio must be at least 1")
        if self.reassociation_confirm_frames < 1:
            raise ValueError("reassociation_confirm_frames must be at least 1")
        if self.max_reassociation_gap_frames < 0:
            raise ValueError("max_reassociation_gap_frames must be non-negative")


@dataclass(frozen=True)
class TargetLockState:
    state: str
    target_track_id: int | None
    box: tuple[int, int, int, int] | None
    missed_frames: int
    confirmation_frames: int
    match_method: str | None

    @property
    def is_locked(self) -> bool:
        return self.state == "locked"

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "target_track_id": self.target_track_id,
            "box": list(self.box) if self.box is not None else None,
            "missed_frames": self.missed_frames,
            "confirmation_frames": self.confirmation_frames,
            "match_method": self.match_method,
        }


def _area(box: Sequence[int]) -> float:
    return float(max(0, box[2] - box[0]) * max(0, box[3] - box[1]))


def _iou(first: Sequence[int], second: Sequence[int]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = _area((x1, y1, x2, y2))
    union = _area(first) + _area(second) - intersection
    return intersection / union if union > 0 else 0.0


def _center(box: Sequence[int]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


class TargetLock:
    """Lock one tracked vehicle and conservatively recover its ID after gaps."""

    def __init__(
        self,
        frame_shape: Sequence[int],
        *,
        target_track_id: int | None = None,
        target_point: tuple[float, float] | None = None,
        config: TargetLockConfig | None = None,
    ) -> None:
        self.frame_height = int(frame_shape[0])
        self.frame_width = int(frame_shape[1])
        if self.frame_width < 1 or self.frame_height < 1:
            raise ValueError("frame dimensions must be positive")
        if target_track_id is None and target_point is None:
            raise ValueError("target_track_id or target_point is required")
        self.requested_track_id = target_track_id
        self.target_point = target_point
        self.config = config or TargetLockConfig()
        self.config.validate()
        self._state = "waiting"
        self._locked_id: int | None = None
        self._last_box: tuple[int, int, int, int] | None = None
        self._missed = 0
        self._confirmation = 0
        self._last_method: str | None = None
        self._last_label: str | None = None
        self._pending_reassociation_id: int | None = None
        self._pending_reassociation_frames = 0

    def _point_candidate(self, tracks: Sequence[Any]) -> Any | None:
        if self.target_point is None:
            return None
        px, py = self.target_point
        inside = [
            track
            for track in tracks
            if track.box[0] <= px <= track.box[2]
            and track.box[1] <= py <= track.box[3]
        ]
        if inside:
            inside.sort(key=lambda track: _area(track.box), reverse=True)
            if len(inside) > 1 and _area(inside[1].box) >= 0.85 * _area(inside[0].box):
                return None
            return inside[0]
        max_distance = self.config.max_center_distance_ratio * min(
            self.frame_width, self.frame_height
        )
        candidates = []
        for track in tracks:
            cx, cy = _center(track.box)
            distance = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            if distance <= max_distance:
                candidates.append((distance, track))
        candidates.sort(key=lambda item: item[0])
        if len(candidates) > 1 and candidates[1][0] - candidates[0][0] < max_distance * 0.05:
            return None
        return candidates[0][1] if candidates else None

    def _find_candidate(self, tracks: Sequence[Any]) -> tuple[Any | None, str | None]:
        # A terminal loss must be explicit. Reusing a point selector after the
        # target has disappeared could silently lock a different vehicle.
        if self._state == "lost":
            return None, None
        if self._locked_id is None:
            if self.requested_track_id is not None:
                for track in tracks:
                    if int(track.track_id) == self.requested_track_id:
                        return track, "requested_track_id"
                return None, None
            track = self._point_candidate(tracks)
            return (track, "target_point") if track is not None else (None, None)

        same_id = [
            track
            for track in tracks
            if int(track.track_id) > 0 and int(track.track_id) == self._locked_id
        ]
        if same_id:
            return same_id[0], "same_track_id"
        if self._last_box is None:
            return None, None
        if self._missed > self.config.max_reassociation_gap_frames:
            return None, None

        candidates = []
        max_distance = self.config.max_center_distance_ratio * min(
            self.frame_width, self.frame_height
        )
        old_cx, old_cy = _center(self._last_box)
        for track in tracks:
            if int(track.track_id) <= 0:
                continue
            if getattr(track, "label", None) != getattr(self, "_last_label", None):
                continue
            iou = _iou(self._last_box, track.box)
            cx, cy = _center(track.box)
            distance = ((cx - old_cx) ** 2 + (cy - old_cy) ** 2) ** 0.5
            distance_score = max(0.0, 1.0 - distance / max_distance)
            score = max(iou, distance_score)
            old_area = _area(self._last_box)
            new_area = _area(track.box)
            if old_area <= 0 or new_area <= 0:
                continue
            area_ratio = max(new_area / old_area, old_area / new_area)
            if (iou >= self.config.min_iou or distance <= max_distance) and score >= self.config.min_match_score:
                if area_ratio > self.config.max_area_change_ratio:
                    continue
                candidates.append((score, iou, distance, track))
        if not candidates:
            return None, None
        candidates.sort(key=lambda item: (item[0], item[1], -item[2]), reverse=True)
        best = candidates[0]
        if len(candidates) > 1 and best[0] - candidates[1][0] < self.config.min_match_margin:
            return None, None
        return best[3], "reassociated"

    def update(self, tracks: Sequence[Any]) -> TargetLockState:
        tracks = [track for track in tracks if int(getattr(track, "track_id", -1)) > 0]
        track, method = self._find_candidate(tracks)
        if track is None:
            self._last_method = "not_found"
            self._pending_reassociation_id = None
            self._pending_reassociation_frames = 0
            if self._locked_id is not None:
                self._missed += 1
                if self._missed > self.config.max_missed_frames:
                    self._state = "lost"
                    self._locked_id = None
                    self._last_box = None
                    self._confirmation = 0
                elif self._state in {"candidate", "locked"}:
                    self._state = "temporarily_lost"
            return self.state

        # A new ByteTrack ID is only accepted after repeated, unambiguous
        # observations. Until then keep the old ID and report a temporary loss.
        if (
            self._locked_id is not None
            and method == "reassociated"
            and int(track.track_id) != self._locked_id
        ):
            candidate_id = int(track.track_id)
            if candidate_id == self._pending_reassociation_id:
                self._pending_reassociation_frames += 1
            else:
                self._pending_reassociation_id = candidate_id
                self._pending_reassociation_frames = 1
            if self._pending_reassociation_frames < self.config.reassociation_confirm_frames:
                self._last_method = "reassociation_pending"
                self._missed += 1
                if self._missed > self.config.max_missed_frames:
                    self._state = "lost"
                    self._locked_id = None
                    self._last_box = None
                    self._confirmation = 0
                    self._pending_reassociation_id = None
                    self._pending_reassociation_frames = 0
                else:
                    self._state = "temporarily_lost"
                return self.state
            self._pending_reassociation_id = None
            self._pending_reassociation_frames = 0

        self._missed = 0
        self._last_method = method
        if method == "same_track_id":
            self._pending_reassociation_id = None
            self._pending_reassociation_frames = 0
        self._last_box = tuple(int(round(value)) for value in track.box)
        self._last_label = getattr(track, "label", "vehicle")
        if self._locked_id is None:
            self._locked_id = int(track.track_id)
            self._confirmation += 1
            self._state = (
                "locked" if self._confirmation >= self.config.confirm_frames else "candidate"
            )
        else:
            self._locked_id = int(track.track_id)
            if self._state == "lost":
                self._confirmation = 1
                self._state = "candidate"
            elif self._state == "temporarily_lost":
                self._state = "locked" if self._confirmation >= self.config.confirm_frames else "candidate"
            elif self._state == "candidate":
                self._confirmation += 1
                if self._confirmation >= self.config.confirm_frames:
                    self._state = "locked"
        return self.state

    @property
    def state(self) -> TargetLockState:
        return TargetLockState(
            state=self._state,
            target_track_id=self._locked_id,
            box=self._last_box,
            missed_frames=self._missed,
            confirmation_frames=self._confirmation,
            match_method=self._last_method,
        )
