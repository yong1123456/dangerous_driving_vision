#!/usr/bin/env python3
"""Temporal class-label smoothing for a stable tracked identity."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass


VEHICLE_LABELS = {"car", "motorcycle", "bus", "truck", "vehicle"}


def _family(label: str) -> str:
    return "vehicle" if label in VEHICLE_LABELS else label


@dataclass(frozen=True)
class StableLabel:
    class_id: int
    label: str
    observations: int


class TrackLabelSmoother:
    """Choose a confidence-weighted label from a recent per-track window."""

    def __init__(self, window_size: int = 15, minimum_observations: int = 3) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least one")
        if not 1 <= minimum_observations <= window_size:
            raise ValueError("minimum_observations must be in [1, window_size]")
        self.window_size = window_size
        self.minimum_observations = minimum_observations
        self._history: dict[int, deque[tuple[int, str, float]]] = {}

    def observe(
        self, track_id: int, class_id: int, label: str, confidence: float
    ) -> StableLabel:
        track_id = int(track_id)
        class_id = int(class_id)
        label = str(label)
        confidence = max(0.0, float(confidence))
        history = self._history.get(track_id)
        if history is None or (history and _family(history[-1][1]) != _family(label)):
            history = deque(maxlen=self.window_size)
            self._history[track_id] = history
        history.append((class_id, label, confidence))
        if len(history) < self.minimum_observations:
            return StableLabel(class_id, label, len(history))

        scores: dict[tuple[int, str], float] = defaultdict(float)
        counts: dict[tuple[int, str], int] = defaultdict(int)
        for observed_class, observed_label, observed_confidence in history:
            key = (observed_class, observed_label)
            scores[key] += observed_confidence
            counts[key] += 1
        current_key = (class_id, label)
        best = max(
            scores,
            key=lambda key: (
                scores[key],
                counts[key],
                key == current_key,
            ),
        )
        return StableLabel(best[0], best[1], len(history))
