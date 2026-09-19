"""Minimal compatibility container required by Supervision ByteTrack 0.18.0."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detections:
    xyxy: np.ndarray
    confidence: np.ndarray | None = None
    class_id: np.ndarray | None = None
    tracker_id: np.ndarray | None = None

    @classmethod
    def empty(cls) -> "Detections":
        return cls(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.empty(0, dtype=np.float32),
            class_id=np.empty(0, dtype=int),
            tracker_id=np.empty(0, dtype=int),
        )
