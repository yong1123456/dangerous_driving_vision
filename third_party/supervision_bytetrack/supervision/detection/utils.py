"""Minimal IoU helper required by Supervision ByteTrack 0.18.0."""

from __future__ import annotations

import numpy as np


def box_iou_batch(boxes_true: np.ndarray, boxes_detection: np.ndarray) -> np.ndarray:
    def box_area(boxes: np.ndarray) -> np.ndarray:
        return (boxes[2] - boxes[0]) * (boxes[3] - boxes[1])

    area_true = box_area(boxes_true.T)
    area_detection = box_area(boxes_detection.T)
    top_left = np.maximum(boxes_true[:, None, :2], boxes_detection[:, :2])
    bottom_right = np.minimum(boxes_true[:, None, 2:], boxes_detection[:, 2:])
    area_intersection = np.prod(
        np.clip(bottom_right - top_left, a_min=0, a_max=None),
        axis=2,
    )
    return area_intersection / (
        area_true[:, None] + area_detection - area_intersection
    )
