#!/usr/bin/env python3
"""Geometry helpers shared by offline detection and tracking tools."""

from __future__ import annotations

from typing import Sequence


def box_area_ratio(box: Sequence[float], width: int, height: int) -> float:
    """Return the clipped box area as a fraction of the image frame."""
    if width <= 0 or height <= 0:
        raise ValueError("frame dimensions must be positive")
    if len(box) != 4:
        raise ValueError("box must contain x1, y1, x2, y2")
    x1, y1, x2, y2 = (float(value) for value in box)
    clipped_width = max(0.0, min(float(width), x2) - max(0.0, x1))
    clipped_height = max(0.0, min(float(height), y2) - max(0.0, y1))
    return clipped_width * clipped_height / float(width * height)
