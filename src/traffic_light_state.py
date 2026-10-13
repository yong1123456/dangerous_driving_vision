#!/usr/bin/env python3
"""Classify the illuminated color inside a detected traffic-light box."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class TrafficLightState:
    state: str
    confidence: float
    color_ratios: dict[str, float]


def _largest_component(mask: np.ndarray) -> int:
    components, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if components <= 1:
        return 0
    return int(stats[1:, cv2.CC_STAT_AREA].max())


def classify_traffic_light_state(
    frame: np.ndarray,
    box: Sequence[float],
    *,
    minimum_colored_ratio: float = 0.01,
    minimum_dominance: float = 0.55,
) -> TrafficLightState:
    """Return a conservative red/yellow/green estimate for one detection."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a BGR image")
    if len(box) != 4:
        raise ValueError("box must contain x1, y1, x2, y2")
    if not 0 < minimum_colored_ratio <= 1:
        raise ValueError("minimum_colored_ratio must be in the range (0, 1]")
    if not 0 < minimum_dominance <= 1:
        raise ValueError("minimum_dominance must be in the range (0, 1]")

    height, width = frame.shape[:2]
    x1 = max(0, min(width, int(np.floor(float(box[0])))))
    y1 = max(0, min(height, int(np.floor(float(box[1])))))
    x2 = max(0, min(width, int(np.ceil(float(box[2])))))
    y2 = max(0, min(height, int(np.ceil(float(box[3])))))
    if x2 <= x1 or y2 <= y1:
        return TrafficLightState("unknown", 0.0, {})

    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    ranges = {
        "red": (
            (np.array([0, 90, 90]), np.array([12, 255, 255])),
            (np.array([165, 90, 90]), np.array([180, 255, 255])),
        ),
        "yellow": (
            (np.array([15, 90, 100]), np.array([38, 255, 255])),
        ),
        "green": (
            (np.array([39, 60, 60]), np.array([95, 255, 255])),
        ),
    }
    component_pixels: dict[str, int] = {}
    for state, state_ranges in ranges.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in state_ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
        component_pixels[state] = _largest_component(mask)

    area = max(1, crop.shape[0] * crop.shape[1])
    ratios = {
        state: round(pixels / area, 6)
        for state, pixels in component_pixels.items()
    }
    winner = max(component_pixels, key=component_pixels.get)
    winner_pixels = component_pixels[winner]
    total_pixels = sum(component_pixels.values())
    required_pixels = max(2, int(np.ceil(area * minimum_colored_ratio)))
    if winner_pixels < required_pixels or total_pixels == 0:
        return TrafficLightState("unknown", 0.0, ratios)

    dominance = winner_pixels / total_pixels
    if dominance < minimum_dominance:
        return TrafficLightState("unknown", round(dominance, 3), ratios)
    coverage = min(1.0, winner_pixels / required_pixels)
    confidence = round(dominance * coverage, 3)
    return TrafficLightState(winner, confidence, ratios)
