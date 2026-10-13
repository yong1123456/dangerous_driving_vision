#!/usr/bin/env python3
"""Create a labelled contact sheet from evenly spaced video frames."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def parse_crop(value: str | None) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    try:
        crop = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise ValueError("crop must use X,Y,WIDTH,HEIGHT format") from error
    if len(crop) != 4 or any(part < 0 for part in crop[:2]) or any(part <= 0 for part in crop[2:]):
        raise ValueError("crop must use non-negative X/Y and positive WIDTH/HEIGHT")
    return crop


def create_contact_sheet(
    video_path: Path,
    output_path: Path,
    *,
    samples: int = 12,
    columns: int = 4,
    tile_width: int = 512,
    crop: Sequence[int] | None = None,
) -> None:
    if samples < 1 or columns < 1 or tile_width < 64:
        raise ValueError("samples/columns must be positive and tile width must be at least 64")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_count < 1 or fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Invalid video metadata: {video_path}")
    if crop is not None:
        if len(crop) != 4:
            capture.release()
            raise ValueError("crop must contain X,Y,WIDTH,HEIGHT")
        crop_x, crop_y, crop_width, crop_height = crop
        if crop_x + crop_width > width or crop_y + crop_height > height:
            capture.release()
            raise ValueError("crop extends beyond the video frame")
    else:
        crop_x, crop_y, crop_width, crop_height = 0, 0, width, height

    frame_indices = np.linspace(0, frame_count - 1, samples, dtype=int)
    tile_height = max(1, round(tile_width * crop_height / crop_width))
    tiles: list[np.ndarray] = []
    try:
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Unable to read frame {frame_index + 1}")
            frame = frame[
                crop_y : crop_y + crop_height,
                crop_x : crop_x + crop_width,
            ]
            tile = cv2.resize(frame, (tile_width, tile_height))
            label = f"frame {frame_index + 1} / {frame_index / fps:.2f}s"
            cv2.rectangle(tile, (0, 0), (tile_width, 36), (0, 0, 0), -1)
            cv2.putText(
                tile,
                label,
                (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            tiles.append(tile)
    finally:
        capture.release()

    rows = math.ceil(samples / columns)
    blank = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
    tiles.extend(blank.copy() for _ in range(rows * columns - len(tiles)))
    sheet = np.vstack(
        [np.hstack(tiles[row * columns : (row + 1) * columns]) for row in range(rows)]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"Unable to write image: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--tile-width", type=int, default=512)
    parser.add_argument("--crop", help="Optional X,Y,WIDTH,HEIGHT crop")
    args = parser.parse_args()
    create_contact_sheet(
        args.video,
        args.output,
        samples=args.samples,
        columns=args.columns,
        tile_width=args.tile_width,
        crop=parse_crop(args.crop),
    )
    print(f"output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
