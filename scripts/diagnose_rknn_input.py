#!/usr/bin/env python3
"""Compare RKNN input layouts, color orders, and numeric scaling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vehicle_detector import letterbox  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        default=PROJECT_ROOT / "recordings" / "detection_test.jpg",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/home/orangepi/yolov8n.rknn"),
    )
    return parser.parse_args()


def prepare_variants(frame: np.ndarray) -> list[tuple[str, np.ndarray, str]]:
    prepared, _, _, _ = letterbox(frame)
    rgb = cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB)
    variants: list[tuple[str, np.ndarray, str]] = []
    for color_name, image in (("rgb", rgb), ("bgr", prepared)):
        uint8_nhwc = np.expand_dims(image, axis=0)
        float_nhwc = np.expand_dims(image.astype(np.float32) / 255.0, axis=0)
        uint8_nchw = np.expand_dims(np.transpose(image, (2, 0, 1)), axis=0)
        float_nchw = np.expand_dims(
            np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1)),
            axis=0,
        )
        variants.extend(
            [
                (f"{color_name}_uint8_nhwc", uint8_nhwc, "nhwc"),
                (f"{color_name}_float_nhwc", float_nhwc, "nhwc"),
                (f"{color_name}_uint8_nchw", uint8_nchw, "nchw"),
                (f"{color_name}_float_nchw", float_nchw, "nchw"),
            ]
        )
    return variants


def summarize(output: np.ndarray) -> str:
    predictions = np.asarray(output).squeeze()
    if predictions.shape[0] == 84:
        predictions = predictions.T
    if predictions.ndim != 2 or predictions.shape[1] != 84:
        return f"unexpected_output={output.shape}"

    class_scores = predictions[:, 4:]
    car_scores = class_scores[:, 2]
    car_index = int(np.argmax(car_scores))
    flat_index = int(np.argmax(class_scores))
    candidate_index, class_id = np.unravel_index(flat_index, class_scores.shape)
    box = predictions[car_index, :4]
    return (
        f"car_max={float(car_scores[car_index]):.6f} "
        f"best_class={int(class_id)} "
        f"best_score={float(class_scores[candidate_index, class_id]):.6f} "
        f"car_box={np.round(box, 2).tolist()}"
    )


def main() -> int:
    args = parse_args()
    frame = cv2.imread(str(args.image))
    if frame is None:
        raise RuntimeError(f"Unable to read image: {args.image}")

    rknn = RKNNLite()
    result = rknn.load_rknn(str(args.model))
    if result != 0:
        raise RuntimeError(f"RKNN model load failed: {result}")
    result = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    if result != 0:
        raise RuntimeError(f"RKNN runtime initialization failed: {result}")

    try:
        print(f"image={args.image} shape={frame.shape}")
        for name, input_data, data_format in prepare_variants(frame):
            try:
                outputs = rknn.inference(
                    inputs=[input_data],
                    data_format=[data_format],
                )
                print(
                    f"{name} input={input_data.shape}/{input_data.dtype} "
                    f"{summarize(outputs[0])}"
                )
            except Exception as error:
                print(f"{name} error={type(error).__name__}: {error}")
    finally:
        rknn.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
