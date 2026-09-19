#!/usr/bin/env python3
"""YOLOv8 RKNN vehicle detector for a single camera frame."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from rknnlite.api import RKNNLite


INPUT_SIZE = 640
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass(frozen=True)
class Detection:
    class_id: int
    label: str
    confidence: float
    box: tuple[int, int, int, int]


def letterbox(image: np.ndarray, size: int = INPUT_SIZE) -> tuple[np.ndarray, float, int, int]:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    pad_x = (size - resized_width) // 2
    pad_y = (size - resized_height) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    return canvas, scale, pad_x, pad_y


def decode_output(
    output: np.ndarray,
    original_shape: Sequence[int],
    scale: float,
    pad_x: int,
    pad_y: int,
    confidence_threshold: float,
    iou_threshold: float,
) -> list[Detection]:
    predictions = np.asarray(output).squeeze()
    if predictions.ndim != 2:
        raise ValueError(f"Unexpected YOLO output shape: {output.shape}")
    if predictions.shape[0] == 84:
        predictions = predictions.T
    if predictions.shape[1] != 84:
        raise ValueError(f"Expected 84 values per prediction, got {predictions.shape}")

    class_scores = predictions[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]
    keep = np.array(
        [
            confidence >= confidence_threshold and int(class_id) in VEHICLE_CLASSES
            for confidence, class_id in zip(confidences, class_ids)
        ],
        dtype=bool,
    )
    if not np.any(keep):
        return []

    boxes_xywh = predictions[keep, :4]
    kept_class_ids = class_ids[keep].astype(int)
    kept_confidences = confidences[keep].astype(float)
    image_height, image_width = int(original_shape[0]), int(original_shape[1])

    boxes_xyxy: list[tuple[int, int, int, int]] = []
    for center_x, center_y, width, height in boxes_xywh:
        x1 = int(round((center_x - width / 2 - pad_x) / scale))
        y1 = int(round((center_y - height / 2 - pad_y) / scale))
        x2 = int(round((center_x + width / 2 - pad_x) / scale))
        y2 = int(round((center_y + height / 2 - pad_y) / scale))
        x1 = max(0, min(image_width - 1, x1))
        y1 = max(0, min(image_height - 1, y1))
        x2 = max(0, min(image_width - 1, x2))
        y2 = max(0, min(image_height - 1, y2))
        boxes_xyxy.append((x1, y1, x2, y2))

    detections: list[Detection] = []
    for class_id in sorted(set(kept_class_ids.tolist())):
        indices = np.flatnonzero(kept_class_ids == class_id)
        nms_boxes = []
        nms_scores = []
        for index in indices:
            x1, y1, x2, y2 = boxes_xyxy[int(index)]
            nms_boxes.append([x1, y1, max(1, x2 - x1), max(1, y2 - y1)])
            nms_scores.append(float(kept_confidences[int(index)]))

        selected = cv2.dnn.NMSBoxes(
            nms_boxes,
            nms_scores,
            confidence_threshold,
            iou_threshold,
        )
        for selected_index in np.asarray(selected).reshape(-1):
            source_index = int(indices[int(selected_index)])
            detections.append(
                Detection(
                    class_id=class_id,
                    label=VEHICLE_CLASSES[class_id],
                    confidence=float(kept_confidences[source_index]),
                    box=boxes_xyxy[source_index],
                )
            )

    return sorted(detections, key=lambda item: item.confidence, reverse=True)


class VehicleDetector:
    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.rknn = RKNNLite()

    def __enter__(self) -> "VehicleDetector":
        result = self.rknn.load_rknn(str(self.model_path))
        if result != 0:
            raise RuntimeError(f"RKNN model load failed: {result}")
        result = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        if result != 0:
            raise RuntimeError(f"RKNN runtime initialization failed: {result}")
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.rknn.release()

    def detect(self, frame: np.ndarray) -> list[Detection]:
        prepared, scale, pad_x, pad_y = letterbox(frame)
        rgb = cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB)
        inputs = np.expand_dims(rgb, axis=0)
        outputs = self.rknn.inference(inputs=[inputs])
        if len(outputs) != 1:
            raise RuntimeError(f"Expected one RKNN output, got {len(outputs)}")
        return decode_output(
            outputs[0],
            frame.shape,
            scale,
            pad_x,
            pad_y,
            self.confidence_threshold,
            self.iou_threshold,
        )


def annotate(frame: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
    annotated = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        color = (0, 220, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text = f"{detection.label} {detection.confidence:.2f}"
        cv2.putText(
            annotated,
            text,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def capture_frame(camera_index: int) -> np.ndarray:
    camera = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    camera.set(cv2.CAP_PROP_FPS, 30)
    if not camera.isOpened():
        raise RuntimeError(f"Unable to open camera index {camera_index}")
    try:
        for _ in range(5):
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera frame read failed")
        return frame
    finally:
        camera.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/home/orangepi/yolov8n.rknn"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--image", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("recordings/detection_test.jpg"),
    )
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            raise RuntimeError(f"Unable to read input image: {args.image}")
    else:
        frame = capture_frame(args.camera)
    with VehicleDetector(args.model, args.confidence, args.iou) as detector:
        detections = detector.detect(frame)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), annotate(frame, detections)):
        raise RuntimeError(f"Unable to write output image: {args.output}")

    print(f"frame={frame.shape[1]}x{frame.shape[0]}")
    print(f"vehicles={len(detections)}")
    for detection in detections:
        print(
            f"{detection.label} confidence={detection.confidence:.3f} "
            f"box={detection.box}"
        )
    print(f"output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
