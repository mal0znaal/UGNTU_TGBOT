from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort


@dataclass(frozen=True)
class Detection:
    bbox: list[float]  # [x1, y1, x2, y2] normalised to [0, 1]
    confidence: float
    class_id: int


def preprocess_rgb(
    image_rgb: np.ndarray,
    target_h: int,
    target_w: int,
) -> tuple[np.ndarray, tuple[int, int, float, int, int]]:
    """
    Letterbox resize -> normalise -> NCHW float32 tensor.
    Returns the tensor and (orig_h, orig_w, scale, pad_x, pad_y) needed to
    map predictions back to original image coordinates.
    """
    orig_h, orig_w = image_rgb.shape[:2]

    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    tensor = canvas.astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)[np.newaxis]

    return tensor, (orig_h, orig_w, scale, pad_x, pad_y)


def postprocess(
    outputs: list[np.ndarray],
    orig_info: tuple[int, int, float, int, int],
    conf_threshold: float,
    iou_threshold: float,
) -> list[Detection]:
    """
    Supports two common YOLO output layouts:
      - YOLOv5/v7: [1, N, 85] - bbox(4) + obj_conf(1) + class_scores(80)
      - YOLOv8/v9: [1, 84, N] - bbox(4) + class_scores(80), no objectness
      - NMS export: [1, N, 6] - bbox(4) + score(1) + class_id(1)

    This logic mirrors the source preprocess/postprocess sample used for the
    exported detector. Boxes are treated as x1, y1, x2, y2 in model-space pixels.
    """
    orig_h, orig_w, scale, pad_x, pad_y = orig_info
    raw: np.ndarray = outputs[0]

    if raw.ndim != 3:
        raise ValueError(f"Unexpected output ndim={raw.ndim}, shape={raw.shape}")

    if raw.shape[2] > raw.shape[1]:
        preds = raw[0].T
    else:
        preds = raw[0]

    num_attrs = preds.shape[1]

    if num_attrs == 6:
        confidences = preds[:, 4]
        class_ids = preds[:, 5].astype(np.int64)
    elif num_attrs == 85:
        objectness = preds[:, 4]
        class_scores = preds[:, 5:]
        class_ids = class_scores.argmax(axis=1)
        confidences = objectness * class_scores[np.arange(len(preds)), class_ids]
    else:
        class_scores = preds[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        confidences = class_scores[np.arange(len(preds)), class_ids]

    keep = confidences >= conf_threshold
    if not keep.any():
        return []

    boxes_xyxy = preds[keep, :4]
    confidences = confidences[keep]
    class_ids = class_ids[keep]

    x1, y1, x2, y2 = boxes_xyxy.T

    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    x2 = (x2 - pad_x) / scale
    y2 = (y2 - pad_y) / scale

    x1 = np.clip(x1 / orig_w, 0.0, 1.0)
    y1 = np.clip(y1 / orig_h, 0.0, 1.0)
    x2 = np.clip(x2 / orig_w, 0.0, 1.0)
    y2 = np.clip(y2 / orig_h, 0.0, 1.0)

    boxes_nms = np.stack(
        [x1 * orig_w, y1 * orig_h, (x2 - x1) * orig_w, (y2 - y1) * orig_h],
        axis=1,
    )
    indices = cv2.dnn.NMSBoxes(
        boxes_nms.tolist(),
        confidences.tolist(),
        float(conf_threshold),
        float(iou_threshold),
    )
    flat = np.array(indices).reshape(-1) if len(indices) else []

    return [
        Detection(
            bbox=[float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
            confidence=float(confidences[i]),
            class_id=int(class_ids[i]),
        )
        for i in flat
    ]


def select_largest_detection(detections: list[Detection]) -> Detection | None:
    if not detections:
        return None

    def area(det: Detection) -> float:
        x1, y1, x2, y2 = det.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    return max(detections, key=area)


class YoloDetector:
    def __init__(
        self,
        model_path: str,
        input_size: int,
        conf_threshold: float,
        iou_threshold: float,
    ) -> None:
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        tensor, orig_info = preprocess_rgb(image_rgb, self.input_size, self.input_size)
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        return postprocess(outputs, orig_info, self.conf_threshold, self.iou_threshold)
