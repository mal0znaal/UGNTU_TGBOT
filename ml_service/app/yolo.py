from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np
import onnxruntime as ort

from app.timing import PhaseTiming


@dataclass(frozen=True)
class Detection:
    bbox: list[float]  # [x1, y1, x2, y2] normalised to [0, 1]
    confidence: float
    class_id: int


@dataclass(frozen=True)
class DetectorResult:
    detections: list[Detection]
    timing: PhaseTiming


def preprocess_rgb(
    image_rgb: np.ndarray,
    target_h: int,
    target_w: int,
) -> tuple[np.ndarray, tuple[int, int, float, int, int]]:
    """Подготавливает изображение и данные для восстановления координат."""
    orig_h, orig_w = image_rgb.shape[:2]

    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = target_w - new_w
    pad_h = target_h - new_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    canvas = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    tensor = canvas.astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)[np.newaxis]

    return tensor, (orig_h, orig_w, scale, pad_left, pad_top)


def postprocess(
    outputs: list[np.ndarray],
    orig_info: tuple[int, int, float, int, int],
    conf_threshold: float,
) -> list[Detection]:
    """Преобразует выход NMS-модели в координаты исходного изображения."""
    orig_h, orig_w, scale, pad_x, pad_y = orig_info
    preds = outputs[0][0]

    if preds.ndim != 2 or preds.shape[1] != 6:
        raise ValueError(f"Expected NMS output shape [N, 6], got {preds.shape}")

    confidences = preds[:, 4]
    class_ids = preds[:, 5].astype(np.int64)

    keep = confidences >= conf_threshold
    if not keep.any():
        return []

    boxes_xyxy = preds[keep, :4]
    confidences = confidences[keep]
    class_ids = class_ids[keep]

    boxes_xyxy[:, 0] = np.clip((boxes_xyxy[:, 0] - pad_x) / scale / orig_w, 0.0, 1.0)
    boxes_xyxy[:, 1] = np.clip((boxes_xyxy[:, 1] - pad_y) / scale / orig_h, 0.0, 1.0)
    boxes_xyxy[:, 2] = np.clip((boxes_xyxy[:, 2] - pad_x) / scale / orig_w, 0.0, 1.0)
    boxes_xyxy[:, 3] = np.clip((boxes_xyxy[:, 3] - pad_y) / scale / orig_h, 0.0, 1.0)

    return [
        Detection(bbox=box.tolist(), confidence=float(conf), class_id=int(cls_id))
        for box, conf, cls_id in zip(boxes_xyxy, confidences, class_ids)
    ]



class YoloDetector:
    def __init__(
        self,
        model_path: str,
        input_size: int,
        conf_threshold: float,
    ) -> None:
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def detect(self, image_rgb: np.ndarray) -> DetectorResult:
        preprocess_start = perf_counter()
        tensor, orig_info = preprocess_rgb(image_rgb, self.input_size, self.input_size)
        preprocess_ms = (perf_counter() - preprocess_start) * 1000

        inference_start = perf_counter()
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        inference_ms = (perf_counter() - inference_start) * 1000

        postprocess_start = perf_counter()
        detections = postprocess(outputs, orig_info, self.conf_threshold)
        postprocess_ms = (perf_counter() - postprocess_start) * 1000

        return DetectorResult(
            detections=detections,
            timing=PhaseTiming(
                preprocess_ms=preprocess_ms,
                inference_ms=inference_ms,
                postprocess_ms=postprocess_ms,
            ),
        )
