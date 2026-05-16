from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import onnxruntime as ort


@dataclass(frozen=True)
class Detection:
    bbox: list[float]  # [x1, y1, x2, y2], нормализовано к [0, 1]
    confidence: float
    class_id: int


@dataclass(frozen=True)
class DetectorTiming:
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float


@dataclass(frozen=True)
class DetectorResult:
    detections: list[Detection]
    timing: DetectorTiming


def decode_image_to_rgb(image_bytes: bytes) -> np.ndarray:
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Cannot decode image")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)[:, :, :3]
    raise ValueError(f"Unsupported channel count: {image.shape[2]}")


def read_image_rgb(path: str | Path) -> np.ndarray:
    return decode_image_to_rgb(Path(path).read_bytes())


def preprocess_rgb(
    image_rgb: np.ndarray,
    target_h: int,
    target_w: int,
) -> tuple[np.ndarray, tuple[int, int, float, int, int]]:
    """Letterbox resize -> normalize /255 -> NCHW float32 tensor."""
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
    return np.ascontiguousarray(tensor), (orig_h, orig_w, scale, pad_x, pad_y)


def postprocess(
    outputs: list[np.ndarray],
    orig_info: tuple[int, int, float, int, int],
    conf_threshold: float,
    iou_threshold: float,
    use_nms: bool = False,
) -> list[Detection]:
    """
    Поддерживает распространенные форматы выхода YOLO ONNX:
    - [1, N, 85]: xyxy + objectness + class scores
    - [1, 84, N]: xyxy + class scores
    - [1, N, 6]: xyxy + score + class_id

    В этом проекте bbox считаются xyxy-координатами в letterbox-пикселях.
    Флаг use_nms нужен для экспериментов: NMS или все bbox после threshold.
    """
    orig_h, orig_w, scale, pad_x, pad_y = orig_info
    raw: np.ndarray = outputs[0]
    if raw.ndim != 3:
        raise ValueError(f"Unexpected output ndim={raw.ndim}, shape={raw.shape}")

    preds = raw[0].T if raw.shape[2] > raw.shape[1] else raw[0]
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

    selected_indices: list[int] | np.ndarray
    if use_nms:
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
        selected_indices = np.array(indices).reshape(-1) if len(indices) else []
    else:
        selected_indices = range(len(confidences))

    return [
        Detection(
            bbox=[float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
            confidence=float(confidences[i]),
            class_id=int(class_ids[i]),
        )
        for i in selected_indices
    ]


class YoloOnnxDetector:
    def __init__(
        self,
        model_path: str | Path,
        input_size: int,
        conf_threshold: float,
        iou_threshold: float,
        use_nms: bool = False,
        providers: list[str] | None = None,
    ) -> None:
        self.model_path = str(model_path)
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.use_nms = use_nms
        self.session = ort.InferenceSession(
            self.model_path,
            providers=providers or ["CPUExecutionProvider"],
        )
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
        detections = postprocess(
            outputs,
            orig_info,
            self.conf_threshold,
            self.iou_threshold,
            self.use_nms,
        )
        postprocess_ms = (perf_counter() - postprocess_start) * 1000

        return DetectorResult(
            detections=detections,
            timing=DetectorTiming(
                preprocess_ms=preprocess_ms,
                inference_ms=inference_ms,
                postprocess_ms=postprocess_ms,
            ),
        )
