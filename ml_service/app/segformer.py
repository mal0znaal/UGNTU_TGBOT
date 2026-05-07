from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort


IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class SegFormerSegmenter:
    def __init__(self, model_path: str, input_size: int, threshold: float) -> None:
        self.input_size = input_size
        self.threshold = threshold
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def _preprocess(self, crop_rgb: np.ndarray) -> np.ndarray:
        resized = cv2.resize(
            crop_rgb,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR,
        )
        tensor = resized.astype(np.float32) / 255.0
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        tensor = tensor.transpose(2, 0, 1)[np.newaxis]
        return np.ascontiguousarray(tensor, dtype=np.float32)

    def _logits_to_mask(self, logits: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
        logits = np.asarray(logits)

        if logits.ndim == 4:
            logits_2d = logits[0, 0]
        elif logits.ndim == 3:
            logits_2d = logits[0]
        elif logits.ndim == 2:
            logits_2d = logits
        else:
            raise ValueError(f"Unexpected SegFormer output shape: {logits.shape}")

        target_h, target_w = target_hw
        if logits_2d.shape != (target_h, target_w):
            logits_2d = cv2.resize(
                logits_2d.astype(np.float32),
                (target_w, target_h),
                interpolation=cv2.INTER_LINEAR,
            )

        probabilities = sigmoid(logits_2d)
        return (probabilities >= self.threshold).astype(np.uint8) * 255

    def predict_mask(self, crop_rgb: np.ndarray) -> np.ndarray:
        tensor = self._preprocess(crop_rgb)
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        return self._logits_to_mask(outputs[0], crop_rgb.shape[:2])
