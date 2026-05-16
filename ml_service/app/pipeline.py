from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from app.config import Settings
from app.image_utils import (
    decode_image_to_rgb,
    encode_rgb_png,
    save_inference_images,
)
from app.yolo import Detection, YoloDetector


logger = logging.getLogger("uvicorn.error")


COCO_CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


class NoDetectionError(RuntimeError):
    pass


class InferenceError(RuntimeError):
    pass


@dataclass
class DetectionPipeline:
    garment_detector: YoloDetector
    bad_classes_detector: YoloDetector
    bad_class_names: tuple[str, ...]
    bad_class_conf: float
    person_conf: float
    save_results: bool = True
    output_dir: Path | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> "DetectionPipeline":
        garment_detector = YoloDetector(
            model_path=settings.garment_model_path,
            input_size=settings.yolo_input_size,
            conf_threshold=settings.garment_conf,
            iou_threshold=settings.yolo_iou_threshold,
        )
        bad_classes_detector = YoloDetector(
            model_path=settings.bad_classes_detector_path,
            input_size=settings.yolo_input_size,
            conf_threshold=min(settings.bad_class_conf, settings.person_conf),
            iou_threshold=settings.yolo_iou_threshold,
        )
        return cls(
            garment_detector=garment_detector,
            bad_classes_detector=bad_classes_detector,
            bad_class_names=settings.bad_class_names,
            bad_class_conf=settings.bad_class_conf,
            person_conf=settings.person_conf,
            save_results=settings.save_inference_results,
            output_dir=Path(settings.inference_output_dir),
        )

    def _make_inference_dir(self) -> Path:
        if self.output_dir is None:
            raise ValueError("output_dir is not configured")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        return self.output_dir / f"{timestamp}_{uuid4().hex[:8]}"

    def process(self, image_bytes: bytes) -> bytes:
        image_rgb = decode_image_to_rgb(image_bytes)

        try:
            detector_result = self.garment_detector.detect_with_timing(image_rgb)
            detections = detector_result.detections
            timing = detector_result.timing
            logger.info(
                "garment detector timings: preprocess=%.2fms inference=%.2fms postprocess=%.2fms detections=%d",
                timing.preprocess_ms,
                timing.inference_ms,
                timing.postprocess_ms,
                len(detections),
            )
            if not detections:
                raise NoDetectionError("YOLO detector found no clothing")

            result_rgb = draw_detections(image_rgb, detections)
            result_png = encode_rgb_png(result_rgb)
            if self.save_results:
                save_inference_images(self._make_inference_dir(), image_rgb, result_png)
            return result_png
        except NoDetectionError:
            raise
        except Exception as exc:
            raise InferenceError(str(exc)) from exc

    def cascade(self, image_bytes: bytes) -> dict:
        image_rgb = decode_image_to_rgb(image_bytes)

        try:
            garment_result = self.garment_detector.detect_with_timing(image_rgb)
            garment_detections = garment_result.detections
            garment_timing = garment_result.timing
            logger.info(
                "garment detector timings: preprocess=%.2fms inference=%.2fms postprocess=%.2fms detections=%d",
                garment_timing.preprocess_ms,
                garment_timing.inference_ms,
                garment_timing.postprocess_ms,
                len(garment_detections),
            )
            if not garment_detections:
                return {
                    "decision": "REJECT",
                    "reason": "no_garment_detected",
                    "garment_detections": [],
                    "bad_class_detections": [],
                }

            bad_result = self.bad_classes_detector.detect_with_timing(image_rgb)
            bad_timing = bad_result.timing
            bad_class_detections = self._filter_bad_classes(bad_result.detections)
            logger.info(
                "bad-class detector timings: preprocess=%.2fms inference=%.2fms postprocess=%.2fms raw_detections=%d bad_detections=%d",
                bad_timing.preprocess_ms,
                bad_timing.inference_ms,
                bad_timing.postprocess_ms,
                len(bad_result.detections),
                len(bad_class_detections),
            )

            if bad_class_detections:
                return {
                    "decision": "REJECT",
                    "reason": "bad_class_detected",
                    "garment_detections": [_serialize_detection(d) for d in garment_detections],
                    "bad_class_detections": bad_class_detections,
                }

            return {
                "decision": "ACCEPT",
                "reason": "ok",
                "garment_detections": [_serialize_detection(d) for d in garment_detections],
                "bad_class_detections": [],
            }
        except Exception as exc:
            raise InferenceError(str(exc)) from exc

    def _filter_bad_classes(self, detections: list[Detection]) -> list[dict]:
        bad_class_names = set(self.bad_class_names)
        result = []
        for detection in detections:
            class_name = _coco_class_name(detection.class_id)
            if class_name not in bad_class_names:
                continue

            threshold = self.person_conf if class_name == "person" else self.bad_class_conf
            if detection.confidence < threshold:
                continue

            serialized = _serialize_detection(detection)
            serialized["class_name"] = class_name
            result.append(serialized)
        return result


def draw_detections(image_rgb: np.ndarray, detections: list[Detection]) -> np.ndarray:
    result = image_rgb.copy()
    height, width = result.shape[:2]
    thickness = max(2, round(min(width, height) / 300))
    color = (255, 0, 0)

    for detection in detections:
        x1, y1, x2, y2 = _bbox_to_pixels(detection.bbox, width, height)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

    return result


def _bbox_to_pixels(
    bbox_norm: list[float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_norm
    left = int(np.clip(round(min(x1, x2) * width), 0, width - 1))
    top = int(np.clip(round(min(y1, y2) * height), 0, height - 1))
    right = int(np.clip(round(max(x1, x2) * width), 0, width - 1))
    bottom = int(np.clip(round(max(y1, y2) * height), 0, height - 1))
    return left, top, right, bottom


def _serialize_detection(detection: Detection) -> dict:
    return {
        "bbox": detection.bbox,
        "confidence": detection.confidence,
        "class_id": detection.class_id,
    }


def _coco_class_name(class_id: int) -> str:
    if 0 <= class_id < len(COCO_CLASS_NAMES):
        return COCO_CLASS_NAMES[class_id]
    return f"class_{class_id}"
