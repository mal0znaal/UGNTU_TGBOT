from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml

from .class_rules import BAD_CLASS_NAMES, coco_class_name
from .yolo_onnx import Detection, YoloOnnxDetector, decode_image_to_rgb


@dataclass(frozen=True)
class CascadeSettings:
    garment_model_path: Path
    bad_classes_detector_path: Path
    yolo_input_size: int = 960
    yolo_iou_threshold: float = 0.45
    garment_conf: float = 0.25
    bad_class_conf: float = 0.35
    person_conf: float = 0.75
    bad_class_names: tuple[str, ...] = tuple(BAD_CLASS_NAMES)
    garment_use_nms: bool = False
    bad_classes_use_nms: bool = False


def load_settings(config_path: str | Path) -> CascadeSettings:
    config_path = Path(config_path).resolve()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_dir = config_path.parent

    def resolve_path(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (base_dir / path).resolve()

    models = data["models"]
    thresholds = data.get("thresholds", {})
    yolo = data.get("yolo", {})
    rules = data.get("rules", {})

    return CascadeSettings(
        garment_model_path=resolve_path(models["garment_model_path"]),
        bad_classes_detector_path=resolve_path(models["bad_classes_detector_path"]),
        yolo_input_size=int(yolo.get("input_size", 960)),
        yolo_iou_threshold=float(yolo.get("iou_threshold", 0.45)),
        garment_conf=float(thresholds.get("garment_conf", 0.25)),
        bad_class_conf=float(thresholds.get("bad_class_conf", 0.35)),
        person_conf=float(thresholds.get("person_conf", 0.75)),
        bad_class_names=tuple(rules.get("bad_class_names", BAD_CLASS_NAMES)),
        garment_use_nms=bool(yolo.get("garment_use_nms", False)),
        bad_classes_use_nms=bool(yolo.get("bad_classes_use_nms", False)),
    )


class CascadeFilter:
    def __init__(self, settings: CascadeSettings) -> None:
        self.settings = settings
        self.garment_detector = YoloOnnxDetector(
            model_path=settings.garment_model_path,
            input_size=settings.yolo_input_size,
            conf_threshold=settings.garment_conf,
            iou_threshold=settings.yolo_iou_threshold,
            use_nms=settings.garment_use_nms,
        )
        self.bad_classes_detector = YoloOnnxDetector(
            model_path=settings.bad_classes_detector_path,
            input_size=settings.yolo_input_size,
            conf_threshold=min(settings.bad_class_conf, settings.person_conf),
            iou_threshold=settings.yolo_iou_threshold,
            use_nms=settings.bad_classes_use_nms,
        )

    @classmethod
    def from_config(cls, config_path: str | Path) -> "CascadeFilter":
        return cls(load_settings(config_path))

    def run_bytes(self, image_bytes: bytes) -> dict[str, Any]:
        return self.run_rgb(decode_image_to_rgb(image_bytes))

    def run_path(self, image_path: str | Path) -> dict[str, Any]:
        return self.run_bytes(Path(image_path).read_bytes())

    def run_rgb(self, image_rgb) -> dict[str, Any]:
        started_at = perf_counter()

        garment_result = self.garment_detector.detect(image_rgb)
        garment_detections = garment_result.detections
        if not garment_detections:
            return {
                "decision": "REJECT",
                "reason": "no_garment_detected",
                "garment_detections": [],
                "bad_class_detections": [],
                "timings_ms": {
                    "garment_preprocess": garment_result.timing.preprocess_ms,
                    "garment_inference": garment_result.timing.inference_ms,
                    "garment_postprocess": garment_result.timing.postprocess_ms,
                    "total": (perf_counter() - started_at) * 1000,
                },
            }

        bad_result = self.bad_classes_detector.detect(image_rgb)
        bad_class_detections = self.filter_bad_classes(bad_result.detections)
        decision = "REJECT" if bad_class_detections else "ACCEPT"
        reason = "bad_class_detected" if bad_class_detections else "ok"

        return {
            "decision": decision,
            "reason": reason,
            "garment_detections": [serialize_detection(d) for d in garment_detections],
            "bad_class_detections": bad_class_detections,
            "timings_ms": {
                "garment_preprocess": garment_result.timing.preprocess_ms,
                "garment_inference": garment_result.timing.inference_ms,
                "garment_postprocess": garment_result.timing.postprocess_ms,
                "bad_preprocess": bad_result.timing.preprocess_ms,
                "bad_inference": bad_result.timing.inference_ms,
                "bad_postprocess": bad_result.timing.postprocess_ms,
                "total": (perf_counter() - started_at) * 1000,
            },
        }

    def filter_bad_classes(self, detections: list[Detection]) -> list[dict[str, Any]]:
        bad_class_names = set(self.settings.bad_class_names)
        filtered = []

        for detection in detections:
            class_name = coco_class_name(detection.class_id)
            if class_name not in bad_class_names:
                continue

            threshold = (
                self.settings.person_conf
                if class_name == "person"
                else self.settings.bad_class_conf
            )
            if detection.confidence < threshold:
                continue

            serialized = serialize_detection(detection)
            serialized["class_name"] = class_name
            serialized["threshold"] = threshold
            filtered.append(serialized)

        return filtered


def serialize_detection(detection: Detection) -> dict[str, Any]:
    return {
        "bbox": detection.bbox,
        "confidence": detection.confidence,
        "class_id": detection.class_id,
    }
