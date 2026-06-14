from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from fastapi import BackgroundTasks

from app.classifier import FashionClassifier
from app.config import Settings
from app.image_utils import decode_image_to_rgb
from app.segformer import SegFormerSegmenter
from app.timing import PhaseTiming
from app.visualizer import save_debug_image
from app.yolo import Detection, YoloDetector


class NoDetectionError(RuntimeError):
    pass


class InferenceError(RuntimeError):
    pass


@dataclass
class DetectionPipeline:
    garment_detector: YoloDetector
    segmenter: SegFormerSegmenter
    classifier: FashionClassifier
    save_results: bool = True
    output_dir: Path | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> "DetectionPipeline":
        return cls(
            garment_detector=YoloDetector(
                model_path=settings.garment_model_path,
                input_size=settings.yolo_input_size,
                conf_threshold=settings.garment_conf,
            ),
            segmenter=SegFormerSegmenter(
                model_path=settings.segformer_model_path,
                input_size=settings.segformer_input_size,
                threshold=settings.segformer_threshold,
            ),
            classifier=FashionClassifier(
                manifest_path=settings.classifier_manifest_path,
                models_dir=settings.classifier_models_dir,
            ),
            save_results=settings.save_inference_results,
            output_dir=Path(settings.inference_output_dir),
        )

    def process(
        self,
        image_bytes: bytes,
        background_tasks: BackgroundTasks | None = None,
    ) -> dict:
        decode_start = perf_counter()
        image_rgb = decode_image_to_rgb(image_bytes)
        timing = PhaseTiming(
            preprocess_ms=(perf_counter() - decode_start) * 1000,
        )

        try:
            detector_result = self.garment_detector.detect(image_rgb)
            timing += detector_result.timing
            if not detector_result.detections:
                raise NoDetectionError("На фото не найдена одежда")

            crop_start = perf_counter()
            crop_rgb = _crop_largest_detection(image_rgb, detector_result.detections)
            crop_ms = (perf_counter() - crop_start) * 1000
            timing += PhaseTiming(postprocess_ms=crop_ms)

            mask, segmenter_timing = self.segmenter.predict_mask(crop_rgb)
            timing += segmenter_timing

            encode_start = perf_counter()
            crop_bgra = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGRA)
            crop_bgra[:, :, 3] = mask
            ok, buffer = cv2.imencode(".png", crop_bgra)
            if not ok:
                raise ValueError("Не удалось закодировать результат в PNG")
            image_base64 = base64.b64encode(buffer).decode("ascii")
            encode_ms = (perf_counter() - encode_start) * 1000
            timing += PhaseTiming(postprocess_ms=encode_ms)

            classification, classifier_timing = self.classifier.classify(crop_rgb)
            timing += classifier_timing

            if self.save_results and background_tasks is not None and self.output_dir is not None:
                background_tasks.add_task(
                    save_debug_image,
                    image_rgb,
                    crop_bgra,
                    classification,
                    self.output_dir,
                )

            return {
                "decision": "ACCEPT",
                "image_base64": image_base64,
                "classification": classification,
                "timings": timing.as_dict(),
            }
        except NoDetectionError:
            raise
        except Exception as exc:
            raise InferenceError(str(exc)) from exc


def _crop_largest_detection(
    image_rgb: np.ndarray,
    detections: list[Detection],
    padding_ratio: float = 0.1,
) -> np.ndarray:
    detection = max(
        detections,
        key=lambda item: max(0.0, item.bbox[2] - item.bbox[0])
        * max(0.0, item.bbox[3] - item.bbox[1]),
    )

    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = _bbox_to_pixels(detection.bbox, width, height)
    pad_x = int((x2 - x1) * padding_ratio)
    pad_y = int((y2 - y1) * padding_ratio)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)

    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("Детектор вернул некорректные координаты")
    return crop


def _bbox_to_pixels(
    bbox: list[float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        int(np.clip(min(x1, x2) * width, 0, width - 1)),
        int(np.clip(min(y1, y2) * height, 0, height - 1)),
        int(np.clip(max(x1, x2) * width, 0, width)),
        int(np.clip(max(y1, y2) * height, 0, height)),
    )
