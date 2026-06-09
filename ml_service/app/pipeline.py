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
import base64
from app.segformer import SegFormerSegmenter
from app.classifier import FashionClassifier
from fastapi import BackgroundTasks
from app.visualizer import save_debug_image


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
    segformer: SegFormerSegmenter
    classifier: FashionClassifier
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
        segformer = SegFormerSegmenter(
            model_path=settings.segformer_model_path,
            input_size=settings.segformer_input_size,
            threshold=settings.segformer_threshold,
        )
        classifier = FashionClassifier(
            manifest_path=settings.classifier_manifest_path,
            models_dir=settings.classifier_models_dir,
        )
        return cls(
            garment_detector=garment_detector,
            segformer=segformer,
            classifier=classifier,
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

    def process_full(self, image_bytes: bytes, bg_tasks: BackgroundTasks = None) -> dict:
        """
        Новый метод, который:
        1. Находит одежду (YOLO)
        2. Вырезает (Crop) самую крупную вещь
        3. Удаляет фон (SegFormer)
        4. Классифицирует цвет, сезон, стиль и категорию (FashionClassifier)
        """
        image_rgb = decode_image_to_rgb(image_bytes)

        try:
            # 1. Детектируем одежду (YOLO)
            garment_result = self.garment_detector.detect_with_timing(image_rgb)
            detections = garment_result.detections
            
            if not detections:
                raise NoDetectionError("На фото не найдена одежда")
                
            # Ищем самый большой BBox (самую крупную вещь) по площади
            best_detection = max(
                detections,
                key=lambda d: max(0.0, d.bbox[2] - d.bbox[0]) * max(0.0, d.bbox[3] - d.bbox[1])
            )
            
            # 2. Вырезаем (crop) найденную одежду
            height, width = image_rgb.shape[:2]
            x1, y1, x2, y2 = _bbox_to_pixels(best_detection.bbox, width, height)
            
            # Расширяем бокс на 10% со всех сторон, чтобы не обрезались края
            bw = x2 - x1
            bh = y2 - y1
            pad_x = int(bw * 0.10)
            pad_y = int(bh * 0.10)
            
            # Не забываем проследить, чтобы не вылезти за границы картинки
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(width, x2 + pad_x)
            y2 = min(height, y2 + pad_y)
            
            crop_rgb = image_rgb[y1:y2, x1:x2]
            if crop_rgb.size == 0:
                raise InferenceError("Кроп одежды пустой (ошибка координат)")
                
            # 3. Применяем SegFormer к кропу, чтобы удалить фон
            # segformer вернет маску, где одежда это 255, а фон это 0
            mask = self.segformer.predict_mask(crop_rgb)
            
            # Превращаем RGB напрямую в BGRA
            crop_bgra = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGRA)
            crop_bgra[:, :, 3] = mask # Кладем маску в канал альфа (прозрачности)
            
            # Кодируем картинку в PNG байты, а потом в строку Base64
            _, buffer = cv2.imencode('.png', crop_bgra)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # 4. Прогоняем кроп (оригинальный RGB, без вырезанного фона) через Классификатор
            classification_info = self.classifier.classify(crop_rgb)
            
            # Если включен флаг сохранения, то в фоне склеиваем и сохраняем картинку
            if self.save_results and bg_tasks is not None:
                bg_tasks.add_task(save_debug_image, image_rgb, crop_bgra, classification_info, self.output_dir)
            
            # 5. Возвращаем всё в одном удобном словаре
            return {
                "decision": "ACCEPT",
                "image_base64": image_base64,
                "classification": classification_info
            }
            
        except NoDetectionError:
            raise
        except Exception as exc:
            raise InferenceError(str(exc)) from exc




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
    return (
        int(np.clip(min(x1, x2) * width, 0, width - 1)),
        int(np.clip(min(y1, y2) * height, 0, height - 1)),
        int(np.clip(max(x1, x2) * width, 0, width - 1)),
        int(np.clip(max(y1, y2) * height, 0, height - 1))
    )


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
