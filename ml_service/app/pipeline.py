from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.config import Settings
from app.image_utils import (
    add_bbox_padding,
    decode_image_to_rgb,
    encode_rgba_png,
    save_inference_images,
)
from app.segformer import SegFormerSegmenter
from app.yolo import YoloDetector, select_largest_detection


class NoDetectionError(RuntimeError):
    pass


class InferenceError(RuntimeError):
    pass


@dataclass
class BackgroundRemovalPipeline:
    detector: YoloDetector
    segmenter: SegFormerSegmenter
    crop_padding: float
    save_results: bool = True
    output_dir: Path | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> "BackgroundRemovalPipeline":
        detector = YoloDetector(
            model_path=settings.detector_onnx_path,
            input_size=settings.yolo_input_size,
            conf_threshold=settings.yolo_conf_threshold,
            iou_threshold=settings.yolo_iou_threshold,
        )
        segmenter = SegFormerSegmenter(
            model_path=settings.segmenter_onnx_path,
            input_size=settings.seg_input_size,
            threshold=settings.mask_threshold,
        )
        return cls(
            detector=detector,
            segmenter=segmenter,
            crop_padding=settings.crop_padding,
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
            detections = self.detector.detect(image_rgb)
            detection = select_largest_detection(detections)
            if detection is None:
                raise NoDetectionError("YOLO detector found no clothing")

            left, top, right, bottom = add_bbox_padding(
                detection.bbox,
                image_rgb.shape,
                self.crop_padding,
            )
            crop_rgb = image_rgb[top:bottom, left:right]
            alpha = self.segmenter.predict_mask(crop_rgb)
            result_png = encode_rgba_png(crop_rgb, alpha)
            if self.save_results:
                save_inference_images(self._make_inference_dir(), image_rgb, result_png)
            return result_png
        except NoDetectionError:
            raise
        except Exception as exc:
            raise InferenceError(str(exc)) from exc
