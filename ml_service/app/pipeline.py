from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.image_utils import add_bbox_padding, decode_image_to_rgb, encode_rgba_png
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
        return cls(detector=detector, segmenter=segmenter, crop_padding=settings.crop_padding)

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
            return encode_rgba_png(crop_rgb, alpha)
        except NoDetectionError:
            raise
        except Exception as exc:
            raise InferenceError(str(exc)) from exc
