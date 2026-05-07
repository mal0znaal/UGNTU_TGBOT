import os
from dataclasses import dataclass


def _get_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _get_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    detector_onnx_path: str = os.getenv("DETECTOR_ONNX_PATH", "/app/models/detector.onnx")
    segmenter_onnx_path: str = os.getenv("SEGMENTER_ONNX_PATH", "/app/models/segmenter.onnx")
    yolo_input_size: int = _get_int("YOLO_INPUT_SIZE", 960)
    seg_input_size: int = _get_int("SEG_INPUT_SIZE", 720)
    yolo_conf_threshold: float = _get_float("YOLO_CONF_THRESHOLD", 0.25)
    yolo_iou_threshold: float = _get_float("YOLO_IOU_THRESHOLD", 0.45)
    crop_padding: float = _get_float("CROP_PADDING", 0.10)
    mask_threshold: float = _get_float("MASK_THRESHOLD", 0.5)
    save_inference_results: bool = _get_bool("SAVE_INFERENCE_RESULTS", True)
    inference_output_dir: str = os.getenv("INFERENCE_OUTPUT_DIR", "/app/inference_results")


def get_settings() -> Settings:
    return Settings()
