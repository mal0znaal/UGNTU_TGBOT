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
    garment_model_path: str = os.getenv("GARMENT_MODEL_PATH", "/app/models/detector.onnx")
    yolo_input_size: int = _get_int("YOLO_INPUT_SIZE", 960)
    garment_conf: float = _get_float("GARMENT_CONF", 0.4)
    save_inference_results: bool = _get_bool("SAVE_INFERENCE_RESULTS", True)
    inference_output_dir: str = os.getenv("INFERENCE_OUTPUT_DIR", "/app/inference_results")
    segformer_model_path: str = os.getenv("SEGFORMER_MODEL_PATH", "/app/models/segmenter.onnx")
    segformer_input_size: int = _get_int("SEGFORMER_INPUT_SIZE", 720)
    segformer_threshold: float = _get_float("SEGFORMER_THRESHOLD", 0.5)

    classifier_manifest_path: str = os.getenv("CLASSIFIER_MANIFEST_PATH", "/app/models/classifier/manifest.json")
    classifier_models_dir: str = os.getenv("CLASSIFIER_MODELS_DIR", "/app/models/classifier")


def get_settings() -> Settings:
    return Settings()
