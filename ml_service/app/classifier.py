from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from app.enrich import EnrichmentRules, detect_color, get_enrichment
from app.timing import PhaseTiming


def softmax(logits: np.ndarray) -> np.ndarray:
    """Преобразует логиты модели в вероятности."""
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


class OnnxImageClassifier:
    """Обёртка над одной ONNX-моделью классификации."""

    def __init__(
        self,
        model_path: str,
        classes: list[str],
        input_size: list[int],
        mean: list[float],
        std: list[float],
    ):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

        self.classes = classes
        self.input_size = input_size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def predict(self, image_rgb: np.ndarray) -> tuple[dict[str, Any], PhaseTiming]:
        preprocess_start = perf_counter()
        resized = cv2.resize(image_rgb, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_LINEAR)
        x = resized.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, axis=0)
        preprocess_ms = (perf_counter() - preprocess_start) * 1000

        inference_start = perf_counter()
        outputs = self.session.run(None, {self.input_name: x})
        inference_ms = (perf_counter() - inference_start) * 1000

        postprocess_start = perf_counter()
        logits = outputs[0]
        probs = softmax(logits)[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        result = {"label": self.classes[pred_idx], "confidence": confidence}
        postprocess_ms = (perf_counter() - postprocess_start) * 1000

        return result, PhaseTiming(preprocess_ms, inference_ms, postprocess_ms)


class FashionClassifier:
    """Определяет категорию, подкатегорию и дополнительные признаки одежды."""

    def __init__(self, manifest_path: str, models_dir: str):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
            
        input_config = manifest["input"]
        self.input_size = input_config["size"]
        self.mean = input_config["mean"]
        self.std = input_config["std"]
        
        self.rules = EnrichmentRules.from_manifest(manifest)

        cat_config = manifest["category_model"]
        cat_path = str(Path(models_dir) / cat_config["path"])
        self.category_classifier = OnnxImageClassifier(
            model_path=cat_path,
            classes=cat_config["classes"],
            input_size=self.input_size,
            mean=self.mean,
            std=self.std
        )
        
        self.subcategory_classifiers = {}
        for category_name, sub_config in manifest["subcategory_models"].items():
            sub_path = str(Path(models_dir) / sub_config["path"])
            self.subcategory_classifiers[category_name] = OnnxImageClassifier(
                model_path=sub_path,
                classes=sub_config["classes"],
                input_size=self.input_size,
                mean=self.mean,
                std=self.std
            )

    def classify(self, image_rgb: np.ndarray) -> tuple[dict[str, Any], PhaseTiming]:
        cat_result, timing = self.category_classifier.predict(image_rgb)
        category_label = cat_result["label"]
        subcategory_label = None

        if category_label in self.subcategory_classifiers:
            sub_classifier = self.subcategory_classifiers[category_label]
            sub_result, sub_timing = sub_classifier.predict(image_rgb)
            timing += sub_timing
            subcategory_label = sub_result["label"]

        enrich_start = perf_counter()
        color_info = detect_color(image_rgb)
        seasons = get_enrichment(subcategory_label, self.rules, "seasons")
        styles = get_enrichment(subcategory_label, self.rules, "styles")
        result = {
            "category": category_label,
            "subcategory": subcategory_label,
            "color": color_info["name"],
            "seasons": seasons,
            "styles": styles,
        }
        enrich_ms = (perf_counter() - enrich_start) * 1000

        return result, timing + PhaseTiming(postprocess_ms=enrich_ms)
