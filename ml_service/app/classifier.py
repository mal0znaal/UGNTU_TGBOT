from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import onnxruntime as ort

from app.enrich import EnrichmentRules, detect_color, get_enrichment


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Превращает "сырые" числа (логиты) от нейросети в вероятности (от 0 до 1).
    Сумма всех вероятностей будет равна 1.
    """
    # Вычитаем максимум для стабильности вычислений (чтобы не было переполнения)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


class OnnxImageClassifier:
    """
    Класс для работы с ОДНОЙ моделью классификации (Category или Subcategory).
    Он умеет подготавливать картинку (preprocess) и предсказывать класс.
    """
    def __init__(self, model_path: str, classes: list[str], input_size: list[int], mean: list[float], std: list[float]):
        # Загружаем модель ONNX (выполняем на процессоре CPU)
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        
        self.classes = classes  # Список названий классов (например, ["Обувь", "Куртка"])
        self.input_size = input_size # Размер картинки для нейросети (например, [224, 224])
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def predict(self, image_rgb: np.ndarray) -> dict[str, Any]:
        """
        Главный метод: принимает картинку, отдает название класса и уверенность нейросети.
        """
        # 1. Меняем размер картинки под нейросеть
        resized = cv2.resize(image_rgb, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_LINEAR)
        
        # 2. Нормализация: переводим пиксели из [0, 255] в [0, 1], а затем применяем mean и std
        x = resized.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        
        # 3. Меняем форму тензора с HWC (Высота, Ширина, Каналы) на NCHW (Батч, Каналы, Высота, Ширина)
        x = np.transpose(x, (2, 0, 1)) # Было (224, 224, 3), стало (3, 224, 224)
        x = np.expand_dims(x, axis=0)  # Добавляем измерение батча: (1, 3, 224, 224)

        # 4. Прогоняем картинку через нейросеть
        outputs = self.session.run(None, {self.input_name: x})
        logits = outputs[0] # Получаем сырые логиты
        
        # 5. Превращаем логиты в вероятности
        probs = softmax(logits)[0]
        
        # 6. Ищем индекс класса с самой большой вероятностью
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        
        return {
            "label": self.classes[pred_idx],
            "confidence": confidence
        }


class FashionClassifier:
    """
    Каскадный классификатор: 
    Сначала определяет основную категорию (Верх, Низ, Обувь),
    Затем берет нужную подмодель и определяет подкатегорию (Футболка, Кроссовки и тд).
    В конце определяет цвет, сезон и стиль одежды.
    """
    def __init__(self, manifest_path: str, models_dir: str):
        # Читаем манифест (там лежат пути к моделям и списки классов)
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
            
        input_config = manifest["input"]
        self.input_size = input_config["size"]
        self.mean = input_config["mean"]
        self.std = input_config["std"]
        
        # Загружаем правила обогащения (стили и сезоны)
        self.rules = EnrichmentRules.from_manifest(manifest)
        
        # Инициализируем главную модель (Категории)
        cat_config = manifest["category_model"]
        cat_path = str(Path(models_dir) / cat_config["path"])
        self.category_classifier = OnnxImageClassifier(
            model_path=cat_path,
            classes=cat_config["classes"],
            input_size=self.input_size,
            mean=self.mean,
            std=self.std
        )
        
        # Инициализируем подмодели (Подкатегории) для каждой категории
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

    def classify(self, image_rgb: np.ndarray) -> dict[str, Any]:
        """
        Прогоняет картинку (кроп одежды) через весь пайплайн классификации.
        """
        # 1. Определяем базовую категорию
        cat_result = self.category_classifier.predict(image_rgb)
        category_label = cat_result["label"]
        
        # 2. Определяем подкатегорию (если для этой категории есть отдельная модель)
        subcategory_label = None
        
        # Если модель для подкатегории найдена в словаре:
        if category_label in self.subcategory_classifiers:
            sub_classifier = self.subcategory_classifiers[category_label]
            sub_result = sub_classifier.predict(image_rgb)
            subcategory_label = sub_result["label"]
            
        # 3. Достаем цвет, сезон и стили
        color_info = detect_color(image_rgb)
        seasons = get_enrichment(subcategory_label, self.rules, "seasons")
        styles = get_enrichment(subcategory_label, self.rules, "styles")
        
        # 4. Возвращаем всё вместе
        return {
            "category": category_label,
            "subcategory": subcategory_label,
            "color": color_info['name'],
            "seasons": seasons,
            "styles": styles
        }
