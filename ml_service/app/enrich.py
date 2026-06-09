from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# Простой тип для RGB цвета
RGB = tuple[int, int, int]


@dataclass(frozen=True)
class EnrichmentRules:
    """
    Правила обогащения: какие сезоны и стили выдавать для каждой вещи.
    Эти данные загружаются из manifest.json.
    """
    default_seasons: list[str]
    default_styles: list[str]
    aliases: dict[str, str]
    season_by_subcategory: dict[str, list[str]]
    style_by_subcategory: dict[str, list[str]]

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any]) -> "EnrichmentRules":
        data = manifest.get("enrichment", {})
        return cls(
            default_seasons=list(data.get("default_seasons", [])),
            default_styles=list(data.get("default_styles", ["Мульти-стиль"])),
            aliases=dict(data.get("aliases", {})),
            season_by_subcategory={str(k): list(v) for k, v in data.get("season_by_subcategory", {}).items()},
            style_by_subcategory={
                str(k).strip().lower(): list(v) for k, v in data.get("style_by_subcategory", {}).items()
            },
        )


BASIC_COLORS = {
    "Черный": (0, 0, 0),
    "Белый": (255, 255, 255),
    "Серый": (128, 128, 128),
    "Светло-серый": (192, 192, 192),
    "Красный": (255, 0, 0),
    "Бордовый": (128, 0, 0),
    "Оранжевый": (255, 128, 0),
    "Желтый": (255, 255, 0),
    "Зеленый": (0, 128, 0),
    "Салатовый": (0, 255, 0),
    "Оливковый / Хаки": (128, 128, 0),
    "Голубой": (0, 255, 255),
    "Синий": (0, 0, 255),
    "Темно-синий": (0, 0, 128),
    "Фиолетовый": (128, 0, 128),
    "Коричневый": (139, 69, 19),
    "Бежевый": (245, 245, 220),
    "Розовый": (255, 192, 203),
    "Малиновый": (220, 20, 60),
    "Персиковый": (255, 218, 185)
}

def rgb_to_name(r: int, g: int, b: int) -> str:
    def dist(c1, c2):
        return (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2 + (c1[2] - c2[2])**2
    return min(BASIC_COLORS.keys(), key=lambda k: dist((r, g, b), BASIC_COLORS[k]))


def detect_color(image_rgb: np.ndarray) -> dict[str, Any]:
    """
    Определяет доминирующий цвет на изображении одежды.
    Возвращает словарь с RGB, HEX и текстовым названием цвета.
    """
    r, g, b = detect_dominant_rgb(image_rgb)
    return {
        "hex": "#{:02X}{:02X}{:02X}".format(r, g, b),
        "rgb": [r, g, b],
        "name": rgb_to_name(r, g, b)
    }


def get_enrichment(subcategory: str | None, rules: EnrichmentRules, enrich_type: str) -> list[str]:
    """
    Универсальная функция для получения сезонов или стилей по подкатегории.
    enrich_type может быть "seasons" или "styles".
    """
    default_vals = rules.default_seasons if enrich_type == "seasons" else rules.default_styles
    if subcategory is None:
        return list(default_vals)

    normalized = rules.aliases.get(subcategory.strip(), subcategory.strip())
    
    if enrich_type == "styles":
        normalized = normalized.lower()
        mapping = rules.style_by_subcategory
    else:
        mapping = rules.season_by_subcategory
        
    return list(mapping.get(normalized, default_vals))


def detect_dominant_rgb(image_rgb: np.ndarray) -> RGB:
    """
    Алгоритм поиска доминирующего цвета (K-Means кластеризация).
    Упрощен для студентов.
    """
    # 1. Ресайзим картинку, чтобы алгоритм работал быстрее
    h, w = image_rgb.shape[:2]
    max_side = 300
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        image_rgb = cv2.resize(image_rgb, (max(1, int(w * scale)), max(1, int(h * scale))))

    # 2. Вырезаем только центр картинки, потому что одежда обычно там
    h, w = image_rgb.shape[:2]
    y1, y2 = int(h * 0.2), int(h * 0.8)
    x1, x2 = int(w * 0.2), int(w * 0.8)
    crop = image_rgb[y1:y2, x1:x2]
    
    if crop.size == 0:
        crop = image_rgb  # Если кроп пустой, берем всю картинку

    # 3. Превращаем 2D картинку в плоский массив пикселей: [[R, G, B], [R, G, B], ...]
    pixels = crop.reshape(-1, 3).astype(np.float32)

    # 4. Чтобы не считать кластеры по тысячам пикселей, берем случайные 5000 штук
    if len(pixels) > 5000:
        np.random.seed(42)  # Чтобы результат был одинаковым при перезапусках
        indices = np.random.choice(len(pixels), 5000, replace=False)
        pixels = pixels[indices]

    # 5. Применяем K-Means (стандартный алгоритм OpenCV) для поиска 3 главных цветов (k=3)
    k = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, flags)

    # 6. Считаем, какой кластер (цвет) встречается чаще всего
    counts = np.bincount(labels.flatten())
    dominant_cluster_index = np.argmax(counts)
    
    # 7. Получаем RGB самого частого цвета
    dominant_color = centers[dominant_cluster_index]
    
    # Преобразуем из float в целые числа 0-255
    r, g, b = np.clip(np.round(dominant_color), 0, 255).astype(int)
    return int(r), int(g), int(b)
