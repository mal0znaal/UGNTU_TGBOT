from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


RGB = tuple[int, int, int]


@dataclass(frozen=True)
class EnrichmentRules:
    """Правила сопоставления подкатегорий с сезонами и стилями."""
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
    """Возвращает доминирующий цвет в нескольких форматах."""
    r, g, b = detect_dominant_rgb(image_rgb)
    return {
        "hex": "#{:02X}{:02X}{:02X}".format(r, g, b),
        "rgb": [r, g, b],
        "name": rgb_to_name(r, g, b)
    }


def get_enrichment(subcategory: str | None, rules: EnrichmentRules, enrich_type: str) -> list[str]:
    """Возвращает сезоны или стили для подкатегории."""
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
    """Находит доминирующий цвет с помощью K-Means."""
    h, w = image_rgb.shape[:2]
    max_side = 300
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        image_rgb = cv2.resize(image_rgb, (max(1, int(w * scale)), max(1, int(h * scale))))

    # Центральная область снижает влияние фона на результат.
    h, w = image_rgb.shape[:2]
    y1, y2 = int(h * 0.2), int(h * 0.8)
    x1, x2 = int(w * 0.2), int(w * 0.8)
    crop = image_rgb[y1:y2, x1:x2]
    
    if crop.size == 0:
        crop = image_rgb

    pixels = crop.reshape(-1, 3).astype(np.float32)

    # Ограничиваем выборку, чтобы время обработки не зависело от разрешения.
    if len(pixels) > 5000:
        np.random.seed(42)
        indices = np.random.choice(len(pixels), 5000, replace=False)
        pixels = pixels[indices]

    k = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, flags)

    counts = np.bincount(labels.flatten())
    dominant_cluster_index = np.argmax(counts)
    
    dominant_color = centers[dominant_cluster_index]
    r, g, b = np.clip(np.round(dominant_color), 0, 255).astype(int)
    return int(r), int(g), int(b)
