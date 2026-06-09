import time
from pathlib import Path
import cv2
import numpy as np


def translit(text: str) -> str:
    """Простая транслитерация для отрисовки текста через cv2.putText (без Pillow)"""
    mapping = {
        'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E', 'Ё': 'E', 'Ж': 'Zh',
        'З': 'Z', 'И': 'I', 'Й': 'Y', 'К': 'K', 'Л': 'L', 'М': 'M', 'Н': 'N', 'О': 'O',
        'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U', 'Ф': 'F', 'Х': 'Kh', 'Ц': 'Ts',
        'Ч': 'Ch', 'Ш': 'Sh', 'Щ': 'Sch', 'Ъ': '', 'Ы': 'Y', 'Ь': '', 'Э': 'E', 'Ю': 'Yu', 'Я': 'Ya',
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'e', 'ж': 'zh',
        'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n', 'о': 'o',
        'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u', 'ф': 'f', 'х': 'kh', 'ц': 'ts',
        'ч': 'ch', 'ш': 'sh', 'щ': 'sch', 'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya'
    }
    return "".join(mapping.get(c, c) for c in text)


def save_debug_image(source_rgb: np.ndarray, crop_bgra: np.ndarray, cls_info: dict, output_dir: Path):
    """
    Эффективно объединяет исходную картинку и вырезанную одежду, 
    используя ТОЛЬКО OpenCV.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Готовим картинки
        target_h = 800
        
        # Ресайз исходника и сразу перевод из RGB в BGR (для OpenCV)
        h_s, w_s = source_rgb.shape[:2]
        scale_s = target_h / max(1, h_s)
        new_w_s = int(w_s * scale_s)
        source_resized = cv2.resize(source_rgb, (new_w_s, target_h))
        source_bgr = cv2.cvtColor(source_resized, cv2.COLOR_RGB2BGR)
        
        # Ресайз кропа (он уже в BGRA)
        h_c, w_c = crop_bgra.shape[:2]
        scale_c = target_h / max(1, h_c)
        new_w_c = int(w_c * scale_c)
        crop_resized = cv2.resize(crop_bgra, (new_w_c, target_h))
        
        # Накладываем кроп на белый фон (работаем в BGR)
        crop_bgr_only = crop_resized[:, :, :3]
        alpha = crop_resized[:, :, 3] / 255.0
        white_bg = np.ones_like(crop_bgr_only) * 255
        crop_on_white = (crop_bgr_only * alpha[..., None] + white_bg * (1 - alpha[..., None])).astype(np.uint8)
        
        # 2. Объединяем их по горизонтали
        merged_bgr = cv2.hconcat([source_bgr, crop_on_white])
        
        # 3. Добавляем плашку с текстом через OpenCV (с транслитерацией)
        overlay = merged_bgr.copy()
        cv2.rectangle(overlay, (20, 20), (500, 220), (0, 0, 0), -1)
        # Применяем прозрачность плашки
        alpha_rect = 0.6
        merged_bgr = cv2.addWeighted(overlay, alpha_rect, merged_bgr, 1 - alpha_rect, 0)
        
        lines = [
            "Category: " + translit(cls_info.get("category", "?")),
            "Subcat: " + translit(cls_info.get("subcategory", "?")),
            "Color: " + cls_info.get("color", "?"),
            "Seasons: " + translit(", ".join(cls_info.get("seasons", []))),
            "Styles: " + translit(", ".join(cls_info.get("styles", []))),
        ]
        
        y_offset = 55
        for line in lines:
            cv2.putText(merged_bgr, line, (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            y_offset += 35
        
        # 4. Сохраняем в папку (без дополнительных конвертаций)
        filename = f"result_{int(time.time()*1000)}.jpg"
        cv2.imwrite(str(output_dir / filename), merged_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
    except Exception as e:
        print(f"Ошибка сохранения дебаг-изображения: {e}")
