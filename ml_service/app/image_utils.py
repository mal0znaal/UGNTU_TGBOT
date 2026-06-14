from __future__ import annotations

import cv2
import numpy as np


class DecodeImageError(ValueError):
    pass


def decode_image_to_rgb(image_bytes: bytes) -> np.ndarray:
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise DecodeImageError("Не удалось прочитать изображение")

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    channels = image.shape[2]
    if channels == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

    raise DecodeImageError(f"Неподдерживаемое число каналов: {channels}")
