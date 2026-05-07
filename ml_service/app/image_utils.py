from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class DecodeImageError(ValueError):
    pass


def decode_image_to_rgb(image_bytes: bytes) -> np.ndarray:
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise DecodeImageError("Cannot decode image")

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    channels = image.shape[2]
    if channels == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)[:, :, :3]

    raise DecodeImageError(f"Unsupported channel count: {channels}")


def add_bbox_padding(
    bbox_norm: list[float],
    image_shape: tuple[int, int, int],
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = bbox_norm

    x1_px = x1 * width
    y1_px = y1 * height
    x2_px = x2 * width
    y2_px = y2 * height

    box_w = max(0.0, x2_px - x1_px)
    box_h = max(0.0, y2_px - y1_px)
    pad_x = box_w * padding_ratio
    pad_y = box_h * padding_ratio

    left = max(0, int(np.floor(x1_px - pad_x)))
    top = max(0, int(np.floor(y1_px - pad_y)))
    right = min(width, int(np.ceil(x2_px + pad_x)))
    bottom = min(height, int(np.ceil(y2_px + pad_y)))

    if right <= left or bottom <= top:
        raise ValueError("Invalid bbox after padding")

    return left, top, right, bottom


def encode_rgba_png(rgb: np.ndarray, alpha: np.ndarray) -> bytes:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image, got shape={rgb.shape}")
    if alpha.shape[:2] != rgb.shape[:2]:
        raise ValueError(f"Alpha mask shape={alpha.shape} does not match image shape={rgb.shape}")

    rgba = np.dstack([rgb, alpha.astype(np.uint8)])
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    ok, encoded = cv2.imencode(".png", bgra)
    if not ok:
        raise ValueError("Failed to encode PNG")
    return encoded.tobytes()


def encode_rgb_png(rgb: np.ndarray) -> bytes:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image, got shape={rgb.shape}")

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", bgr)
    if not ok:
        raise ValueError("Failed to encode source PNG")
    return encoded.tobytes()


def save_inference_images(output_dir: Path, source_rgb: np.ndarray, result_png: bytes) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "source.png").write_bytes(encode_rgb_png(source_rgb))
    (output_dir / "result.png").write_bytes(result_png)
    return output_dir
