from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np


def bbox_to_pixels(
    bbox_norm: list[float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_norm
    left = int(np.clip(round(min(x1, x2) * width), 0, width - 1))
    top = int(np.clip(round(min(y1, y2) * height), 0, height - 1))
    right = int(np.clip(round(max(x1, x2) * width), 0, width - 1))
    bottom = int(np.clip(round(max(y1, y2) * height), 0, height - 1))
    return left, top, right, bottom


def draw_detections(
    image_rgb: np.ndarray,
    detections: list[dict],
    color: tuple[int, int, int],
    label_key: str | None = None,
) -> np.ndarray:
    result = image_rgb.copy()
    height, width = result.shape[:2]
    thickness = max(2, round(min(width, height) / 300))
    font_scale = max(0.45, min(width, height) / 1000)

    for detection in detections:
        x1, y1, x2, y2 = bbox_to_pixels(detection["bbox"], width, height)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

        label = f'{detection.get(label_key, "cls")} {detection.get("confidence", 0):.2f}' if label_key else f'{detection.get("confidence", 0):.2f}'
        (label_w, label_h), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            max(1, thickness - 1),
        )
        label_top = max(0, y1 - label_h - baseline - thickness)
        label_bottom = label_top + label_h + baseline + thickness
        label_right = min(width, x1 + label_w + thickness * 2)
        cv2.rectangle(result, (x1, label_top), (label_right, label_bottom), color, -1)
        cv2.putText(
            result,
            label,
            (x1 + thickness, label_bottom - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            max(1, thickness - 1),
            cv2.LINE_AA,
        )

    return result


def draw_cascade_result(image_rgb: np.ndarray, result: dict) -> np.ndarray:
    drawn = draw_detections(
        image_rgb,
        result["garment_detections"],
        color=(255, 0, 0),
        label_key=None,
    )
    return draw_detections(
        drawn,
        result["bad_class_detections"],
        color=(255, 190, 0),
        label_key="class_name",
    )


def draw_cascade_result_with_gt(
    image_rgb: np.ndarray,
    result: dict,
    gt_detections: list[dict],
) -> np.ndarray:
    drawn = draw_cascade_result(image_rgb, result)
    return draw_detections(
        drawn,
        gt_detections,
        color=(0, 210, 80),
        label_key="class_name",
    )


def show_image(image_rgb: np.ndarray, title: str | None = None, figsize=(9, 9)) -> None:
    plt.figure(figsize=figsize)
    plt.imshow(image_rgb)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()


def show_images_grid(
    items: list[tuple[np.ndarray, str]],
    columns: int = 2,
    figsize_per_image: tuple[int, int] = (7, 7),
) -> None:
    if not items:
        print("Нет картинок для отображения")
        return

    rows = int(np.ceil(len(items) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(figsize_per_image[0] * columns, figsize_per_image[1] * rows),
    )
    axes_array = np.array(axes).reshape(-1)

    for axis, (image_rgb, title) in zip(axes_array, items):
        axis.imshow(image_rgb)
        axis.set_title(title)
        axis.axis("off")

    for axis in axes_array[len(items) :]:
        axis.axis("off")

    plt.tight_layout()
    plt.show()
