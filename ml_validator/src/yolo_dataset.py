from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class YoloSample:
    image_path: Path
    label_path: Path
    has_gt_bbox: bool


def find_images(dataset_root: str | Path, split: str) -> list[Path]:
    image_dir = Path(dataset_root) / "images" / split
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def label_path_for(dataset_root: str | Path, split: str, image_path: str | Path) -> Path:
    image_path = Path(image_path)
    return Path(dataset_root) / "labels" / split / f"{image_path.stem}.txt"


def has_gt_bbox(label_path: str | Path) -> bool:
    label_path = Path(label_path)
    if not label_path.exists():
        return False
    return bool(label_path.read_text(encoding="utf-8").strip())


def read_yolo_label_detections(label_path: str | Path) -> list[dict]:
    label_path = Path(label_path)
    if not label_path.exists():
        return []

    detections = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        class_id_raw, x_center_raw, y_center_raw, width_raw, height_raw, *_ = line.split()
        class_id = int(float(class_id_raw))
        x_center = float(x_center_raw)
        y_center = float(y_center_raw)
        box_w = float(width_raw)
        box_h = float(height_raw)

        detections.append(
            {
                "bbox": [
                    x_center - box_w / 2,
                    y_center - box_h / 2,
                    x_center + box_w / 2,
                    y_center + box_h / 2,
                ],
                "confidence": 1.0,
                "class_id": class_id,
                "class_name": f"GT class {class_id}",
            }
        )
    return detections


def iter_yolo_samples(
    dataset_root: str | Path,
    split: str,
    max_images: int | None = None,
) -> list[YoloSample]:
    images = find_images(dataset_root, split)
    if max_images is not None:
        images = images[:max_images]

    samples = []
    for image_path in images:
        label_path = label_path_for(dataset_root, split, image_path)
        samples.append(
            YoloSample(
                image_path=image_path,
                label_path=label_path,
                has_gt_bbox=has_gt_bbox(label_path),
            )
        )
    return samples
