from __future__ import annotations

from collections import Counter


def confusion_bucket(decision: str, has_gt_bbox: bool) -> str:
    accepted = decision == "ACCEPT"
    if accepted and has_gt_bbox:
        return "TP"
    if accepted and not has_gt_bbox:
        return "FP"
    if not accepted and not has_gt_bbox:
        return "TN"
    return "FN"


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def compute_binary_metrics(counts: Counter) -> dict[str, float | int]:
    tp = counts["TP"]
    tn = counts["TN"]
    fp = counts["FP"]
    fn = counts["FN"]
    total = tp + tn + fp + fn

    return {
        "total": total,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": safe_div(tp + tn, total),
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
    }
