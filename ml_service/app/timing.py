from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseTiming:
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0

    def __add__(self, other: "PhaseTiming") -> "PhaseTiming":
        return PhaseTiming(
            preprocess_ms=self.preprocess_ms + other.preprocess_ms,
            inference_ms=self.inference_ms + other.inference_ms,
            postprocess_ms=self.postprocess_ms + other.postprocess_ms,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "preprocess_ms": round(self.preprocess_ms, 2),
            "inference_ms": round(self.inference_ms, 2),
            "postprocess_ms": round(self.postprocess_ms, 2),
        }
