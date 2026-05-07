from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch import nn
from transformers import SegformerForSemanticSegmentation


class SegFormerLogitsWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).logits


def build_segformer_b0(pretrained_name: str) -> SegformerForSemanticSegmentation:
    return SegformerForSemanticSegmentation.from_pretrained(
        pretrained_name,
        num_labels=1,
        ignore_mismatched_sizes=True,
    )


def load_pretrained_name(config_path: Path) -> str:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["model"]["pretrained_name"]


def export_segformer(
    checkpoint_path: Path,
    config_path: Path,
    output_path: Path,
    input_size: int,
    opset: int,
) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    pretrained_name = load_pretrained_name(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain model_state_dict")

    model = build_segformer_b0(pretrained_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    wrapped = SegFormerLogitsWrapper(model).eval()
    dummy_input = torch.randn(1, 3, input_size, input_size, dtype=torch.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped,
        dummy_input,
        output_path.as_posix(),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes=None,
        dynamo=False,
    )

    print(f"Exported SegFormer ONNX to: {output_path}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export binary SegFormer checkpoint to fixed-shape ONNX logits model."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--input-size", default=720, type=int)
    parser.add_argument("--opset", default=17, type=int)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    export_segformer(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_path=args.output,
        input_size=args.input_size,
        opset=args.opset,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
