from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from pneumonia_detection.model import PneumoniaModel


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def export_onnx(cfg: DictConfig, checkpoint_path: Path, onnx_path: Path) -> Path:
    checkpoint_path = _resolve(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    onnx_path = _resolve(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    model = PneumoniaModel.load_from_checkpoint(checkpoint_path, cfg=cfg.model)
    model.eval()

    img_size = int(OmegaConf.select(cfg, "img_size"))
    input_channels = int(OmegaConf.select(cfg, "input_channels"))
    opset_ver = int(OmegaConf.select(cfg, "opset_version"))

    device = next(model.parameters()).device
    dummy = torch.randn(1, input_channels, img_size, img_size, device=device)

    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            opset_version=opset_ver,
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
    return onnx_path


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    ckpt = OmegaConf.select(cfg, "ckpt_path")
    export_onnx(cfg, Path(ckpt), Path(cfg.export.onnx_path))


if __name__ == "__main__":
    main()
