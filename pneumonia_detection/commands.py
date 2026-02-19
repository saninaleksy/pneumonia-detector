from __future__ import annotations

from pathlib import Path
from typing import Sequence

import fire
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from pneumonia_detection.export_to_onnx import export_onnx
from pneumonia_detection.import_data import download_data
from pneumonia_detection.inference import infer_one
from pneumonia_detection.onnx_to_tensorrt import convert_onnx_to_tensorrt
from pneumonia_detection.run_triton import run_triton_server
from pneumonia_detection.train import train_model
from pneumonia_detection.triton_setup import create_triton_model_repository


def _compose_cfg(overrides: Sequence[str]) -> DictConfig:
    with initialize(version_base="1.3", config_path="../configs"):
        return compose(config_name="config", overrides=list(overrides))


def train(*overrides: str) -> None:
    cfg = _compose_cfg(overrides)
    train_model(cfg)


def export(*overrides: str) -> None:
    cfg = _compose_cfg(overrides)

    ckpt = OmegaConf.select(cfg, "export.ckpt_path")
    if not ckpt:
        raise ValueError("Set ckpt_path=...")

    onnx_path = OmegaConf.select(cfg, "export.onnx_path")
    if not onnx_path:
        raise ValueError("Set export.onnx_path=...")

    export_onnx(cfg, checkpoint_path=Path(str(ckpt)), onnx_path=Path(str(onnx_path)))


def onnx_to_tensorrt(*overrides: str) -> None:
    cfg = _compose_cfg(overrides)

    onnx_path = OmegaConf.select(cfg, "export.onnx_path")
    if not onnx_path:
        raise ValueError("Set export.onnx_path=...")

    engine_path = OmegaConf.select(cfg, "export.engine_path")
    if not engine_path:
        raise ValueError("Set export.engine_path=...")

    workspace = int(OmegaConf.select(cfg, "export.workspace_size"))
    fp16 = bool(OmegaConf.select(cfg, "export.fp16"))
    int8 = bool(OmegaConf.select(cfg, "export.int8"))

    convert_onnx_to_tensorrt(
        Path(str(onnx_path)),
        Path(str(engine_path)),
        workspace_size=workspace,
        fp16=fp16,
        int8=int8,
    )


def setup_triton(*overrides: str) -> None:
    cfg = _compose_cfg(overrides)
    try:
        create_triton_model_repository(cfg)
    except Exception:
        raise RuntimeError("Failed to create Triton model repository.")


def infer(*overrides: str) -> None:
    cfg = _compose_cfg(overrides)
    prob = infer_one(cfg)
    print(f"Predicted probability of pneumonia: {prob:.4f}")


def import_data(*overrides: str) -> None:
    cfg = _compose_cfg(overrides)
    root_dir = Path(str(OmegaConf.select(cfg, "data.root_dir")))
    download_data(str(root_dir))


def run_triton(*overrides: str) -> None:
    cfg = _compose_cfg(overrides)
    run_triton_server(cfg)


def main() -> None:
    fire.Fire(
        {
            "train": train,
            "export": export,
            "onnx_to_tensorrt": onnx_to_tensorrt,
            "setup_triton": setup_triton,
            "infer": infer,
            "import_data": import_data,
            "run_triton": run_triton,
        }
    )


if __name__ == "__main__":
    main()
