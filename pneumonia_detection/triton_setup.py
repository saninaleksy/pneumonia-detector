from __future__ import annotations

import logging
import shutil
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


TRITON_CONFIG_TEMPLATE = """name: "{model_name}"
platform: "{platform}"
max_batch_size: {max_batch_size}

input [
  {{
    name: "{input_name}"
    data_type: TYPE_FP32
    dims: [ {channels}, {height}, {width} ]
  }}
]

output [
  {{
    name: "{output_name}"
    data_type: TYPE_FP32
    dims: [ {output_dims} ]
  }}
]

instance_group [
  {{
    count: {instance_count}
    kind: {instance_kind}
  }}
]

{dynamic_batching_block}
"""


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def _dynamic_batching_block(cfg: DictConfig) -> str:
    enabled = bool(OmegaConf.select(cfg, "triton.dynamic_batching.enabled"))
    if not enabled:
        return ""

    max_delay = int(
        OmegaConf.select(cfg, "triton.dynamic_batching.max_queue_delay_microseconds")
    )
    return f"""dynamic_batching {{
  max_queue_delay_microseconds: {max_delay}
}}"""


def create_triton_model_repository(cfg: DictConfig) -> None:
    triton_conf = cfg.triton

    root = _project_root()
    repo_dir = _resolve(root / Path(str(triton_conf.repository_dir)))

    model_name = str(triton_conf.model_name)
    model_version = str(triton_conf.model_version)

    model_dir = repo_dir / model_name
    version_dir = model_dir / model_version
    version_dir.mkdir(parents=True, exist_ok=True)

    backend = str(OmegaConf.select(cfg, "triton.backend")).lower()
    if backend == "onnx":
        platform = "onnxruntime_onnx"
        src = _resolve(Path(str(triton_conf.onnx_source)))
        dst = version_dir / "model.onnx"
    elif backend == "tensorrt":
        platform = "tensorrt_plan"
        src = _resolve(root / Path(str(triton_conf.onnx_source)))
        dst = version_dir / "model.plan"
    else:
        raise ValueError("triton.backend must be one of: onnx, tensorrt")

    if not src.is_file():
        raise FileNotFoundError(f"Model file not found: {src}")

    shutil.copy2(src, dst)
    log.info("Copied model file to: %s", dst)

    kind = str(OmegaConf.select(cfg, "triton.instance_group.kind")).lower()
    instance_kind = "KIND_GPU" if kind == "gpu" else "KIND_CPU"
    instance_count = int(OmegaConf.select(cfg, "triton.instance_group.count"))

    input_name = str(OmegaConf.select(cfg, "triton.input.name"))
    channels = int(OmegaConf.select(cfg, "triton.input.channels"))
    height = int(OmegaConf.select(cfg, "triton.input.height"))
    width = int(OmegaConf.select(cfg, "triton.input.width"))

    output_name = str(OmegaConf.select(cfg, "triton.output.name"))
    output_dims = int(OmegaConf.select(cfg, "triton.output.dims"))

    max_batch_size = int(OmegaConf.select(cfg, "triton.max_batch_size"))
    dyn_block = _dynamic_batching_block(cfg)

    config_content = TRITON_CONFIG_TEMPLATE.format(
        model_name=model_name,
        platform=platform,
        max_batch_size=max_batch_size,
        input_name=input_name,
        channels=channels,
        height=height,
        width=width,
        output_name=output_name,
        output_dims=output_dims,
        instance_count=instance_count,
        instance_kind=instance_kind,
        dynamic_batching_block=dyn_block,
    )

    (model_dir / "config.pbtxt").write_text(config_content, encoding="utf-8")

    labels = OmegaConf.select(cfg, "triton.labels")
    (model_dir / "labels.txt").write_text("\n".join(list(labels)), encoding="utf-8")

    log.info("Triton model repository created at: %s", model_dir)
