from __future__ import annotations

import subprocess
from pathlib import Path

from omegaconf import DictConfig


def run_triton_server(cfg: DictConfig) -> None:
    triton_conf = cfg.triton

    repository_dir = Path(str(triton_conf.repository_dir)).expanduser().resolve()
    image = str(triton_conf.docker_image)
    use_gpu = bool(triton_conf.use_gpu)

    cmd = [
        "docker",
        "run",
        "--rm",
    ]

    if use_gpu:
        cmd += ["--gpus", "all"]

    cmd += [
        "-p8000:8000",
        "-p8001:8001",
        "-p8002:8002",
        "-v",
        f"{repository_dir}:/models",
        image,
        "tritonserver",
        "--model-repository=/models",
    ]

    subprocess.run(cmd, check=True)
