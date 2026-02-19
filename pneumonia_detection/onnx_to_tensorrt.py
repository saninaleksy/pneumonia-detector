from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _resolve(p: Path) -> Path:
    return p.expanduser().resolve()


def convert_onnx_to_tensorrt(
    onnx_path: Path,
    engine_path: Path,
    *,
    workspace_size: int,
    fp16: bool,
    int8: bool,
) -> Path:
    onnx_path = _resolve(onnx_path)
    engine_path = _resolve(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    trtexec = shutil.which("trtexec")
    if trtexec is None:
        raise RuntimeError(
            "trtexec not found. Install TensorRT and ensure it is in PATH."
        )

    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--workspace={workspace_size}",
        "--explicitBatch",
        "--verbose",
    ]

    if fp16:
        cmd.append("--fp16")
    if int8:
        cmd.append("--int8")

    subprocess.run(cmd, check=True)
    return engine_path
