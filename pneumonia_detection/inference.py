from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image

try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    import tensorrt as trt
except Exception:
    trt = None

try:
    import pycuda.driver as cuda
except Exception:
    cuda = None

try:
    from tritonclient.http import (
        InferenceServerClient,
        InferInput,
        InferRequestedOutput,
    )
except Exception:
    InferenceServerClient = None
    InferInput = None
    InferRequestedOutput = None


def _sigmoid(arg: float) -> float:
    if arg >= 0:
        exp = np.exp(-arg)
        return float(1.0 / (1.0 + exp))
    exp = np.exp(arg)
    return float(exp / (1.0 + exp))


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def _preprocess(image_path: Path, img_size: int, input_channels: int) -> np.ndarray:
    mode = "L" if input_channels == 1 else "RGB"
    img = Image.open(image_path).convert(mode).resize((img_size, img_size))

    img_norm = np.asarray(img, dtype=np.float32) / 255.0
    if input_channels == 1:
        return img_norm[None, None, :, :]

    img_norm = np.transpose(img_norm, (2, 0, 1))
    return img_norm[None, :, :, :]


def _infer_onnx(img: np.ndarray, onnx_path: Path) -> float:
    if ort is None:
        raise RuntimeError(
            "onnxruntime is not installed. Install: `pip install onnxruntime`."
        )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    out = session.run(None, {input_name: img})[0]
    return float(np.asarray(out).squeeze())


def _load_trt_engine(engine_path: Path) -> "trt.ICudaEngine":
    if trt is None:
        raise RuntimeError("tensorrt is not installed.")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as file, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(file.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
    return engine


def _find_bindings(engine: "trt.ICudaEngine") -> Tuple[int, int]:
    input_idx: Optional[int] = None
    output_idx: Optional[int] = None
    for bind in range(engine.num_bindings):
        name = engine.get_binding_name(bind)
        if name == "input":
            input_idx = bind
        elif name == "output":
            output_idx = bind
    if input_idx is not None and output_idx is not None:
        return input_idx, output_idx
    return 0, 1


def _infer_tensorrt(img: np.ndarray, engine_path: Path) -> float:
    if trt is None or cuda is None:
        raise RuntimeError("TensorRT backend requires tensorrt + pycuda.")

    engine = _load_trt_engine(engine_path)
    context = engine.create_execution_context()
    input_idx, output_idx = _find_bindings(engine)

    try:
        context.set_binding_shape(input_idx, tuple(img.shape))
    except Exception:
        pass

    host_input = img.astype(np.float32, copy=False)

    out_shape = tuple(context.get_binding_shape(output_idx))
    if any(dim < 0 for dim in out_shape):
        out_shape = (img.shape[0], 1)

    host_output = np.empty(out_shape, dtype=np.float32)

    d_input = cuda.mem_alloc(host_input.nbytes)
    d_output = cuda.mem_alloc(host_output.nbytes)

    bindings = [0] * engine.num_bindings
    bindings[input_idx] = int(d_input)
    bindings[output_idx] = int(d_output)

    stream = cuda.Stream()
    cuda.memcpy_htod_async(d_input, host_input, stream)

    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    cuda.memcpy_dtoh_async(host_output, d_output, stream)
    stream.synchronize()

    return float(host_output.squeeze())


def _normalize_triton_url(url: str) -> str:
    url = url.strip()
    if url.startswith("http://"):
        return url.removeprefix("http://")
    if url.startswith("https://"):
        return url.removeprefix("https://")
    return url


def _infer_triton(
    img: np.ndarray,
    *,
    url: str,
    model_name: str,
    input_name: str,
    output_name: str,
) -> float:
    if (
        InferenceServerClient is None
        or InferInput is None
        or InferRequestedOutput is None
    ):
        raise RuntimeError("Install Triton client: `pip install tritonclient[http]`.")

    client = InferenceServerClient(url=_normalize_triton_url(url))

    inp = InferInput(input_name, img.shape, "FP32")
    inp.set_data_from_numpy(img)

    out = InferRequestedOutput(output_name)
    resp = client.infer(model_name=model_name, inputs=[inp], outputs=[out])

    out = resp.as_numpy(output_name)
    return float(np.asarray(out).squeeze())


def infer_one(cfg: DictConfig) -> float:
    image_path = OmegaConf.select(cfg, "inference.img_path")
    image_path = _resolve(Path(str(image_path)))
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    backend = str(OmegaConf.select(cfg, "inference.backend")).lower()

    img_size = int(OmegaConf.select(cfg, "inference.img_size"))
    input_channels = int(OmegaConf.select(cfg, "inference.input_channels"))
    img = _preprocess(image_path, img_size, input_channels)

    if backend == "onnx":
        onnx_path = OmegaConf.select(cfg, "inference.onnx_path")
        model_path = _resolve(Path(str(onnx_path)))
        logit = _infer_onnx(img, model_path)
        return _sigmoid(logit)

    if backend == "tensorrt":
        engine_path = OmegaConf.select(cfg, "inference.engine_path")
        model_path = _resolve(Path(str(engine_path)))
        logit = _infer_tensorrt(img, model_path)
        return _sigmoid(logit)

    if backend == "triton":
        url = str(OmegaConf.select(cfg, "inference.triton_url"))
        model_name = str(OmegaConf.select(cfg, "inference.triton_model_name"))
        input_name = str(OmegaConf.select(cfg, "inference.triton_input_name"))
        output_name = str(OmegaConf.select(cfg, "inference.triton_output_name"))
        logit = _infer_triton(
            img,
            url=url,
            model_name=model_name,
            input_name=input_name,
            output_name=output_name,
        )
        return _sigmoid(logit)

    raise ValueError("inference.backend must be one of: onnx, tensorrt, triton")


def main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(config_path="../configs", config_name="config", version_base="1.3")
    def _main(cfg: DictConfig) -> None:
        img_path = OmegaConf.select(cfg, "inference.img_path")
        if not img_path:
            raise ValueError("Set inference.img_path=...")

        prob = infer_one(cfg, Path(str(img_path)))
        print(f"Predicted probability of pneumonia: {prob:.4f}")

    _main()


if __name__ == "__main__":
    main()
