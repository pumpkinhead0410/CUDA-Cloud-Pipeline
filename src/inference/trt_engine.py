"""
src/inference/trt_engine.py

TensorRT inference engine wrapper.

Responsibilities
----------------
- Load a serialised TensorRT engine from disk.
- Allocate pinned host + device memory buffers.
- Execute synchronous and asynchronous inference.
- Report per-layer profiling data when requested.

Usage
-----
    engine = TRTEngine("model.engine")
    outputs = engine.infer({"input": input_array})
    engine.close()

    # Context manager
    with TRTEngine("model.engine") as engine:
        outputs = engine.infer({"input": input_array})
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import pycuda.autoinit  # noqa: F401  – initialises the CUDA context
    import pycuda.driver as cuda  # type: ignore
    import tensorrt as trt  # type: ignore

    TRT_AVAILABLE = True
except ImportError:  # pragma: no cover
    TRT_AVAILABLE = False
    logger.warning("TensorRT / PyCUDA not available – TRTEngine will run in stub mode.")


class _SimpleProfiler:
    """Collect per-layer latency for profiling."""

    def __init__(self) -> None:
        self.layers: dict[str, float] = {}

    def report_layer_time(self, layer_name: str, ms: float) -> None:
        self.layers[layer_name] = self.layers.get(layer_name, 0.0) + ms

    def summary(self) -> dict[str, float]:
        return dict(sorted(self.layers.items(), key=lambda x: -x[1]))


class HostDeviceMem:
    """Pair of pinned host buffer and matching device buffer."""

    def __init__(self, host_mem: Any, device_mem: Any) -> None:
        self.host = host_mem
        self.device = device_mem

    def free(self) -> None:
        if TRT_AVAILABLE:
            self.host.base.free()
            self.device.free()


class TRTEngine:
    """
    Thin wrapper around a serialised TensorRT engine.

    Parameters
    ----------
    engine_path : str
        Path to the ``*.engine`` / ``*.trt`` serialised plan file.
    enable_profiling : bool
        When True attach a simple profiler to the execution context.
    """

    def __init__(self, engine_path: str, *, enable_profiling: bool = False) -> None:
        if not os.path.isfile(engine_path):
            raise FileNotFoundError(f"TRT engine not found: {engine_path}")

        if not TRT_AVAILABLE:
            raise RuntimeError(
                "TensorRT is not installed. Install tensorrt and pycuda to use TRTEngine."
            )

        self._engine_path = engine_path
        self._enable_profiling = enable_profiling
        self._runtime: Any = None
        self._engine: Any = None
        self._context: Any = None
        self._stream: Any = None
        self._buffers: dict[str, HostDeviceMem] = {}
        self._bindings: list[Any] = []

        self._load()

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def _load(self) -> None:
        trt_logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(trt_logger)

        with open(self._engine_path, "rb") as f:
            engine_bytes = f.read()

        self._engine = self._runtime.deserialize_cuda_engine(engine_bytes)
        self._context = self._engine.create_execution_context()
        self._stream = cuda.Stream()

        if self._enable_profiling:
            self._profiler = _SimpleProfiler()
            self._context.profiler = self._profiler

        self._allocate_buffers()
        logger.info("TRT engine loaded from %s", self._engine_path)

    def _allocate_buffers(self) -> None:
        self._bindings = []
        for binding in self._engine:
            shape = self._engine.get_binding_shape(binding)
            dtype = trt.nptype(self._engine.get_binding_dtype(binding))
            size = int(np.prod(shape))

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self._bindings.append(int(device_mem))
            self._buffers[binding] = HostDeviceMem(host_mem, device_mem)

    def close(self) -> None:
        """Free GPU resources."""
        for buf in self._buffers.values():
            buf.free()
        self._buffers.clear()
        self._bindings.clear()
        logger.info("TRT engine resources released.")

    # ── Inference ─────────────────────────────────────────────────────────

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Run synchronous inference.

        Parameters
        ----------
        inputs : dict mapping binding name → numpy array

        Returns
        -------
        dict mapping output binding name → numpy array
        """
        # Copy inputs to pinned memory → device
        for name, array in inputs.items():
            buf = self._buffers[name]
            np.copyto(buf.host, array.ravel())
            cuda.memcpy_htod_async(buf.device, buf.host, self._stream)

        # Execute
        self._context.execute_async_v2(bindings=self._bindings, stream_handle=self._stream.handle)

        # Copy outputs device → host
        outputs: dict[str, np.ndarray] = {}
        for binding in self._engine:
            if not self._engine.binding_is_input(binding):
                buf = self._buffers[binding]
                cuda.memcpy_dtoh_async(buf.host, buf.device, self._stream)
                outputs[binding] = np.copy(buf.host)

        self._stream.synchronize()
        return outputs

    def profiling_summary(self) -> dict[str, float] | None:
        """Return per-layer timing dict (ms) if profiling is enabled."""
        if self._enable_profiling and hasattr(self, "_profiler"):
            return self._profiler.summary()
        return None

    # ── Context manager ───────────────────────────────────────────────────

    def __enter__(self) -> TRTEngine:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ─────────────────────────────────────────────────────────────────────────────
# Stub engine for CPU-only / CI environments
# ─────────────────────────────────────────────────────────────────────────────


class StubEngine:
    """
    A no-op inference engine used in CPU-only / testing environments.
    Returns zero-filled arrays matching the requested output shapes.
    """

    def __init__(self, output_shapes: dict[str, tuple[int, ...]]) -> None:
        self._output_shapes = output_shapes

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            name: np.zeros(shape, dtype=np.float32) for name, shape in self._output_shapes.items()
        }

    def __enter__(self) -> StubEngine:
        return self

    def __exit__(self, *_: Any) -> None:
        pass

    def close(self) -> None:
        pass
