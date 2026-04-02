"""
src/api/main.py

FastAPI application that serves TensorRT (or stub) inference over HTTP.

Endpoints
---------
GET  /healthz          – liveness / readiness probe
POST /infer            – run inference on a single sample
GET  /profile          – per-layer TRT profiling summary

Environment variables
---------------------
ENGINE_PATH   : path to the serialised TRT engine file
                default: "models/model.engine"
MODEL_NAME    : human-readable model identifier
                default: "default"
USE_STUB      : "1" to force the StubEngine (CPU-only / CI)
                default: "0"
STUB_OUTPUT_DIM : output dimension for StubEngine
                default: "1000"
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    ProfileResponse,
)
from src.inference.trt_engine import StubEngine, TRTEngine

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuration from environment ────────────────────────────────────────
ENGINE_PATH = os.getenv("ENGINE_PATH", "models/model.engine")
MODEL_NAME = os.getenv("MODEL_NAME", "default")
USE_STUB = os.getenv("USE_STUB", "0") == "1"
STUB_OUTPUT_DIM = int(os.getenv("STUB_OUTPUT_DIM", "1000"))

# ── Global engine handle ──────────────────────────────────────────────────
_engine: Any = None


# ── App lifecycle ─────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    global _engine
    if USE_STUB or not os.path.isfile(ENGINE_PATH):
        logger.warning(
            "Using StubEngine (USE_STUB=%s, engine file exists=%s)",
            USE_STUB,
            os.path.isfile(ENGINE_PATH),
        )
        _engine = StubEngine({"output": (1, STUB_OUTPUT_DIM)})
    else:
        try:
            _engine = TRTEngine(ENGINE_PATH, enable_profiling=True)
            logger.info("TRTEngine loaded from %s", ENGINE_PATH)
        except Exception as exc:
            logger.error("Failed to load TRTEngine: %s – falling back to stub", exc)
            _engine = StubEngine({"output": (1, STUB_OUTPUT_DIM)})
    yield
    # Cleanup
    if hasattr(_engine, "close"):
        _engine.close()


app = FastAPI(
    title="CUDA Cloud Inference API",
    description=(
        "High-performance ML inference backed by TensorRT GPU acceleration, "
        "deployed on AWS EKS via Kubernetes."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────


@app.get("/healthz", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Liveness and readiness probe."""
    try:
        import pycuda.driver as cuda  # type: ignore

        gpu_ok = cuda.Device.count() > 0
    except Exception:
        gpu_ok = False

    return HealthResponse(
        status="ok",
        gpu_available=gpu_ok,
        engine_loaded=_engine is not None,
        model_name=MODEL_NAME,
    )


@app.post(
    "/infer",
    response_model=InferenceResponse,
    status_code=status.HTTP_200_OK,
    tags=["inference"],
)
async def infer(request: InferenceRequest) -> InferenceResponse:
    """
    Run inference on a single sample.

    The caller provides flat float arrays for each input binding together
    with the corresponding shapes.  The server reconstructs the NumPy
    arrays, forwards them to the engine, and returns the flat output arrays.
    """
    if _engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not ready.",
        )

    # Reconstruct numpy arrays from flat lists + shapes
    np_inputs: dict[str, np.ndarray] = {}
    for name, flat in request.inputs.items():
        shape = tuple(request.input_shapes[name])
        try:
            np_inputs[name] = np.array(flat, dtype=np.float32).reshape(shape)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Cannot reshape input '{name}' to {shape}: {exc}",
            ) from exc

    t0 = time.perf_counter()
    try:
        raw_outputs = _engine.infer(np_inputs)
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc}",
        ) from exc
    latency_ms = (time.perf_counter() - t0) * 1_000

    outputs = {k: v.ravel().tolist() for k, v in raw_outputs.items()}

    return InferenceResponse(
        outputs=outputs,
        latency_ms=round(latency_ms, 3),
        model_name=MODEL_NAME,
    )


@app.get("/profile", response_model=ProfileResponse, tags=["ops"])
async def profile() -> ProfileResponse:
    """Return per-layer TRT profiling data (requires TRTEngine with profiling)."""
    if not isinstance(_engine, TRTEngine):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Profiling is only available when TRTEngine is active.",
        )
    summary = _engine.profiling_summary() or {}
    return ProfileResponse(layers=summary)
