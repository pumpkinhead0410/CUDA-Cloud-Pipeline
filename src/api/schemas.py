"""
src/api/schemas.py

Pydantic request / response models for the inference REST API.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class InferenceRequest(BaseModel):
    """Single-sample inference request."""

    inputs: dict[str, list[float]] = Field(
        ...,
        description="Named input tensors.  Key = binding name, "
        "value = flat float list (row-major).",
        examples=[{"input": [0.1, 0.2, 0.3]}],
    )
    input_shapes: dict[str, list[int]] = Field(
        ...,
        description='Shape of each input tensor, e.g. {"input": [1, 3, 224, 224]}.',
    )

    @field_validator("inputs")
    @classmethod
    def inputs_not_empty(cls, v: dict[str, list[float]]) -> dict[str, list[float]]:
        if not v:
            raise ValueError("inputs must not be empty")
        return v

    @field_validator("input_shapes")
    @classmethod
    def shapes_not_empty(cls, v: dict[str, list[int]]) -> dict[str, list[int]]:
        if not v:
            raise ValueError("input_shapes must not be empty")
        return v


class InferenceResponse(BaseModel):
    """Inference result returned to the caller."""

    outputs: dict[str, list[float]] = Field(
        ...,
        description="Named output tensors as flat float lists.",
    )
    latency_ms: float = Field(
        ...,
        description="End-to-end server-side inference latency in milliseconds.",
    )
    model_name: str = Field(..., description="Name/version of the loaded model.")


class HealthResponse(BaseModel):
    """Liveness / readiness probe response."""

    status: str = Field(default="ok")
    gpu_available: bool
    engine_loaded: bool
    model_name: str


class ProfileResponse(BaseModel):
    """Per-layer TRT profiling report."""

    layers: dict[str, float] = Field(
        ...,
        description="Mapping of layer name → cumulative latency in ms.",
    )
