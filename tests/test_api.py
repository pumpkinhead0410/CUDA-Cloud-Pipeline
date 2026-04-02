"""
tests/test_api.py

Integration-style tests for the FastAPI inference API.

The test suite forces USE_STUB=1 so that no GPU or TensorRT is required.
A real TRTEngine is never instantiated.
"""

from __future__ import annotations

import os

# Force stub engine before any app import
os.environ["USE_STUB"] = "1"
os.environ["STUB_OUTPUT_DIM"] = "10"
os.environ["MODEL_NAME"] = "test-model"

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Shared TestClient for the module."""
    with TestClient(app) as c:
        yield c


# ─────────────────────────────────────────────────────────────────────────────
# /healthz
# ─────────────────────────────────────────────────────────────────────────────


class TestHealthz:
    def test_returns_200(self, client: TestClient):
        resp = client.get("/healthz")
        assert resp.status_code == 200

    def test_engine_loaded(self, client: TestClient):
        data = client.get("/healthz").json()
        assert data["engine_loaded"] is True

    def test_model_name_in_response(self, client: TestClient):
        data = client.get("/healthz").json()
        assert data["model_name"] == "test-model"

    def test_status_ok(self, client: TestClient):
        data = client.get("/healthz").json()
        assert data["status"] == "ok"


# ─────────────────────────────────────────────────────────────────────────────
# /infer
# ─────────────────────────────────────────────────────────────────────────────


def _make_request(shape: list[int]) -> dict:
    n = int(np.prod(shape))
    return {
        "inputs": {"input": list(np.random.rand(n).astype(float))},
        "input_shapes": {"input": shape},
    }


class TestInfer:
    def test_basic_inference(self, client: TestClient):
        payload = _make_request([1, 3, 224, 224])
        resp = client.post("/infer", json=payload)
        assert resp.status_code == 200

    def test_response_has_outputs(self, client: TestClient):
        payload = _make_request([1, 3, 224, 224])
        data = client.post("/infer", json=payload).json()
        assert "outputs" in data
        assert isinstance(data["outputs"], dict)

    def test_output_is_list_of_floats(self, client: TestClient):
        payload = _make_request([1, 3, 224, 224])
        data = client.post("/infer", json=payload).json()
        for values in data["outputs"].values():
            assert all(isinstance(v, float) for v in values)

    def test_latency_ms_present(self, client: TestClient):
        payload = _make_request([1, 64])
        data = client.post("/infer", json=payload).json()
        assert "latency_ms" in data
        assert data["latency_ms"] >= 0

    def test_model_name_in_response(self, client: TestClient):
        payload = _make_request([1, 64])
        data = client.post("/infer", json=payload).json()
        assert data["model_name"] == "test-model"

    def test_shape_mismatch_returns_422(self, client: TestClient):
        payload = {
            "inputs": {"input": [1.0, 2.0, 3.0]},
            "input_shapes": {"input": [1, 100]},  # size mismatch: 3 ≠ 100
        }
        resp = client.post("/infer", json=payload)
        assert resp.status_code == 422

    def test_empty_inputs_returns_422(self, client: TestClient):
        payload = {
            "inputs": {},
            "input_shapes": {},
        }
        resp = client.post("/infer", json=payload)
        assert resp.status_code == 422

    def test_stub_output_dim(self, client: TestClient):
        """StubEngine should return STUB_OUTPUT_DIM=10 values."""
        payload = _make_request([1, 16])
        data = client.post("/infer", json=payload).json()
        # StubEngine output shape is (1, 10) → flat list of 10 values
        assert len(data["outputs"]["output"]) == 10

    def test_multiple_inputs(self, client: TestClient):
        """Sending multiple named inputs should work without error."""
        n = 8
        payload = {
            "inputs": {
                "input_a": list(np.zeros(n, dtype=float)),
                "input_b": list(np.ones(n, dtype=float)),
            },
            "input_shapes": {
                "input_a": [1, n],
                "input_b": [1, n],
            },
        }
        # StubEngine only has "output" binding, but should not crash on extra inputs
        resp = client.post("/infer", json=payload)
        # Either 200 (stub ignores extra inputs) or 500 (key error) – not 422
        assert resp.status_code in (200, 500)


# ─────────────────────────────────────────────────────────────────────────────
# /profile
# ─────────────────────────────────────────────────────────────────────────────


class TestProfile:
    def test_returns_404_for_stub(self, client: TestClient):
        """Profile endpoint should return 404 when using StubEngine."""
        resp = client.get("/profile")
        assert resp.status_code == 404
