"""
tests/test_numba_ops.py

Unit tests for src/cuda/numba_ops.py

These tests run on CPU using NumPy reference implementations to verify
numerical correctness.  On machines with a compatible NVIDIA GPU the
Numba kernels will run on the GPU; on CPU-only machines Numba falls back
to a CUDA simulator when NUMBA_ENABLE_CUDASIM=1 is set.

Set the environment variable  NUMBA_ENABLE_CUDASIM=1  to run without a GPU.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

# ── CUDA simulator shim ───────────────────────────────────────────────────
# When there is no GPU available we rely on Numba's CUDA simulator.
# This must be set BEFORE numba is imported.
if os.getenv("NUMBA_ENABLE_CUDASIM") is None and os.getenv("USE_STUB") == "1":
    os.environ["NUMBA_ENABLE_CUDASIM"] = "1"

try:
    from src.cuda.numba_ops import layer_norm, matrix_multiply, relu, softmax

    CUDA_OPS_AVAILABLE = True
except Exception:
    CUDA_OPS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not CUDA_OPS_AVAILABLE,
    reason="CUDA ops not available (no GPU and NUMBA_ENABLE_CUDASIM not set)",
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _np_softmax(x: np.ndarray) -> np.ndarray:
    """Reference row-wise softmax."""
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _np_layer_norm(
    x: np.ndarray,
    gamma: np.ndarray | None = None,
    beta: np.ndarray | None = None,
    eps: float = 1e-5,
) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    std = np.sqrt(x.var(axis=-1, keepdims=True) + eps)
    norm = (x - mean) / std
    if gamma is not None:
        norm = norm * gamma
    if beta is not None:
        norm = norm + beta
    return norm


# ─────────────────────────────────────────────────────────────────────────────
# matrix_multiply
# ─────────────────────────────────────────────────────────────────────────────


class TestMatrixMultiply:
    def test_square_matrices(self):
        rng = np.random.default_rng(0)
        A = rng.random((32, 32)).astype(np.float32)
        B = rng.random((32, 32)).astype(np.float32)
        expected = A @ B
        result = matrix_multiply(A, B)
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)

    def test_non_square_matrices(self):
        rng = np.random.default_rng(1)
        A = rng.random((16, 64)).astype(np.float32)
        B = rng.random((64, 8)).astype(np.float32)
        expected = A @ B
        result = matrix_multiply(A, B)
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)

    def test_identity_matrix(self):
        eye = np.eye(16, dtype=np.float32)
        A = np.arange(256, dtype=np.float32).reshape(16, 16)
        result = matrix_multiply(A, eye)
        np.testing.assert_allclose(result, A, rtol=1e-5, atol=1e-5)

    def test_shape_mismatch_raises(self):
        A = np.ones((4, 8), dtype=np.float32)
        B = np.ones((4, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="Shape mismatch"):
            matrix_multiply(A, B)

    def test_output_shape(self):
        A = np.ones((7, 13), dtype=np.float32)
        B = np.ones((13, 5), dtype=np.float32)
        result = matrix_multiply(A, B)
        assert result.shape == (7, 5)


# ─────────────────────────────────────────────────────────────────────────────
# relu
# ─────────────────────────────────────────────────────────────────────────────


class TestRelu:
    def test_all_positive(self):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        np.testing.assert_allclose(relu(x), x)

    def test_all_negative(self):
        x = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
        np.testing.assert_allclose(relu(x), np.zeros_like(x))

    def test_mixed(self):
        x = np.array([-1.0, 0.0, 1.0, -0.5, 2.5], dtype=np.float32)
        expected = np.array([0.0, 0.0, 1.0, 0.0, 2.5], dtype=np.float32)
        np.testing.assert_allclose(relu(x), expected)

    def test_preserves_shape(self):
        x = np.random.randn(4, 8).astype(np.float32)
        assert relu(x).shape == x.shape

    def test_2d_input(self):
        x = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
        expected = np.array([[0.0, 2.0], [3.0, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(relu(x), expected)


# ─────────────────────────────────────────────────────────────────────────────
# softmax
# ─────────────────────────────────────────────────────────────────────────────


class TestSoftmax:
    def test_sums_to_one(self):
        x = np.random.randn(4, 10).astype(np.float32)
        result = softmax(x)
        np.testing.assert_allclose(result.sum(axis=-1), np.ones(4), atol=1e-5)

    def test_all_positive(self):
        x = np.random.randn(4, 10).astype(np.float32)
        assert (softmax(x) >= 0).all()

    def test_matches_numpy(self):
        x = np.random.randn(3, 5).astype(np.float32)
        expected = _np_softmax(x)
        result = softmax(x)
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_1d_input(self):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = softmax(x)
        assert result.shape == (3,)
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-5)

    def test_numerical_stability_large_values(self):
        """Softmax should not overflow with large inputs."""
        x = np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float32)
        result = softmax(x)
        assert np.all(np.isfinite(result))
        np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# layer_norm
# ─────────────────────────────────────────────────────────────────────────────


class TestLayerNorm:
    def test_zero_mean_unit_var(self):
        x = np.random.randn(8, 32).astype(np.float32) * 5 + 3
        result = layer_norm(x)
        np.testing.assert_allclose(result.mean(axis=-1), np.zeros(8), atol=1e-4)
        np.testing.assert_allclose(result.var(axis=-1), np.ones(8), atol=1e-3)

    def test_matches_numpy(self):
        rng = np.random.default_rng(42)
        x = rng.random((4, 16)).astype(np.float32)
        gamma = rng.random(16).astype(np.float32)
        beta = rng.random(16).astype(np.float32)
        expected = _np_layer_norm(x, gamma, beta)
        result = layer_norm(x, gamma, beta)
        np.testing.assert_allclose(result, expected, atol=1e-4)

    def test_1d_input(self):
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = layer_norm(x)
        assert result.shape == (4,)

    def test_preserves_shape(self):
        x = np.random.randn(6, 24).astype(np.float32)
        assert layer_norm(x).shape == x.shape
