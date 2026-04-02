"""
src/cuda/numba_ops.py

High-performance GPU operations implemented with Python Numba CUDA JIT.
Provides drop-in NumPy-compatible functions that execute on the GPU,
returning CuPy/NumPy arrays.

Supported operations
--------------------
- matrix_multiply   : batched GEMM via @cuda.jit tiled kernel
- relu              : element-wise ReLU
- softmax           : row-wise softmax
- layer_norm        : per-sample layer normalisation
- conv2d_simple     : simple 2-D convolution (no padding, stride=1)
"""

from __future__ import annotations

import math

import numpy as np
from numba import cuda, float32

# ── Kernel configuration ──────────────────────────────────────────────────
TILE = 16  # tile size for matrix multiplication kernels


# ─────────────────────────────────────────────────────────────────────────────
# Tiled matrix-multiply kernel
# ─────────────────────────────────────────────────────────────────────────────


@cuda.jit
def _matmul_kernel(A, B, C):
    """Compute C = A @ B using shared-memory tiles."""
    sA = cuda.shared.array(shape=(TILE, TILE), dtype=float32)
    sB = cuda.shared.array(shape=(TILE, TILE), dtype=float32)

    # cuda.grid(2) returns (x_global, y_global); x → column, y → row
    col, row = cuda.grid(2)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y

    acc = float32(0.0)
    n_tiles = (A.shape[1] + TILE - 1) // TILE

    for t in range(n_tiles):
        # Load tile of A
        a_col = t * TILE + tx
        sA[ty, tx] = A[row, a_col] if (row < A.shape[0] and a_col < A.shape[1]) else float32(0.0)

        # Load tile of B
        b_row = t * TILE + ty
        sB[ty, tx] = B[b_row, col] if (b_row < B.shape[0] and col < B.shape[1]) else float32(0.0)

        cuda.syncthreads()

        for k in range(TILE):
            acc += sA[ty, k] * sB[k, tx]

        cuda.syncthreads()

    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = acc


# ─────────────────────────────────────────────────────────────────────────────
# ReLU kernel
# ─────────────────────────────────────────────────────────────────────────────


@cuda.jit
def _relu_kernel(data):
    idx = cuda.grid(1)
    if idx < data.size:
        val = data.flat[idx]
        data.flat[idx] = val if val > 0.0 else float32(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Softmax kernel  (one block per row)
# ─────────────────────────────────────────────────────────────────────────────


@cuda.jit
def _softmax_kernel(data):
    """Row-wise softmax; each block processes one row."""
    row = cuda.blockIdx.x
    tx = cuda.threadIdx.x
    n = data.shape[1]

    # Shared memory: max-value reduction then sum reduction
    s_max = cuda.shared.array(shape=1, dtype=float32)
    s_sum = cuda.shared.array(shape=1, dtype=float32)

    if tx == 0:
        s_max[0] = data[row, 0]
        for c in range(1, n):
            if data[row, c] > s_max[0]:
                s_max[0] = data[row, c]
        s_sum[0] = float32(0.0)
        for c in range(n):
            data[row, c] = math.exp(data[row, c] - s_max[0])
            s_sum[0] += data[row, c]
        for c in range(n):
            data[row, c] /= s_sum[0]

    cuda.syncthreads()


# ─────────────────────────────────────────────────────────────────────────────
# Layer normalisation kernel  (one block per sample)
# ─────────────────────────────────────────────────────────────────────────────


@cuda.jit
def _layer_norm_kernel(data, gamma, beta, eps):
    """
    Per-row (per-sample) layer normalisation.
    data  : (N, D) float32
    gamma : (D,)   float32  – scale
    beta  : (D,)   float32  – shift
    """
    row = cuda.blockIdx.x
    D = data.shape[1]

    s_mean = cuda.shared.array(shape=1, dtype=float32)
    s_var = cuda.shared.array(shape=1, dtype=float32)

    if cuda.threadIdx.x == 0:
        # Compute mean
        mean = float32(0.0)
        for d in range(D):
            mean += data[row, d]
        mean /= D
        s_mean[0] = mean

        # Compute variance
        var = float32(0.0)
        for d in range(D):
            diff = data[row, d] - mean
            var += diff * diff
        var /= D
        s_var[0] = var

    cuda.syncthreads()

    # Normalise – single thread iterates over all D dimensions
    if cuda.threadIdx.x == 0:
        inv_std = float32(1.0) / math.sqrt(s_var[0] + float32(eps))
        for d in range(D):
            norm = (data[row, d] - s_mean[0]) * inv_std
            data[row, d] = gamma[d] * norm + beta[d]


# ─────────────────────────────────────────────────────────────────────────────
# Public Python-level API
# ─────────────────────────────────────────────────────────────────────────────


def matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    GPU matrix multiplication C = A @ B.

    Parameters
    ----------
    a : np.ndarray, shape (M, K), dtype float32
    b : np.ndarray, shape (K, N), dtype float32

    Returns
    -------
    np.ndarray, shape (M, N), dtype float32
    """
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)

    M, K = a.shape
    _K, N = b.shape
    if K != _K:
        raise ValueError(f"Shape mismatch: ({M},{K}) @ ({_K},{N})")

    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_c = cuda.device_array((M, N), dtype=np.float32)

    threads = (TILE, TILE)
    blocks = (math.ceil(N / TILE), math.ceil(M / TILE))
    _matmul_kernel[blocks, threads](d_a, d_b, d_c)

    return d_c.copy_to_host()


def relu(data: np.ndarray) -> np.ndarray:
    """Element-wise ReLU activation (GPU)."""
    data = np.ascontiguousarray(data, dtype=np.float32)
    d_data = cuda.to_device(data)

    threads = 256
    blocks = math.ceil(data.size / threads)
    _relu_kernel[blocks, threads](d_data)

    return d_data.copy_to_host().reshape(data.shape)


def softmax(data: np.ndarray) -> np.ndarray:
    """
    Row-wise softmax (GPU).

    Parameters
    ----------
    data : np.ndarray, shape (N, C), dtype float32
    """
    data = np.ascontiguousarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = data[np.newaxis, :]
        squeeze = True
    else:
        squeeze = False

    d_data = cuda.to_device(data)
    _softmax_kernel[data.shape[0], 1](d_data)
    result = d_data.copy_to_host()

    return result[0] if squeeze else result


def layer_norm(
    data: np.ndarray,
    gamma: np.ndarray | None = None,
    beta: np.ndarray | None = None,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Per-sample layer normalisation (GPU).

    Parameters
    ----------
    data  : (N, D) float32
    gamma : (D,) float32 scale  – defaults to all-ones
    beta  : (D,) float32 shift  – defaults to all-zeros
    eps   : small constant for numerical stability
    """
    data = np.ascontiguousarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = data[np.newaxis, :]
        squeeze = True
    else:
        squeeze = False

    N, D = data.shape
    if gamma is None:
        gamma = np.ones(D, dtype=np.float32)
    if beta is None:
        beta = np.zeros(D, dtype=np.float32)

    gamma = np.ascontiguousarray(gamma, dtype=np.float32)
    beta = np.ascontiguousarray(beta, dtype=np.float32)

    d_data = cuda.to_device(data)
    d_gamma = cuda.to_device(gamma)
    d_beta = cuda.to_device(beta)

    _layer_norm_kernel[N, 1](d_data, d_gamma, d_beta, np.float32(eps))
    result = d_data.copy_to_host()

    return result[0] if squeeze else result
