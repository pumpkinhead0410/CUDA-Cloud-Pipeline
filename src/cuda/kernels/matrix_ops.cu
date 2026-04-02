/**
 * matrix_ops.cu
 *
 * CUDA kernels for high-performance matrix operations used in ML inference.
 * Implements tiled GEMM (General Matrix Multiply) with shared memory to
 * maximise memory bandwidth utilisation on GPU hardware.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include "matrix_ops.h"

#define TILE_SIZE 32

/**
 * Tiled matrix multiplication kernel.
 *
 * Each thread block handles a TILE_SIZE × TILE_SIZE output tile.
 * Shared-memory staging eliminates redundant global memory reads.
 *
 * C = alpha * A * B + beta * C   (SGEMM-style signature)
 */
__global__ void tiledMatMulKernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float*       __restrict__ C,
                                   int M, int N, int K,
                                   float alpha, float beta)
{
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        /* Load tile of A */
        int aCol = t * TILE_SIZE + threadIdx.x;
        tileA[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;

        /* Load tile of B */
        int bRow = t * TILE_SIZE + threadIdx.y;
        tileB[threadIdx.y][threadIdx.x] =
            (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k)
            acc += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

/**
 * Element-wise ReLU activation kernel.
 */
__global__ void reluKernel(float* __restrict__ data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        data[idx] = fmaxf(0.0f, data[idx]);
}

/**
 * Softmax kernel (numerically stable, single-block for small last-dim).
 */
__global__ void softmaxKernel(float* __restrict__ data, int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    float* rowPtr = data + row * cols;

    /* Find max for numerical stability */
    float maxVal = rowPtr[0];
    for (int c = 1; c < cols; ++c)
        maxVal = fmaxf(maxVal, rowPtr[c]);

    /* Compute exp and sum */
    float sum = 0.0f;
    for (int c = 0; c < cols; ++c) {
        rowPtr[c] = expf(rowPtr[c] - maxVal);
        sum += rowPtr[c];
    }

    /* Normalise */
    for (int c = 0; c < cols; ++c)
        rowPtr[c] /= sum;
}

/* ── Host-side launcher wrappers ────────────────────────────────────────── */

extern "C" {

void launchTiledMatMul(const float* A, const float* B, float* C,
                       int M, int N, int K,
                       float alpha, float beta,
                       cudaStream_t stream)
{
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);
    tiledMatMulKernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
}

void launchRelu(float* data, int n, cudaStream_t stream)
{
    int block = 256;
    int grid  = (n + block - 1) / block;
    reluKernel<<<grid, block, 0, stream>>>(data, n);
}

void launchSoftmax(float* data, int rows, int cols, cudaStream_t stream)
{
    softmaxKernel<<<rows, 1, 0, stream>>>(data, rows, cols);
}

} /* extern "C" */
