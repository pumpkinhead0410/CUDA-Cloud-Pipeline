#pragma once
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Tiled GEMM: C = alpha * A * B + beta * C
 * Matrices are row-major.  M×K × K×N → M×N.
 */
void launchTiledMatMul(const float* A, const float* B, float* C,
                       int M, int N, int K,
                       float alpha, float beta,
                       cudaStream_t stream);

/** In-place ReLU on a flat array of n floats. */
void launchRelu(float* data, int n, cudaStream_t stream);

/** In-place row-wise softmax on a (rows × cols) row-major matrix. */
void launchSoftmax(float* data, int rows, int cols, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
