// RMSNorm forward + backward kernels.
//
// RMSNorm is LayerNorm without the mean-subtract and without the
// beta bias:
//     y_j = (x_j / sqrt(mean_k(x_k^2) + eps)) * gamma_j
//
// One block per row, 256 threads cooperating on the per-row reductions
// via `block_reduce_sum_bcast`. Backward recomputes rstd from x rather
// than caching across the FFI boundary.
//
// I/O contract:
//   forward:  x [R,C], gamma [C]  ->  out [R,C]
//   backward: x [R,C], gamma [C], gO [R,C]  ->
//               gX  [R,C]  (written; caller allocates)
//               gG  [C]    (atomicAdd; caller pre-allocates)

#pragma once
#include "common.cuh"

__global__ void rmsnorm_fwd(const float *x, const float *gamma,
                            float *out, int R, int C, float eps) {
  int row = blockIdx.x;
  if (row >= R) return;

  __shared__ float smem[32];
  __shared__ float s_inv_rms;

  const float *xrow = x + row * C;
  float *yrow = out + row * C;

  // Pass 1: mean of squares.
  float local_sq = 0.0f;
  for (int j = threadIdx.x; j < C; j += blockDim.x) {
    float v = xrow[j];
    local_sq += v * v;
  }
  float total_sq = block_reduce_sum_bcast(local_sq, smem);
  if (threadIdx.x == 0) s_inv_rms = rsqrtf(total_sq / (float)C + eps);
  __syncthreads();
  float inv_rms = s_inv_rms;

  // Pass 2: scale.
  for (int j = threadIdx.x; j < C; j += blockDim.x) {
    yrow[j] = xrow[j] * inv_rms * gamma[j];
  }
}

__global__ void rmsnorm_bwd(const float *x, const float *gamma,
                            const float *gO, float *gX, float *gGamma,
                            int R, int C, float eps) {
  int row = blockIdx.x;
  if (row >= R) return;

  __shared__ float smem[32];
  __shared__ float s_inv_rms, s_dot;

  const float *xrow = x + row * C;
  const float *grow = gO + row * C;
  float *gxrow = gX + row * C;

  // Recompute inv_rms.
  float local_sq = 0.0f;
  for (int j = threadIdx.x; j < C; j += blockDim.x) {
    float v = xrow[j];
    local_sq += v * v;
  }
  float total_sq = block_reduce_sum_bcast(local_sq, smem);
  if (threadIdx.x == 0) s_inv_rms = rsqrtf(total_sq / (float)C + eps);
  __syncthreads();
  float inv_rms = s_inv_rms;

  // Accumulate dGamma and the single per-row reduction we need for dX:
  //   dot = sum_k( gO_k * gamma_k * x_k )
  float local_dot = 0.0f;
  for (int j = threadIdx.x; j < C; j += blockDim.x) {
    float gj = grow[j];
    float xj = xrow[j];
    local_dot += gj * gamma[j] * xj;
    // dGamma_j += gO_j * (x_j * inv_rms) — accumulate into caller's grad.
    atomicAdd(&gGamma[j], gj * xj * inv_rms);
  }
  float dot = block_reduce_sum_bcast(local_dot, smem);
  if (threadIdx.x == 0) s_dot = dot;
  __syncthreads();
  float dot_bc = s_dot;

  // Input gradient:
  //   dX_j = inv_rms * (gO_j * gamma_j - x_j * inv_rms^2 * dot / C)
  float invC = 1.0f / (float)C;
  float inv_rms3 = inv_rms * inv_rms * inv_rms;
  for (int j = threadIdx.x; j < C; j += blockDim.x) {
    float gj = grow[j];
    float xj = xrow[j];
    gxrow[j] = gj * gamma[j] * inv_rms - xj * inv_rms3 * dot_bc * invC;
  }
}
