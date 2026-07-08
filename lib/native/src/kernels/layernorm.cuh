// LayerNorm forward + backward kernels.
//
// Design mirrors the reference in dart_cuda: two-pass mean/variance
// (centered form is ~1 ulp accurate), one block per row, 256 threads
// cooperating on the column-wise reductions via `block_reduce_sum_bcast`.
//
// The backward kernel recomputes mean and rstd from `x` rather than
// caching across the FFI boundary — the extra passes are negligible
// vs. the FFI + Dart plumbing that caching would demand.
//
// I/O contract:
//   forward:  x [R,C], gamma [C], beta [C]  ->  out [R,C]
//   backward: x [R,C], gamma [C], gO [R,C]  ->
//               gX  [R,C]  (written; caller allocates)
//               gG  [C]    (atomicAdd; caller pre-allocates and typically
//                            passes an existing grad accumulator)
//               gB  [C]    (atomicAdd; same as gG)

#pragma once
#include "common.cuh"

__global__ void layernorm_fwd(const float *x, const float *gamma,
                              const float *beta, float *out, int R, int C,
                              float eps) {
  int row = blockIdx.x;
  if (row >= R) return;

  __shared__ float smem[32];
  __shared__ float s_mean, s_inv_std;

  const float *xrow = x + row * C;
  float *yrow = out + row * C;

  // Pass 1: mean.
  float local_sum = 0.0f;
  for (int j = threadIdx.x; j < C; j += blockDim.x) local_sum += xrow[j];
  float total = block_reduce_sum_bcast(local_sum, smem);
  if (threadIdx.x == 0) s_mean = total / (float)C;
  __syncthreads();
  float mean = s_mean;

  // Pass 2: variance.
  float local_sq = 0.0f;
  for (int j = threadIdx.x; j < C; j += blockDim.x) {
    float d = xrow[j] - mean;
    local_sq += d * d;
  }
  float total_sq = block_reduce_sum_bcast(local_sq, smem);
  if (threadIdx.x == 0) s_inv_std = rsqrtf(total_sq / (float)C + eps);
  __syncthreads();
  float inv_std = s_inv_std;

  // Pass 3: scale + shift.
  for (int j = threadIdx.x; j < C; j += blockDim.x) {
    float xh = (xrow[j] - mean) * inv_std;
    yrow[j] = xh * gamma[j] + beta[j];
  }
}

__global__ void layernorm_bwd(const float *x, const float *gamma,
                              const float *gO, float *gX, float *gGamma,
                              float *gBeta, int R, int C, float eps) {
  int row = blockIdx.x;
  if (row >= R) return;

  __shared__ float smem[32];
  __shared__ float s_mean, s_inv_std, s_dxh_sum, s_dxh_xh_sum;

  const float *xrow = x + row * C;
  const float *grow = gO + row * C;
  float *gxrow = gX + row * C;

  // Recompute mean.
  float local_sum = 0.0f;
  for (int j = threadIdx.x; j < C; j += blockDim.x) local_sum += xrow[j];
  float total = block_reduce_sum_bcast(local_sum, smem);
  if (threadIdx.x == 0) s_mean = total / (float)C;
  __syncthreads();
  float mean = s_mean;

  // Recompute inv_std.
  float local_sq = 0.0f;
  for (int j = threadIdx.x; j < C; j += blockDim.x) {
    float d = xrow[j] - mean;
    local_sq += d * d;
  }
  float total_sq = block_reduce_sum_bcast(local_sq, smem);
  if (threadIdx.x == 0) s_inv_std = rsqrtf(total_sq / (float)C + eps);
  __syncthreads();
  float inv_std = s_inv_std;

  // Accumulate dGamma / dBeta and the two per-row reductions needed
  // for the input gradient.
  float local_dxh = 0.0f;
  float local_dxh_xh = 0.0f;
  for (int j = threadIdx.x; j < C; j += blockDim.x) {
    float gj = grow[j];
    float gm = gamma[j];
    float xh = (xrow[j] - mean) * inv_std;
    float dxh = gj * gm;
    local_dxh += dxh;
    local_dxh_xh += dxh * xh;
    atomicAdd(&gGamma[j], gj * xh);
    atomicAdd(&gBeta[j], gj);
  }
  float DXH = block_reduce_sum_bcast(local_dxh, smem);
  if (threadIdx.x == 0) s_dxh_sum = DXH;
  __syncthreads();
  float DXHX = block_reduce_sum_bcast(local_dxh_xh, smem);
  if (threadIdx.x == 0) s_dxh_xh_sum = DXHX;
  __syncthreads();
  float dl_sum = s_dxh_sum;
  float dl_y_sum = s_dxh_xh_sum;

  // Input gradient (write-through: caller allocated gX fresh).
  float invC = 1.0f / (float)C;
  for (int j = threadIdx.x; j < C; j += blockDim.x) {
    float xh = (xrow[j] - mean) * inv_std;
    float dxh = grow[j] * gamma[j];
    gxrow[j] = inv_std * (dxh - dl_sum * invC - xh * dl_y_sum * invC);
  }
}
