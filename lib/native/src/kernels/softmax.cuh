// Softmax + fused cross-entropy kernels.
//
// Softmax (row-wise, block-per-row):
//   forward:  y[r,c] = exp(x[r,c] - max_c x[r,:]) / sum_c exp(...)
//   backward: dx[r,c] = (dO[r,c] - sum_c(dO[r,:] * y[r,:])) * y[r,c]
//
// Cross-entropy with integer labels (fused softmax + NLL, block-per-row):
//   forward:  loss[r] = logsumexp(x[r,:]) - x[r, target[r]]
//                     -- returns per-sample losses; Dart applies mean/sum.
//   backward: dx[r,c] = (softmax(x[r])[c] - (c == target[r])) * gLoss[r]
//
// Targets are passed as float32 pointers and rounded to int inside the
// kernel (cheaper than plumbing a separate int32 tensor type through
// the FFI). Valid up to ~16M classes, which is well beyond any realistic
// vocab size for the kinds of models this library will train.

#pragma once
#include "common.cuh"

// ---------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------

__global__ void softmax_fwd(const float *x, float *out, int R, int C) {
  int row = blockIdx.x;
  if (row >= R) return;

  __shared__ float smem[32];
  __shared__ float s_max, s_sum;

  const float *xrow = x + row * C;
  float *yrow = out + row * C;

  // Pass 1: row max (for numerical stability).
  float local_max = -1e30f;
  for (int j = threadIdx.x; j < C; j += blockDim.x)
    local_max = fmaxf(local_max, xrow[j]);
  float row_max = block_reduce_max_bcast(local_max, smem);
  if (threadIdx.x == 0) s_max = row_max;
  __syncthreads();
  float m = s_max;

  // Pass 2: exp(x - max) and running sum.
  float local_sum = 0.0f;
  for (int j = threadIdx.x; j < C; j += blockDim.x) {
    float e = expf(xrow[j] - m);
    yrow[j] = e;
    local_sum += e;
  }
  float row_sum = block_reduce_sum_bcast(local_sum, smem);
  if (threadIdx.x == 0) s_sum = row_sum;
  __syncthreads();
  float inv = 1.0f / s_sum;

  // Pass 3: normalize.
  for (int j = threadIdx.x; j < C; j += blockDim.x) yrow[j] *= inv;
}

__global__ void softmax_bwd(const float *y, const float *gO, float *gX, int R,
                            int C) {
  int row = blockIdx.x;
  if (row >= R) return;

  __shared__ float smem[32];
  __shared__ float s_dot;

  const float *yrow = y + row * C;
  const float *gOrow = gO + row * C;
  float *gXrow = gX + row * C;

  // dot = sum_c(dO * y)
  float local = 0.0f;
  for (int j = threadIdx.x; j < C; j += blockDim.x)
    local += gOrow[j] * yrow[j];
  float dot = block_reduce_sum_bcast(local, smem);
  if (threadIdx.x == 0) s_dot = dot;
  __syncthreads();
  float d = s_dot;

  for (int j = threadIdx.x; j < C; j += blockDim.x)
    gXrow[j] = (gOrow[j] - d) * yrow[j];
}

// ---------------------------------------------------------------------
// Fused cross-entropy (softmax + NLL). Recomputes softmax for backward
// rather than caching it — the extra pass is far cheaper than the extra
// memory + FFI round-trips a cache would need.
// ---------------------------------------------------------------------

__global__ void cross_entropy_fwd(const float *x, const float *targets,
                                  float *loss, int R, int C) {
  int row = blockIdx.x;
  if (row >= R) return;

  __shared__ float smem[32];
  __shared__ float s_max, s_lse;

  const float *xrow = x + row * C;

  // Row max.
  float local_max = -1e30f;
  for (int j = threadIdx.x; j < C; j += blockDim.x)
    local_max = fmaxf(local_max, xrow[j]);
  float row_max = block_reduce_max_bcast(local_max, smem);
  if (threadIdx.x == 0) s_max = row_max;
  __syncthreads();
  float m = s_max;

  // Row sum of exp(x - max).
  float local_sum = 0.0f;
  for (int j = threadIdx.x; j < C; j += blockDim.x)
    local_sum += expf(xrow[j] - m);
  float row_sum = block_reduce_sum_bcast(local_sum, smem);
  if (threadIdx.x == 0) s_lse = m + logf(row_sum);
  __syncthreads();

  if (threadIdx.x == 0) {
    int tgt = (int)(targets[row] + 0.5f);
    loss[row] = s_lse - xrow[tgt];
  }
}

__global__ void cross_entropy_bwd(const float *x, const float *targets,
                                  const float *gLoss, float *gX, int R,
                                  int C) {
  int row = blockIdx.x;
  if (row >= R) return;

  __shared__ float smem[32];
  __shared__ float s_max, s_inv_sum;

  const float *xrow = x + row * C;
  float *gXrow = gX + row * C;

  // Row max.
  float local_max = -1e30f;
  for (int j = threadIdx.x; j < C; j += blockDim.x)
    local_max = fmaxf(local_max, xrow[j]);
  float row_max = block_reduce_max_bcast(local_max, smem);
  if (threadIdx.x == 0) s_max = row_max;
  __syncthreads();
  float m = s_max;

  // Row sum of exp(x - max).
  float local_sum = 0.0f;
  for (int j = threadIdx.x; j < C; j += blockDim.x)
    local_sum += expf(xrow[j] - m);
  float row_sum = block_reduce_sum_bcast(local_sum, smem);
  if (threadIdx.x == 0) s_inv_sum = 1.0f / row_sum;
  __syncthreads();
  float inv = s_inv_sum;

  int tgt = (int)(targets[row] + 0.5f);
  float g = gLoss[row];

  for (int j = threadIdx.x; j < C; j += blockDim.x) {
    float p = expf(xrow[j] - m) * inv;
    gXrow[j] = (p - (j == tgt ? 1.0f : 0.0f)) * g;
  }
}
