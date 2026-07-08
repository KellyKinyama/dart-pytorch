// Embedding table lookup.
//
// Forward:  out[n, d] = table[ indices[n], d ]
// Backward: gTable[v, d] += sum over n where indices[n]==v of gOut[n, d]
//           (scatter-add via atomicAdd; caller supplies zero-init gTable).
//
// Layout:
//   table   [V, D]
//   indices [N]      -- float32, rounded to int (same convention as CE)
//   out     [N, D]
//
// Launch:
//   forward:  <<<N, min(D, 256)>>>   -- one block per output row
//   backward: <<<N, min(D, 256)>>>   -- one block per input row; atomicAdd

#pragma once
#include "common.cuh"

__global__ void embedding_fwd(const float *table, const float *indices,
                              float *out, int V, int D, int N) {
  int n = blockIdx.x;
  if (n >= N) return;

  int v = (int)(indices[n] + 0.5f);
  if (v < 0 || v >= V) {
    // Out-of-range index: zero the row. Cheaper than aborting.
    for (int d = threadIdx.x; d < D; d += blockDim.x) out[n * D + d] = 0.0f;
    return;
  }

  const float *src = table + v * D;
  float *dst = out + n * D;
  for (int d = threadIdx.x; d < D; d += blockDim.x) dst[d] = src[d];
}

__global__ void embedding_bwd(const float *indices, const float *gOut,
                              float *gTable, int V, int D, int N) {
  int n = blockIdx.x;
  if (n >= N) return;

  int v = (int)(indices[n] + 0.5f);
  if (v < 0 || v >= V) return;

  const float *src = gOut + n * D;
  float *dst = gTable + v * D;
  for (int d = threadIdx.x; d < D; d += blockDim.x)
    atomicAdd(&dst[d], src[d]);
}
