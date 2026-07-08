// attention.cuh — Attention-Free Transformer (AFT-full) kernels.
//
// Ported from dart_cuda/native/src/engine_v2.cu. Forward computes a
// numerically-stable softmax over t' for each (t, d), weighted-summed
// against V and gated by sigmoid(Q). Backward recomputes the same
// stable weights (avoids allocating a [T,T,D] scratch buffer) and
// atomicAdds analytical gradients into caller-supplied grad tensors.
//
// Layout: Q, K, V, out are [T, D]; WB is [T, T]. Both fwd and bwd
// launch one thread per output row (`t`) with block size 256.

#pragma once

#include "common.cuh"

// masked -> restrict inner sum to t' <= t (causal decoder AFT).
__global__ void aft_full_fwd(const float *Q, const float *K, const float *V,
                             const float *WB, float *out,
                             int T, int D, int masked)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    for (int d = 0; d < D; d++)
    {
        int limit = masked ? (t + 1) : T;

        // Phase 1: per-(t,d) max for numerical stability.
        float max_val = -1e20f;
        for (int tp = 0; tp < limit; tp++)
        {
            float v = K[tp * D + d] + WB[t * T + tp];
            if (v > max_val) max_val = v;
        }

        // Phase 2: stable weighted sum.
        float num = 0.0f;
        float den = 0.0f;
        for (int tp = 0; tp < limit; tp++)
        {
            float w = expf(K[tp * D + d] + WB[t * T + tp] - max_val);
            num += w * V[tp * D + d];
            den += w;
        }

        float sigQ = 1.0f / (1.0f + expf(-Q[t * D + d]));
        out[t * D + d] = sigQ * (num / (den + 1e-6f));
    }
}

// grad_Q/K/V/WB are caller-allocated, zero-init, and atomicAdd'd into
// (Q, K, V, out and grad_out are read-only).
__global__ void aft_full_bwd(const float *Q, const float *K, const float *V,
                             const float *WB, const float *grad_out,
                             float *grad_Q, float *grad_K,
                             float *grad_V, float *grad_WB,
                             int T, int D, int masked)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    for (int d = 0; d < D; d++)
    {
        int limit = masked ? (t + 1) : T;

        // Recompute stable max + partition function to match fwd numerics.
        float max_val = -1e20f;
        for (int tp = 0; tp < limit; tp++)
        {
            float v = K[tp * D + d] + WB[t * T + tp];
            if (v > max_val) max_val = v;
        }

        float num = 0.0f;
        float den = 0.0f;
        for (int tp = 0; tp < limit; tp++)
        {
            float w = expf(K[tp * D + d] + WB[t * T + tp] - max_val);
            num += w * V[tp * D + d];
            den += w;
        }

        float den_inv = 1.0f / (den + 1e-6f);
        float ratio = num * den_inv;                     // == fwd's num/(den+eps)
        float sigQ = 1.0f / (1.0f + expf(-Q[t * D + d]));
        float dSigQ = sigQ * (1.0f - sigQ);
        float gO = grad_out[t * D + d];

        // dQ: out = sigQ * ratio  =>  dQ = gO * ratio * dSigQ.
        atomicAdd(&grad_Q[t * D + d], gO * ratio * dSigQ);

        // dV, dK, dW: quotient rule (V - ratio) * norm_weight, gated by sigQ.
        for (int tp = 0; tp < limit; tp++)
        {
            float w = expf(K[tp * D + d] + WB[t * T + tp] - max_val);
            float norm_w = w * den_inv;

            float dV = gO * sigQ * norm_w;
            atomicAdd(&grad_V[tp * D + d], dV);

            float dW = gO * sigQ * norm_w * (V[tp * D + d] - ratio);
            atomicAdd(&grad_K[tp * D + d], dW);
            atomicAdd(&grad_WB[t * T + tp], dW);
        }
    }
}
