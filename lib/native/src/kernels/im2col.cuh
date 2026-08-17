// im2col NHWC-flat convolution unfold for 8x8 boards.
//
// Input:  [B*64, C]    NHWC-flat activations
// Output: [B*64, C*k*k] unfolded column matrix
//
// Column ordering matches the LC0 weight layout `[Cout, Cin, Kh, Kw]`
// so a matmul with `wT [C*k*k, Cout]` produces `[B*64, Cout]` directly.
//
// One thread per output float; each thread computes its (b, y, x, ci,
// ky, kx) coordinates from the flat index and either copies the
// corresponding input value or writes a zero for the padded borders.

#pragma once

#include <cuda_runtime.h>

__global__ void im2col_nhwc_fwd(const float *__restrict__ input,
                                float *__restrict__ output,
                                int b, int c, int k, int pad)
{
    const int totalCols = c * k * k;
    const int totalRows = b * 64;
    const int total = totalRows * totalCols;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int row = idx / totalCols;
    const int col = idx - row * totalCols;

    const int bi = row / 64;
    const int spatial = row - bi * 64;
    const int y = spatial >> 3;      // /8
    const int x = spatial & 7;        // %8

    const int kk = k * k;
    const int ci = col / kk;
    const int rem = col - ci * kk;
    const int ky = rem / k;
    const int kx = rem - ky * k;

    const int yin = y + ky - pad;
    const int xin = x + kx - pad;

    float v = 0.0f;
    if ((unsigned)yin < 8u && (unsigned)xin < 8u) {
        const int inRow = bi * 64 + yin * 8 + xin;
        v = input[inRow * c + ci];
    }
    output[idx] = v;
}
