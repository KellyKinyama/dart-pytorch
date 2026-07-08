// transpose.cuh - 2D tile-based transpose with autograd-friendly backward.
// Edit this file directly; engine.cu just #includes us.

#pragma once

// Tile-based transpose: each block reads a 32x32 tile of `in`, writes the
// transposed tile to `out`. Bank-conflict-free via +1 padding.
#ifndef DC_TRANSPOSE_TILE
#define DC_TRANSPOSE_TILE 32
#endif

__global__ void transpose_fwd_kernel(const float *in, float *out, int rows, int cols)
{
    __shared__ float tile[DC_TRANSPOSE_TILE][DC_TRANSPOSE_TILE + 1];

    int x = blockIdx.x * DC_TRANSPOSE_TILE + threadIdx.x; // column in input
    int y = blockIdx.y * DC_TRANSPOSE_TILE + threadIdx.y; // row    in input

    if (x < cols && y < rows)
        tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
    __syncthreads();

    // Output index: rows of out == cols of in, cols of out == rows of in.
    int out_x = blockIdx.y * DC_TRANSPOSE_TILE + threadIdx.x; // column in output (was row in input)
    int out_y = blockIdx.x * DC_TRANSPOSE_TILE + threadIdx.y; // row    in output (was col in input)

    if (out_x < rows && out_y < cols)
        out[out_y * rows + out_x] = tile[threadIdx.x][threadIdx.y];
}

// Backward of out = in^T: grad_in += grad_out^T (same transpose, accumulating).
// Inputs to this kernel: grad_out has shape [cols, rows]; grad_in has [rows, cols].
__global__ void transpose_bwd_kernel(const float *grad_out, float *grad_in,
                                     int rows, int cols)
{
    __shared__ float tile[DC_TRANSPOSE_TILE][DC_TRANSPOSE_TILE + 1];

    // grad_out is [cols, rows]; treat its rows=cols and cols=rows for indexing.
    int x = blockIdx.x * DC_TRANSPOSE_TILE + threadIdx.x; // column index in grad_out (==rows)
    int y = blockIdx.y * DC_TRANSPOSE_TILE + threadIdx.y; // row    index in grad_out (==cols)

    if (x < rows && y < cols)
        tile[threadIdx.y][threadIdx.x] = grad_out[y * rows + x];
    __syncthreads();

    int out_x = blockIdx.y * DC_TRANSPOSE_TILE + threadIdx.x; // column in grad_in (==cols)
    int out_y = blockIdx.x * DC_TRANSPOSE_TILE + threadIdx.y; // row    in grad_in (==rows)

    if (out_x < cols && out_y < rows)
        atomicAdd(&grad_in[out_y * cols + out_x], tile[threadIdx.x][threadIdx.y]);
}

// Row-wise softmax backward. For each row, given y = softmax(x) and gy = dL/dy,
// dL/dx_i = y_i * (gy_i - sum_j y_j * gy_j). One block per row mirrors the
// forward kernel layout.
__global__ void softmax_bwd_kernel(const float *y, const float *gy, float *gx,
                                   int T, int V)
{
    int t = blockIdx.x;
    if (t >= T) return;

    __shared__ float smem[32];
    __shared__ float s_dot;

    const float *yr  = y  + t * V;
    const float *gyr = gy + t * V;
    float       *gxr = gx + t * V;

    float local = 0.0f;
    for (int v = threadIdx.x; v < V; v += blockDim.x)
        local += yr[v] * gyr[v];
    float dot = block_reduce_sum_bcast(local, smem);
    if (threadIdx.x == 0) s_dot = dot;
    __syncthreads();
    float row_dot = s_dot;

    for (int v = threadIdx.x; v < V; v += blockDim.x)
        gxr[v] += yr[v] * (gyr[v] - row_dot);
}
