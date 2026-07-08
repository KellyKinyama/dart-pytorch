// engine.cu — forward-only entry point for the Dart FFI layer.
//
// Kernels are pulled from dart_cuda/native/src/kernels/*.cuh. We only
// wrap the forward passes here; autograd wiring lives in Dart and is
// still WIP, so all backward kernels / lambdas / children tracking are
// intentionally omitted.
//
// Build: nvcc --shared -Xcompiler -fPIC \
//              -o native/lib/libmat_mul.so lib/native/src/engine.cu

#include "kernels/common.cuh"
#include "kernels/matmul.cuh"
#include "kernels/elementwise.cuh"
#include "kernels/transpose.cuh"
#include "kernels/layernorm.cuh"
#include "kernels/softmax.cuh"
#include "kernels/embedding.cuh"
#include "kernels/attention.cuh"

extern "C"
{
    struct Tensor
    {
        float *data_gpu, *grad_gpu;
        int rows, cols, size;

        Tensor(int r, int c) : rows(r), cols(c), size(r * c)
        {
            cudaMalloc(&data_gpu, size * sizeof(float));
            cudaMalloc(&grad_gpu, size * sizeof(float));
            cudaMemset(grad_gpu, 0, size * sizeof(float));
        }

        ~Tensor()
        {
            if (data_gpu) cudaFree(data_gpu);
            if (grad_gpu) cudaFree(grad_gpu);
        }
    };

    // ---------------------------------------------------------------------
    // Scalar-broadcast forward kernels (copied inline from dart_cuda's
    // engine.cu since they're not in a shared header). Only fwd here.
    // ---------------------------------------------------------------------
    __global__ void add_scalar_fwd_k(const float *a, const float *b,
                                     float *out, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) out[i] = a[i] + b[0];
    }
    __global__ void sub_scalar_fwd_k(const float *a, const float *b,
                                     float *out, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) out[i] = a[i] - b[0];
    }
    __global__ void mul_scalar_fwd_k(const float *a, const float *b,
                                     float *out, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) out[i] = a[i] * b[0];
    }
    __global__ void div_scalar_fwd_k(const float *a, const float *b,
                                     float *out, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) out[i] = a[i] / b[0];
    }

    // ---------------------------------------------------------------------
    // Lifecycle + I/O
    // ---------------------------------------------------------------------

    DLLEXPORT void *create_tensor(int r, int c, float *d)
    {
        Tensor *t = new Tensor(r, c);
        if (d)
            cudaMemcpy(t->data_gpu, d, t->size * sizeof(float),
                       cudaMemcpyHostToDevice);
        return (void *)t;
    }

    DLLEXPORT void destroy_tensor(void *h)
    {
        if (!h) return;
        delete (Tensor *)h;
    }

    DLLEXPORT void get_tensor_data(void *h, float *b)
    {
        Tensor *t = (Tensor *)h;
        cudaMemcpy(b, t->data_gpu, t->size * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }

    // ---------------------------------------------------------------------
    // Matmul (tiled 32x32).
    // ---------------------------------------------------------------------

    DLLEXPORT void *matmul_tensors(void *ah, void *bh)
    {
        Tensor *a = (Tensor *)ah;
        Tensor *b = (Tensor *)bh;
        int M = a->rows, K = a->cols, N = b->cols;
        Tensor *out = new Tensor(M, N);

        dim3 th(32, 32);
        dim3 bl((N + 31) / 32, (M + 31) / 32);
        matmul_fwd<<<bl, th>>>(a->data_gpu, b->data_gpu, out->data_gpu,
                               M, K, N);
        return (void *)out;
    }

    // ---------------------------------------------------------------------
    // Elementwise binary (exact-shape). float4-vectorized fwd kernels.
    // ---------------------------------------------------------------------

    static inline int vec4_blocks(int n)
    {
        return (n + (256 * 4) - 1) / (256 * 4);
    }

    DLLEXPORT void *add_tensors(void *ah, void *bh)
    {
        Tensor *a = (Tensor *)ah, *b = (Tensor *)bh;
        Tensor *out = new Tensor(a->rows, a->cols);
        add_fwd<<<vec4_blocks(a->size), 256>>>(a->data_gpu, b->data_gpu,
                                               out->data_gpu, a->size);
        return (void *)out;
    }
    DLLEXPORT void *sub_tensors(void *ah, void *bh)
    {
        Tensor *a = (Tensor *)ah, *b = (Tensor *)bh;
        Tensor *out = new Tensor(a->rows, a->cols);
        sub_fwd<<<vec4_blocks(a->size), 256>>>(a->data_gpu, b->data_gpu,
                                               out->data_gpu, a->size);
        return (void *)out;
    }
    DLLEXPORT void *mul_tensors(void *ah, void *bh)
    {
        Tensor *a = (Tensor *)ah, *b = (Tensor *)bh;
        Tensor *out = new Tensor(a->rows, a->cols);
        mul_fwd<<<vec4_blocks(a->size), 256>>>(a->data_gpu, b->data_gpu,
                                               out->data_gpu, a->size);
        return (void *)out;
    }
    DLLEXPORT void *div_tensors(void *ah, void *bh)
    {
        Tensor *a = (Tensor *)ah, *b = (Tensor *)bh;
        Tensor *out = new Tensor(a->rows, a->cols);
        div_fwd<<<(a->size + 255) / 256, 256>>>(a->data_gpu, b->data_gpu,
                                                out->data_gpu, a->size);
        return (void *)out;
    }

    // ---------------------------------------------------------------------
    // Scalar broadcast (B is a 1-elem Tensor).
    // ---------------------------------------------------------------------

    DLLEXPORT void *add_tensor_scalar(void *ah, void *bh)
    {
        Tensor *a = (Tensor *)ah, *b = (Tensor *)bh;
        Tensor *out = new Tensor(a->rows, a->cols);
        int blocks = (a->size + 255) / 256;
        add_scalar_fwd_k<<<blocks, 256>>>(a->data_gpu, b->data_gpu,
                                          out->data_gpu, a->size);
        return (void *)out;
    }
    DLLEXPORT void *sub_tensor_scalar(void *ah, void *bh)
    {
        Tensor *a = (Tensor *)ah, *b = (Tensor *)bh;
        Tensor *out = new Tensor(a->rows, a->cols);
        int blocks = (a->size + 255) / 256;
        sub_scalar_fwd_k<<<blocks, 256>>>(a->data_gpu, b->data_gpu,
                                          out->data_gpu, a->size);
        return (void *)out;
    }
    DLLEXPORT void *mul_tensor_scalar(void *ah, void *bh)
    {
        Tensor *a = (Tensor *)ah, *b = (Tensor *)bh;
        Tensor *out = new Tensor(a->rows, a->cols);
        int blocks = (a->size + 255) / 256;
        mul_scalar_fwd_k<<<blocks, 256>>>(a->data_gpu, b->data_gpu,
                                          out->data_gpu, a->size);
        return (void *)out;
    }
    DLLEXPORT void *div_tensor_scalar(void *ah, void *bh)
    {
        Tensor *a = (Tensor *)ah, *b = (Tensor *)bh;
        Tensor *out = new Tensor(a->rows, a->cols);
        int blocks = (a->size + 255) / 256;
        div_scalar_fwd_k<<<blocks, 256>>>(a->data_gpu, b->data_gpu,
                                          out->data_gpu, a->size);
        return (void *)out;
    }

    // Row-broadcast add: out[M,N] = a[M,N] + b[1,N]. For Linear bias-add.
    DLLEXPORT void *add_tensor_row_broadcast(void *ah, void *bh)
    {
        Tensor *a = (Tensor *)ah, *b = (Tensor *)bh;
        int M = a->rows, N = a->cols;
        Tensor *out = new Tensor(M, N);
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x,
                  (M + block.y - 1) / block.y);
        add_row_broadcast_fwd<<<grid, block>>>(a->data_gpu, b->data_gpu,
                                               out->data_gpu, M, N);
        return (void *)out;
    }

    // ---------------------------------------------------------------------
    // Unary elementwise + activations.
    // ---------------------------------------------------------------------

    DLLEXPORT void *abs_tensor(void *ah)
    {
        Tensor *a = (Tensor *)ah;
        Tensor *out = new Tensor(a->rows, a->cols);
        abs_fwd<<<(a->size + 255) / 256, 256>>>(a->data_gpu, out->data_gpu,
                                                a->size);
        return (void *)out;
    }
    DLLEXPORT void *log_tensor(void *ah)
    {
        Tensor *a = (Tensor *)ah;
        Tensor *out = new Tensor(a->rows, a->cols);
        log_fwd<<<(a->size + 255) / 256, 256>>>(a->data_gpu, out->data_gpu,
                                                a->size);
        return (void *)out;
    }
    DLLEXPORT void *pow_tensor(void *ah, float exp)
    {
        Tensor *a = (Tensor *)ah;
        Tensor *out = new Tensor(a->rows, a->cols);
        pow_fwd<<<(a->size + 255) / 256, 256>>>(a->data_gpu, exp,
                                                out->data_gpu, a->size);
        return (void *)out;
    }
    DLLEXPORT void *relu_tensor(void *ah)
    {
        Tensor *a = (Tensor *)ah;
        Tensor *out = new Tensor(a->rows, a->cols);
        relu_fwd<<<(a->size + 255) / 256, 256>>>(a->data_gpu, out->data_gpu,
                                                 a->size);
        return (void *)out;
    }
    DLLEXPORT void *sigmoid_tensor(void *ah)
    {
        Tensor *a = (Tensor *)ah;
        Tensor *out = new Tensor(a->rows, a->cols);
        sigmoid_fwd<<<(a->size + 255) / 256, 256>>>(a->data_gpu, out->data_gpu,
                                                    a->size);
        return (void *)out;
    }
    DLLEXPORT void *tanh_tensor(void *ah)
    {
        Tensor *a = (Tensor *)ah;
        Tensor *out = new Tensor(a->rows, a->cols);
        tanh_fwd<<<(a->size + 255) / 256, 256>>>(a->data_gpu, out->data_gpu,
                                                 a->size);
        return (void *)out;
    }

    // ReLU backward: gA += (a > 0) * gO. `gA` is caller-allocated
    // zero-init; kernel uses atomicAdd (safe with grad accumulation
    // from other branches). Same pattern as layernorm/embedding bwd.
    DLLEXPORT void relu_backward_op(void *ah, void *goh, void *gaH)
    {
        Tensor *a = (Tensor *)ah;
        Tensor *gO = (Tensor *)goh;
        Tensor *gA = (Tensor *)gaH;
        relu_bwd<<<(a->size + 255) / 256, 256>>>(
            a->data_gpu, gO->data_gpu, gA->data_gpu, a->size);
    }

    // ---------------------------------------------------------------------
    // Rearrangement.
    // ---------------------------------------------------------------------

    DLLEXPORT void *transpose_tensor(void *ah)
    {
        Tensor *a = (Tensor *)ah;
        Tensor *out = new Tensor(a->cols, a->rows);
        dim3 block(DC_TRANSPOSE_TILE, DC_TRANSPOSE_TILE);
        dim3 grid((a->cols + DC_TRANSPOSE_TILE - 1) / DC_TRANSPOSE_TILE,
                  (a->rows + DC_TRANSPOSE_TILE - 1) / DC_TRANSPOSE_TILE);
        transpose_fwd_kernel<<<grid, block>>>(a->data_gpu, out->data_gpu,
                                              a->rows, a->cols);
        return (void *)out;
    }

    // ---------------------------------------------------------------------
    // Reductions. Result is a 1x1 tensor.
    // ---------------------------------------------------------------------

    DLLEXPORT void *sum_tensor(void *ah)
    {
        Tensor *a = (Tensor *)ah;
        Tensor *out = new Tensor(1, 1);
        cudaMemset(out->data_gpu, 0, sizeof(float));
        int threads = 256;
        int blocks = (a->size + threads - 1) / threads;
        sum_fwd_kernel<<<blocks, threads>>>(a->data_gpu, out->data_gpu,
                                            a->size);
        return (void *)out;
    }
    DLLEXPORT void *mean_tensor(void *ah)
    {
        Tensor *a = (Tensor *)ah;
        Tensor *out = new Tensor(1, 1);
        cudaMemset(out->data_gpu, 0, sizeof(float));
        int threads = 256;
        int blocks = (a->size + threads - 1) / threads;
        mean_fwd_kernel<<<blocks, threads>>>(a->data_gpu, out->data_gpu,
                                             a->size);
        return (void *)out;
    }

    // ---------------------------------------------------------------------
    // LayerNorm forward + backward.
    //
    // Dart-side autograd owns the gradient tensors; the backward wrapper
    // takes pre-allocated `gGamma` / `gBeta` handles and atomicAdds into
    // them (matches how gradients accumulate across multiple call sites).
    // `gX` is freshly allocated and returned as a handle.
    // ---------------------------------------------------------------------

    DLLEXPORT void *layernorm_forward(void *xh, void *gh, void *bh, float eps)
    {
        Tensor *x = (Tensor *)xh;
        Tensor *gamma = (Tensor *)gh;
        Tensor *beta = (Tensor *)bh;
        Tensor *out = new Tensor(x->rows, x->cols);
        layernorm_fwd<<<x->rows, 256>>>(x->data_gpu, gamma->data_gpu,
                                        beta->data_gpu, out->data_gpu,
                                        x->rows, x->cols, eps);
        return (void *)out;
    }

    DLLEXPORT void *layernorm_backward(void *xh, void *gh, void *goh,
                                       void *gGammaH, void *gBetaH, float eps)
    {
        Tensor *x = (Tensor *)xh;
        Tensor *gamma = (Tensor *)gh;
        Tensor *gO = (Tensor *)goh;
        Tensor *gGamma = (Tensor *)gGammaH;
        Tensor *gBeta = (Tensor *)gBetaH;
        Tensor *gX = new Tensor(x->rows, x->cols);
        layernorm_bwd<<<x->rows, 256>>>(x->data_gpu, gamma->data_gpu,
                                        gO->data_gpu, gX->data_gpu,
                                        gGamma->data_gpu, gBeta->data_gpu,
                                        x->rows, x->cols, eps);
        return (void *)gX;
    }

    // ---------------------------------------------------------------------
    // Softmax forward + backward (row-wise, block-per-row).
    // ---------------------------------------------------------------------

    DLLEXPORT void *softmax_forward(void *xh)
    {
        Tensor *x = (Tensor *)xh;
        Tensor *out = new Tensor(x->rows, x->cols);
        softmax_fwd<<<x->rows, 256>>>(x->data_gpu, out->data_gpu,
                                      x->rows, x->cols);
        return (void *)out;
    }

    DLLEXPORT void *softmax_backward(void *yh, void *goh)
    {
        Tensor *y = (Tensor *)yh;
        Tensor *gO = (Tensor *)goh;
        Tensor *gX = new Tensor(y->rows, y->cols);
        softmax_bwd<<<y->rows, 256>>>(y->data_gpu, gO->data_gpu,
                                      gX->data_gpu, y->rows, y->cols);
        return (void *)gX;
    }

    // ---------------------------------------------------------------------
    // Fused cross-entropy (softmax + NLL).
    //   forward:  x [R, C], targets [R] -> loss [R] (per-sample, no reduce)
    //   backward: x [R, C], targets [R], gLoss [R] -> gX [R, C]
    // ---------------------------------------------------------------------

    DLLEXPORT void *cross_entropy_forward(void *xh, void *targetsH)
    {
        Tensor *x = (Tensor *)xh;
        Tensor *targets = (Tensor *)targetsH;
        Tensor *loss = new Tensor(x->rows, 1);
        cross_entropy_fwd<<<x->rows, 256>>>(x->data_gpu, targets->data_gpu,
                                            loss->data_gpu, x->rows, x->cols);
        return (void *)loss;
    }

    DLLEXPORT void *cross_entropy_backward(void *xh, void *targetsH,
                                           void *gLossH)
    {
        Tensor *x = (Tensor *)xh;
        Tensor *targets = (Tensor *)targetsH;
        Tensor *gLoss = (Tensor *)gLossH;
        Tensor *gX = new Tensor(x->rows, x->cols);
        cross_entropy_bwd<<<x->rows, 256>>>(x->data_gpu, targets->data_gpu,
                                            gLoss->data_gpu, gX->data_gpu,
                                            x->rows, x->cols);
        return (void *)gX;
    }

    // ---------------------------------------------------------------------
    // Embedding lookup.
    //   forward:  table [V, D], indices [N] -> out [N, D]
    //   backward: indices [N], gOut [N, D] -> atomicAdd into gTable [V, D]
    //             (caller pre-allocates gTable — matches LayerNorm bwd).
    // ---------------------------------------------------------------------

    DLLEXPORT void *embedding_forward(void *tableH, void *indicesH, int N)
    {
        Tensor *table = (Tensor *)tableH;
        Tensor *indices = (Tensor *)indicesH;
        int D = table->cols;
        Tensor *out = new Tensor(N, D);
        int threads = D < 256 ? D : 256;
        embedding_fwd<<<N, threads>>>(table->data_gpu, indices->data_gpu,
                                      out->data_gpu, table->rows, D, N);
        return (void *)out;
    }

    DLLEXPORT void embedding_backward(void *indicesH, void *gOutH,
                                      void *gTableH, int N)
    {
        Tensor *indices = (Tensor *)indicesH;
        Tensor *gOut = (Tensor *)gOutH;
        Tensor *gTable = (Tensor *)gTableH;
        int D = gTable->cols;
        int threads = D < 256 ? D : 256;
        embedding_bwd<<<N, threads>>>(indices->data_gpu, gOut->data_gpu,
                                      gTable->data_gpu, gTable->rows, D, N);
    }

    // ---------------------------------------------------------------------
    // AFT-full forward + backward.
    //   Q, K, V, out : [T, D]     WB : [T, T]
    //   forward:  (Q, K, V, WB, masked) -> out
    //   backward: (Q, K, V, WB, gOut, masked, gQ, gK, gV, gWB) writes into
    //             pre-allocated grad tensors via atomicAdd — same pattern
    //             as layernorm_backward / embedding_backward.
    // ---------------------------------------------------------------------

    DLLEXPORT void *aft_full_forward(void *qh, void *kh, void *vh, void *wbh,
                                     int masked)
    {
        Tensor *Q = (Tensor *)qh;
        Tensor *K = (Tensor *)kh;
        Tensor *V = (Tensor *)vh;
        Tensor *WB = (Tensor *)wbh;
        int T = Q->rows;
        int D = Q->cols;
        Tensor *out = new Tensor(T, D);
        aft_full_fwd<<<(T + 255) / 256, 256>>>(
            Q->data_gpu, K->data_gpu, V->data_gpu, WB->data_gpu,
            out->data_gpu, T, D, masked);
        return (void *)out;
    }

    DLLEXPORT void aft_full_backward(void *qh, void *kh, void *vh, void *wbh,
                                     void *goh, int masked,
                                     void *gqh, void *gkh, void *gvh,
                                     void *gwbh)
    {
        Tensor *Q = (Tensor *)qh;
        Tensor *K = (Tensor *)kh;
        Tensor *V = (Tensor *)vh;
        Tensor *WB = (Tensor *)wbh;
        Tensor *gOut = (Tensor *)goh;
        Tensor *gQ = (Tensor *)gqh;
        Tensor *gK = (Tensor *)gkh;
        Tensor *gV = (Tensor *)gvh;
        Tensor *gWB = (Tensor *)gwbh;
        int T = Q->rows;
        int D = Q->cols;
        aft_full_bwd<<<(T + 255) / 256, 256>>>(
            Q->data_gpu, K->data_gpu, V->data_gpu, WB->data_gpu,
            gOut->data_gpu,
            gQ->data_gpu, gK->data_gpu, gV->data_gpu, gWB->data_gpu,
            T, D, masked);
    }

    // ---------------------------------------------------------------------
    // Top-left slice: extract a `[rows, cols]` submatrix from `[R, C]`
    // input. Backward pads with zeros back to `[R, C]`. Both directions
    // use cudaMemcpy2D since the copy is contiguous per-row with a
    // stride mismatch (source pitch = C, dest pitch = cols).
    // ---------------------------------------------------------------------

    DLLEXPORT void *slice_top_left_forward(void *xh, int rows, int cols)
    {
        Tensor *x = (Tensor *)xh;
        Tensor *out = new Tensor(rows, cols);
        cudaMemcpy2D(out->data_gpu, cols * sizeof(float),
                     x->data_gpu, x->cols * sizeof(float),
                     cols * sizeof(float), rows,
                     cudaMemcpyDeviceToDevice);
        return (void *)out;
    }

    DLLEXPORT void *slice_top_left_backward(void *gOutH, int R, int C)
    {
        Tensor *gOut = (Tensor *)gOutH;
        Tensor *gIn = new Tensor(R, C);
        cudaMemset(gIn->data_gpu, 0, R * C * sizeof(float));
        cudaMemcpy2D(gIn->data_gpu, C * sizeof(float),
                     gOut->data_gpu, gOut->cols * sizeof(float),
                     gOut->cols * sizeof(float), gOut->rows,
                     cudaMemcpyDeviceToDevice);
        return (void *)gIn;
    }
}
