# **1. THE AFT OPERATOR**

The AFT operator lives entirely in one file — [lib/core/tensor/aft.dart](../../lib/core/tensor/aft.dart) — as the
`TensorAft.aftFull` static extension method. This chapter walks
the CPU forward pass; the backward is [chapter 5](./05-CUSTOM-BACKWARD.md), the
numerical-stability trick is [chapter 4](./04-NUMERICAL-STABILITY.md).

## **1.1. Signature and shape check**

<sup>from [lib/core/tensor/aft.dart](../../lib/core/tensor/aft.dart):</sup>

```dart
extension TensorAft on Tensor {
  static Tensor aftFull(
    Tensor q,
    Tensor k,
    Tensor v,
    Tensor w, {
    bool masked = false,
  }) {
    if (q.shape.length != 2 || k.shape.length != 2 || v.shape.length != 2) {
      throw ArgumentError(
        'aftFull: q/k/v must be 2D [T, D]; got q=${q.shape} k=${k.shape} '
        'v=${v.shape}',
      );
    }
    final t = q.shape[0];
    final d = q.shape[1];
    if (k.shape[0] != t || k.shape[1] != d ||
        v.shape[0] != t || v.shape[1] != d) {
      throw ArgumentError(
        'aftFull: q, k, v must share shape [T, D]; ...',
      );
    }
    if (w.shape.length != 2 || w.shape[0] != t || w.shape[1] != t) {
      throw ArgumentError('aftFull: w must be [T, T]=[$t, $t]; got ${w.shape}');
    }
    ...
```

Inputs and shapes at a glance:

| Tensor | Shape    | What it is                                        |
|--------|----------|---------------------------------------------------|
| `q`    | `[T, D]` | Query projection of the input                     |
| `k`    | `[T, D]` | Key projection of the input                       |
| `v`    | `[T, D]` | Value projection of the input                     |
| `w`    | `[T, T]` | Learned position-bias matrix (see chapter 2)      |
| `out`  | `[T, D]` | Same shape as `q`, `k`, `v`                       |

Two additional invariants enforced up front:

- **`k` and `v` must share shape with `q`.** No cross-attention
  form; AFT is self-attention only.
- **`w` must be square `[T, T]`.** For sequences shorter than the
  module's `maxSeqLen`, the top-left `[T, T]` slice is used (see
  [chapter 6](./06-AFT-ATTENTION-MODULE.md)).

Device is also checked (`q.device == k.device == v.device ==
w.device`) — the op does not auto-move tensors between CPU and GPU.

## **1.2. CPU forward: three nested loops**

The core is a triple loop over `(i, dd, t')` where `i` is the output
position, `dd` is the feature-axis index, and `t'` is the context
position:

<sup>from [lib/core/tensor/aft.dart](../../lib/core/tensor/aft.dart):</sup>

```dart
static Tensor _aftFullCpu(
  Tensor q, Tensor k, Tensor v, Tensor w, {required bool masked},
) {
  final t = q.shape[0];
  final d = q.shape[1];

  final qd = q._cpuData!;
  final kd = k._cpuData!;
  final vd = v._cpuData!;
  final wd = w._cpuData!;

  final weights = Float32List(t * t * d); // [t_out, t_prime, d]
  final wv = Float32List(t * d);          // weighted V per output pos
  final sigQ = Float32List(t * d);
  final outData = Float32List(t * d);

  for (int i = 0; i < t; i++) {
    final limit = masked ? i + 1 : t;
    for (int dd = 0; dd < d; dd++) {
      // ... (see below)
    }
  }

  final out = Tensor._cpu([t, d], outData);
  ...
}
```

The three intermediate buffers exist because we keep them alive for
the backward pass:

- `weights[i, t', dd]` — the normalized `[T, T, D]` mixing weights
  after per-(i, dd) softmax. Chapter 5 will use these for the
  gradient of `K`, `V`, `W`.
- `wv[i, dd]` — the weighted sum of `V[:, dd]` for output position
  `i` (the "aggregated value" before gating).
- `sigQ[i, dd]` — `sigmoid(Q[i, dd])`, the elementwise gate. Kept
  because its derivative is `sq * (1 - sq)`.

`outData[i, dd] = sigQ[i, dd] * wv[i, dd]`.

## **1.3. Inside the double loop**

For a fixed `(i, dd)`, the code does:

<sup>from [lib/core/tensor/aft.dart](../../lib/core/tensor/aft.dart):</sup>

```dart
double maxVal = -1e30;
for (int tp = 0; tp < limit; tp++) {
  final s = kd[tp * d + dd] + wd[i * t + tp];
  if (s > maxVal) maxVal = s;
}
double denom = 0.0;
for (int tp = 0; tp < limit; tp++) {
  final e = math.exp(kd[tp * d + dd] + wd[i * t + tp] - maxVal);
  weights[(i * t + tp) * d + dd] = e;
  denom += e;
}
final invDenom = 1.0 / (denom + 1e-6);
double num = 0.0;
for (int tp = 0; tp < limit; tp++) {
  final wt = weights[(i * t + tp) * d + dd] * invDenom;
  weights[(i * t + tp) * d + dd] = wt;
  num += wt * vd[tp * d + dd];
}
wv[i * d + dd] = num;
final sq = 1.0 / (1.0 + math.exp(-qd[i * d + dd]));
sigQ[i * d + dd] = sq;
outData[i * d + dd] = sq * num;
```

Three sweeps over `t'` in the inner loop:

1. **Find the max** `max_{t'} (K[t', dd] + W[i, t'])`. This is the
   numerical-stability step ([chapter 4](./04-NUMERICAL-STABILITY.md)) — subtracting the max keeps
   the exponents in a safe range.

2. **Exponentiate and accumulate the denominator.** Each `e = exp(K
   + W - max)` becomes an unnormalized softmax weight, and their sum
   is the denominator.

   The `+ 1e-6` in `invDenom` is a safety margin against denormals
   when the entire row is heavily suppressed; without it a row of
   all-`-1e9` scores could produce `denom == 0`.

3. **Normalize and mix.** Multiply each stored `e` by `1 / denom` to
   get the true softmax weight `wt`, then accumulate `wt * V[t',
   dd]` into `num`.

Then the elementwise gate:

- `sq = sigmoid(Q[i, dd])` — computed as `1 / (1 + exp(-q))`.
- `out[i, dd] = sq * num`.

That's the whole forward pass. About 20 lines of arithmetic inside
two Dart `for` loops.

## **1.4. GPU forward**

The GPU path just calls into a CUDA kernel and wires up autograd:

<sup>from [lib/core/tensor/aft.dart](../../lib/core/tensor/aft.dart):</sup>

```dart
static Tensor _aftFullGpu(
  Tensor q, Tensor k, Tensor v, Tensor w, {required bool masked},
) {
  final t = q.shape[0];
  final d = q.shape[1];
  final outHandle = engine.aftFullForward(
    q._handle!, k._handle!, v._handle!, w._handle!,
    masked ? 1 : 0,
  );
  final out = Tensor._gpu([t, d], outHandle);
  if (q.requiresGrad || k.requiresGrad || v.requiresGrad || w.requiresGrad) {
    out._setBackward([q, k, v, w], () {
      // Fresh zero-init grad tensors for the kernel to atomicAdd into.
      final gQ = Tensor.fill(q.shape, 0.0, device: Device.GPU);
      final gK = Tensor.fill(k.shape, 0.0, device: Device.GPU);
      final gV = Tensor.fill(v.shape, 0.0, device: Device.GPU);
      final gW = Tensor.fill(w.shape, 0.0, device: Device.GPU);
      engine.aftFullBackward(
        q._handle!, k._handle!, v._handle!, w._handle!,
        out._grad!._handle!,
        masked ? 1 : 0,
        gQ._handle!, gK._handle!, gV._handle!, gW._handle!,
      );
      ...
    });
  }
  return out;
}
```

The kernels are `aft_full_forward` and `aft_full_backward` on the
native side. Note the pre-allocated zero-init gradient buffers —
the backward kernel uses `atomicAdd` to accumulate, so it needs
zeroed memory to accumulate into.

The CPU and GPU paths produce numerically identical output to
several decimal places (verified by the numerical checks in
[test/aft_test.dart](../../test/aft_test.dart)).

## **1.5. Why per-`d` softmax matters**

The single most surprising aspect of AFT is that the softmax is done
**per feature dimension `d`**, not globally. Standard attention
softmaxes once over `t'` per output position `i` and uses that same
distribution for every `d`. AFT computes a **different** softmax
distribution for every `(i, d)` pair.

Concretely, in the code above, `weights[(i * t + tp) * d + dd]` is
indexed by all three of `(i, t', dd)`. Rearrange as `[T, T, D]` and
you'll see there's no shared axis being softmax'd — every column of
depth `D` gets its own normalization.

This is what gives AFT its expressive power despite the missing
`Q @ K.T`. Every feature dimension gets its own reason to mix
positions differently, and those `D` reasons compensate for the
fact that `Q` has been reduced to just a gate.

---

<div align="right">

[&lt;&lt;&nbsp;&nbsp;Previous: INTUITION](./00-INTUITION.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next chapter: POSITION BIAS&nbsp;&nbsp;&gt;](./02-POSITION-BIAS.md)

</div>
