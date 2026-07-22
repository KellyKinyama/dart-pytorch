# **4. NUMERICAL STABILITY**

The inner `softmax(K[:, d] + W[i, :])` in AFT is a textbook softmax,
so it needs the same numerical care as any other softmax: subtract
the row max before exponentiating. This chapter walks the two
stability tricks in `_aftFullCpu` and explains why each is
necessary.

## **4.1. Recap: why softmax needs stabilizing**

For a vector `s`, the softmax weight of entry `j` is

    softmax(s)[j] = exp(s[j]) / sum_k exp(s[k])

If any `s[j]` is large (say 100), `exp(100)` is around `2.7 * 10^43`
— fits in fp32 barely, but a few of them summed together overflow.
If `s[j]` is 800, `exp(800)` is `inf` and the whole computation is
ruined.

The fix is that softmax is **shift-invariant**:

    softmax(s)[j] = exp(s[j] - m) / sum_k exp(s[k] - m)   for any m

Choose `m = max_k s[k]`. Then the largest exponent is exactly
`exp(0) = 1`, and every other one is between `0` and `1`. No
overflow, minimal precision loss.

The regular `softmax` op does this in [lib/core/tensor/softmax.dart](../../lib/core/tensor/softmax.dart);
AFT reproduces the trick inline because the softmax happens per-`d`
and can't reuse the tensor-op version.

## **4.2. Sweep 1: find the max**

<sup>from [lib/core/tensor/aft.dart](../../lib/core/tensor/aft.dart):</sup>

```dart
for (int dd = 0; dd < d; dd++) {
  double maxVal = -1e30;
  for (int tp = 0; tp < limit; tp++) {
    final s = kd[tp * d + dd] + wd[i * t + tp];
    if (s > maxVal) maxVal = s;
  }
  ...
```

For a fixed `(i, dd)`, we scan `t' = 0..limit-1` and track the
maximum of `K[t', dd] + W[i, t']`. That's the `m` we're going to
subtract from every exponent.

The initial value `-1e30` is chosen so that any real fp32 score
(even something like `-1e9` from a masked-out contribution — although
AFT skips those via `limit`, not via a mask) will beat it. If the
loop range is empty (which never happens because `t >= 1` is enforced
upstream), you'd get `maxVal = -1e30` and every exponent would
underflow, producing `denom = 0`.

## **4.3. Sweep 2: exponentiate and accumulate**

```dart
double denom = 0.0;
for (int tp = 0; tp < limit; tp++) {
  final e = math.exp(kd[tp * d + dd] + wd[i * t + tp] - maxVal);
  weights[(i * t + tp) * d + dd] = e;
  denom += e;
}
```

Because we subtracted `maxVal`, the exponent argument is guaranteed
to be `<= 0`, so `e` is in `(0, 1]` — no overflow possible. The
largest `e` (the one that hit `maxVal`) is exactly `1.0`.

The unnormalized weights `e` are **stored** into the `weights`
buffer at this stage, not the final normalized weights. Sweep 3
will multiply them by `1 / denom` in place.

## **4.4. Sweep 3: the `+ 1e-6` denominator safety**

```dart
final invDenom = 1.0 / (denom + 1e-6);
double num = 0.0;
for (int tp = 0; tp < limit; tp++) {
  final wt = weights[(i * t + tp) * d + dd] * invDenom;
  weights[(i * t + tp) * d + dd] = wt;
  num += wt * vd[tp * d + dd];
}
wv[i * d + dd] = num;
```

The `+ 1e-6` in `denom + 1e-6` is a guard against **underflow of the
denominator**. When could this bite?

- On the first forward, all `K`, `W` are small random values, so no
  problem.
- After training pushes `K` or `W` to strongly negative values in
  certain rows, `exp(K + W - max)` for the non-max entries can
  underflow to `0.0`. The max entry itself is always `exp(0) = 1`,
  so `denom` is at least `1.0` in exact arithmetic.
- In fp32 with catastrophic cancellation, denom can very
  occasionally round to `0.0` — the `+ 1e-6` prevents that from
  causing an `inf` when we divide.

The trade-off: `+ 1e-6` slightly biases every weight toward 0
(`wt = e / (denom + 1e-6) < e / denom`). At `denom = 1`, the
relative error is `1e-6`. At `denom = 100`, it's `1e-8`. Both
negligible vs. fp32's `~1e-7` machine epsilon.

If you needed higher precision you'd switch the accumulator to fp64
(`double` on the Dart side is already fp64, so the CPU path
actually gets this for free — the `+ 1e-6` is more defensive than
strictly required).

## **4.5. Sigmoid of `Q`: also stable but for a different reason**

```dart
final sq = 1.0 / (1.0 + math.exp(-qd[i * d + dd]));
```

`sigmoid(q) = 1 / (1 + exp(-q))`. The direct formula:

- For `q >> 0`, `exp(-q) -> 0`, so `sq -> 1`. Safe.
- For `q << 0`, `exp(-q) -> inf`, so `sq -> 0`. **Safe:** `1 / inf =
  0` in IEEE-754, no NaN.

The "traditional" concern about sigmoid is that `1 / (1 + exp(x))`
(the alternate form) blows up when `x >> 0`. This code uses the
`exp(-q)` form, which is safe for negative `q`. For positive `q`,
`exp(-q)` is small and no issue arises. So the sigmoid here needs
no shift.

## **4.6. GPU kernel does the same thing**

The `aft_full_forward` kernel is a per-`(i, dd)` block that runs the
same three sweeps: max reduce, exponentiate, normalize + weighted
sum. The `+ 1e-6` denominator guard is baked into the kernel too, so
CPU and GPU produce numerically identical results.

## **4.7. Backward stability**

The custom backward (chapter 5) also reads the cached `weights`
buffer, which is already normalized and in `[0, 1]`. No further
exponentiation happens in the backward, so no additional stability
tricks are needed.

## **4.8. Summary**

Two lines of defense in the AFT forward:

- **`- maxVal`** before `exp` — protects against overflow of large
  scores. Necessary; without it any well-trained AFT model will
  eventually produce `inf`.
- **`+ 1e-6`** in the denominator — protects against underflow /
  fp32 cancellation. Defensive; unlikely to fire in practice but
  cheap insurance.

These two lines account for the difference between "AFT works" and
"AFT produces `nan` after 500 training steps."

---

<div align="right">

[&lt;&lt;&nbsp;&nbsp;Previous: CAUSAL MASKING](./03-CAUSAL-MASKING.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next chapter: CUSTOM BACKWARD&nbsp;&nbsp;&gt;](./05-CUSTOM-BACKWARD.md)

</div>
