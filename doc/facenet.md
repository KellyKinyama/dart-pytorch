# FaceNet in dart-pytorch

Port of [timesler/facenet-pytorch](https://github.com/timesler/facenet-pytorch)'s
`InceptionResnetV1(pretrained='vggface2', classify=False)`. Face
image → 512-d L2-normalized embedding; cosine similarity between two
embeddings = face similarity. Produces **bit-exact** output vs. the
Python reference (cosine 1.000000, max |Δ| ≈ 0 on the same fp32 input).

- [`lib/core/nn/vision/facenet.dart`](../../lib/core/nn/vision/facenet.dart)
  — `InceptionResnetV1`, `BasicConv2d`, `Block35 / Block17 / Block8`,
  `Mixed6a / Mixed7a`.
- [`lib/core/nn/vision/facenet_loader.dart`](../../lib/core/nn/vision/facenet_loader.dart)
  — walks the converted safetensors, folds Conv2d + BatchNorm2d pairs
  at load time via [`conv_bn_fold.dart`](../../lib/core/nn/vision/conv_bn_fold.dart).
- [`lib/core/nn/vision/pool2d.dart`](../../lib/core/nn/vision/pool2d.dart),
  [`nchw.dart`](../../lib/core/nn/vision/nchw.dart) — reusable
  building blocks (max-pool, global avg-pool, channel-axis concat,
  row-wise L2 normalize).
- [`bin/facenet_demo.dart`](../../bin/facenet_demo.dart),
  [`bin/facenet_gpu_demo.dart`](../../bin/facenet_gpu_demo.dart),
  [`bin/facenet_verify_demo.dart`](../../bin/facenet_verify_demo.dart)
  — runnable demos.
- [`test/facenet_test.dart`](../../test/facenet_test.dart),
  [`test/facenet_gpu_test.dart`](../../test/facenet_gpu_test.dart)
  — fast structural + bit-exact-vs-oracle tests.

## Quick start

One-time weight conversion (needs Python + `facenet-pytorch` +
`safetensors`; downloads ~107 MB of `.pt` weights on first run and
writes a 145 MB safetensors file):

```bash
python3 -m pip install --break-system-packages --user facenet-pytorch safetensors

mkdir -p models/facenet-vggface2
python3 scripts/convert_facenet_pt_to_safetensors.py \
    models/facenet-vggface2/model.safetensors
```

Optional: one-shot Python oracle so the demos can diff against a
reference (`/tmp/facenet_input.raw` + `/tmp/facenet_ref.raw`):

```bash
python3 scripts/facenet_reference.py "faces_gallery/Brad Pitt/sample_0.jpg"
```

CPU embedding + diff:

```bash
dart run bin/facenet_demo.dart
```

GPU embedding + diff (WSL2 / Linux with the CUDA build in place):

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  dart run bin/facenet_gpu_demo.dart
```

Face-verification on real JPEG crops from `faces_gallery/`:

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  dart run bin/facenet_verify_demo.dart --gpu
```

Prints something like:

```
== pairwise cosine similarity ==
  SAME       0.6342   Brad Pitt/sample_0.jpg  ↔  Brad Pitt/sample_1.jpg
  DIFFERENT  -0.0148  Brad Pitt/sample_0.jpg  ↔  Alia Bhatt/sample_0.jpg
  DIFFERENT   0.1679  Brad Pitt/sample_1.jpg  ↔  Alia Bhatt/sample_0.jpg
```

## Architecture

Everything below the 512-d Linear is `Conv2d + ReLU` (BN folded in),
`catChannels`, and pooling. Input is `[N, 3, 160, 160]` in RGB with
`(x − 127.5) / 128.0` normalization.

```
Input [N, 3, 160, 160]                     RGB, (x−127.5)/128
  │
  ▼ Stem (6 × BasicConv2d + one maxPool)
     conv2d_1a (3   → 32,  k=3, s=2)       [N, 32, 79, 79]
     conv2d_2a (32  → 32,  k=3)            [N, 32, 77, 77]
     conv2d_2b (32  → 64,  k=3, p=1)       [N, 64, 77, 77]
     maxPool2d(k=3, s=2)                   [N, 64, 38, 38]
     conv2d_3b (64  → 80,  k=1)            [N, 80, 38, 38]
     conv2d_4a (80  → 192, k=3)            [N,192, 36, 36]
     conv2d_4b (192 → 256, k=3, s=2)       [N,256, 17, 17]
  │
  ▼ 5 × Block35(scale=0.17)                (Inception-ResNet-A, 256ch)
  ▼ Mixed6a — reduction                    [N,896,  8,  8]
  ▼ 10 × Block17(scale=0.10)               (Inception-ResNet-B, 896ch)
  ▼ Mixed7a — reduction                    [N,1792, 3,  3]
  ▼ 5 × Block8(scale=0.20)                 (Inception-ResNet-C, 1792ch)
  ▼ 1 × Block8(scale=1.0, noReLU)          [N,1792, 3,  3]
  │
  ▼ globalAvgPool2d                        [N,1792]
  ▼ Dropout(p=0.6)                         (eval mode: identity)
  ▼ Linear(1792 → 512, bias=False)
  ▼ folded last_bn (scale·x + offset)
  ▼ L2 normalize                           [N, 512], ‖·‖ = 1
```

Each `BlockXX` is a small Inception-ResNet residual: 2–3 parallel
`BasicConv2d` branches → `catChannels` → 1×1 `conv2d` (with bias, no
BN) → `x + scale * up` → optional ReLU. `Block17` and `Block8` use
the classic rectangular `1×7 / 7×1` and `1×3 / 3×1` factorised kernels.

## Batch-norm folding

Every `Conv2d(bias=False)` + `BatchNorm2d(eps=1e-3)` pair is fused into
a single `Conv2d(bias=True)` when the loader reads the safetensors:

```
scale  = γ / √(σ² + ε)
W'     = scale · W
b'     = β − μ · scale
```

(See [`conv_bn_fold.dart`](../../lib/core/nn/vision/conv_bn_fold.dart).)
`last_bn` uses `affine=True` on the 512-d embedding, so it's folded
into a per-dim `scale` and `offset` applied right after `last_linear`
and before L2 normalization.

The BN eps is `1e-3` throughout (matches facenet-pytorch), *not* the
PyTorch default `1e-5`.

## HF safetensors key layout

`scripts/convert_facenet_pt_to_safetensors.py` writes 604 tensors
directly from `InceptionResnetV1.state_dict()`, dropping only
`.num_batches_tracked`. The loader consumes 602 and reports two
unused keys (`logits.weight`, `logits.bias`) — those only exist when
the checkpoint was saved with `classify=True`, which is irrelevant
for embedding extraction.

```
conv2d_1a.conv.weight            [32, 3, 3, 3]      ┐
conv2d_1a.bn.{weight,bias}       [32]               │  6 stem blocks
conv2d_1a.bn.running_{mean,var}  [32]               ┘
...
repeat_1.0.branch0.conv.weight   [32, 256, 1, 1]    ┐  Block35: 3 branches
repeat_1.0.branch0.bn.*          [32]               │  of BasicConv2d, then
repeat_1.0.branch1.{0,1}.*                          │  a plain conv2d with
repeat_1.0.branch2.{0,1,2}.*                        │  bias (no BN).
repeat_1.0.conv2d.{weight,bias}  [256, 96, 1, 1]    ┘
...
mixed_6a.branch{0,1.{0..2}}.*                       ┐  reductions
mixed_7a.branch{0.{0..1},1.{0..1},2.{0..2}}.*       ┘
...
block8.branch{0,1.{0..2}}.*                         ← final Block8, noReLU
last_linear.weight               [512, 1792]        ← bias=False
last_bn.{weight,bias}            [512]              ← γ, β (affine=True)
last_bn.running_{mean,var}       [512]              ← μ, σ²
```

Fine-grained shapes for all 604 keys are dumped by the converter
script (last five printed to stdout).

## Numerical accuracy

Feeding the exact fp32 input tensor produced by facenet-pytorch's own
preprocessing (`/tmp/facenet_input.raw`, `[3, 160, 160]`), the Dart
port produces an embedding **bit-identical** to
`InceptionResnetV1(pretrained='vggface2')(x)`:

| | CPU | GPU |
|---|---|---|
| Cosine vs. reference | 1.000000 | 1.000000 |
| Mean \|Δ\|            | 0.000000 | 0.000000 |
| Max \|Δ\|             | 0.000000 | 0.000000 |
| Forward wall-clock   | 2.3 s   | 1.3 s   |

Semantic sanity on the shipped face gallery
(`bin/facenet_verify_demo.dart --gpu`, default paths):

- Brad Pitt sample_0 ↔ sample_1: cosine ≈ 0.63 → **SAME**
- Brad Pitt ↔ Alia Bhatt (both pairs): cosine ≈ −0.01 / 0.17 → **DIFFERENT**

Standard cosine thresholds for VGGFace2:

- `> 0.4` → same person (95%+ precision)
- `0.25 – 0.4` → unclear; use more samples per identity
- `< 0.25` → different

## Fine-tuning

Every folded `Conv2d.weight` + `.bias` and the `last_linear.weight`
carry `requiresGrad: true`, so an optimizer sees the full parameter
list via `model.parameters()`.

**What actually trains today** (with our current Conv2d autograd):

- `last_linear` — the 512-d projection head.
- Any Linear / MLP head you attach on top of the 512-d embedding
  (triplet, ArcFace, contrastive, classification, whatever).

**What doesn't train yet** — backprop *through* Conv2d layers. Our
`Conv2d` uses host-side `im2col` and NHWC→NCHW permutes, which use
`.toList()` and `Tensor.fromFloat32List` and so break the autograd
tape between adjacent Conv2d layers. Gradients flow into the *last*
Conv2d's weight (via the matmul) but not into earlier ones. A native
CUDA `conv2d_backward` would unblock full backbone fine-tuning; it's
a separate follow-up.

Typical head-only recipe with the current code:

```dart
final model = InceptionResnetV1(device: Device.GPU);
FaceNetLoader.loadFile(model, 'models/facenet-vggface2/model.safetensors');
model.eval();       // keep BN folded, disable Dropout

// Freeze everything below last_linear.
for (final p in model.parameters()) {
  p.requiresGrad = false;
}
model.lastLinear.weight.requiresGrad = true;

// Attach your own head on top of the 512-d embedding.
final head = Linear(512, numClasses, device: Device.GPU);
final optim = Adam([
  model.lastLinear.weight,
  ...head.parameters(),
], lr: 1e-3);
```

## Known limitations

- **No face detection.** Assumes an already-aligned 160×160 face crop
  (or something the resize step will produce a sensible one from).
  For raw photos, add [MTCNN](https://github.com/timesler/facenet-pytorch#usage)
  or [retinaface](https://github.com/serengil/retinaface) upstream —
  a Dart MTCNN port is a plausible follow-up.
- **Fixed-input Conv2d.** No autograd through convs (see previous
  section). Fine-tuning is head-only until that lands.
- **CPU pool ops.** `maxPool2d` and `globalAvgPool2d` still run on
  the host and re-upload to the device. Not a bottleneck (they take
  < 10 ms out of a 1.3 s GPU forward), but worth a note.
- **Single-frame demo.** All the CLI demos assume `N = 1`; the model
  itself handles arbitrary `N`, but for the JPEG-in demos you'd want
  to batch the disk-reads and the CPU-side `_decodeAndPrep`
  yourself for real throughput.
