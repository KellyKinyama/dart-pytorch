# Running dart-pytorch on a free cloud GPU

This library runs on any Debian/Ubuntu Linux with a CUDA toolkit and
NVIDIA driver installed. That describes every mainstream **free GPU
notebook** service, which is a nice way to try the demos when you don't
own a discrete NVIDIA card.

| Service                         | GPU                | Weekly quota    | Sign-in                |
| ------------------------------- | ------------------ | --------------- | ---------------------- |
| Google Colab (free tier)        | T4 (~16 GB)        | ~12 h/day soft  | Google account         |
| Kaggle Notebooks                | T4 x2 or P100      | 30 h/week       | Kaggle account         |
| Paperspace Gradient (Free-M4000)| M4000              | 6 h/session     | Paperspace account     |
| Lightning AI Studio             | T4 (limited)       | monthly credits | Lightning AI account   |

The three ingredients on all of them are:

1. Enable the GPU on the runtime.
2. Install the Dart SDK.
3. `nvcc`-compile `lib/native/src/engine.cu` into `native/lib/libmat_mul.so`.

Steps 2 and 3 are wrapped in [scripts/setup_colab.sh](../scripts/setup_colab.sh);
you can also copy the cells below.

---

## Google Colab — cell-by-cell recipe

1. Open <https://colab.research.google.com/>, create a new notebook.
2. **Runtime -> Change runtime type -> Hardware accelerator = T4 GPU** (or whichever GPU is offered), Save.
3. Paste the cells below one by one.

**Cell 1 — sanity-check the GPU**

```python
!nvidia-smi | head -n 20
!nvcc --version | tail -n 1
```

**Cell 2 — clone your repo** (replace the URL with wherever you push it)

```python
!git clone https://github.com/<you>/dart-pytorch.git
%cd dart-pytorch
```

If your repo is private, either make it public temporarily or upload a
tarball with the Files panel and `!tar -xzf dart-pytorch.tgz`.

**Cell 3 — one-shot setup (Dart SDK + native library + pub get)**

```python
!bash scripts/setup_colab.sh
```

You should see something like:

```
>>> [1/4] verifying CUDA toolkit
release 12.2, V12.2.128
Tesla T4
>>> [2/4] installing Dart SDK (if missing)
Dart SDK version: 3.6.0 (stable) ...
>>> [3/4] building native/lib/libmat_mul.so with nvcc
-rwxr-xr-x 1 root root 4.2M ... native/lib/libmat_mul.so
>>> [4/4] dart pub get
Got dependencies!
Done.
```

**Cell 4 — run the word2vec demo on the GPU**

```python
!dart run bin/word2vec_demo.dart \
    --device=gpu \
    --max-chars=1200000 \
    --epochs=20 \
    --embed-dim=128 \
    --vocab-size=4096 \
    --num-ns=8 \
    --batch=1024
```

Bigger `--embed-dim`, `--batch`, and `--num-ns` are exactly the knobs
that turn CPU runs into agony but that a T4 chews through happily.

**Cell 5 — download the trained embeddings** (optional)

```python
!dart run bin/word2vec_demo.dart \
    --device=gpu --max-chars=1200000 --epochs=15 \
    --embed-dim=128 --vocab-size=4096 --num-ns=8 --batch=1024 \
    > run.log 2>&1
!tail -n 40 run.log

from google.colab import files
files.download('run.log')
```

---

## Kaggle Notebooks — recipe

Kaggle differs only in the accelerator toggle and the persistent home
directory:

1. New Notebook -> right sidebar **Settings -> Accelerator = GPU T4 x2** (or P100).
2. Also flip **Internet = ON** (required to download the Dart SDK).
3. Same three cells as Colab, except:

```python
!git clone https://github.com/<you>/dart-pytorch.git
%cd /kaggle/working/dart-pytorch
!bash scripts/setup_colab.sh
```

Kaggle's `/kaggle/working` is the only writable location that persists
across the cell run; do everything there.

---

## Troubleshooting

**`Failed to load dynamic library ... libmat_mul.so: cannot open shared object file`**

- The build step failed silently. Re-run `!bash scripts/setup_colab.sh`
  and check its output.
- `cuda_engine.dart` looks for `${cwd}/native/lib/libmat_mul.so` — make
  sure the `dart run ...` command is invoked from the repository root.

**`nvcc: command not found`**

- The runtime is CPU-only. On Colab: Runtime -> Change runtime type ->
  GPU. On Kaggle: Settings -> Accelerator -> GPU.

**`CUDA driver version is insufficient for CUDA runtime version`**

- The base image is older than the CUDA toolkit that nvcc targets. Add
  `-arch=sm_75` (T4 / RTX 20xx) or `-arch=sm_60` (P100) to the nvcc
  command in [scripts/setup_colab.sh](../scripts/setup_colab.sh), or
  ask nvcc to target the driver's max: `nvcc --gpu-architecture=native ...`

**"It works on CPU but crashes on GPU"**

- Every op the demo touches (embedding, elementwise mul, matmul, fused
  crossEntropy, mean) has a GPU path. If you added an op that lacks a
  GPU kernel you'll see a clear "mixed devices" `ArgumentError` — call
  `.to(Device.CPU)` on the offending tensor, do the op, then `.to(cfg.device)`
  before continuing. See [device-placement.md](device-placement.md).

**The free session dies after N hours**

- Colab / Kaggle both kill idle or long-running sessions. Save the
  trained target embedding matrix (`Tensor.toList()`) to a `.tsv` file
  and download it with `google.colab.files.download` / Kaggle's
  right-sidebar Output panel *before* the session dies.
