/// Tensor-parallel transformer encoder block — a full pre-LN block whose
/// attention and MLP are split across several GPUs in one process.
///
///     h = x + mha(ln1(x))
///     y = h + mlp(ln2(h))
///
/// Attention and the feed-forward are tensor-parallel (see
/// [TensorParallelMultiHeadAttention] and [TensorParallelMLP]); the two
/// LayerNorms and the residual adds are cheap, so they are **replicated**
/// on the output device rather than sharded. Only the block's input and
/// output activations (and the small per-layer gather/all-reduce inside
/// the sharded sublayers) cross GPU boundaries.
///
/// Inference-oriented first cut (mirrors the inference TP attention):
/// 2D `[N, embedDim]` input, ReLU FFN, optional additive mask. Build one
/// from a reference [TransformerBlock] with [fromBlock] to get a layer
/// numerically equivalent to the single-device block.
library;

import '../tensor/tensor.dart';
import 'module.dart';
import 'transformer.dart';
import 'parallel_linear.dart';
import 'attention/tensor_parallel_attention.dart';

class TensorParallelTransformerBlock extends Module {
  final int embedDim;
  final int numHeads;
  final int ffnDim;
  final int outputDevice;
  final double eps;

  // LayerNorm params, replicated on [outputDevice].
  final Tensor ln1Gamma;
  final Tensor ln1Beta;
  final Tensor ln2Gamma;
  final Tensor ln2Beta;

  final TensorParallelMultiHeadAttention mha;
  final TensorParallelMLP mlp;

  /// Whether the sharded sublayers + LayerNorms are trainable leaves.
  final bool trainable;

  TensorParallelTransformerBlock._(
    this.embedDim,
    this.numHeads,
    this.ffnDim,
    this.outputDevice,
    this.eps,
    this.ln1Gamma,
    this.ln1Beta,
    this.ln2Gamma,
    this.ln2Beta,
    this.mha,
    this.mlp,
    this.trainable,
  );

  /// Shard a reference [block] across [devices] (defaults to all visible
  /// GPUs). Supports the block's ReLU / tanh-GELU / quick-GELU FFN. Pass
  /// `trainable: true` to make the sharded sublayers and LayerNorms
  /// autograd leaves.
  factory TensorParallelTransformerBlock.fromBlock(
    TransformerBlock block, {
    List<int>? devices,
    int? outputDevice,
    bool trainable = false,
  }) {
    final devs = devices ?? List<int>.generate(Tensor.gpuCount, (i) => i);
    final outDev = outputDevice ?? devs.first;

    final mha = TensorParallelMultiHeadAttention.fromAttention(
      block.mha,
      devices: devs,
      outputDevice: outDev,
      trainable: trainable,
    );
    final mlp = TensorParallelMLP.fromWeights(
      upWeight: block.ffn1.weight,
      upBias: block.ffn1.bias,
      downWeight: block.ffn2.weight,
      downBias: block.ffn2.bias,
      devices: devs,
      outputDevice: outDev,
      trainable: trainable,
      activation: _activationFn(block.activation),
    );

    Tensor onOut(Tensor t) {
      final r = t.detach().toGpu(outDev);
      if (trainable) r.requiresGrad = true;
      return r;
    }

    return TensorParallelTransformerBlock._(
      block.embedDim,
      block.numHeads,
      block.ffnDim,
      outDev,
      block.ln1.eps,
      onOut(block.ln1.gamma),
      onOut(block.ln1.beta),
      onOut(block.ln2.gamma),
      onOut(block.ln2.beta),
      mha,
      mlp,
      trainable,
    );
  }

  /// Forward over a 2D `[N, embedDim]` sequence with optional additive
  /// mask `[N, N]`. For autoregressive decoding pass a per-layer [cache]
  /// (from [newCache]) and [startPos]; the attention appends each token's
  /// K/V and continues RoPE positions from the cached length.
  /// Returns `[N, embedDim]` on [outputDevice].
  Tensor call(Tensor x, {Tensor? mask, TPMHACache? cache, int startPos = 0}) {
    final xo = x.toGpu(outputDevice);
    final normed1 = Tensor.onGpu(
      outputDevice,
      () => xo.layerNorm(ln1Gamma, ln1Beta, eps: eps),
    );
    final attn = mha(normed1, mask: mask, cache: cache, startPos: startPos);
    final h = Tensor.onGpu(outputDevice, () => xo + attn);

    final normed2 = Tensor.onGpu(
      outputDevice,
      () => h.layerNorm(ln2Gamma, ln2Beta, eps: eps),
    );
    final ff = mlp(normed2);
    return Tensor.onGpu(outputDevice, () => h + ff);
  }

  /// A fresh, empty KV cache for this block's attention.
  TPMHACache newCache() => mha.newCache();

  @override
  List<Module> submodules() => [mha, mlp];

  @override
  List<Tensor> parameters() => trainable
      ? [
          ...mha.parameters(),
          ...mlp.parameters(),
          ln1Gamma,
          ln1Beta,
          ln2Gamma,
          ln2Beta,
        ]
      : const [];
}

/// Maps a [TransformerBlock] activation to the elementwise function the
/// tensor-parallel MLP applies to its (sharded) hidden activation.
Tensor Function(Tensor) _activationFn(Activation a) => switch (a) {
      Activation.relu => (t) => t.relu(),
      Activation.geluTanh => _geluTanh,
      Activation.quickGelu => _quickGelu,
    };

/// GPT-2 tanh-approximation GELU (mirrors `TransformerBlock._geluTanh`).
Tensor _geluTanh(Tensor x) {
  const c = 0.7978845608028654; // sqrt(2 / pi)
  final inner = (x + x.pow(3.0) * 0.044715) * c;
  final t = inner.tanh();
  return x * (t + 1.0) * 0.5;
}

/// OpenAI CLIP QuickGELU: `x * sigmoid(1.702 * x)`.
Tensor _quickGelu(Tensor x) => x * (x * 1.702).sigmoid();

/// Even split of `total` into `parts` contiguous chunks; earlier chunks
/// take the remainder. Returns `parts + 1` boundary offsets.
List<int> _splitOffsets(int total, int parts) {
  final offs = <int>[0];
  final base = total ~/ parts;
  var rem = total % parts;
  var acc = 0;
  for (var i = 0; i < parts; i++) {
    acc += base + (rem > 0 ? 1 : 0);
    if (rem > 0) rem--;
    offs.add(acc);
  }
  return offs;
}

/// A stack of [TensorParallelTransformerBlock]s — a full tensor-parallel
/// encoder. Supports two placement modes:
///
///   * **Tensor-parallel only** (default): every block is sharded across
///     *all* devices; the activation stays on one output device between
///     blocks.
///   * **Tensor + pipeline parallel** (`pipelineStages > 1`): the devices
///     are split into that many groups and the blocks into that many
///     contiguous stages, so stage `s`'s blocks are tensor-parallel
///     within device group `s`. The activation is handed from one stage's
///     device group to the next automatically (each block uploads its
///     input to its own output device). This lets a model with more
///     blocks than fit per card still run end to end.
///
/// Inference-oriented, 2D `[N, embedDim]`, ReLU FFN — same scope as the
/// block it stacks.
class TensorParallelTransformerStack extends Module {
  final List<TensorParallelTransformerBlock> blocks;

  TensorParallelTransformerStack(this.blocks);

  /// Shard a list of reference [blocks] across [devices] (defaults to all
  /// visible GPUs). With [pipelineStages] > 1, partition devices + blocks
  /// into that many pipeline stages (clamped to the device count). Pass
  /// `trainable: true` to make every block's sharded weights trainable.
  factory TensorParallelTransformerStack.fromBlocks(
    List<TransformerBlock> blocks, {
    List<int>? devices,
    int pipelineStages = 1,
    bool trainable = false,
  }) {
    if (blocks.isEmpty) {
      throw ArgumentError('TensorParallelTransformerStack: no blocks');
    }
    final devs = devices ?? List<int>.generate(Tensor.gpuCount, (i) => i);
    if (devs.isEmpty) {
      throw ArgumentError('TensorParallelTransformerStack: no devices');
    }

    final stages = pipelineStages.clamp(1, devs.length);
    if (stages <= 1) {
      // Pure tensor parallel: every block across all devices.
      return TensorParallelTransformerStack([
        for (final b in blocks)
          TensorParallelTransformerBlock.fromBlock(
            b,
            devices: devs,
            outputDevice: devs.first,
            trainable: trainable,
          ),
      ]);
    }

    // Pipeline: device group `s` holds the blocks of stage `s`.
    final devOffs = _splitOffsets(devs.length, stages);
    final blockOffs = _splitOffsets(blocks.length, stages);
    final out = <TensorParallelTransformerBlock>[];
    for (var s = 0; s < stages; s++) {
      var groupDevs = devs.sublist(devOffs[s], devOffs[s + 1]);
      if (groupDevs.isEmpty) groupDevs = [devs.last];
      for (var bi = blockOffs[s]; bi < blockOffs[s + 1]; bi++) {
        out.add(
          TensorParallelTransformerBlock.fromBlock(
            blocks[bi],
            devices: groupDevs,
            outputDevice: groupDevs.first,
            trainable: trainable,
          ),
        );
      }
    }
    return TensorParallelTransformerStack(out);
  }

  /// Device each block's output lands on (the hand-off points).
  List<int> get blockOutputDevices =>
      [for (final b in blocks) b.outputDevice];

  /// Fresh KV caches, one per block, for autoregressive decoding.
  List<TPMHACache> newCache() => [for (final b in blocks) b.newCache()];

  Tensor call(Tensor x,
      {Tensor? mask, List<TPMHACache>? caches, int startPos = 0}) {
    var h = x;
    for (var i = 0; i < blocks.length; i++) {
      h = blocks[i](h, mask: mask, cache: caches?[i], startPos: startPos);
    }
    return h;
  }

  @override
  List<Module> submodules() => blocks;

  @override
  List<Tensor> parameters() =>
      [for (final b in blocks) ...b.parameters()];
}

