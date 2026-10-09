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
  );

  /// Shard a reference [block] across [devices] (defaults to all visible
  /// GPUs). The FFN activation must be ReLU (the default). The result is
  /// a detached, inference-only block numerically equivalent to [block].
  factory TensorParallelTransformerBlock.fromBlock(
    TransformerBlock block, {
    List<int>? devices,
    int? outputDevice,
  }) {
    if (block.activation != Activation.relu) {
      throw ArgumentError(
        'TensorParallelTransformerBlock: only the ReLU FFN is supported '
        '(got ${block.activation}).',
      );
    }
    final devs = devices ?? List<int>.generate(Tensor.gpuCount, (i) => i);
    final outDev = outputDevice ?? devs.first;

    final mha = TensorParallelMultiHeadAttention.fromAttention(
      block.mha,
      devices: devs,
      outputDevice: outDev,
    );
    final mlp = TensorParallelMLP.fromWeights(
      upWeight: block.ffn1.weight,
      upBias: block.ffn1.bias,
      downWeight: block.ffn2.weight,
      downBias: block.ffn2.bias,
      devices: devs,
      outputDevice: outDev,
    );

    Tensor onOut(Tensor t) => t.detach().toGpu(outDev);
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
    );
  }

  /// Forward over a 2D `[N, embedDim]` sequence with optional additive
  /// mask `[N, N]`. Returns `[N, embedDim]` on [outputDevice].
  Tensor call(Tensor x, {Tensor? mask}) {
    final xo = x.toGpu(outputDevice);
    final normed1 = Tensor.onGpu(
      outputDevice,
      () => xo.layerNorm(ln1Gamma, ln1Beta, eps: eps),
    );
    final attn = mha(normed1, mask: mask);
    final h = Tensor.onGpu(outputDevice, () => xo + attn);

    final normed2 = Tensor.onGpu(
      outputDevice,
      () => h.layerNorm(ln2Gamma, ln2Beta, eps: eps),
    );
    final ff = mlp(normed2);
    return Tensor.onGpu(outputDevice, () => h + ff);
  }

  @override
  List<Module> submodules() => [mha, mlp];

  @override
  List<Tensor> parameters() => const [];
}

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
  /// into that many pipeline stages (clamped to the device count).
  factory TensorParallelTransformerStack.fromBlocks(
    List<TransformerBlock> blocks, {
    List<int>? devices,
    int pipelineStages = 1,
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
          ),
        );
      }
    }
    return TensorParallelTransformerStack(out);
  }

  /// Device each block's output lands on (the hand-off points).
  List<int> get blockOutputDevices =>
      [for (final b in blocks) b.outputDevice];

  Tensor call(Tensor x, {Tensor? mask}) {
    var h = x;
    for (final b in blocks) {
      h = b(h, mask: mask);
    }
    return h;
  }

  @override
  List<Module> submodules() => blocks;

  @override
  List<Tensor> parameters() => const [];
}

