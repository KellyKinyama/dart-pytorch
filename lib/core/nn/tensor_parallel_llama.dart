/// Tensor-parallel Llama decoder — RMSNorm + GQA attention + SwiGLU,
/// sharded across several GPUs in one process.
///
/// Reuses [TensorParallelMultiHeadAttention] (which already handles GQA,
/// per-shard RoPE, and a KV cache) for the attention, adds a
/// tensor-parallel SwiGLU whose wide hidden dimension is split across
/// cards (gate/up column-wise, down row-wise), and replicates the two
/// RMSNorms on the output device. Built from the repo's reference
/// [LlamaBlock] / [Llama] objects, so a loaded checkpoint can be sharded
/// block-by-block. Inference- and training-capable (`trainable: true`),
/// 2D `[N, embedDim]` path, with an optional KV cache for decoding.
library;

import '../tensor/tensor.dart';
import 'ffn/swiglu.dart';
import 'llama.dart';
import 'module.dart';
import 'attention/tensor_parallel_attention.dart';

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

Tensor _sliceCols(Tensor w2d, int c0, int c1) {
  final rows = w2d.shape[0];
  final cols = w2d.shape[1];
  final src = w2d.toFloat32List();
  final w = c1 - c0;
  final out = List<double>.filled(rows * w, 0);
  for (var r = 0; r < rows; r++) {
    for (var j = 0; j < w; j++) {
      out[r * w + j] = src[r * cols + c0 + j];
    }
  }
  return Tensor.fromList([rows, w], out);
}

/// One GPU's slice of a SwiGLU FFN: the hidden-dim band `[h0, h1)` it
/// owns, as `gate`/`up` weights `[hBand, dim]` and a `down` weight
/// `[dim, hBand]`, all resident on [device].
class _SwiGluShard {
  _SwiGluShard(this.device, this.gateW, this.upW, this.downW);
  final int device;
  final Tensor gateW; // [hBand, dim]
  final Tensor upW; // [hBand, dim]
  final Tensor downW; // [dim, hBand]
}

/// Tensor-parallel SwiGLU: `down( silu(gate(x)) * up(x) )` with the wide
/// hidden dimension partitioned across GPUs. Each card computes its hidden
/// band locally (gate, up, SiLU, elementwise product) and contributes a
/// partial `down` projection; the partials are summed on [outputDevice].
class TensorParallelSwiGlu extends Module {
  final int dim;
  final int hiddenDim;
  final List<_SwiGluShard> shards;
  final int outputDevice;
  final bool trainable;

  TensorParallelSwiGlu._(
    this.dim,
    this.hiddenDim,
    this.shards,
    this.outputDevice,
    this.trainable,
  );

  /// Shard a reference [ffn] across [devices] (defaults to all GPUs).
  factory TensorParallelSwiGlu.fromSwiGlu(
    SwiGluFfn ffn, {
    List<int>? devices,
    int? outputDevice,
    bool trainable = false,
  }) {
    final devs = devices ?? List<int>.generate(Tensor.gpuCount, (i) => i);
    if (devs.isEmpty) {
      throw ArgumentError('TensorParallelSwiGlu: no devices');
    }
    final dim = ffn.dim;
    final hidden = ffn.hiddenDim;
    final outDev = outputDevice ?? devs.first;

    // Host copies for bit-exact slicing.
    final gateCpu = ffn.gateProj.weight.to(Device.CPU); // [hidden, dim]
    final upCpu = ffn.upProj.weight.to(Device.CPU); // [hidden, dim]
    final downCpu = ffn.downProj.weight.to(Device.CPU); // [dim, hidden]

    final g = devs.length < hidden ? devs.length : hidden;
    final offs = _splitOffsets(hidden, g);
    final shards = <_SwiGluShard>[];
    for (var d = 0; d < g; d++) {
      final h0 = offs[d];
      final h1 = offs[d + 1];
      if (h1 <= h0) continue;
      final dev = devs[d];
      final gateW = gateCpu.sliceRows(h0, h1).toGpu(dev);
      final upW = upCpu.sliceRows(h0, h1).toGpu(dev);
      final downW = _sliceCols(downCpu, h0, h1).toGpu(dev);
      if (trainable) {
        gateW.requiresGrad = true;
        upW.requiresGrad = true;
        downW.requiresGrad = true;
      }
      shards.add(_SwiGluShard(dev, gateW, upW, downW));
    }
    return TensorParallelSwiGlu._(dim, hidden, shards, outDev, trainable);
  }

  Tensor call(Tensor x) {
    Tensor? acc;
    for (final s in shards) {
      final dev = s.device;
      final xg = x.toGpu(dev);
      final partial = Tensor.onGpu(dev, () {
        final gate = xg.matmul(s.gateW.transpose());
        final up = xg.matmul(s.upW.transpose());
        final silu = gate * gate.sigmoid();
        final hs = silu * up; // sharded hidden stays on-card
        return hs.matmul(s.downW.transpose());
      });
      final onOut = partial.toGpu(outputDevice);
      acc = acc == null
          ? onOut
          : Tensor.onGpu(outputDevice, () => acc! + onOut);
    }
    return acc!;
  }

  @override
  List<Tensor> parameters() {
    if (!trainable) return const [];
    final ps = <Tensor>[];
    for (final s in shards) {
      ps..add(s.gateW)..add(s.upW)..add(s.downW);
    }
    return ps;
  }
}

/// Tensor-parallel Llama decoder block: pre-norm residuals with a
/// sharded GQA attention and a sharded SwiGLU, RMSNorms replicated on the
/// output device.
class TensorParallelLlamaBlock extends Module {
  final int embedDim;
  final int outputDevice;
  final double attnEps;
  final double ffnEps;
  final Tensor attnGamma; // RMSNorm weight on outputDevice
  final Tensor ffnGamma;
  final TensorParallelMultiHeadAttention attn;
  final TensorParallelSwiGlu ffn;

  TensorParallelLlamaBlock._(
    this.embedDim,
    this.outputDevice,
    this.attnEps,
    this.ffnEps,
    this.attnGamma,
    this.ffnGamma,
    this.attn,
    this.ffn,
  );

  /// Shard a reference [block] across [devices] (defaults to all GPUs).
  /// The block's RoPE cache is carried into the sharded attention.
  factory TensorParallelLlamaBlock.fromBlock(
    LlamaBlock block, {
    List<int>? devices,
    int? outputDevice,
    bool trainable = false,
  }) {
    final devs = devices ?? List<int>.generate(Tensor.gpuCount, (i) => i);
    final outDev = outputDevice ?? devs.first;

    final attn = TensorParallelMultiHeadAttention.fromAttention(
      block.attn,
      devices: devs,
      outputDevice: outDev,
      trainable: trainable,
    );
    final ffn = TensorParallelSwiGlu.fromSwiGlu(
      block.ffn,
      devices: devs,
      outputDevice: outDev,
      trainable: trainable,
    );

    Tensor gammaOnOut(Tensor t) {
      final r = t.detach().toGpu(outDev);
      if (trainable) r.requiresGrad = true;
      return r;
    }

    return TensorParallelLlamaBlock._(
      block.attnNorm.dim,
      outDev,
      block.attnNorm.eps,
      block.ffnNorm.eps,
      gammaOnOut(block.attnNorm.gamma),
      gammaOnOut(block.ffnNorm.gamma),
      attn,
      ffn,
    );
  }

  /// A fresh, empty KV cache for this block's attention.
  TPMHACache newCache() => attn.newCache();

  Tensor call(Tensor x, {Tensor? mask, TPMHACache? cache, int startPos = 0}) {
    final xo = x.toGpu(outputDevice);
    final an = Tensor.onGpu(outputDevice, () => xo.rmsNorm(attnGamma, eps: attnEps));
    final a = attn(an, mask: mask, cache: cache, startPos: startPos);
    final h = Tensor.onGpu(outputDevice, () => xo + a);
    final fn = Tensor.onGpu(outputDevice, () => h.rmsNorm(ffnGamma, eps: ffnEps));
    final m = ffn(fn);
    return Tensor.onGpu(outputDevice, () => h + m);
  }

  @override
  List<Module> submodules() => [attn, ffn];

  @override
  List<Tensor> parameters() => [
        ...attn.parameters(),
        ...ffn.parameters(),
        if (attnGamma.requiresGrad) attnGamma,
        if (ffnGamma.requiresGrad) ffnGamma,
      ];
}

/// A stack of [TensorParallelLlamaBlock]s with optional pipeline
/// parallelism across device groups (TP within a stage, PP across
/// stages). Supports a per-block KV cache for autoregressive decoding.
class TensorParallelLlamaStack extends Module {
  final List<TensorParallelLlamaBlock> blocks;

  TensorParallelLlamaStack(this.blocks);

  factory TensorParallelLlamaStack.fromBlocks(
    List<LlamaBlock> blocks, {
    List<int>? devices,
    int pipelineStages = 1,
    bool trainable = false,
  }) {
    if (blocks.isEmpty) {
      throw ArgumentError('TensorParallelLlamaStack: no blocks');
    }
    final devs = devices ?? List<int>.generate(Tensor.gpuCount, (i) => i);
    if (devs.isEmpty) {
      throw ArgumentError('TensorParallelLlamaStack: no devices');
    }
    final stages = pipelineStages.clamp(1, devs.length);
    if (stages <= 1) {
      return TensorParallelLlamaStack([
        for (final b in blocks)
          TensorParallelLlamaBlock.fromBlock(
            b,
            devices: devs,
            outputDevice: devs.first,
            trainable: trainable,
          ),
      ]);
    }
    final devOffs = _splitOffsets(devs.length, stages);
    final blockOffs = _splitOffsets(blocks.length, stages);
    final out = <TensorParallelLlamaBlock>[];
    for (var s = 0; s < stages; s++) {
      var groupDevs = devs.sublist(devOffs[s], devOffs[s + 1]);
      if (groupDevs.isEmpty) groupDevs = [devs.last];
      for (var bi = blockOffs[s]; bi < blockOffs[s + 1]; bi++) {
        out.add(
          TensorParallelLlamaBlock.fromBlock(
            blocks[bi],
            devices: groupDevs,
            outputDevice: groupDevs.first,
            trainable: trainable,
          ),
        );
      }
    }
    return TensorParallelLlamaStack(out);
  }

  List<int> get blockOutputDevices => [for (final b in blocks) b.outputDevice];

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
  List<Tensor> parameters() => [for (final b in blocks) ...b.parameters()];
}
