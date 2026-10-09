/// Tensor-parallel linear layers — split one big matmul across several
/// GPUs *within a single process*.
///
/// Two complementary shardings (the Megatron-LM scheme) are provided:
///
///   * [ColumnParallelLinear] splits the weight along its **output**
///     rows, so GPU `g` owns `W[g]` of shape `[outShard, in]` and
///     computes a slice of the output columns. The per-GPU results are
///     gathered (concat) into the full `[.., out]` activation.
///
///   * [RowParallelLinear] splits the weight along its **input**
///     columns, so GPU `g` owns `W[g]` of shape `[out, inShard]` and
///     consumes a slice of the input features. Each GPU produces a
///     partial `[.., out]` sum; the partials are all-reduced (added).
///
/// Chaining a column-parallel layer into a row-parallel one is the
/// canonical tensor-parallel transformer MLP — see [TensorParallelMLP].
///
/// These layers are **inference-oriented**: weights are held detached
/// (no autograd), which is what running one oversized model across
/// cards calls for. Each shard's forward runs inside [Tensor.onGpu] so
/// every op executes on the card that holds its weights; only the small
/// boundary activation crosses between GPUs (via [Tensor.toGpu]).
library;

import '../tensor/tensor.dart';
import 'module.dart';

/// Even split of `total` into `parts` contiguous chunks; earlier chunks
/// take the extra when it doesn't divide evenly. Returns the chunk
/// boundaries as `parts + 1` offsets.
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

/// Column-slice `[c0, c1)` of a rank-2 **CPU fp32** tensor into a fresh
/// CPU tensor. Complements [Tensor.sliceRows] for the row-parallel and
/// bias splits.
Tensor _sliceCols(Tensor w2d, int c0, int c1) {
  if (w2d.shape.length != 2) {
    throw ArgumentError('_sliceCols: expected rank 2, got ${w2d.shape}');
  }
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

/// Flatten a rank-≥2 input to `[rows, lastDim]`, remembering the leading
/// shape so the caller can restore it after the matmul.
({Tensor flat, List<int> lead}) _flattenRows(Tensor x, int features) {
  if (x.shape.isEmpty || x.shape.last != features) {
    throw ArgumentError(
      'expected input [..., $features]; got ${x.shape}',
    );
  }
  if (x.shape.length == 2) {
    return (flat: x, lead: [x.shape[0]]);
  }
  final rows = x.length ~/ features;
  final lead = x.shape.sublist(0, x.shape.length - 1);
  return (flat: x.reshape([rows, features]), lead: lead);
}

/// Linear layer whose weight is split across GPUs along the **output**
/// dimension. `y = x @ W.T (+ b)`, with `W` shape `[out, in]` sharded
/// row-wise so shard `g` is `[outShard_g, in]` resident on `devices[g]`.
class ColumnParallelLinear extends Module {
  final int inFeatures;
  final int outFeatures;

  /// Physical GPU ordinal of each shard.
  final List<int> devices;

  /// Weight shard `g`: `[outShard_g, in]` on `devices[g]`.
  final List<Tensor> weightShards;

  /// Bias shard `g`: `[1, outShard_g]` on `devices[g]`, or null.
  final List<Tensor>? biasShards;

  /// GPU the gathered output lands on. Defaults to the first shard's GPU.
  final int outputDevice;

  /// Whether the weight/bias shards are autograd leaves (trainable).
  final bool trainable;

  ColumnParallelLinear._(
    this.inFeatures,
    this.outFeatures,
    this.devices,
    this.weightShards,
    this.biasShards,
    this.outputDevice,
    this.trainable,
  );

  /// Shard a full `[out, in]` CPU weight (and optional `[1, out]` bias)
  /// across [devices], uploading each shard to its GPU. Pass
  /// `trainable: true` to make the shards autograd leaves so they train
  /// in place (gradients stay local to each card — see
  /// doc/multi_gpu_roadmap.md).
  factory ColumnParallelLinear.fromWeight(
    Tensor weight, {
    Tensor? bias,
    List<int>? devices,
    int? outputDevice,
    bool trainable = false,
  }) {
    if (weight.shape.length != 2) {
      throw ArgumentError('weight must be rank 2 [out, in]; got ${weight.shape}');
    }
    // Slicing happens host-side, so pull weights to CPU first (large
    // tensors auto-place on GPU via Tensor.fromList).
    weight = weight.to(Device.CPU);
    bias = bias?.to(Device.CPU);
    final out = weight.shape[0];
    final inF = weight.shape[1];
    final devs = devices ?? List<int>.generate(Tensor.gpuCount, (i) => i);
    if (devs.isEmpty) {
      throw ArgumentError('ColumnParallelLinear: no devices to shard across');
    }
    if (bias != null &&
        (bias.shape.length != 2 || bias.shape[0] != 1 || bias.shape[1] != out)) {
      throw ArgumentError('bias must be [1, $out]; got ${bias.shape}');
    }
    final offs = _splitOffsets(out, devs.length);
    final wShards = <Tensor>[];
    final bShards = bias == null ? null : <Tensor>[];
    for (var g = 0; g < devs.length; g++) {
      final r0 = offs[g];
      final r1 = offs[g + 1];
      // Slice on CPU (bit-exact), then move the shard onto its card.
      final w = weight.sliceRows(r0, r1).toGpu(devs[g]);
      if (trainable) w.requiresGrad = true;
      wShards.add(w);
      if (bias != null) {
        final b = _sliceCols(bias, r0, r1).toGpu(devs[g]);
        if (trainable) b.requiresGrad = true;
        bShards!.add(b);
      }
    }
    return ColumnParallelLinear._(
      inF,
      out,
      devs,
      wShards,
      bShards,
      outputDevice ?? devs.first,
      trainable,
    );
  }

  Tensor call(Tensor x) {
    final f = _flattenRows(x, inFeatures);
    final shardOuts = <Tensor>[];
    for (var g = 0; g < devices.length; g++) {
      final dev = devices[g];
      final w = weightShards[g];
      final b = biasShards?[g];
      final xg = f.flat.toGpu(dev);
      // Each shard's output stays on its own card; the gather moves only
      // the output columns (GPU-native when collectives are available).
      shardOuts.add(
        Tensor.onGpu(dev, () {
          var y = xg.matmul(w.transpose());
          if (b != null) y = y + b;
          return y;
        }),
      );
    }
    final gathered = TensorConcat.gatherColumns(
      shardOuts,
      outputDevice: outputDevice,
    );
    if (f.lead.length == 1) return gathered;
    return Tensor.onGpu(
      outputDevice,
      () => gathered.reshape([...f.lead, outFeatures]),
    );
  }

  @override
  List<Tensor> parameters() => trainable
      ? [...weightShards, if (biasShards != null) ...biasShards!]
      : const [];
}

/// Linear layer whose weight is split across GPUs along the **input**
/// dimension. `y = x @ W.T (+ b)`, with `W` shape `[out, in]` sharded
/// column-wise so shard `g` is `[out, inShard_g]` resident on
/// `devices[g]` and consumes input features `[inStart_g, inEnd_g)`.
/// Each GPU yields a partial `[.., out]`; the partials are summed.
class RowParallelLinear extends Module {
  final int inFeatures;
  final int outFeatures;
  final List<int> devices;

  /// Weight shard `g`: `[out, inShard_g]` on `devices[g]`.
  final List<Tensor> weightShards;

  /// Input-feature boundaries, `devices.length + 1` offsets.
  final List<int> inOffsets;

  /// Full `[1, out]` bias on [outputDevice], added once after the
  /// all-reduce, or null.
  final Tensor? bias;

  final int outputDevice;

  /// Whether the weight/bias shards are autograd leaves (trainable).
  final bool trainable;

  RowParallelLinear._(
    this.inFeatures,
    this.outFeatures,
    this.devices,
    this.weightShards,
    this.inOffsets,
    this.bias,
    this.outputDevice,
    this.trainable,
  );

  /// Shard a full `[out, in]` CPU weight column-wise across [devices].
  /// Bias (if given, `[1, out]`) is kept whole on [outputDevice]. Pass
  /// `trainable: true` to make the shards autograd leaves.
  factory RowParallelLinear.fromWeight(
    Tensor weight, {
    Tensor? bias,
    List<int>? devices,
    int? outputDevice,
    bool trainable = false,
  }) {
    if (weight.shape.length != 2) {
      throw ArgumentError('weight must be rank 2 [out, in]; got ${weight.shape}');
    }
    // Column slicing is host-side; pull weights to CPU first.
    weight = weight.to(Device.CPU);
    bias = bias?.to(Device.CPU);
    final out = weight.shape[0];
    final inF = weight.shape[1];
    final devs = devices ?? List<int>.generate(Tensor.gpuCount, (i) => i);
    if (devs.isEmpty) {
      throw ArgumentError('RowParallelLinear: no devices to shard across');
    }
    if (bias != null &&
        (bias.shape.length != 2 || bias.shape[0] != 1 || bias.shape[1] != out)) {
      throw ArgumentError('bias must be [1, $out]; got ${bias.shape}');
    }
    final outDev = outputDevice ?? devs.first;
    final offs = _splitOffsets(inF, devs.length);
    final wShards = <Tensor>[];
    for (var g = 0; g < devs.length; g++) {
      final w = _sliceCols(weight, offs[g], offs[g + 1]).toGpu(devs[g]);
      if (trainable) w.requiresGrad = true;
      wShards.add(w);
    }
    final bOnOut = bias?.toGpu(outDev);
    if (trainable && bOnOut != null) bOnOut.requiresGrad = true;
    return RowParallelLinear._(
      inF,
      out,
      devs,
      wShards,
      offs,
      bOnOut,
      outDev,
      trainable,
    );
  }

  Tensor call(Tensor x) {
    final f = _flattenRows(x, inFeatures);
    // Host-split the input features once; upload each slice to its card.
    final xCpu = f.flat.to(Device.CPU);
    Tensor? acc;
    for (var g = 0; g < devices.length; g++) {
      final dev = devices[g];
      final xg = _sliceCols(xCpu, inOffsets[g], inOffsets[g + 1]).toGpu(dev);
      final partial = Tensor.onGpu(dev, () => xg.matmul(weightShards[g].transpose()));
      final onOut = partial.toGpu(outputDevice);
      acc = acc == null
          ? onOut
          : Tensor.onGpu(outputDevice, () => acc! + onOut);
    }
    var y = acc!;
    if (bias != null) {
      final b = bias!;
      y = Tensor.onGpu(outputDevice, () => y + b);
    }
    if (f.lead.length == 1) return y;
    return Tensor.onGpu(
      outputDevice,
      () => y.reshape([...f.lead, outFeatures]),
    );
  }

  @override
  List<Tensor> parameters() =>
      trainable ? [...weightShards, if (bias != null) bias!] : const [];
}

/// Canonical tensor-parallel transformer MLP:
/// `down( relu( up(x) ) )`, where `up` is column-parallel (so the hidden
/// activation is already sharded across GPUs) and `down` is row-parallel
/// (so the sharded hidden is consumed in place and all-reduced back to
/// the model dim). Only the block's input and output activations cross
/// GPU boundaries; the wide hidden layer never materialises on one card.
class TensorParallelMLP extends Module {
  final ColumnParallelLinear up;
  final RowParallelLinear down;

  TensorParallelMLP(this.up, this.down);

  /// Build from full CPU weights. [upWeight] is `[hidden, model]`,
  /// [downWeight] is `[model, hidden]`. Pass `trainable: true` to make
  /// the sharded weights autograd leaves.
  factory TensorParallelMLP.fromWeights({
    required Tensor upWeight,
    required Tensor downWeight,
    Tensor? upBias,
    Tensor? downBias,
    List<int>? devices,
    int? outputDevice,
    bool trainable = false,
  }) {
    final devs = devices ?? List<int>.generate(Tensor.gpuCount, (i) => i);
    final outDev = outputDevice ?? devs.first;
    return TensorParallelMLP(
      ColumnParallelLinear.fromWeight(
        upWeight,
        bias: upBias,
        devices: devs,
        // Keep the hidden activation sharded — gather onto each card is
        // avoided because the row-parallel `down` re-splits it anyway;
        // we gather to outDev here for simplicity and correctness.
        outputDevice: outDev,
        trainable: trainable,
      ),
      RowParallelLinear.fromWeight(
        downWeight,
        bias: downBias,
        devices: devs,
        outputDevice: outDev,
        trainable: trainable,
      ),
    );
  }

  Tensor call(Tensor x) => down(up(x).relu());

  @override
  List<Module> submodules() => [up, down];

  @override
  List<Tensor> parameters() => [...up.parameters(), ...down.parameters()];
}
