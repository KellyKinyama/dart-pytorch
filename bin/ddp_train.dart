/// Distributed data-parallel (DDP) training for `dart_pytorch`, ported from
/// nanoGPT's `train.py`. Each rank trains the same model on a different data
/// shard; gradients are averaged across ranks every step (see [Dist]), so all
/// replicas stay in lock-step — the exact mechanic `torch.nn.parallel.DDP`
/// provides, minus NCCL.
///
/// Single process (no DDP):
///   dart run bin/ddp_train.dart
///
/// Multiple local ranks (one process each) via the launcher:
///   dart run bin/ddp_launch.dart 4
///
/// Across servers (run on each host, like torchrun's multi-node example):
///   RANK=0 WORLD_SIZE=8 LOCAL_RANK=0 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
///     dart run bin/ddp_train.dart           # ... node 0 ranks ...
///   RANK=4 WORLD_SIZE=8 LOCAL_RANK=0 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
///     dart run bin/ddp_train.dart           # ... node 1 ranks ...
///
/// GPU pinning: set CUDA_VISIBLE_DEVICES=`<LOCAL_RANK>` per process so each rank's
/// default CUDA device maps to a distinct physical GPU. This demo forces CPU so
/// it runs anywhere; the DDP logic is identical on GPU.
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';

import 'ddp_dist.dart';

const _corpus =
    'to every thing there is a season and a time to every purpose under the heaven. '
    'a time to be born and a time to die. a time to plant and a time to pluck up '
    'that which is planted. a time to kill and a time to heal. a time to break down '
    'and a time to build up. a time to weep and a time to laugh. a time to mourn '
    'and a time to dance. a time to cast away stones and a time to gather stones '
    'together. a time to embrace and a time to refrain from embracing. a time to '
    'get and a time to lose. a time to keep and a time to cast away. a time to '
    'rend and a time to sew. a time to keep silence and a time to speak. ';

Future<void> main() async {
  // Keep the demo CPU-only and deterministic across ranks.
  Tensor.disableAutoGpu = true;

  final dist = await Dist.init();
  void log(String m) {
    if (dist.isMaster) print(m);
  }

  log('=== dart-pytorch DDP training (world_size=${dist.worldSize}) ===');

  // ---- tokenizer (identical on every rank: same corpus + deterministic BPE) ----
  final tok = BpeTokenizer.train(_corpus, targetVocabSize: 320);
  final ids = tok.encode(_corpus);

  // ---- model (same seed → identical init on every rank) ----
  const maxCtx = 32;
  final gpt = GPT(GPTConfig(
    vocabSize: tok.vocabSize,
    maxCtx: maxCtx,
    embedDim: 32,
    numLayers: 2,
    numHeads: 4,
    dropoutP: 0.0,
    tieWeights: true,
    seed: 1,
  ));
  final params = gpt.parameters();
  final totalScalars = params.fold<int>(0, (a, p) => a + p.length);
  final flat = Float32List(totalScalars);

  // Optional resume: rank 0 loads weights from a checkpoint; the broadcast
  // below then propagates them to every rank.
  final resumePath = Platform.environment['DDP_RESUME'];
  if (resumePath != null && dist.isMaster && File(resumePath).existsSync()) {
    Checkpoint.loadIntoFile(gpt, resumePath);
    log('resumed weights from $resumePath');
  }

  // Broadcast rank 0's weights so every replica starts byte-identical
  // (whether freshly initialized or just resumed).
  _gather(params, flat, grads: false);
  await dist.broadcastFromMaster(flat);
  _scatter(params, flat, grads: false);

  // ---- data sharding: each rank gets a disjoint shard of a shared shuffle ----
  final trainable = ids.length - maxCtx - 1;
  const microBatch = 4; // per-rank micro-batch; effective batch scales with world
  final perRank = trainable ~/ dist.worldSize; // disjoint, equal-size shards
  final batchesPerEpoch = perRank ~/ microBatch; // identical on every rank
  if (batchesPerEpoch == 0) {
    throw StateError('corpus too small for world_size=${dist.worldSize} '
        '(trainable=$trainable, microBatch=$microBatch)');
  }
  const targetSteps = 160;
  final epochs = math.max(1, targetSteps ~/ batchesPerEpoch);
  final totalSteps = epochs * batchesPerEpoch;

  // Gradient all-reduce bucket size (scalars). Bounds per-message size for
  // large models; defaults to one bucket (the whole gradient buffer).
  final bucketScalars =
      int.tryParse(Platform.environment['DDP_BUCKET_SCALARS'] ?? '') ??
          totalScalars;

  // ---- optimizer + schedule ----
  final warmupSteps = math.max(1, (totalSteps * 0.1).round());
  final opt = Adam(params, lr: 0.0);
  final sched = LinearWarmupCosineDecay(
    opt,
    warmupSteps: warmupSteps,
    totalSteps: totalSteps,
    maxLr: 3e-3,
    minLr: 3e-4,
  );
  log('params: ${params.length} tensors / $totalScalars scalars  '
      'effective batch: ${microBatch * dist.worldSize}');
  log('data: trainable=$trainable  perRank=$perRank  '
      'epochs=$epochs x $batchesPerEpoch batches = $totalSteps steps');

  final xBuf = List<double>.filled(microBatch * maxCtx, 0.0);
  final yBuf = List<double>.filled(microBatch * maxCtx, 0.0);
  final sw = Stopwatch()..start();

  var step = 0;
  for (var epoch = 0; epoch < epochs; epoch++) {
    // Deterministic, identical shuffle on every rank, then disjoint stride
    // selection — rank r takes positions r, r+W, r+2W, ... (trimmed equal).
    final order = List<int>.generate(trainable, (i) => i)
      ..shuffle(math.Random(1234 + epoch));
    final shard = <int>[];
    for (var i = dist.rank; i < order.length && shard.length < perRank; i += dist.worldSize) {
      shard.add(order[i]);
    }

    for (var bIdx = 0; bIdx < batchesPerEpoch; bIdx++) {
      step++;
      opt.zeroGrad();

      for (var b = 0; b < microBatch; b++) {
        final start = shard[bIdx * microBatch + b];
        for (var t = 0; t < maxCtx; t++) {
          xBuf[b * maxCtx + t] = ids[start + t].toDouble();
          yBuf[b * maxCtx + t] = ids[start + t + 1].toDouble();
        }
      }
      final x = Tensor.fromList([microBatch, maxCtx], List<double>.from(xBuf));
      final y = Tensor.fromList([microBatch, maxCtx], List<double>.from(yBuf));

      final loss = gpt(x).crossEntropy(y).mean();
      loss.backward();
      final localLoss = loss.toList()[0];

      // --- DDP gradient all-reduce: average grads across all ranks ---
      _gather(params, flat, grads: true);
      for (var off = 0; off < flat.length; off += bucketScalars) {
        final end = math.min(off + bucketScalars, flat.length);
        await dist.allReduceMean(Float32List.sublistView(flat, off, end));
      }
      _scatter(params, flat, grads: true);

      clipGradNorm(params, 1.0);
      opt.step();
      sched.step();

      if (dist.isMaster && (step == 1 || step % 40 == 0 || step == totalSteps)) {
        log('  step ${step.toString().padLeft(3)}/$totalSteps  '
            'epoch $epoch  loss=${localLoss.toStringAsFixed(4)}  '
            'lr=${opt.lr.toStringAsExponential(2)}');
      }
    }
    await dist.barrier(); // all ranks finish the epoch together
  }
  sw.stop();

  // Optional checkpoint: rank 0 saves after everyone is done.
  await dist.barrier();
  final savePath = Platform.environment['DDP_SAVE'];
  if (savePath != null && dist.isMaster) {
    Checkpoint.saveFile(gpt, savePath);
    log('saved checkpoint to $savePath');
  }

  // Every rank now holds identical weights; print a checksum to prove it.
  _gather(params, flat, grads: false);
  final checksum = flat.fold<double>(0.0, (a, v) => a + v);
  print('[rank ${dist.rank}] done in ${sw.elapsedMilliseconds} ms  '
      'param_checksum=${checksum.toStringAsFixed(6)}');

  await dist.close();
}

/// Flattens each param's data (or `.grad`) into [flat].
void _gather(List<Tensor> params, Float32List flat, {required bool grads}) {
  var off = 0;
  for (final p in params) {
    final src = grads ? p.grad : p;
    if (src != null) {
      flat.setAll(off, src.toFloat32List());
    } else {
      flat.fillRange(off, off + p.length, 0.0);
    }
    off += p.length;
  }
}

/// Writes [flat] back into each param's data (or `.grad`) via in-place assign.
void _scatter(List<Tensor> params, Float32List flat, {required bool grads}) {
  var off = 0;
  for (final p in params) {
    final n = p.length;
    final slice = Float32List.sublistView(flat, off, off + n);
    if (grads) {
      p.grad?.assign(Tensor.fromFloat32List(p.shape, slice));
    } else {
      p.assign(Tensor.fromFloat32List(p.shape, slice));
    }
    off += n;
  }
}
