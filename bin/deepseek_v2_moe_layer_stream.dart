/// AirLLM-style per-expert streaming on a **real** MoE layer from
/// `deepseek-ai/DeepSeek-V2-Lite-Chat`.
///
/// This is the smoke test of the pattern in
/// [bin/moe_streaming_demo.dart](moe_streaming_demo.dart) against a
/// production checkpoint. Loads only one MoE layer (--layer, default
/// 5), verifies the shapes match the config
/// (E=64, K=6, D=2048, moe_intermediate=1408, shared=2×1408),
/// streams the routed experts the router actually picks for a random
/// input, and reports the bytes-saved ratio.
///
/// Does **not** run a full end-to-end forward through the model —
/// that would require porting MLA + dense-vs-MoE layer switching +
/// persistent embed/head in the same shape as `LlamaStreamingRunner`.
/// See [doc/layer_streaming.md](../doc/layer_streaming.md).
///
///   dart run bin/deepseek_v2_moe_layer_stream.dart \
///     --index ~/models/deepseek-v2-lite-chat/model.safetensors.index.json \
///     --layer 5 --tokens 1 --seed 0
library;

import 'dart:io';
import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

// Config values pulled from DeepSeek-V2-Lite-Chat config.json.
const int _d = 2048;
const int _e = 64;
const int _k = 6;
const int _moeHidden = 1408;
const int _sharedExperts = 2;
const int _sharedHidden = _sharedExperts * _moeHidden;
const int _numLayers = 27;
const int _firstDense = 1;

Future<void> main(List<String> args) async {
  var indexPath =
      '${_defaultHome()}/models/deepseek-v2-lite-chat/'
      'model.safetensors.index.json';
  var layer = 5;
  var tokens = 1;
  var seed = 0;
  var reportOnly = false;
  for (int i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--index':
        indexPath = args[++i];
        break;
      case '--layer':
        layer = int.parse(args[++i]);
        break;
      case '--tokens':
        tokens = int.parse(args[++i]);
        break;
      case '--seed':
        seed = int.parse(args[++i]);
        break;
      case '--report-only':
        reportOnly = true;
        break;
      case '-h':
      case '--help':
        stdout.writeln(
          'usage: deepseek_v2_moe_layer_stream [--index P] '
          '[--layer N] [--tokens N] [--seed N] [--report-only]',
        );
        return;
    }
  }

  if (!File(indexPath).existsSync()) {
    stderr.writeln('missing: $indexPath');
    stderr.writeln('download first with:');
    stderr.writeln('  mkdir -p ~/models/deepseek-v2-lite-chat && cd \$_');
    stderr.writeln('  # then curl the config, index, and 4 shards from HF');
    exit(2);
  }
  if (layer < _firstDense || layer >= _numLayers) {
    stderr.writeln(
      'layer $layer is dense or out of range — MoE layers '
      'are [$_firstDense, $_numLayers)',
    );
    exit(2);
  }

  print('== DeepSeek-V2-Lite per-expert streaming (real weights) ==');
  print('  index          : $indexPath');
  print(
    '  layer          : $layer (of $_numLayers, first '
    '$_firstDense dense)',
  );
  print('  D (embed)      : $_d');
  print('  E (routed)     : $_e');
  print('  K (top-K)      : $_k');
  print('  hidden (routed): $_moeHidden');
  print(
    '  shared         : $_sharedExperts experts fused as SwiGLU '
    'hidden=$_sharedHidden',
  );
  print('  gate           : softmax + no-renormalize (V2 config)');
  print('  tokens         : $tokens');

  final swOpen = Stopwatch()..start();
  final reader = ShardedSafeTensorsReader.open(indexPath);
  swOpen.stop();
  print('  header parse   : ${swOpen.elapsedMilliseconds} ms');

  final p = 'model.layers.$layer.mlp';
  final gateKey = '$p.gate.weight';
  if (!reader.contains(gateKey)) {
    stderr.writeln('layer $layer is not a MoE layer (no "$gateKey")');
    exit(2);
  }

  // Verify layout on disk matches the config.
  final routedBytes = _reportShape(reader, '$p.experts.0.gate_proj.weight');
  final sharedGateBytes = _reportShape(
    reader,
    '$p.shared_experts.gate_proj.weight',
  );
  final gateWBytes = _reportShape(reader, gateKey);
  final expertBytes = 3 * routedBytes;
  final totalExpertsBytes = _e * expertBytes;
  final sharedBytes = 3 * sharedGateBytes;
  print('');
  print('== on-disk footprint (one layer) ==');
  print('  gate (router)  : ${_fmtBytes(gateWBytes)}');
  print(
    '  shared         : ${_fmtBytes(sharedBytes)} '
    '(3 × ${_fmtBytes(sharedGateBytes)})',
  );
  print(
    '  1 routed expert: ${_fmtBytes(expertBytes)} '
    '(3 × ${_fmtBytes(routedBytes)})',
  );
  print(
    '  all routed     : ${_fmtBytes(totalExpertsBytes)} '
    '($_e × ${_fmtBytes(expertBytes)})',
  );

  // Persistent load: router weight + shared experts. These are the
  // parts that a real DeepSeekV2StreamingRunner would keep across
  // token forwards in the resident block; we materialise them now to
  // exercise real bf16 → fp32 decode from disk.
  final swLoad = Stopwatch()..start();
  final gateW = reader
      .readTensor(gateKey)
      .transpose(); // HF ships [E, D]; we want [D, E]
  swLoad.stop();
  print('');
  print('== persistent load ==');
  print(
    '  gateW shape    : ${gateW.shape} '
    '(HF [E, D] transposed to [D, E], ${swLoad.elapsedMilliseconds} ms)',
  );

  final sharedGate = reader.readTensor('$p.shared_experts.gate_proj.weight');
  final sharedUp = reader.readTensor('$p.shared_experts.up_proj.weight');
  final sharedDown = reader.readTensor('$p.shared_experts.down_proj.weight');
  print(
    '  shared_gate    : ${sharedGate.shape} '
    '(expected [$_sharedHidden, $_d])',
  );
  print('  shared_up      : ${sharedUp.shape}');
  print(
    '  shared_down    : ${sharedDown.shape} '
    '(expected [$_d, $_sharedHidden])',
  );

  // A DeepSeekV2StreamingRunner would run MLA first and pass its
  // output to the FFN; here we test with random inputs to exercise
  // routing and per-expert streaming only.
  final rng = math.Random(seed);
  final xVals = List<double>.generate(
    tokens * _d,
    (_) => (rng.nextDouble() - 0.5) * 0.1,
  );
  final x = Tensor.fromList([tokens, _d], xVals, device: Device.CPU);

  // Route in fp32.
  final gateLogits = x.matmul(gateW); // [T, E]
  final scores = gateLogits.softmax();
  final flat = scores.toList();
  final used = <int>{};
  for (int i = 0; i < tokens; i++) {
    final indexed = List<MapEntry<int, double>>.generate(
      _e,
      (j) => MapEntry(j, flat[i * _e + j]),
    );
    indexed.sort((a, b) => b.value.compareTo(a.value));
    for (int r = 0; r < _k; r++) {
      used.add(indexed[r].key);
    }
  }
  final sortedUsed = used.toList()..sort();
  print('');
  print('== routing ==');
  print(
    '  top-K experts  : $sortedUsed (${used.length} of $_e = '
    '${(used.length * 100 / _e).toStringAsFixed(1)}%)',
  );
  final streamedBytes = used.length * expertBytes;
  print(
    '  streamed       : ${_fmtBytes(streamedBytes)} '
    '(vs ${_fmtBytes(totalExpertsBytes)} for full layer)',
  );
  print(
    '  saved          : '
    '${_fmtBytes(totalExpertsBytes - streamedBytes)} '
    '(${((totalExpertsBytes - streamedBytes) * 100 / totalExpertsBytes).toStringAsFixed(1)}%)',
  );

  if (reportOnly) {
    reader.close();
    return;
  }

  // Actually stream the selected experts into fp32 tensors, run
  // their SwiGLU FFN on the routed subset of x, weight-sum with the
  // gate scores, and add the shared branch.
  final swExperts = Stopwatch()..start();
  final expertGate = <int, Tensor>{};
  final expertUp = <int, Tensor>{};
  final expertDown = <int, Tensor>{};
  for (final j in sortedUsed) {
    expertGate[j] = reader.readTensor('$p.experts.$j.gate_proj.weight');
    expertUp[j] = reader.readTensor('$p.experts.$j.up_proj.weight');
    expertDown[j] = reader.readTensor('$p.experts.$j.down_proj.weight');
  }
  swExperts.stop();
  print(
    '  load           : ${swExperts.elapsedMilliseconds} ms '
    '(${used.length} experts)',
  );

  // Softmax + top-K mask (softmax weights sum to 1 across full E; we
  // multiply expert outputs by that weight but keep only top-K
  // active). V2 has norm_topk_prob=false so no renormalization.
  final maskVals = List<double>.filled(tokens * _e, 0.0);
  for (int i = 0; i < tokens; i++) {
    final indexed = List<MapEntry<int, double>>.generate(
      _e,
      (j) => MapEntry(j, flat[i * _e + j]),
    );
    indexed.sort((a, b) => b.value.compareTo(a.value));
    for (int r = 0; r < _k; r++) {
      maskVals[i * _e + indexed[r].key] = 1.0;
    }
  }

  final swF = Stopwatch()..start();
  var acc = _swiGluBatch(x, sharedGate, sharedUp, sharedDown); // shared
  for (final j in sortedUsed) {
    // Per-token weight for expert j: scores[:, j] * mask[:, j].
    final wjVals = List<double>.filled(tokens, 0.0);
    for (int i = 0; i < tokens; i++) {
      wjVals[i] = flat[i * _e + j] * maskVals[i * _e + j];
    }
    final wj = Tensor.fromList([tokens, 1], wjVals, device: Device.CPU);
    final onesD = Tensor.fill([1, _d], 1.0, device: Device.CPU);
    final wjBcast = wj.matmul(onesD); // [T, D]
    final expertOut = _swiGluBatch(
      x,
      expertGate[j]!,
      expertUp[j]!,
      expertDown[j]!,
    );
    acc = acc + (expertOut * wjBcast);
  }
  swF.stop();
  print('');
  print('== forward ==');
  print(
    '  wall           : ${swF.elapsedMilliseconds} ms (routed '
    'via ${used.length} experts + shared)',
  );
  print('  output shape   : ${acc.shape} (expected [$tokens, $_d])');

  final row = acc.toList();
  var mn = double.infinity;
  var mx = double.negativeInfinity;
  var sum = 0.0;
  var nan = 0;
  for (final v in row) {
    if (v.isNaN) {
      nan++;
      continue;
    }
    if (v < mn) mn = v;
    if (v > mx) mx = v;
    sum += v;
  }
  final finite = row.length - nan;
  print(
    '  stats          : min=${mn.toStringAsFixed(4)} '
    'max=${mx.toStringAsFixed(4)} '
    'mean=${(sum / (finite == 0 ? 1 : finite)).toStringAsFixed(6)} '
    'nan=$nan/${row.length}',
  );
  if (nan == 0) {
    print(
      '  status         : OK — per-expert streaming against real '
      'DeepSeek-V2-Lite weights produces finite outputs.',
    );
  }

  reader.close();
}

/// SwiGLU: `down(silu(gate(x)) * up(x))`. Weights are the on-disk
/// HF layout: gate/up are `[hidden, D]`, down is `[D, hidden]`.
Tensor _swiGluBatch(Tensor x, Tensor gW, Tensor uW, Tensor dW) {
  final gate = x.matmul(gW.transpose());
  final up = x.matmul(uW.transpose());
  final act = gate * gate.sigmoid();
  return (act * up).matmul(dW.transpose());
}

int _reportShape(ShardedSafeTensorsReader reader, String name) {
  final e = reader.entry(name);
  if (e == null) {
    stderr.writeln('missing tensor: $name');
    exit(3);
  }
  return e.dataEnd - e.dataStart;
}

String _fmtBytes(int b) {
  const units = ['B', 'KB', 'MB', 'GB'];
  var i = 0;
  double v = b.toDouble();
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return '${v.toStringAsFixed(2)} ${units[i]}';
}

String _defaultHome() =>
    Platform.environment['HOME'] ?? Platform.environment['USERPROFILE'] ?? '.';
