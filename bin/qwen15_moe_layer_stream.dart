/// AirLLM-style per-expert streaming on a real MoE layer of
/// `Qwen/Qwen1.5-MoE-A2.7B-Chat`.
///
/// Same shape as [bin/deepseek_v2_moe_layer_stream.dart] but for
/// Qwen2-MoE architecture, which uses plain MHA (not MLA), all-MoE
/// layers, and a single shared expert with `hidden = 5632` gated by
/// a learned scalar (`shared_expert_gate.weight`).
///
///   dart run bin/qwen15_moe_layer_stream.dart \
///     --index ~/models/qwen1.5-moe-a2.7b-chat/model.safetensors.index.json \
///     --layer 0 --tokens 1 --seed 0
///
/// Layer 0 fits entirely in shard 1, so this demo works once the
/// first shard of the checkpoint has finished downloading.
library;

import 'dart:io';
import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

// Config values pulled from Qwen1.5-MoE-A2.7B-Chat config.json.
const int _d = 2048;
const int _e = 60;
const int _k = 4;
const int _moeHidden = 1408;
const int _sharedHidden = 5632; // shared_expert_intermediate_size
const int _numLayers = 24;

Future<void> main(List<String> args) async {
  var indexPath = '${_defaultHome()}/models/qwen1.5-moe-a2.7b-chat/'
      'model.safetensors.index.json';
  var layer = 0;
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
          'usage: qwen15_moe_layer_stream [--index P] [--layer N] '
          '[--tokens N] [--seed N] [--report-only]',
        );
        return;
    }
  }

  if (!File(indexPath).existsSync()) {
    stderr.writeln('missing: $indexPath');
    stderr.writeln('download: hf download Qwen/Qwen1.5-MoE-A2.7B-Chat '
        '--local-dir ~/models/qwen1.5-moe-a2.7b-chat');
    exit(2);
  }
  if (layer < 0 || layer >= _numLayers) {
    stderr.writeln('layer $layer out of range [0, $_numLayers)');
    exit(2);
  }

  print('== Qwen1.5-MoE-A2.7B per-expert streaming (real weights) ==');
  print('  index          : $indexPath');
  print('  layer          : $layer (of $_numLayers, all MoE)');
  print('  D (embed)      : $_d');
  print('  E (routed)     : $_e');
  print('  K (top-K)      : $_k');
  print('  hidden (routed): $_moeHidden');
  print('  shared         : 1 fused expert hidden=$_sharedHidden '
      '(+ shared_expert_gate scalar)');
  print('  gate           : softmax + no-renormalize (Qwen2-MoE)');
  print('  tokens         : $tokens');

  final swOpen = Stopwatch()..start();
  final reader = ShardedSafeTensorsReader.open(indexPath);
  swOpen.stop();
  print('  header parse   : ${swOpen.elapsedMilliseconds} ms');

  final p = 'model.layers.$layer.mlp';
  final gateKey = '$p.gate.weight';
  if (!reader.contains(gateKey)) {
    stderr.writeln('no "$gateKey" — layer $layer might not be in a '
        'downloaded shard yet, or this checkpoint has a different layout');
    exit(2);
  }
  // Verify the layer's shard is actually on disk (hf writes 0-byte
  // placeholders in the local-dir before download completes).
  for (final key in [
    gateKey,
    '$p.experts.0.gate_proj.weight',
    '$p.shared_expert.gate_proj.weight',
  ]) {
    try {
      reader.entry(key);
    } catch (_) {
      // fall-through; the read below will fail with a better message.
    }
  }

  // Report on-disk shapes.
  final routedBytes = _bytesOf(reader, '$p.experts.0.gate_proj.weight');
  final sharedGateBytes =
      _bytesOf(reader, '$p.shared_expert.gate_proj.weight');
  final gateWBytes = _bytesOf(reader, gateKey);
  final expertBytes = 3 * routedBytes;
  final totalExpertsBytes = _e * expertBytes;
  final sharedBytes = 3 * sharedGateBytes;
  print('');
  print('== on-disk footprint (one layer) ==');
  print('  gate (router)  : ${_fmtBytes(gateWBytes)}');
  print('  shared         : ${_fmtBytes(sharedBytes)}');
  print('  1 routed expert: ${_fmtBytes(expertBytes)} '
      '(3 × ${_fmtBytes(routedBytes)})');
  print('  all routed     : ${_fmtBytes(totalExpertsBytes)} '
      '($_e × ${_fmtBytes(expertBytes)})');

  final swLoad = Stopwatch()..start();
  // HF gate.weight shape is [E, D]; transpose to [D, E] for x @ gW.
  final gateW = reader.readTensor(gateKey).transpose();
  swLoad.stop();
  print('');
  print('== persistent load ==');
  print('  gateW shape    : ${gateW.shape} '
      '(${swLoad.elapsedMilliseconds} ms)');

  final sharedGate = reader.readTensor('$p.shared_expert.gate_proj.weight');
  final sharedUp = reader.readTensor('$p.shared_expert.up_proj.weight');
  final sharedDown = reader.readTensor('$p.shared_expert.down_proj.weight');
  final sharedExpertGate =
      reader.readTensor('$p.shared_expert_gate.weight'); // [1, D]
  print('  shared_gate    : ${sharedGate.shape} '
      '(expected [$_sharedHidden, $_d])');
  print('  shared_down    : ${sharedDown.shape} '
      '(expected [$_d, $_sharedHidden])');
  print('  shared_expert_gate: ${sharedExpertGate.shape} '
      '(expected [1, $_d])');

  final rng = math.Random(seed);
  final xVals = List<double>.generate(
    tokens * _d,
    (_) => (rng.nextDouble() - 0.5) * 0.1,
  );
  final x = Tensor.fromList([tokens, _d], xVals, device: Device.CPU);

  // Route.
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
  print('  top-K experts  : $sortedUsed (${used.length} of $_e = '
      '${(used.length * 100 / _e).toStringAsFixed(1)}%)');
  final streamedBytes = used.length * expertBytes;
  print('  streamed       : ${_fmtBytes(streamedBytes)} '
      '(vs ${_fmtBytes(totalExpertsBytes)} for full layer)');
  print('  saved          : '
      '${_fmtBytes(totalExpertsBytes - streamedBytes)} '
      '(${((totalExpertsBytes - streamedBytes) * 100 / totalExpertsBytes).toStringAsFixed(1)}%)');

  if (reportOnly) {
    reader.close();
    return;
  }

  final swExperts = Stopwatch()..start();
  final eGate = <int, Tensor>{};
  final eUp = <int, Tensor>{};
  final eDown = <int, Tensor>{};
  for (final j in sortedUsed) {
    eGate[j] = reader.readTensor('$p.experts.$j.gate_proj.weight');
    eUp[j] = reader.readTensor('$p.experts.$j.up_proj.weight');
    eDown[j] = reader.readTensor('$p.experts.$j.down_proj.weight');
  }
  swExperts.stop();
  print('  load           : ${swExperts.elapsedMilliseconds} ms '
      '(${used.length} experts)');

  // Top-K mask (softmax scores, no renormalize).
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

  // Shared expert output, scalar-gated by sigmoid(x @ shared_expert_gate^T).
  final sharedOut = _swiGlu(x, sharedGate, sharedUp, sharedDown); // [T, D]
  final sharedGateLogit = x.matmul(sharedExpertGate.transpose()); // [T, 1]
  final sharedGateScore = sharedGateLogit.sigmoid();
  final onesD = Tensor.fill([1, _d], 1.0, device: Device.CPU);
  final sharedGateBcast = sharedGateScore.matmul(onesD); // [T, D]
  var acc = sharedOut * sharedGateBcast;

  for (final j in sortedUsed) {
    final wjVals = List<double>.filled(tokens, 0.0);
    for (int i = 0; i < tokens; i++) {
      wjVals[i] = flat[i * _e + j] * maskVals[i * _e + j];
    }
    final wj = Tensor.fromList([tokens, 1], wjVals, device: Device.CPU);
    final wjBcast = wj.matmul(onesD); // [T, D]
    final expertOut = _swiGlu(x, eGate[j]!, eUp[j]!, eDown[j]!);
    acc = acc + (expertOut * wjBcast);
  }
  swF.stop();
  print('');
  print('== forward ==');
  print('  wall           : ${swF.elapsedMilliseconds} ms');
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
  print('  stats          : min=${mn.toStringAsFixed(4)} '
      'max=${mx.toStringAsFixed(4)} '
      'mean=${(sum / (finite == 0 ? 1 : finite)).toStringAsFixed(6)} '
      'nan=$nan/${row.length}');
  if (nan == 0) {
    print('  status         : OK — per-expert streaming against real '
        'Qwen1.5-MoE weights produces finite outputs.');
  }

  reader.close();
}

Tensor _swiGlu(Tensor x, Tensor gW, Tensor uW, Tensor dW) {
  final gate = x.matmul(gW.transpose());
  final up = x.matmul(uW.transpose());
  final act = gate * gate.sigmoid();
  return (act * up).matmul(dW.transpose());
}

int _bytesOf(ShardedSafeTensorsReader reader, String name) {
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
