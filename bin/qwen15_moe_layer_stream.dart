/// AirLLM-style per-expert streaming on a real MoE layer of
/// `Qwen/Qwen1.5-MoE-A2.7B-Chat`. Uses the reusable
/// [MoeStreamingLayer] helper.
///
///   dart run bin/qwen15_moe_layer_stream.dart \
///     --index ~/models/qwen1.5-moe-a2.7b-chat/model.safetensors.index.json \
///     --layer 0 --tokens 1 --seed 0
///
/// Layer 0 fits entirely in shard 1, so this works once the first
/// shard finishes downloading.
library;

import 'dart:io';
import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

// Config values pulled from Qwen1.5-MoE-A2.7B-Chat config.json.
const int _d = 2048;
const int _e = 60;
const int _k = 4;
const int _moeHidden = 1408;
const int _sharedHidden = 5632;
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
  print('  D              : $_d');
  print('  E / K          : $_e / $_k');
  print('  moe_hidden     : $_moeHidden');
  print('  shared_hidden  : $_sharedHidden (scalar-gated shared_expert)');
  print('  gate           : softmax + no-renormalize (Qwen2-MoE)');
  print('  tokens         : $tokens');

  final swOpen = Stopwatch()..start();
  final reader = ShardedSafeTensorsReader.open(indexPath);
  swOpen.stop();
  print('  header parse   : ${swOpen.elapsedMilliseconds} ms');

  final p = 'model.layers.$layer.mlp';
  final moe = MoeStreamingLayer(
    prefix: p,
    numExperts: _e,
    topK: _k,
    reader: reader,
    gate: GateFunction.softmax,
  );
  if (!reader.contains(moe.routerKey)) {
    stderr.writeln('no "${moe.routerKey}" — layer $layer probably not '
        'in a downloaded shard yet');
    exit(2);
  }

  final routedBytes = moe.expertBytes();
  final totalRouted = _e * routedBytes;
  final sharedGateEntry = reader.entry('$p.shared_expert.gate_proj.weight')!;
  final sharedBytes = 3 * (sharedGateEntry.dataEnd - sharedGateEntry.dataStart);
  print('');
  print('== on-disk footprint (one layer) ==');
  print('  1 routed expert: ${_fmtBytes(routedBytes)}');
  print('  all routed     : ${_fmtBytes(totalRouted)}');
  print('  shared         : ${_fmtBytes(sharedBytes)}');

  final swLoad = Stopwatch()..start();
  final routerW = moe.loadRouter();
  swLoad.stop();
  print('');
  print('== persistent load ==');
  print('  routerW shape  : ${routerW.shape} '
      '(${swLoad.elapsedMilliseconds} ms)');

  final sharedGate = reader.readTensor('$p.shared_expert.gate_proj.weight');
  final sharedUp = reader.readTensor('$p.shared_expert.up_proj.weight');
  final sharedDown = reader.readTensor('$p.shared_expert.down_proj.weight');
  final sharedExpertGate =
      reader.readTensor('$p.shared_expert_gate.weight'); // [1, D]

  final rng = math.Random(seed);
  final xVals = List<double>.generate(
    tokens * _d,
    (_) => (rng.nextDouble() - 0.5) * 0.1,
  );
  final x = Tensor.fromList([tokens, _d], xVals, device: Device.CPU);

  final decision = moe.route(routerW, x);
  final report = moe.report(decision, bytesPerExpert: routedBytes);
  print('');
  print('== routing ==');
  print('  top-K experts  : ${decision.sortedUnion} '
      '(${decision.union.length} of $_e = '
      '${(decision.union.length * 100 / _e).toStringAsFixed(1)}%)');
  print('  streamed       : ${_fmtBytes(report.streamedBytes)} '
      '(vs ${_fmtBytes(report.totalBytes)} for full layer)');
  print('  saved          : ${_fmtBytes(report.savedBytes)} '
      '(${report.savedPercent.toStringAsFixed(1)}%)');

  if (reportOnly) {
    reader.close();
    return;
  }

  final swExperts = Stopwatch()..start();
  final loaded = <int, RoutedExpertWeights>{};
  for (final j in decision.sortedUnion) {
    loaded[j] = moe.loadRoutedExpert(j);
  }
  swExperts.stop();
  print('  load           : ${swExperts.elapsedMilliseconds} ms '
      '(${decision.union.length} experts)');

  final swF = Stopwatch()..start();
  final sharedOut = swiGluForwardRaw(x, sharedGate, sharedUp, sharedDown);
  final sharedGateLogit = x.matmul(sharedExpertGate.transpose()); // [T, 1]
  final sharedGateScore = sharedGateLogit.sigmoid();
  final onesD = Tensor.fill([1, _d], 1.0, device: Device.CPU);
  final sharedGateBcast = sharedGateScore.matmul(onesD);
  var acc = sharedOut * sharedGateBcast;

  for (final j in decision.sortedUnion) {
    final wj = Tensor.fromList(
      [tokens, 1],
      decision.weightForExpert(j),
      device: Device.CPU,
    );
    final wjBcast = wj.matmul(onesD);
    final expertOut = swiGluForward(x, loaded[j]!);
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
