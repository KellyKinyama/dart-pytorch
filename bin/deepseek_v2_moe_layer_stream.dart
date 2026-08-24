/// AirLLM-style per-expert streaming on a real MoE layer from
/// `deepseek-ai/DeepSeek-V2-Lite-Chat`, using the reusable
/// [MoeStreamingLayer] helper.
///
/// Does **not** run a full end-to-end forward through the model —
/// that would require porting MLA + dense-vs-MoE layer switching +
/// persistent embed/head.
///
///   dart run bin/deepseek_v2_moe_layer_stream.dart \
///     --index ~/models/deepseek-v2-lite-chat/model.safetensors.index.json \
///     --layer 5 --tokens 1 --seed 0
library;

import 'dart:io';
import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

// From DeepSeek-V2-Lite-Chat config.json.
const int _d = 2048;
const int _e = 64;
const int _k = 6;
const int _moeHidden = 1408;
const int _sharedExperts = 2;
const int _sharedHidden = _sharedExperts * _moeHidden;
const int _numLayers = 27;
const int _firstDense = 1;

Future<void> main(List<String> args) async {
  var indexPath = '${_defaultHome()}/models/deepseek-v2-lite-chat/'
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
    stderr.writeln('download: hf download deepseek-ai/DeepSeek-V2-Lite-Chat '
        '--local-dir ~/models/deepseek-v2-lite-chat');
    exit(2);
  }
  if (layer < _firstDense || layer >= _numLayers) {
    stderr.writeln('layer $layer is dense or out of range — MoE layers '
        'are [$_firstDense, $_numLayers)');
    exit(2);
  }

  print('== DeepSeek-V2-Lite per-expert streaming (real weights) ==');
  print('  index          : $indexPath');
  print('  layer          : $layer (of $_numLayers, first $_firstDense dense)');
  print('  D              : $_d');
  print('  E / K          : $_e / $_k');
  print('  moe_hidden     : $_moeHidden');
  print('  shared         : $_sharedExperts experts fused as SwiGLU '
      'hidden=$_sharedHidden');
  print('  gate           : softmax + no-renormalize (V2 config)');
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
    stderr.writeln('layer $layer is not a MoE layer or its shard is '
        'not on disk yet (no "${moe.routerKey}")');
    exit(2);
  }

  final routedBytes = moe.expertBytes();
  final totalRouted = _e * routedBytes;
  final sharedGateEntry = reader.entry('$p.shared_experts.gate_proj.weight')!;
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

  final sharedGate =
      reader.readTensor('$p.shared_experts.gate_proj.weight');
  final sharedUp = reader.readTensor('$p.shared_experts.up_proj.weight');
  final sharedDown = reader.readTensor('$p.shared_experts.down_proj.weight');

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
  var acc = swiGluForwardRaw(x, sharedGate, sharedUp, sharedDown);
  final onesD = Tensor.fill([1, _d], 1.0, device: Device.CPU);
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
  print('  wall           : ${swF.elapsedMilliseconds} ms (routed via '
      '${decision.union.length} experts + shared)');
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
        'DeepSeek-V2-Lite weights produces finite outputs.');
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
