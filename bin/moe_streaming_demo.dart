/// Per-expert streaming demo for MoE FFN — the distinctive AirLLM
/// trick that lets a 671 B DeepSeek-V3 fit in ~12 GB and Kimi K3 in
/// under 4 GB: for each token forward, only stream the top-K of E
/// experts the router actually picks, not the whole layer.
///
/// This demo builds a small [MoEFeedForward] layer, writes a random
/// fp16 safetensors "checkpoint" for its expert weights, then runs a
/// forward that:
///
///   1. Runs the router on a batch of tokens to get per-token top-K
///      expert indices.
///   2. Computes the union of routed experts across the batch.
///   3. Streams *only* those experts' `w1`/`w2`/`w3` weights from
///      disk into the resident module.
///   4. Runs the MoE forward with `sparseExecution: true` so the
///      unused experts (whose in-memory weights are still fp32 random
///      init) contribute nothing.
///
/// Reports what percentage of the layer's expert bytes were actually
/// streamed. For single-token generation (`--tokens 1`) with K=4,
/// E=16 you'd expect ~25%; with K=6, E=64 (DeepSeek-V2-Lite scale)
/// you'd expect ~9% — those are AirLLM's headline savings.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';

Future<void> main(List<String> args) async {
  var d = 128;
  var e = 16;
  var h = 256;
  var k = 4;
  var t = 1;
  var seed = 0;
  String? outDirArg;
  var keep = false;
  for (int i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--d':
        d = int.parse(args[++i]);
        break;
      case '--experts':
        e = int.parse(args[++i]);
        break;
      case '--hidden':
        h = int.parse(args[++i]);
        break;
      case '--topk':
        k = int.parse(args[++i]);
        break;
      case '--tokens':
        t = int.parse(args[++i]);
        break;
      case '--seed':
        seed = int.parse(args[++i]);
        break;
      case '--out-dir':
        outDirArg = args[++i];
        break;
      case '--keep':
        keep = true;
        break;
      case '-h':
      case '--help':
        stdout.writeln(
          'usage: moe_streaming_demo [--d N] [--experts N] [--hidden N] '
          '[--topk K] [--tokens N] [--seed N] [--out-dir DIR] [--keep]',
        );
        return;
    }
  }

  final outDir = outDirArg ?? '/tmp/moe_streaming';
  Directory(outDir).createSync(recursive: true);
  final ckptPath = '$outDir/moe.safetensors';

  print('== MoE per-expert streaming demo (AirLLM-style) ==');
  print('  D (embed)      : $d');
  print('  E (experts)    : $e');
  print('  H (hidden)     : $h');
  print('  K (top-K)      : $k');
  print('  T (tokens)     : $t');
  print('  variant        : SwiGLU (w1, w2, w3)');
  print('  gate           : sigmoid + renormalize (DeepSeek-V3 style)');
  print('  ckpt           : $ckptPath');

  final expertBytes = 3 * h * d * 2; // fp16: w1[hd], w2[dh], w3[hd]
  final totalExpertBytes = e * expertBytes;
  print('  per-expert     : ${_fmtBytes(expertBytes)} fp16');
  print('  all experts    : ${_fmtBytes(totalExpertBytes)} fp16');

  final swGen = Stopwatch()..start();
  final specs = _expertSpecs(e, d, h);
  _writeShardFile(ckptPath, specs, math.Random(seed));
  swGen.stop();
  print('  gen            : ${swGen.elapsedMilliseconds} ms');

  final moe = MoEFeedForward(
    embedDim: d,
    numRoutedExperts: e,
    numSharedExperts: 0,
    topK: k,
    expertHiddenDim: h,
    activation: ExpertActivation.silu,
    expertVariant: ExpertVariant.swiGlu,
    gateFunction: GateFunction.sigmoid,
    sparseExecution: true,
    device: Device.CPU,
    seed: seed,
  );

  final reader = ShardedSafeTensorsReader.open(ckptPath);

  // Random 2D input [T, D].
  final rng = math.Random(seed + 42);
  final xVals = List<double>.generate(t * d, (_) => (rng.nextDouble() - 0.5));
  final x = Tensor.fromList([t, d], xVals, device: Device.CPU);

  // 1. Compute routing decision — which experts do these T tokens
  //    actually need? Mimics the internal routing in MoEFeedForward
  //    but as a pure inspection (no side effects on _expertLoad).
  final used = _topKExpertsForBatch(moe, x, k);
  print('');
  print('== routing ==');
  print('  top-K experts  : ${used.toList()..sort()} '
      '(${used.length} of $e = ${(used.length * 100 / e).toStringAsFixed(1)}%)');
  final streamedBytes = used.length * expertBytes;
  print('  streamed       : ${_fmtBytes(streamedBytes)} '
      '(vs ${_fmtBytes(totalExpertBytes)} for full layer)');
  print('  saved          : ${_fmtBytes(totalExpertBytes - streamedBytes)} '
      '(${((totalExpertBytes - streamedBytes) * 100 / totalExpertBytes).toStringAsFixed(1)}%)');

  // 2. Load only those experts' weights from disk into the resident
  //    module. Other experts keep their fp32 random init — they get
  //    skipped by sparseExecution.
  final swLoad = Stopwatch()..start();
  for (final j in used) {
    _streamExpert(reader, j, moe.routedExperts[j]);
  }
  swLoad.stop();
  print('  load           : ${swLoad.elapsedMilliseconds} ms '
      '(${used.length} experts)');

  // 3. Forward — sparseExecution ensures only loaded experts run.
  final swF = Stopwatch()..start();
  final y = Tensor.noGrad(() => moe(x));
  swF.stop();
  print('');
  print('== forward ==');
  print('  output shape   : ${y.shape} (expected [$t, $d])');
  print('  wall           : ${swF.elapsedMilliseconds} ms');

  final row = y.toList();
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
  print('  stats          : min=${mn.toStringAsFixed(3)} '
      'max=${mx.toStringAsFixed(3)} '
      'mean=${(sum / (finite == 0 ? 1 : finite)).toStringAsFixed(5)} '
      'nan=$nan/${row.length}');
  if (nan == 0) {
    print('  status         : OK — per-expert streaming produces '
        'finite outputs; unused experts skipped by sparseExecution.');
  }

  reader.close();
  if (!keep) {
    try {
      File(ckptPath).deleteSync();
    } catch (_) {}
  }
}

/// Inspect which experts the router would pick for `x` without
/// touching the module's load counters or bias.
Set<int> _topKExpertsForBatch(MoEFeedForward moe, Tensor x, int k) {
  final t = x.shape[0];
  final e = moe.numRoutedExperts;
  final gateLogits = x.matmul(moe.gateW); // [T, E]
  final scores = moe.gateFunction == GateFunction.sigmoid
      ? gateLogits.sigmoid()
      : gateLogits.softmax();
  final flat = scores.toList();
  final used = <int>{};
  for (int i = 0; i < t; i++) {
    final indexed = List<MapEntry<int, double>>.generate(
      e,
      (j) => MapEntry(j, flat[i * e + j] + moe.routingBias[j]),
    );
    indexed.sort((a, b) => b.value.compareTo(a.value));
    for (int r = 0; r < k; r++) {
      used.add(indexed[r].key);
    }
  }
  return used;
}

/// Load expert `j`'s w1/w2/w3 fp16 from the reader into the resident
/// module's tensors via `adoptCpuStorageFrom`.
void _streamExpert(
  ShardedSafeTensorsReader reader,
  int j,
  Expert dst,
) {
  void load(String name, Tensor into) {
    final src = reader.readTensor(name, keepFp16: true);
    if (src.length != into.length) {
      throw StateError(
        'expert $j: shape mismatch for $name — src=${src.shape}, '
        'dst=${into.shape}',
      );
    }
    if (src.dtype == DType.fp16 && into.device == Device.CPU) {
      into.adoptCpuStorageFrom(src);
    } else {
      into.assign(
        Tensor.fromList(into.shape, src.toList(), device: into.device),
      );
    }
  }

  load('experts.$j.w1.weight', dst.w1.weight);
  load('experts.$j.w2.weight', dst.w2.weight);
  if (dst.w3 != null) load('experts.$j.w3.weight', dst.w3!.weight);
}

// ---------------------------------------------------------------------------
// Random-weight safetensors writer for the expert stack.
// ---------------------------------------------------------------------------

class _Spec {
  final String name;
  final List<int> shape;
  const _Spec(this.name, this.shape);
  int get numel => shape.fold(1, (a, b) => a * b);
  int get bytes => numel * 2; // fp16
}

List<_Spec> _expertSpecs(int e, int d, int h) {
  final s = <_Spec>[];
  for (int j = 0; j < e; j++) {
    s.add(_Spec('experts.$j.w1.weight', [h, d]));
    s.add(_Spec('experts.$j.w2.weight', [d, h]));
    s.add(_Spec('experts.$j.w3.weight', [h, d]));
  }
  return s;
}

void _writeShardFile(String path, List<_Spec> specs, math.Random rng) {
  var running = 0;
  final offsets = <int>[];
  for (final s in specs) {
    offsets.add(running);
    running += s.bytes;
  }
  final headerMap = <String, dynamic>{};
  for (int i = 0; i < specs.length; i++) {
    final s = specs[i];
    headerMap[s.name] = {
      'dtype': 'F16',
      'shape': s.shape,
      'data_offsets': [offsets[i], offsets[i] + s.bytes],
    };
  }
  final rawHeader = utf8.encode(jsonEncode(headerMap));
  final padLen = ((rawHeader.length + 7) ~/ 8) * 8;
  final padded = Uint8List(padLen);
  padded.setRange(0, rawHeader.length, rawHeader);
  for (int i = rawHeader.length; i < padLen; i++) {
    padded[i] = 0x20;
  }

  final raf = File(path).openSync(mode: FileMode.write);
  try {
    final lenBd = ByteData(8)..setUint64(0, padLen, Endian.little);
    raf.writeFromSync(lenBd.buffer.asUint8List());
    raf.writeFromSync(padded);
    for (final s in specs) {
      _streamRandomBytes(raf, s.bytes, rng);
    }
  } finally {
    raf.closeSync();
  }
}

const int _chunk = 1 << 20; // 1 MB
void _streamRandomBytes(RandomAccessFile raf, int nBytes, math.Random rng) {
  final buf = Uint8List(_chunk);
  final bd = ByteData.sublistView(buf);
  var remaining = nBytes;
  while (remaining > 0) {
    final take = remaining >= _chunk ? _chunk : remaining;
    for (int i = 0; i < take; i += 2) {
      final f32 = (rng.nextDouble() * 0.04) - 0.02;
      bd.setUint16(i, _f32ToF16(f32), Endian.little);
    }
    raf.writeFromSync(buf, 0, take);
    remaining -= take;
  }
}

int _f32ToF16(double x) {
  final bd = ByteData(4)..setFloat32(0, x, Endian.little);
  final bits = bd.getUint32(0, Endian.little);
  final sign = (bits >> 31) & 1;
  final exp32 = (bits >> 23) & 0xFF;
  final mant32 = bits & 0x7FFFFF;
  if (exp32 == 0) return sign << 15;
  if (exp32 == 0xFF) {
    if (mant32 == 0) return (sign << 15) | 0x7C00;
    return (sign << 15) | 0x7C00 | 1;
  }
  final ee = exp32 - 127 + 15;
  if (ee >= 0x1F) return (sign << 15) | 0x7C00;
  if (ee <= 0) return sign << 15;
  return (sign << 15) | (ee << 10) | (mant32 >> 13);
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
