/// Random-weight validator for [Qwen15MoEStreamingRunner].
///
/// Generates a small (or full-scale) synthetic checkpoint matching
/// the Qwen1.5-MoE-A2.7B HF key layout, then runs a full forward
/// through the streaming runner to prove the end-to-end pipeline
/// (embed → 24 × (MHA + MoE + streamed top-K experts) → head) works
/// on real-sized tensors, without needing the actual 28.6 GB
/// checkpoint on disk.
///
///   dart run bin/qwen15_moe_streaming_random_demo.dart              # tiny preset
///   dart run bin/qwen15_moe_streaming_random_demo.dart --preset full  # real config
///   dart run bin/qwen15_moe_streaming_random_demo.dart --preset full --seq-len 4 --keep
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';

Future<void> main(List<String> args) async {
  var preset = 'tiny';
  var seqLen = 4;
  var seed = 0;
  var keep = false;
  var profile = false;
  String? outDirArg;
  for (int i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--preset':
        preset = args[++i];
        break;
      case '--seq-len':
        seqLen = int.parse(args[++i]);
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
      case '--profile':
        profile = true;
        break;
      case '-h':
      case '--help':
        stdout.writeln(
          'usage: qwen15_moe_streaming_random_demo [--preset tiny|full] '
          '[--seq-len N] [--seed N] [--out-dir DIR] [--keep] [--profile]',
        );
        return;
    }
  }

  final cfg = switch (preset.toLowerCase()) {
    'tiny' => const Qwen15MoEConfig(
        vocabSize: 512,
        dim: 128,
        numHeads: 4,
        numKvHeads: 4,
        numLayers: 2,
        moeHidden: 64,
        sharedHidden: 256,
        numExperts: 8,
        topK: 2,
        maxCtx: 128,
      ),
    'full' => const Qwen15MoEConfig(), // A2.7B defaults
    _ => throw ArgumentError('unknown --preset "$preset" (tiny|full)'),
  };

  final outDir = outDirArg ?? '/tmp/qwen15_moe_random_$preset';
  Directory(outDir).createSync(recursive: true);
  final ckptPath = '$outDir/model.safetensors';

  final specs = _qwen15Specs(cfg);
  final totalBytes = specs.fold<int>(0, (a, s) => a + s.bytes);
  print('== Qwen1.5-MoE streaming runner (random weights) ==');
  print('  preset       : $preset');
  print('  layers       : ${cfg.numLayers}');
  print('  D            : ${cfg.dim}');
  print('  heads        : ${cfg.numHeads} (headDim ${cfg.headDim})');
  print('  E / K        : ${cfg.numExperts} / ${cfg.topK}');
  print('  moe_hidden   : ${cfg.moeHidden}');
  print('  shared_hidden: ${cfg.sharedHidden}');
  print('  vocab        : ${cfg.vocabSize}');
  print('  ckpt         : $ckptPath (${_fmtBytes(totalBytes)} fp16)');

  final free = _freeRamBytes();
  final expertsPerLayer = cfg.numExperts * 3 * cfg.moeHidden * cfg.dim * 2;
  final peakEst = cfg.vocabSize * cfg.dim * 2 * 2 // embed + lm_head fp16
      +
      100 * 1024 * 1024 + // one resident block
      3 * cfg.topK * cfg.moeHidden * cfg.dim * 2 + // streamed scratch
      200 * 1024 * 1024; // Dart runtime + activations
  print('  RAM est peak : ${_fmtBytes(peakEst)}');
  print('  per-layer exp: ${_fmtBytes(expertsPerLayer)} on disk '
      '(saved ${((cfg.numExperts - cfg.topK) * 100 / cfg.numExperts).toStringAsFixed(1)}% by streaming top-${cfg.topK})');
  if (free != null) {
    print('  RAM free     : ${_fmtBytes(free)} '
        '(from /proc/meminfo MemAvailable)');
    if (peakEst > free) {
      stderr.writeln('ABORT: predicted peak exceeds free RAM');
      exit(3);
    }
  }

  if (!File(ckptPath).existsSync()) {
    print('');
    final swGen = Stopwatch()..start();
    _writeShardFile(ckptPath, specs, math.Random(seed));
    swGen.stop();
    print('  gen          : ${swGen.elapsedMilliseconds} ms '
        '(${(totalBytes / (1024 * 1024) / (swGen.elapsedMilliseconds / 1000.0)).toStringAsFixed(1)} MB/s write)');
  } else {
    print('  gen          : (exists, skipping)');
  }

  final swOpen = Stopwatch()..start();
  final reader = ShardedSafeTensorsReader.open(ckptPath);
  swOpen.stop();
  print('  header parse : ${swOpen.elapsedMilliseconds} ms');

  final swInit = Stopwatch()..start();
  final runner =
      Qwen15MoEStreamingRunner(cfg, reader, profile: profile);
  swInit.stop();
  print('  runner init  : ${swInit.elapsedMilliseconds} ms '
      '(persistent embed + head + rope + resident block)');

  // Random prompt.
  final rng = math.Random(seed + 1);
  final promptIds = List<int>.generate(
    seqLen,
    (_) => rng.nextInt(cfg.vocabSize),
  );
  final promptTensor = Tensor.fromList(
    [seqLen],
    promptIds.map((i) => i.toDouble()).toList(),
    device: Device.CPU,
  );

  print('');
  print('== forward ==');
  print('  seq len      : $seqLen tokens');
  print('  prompt       : $promptIds');
  final swF = Stopwatch()..start();
  final logits = runner.forward(promptTensor);
  swF.stop();
  print('  wall         : ${swF.elapsedMilliseconds} ms');
  print('  logits shape : ${logits.shape} '
      '(expected [$seqLen, ${cfg.vocabSize}])');

  final row = logits.toList();
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
  print('  stats        : min=${mn.toStringAsFixed(4)} '
      'max=${mx.toStringAsFixed(4)} '
      'mean=${(sum / (finite == 0 ? 1 : finite)).toStringAsFixed(5)} '
      'nan=$nan/${row.length}');
  if (nan == 0) {
    print('  status       : OK — Qwen1.5-MoE streaming runner produces '
        'finite outputs end-to-end.');
  } else {
    stderr.writeln('WARNING: $nan NaN(s) in output.');
  }

  // Also exercise the generate() path (prompt fill + KV-cached
  // single-token appends).
  print('');
  print('== generate (2 new tokens, greedy w/ KV cache) ==');
  final swG = Stopwatch()..start();
  final out = runner.generate(
    promptIds.map((i) => i.toDouble()).toList(),
    maxNewTokens: 2,
  );
  swG.stop();
  final newIds = out.skip(seqLen).map((v) => v.toInt()).toList();
  print('  wall         : ${swG.elapsedMilliseconds} ms');
  print('  new ids      : $newIds');
  print('  status       : OK — KV cache survives per-layer swaps and '
      'per-token expert streaming.');

  runner.close();
  if (!keep) {
    try {
      Directory(outDir).deleteSync(recursive: true);
      print('');
      print('  cleaned      : $outDir (pass --keep to preserve)');
    } catch (_) {}
  }
}

// ---------------------------------------------------------------------------
// Tensor specs — mirror the HF layout that Qwen15MoEStreamingRunner reads.
// ---------------------------------------------------------------------------

class _Spec {
  final String name;
  final List<int> shape;
  const _Spec(this.name, this.shape);
  int get numel {
    var p = 1;
    for (final s in shape) {
      p *= s;
    }
    return p;
  }

  int get bytes => numel * 2; // fp16
}

List<_Spec> _qwen15Specs(Qwen15MoEConfig cfg) {
  final specs = <_Spec>[];
  final d = cfg.dim;
  final e = cfg.numExperts;
  final h = cfg.numHeads;
  final kvH = cfg.numKvHeads;
  final headDim = d ~/ h;
  specs.add(_Spec('model.embed_tokens.weight', [cfg.vocabSize, d]));
  specs.add(_Spec('model.norm.weight', [d]));
  specs.add(_Spec('lm_head.weight', [cfg.vocabSize, d]));
  for (int i = 0; i < cfg.numLayers; i++) {
    final p = 'model.layers.$i';
    specs.add(_Spec('$p.input_layernorm.weight', [d]));
    specs.add(_Spec('$p.post_attention_layernorm.weight', [d]));
    specs.add(_Spec('$p.self_attn.q_proj.weight', [h * headDim, d]));
    specs.add(_Spec('$p.self_attn.q_proj.bias', [h * headDim]));
    specs.add(_Spec('$p.self_attn.k_proj.weight', [kvH * headDim, d]));
    specs.add(_Spec('$p.self_attn.k_proj.bias', [kvH * headDim]));
    specs.add(_Spec('$p.self_attn.v_proj.weight', [kvH * headDim, d]));
    specs.add(_Spec('$p.self_attn.v_proj.bias', [kvH * headDim]));
    specs.add(_Spec('$p.self_attn.o_proj.weight', [d, d]));
    specs.add(_Spec('$p.mlp.gate.weight', [e, d]));
    specs.add(
      _Spec('$p.mlp.shared_expert.gate_proj.weight', [cfg.sharedHidden, d]),
    );
    specs.add(
      _Spec('$p.mlp.shared_expert.up_proj.weight', [cfg.sharedHidden, d]),
    );
    specs.add(
      _Spec('$p.mlp.shared_expert.down_proj.weight', [d, cfg.sharedHidden]),
    );
    specs.add(_Spec('$p.mlp.shared_expert_gate.weight', [1, d]));
    for (int j = 0; j < e; j++) {
      specs.add(
        _Spec('$p.mlp.experts.$j.gate_proj.weight', [cfg.moeHidden, d]),
      );
      specs.add(_Spec('$p.mlp.experts.$j.up_proj.weight', [cfg.moeHidden, d]));
      specs.add(
        _Spec('$p.mlp.experts.$j.down_proj.weight', [d, cfg.moeHidden]),
      );
    }
  }
  return specs;
}

// ---------------------------------------------------------------------------
// Safetensors writer — chunked to bound peak RAM.
// ---------------------------------------------------------------------------

void _writeShardFile(
  String path,
  List<_Spec> specs,
  math.Random rng,
) {
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

const int _chunk = 4 * 1024 * 1024;

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
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  var i = 0;
  double v = b.toDouble();
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return '${v.toStringAsFixed(2)} ${units[i]}';
}

int? _freeRamBytes() {
  try {
    final txt = File('/proc/meminfo').readAsStringSync();
    for (final line in txt.split('\n')) {
      if (line.startsWith('MemAvailable:')) {
        final parts = line.split(RegExp(r'\s+'));
        return int.parse(parts[1]) * 1024;
      }
    }
  } catch (_) {}
  return null;
}
