/// Layer-streaming validation with **random weights** — proves the
/// runner scales to real-sized Llama-family configs without needing
/// a downloaded checkpoint.
///
/// The output is meaningless (random logits, not real completions),
/// but this exercises every code path in the pipeline against
/// real-sized tensors:
///
///   * safetensors header at production scale (100s of entries),
///   * `ShardedSafeTensorsReader.fromIndex` (with `--shards N > 1`),
///   * per-layer seek + fp16 decode,
///   * `adoptCpuStorageFrom` swapping into the resident block,
///   * a full 16-/24-/28-layer forward pass, RMSNorm + GQA + SwiGLU
///     + RoPE + tied/untied lm_head, without OOM.
///
///   dart run bin/llama_streaming_random_demo.dart --preset llama-3.2-1b
///   dart run bin/llama_streaming_random_demo.dart --preset llama-3.2-3b --shards 4
///   dart run bin/llama_streaming_random_demo.dart --preset smollm2-1.7b --seq-len 16
///
/// The generated checkpoint lives under `/tmp/random_llama_<preset>/`
/// by default. Pass `--keep` to leave it on disk between runs,
/// otherwise it's deleted after the forward pass.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';

import '_llama_encoder.dart';

Future<void> main(List<String> args) async {
  var preset = 'llama-3.2-1b';
  String? outDirArg;
  var shards = 1;
  var seqLen = 8;
  var seed = 0;
  var keep = false;
  var profile = false;
  for (int i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--preset':
        preset = args[++i];
        break;
      case '--out-dir':
        outDirArg = args[++i];
        break;
      case '--shards':
        shards = int.parse(args[++i]);
        break;
      case '--seq-len':
        seqLen = int.parse(args[++i]);
        break;
      case '--seed':
        seed = int.parse(args[++i]);
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
          'usage: llama_streaming_random_demo [--preset NAME] '
          '[--out-dir DIR] [--shards N] [--seq-len N] [--seed N] '
          '[--keep] [--profile]',
        );
        return;
    }
  }

  final cfg = configForLlamaPreset(preset, Device.CPU);
  final outDir = outDirArg ?? '/tmp/random_llama_$preset';
  Directory(outDir).createSync(recursive: true);

  final ckptPath = shards == 1
      ? '$outDir/model.safetensors'
      : '$outDir/model.safetensors.index.json';

  final specs = _llamaSpecs(cfg);
  final totalBytes = specs.fold<int>(0, (a, s) => a + s.bytes);
  print('== streaming Llama random-weight validator ==');
  print('  preset  : $preset');
  print('  layers  : ${cfg.numLayers}');
  print(
    '  D / H   : ${cfg.embedDim} / ${cfg.numHeads} '
    '(kv=${cfg.numKvHeads}, headDim=${cfg.embedDim ~/ cfg.numHeads})',
  );
  print('  vocab   : ${cfg.vocabSize}');
  print('  ffn     : ${cfg.ffnDim}');
  print('  tie     : ${cfg.tieWeights}');
  print('  attnBias: ${cfg.attentionBias}');
  print(
    '  ckpt    : $ckptPath  ($shards shard${shards == 1 ? "" : "s"}, '
    '${_fmtBytes(totalBytes)} on disk when fp16)',
  );
  print('  out-dir : $outDir');

  if (!File(ckptPath).existsSync()) {
    final swGen = Stopwatch()..start();
    if (shards == 1) {
      _writeShardFile('$outDir/model.safetensors', specs, math.Random(seed));
    } else {
      _writeShardedCheckpoint(outDir, specs, shards, seed);
    }
    swGen.stop();
    print(
      '  gen     : ${swGen.elapsedMilliseconds} ms '
      '(${(totalBytes / (1024 * 1024) / (swGen.elapsedMilliseconds / 1000.0)).toStringAsFixed(1)} MB/s write)',
    );
  } else {
    print('  gen     : (already exists, skipping)');
  }

  final swOpen = Stopwatch()..start();
  final reader = ShardedSafeTensorsReader.open(ckptPath);
  swOpen.stop();
  print('  header  : ${swOpen.elapsedMilliseconds} ms');

  final swInit = Stopwatch()..start();
  final runner = LlamaStreamingRunner(cfg, reader, profile: profile);
  swInit.stop();
  print(
    '  init    : ${swInit.elapsedMilliseconds} ms '
    '(persistent tensors loaded)',
  );

  final layerBytes = estimateLayerBytes(reader, cfg);
  print(
    '  layer   : ${_fmtBytes(layerBytes)} on disk per layer '
    '(× ${cfg.numLayers} = ${_fmtBytes(layerBytes * cfg.numLayers)} '
    'streamed per forward)',
  );

  final rng = math.Random(seed + 1);
  final promptIds = List.generate(seqLen, (_) => rng.nextInt(cfg.vocabSize));
  print('');
  print('== forward ==');
  print('  seq len : $seqLen tokens');
  print('  prompt  : $promptIds');

  final promptTensor = Tensor.fromList(
    [seqLen],
    promptIds.map((i) => i.toDouble()).toList(),
    device: Device.CPU,
  );

  final swF = Stopwatch()..start();
  final logits = runner.forward(promptTensor);
  swF.stop();
  print(
    '  wall    : ${swF.elapsedMilliseconds} ms '
    '(${(cfg.numLayers * 1000.0 / swF.elapsedMilliseconds).toStringAsFixed(2)} '
    'layers/s effective)',
  );
  print('  logits  : ${logits.shape}');

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
  print(
    '  stats   : min=${mn.toStringAsFixed(3)} '
    'max=${mx.toStringAsFixed(3)} '
    'mean=${(sum / (finite == 0 ? 1 : finite)).toStringAsFixed(5)} '
    'nan=$nan/${row.length}',
  );
  if (nan > 0) {
    stderr.writeln(
      'WARNING: forward produced $nan NaN(s) — random weights can '
      'saturate softmax; try smaller --seed variation or --seq-len',
    );
  } else {
    print(
      '  status  : OK — all logits finite, layer-streaming pipeline '
      'validated end-to-end.',
    );
  }

  runner.close();
  if (!keep) {
    try {
      Directory(outDir).deleteSync(recursive: true);
      print('');
      print('  cleaned : $outDir (pass --keep to preserve)');
    } catch (e) {
      stderr.writeln('cleanup failed: $e');
    }
  }
}

// ---------------------------------------------------------------------------
// Tensor specs — mirror what LlamaHFLoader.loadMap expects.
// ---------------------------------------------------------------------------

class _TensorSpec {
  final String name;
  final List<int> shape;
  final String dtype; // 'F16' or 'F32'
  const _TensorSpec(this.name, this.shape, this.dtype);
  int get numel {
    var p = 1;
    for (final s in shape) {
      p *= s;
    }
    return p;
  }

  int get bytes => numel * (dtype == 'F16' ? 2 : 4);
}

List<_TensorSpec> _llamaSpecs(LlamaConfig cfg) {
  final specs = <_TensorSpec>[];
  final d = cfg.embedDim;
  final h = cfg.numHeads;
  final kvH = cfg.numKvHeads;
  final headDim = d ~/ h;
  final ffn = cfg.ffnDim;
  specs.add(
    _TensorSpec('model.embed_tokens.weight', [cfg.vocabSize, d], 'F16'),
  );
  for (int i = 0; i < cfg.numLayers; i++) {
    final p = 'model.layers.$i';
    specs.add(_TensorSpec('$p.input_layernorm.weight', [d], 'F16'));
    specs.add(
      _TensorSpec('$p.self_attn.q_proj.weight', [h * headDim, d], 'F16'),
    );
    if (cfg.attentionBias) {
      specs.add(_TensorSpec('$p.self_attn.q_proj.bias', [h * headDim], 'F16'));
    }
    specs.add(
      _TensorSpec('$p.self_attn.k_proj.weight', [kvH * headDim, d], 'F16'),
    );
    if (cfg.attentionBias) {
      specs.add(
        _TensorSpec('$p.self_attn.k_proj.bias', [kvH * headDim], 'F16'),
      );
    }
    specs.add(
      _TensorSpec('$p.self_attn.v_proj.weight', [kvH * headDim, d], 'F16'),
    );
    if (cfg.attentionBias) {
      specs.add(
        _TensorSpec('$p.self_attn.v_proj.bias', [kvH * headDim], 'F16'),
      );
    }
    specs.add(_TensorSpec('$p.self_attn.o_proj.weight', [d, d], 'F16'));
    specs.add(_TensorSpec('$p.post_attention_layernorm.weight', [d], 'F16'));
    specs.add(_TensorSpec('$p.mlp.gate_proj.weight', [ffn, d], 'F16'));
    specs.add(_TensorSpec('$p.mlp.up_proj.weight', [ffn, d], 'F16'));
    specs.add(_TensorSpec('$p.mlp.down_proj.weight', [d, ffn], 'F16'));
  }
  specs.add(_TensorSpec('model.norm.weight', [d], 'F16'));
  if (!cfg.tieWeights) {
    specs.add(_TensorSpec('lm_head.weight', [cfg.vocabSize, d], 'F16'));
  }
  return specs;
}

// ---------------------------------------------------------------------------
// Safetensors writer — chunked so peak RAM is bounded regardless of
// tensor size. Random data in fp16 with |x| < 0.02 (well inside the
// fp16 normal range, no NaN/Inf).
// ---------------------------------------------------------------------------

void _writeShardFile(String path, List<_TensorSpec> specs, math.Random rng) {
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
      'dtype': s.dtype,
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

const int _writeChunkBytes = 4 * 1024 * 1024;

void _streamRandomBytes(RandomAccessFile raf, int nBytes, math.Random rng) {
  final buf = Uint8List(_writeChunkBytes);
  final bd = ByteData.sublistView(buf);
  var remaining = nBytes;
  while (remaining > 0) {
    final take = remaining >= _writeChunkBytes ? _writeChunkBytes : remaining;
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
  final e = exp32 - 127 + 15;
  if (e >= 0x1F) return (sign << 15) | 0x7C00;
  if (e <= 0) return sign << 15;
  return (sign << 15) | (e << 10) | (mant32 >> 13);
}

// ---------------------------------------------------------------------------
// Multi-shard writer — HF layout: model-00001-of-000NN.safetensors +
// model.safetensors.index.json.
// ---------------------------------------------------------------------------

void _writeShardedCheckpoint(
  String outDir,
  List<_TensorSpec> specs,
  int nShards,
  int seed,
) {
  final buckets = _splitShards(specs, nShards);
  final weightMap = <String, String>{};
  var totalBytes = 0;
  final rng = math.Random(seed);
  for (int i = 0; i < buckets.length; i++) {
    final shardName = 'model-${_pad5(i + 1)}-of-${_pad5(nShards)}.safetensors';
    final path = '$outDir/$shardName';
    _writeShardFile(path, buckets[i], rng);
    for (final s in buckets[i]) {
      weightMap[s.name] = shardName;
      totalBytes += s.bytes;
    }
  }
  final index = {
    'metadata': {'total_size': totalBytes},
    'weight_map': weightMap,
  };
  File(
    '$outDir/model.safetensors.index.json',
  ).writeAsStringSync(const JsonEncoder.withIndent('  ').convert(index));
}

List<List<_TensorSpec>> _splitShards(List<_TensorSpec> specs, int n) {
  if (n <= 1) return [specs];
  final total = specs.fold<int>(0, (a, s) => a + s.bytes);
  final target = total / n;
  final out = List.generate(n, (_) => <_TensorSpec>[]);
  var acc = 0;
  var idx = 0;
  for (final s in specs) {
    while (idx < n - 1 && acc >= (idx + 1) * target) {
      idx++;
    }
    out[idx].add(s);
    acc += s.bytes;
  }
  return out;
}

String _pad5(int n) => n.toString().padLeft(5, '0');

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
