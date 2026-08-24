/// AirLLM-style **layer-streaming** GPT-J runner.
///
/// GPT-J-6B is ~24 GB fp32 / ~12 GB fp16 — it does NOT fit on a 6 GB
/// GPU, and even on CPU the merged safetensors map plus a fresh
/// fp32 model init peaks at ~40 GB. This runner keeps only:
///
///   * `transformer.wte.weight` (~800 MB fp32 or ~400 MB fp16),
///   * `transformer.ln_f.{weight,bias}` (tiny),
///   * `lm_head.{weight,bias}` (~800 MB fp32 or ~400 MB fp16),
///   * one resident [GPTJBlock],
///   * a [RopeCache].
///
/// For GPT-J-6B fp16 that peaks at ~1.5–2 GB resident — fits on any
/// laptop. Per-token latency is dominated by streaming 28 layers ×
/// ~400 MB from disk on each forward pass, so this is an "it fits"
/// runner, not a "fast" runner.
///
/// Only difference from [LlamaStreamingRunner] besides the key
/// namespace: GPT-J's Q/K weights need the same **interleaved →
/// half-split rotary row permutation** applied per head that the
/// non-streaming [GPTJHFLoader] does. That happens inline in
/// [_swapLayer].
library;

import 'dart:typed_data';

import '../tensor/tensor.dart';
import '../tensor/dtype.dart';
import 'embedding.dart';
import 'gptj.dart';
import 'kv_cache.dart';
import 'layer_norm.dart';
import 'linear.dart';
import 'masks.dart';
import 'rotary.dart';
import 'safetensors_reader.dart';

class GPTJStreamingRunner {
  final GPTJConfig config;
  final ShardedSafeTensorsReader reader;

  /// Load fp16 blobs as raw Uint16List (half the resident RAM).
  final bool keepFp16;

  /// Print per-layer swap timings.
  final bool profile;

  final Embedding wte;
  final LayerNorm finalLn;
  final Linear lmHead;
  final RopeCache rope;

  final GPTJBlock residentBlock;

  GPTJStreamingRunner(
    this.config,
    this.reader, {
    this.keepFp16 = true,
    this.profile = false,
  }) : wte = Embedding(
         config.vocabSize,
         config.embedDim,
         device: config.embedDevice,
         seed: config.seed,
       ),
       finalLn = LayerNorm(config.embedDim, device: config.lmDevice),
       lmHead = Linear(
         config.embedDim,
         config.vocabSize,
         bias: true,
         device: config.lmDevice,
         seed: config.seed + 900000,
       ),
       rope = RopeCache(
         maxCtx: config.maxCtx,
         headDim: config.embedDim ~/ config.numHeads,
         rotaryDim: config.rotaryDim,
         base: config.ropeBase,
         device: Device.CPU,
       ),
       residentBlock = GPTJBlock(
         config.embedDim,
         config.numHeads,
         ffnDim: config.ffnDim,
         rope: RopeCache(
           maxCtx: config.maxCtx,
           headDim: config.embedDim ~/ config.numHeads,
           rotaryDim: config.rotaryDim,
           base: config.ropeBase,
           device: Device.CPU,
         ),
         device: Device.CPU,
         seed: config.seed + 100000,
       ) {
    if (config.embedDevice != Device.CPU || config.lmDevice != Device.CPU) {
      throw StateError(
        'GPTJStreamingRunner: only CPU is supported '
        '(embed=${config.embedDevice}, lm=${config.lmDevice})',
      );
    }
    _loadPersistent();
  }

  void _loadPersistent() {
    final d = config.embedDim;
    _copy(
      wte.weight,
      _expectShape(
        reader.readTensor('transformer.wte.weight', keepFp16: keepFp16),
        [config.vocabSize, d],
        'transformer.wte.weight',
      ),
    );
    _copy(
      finalLn.gamma,
      _expectShape(reader.readTensor('transformer.ln_f.weight'), [
        d,
      ], 'transformer.ln_f.weight'),
    );
    _copy(
      finalLn.beta,
      _expectShape(reader.readTensor('transformer.ln_f.bias'), [
        d,
      ], 'transformer.ln_f.bias'),
    );
    _copy(
      lmHead.weight,
      _expectShape(reader.readTensor('lm_head.weight', keepFp16: keepFp16), [
        config.vocabSize,
        d,
      ], 'lm_head.weight'),
    );
    _copy(
      lmHead.bias!,
      _reshape1xN(
        _expectShape(reader.readTensor('lm_head.bias'), [
          config.vocabSize,
        ], 'lm_head.bias'),
      ),
    );
  }

  /// Swap layer [i]'s weights into `residentBlock`. Applies the
  /// GPT-J interleaved→half-split rotary row permutation on the
  /// Q and K per-head slices.
  void _swapLayer(int i) {
    final sw = profile ? (Stopwatch()..start()) : null;

    final cfg = config;
    final d = cfg.embedDim;
    final h = cfg.numHeads;
    final headDim = d ~/ h;
    final ffn = cfg.ffnDim;
    final rDim = cfg.rotaryDim;
    final p = 'transformer.h.$i';

    _copy(
      residentBlock.ln.gamma,
      _expectShape(reader.readTensor('$p.ln_1.weight'), [d], '$p.ln_1.weight'),
    );
    _copy(
      residentBlock.ln.beta,
      _expectShape(reader.readTensor('$p.ln_1.bias'), [d], '$p.ln_1.bias'),
    );

    final qW = _expectShape(
      reader.readTensor('$p.attn.q_proj.weight', keepFp16: keepFp16),
      [d, d],
      '$p.attn.q_proj.weight',
    );
    final kW = _expectShape(
      reader.readTensor('$p.attn.k_proj.weight', keepFp16: keepFp16),
      [d, d],
      '$p.attn.k_proj.weight',
    );
    final vW = _expectShape(
      reader.readTensor('$p.attn.v_proj.weight', keepFp16: keepFp16),
      [d, d],
      '$p.attn.v_proj.weight',
    );
    for (int hh = 0; hh < h; hh++) {
      final start = hh * headDim;
      final qHead = qW.sliceRows(start, start + headDim);
      final kHead = kW.sliceRows(start, start + headDim);
      final vHead = vW.sliceRows(start, start + headDim);
      _copy(residentBlock.attn.wq[hh].weight, _permuteRotaryRows(qHead, rDim));
      _copy(residentBlock.attn.wk[hh].weight, _permuteRotaryRows(kHead, rDim));
      _copy(residentBlock.attn.wv[hh].weight, vHead);
    }

    _copy(
      residentBlock.attn.wo.weight,
      _expectShape(
        reader.readTensor('$p.attn.out_proj.weight', keepFp16: keepFp16),
        [d, d],
        '$p.attn.out_proj.weight',
      ),
    );

    _copy(
      residentBlock.ffn1.weight,
      _expectShape(
        reader.readTensor('$p.mlp.fc_in.weight', keepFp16: keepFp16),
        [ffn, d],
        '$p.mlp.fc_in.weight',
      ),
    );
    _copy(
      residentBlock.ffn1.bias!,
      _reshape1xN(
        _expectShape(reader.readTensor('$p.mlp.fc_in.bias'), [
          ffn,
        ], '$p.mlp.fc_in.bias'),
      ),
    );
    _copy(
      residentBlock.ffn2.weight,
      _expectShape(
        reader.readTensor('$p.mlp.fc_out.weight', keepFp16: keepFp16),
        [d, ffn],
        '$p.mlp.fc_out.weight',
      ),
    );
    _copy(
      residentBlock.ffn2.bias!,
      _reshape1xN(
        _expectShape(reader.readTensor('$p.mlp.fc_out.bias'), [
          d,
        ], '$p.mlp.fc_out.bias'),
      ),
    );

    if (sw != null) {
      sw.stop();
      // ignore: avoid_print
      print('  [layer $i] swap ${sw.elapsedMilliseconds} ms');
    }
  }

  Tensor forward(
    Tensor tokens, {
    int startPos = 0,
    EncoderCache? cache,
  }) {
    if (tokens.shape.length != 1) {
      throw ArgumentError(
        'GPTJStreamingRunner: tokens must be 1D [seqLen]; '
        'got ${tokens.shape}',
      );
    }
    final n = tokens.shape.last;
    if (startPos + n > config.maxCtx) {
      throw ArgumentError(
        'GPTJStreamingRunner: window [$startPos, ${startPos + n}) '
        'exceeds maxCtx ${config.maxCtx}',
      );
    }
    return Tensor.noGrad(() {
      var x = wte(tokens);
      final mask = n > 1 ? causalMask(n, device: x.device) : null;
      for (int i = 0; i < config.numLayers; i++) {
        _swapLayer(i);
        final layerCache = cache?.layers[i];
        x = residentBlock(
          x,
          mask: mask,
          cache: layerCache,
          startPos: startPos,
        );
      }
      x = finalLn(x);
      return lmHead(x);
    });
  }

  /// Greedy autoregressive decode. With [useCache] on (default), a
  /// persistent per-layer KV cache is kept across steps so each
  /// generated token only runs a **single-token** forward through
  /// every streamed layer, rather than re-projecting the entire
  /// prefix.
  List<double> generate(
    List<double> prompt, {
    required int maxNewTokens,
    bool useCache = true,
  }) {
    if (prompt.isEmpty) {
      throw ArgumentError('generate: prompt must be non-empty');
    }
    if (useCache) return _generateCached(prompt, maxNewTokens);
    return _generateNoCache(prompt, maxNewTokens);
  }

  List<double> _generateCached(List<double> prompt, int maxNewTokens) {
    final v = config.vocabSize;
    final out = List<double>.of(prompt);
    final cache = EncoderCache.empty(config.numLayers, config.numHeads);
    final promptT = Tensor.fromList([prompt.length], prompt, device: Device.CPU);
    var logits = forward(promptT, startPos: 0, cache: cache).toList();
    out.add(_argmax(logits, (prompt.length - 1) * v, v).toDouble());
    for (int step = 1; step < maxNewTokens; step++) {
      if (cache.seqLen >= config.maxCtx) break;
      final oneT = Tensor.fromList([1], [out.last], device: Device.CPU);
      logits = forward(oneT, startPos: cache.seqLen, cache: cache).toList();
      out.add(_argmax(logits, 0, v).toDouble());
    }
    return out;
  }

  List<double> _generateNoCache(List<double> prompt, int maxNewTokens) {
    final v = config.vocabSize;
    final out = List<double>.of(prompt);
    for (int step = 0; step < maxNewTokens; step++) {
      if (out.length >= config.maxCtx) break;
      final ctx = Tensor.fromList([out.length], out, device: Device.CPU);
      final logits = forward(ctx).toList();
      out.add(_argmax(logits, (out.length - 1) * v, v).toDouble());
    }
    return out;
  }

  static int _argmax(List<double> row, int base, int len) {
    var best = double.negativeInfinity;
    var arg = 0;
    for (int t = 0; t < len; t++) {
      final val = row[base + t];
      if (val > best) {
        best = val;
        arg = t;
      }
    }
    return arg;
  }

  void close() => reader.close();

  // ---------------- helpers ----------------

  static Tensor _expectShape(Tensor t, List<int> expected, String name) {
    if (t.shape.length != expected.length) {
      throw ArgumentError(
        'gpt-j streaming: "$name" expected shape $expected, got ${t.shape}',
      );
    }
    for (int i = 0; i < expected.length; i++) {
      if (t.shape[i] != expected[i]) {
        throw ArgumentError(
          'gpt-j streaming: "$name" expected shape $expected, got ${t.shape}',
        );
      }
    }
    return t;
  }

  static void _copy(Tensor dst, Tensor src) {
    if (dst.length != src.length) {
      throw ArgumentError(
        'gpt-j streaming: copy length mismatch dst=${dst.shape} '
        'src=${src.shape}',
      );
    }
    if (src.dtype == DType.fp16 && dst.device == Device.CPU) {
      dst.adoptCpuStorageFrom(src);
      return;
    }
    final vals = src.toList();
    final matched = Tensor.fromList(dst.shape, vals, device: dst.device);
    dst.assign(matched);
  }

  static Tensor _reshape1xN(Tensor v) {
    if (v.shape.length != 1) {
      throw ArgumentError('_reshape1xN: expected rank 1, got ${v.shape}');
    }
    return Tensor.fromList([1, v.shape[0]], v.toList(), device: Device.CPU);
  }

  /// GPT-J's interleaved-pair rotary → our half-split rotary
  /// permutation, applied to the first `rDim` rows of a [headDim, D]
  /// weight slice. Rows [rDim, headDim) pass through.
  ///
  /// Produces an fp32 buffer even when the input is fp16 — this is
  /// unavoidable because the permutation reads and re-writes rows.
  /// The peak cost is one Q + one K head slice per layer (headDim=256,
  /// D=4096 → 4 MB fp32 each). Fine.
  static Tensor _permuteRotaryRows(Tensor headSlice, int rDim) {
    if (headSlice.shape.length != 2) {
      throw ArgumentError(
        '_permuteRotaryRows: expected rank 2, got ${headSlice.shape}',
      );
    }
    final headDim = headSlice.shape[0];
    final d = headSlice.shape[1];
    if (rDim <= 0 || rDim > headDim || rDim.isOdd) {
      throw ArgumentError(
        '_permuteRotaryRows: rDim=$rDim invalid for headDim=$headDim',
      );
    }
    final halfR = rDim ~/ 2;
    final src = headSlice.toList();
    final out = Float32List(headDim * d);
    for (int i = 0; i < halfR; i++) {
      final srcBase0 = (2 * i) * d;
      final dstBase0 = i * d;
      for (int c = 0; c < d; c++) {
        out[dstBase0 + c] = src[srcBase0 + c];
      }
      final srcBase1 = (2 * i + 1) * d;
      final dstBase1 = (i + halfR) * d;
      for (int c = 0; c < d; c++) {
        out[dstBase1 + c] = src[srcBase1 + c];
      }
    }
    for (int r = rDim; r < headDim; r++) {
      final base = r * d;
      for (int c = 0; c < d; c++) {
        out[base + c] = src[base + c];
      }
    }
    return Tensor.fromList([headDim, d], out, device: Device.CPU);
  }
}

/// Sum of on-disk bytes for one GPT-J transformer layer, computed
/// from the safetensors header (no tensor decoding).
int estimateGPTJLayerBytes(
  ShardedSafeTensorsReader reader,
  GPTJConfig cfg, {
  int layer = 0,
}) {
  final p = 'transformer.h.$layer';
  final keys = <String>[
    '$p.ln_1.weight',
    '$p.ln_1.bias',
    '$p.attn.q_proj.weight',
    '$p.attn.k_proj.weight',
    '$p.attn.v_proj.weight',
    '$p.attn.out_proj.weight',
    '$p.mlp.fc_in.weight',
    '$p.mlp.fc_in.bias',
    '$p.mlp.fc_out.weight',
    '$p.mlp.fc_out.bias',
  ];
  var total = 0;
  for (final k in keys) {
    final b = reader.tensorBytes(k);
    if (b != null) total += b;
  }
  return total;
}
