/// AirLLM-style **layer-streaming** Llama runner.
///
/// Instead of holding all `L` transformer layers in memory at once,
/// this runner keeps:
///
///   * the token embedding (`embed_tokens.weight`),
///   * the final RMSNorm,
///   * the untied lm_head (if any),
///   * a **single** resident [LlamaBlock],
///   * a [RopeCache].
///
/// During `forward`, each layer's parameters are pulled from disk
/// via a [ShardedSafeTensorsReader], `adoptCpuStorageFrom`'d into the
/// resident block's tensors, and the block is run once. Peak resident
/// memory ≈ embed table + one layer's weights + activations.
///
/// This is the trick AirLLM uses to fit 70B models on a 4 GB card:
/// per-token latency is dominated by disk→RAM bandwidth × numLayers,
/// but the model *fits*.
///
/// **Constraints:**
///   * CPU only. GPU streaming would require a re-upload per layer,
///     which the underlying [Tensor.adoptCpuStorageFrom] fast path
///     does not support.
///   * fp16 checkpoints strongly preferred — the fp16 fast path in
///     the loader avoids fp32 promotion, halving both disk I/O and
///     resident layer bytes.
///   * No KV cache in this first cut. Every forward re-processes the
///     full prefix. For research / porting demos this is fine; for
///     production autoregression a resident-per-layer KV cache is
///     the natural next step.
///
///   final reader = ShardedSafeTensorsReader.open('model.safetensors');
///   final runner = LlamaStreamingRunner(cfg, reader);
///   final out = runner.generate(promptIds, maxNewTokens: 20);
library;

import 'dart:math' as math;
import 'dart:typed_data';

import '../tensor/tensor.dart';
import '../tensor/dtype.dart';
import 'kv_cache.dart';
import 'llama.dart';
import 'masks.dart';
import 'rms_norm.dart';
import 'rotary.dart';
import 'safetensors_reader.dart';

class LlamaStreamingRunner {
  final LlamaConfig config;
  final ShardedSafeTensorsReader reader;

  /// Load fp16 blobs as raw Uint16List (half the resident RAM per
  /// layer). Requires the checkpoint itself to be fp16.
  final bool keepFp16;

  /// Print per-layer swap timings.
  final bool profile;

  /// Persistent token embedding table, loaded fp16 directly from
  /// disk — never allocated as fp32 to keep peak RAM low. Shape
  /// `[vocabSize, embedDim]`.
  late final Tensor _embedWeight;

  /// Persistent untied lm_head weight (null for tied models). Shape
  /// `[vocabSize, embedDim]`, fp16.
  late final Tensor? _untiedHeadWeight;

  final RMSNorm finalNorm;
  final RopeCache rope;

  /// The single resident block whose weight storage is swapped in
  /// place before every layer's forward.
  final LlamaBlock residentBlock;

  LlamaStreamingRunner(
    this.config,
    this.reader, {
    this.keepFp16 = true,
    this.profile = false,
  }) : finalNorm = RMSNorm(
         config.embedDim,
         eps: config.rmsNormEps,
         device: config.device,
       ),
       rope = RopeCache(
         maxCtx: config.maxCtx,
         headDim: config.embedDim ~/ config.numHeads,
         base: config.ropeBase,
         device: config.device,
       ),
       residentBlock = LlamaBlock(
         config.embedDim,
         config.numHeads,
         numKvHeads: config.numKvHeads,
         ffnDim: config.ffnDim,
         rope: RopeCache(
           maxCtx: config.maxCtx,
           headDim: config.embedDim ~/ config.numHeads,
           base: config.ropeBase,
           device: config.device,
         ),
         rmsNormEps: config.rmsNormEps,
         attentionBias: config.attentionBias,
         outBias: config.outBias,
         device: config.device,
         seed: config.seed + 100000,
       ) {
    if (config.device != Device.CPU) {
      throw StateError(
        'LlamaStreamingRunner: only CPU is supported (got ${config.device})',
      );
    }
    _loadPersistent();
  }

  void _loadPersistent() {
    final d = config.embedDim;
    _embedWeight = _expectShape(
      reader.readTensor('model.embed_tokens.weight', keepFp16: keepFp16),
      [config.vocabSize, d],
      'model.embed_tokens.weight',
    );
    _copy(
      finalNorm.gamma,
      _expectShape(reader.readTensor('model.norm.weight'), [
        d,
      ], 'model.norm.weight'),
    );
    if (config.tieWeights) {
      _untiedHeadWeight = null;
    } else {
      _untiedHeadWeight = _expectShape(
        reader.readTensor('lm_head.weight', keepFp16: keepFp16),
        [config.vocabSize, d],
        'lm_head.weight',
      );
    }
  }

  /// Swap layer [i]'s weights into `residentBlock`. Mirrors
  /// `LlamaHFLoader.loadMap`'s per-layer section but pulls each
  /// tensor via [reader] instead of a merged in-memory map.
  void _swapLayer(int i) {
    final sw = profile ? (Stopwatch()..start()) : null;

    final cfg = config;
    final d = cfg.embedDim;
    final h = cfg.numHeads;
    final kvH = cfg.numKvHeads;
    final headDim = d ~/ h;
    final ffn = cfg.ffnDim;
    final p = 'model.layers.$i';

    _copy(
      residentBlock.attnNorm.gamma,
      _expectShape(
        reader.readTensor('$p.input_layernorm.weight'),
        [d],
        '$p.input_layernorm.weight',
      ),
    );
    _copy(
      residentBlock.ffnNorm.gamma,
      _expectShape(
        reader.readTensor('$p.post_attention_layernorm.weight'),
        [d],
        '$p.post_attention_layernorm.weight',
      ),
    );

    final qW = _expectShape(
      reader.readTensor('$p.self_attn.q_proj.weight', keepFp16: keepFp16),
      [h * headDim, d],
      '$p.self_attn.q_proj.weight',
    );
    for (int hh = 0; hh < h; hh++) {
      _copy(
        residentBlock.attn.wq[hh].weight,
        qW.sliceRows(hh * headDim, (hh + 1) * headDim),
      );
    }
    final kW = _expectShape(
      reader.readTensor('$p.self_attn.k_proj.weight', keepFp16: keepFp16),
      [kvH * headDim, d],
      '$p.self_attn.k_proj.weight',
    );
    for (int hh = 0; hh < kvH; hh++) {
      _copy(
        residentBlock.attn.wk[hh].weight,
        kW.sliceRows(hh * headDim, (hh + 1) * headDim),
      );
    }
    final vW = _expectShape(
      reader.readTensor('$p.self_attn.v_proj.weight', keepFp16: keepFp16),
      [kvH * headDim, d],
      '$p.self_attn.v_proj.weight',
    );
    for (int hh = 0; hh < kvH; hh++) {
      _copy(
        residentBlock.attn.wv[hh].weight,
        vW.sliceRows(hh * headDim, (hh + 1) * headDim),
      );
    }

    if (cfg.attentionBias) {
      final qB = _expectShape(reader.readTensor('$p.self_attn.q_proj.bias'), [
        h * headDim,
      ], '$p.self_attn.q_proj.bias');
      for (int hh = 0; hh < h; hh++) {
        _copy(
          residentBlock.attn.wq[hh].bias!,
          _reshape1xN(_sliceVector(qB, hh * headDim, (hh + 1) * headDim)),
        );
      }
      final kB = _expectShape(reader.readTensor('$p.self_attn.k_proj.bias'), [
        kvH * headDim,
      ], '$p.self_attn.k_proj.bias');
      for (int hh = 0; hh < kvH; hh++) {
        _copy(
          residentBlock.attn.wk[hh].bias!,
          _reshape1xN(_sliceVector(kB, hh * headDim, (hh + 1) * headDim)),
        );
      }
      final vB = _expectShape(reader.readTensor('$p.self_attn.v_proj.bias'), [
        kvH * headDim,
      ], '$p.self_attn.v_proj.bias');
      for (int hh = 0; hh < kvH; hh++) {
        _copy(
          residentBlock.attn.wv[hh].bias!,
          _reshape1xN(_sliceVector(vB, hh * headDim, (hh + 1) * headDim)),
        );
      }
    }

    _copy(
      residentBlock.attn.wo.weight,
      _expectShape(
        reader.readTensor('$p.self_attn.o_proj.weight', keepFp16: keepFp16),
        [d, d],
        '$p.self_attn.o_proj.weight',
      ),
    );
    _copy(
      residentBlock.ffn.gateProj.weight,
      _expectShape(
        reader.readTensor('$p.mlp.gate_proj.weight', keepFp16: keepFp16),
        [ffn, d],
        '$p.mlp.gate_proj.weight',
      ),
    );
    _copy(
      residentBlock.ffn.upProj.weight,
      _expectShape(
        reader.readTensor('$p.mlp.up_proj.weight', keepFp16: keepFp16),
        [ffn, d],
        '$p.mlp.up_proj.weight',
      ),
    );
    _copy(
      residentBlock.ffn.downProj.weight,
      _expectShape(
        reader.readTensor('$p.mlp.down_proj.weight', keepFp16: keepFp16),
        [d, ffn],
        '$p.mlp.down_proj.weight',
      ),
    );

    if (sw != null) {
      sw.stop();
      // ignore: avoid_print
      print('  [layer $i] swap ${sw.elapsedMilliseconds} ms');
    }
  }

  /// Single forward. `tokens` is a 1D `[seqLen]` tensor of token ids
  /// as float32. Returns `[seqLen, vocab]` logits.
  ///
  /// If [cache] is supplied, per-layer K/V is appended to it (prompt
  /// fill: `startPos=0`, `seqLen>=1`; single-token append:
  /// `seqLen==1`, `startPos == cache.seqLen`). The cache survives
  /// layer swaps because its K/V tensors are separate from the
  /// resident block's weight tensors.
  Tensor forward(
    Tensor tokens, {
    int startPos = 0,
    EncoderCache? cache,
  }) {
    if (tokens.shape.length != 1) {
      throw ArgumentError(
        'LlamaStreamingRunner: tokens must be 1D [seqLen]; got ${tokens.shape}',
      );
    }
    final n = tokens.shape.last;
    if (startPos + n > config.maxCtx) {
      throw ArgumentError(
        'LlamaStreamingRunner: window [$startPos, ${startPos + n}) '
        'exceeds maxCtx ${config.maxCtx}',
      );
    }
    return Tensor.noGrad(() {
      var x = _embedWeight.embedding(tokens);
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
      x = finalNorm(x);
      final head = config.tieWeights ? _embedWeight : _untiedHeadWeight!;
      return x.matmul(head.transpose());
    });
  }

  /// Greedy autoregressive decode. With [useCache] on (default), a
  /// persistent per-layer KV cache is kept across steps so each
  /// generated token only runs a **single-token** forward through
  /// each streamed layer, rather than re-projecting the entire
  /// prefix. Layer disk I/O is unchanged (still numLayers reads per
  /// token), but per-layer matmul cost drops from `O(prefix × D²)`
  /// to `O(1 × D²)` after the prompt.
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
    final cache = EncoderCache.empty(config.numLayers, config.numKvHeads);

    final promptT = Tensor.fromList(
      [prompt.length],
      prompt,
      device: config.device,
    );
    var logits = forward(promptT, startPos: 0, cache: cache).toList();
    var lastBase = (prompt.length - 1) * v;
    out.add(_argmax(logits, lastBase, v).toDouble());

    for (int step = 1; step < maxNewTokens; step++) {
      if (cache.seqLen >= config.maxCtx) break;
      final oneT = Tensor.fromList(
        [1],
        [out.last],
        device: config.device,
      );
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
      final ctx = Tensor.fromList([out.length], out, device: config.device);
      final logits = forward(ctx).toList();
      final base = (out.length - 1) * v;
      out.add(_argmax(logits, base, v).toDouble());
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

  // ---------------- helpers (mirror LlamaHFLoader's privates) ----------------

  static Tensor _expectShape(Tensor t, List<int> expected, String name) {
    if (t.shape.length != expected.length) {
      throw ArgumentError(
        'streaming: "$name" expected shape $expected, got ${t.shape}',
      );
    }
    for (int i = 0; i < expected.length; i++) {
      if (t.shape[i] != expected[i]) {
        throw ArgumentError(
          'streaming: "$name" expected shape $expected, got ${t.shape}',
        );
      }
    }
    return t;
  }

  static void _copy(Tensor dst, Tensor src) {
    if (dst.length != src.length) {
      throw ArgumentError(
        'streaming: copy length mismatch dst=${dst.shape} src=${src.shape}',
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

  static Tensor _sliceVector(Tensor t, int start, int end) {
    final src = t.toList();
    final n = end - start;
    final out = Float32List(n);
    for (int i = 0; i < n; i++) {
      out[i] = src[start + i];
    }
    return Tensor.fromList([n], out, device: Device.CPU);
  }

  static Tensor _reshape1xN(Tensor v) {
    if (v.shape.length != 1) {
      throw ArgumentError('_reshape1xN: expected rank 1, got ${v.shape}');
    }
    return Tensor.fromList([1, v.shape[0]], v.toList(), device: Device.CPU);
  }
}

/// Sum of on-disk weight bytes for one layer, computed from the
/// reader's header (no tensor decoding). Useful for planning: multiply
/// by the number of resident layers you want to fit.
int estimateLayerBytes(
  ShardedSafeTensorsReader reader,
  LlamaConfig cfg, {
  int layer = 0,
}) {
  final p = 'model.layers.$layer';
  final keys = <String>[
    '$p.input_layernorm.weight',
    '$p.post_attention_layernorm.weight',
    '$p.self_attn.q_proj.weight',
    '$p.self_attn.k_proj.weight',
    '$p.self_attn.v_proj.weight',
    '$p.self_attn.o_proj.weight',
    '$p.mlp.gate_proj.weight',
    '$p.mlp.up_proj.weight',
    '$p.mlp.down_proj.weight',
  ];
  if (cfg.attentionBias) {
    keys.addAll([
      '$p.self_attn.q_proj.bias',
      '$p.self_attn.k_proj.bias',
      '$p.self_attn.v_proj.bias',
    ]);
  }
  var total = 0;
  for (final k in keys) {
    final b = reader.tensorBytes(k);
    if (b != null) total += b;
  }
  return total;
}

// Unused import guard so `math` stays if we add sampling later.
// ignore: unused_element
math.Random _rngPlaceholder() => math.Random(0);
