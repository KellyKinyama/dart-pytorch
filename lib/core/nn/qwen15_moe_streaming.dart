/// AirLLM-style layer-streaming runner for
/// `Qwen/Qwen1.5-MoE-A2.7B-Chat` (Qwen2-MoE architecture).
///
/// Same pattern as [LlamaStreamingRunner] with the MoE FFN swapped
/// in: per-token, only the routed top-K of E=60 experts are actually
/// streamed off disk per layer (via [MoeStreamingLayer]). Everything
/// else that's not a per-expert weight — MHA (plain, `numKvHeads ==
/// numHeads == 16`, Q/K/V bias, O no-bias), RMSNorms, router weight,
/// shared expert (fused SwiGLU with hidden=5632 scalar-gated by
/// `shared_expert_gate`) — is loaded once per layer swap.
///
/// **CPU only** (uses `Tensor.adoptCpuStorageFrom` for the fp16 fast
/// path). **fp16 checkpoints only**. Handles bf16 checkpoints too
/// but promotes to fp32 on load, doubling the resident cost.
///
/// Peak resident memory (fp16 checkpoint, one-token generation):
///
///   * `embed_tokens.weight`: ~600 MB (151936 × 2048 × 2)
///   * `lm_head.weight`: ~600 MB (untied)
///   * one resident block: ~100 MB (MHA + norms + router +
///     shared_expert + shared_expert_gate)
///   * streaming scratch: K × ~11 MB fp16 = ~44 MB
///   * activations: few MB per token
///
/// Total ~1.4 GB — well inside a 15 GB WSL RAM cap despite the
/// checkpoint's 28.6 GB on-disk footprint. This is the AirLLM claim
/// applied concretely: sparse-MoE lets a 14B-total / 2.7B-active
/// model run in ~1.5 GB.
library;

import 'dart:io' as io;
import 'dart:typed_data';

import '../tensor/tensor.dart';
import '../tensor/dtype.dart';
import 'attention/multi_head_attention.dart';
import 'kv_cache.dart';
import 'masks.dart';
import 'moe.dart' show GateFunction;
import 'moe_streaming.dart';
import 'rms_norm.dart';
import 'rotary.dart';
import 'safetensors_reader.dart';

/// Config for one Qwen2-MoE model. Defaults match
/// `Qwen/Qwen1.5-MoE-A2.7B-Chat` config.json.
class Qwen15MoEConfig {
  final int vocabSize;
  final int dim;
  final int numHeads;
  final int numKvHeads;
  final int numLayers;
  final int moeHidden;
  final int sharedHidden;
  final int numExperts;
  final int topK;
  final int maxCtx;
  final double ropeBase;
  final double rmsNormEps;
  final bool tieWeights;

  const Qwen15MoEConfig({
    this.vocabSize = 151936,
    this.dim = 2048,
    this.numHeads = 16,
    this.numKvHeads = 16,
    this.numLayers = 24,
    this.moeHidden = 1408,
    this.sharedHidden = 5632,
    this.numExperts = 60,
    this.topK = 4,
    this.maxCtx = 32768,
    this.ropeBase = 1000000.0,
    this.rmsNormEps = 1e-6,
    this.tieWeights = false,
  });

  int get headDim => dim ~/ numHeads;
}

class Qwen15MoEStreamingRunner {
  final Qwen15MoEConfig config;
  final ShardedSafeTensorsReader reader;
  final bool keepFp16;
  final bool profile;

  /// Persistent — loaded fp16 directly from disk.
  late final Tensor _embedWeight;

  /// Pre-transposed output projection `[D, vocab]`, cached at load
  /// time so `forward` doesn't rebuild a 622 MB tensor every call.
  /// For tied weights this points at `_embedWeight.transpose()` (also
  /// cached, so both the lookup table and the head projection are
  /// resident). For untied weights only this transposed copy is kept —
  /// the raw `[vocab, D]` layout is dropped after transposing.
  late final Tensor _headT;

  final RMSNorm finalNorm;
  final RopeCache rope;

  /// Single resident MHA + norms + router + shared expert. Weight
  /// storage is swapped per layer via `adoptCpuStorageFrom`.
  final RMSNorm attnNorm;
  final RMSNorm ffnNorm;
  final MultiHeadAttention attn;

  /// Router weight `[D, E]`. Loaded per-layer swap from HF's
  /// `mlp.gate.weight` (shape `[E, D]`) via `Tensor.transpose()`
  /// which promotes to fp32 — held as fp32.
  Tensor _routerW;

  /// Shared expert SwiGLU: shape `[hidden_shared, D]` for
  /// `gate`/`up`, `[D, hidden_shared]` for `down`. fp16 storage
  /// preserved across swaps.
  Tensor _sharedGate;
  Tensor _sharedUp;
  Tensor _sharedDown;

  /// Pre-transposed shared expert weights, computed at swap time so
  /// `forward` doesn't allocate 3 × 22 MB fp16 transposes per layer
  /// per token. `_sharedGateT` and `_sharedUpT` are `[D, hidden]`,
  /// `_sharedDownT` is `[hidden, D]`.
  Tensor _sharedGateT;
  Tensor _sharedUpT;
  Tensor _sharedDownT;

  /// Learned scalar gate `[1, D]` applied via `sigmoid(x @
  /// shared_expert_gate.T)` to weight the shared expert output.
  Tensor _sharedExpertGate;

  /// Pre-transposed `[D, 1]` version cached at swap.
  Tensor _sharedExpertGateT;

  /// Reused across forwards for row-broadcasting `[T, 1] @ [1, D]
  /// → [T, D]`. Allocated once.
  late final Tensor _onesD;

  Qwen15MoEStreamingRunner(
    this.config,
    this.reader, {
    this.keepFp16 = true,
    this.profile = false,
  }) : finalNorm = RMSNorm(config.dim, eps: config.rmsNormEps),
       rope = RopeCache(
         maxCtx: config.maxCtx,
         headDim: config.dim ~/ config.numHeads,
         base: config.ropeBase,
       ),
       attnNorm = RMSNorm(config.dim, eps: config.rmsNormEps),
       ffnNorm = RMSNorm(config.dim, eps: config.rmsNormEps),
       attn = MultiHeadAttention(
         config.dim,
         config.numHeads,
         numKvHeads: config.numKvHeads,
         bias: true, // Qwen has bias on Q/K/V
         outBias: false, // but not on O
         seed: 0,
       ),
       _routerW = Tensor.fill([config.dim, config.numExperts], 0.0),
       _sharedGate = Tensor.fill([config.sharedHidden, config.dim], 0.0),
       _sharedUp = Tensor.fill([config.sharedHidden, config.dim], 0.0),
       _sharedDown = Tensor.fill([config.dim, config.sharedHidden], 0.0),
       _sharedGateT = Tensor.fill([config.dim, config.sharedHidden], 0.0),
       _sharedUpT = Tensor.fill([config.dim, config.sharedHidden], 0.0),
       _sharedDownT = Tensor.fill([config.sharedHidden, config.dim], 0.0),
       _sharedExpertGate = Tensor.fill([1, config.dim], 0.0),
       _sharedExpertGateT = Tensor.fill([config.dim, 1], 0.0) {
    attn.rope = rope;
    _onesD = Tensor.fill([1, config.dim], 1.0);
    _ramGuard();
    _loadPersistent();
  }

  /// Pessimistic peak-RAM estimate and abort if it exceeds
  /// `/proc/meminfo` `MemAvailable`. Prevents the WSL crush loop when
  /// the caller picks a config that can't fit.
  void _ramGuard() {
    final d = config.dim;
    final embedBytes = config.vocabSize * d * 2; // fp16
    final headBytes = config.tieWeights ? 0 : config.vocabSize * d * 2;
    // One resident block: MHA fp16 + shared expert fp16 + router +
    // scratch for one transposed shared_gate/up/down (fp32
    // promotion at forward time).
    final residentBlockBytes =
        4 * d * d * 2 + // Q/K/V/O fp16
        3 * config.sharedHidden * d * 2 + // shared fp16
        3 * config.sharedHidden * d * 4; // shared transposes fp32
    // Per-forward transient: K experts × (fp16 read + fp32 transpose)
    // held briefly.
    final perLayerTransient = config.topK * 3 * config.moeHidden * d * (2 + 4);
    final peakEst =
        embedBytes +
        headBytes +
        residentBlockBytes +
        perLayerTransient +
        300 * 1024 * 1024; // Dart runtime + activations

    final free = _freeRamBytes();
    if (free == null) return;
    if (peakEst > free) {
      throw StateError(
        'Qwen15MoEStreamingRunner: predicted peak ~${_fmtBytes(peakEst)} '
        '(embed ${_fmtBytes(embedBytes)}'
        '${headBytes > 0 ? ' + lm_head ${_fmtBytes(headBytes)}' : ''} '
        '+ resident block ${_fmtBytes(residentBlockBytes)} '
        '+ per-forward transient ${_fmtBytes(perLayerTransient)}) '
        'exceeds free RAM ${_fmtBytes(free)}. Either shrink the '
        'preset, close other processes, or raise the WSL memory '
        'limit in %USERPROFILE%\\.wslconfig.',
      );
    }
  }

  static int? _freeRamBytes() {
    try {
      final txt = io.File('/proc/meminfo').readAsStringSync();
      for (final line in txt.split('\n')) {
        if (line.startsWith('MemAvailable:')) {
          final parts = line.split(RegExp(r'\s+'));
          return int.parse(parts[1]) * 1024;
        }
      }
    } catch (_) {}
    return null;
  }

  static String _fmtBytes(int b) {
    const units = ['B', 'KB', 'MB', 'GB'];
    var i = 0;
    double v = b.toDouble();
    while (v >= 1024 && i < units.length - 1) {
      v /= 1024;
      i++;
    }
    return '${v.toStringAsFixed(2)} ${units[i]}';
  }

  void _loadPersistent() {
    final d = config.dim;
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
      _headT = _embedWeight.transpose();
    } else {
      final head = _expectShape(
        reader.readTensor('lm_head.weight', keepFp16: keepFp16),
        [config.vocabSize, d],
        'lm_head.weight',
      );
      _headT = head.transpose(); // fp16-preserving; head then GC'd
    }
  }

  /// Swap layer [i]'s non-expert weights (MHA + norms + router +
  /// shared expert) into the resident block. Expert weights are
  /// streamed on-demand inside `forward` after routing.
  void _swapLayerNonExperts(int i) {
    final sw = profile ? (Stopwatch()..start()) : null;
    final cfg = config;
    final d = cfg.dim;
    final h = cfg.numHeads;
    final kvH = cfg.numKvHeads;
    final headDim = d ~/ h;
    final p = 'model.layers.$i';

    _copy(
      attnNorm.gamma,
      _expectShape(
        reader.readTensor('$p.input_layernorm.weight'),
        [d],
        '$p.input_layernorm.weight',
      ),
    );
    _copy(
      ffnNorm.gamma,
      _expectShape(
        reader.readTensor('$p.post_attention_layernorm.weight'),
        [d],
        '$p.post_attention_layernorm.weight',
      ),
    );

    // Q / K / V (weights + biases, Qwen has all three biased).
    final qW = _expectShape(
      reader.readTensor('$p.self_attn.q_proj.weight', keepFp16: keepFp16),
      [h * headDim, d],
      '$p.self_attn.q_proj.weight',
    );
    for (int hh = 0; hh < h; hh++) {
      _copy(attn.wq[hh].weight, qW.sliceRows(hh * headDim, (hh + 1) * headDim));
    }
    final kW = _expectShape(
      reader.readTensor('$p.self_attn.k_proj.weight', keepFp16: keepFp16),
      [kvH * headDim, d],
      '$p.self_attn.k_proj.weight',
    );
    for (int hh = 0; hh < kvH; hh++) {
      _copy(attn.wk[hh].weight, kW.sliceRows(hh * headDim, (hh + 1) * headDim));
    }
    final vW = _expectShape(
      reader.readTensor('$p.self_attn.v_proj.weight', keepFp16: keepFp16),
      [kvH * headDim, d],
      '$p.self_attn.v_proj.weight',
    );
    for (int hh = 0; hh < kvH; hh++) {
      _copy(attn.wv[hh].weight, vW.sliceRows(hh * headDim, (hh + 1) * headDim));
    }

    final qB = _expectShape(reader.readTensor('$p.self_attn.q_proj.bias'), [
      h * headDim,
    ], '$p.self_attn.q_proj.bias');
    for (int hh = 0; hh < h; hh++) {
      _copy(
        attn.wq[hh].bias!,
        _reshape1xN(_sliceVector(qB, hh * headDim, (hh + 1) * headDim)),
      );
    }
    final kB = _expectShape(reader.readTensor('$p.self_attn.k_proj.bias'), [
      kvH * headDim,
    ], '$p.self_attn.k_proj.bias');
    for (int hh = 0; hh < kvH; hh++) {
      _copy(
        attn.wk[hh].bias!,
        _reshape1xN(_sliceVector(kB, hh * headDim, (hh + 1) * headDim)),
      );
    }
    final vB = _expectShape(reader.readTensor('$p.self_attn.v_proj.bias'), [
      kvH * headDim,
    ], '$p.self_attn.v_proj.bias');
    for (int hh = 0; hh < kvH; hh++) {
      _copy(
        attn.wv[hh].bias!,
        _reshape1xN(_sliceVector(vB, hh * headDim, (hh + 1) * headDim)),
      );
    }

    _copy(
      attn.wo.weight,
      _expectShape(
        reader.readTensor('$p.self_attn.o_proj.weight', keepFp16: keepFp16),
        [d, d],
        '$p.self_attn.o_proj.weight',
      ),
    );

    // Router weight: HF ships [E, D], we want [D, E] for x @ router.
    // .transpose() promotes to fp32; router is small (~240 KB fp32).
    _routerW = reader
        .readTensor('$p.mlp.gate.weight', keepFp16: false)
        .transpose();

    _sharedGate = _expectShape(
      reader.readTensor(
        '$p.mlp.shared_expert.gate_proj.weight',
        keepFp16: keepFp16,
      ),
      [cfg.sharedHidden, d],
      '$p.mlp.shared_expert.gate_proj.weight',
    );
    _sharedUp = _expectShape(
      reader.readTensor(
        '$p.mlp.shared_expert.up_proj.weight',
        keepFp16: keepFp16,
      ),
      [cfg.sharedHidden, d],
      '$p.mlp.shared_expert.up_proj.weight',
    );
    _sharedDown = _expectShape(
      reader.readTensor(
        '$p.mlp.shared_expert.down_proj.weight',
        keepFp16: keepFp16,
      ),
      [d, cfg.sharedHidden],
      '$p.mlp.shared_expert.down_proj.weight',
    );
    _sharedExpertGate = _expectShape(
      reader.readTensor('$p.mlp.shared_expert_gate.weight', keepFp16: false),
      [1, d],
      '$p.mlp.shared_expert_gate.weight',
    );

    // Cache pre-transposed shared expert weights so per-token
    // forward doesn't allocate 3 × 22 MB fp16 transposes per layer.
    // Transpose preserves fp16 storage — same bytes, rearranged.
    _sharedGateT = _sharedGate.transpose();
    _sharedUpT = _sharedUp.transpose();
    _sharedDownT = _sharedDown.transpose();
    _sharedExpertGateT = _sharedExpertGate.transpose();

    if (sw != null) {
      sw.stop();
      // ignore: avoid_print
      print('  [layer $i] non-expert swap ${sw.elapsedMilliseconds} ms');
    }
  }

  /// Run one layer's forward. Assumes non-expert weights are already
  /// resident (via `_swapLayerNonExperts(i)`).
  Tensor _forwardLayer(
    int layerIdx,
    Tensor x, {
    required Tensor? mask,
    MHACache? cache,
    required int startPos,
  }) {
    // Self-attention half.
    final h = attn(attnNorm(x), mask: mask, cache: cache, startPos: startPos);
    final x2 = x + h;

    // MoE FFN half.
    final hFfn = ffnNorm(x2);

    final moeStreaming = MoeStreamingLayer(
      prefix: 'model.layers.$layerIdx.mlp',
      numExperts: config.numExperts,
      topK: config.topK,
      reader: reader,
      gate: GateFunction.softmax, // Qwen2-MoE
    );
    final decision = moeStreaming.route(_routerW, hFfn);

    // Shared expert path (fused SwiGLU + scalar sigmoid gate). Uses
    // the pre-transposed weights cached at swap time — saves 3 × 22
    // MB fp16 transpose allocations per layer per token.
    final gate = hFfn.matmul(_sharedGateT);
    final up = hFfn.matmul(_sharedUpT);
    final act = gate * gate.sigmoid();
    final sharedOut = (act * up).matmul(_sharedDownT);

    final sharedGateLogit = hFfn.matmul(_sharedExpertGateT);
    final sharedGateScore = sharedGateLogit.sigmoid();
    final sharedGateBcast = sharedGateScore.matmul(_onesD);
    var acc = sharedOut * sharedGateBcast;

    // Routed experts — stream only the union of top-K.
    for (final j in decision.sortedUnion) {
      final w = moeStreaming.loadRoutedExpert(j, keepFp16: keepFp16);
      final expertOut = swiGluForward(hFfn, w);
      final wj = Tensor.fromList([
        hFfn.shape[0],
        1,
      ], decision.weightForExpert(j));
      final wjBcast = wj.matmul(_onesD);
      acc = acc + (expertOut * wjBcast);
    }

    return x2 + acc;
  }

  /// Full forward through all N layers. `tokens` is `[seqLen]`
  /// float32 token ids. Returns `[seqLen, vocab]` logits.
  Tensor forward(Tensor tokens, {int startPos = 0, EncoderCache? cache}) {
    if (tokens.shape.length != 1) {
      throw ArgumentError(
        'Qwen15MoEStreamingRunner: tokens must be 1D [seqLen]; '
        'got ${tokens.shape}',
      );
    }
    final n = tokens.shape.last;
    if (startPos + n > config.maxCtx) {
      throw ArgumentError(
        'Qwen15MoEStreamingRunner: window [$startPos, ${startPos + n}) '
        'exceeds maxCtx ${config.maxCtx}',
      );
    }
    return Tensor.noGrad(() {
      var x = _embedWeight.embedding(tokens);
      final mask = n > 1 ? causalMask(n, device: x.device) : null;
      for (int i = 0; i < config.numLayers; i++) {
        _swapLayerNonExperts(i);
        x = _forwardLayer(
          i,
          x,
          mask: mask,
          cache: cache?.layers[i],
          startPos: startPos,
        );
      }
      x = finalNorm(x);
      return x.matmul(_headT);
    });
  }

  /// Greedy autoregressive decode with persistent per-layer KV cache.
  List<double> generate(List<double> prompt, {required int maxNewTokens}) {
    if (prompt.isEmpty) {
      throw ArgumentError('generate: prompt must be non-empty');
    }
    final v = config.vocabSize;
    final out = List<double>.of(prompt);
    final cache = EncoderCache.empty(config.numLayers, config.numKvHeads);

    final promptT = Tensor.fromList([prompt.length], prompt);
    var logits = forward(promptT, startPos: 0, cache: cache).toList();
    out.add(_argmax(logits, (prompt.length - 1) * v, v).toDouble());

    for (int step = 1; step < maxNewTokens; step++) {
      if (cache.seqLen >= config.maxCtx) break;
      final oneT = Tensor.fromList([1], [out.last]);
      logits = forward(oneT, startPos: cache.seqLen, cache: cache).toList();
      out.add(_argmax(logits, 0, v).toDouble());
    }
    return out;
  }

  void close() => reader.close();

  // ---------------- helpers ----------------

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

  static Tensor _expectShape(Tensor t, List<int> expected, String name) {
    if (t.shape.length != expected.length) {
      throw ArgumentError(
        'qwen15-moe streaming: "$name" expected shape $expected, got '
        '${t.shape}',
      );
    }
    for (int i = 0; i < expected.length; i++) {
      if (t.shape[i] != expected[i]) {
        throw ArgumentError(
          'qwen15-moe streaming: "$name" expected shape $expected, got '
          '${t.shape}',
        );
      }
    }
    return t;
  }

  static void _copy(Tensor dst, Tensor src) {
    if (dst.length != src.length) {
      throw ArgumentError(
        'qwen15-moe streaming: copy length mismatch dst=${dst.shape} '
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

  static Tensor _sliceVector(Tensor t, int start, int end) {
    final src = t.toList();
    final n = end - start;
    final out = Float32List(n);
    for (int i = 0; i < n; i++) {
      out[i] = src[start + i];
    }
    return Tensor.fromList([n], out);
  }

  static Tensor _reshape1xN(Tensor v) {
    if (v.shape.length != 1) {
      throw ArgumentError('_reshape1xN: expected rank 1, got ${v.shape}');
    }
    return Tensor.fromList([1, v.shape[0]], v.toList());
  }
}
