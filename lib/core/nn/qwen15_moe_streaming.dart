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

  /// Per-stage timing accumulator. Only populated when [profile] is
  /// true; reset at the start of every [forward].
  final _StageProfile _prof = _StageProfile();

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

  /// Pre-transposed shared expert weights, computed at swap time so
  /// `forward` doesn't allocate 3 × 22 MB fp16 transposes per layer
  /// per token. `_sharedGateT` and `_sharedUpT` are `[D, hidden]`,
  /// `_sharedDownT` is `[hidden, D]`. Reassigned per swap from the
  /// `_sharedGateTs` / `_sharedUpTs` / `_sharedDownTs` caches.
  Tensor _sharedGateT;
  Tensor _sharedUpT;
  Tensor _sharedDownT;

  /// Pre-transposed `[D, 1]` version of the scalar shared_expert_gate.
  Tensor _sharedExpertGateT;

  /// Persistent per-layer caches of the transposed shared expert and
  /// router weights. Populated once at construction — after that,
  /// `_swapLayerNonExperts` skips disk I/O for these and just points
  /// at the right layer's cached tensor. Costs ~1.6 GB extra fp16
  /// storage (24 × 66 MB shared + 24 × 240 KB router) to save
  /// ~5 seconds per forward on real Qwen1.5-MoE weights.
  late final List<Tensor> _sharedGateTs;
  late final List<Tensor> _sharedUpTs;
  late final List<Tensor> _sharedDownTs;
  late final List<Tensor> _sharedExpertGateTs;
  late final List<Tensor> _routerWs;
  late final List<Tensor> _attnNormGammas;
  late final List<Tensor> _ffnNormGammas;

  /// Persistent per-layer MHA weight caches. Each entry is the full
  /// `[H*headDim, D]` fp16 blob (biases as `[H*headDim]` fp32). Per
  /// swap, sliceRows produces fresh per-head views (native memmove);
  /// no disk I/O. Costs ~24 x 32 MB fp16 = 768 MB persistent for
  /// Qwen1.5-MoE's Q/K/V/O with biases.
  late final List<Tensor> _qWs;
  late final List<Tensor> _kWs;
  late final List<Tensor> _vWs;
  late final List<Tensor> _oWs;
  late final List<Tensor> _qBs;
  late final List<Tensor> _kBs;
  late final List<Tensor> _vBs;

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
       _sharedGateT = Tensor.fill([1, 1], 0.0),
       _sharedUpT = Tensor.fill([1, 1], 0.0),
       _sharedDownT = Tensor.fill([1, 1], 0.0),
       _sharedExpertGateT = Tensor.fill([1, 1], 0.0) {
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

    // Pre-load all layers' router + shared expert transposes + norms.
    // Big memory win but eliminates ~66 MB of disk I/O per layer per
    // forward. ~1.6 GB extra fp16 storage for the shared expert alone.
    _sharedGateTs = List<Tensor>.filled(config.numLayers, _sharedGateT);
    _sharedUpTs = List<Tensor>.filled(config.numLayers, _sharedUpT);
    _sharedDownTs = List<Tensor>.filled(config.numLayers, _sharedDownT);
    _sharedExpertGateTs =
        List<Tensor>.filled(config.numLayers, _sharedExpertGateT);
    _routerWs = List<Tensor>.filled(config.numLayers, _routerW);
    _attnNormGammas = List<Tensor>.filled(config.numLayers, attnNorm.gamma);
    _ffnNormGammas = List<Tensor>.filled(config.numLayers, ffnNorm.gamma);
    for (int i = 0; i < config.numLayers; i++) {
      final p = 'model.layers.$i.mlp';
      final sG = _expectShape(
        reader.readTensor('$p.shared_expert.gate_proj.weight',
            keepFp16: keepFp16),
        [config.sharedHidden, d],
        '$p.shared_expert.gate_proj.weight',
      );
      final sU = _expectShape(
        reader.readTensor('$p.shared_expert.up_proj.weight',
            keepFp16: keepFp16),
        [config.sharedHidden, d],
        '$p.shared_expert.up_proj.weight',
      );
      final sD = _expectShape(
        reader.readTensor('$p.shared_expert.down_proj.weight',
            keepFp16: keepFp16),
        [d, config.sharedHidden],
        '$p.shared_expert.down_proj.weight',
      );
      final sEG = _expectShape(
        reader.readTensor('$p.shared_expert_gate.weight', keepFp16: false),
        [1, d],
        '$p.shared_expert_gate.weight',
      );
      _sharedGateTs[i] = sG.transpose();
      _sharedUpTs[i] = sU.transpose();
      _sharedDownTs[i] = sD.transpose();
      _sharedExpertGateTs[i] = sEG.transpose();
      _routerWs[i] = reader
          .readTensor('model.layers.$i.mlp.gate.weight', keepFp16: false)
          .transpose();
      // Norms are tiny [D] fp32 gammas — just cache the tensor ref.
      _attnNormGammas[i] = _expectShape(
        reader.readTensor('model.layers.$i.input_layernorm.weight'),
        [d],
        'model.layers.$i.input_layernorm.weight',
      );
      _ffnNormGammas[i] = _expectShape(
        reader.readTensor('model.layers.$i.post_attention_layernorm.weight'),
        [d],
        'model.layers.$i.post_attention_layernorm.weight',
      );
    }

    // Cache MHA weights per layer. Bulk fp16 blobs stay on the runner;
    // per-swap `sliceRows` creates fresh per-head views via native
    // memmove and the resident MHA adopts them.
    final h = config.numHeads;
    final kvH = config.numKvHeads;
    final headDim = d ~/ h;
    _qWs = List<Tensor>.filled(config.numLayers, Tensor.fill([1, 1], 0.0));
    _kWs = List<Tensor>.filled(config.numLayers, Tensor.fill([1, 1], 0.0));
    _vWs = List<Tensor>.filled(config.numLayers, Tensor.fill([1, 1], 0.0));
    _oWs = List<Tensor>.filled(config.numLayers, Tensor.fill([1, 1], 0.0));
    _qBs = List<Tensor>.filled(config.numLayers, Tensor.fill([1], 0.0));
    _kBs = List<Tensor>.filled(config.numLayers, Tensor.fill([1], 0.0));
    _vBs = List<Tensor>.filled(config.numLayers, Tensor.fill([1], 0.0));
    for (int i = 0; i < config.numLayers; i++) {
      final p = 'model.layers.$i.self_attn';
      _qWs[i] = _expectShape(
        reader.readTensor('$p.q_proj.weight', keepFp16: keepFp16),
        [h * headDim, d],
        '$p.q_proj.weight',
      );
      _kWs[i] = _expectShape(
        reader.readTensor('$p.k_proj.weight', keepFp16: keepFp16),
        [kvH * headDim, d],
        '$p.k_proj.weight',
      );
      _vWs[i] = _expectShape(
        reader.readTensor('$p.v_proj.weight', keepFp16: keepFp16),
        [kvH * headDim, d],
        '$p.v_proj.weight',
      );
      _oWs[i] = _expectShape(
        reader.readTensor('$p.o_proj.weight', keepFp16: keepFp16),
        [d, d],
        '$p.o_proj.weight',
      );
      _qBs[i] = _expectShape(
        reader.readTensor('$p.q_proj.bias'),
        [h * headDim],
        '$p.q_proj.bias',
      );
      _kBs[i] = _expectShape(
        reader.readTensor('$p.k_proj.bias'),
        [kvH * headDim],
        '$p.k_proj.bias',
      );
      _vBs[i] = _expectShape(
        reader.readTensor('$p.v_proj.bias'),
        [kvH * headDim],
        '$p.v_proj.bias',
      );
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

    // All non-expert weights come from per-layer caches populated at
    // construction. No disk I/O in this method.
    _copy(attnNorm.gamma, _attnNormGammas[i]);
    _copy(ffnNorm.gamma, _ffnNormGammas[i]);
    _routerW = _routerWs[i];
    _sharedGateT = _sharedGateTs[i];
    _sharedUpT = _sharedUpTs[i];
    _sharedDownT = _sharedDownTs[i];
    _sharedExpertGateT = _sharedExpertGateTs[i];

    // Q / K / V per-head from cached full weight blobs. sliceRows
    // creates a fresh fp16 buffer (native memmove) so the cache
    // survives the adopt.
    final qW = _qWs[i];
    for (int hh = 0; hh < h; hh++) {
      _copy(attn.wq[hh].weight, qW.sliceRows(hh * headDim, (hh + 1) * headDim));
    }
    final kW = _kWs[i];
    for (int hh = 0; hh < kvH; hh++) {
      _copy(attn.wk[hh].weight, kW.sliceRows(hh * headDim, (hh + 1) * headDim));
    }
    final vW = _vWs[i];
    for (int hh = 0; hh < kvH; hh++) {
      _copy(attn.wv[hh].weight, vW.sliceRows(hh * headDim, (hh + 1) * headDim));
    }

    final qB = _qBs[i];
    for (int hh = 0; hh < h; hh++) {
      _copy(
        attn.wq[hh].bias!,
        _reshape1xN(_sliceVector(qB, hh * headDim, (hh + 1) * headDim)),
      );
    }
    final kB = _kBs[i];
    for (int hh = 0; hh < kvH; hh++) {
      _copy(
        attn.wk[hh].bias!,
        _reshape1xN(_sliceVector(kB, hh * headDim, (hh + 1) * headDim)),
      );
    }
    final vB = _vBs[i];
    for (int hh = 0; hh < kvH; hh++) {
      _copy(
        attn.wv[hh].bias!,
        _reshape1xN(_sliceVector(vB, hh * headDim, (hh + 1) * headDim)),
      );
    }

    _copy(attn.wo.weight, _oWs[i].sliceRows(0, _oWs[i].shape[0]));

    if (sw != null) {
      sw.stop();
      _prof.swap += sw.elapsedMicroseconds;
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
    final sw = profile ? Stopwatch() : null;
    // Self-attention half.
    sw?.reset();
    sw?.start();
    final h = attn(attnNorm(x), mask: mask, cache: cache, startPos: startPos);
    final x2 = x + h;
    if (sw != null) {
      sw.stop();
      _prof.mha += sw.elapsedMicroseconds;
    }

    // MoE FFN half.
    sw?.reset();
    sw?.start();
    final hFfn = ffnNorm(x2);
    final moeStreaming = MoeStreamingLayer(
      prefix: 'model.layers.$layerIdx.mlp',
      numExperts: config.numExperts,
      topK: config.topK,
      reader: reader,
      gate: GateFunction.softmax, // Qwen2-MoE
    );
    final decision = moeStreaming.route(_routerW, hFfn);
    if (sw != null) {
      sw.stop();
      _prof.router += sw.elapsedMicroseconds;
    }

    // Shared expert path (fused SwiGLU + scalar sigmoid gate). Uses
    // the pre-transposed weights cached at swap time — saves 3 × 22
    // MB fp16 transpose allocations per layer per token.
    sw?.reset();
    sw?.start();
    final gate = hFfn.matmul(_sharedGateT);
    final up = hFfn.matmul(_sharedUpT);
    final act = gate * gate.sigmoid();
    final sharedOut = (act * up).matmul(_sharedDownT);
    final sharedGateLogit = hFfn.matmul(_sharedExpertGateT);
    final sharedGateScore = sharedGateLogit.sigmoid();
    final sharedGateBcast = sharedGateScore.matmul(_onesD);
    var acc = sharedOut * sharedGateBcast;
    if (sw != null) {
      sw.stop();
      _prof.sharedExpert += sw.elapsedMicroseconds;
    }

    // Routed experts — stream only the union of top-K.
    for (final j in decision.sortedUnion) {
      sw?.reset();
      sw?.start();
      final w = moeStreaming.loadRoutedExpert(j, keepFp16: keepFp16);
      if (sw != null) {
        sw.stop();
        _prof.expertStream += sw.elapsedMicroseconds;
      }
      sw?.reset();
      sw?.start();
      final expertOut = swiGluForward(hFfn, w);
      final wj = Tensor.fromList([
        hFfn.shape[0],
        1,
      ], decision.weightForExpert(j));
      final wjBcast = wj.matmul(_onesD);
      acc = acc + (expertOut * wjBcast);
      if (sw != null) {
        sw.stop();
        _prof.expertCompute += sw.elapsedMicroseconds;
      }
    }
    _prof.expertsCounted += decision.sortedUnion.length;

    return x2 + acc;
  }

  /// Full forward through all N layers. `tokens` is `[seqLen]`
  /// float32 token ids. Returns `[seqLen, vocab]` logits — or, when
  /// [lastRowOnly] is true, just the last row `[1, vocab]`. Slicing
  /// off the prefix before the final head projection saves
  /// `(seqLen - 1) × D × vocab` matmul ops per forward. Only safe
  /// when the caller uses just the last-position logits (which is
  /// what [generate] does for prompt-fill).
  Tensor forward(
    Tensor tokens, {
    int startPos = 0,
    EncoderCache? cache,
    bool lastRowOnly = false,
  }) {
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
      final sw = profile ? Stopwatch() : null;
      _prof.reset();
      sw?.reset();
      sw?.start();
      var x = _embedWeight.embedding(tokens);
      final mask = n > 1 ? causalMask(n, device: x.device) : null;
      if (sw != null) {
        sw.stop();
        _prof.embed += sw.elapsedMicroseconds;
      }
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
      sw?.reset();
      sw?.start();
      x = finalNorm(x);
      if (lastRowOnly && n > 1) {
        x = x.sliceRows(n - 1, n);
      }
      final logits = x.matmul(_headT);
      if (sw != null) {
        sw.stop();
        _prof.head += sw.elapsedMicroseconds;
        _prof.report(config.numLayers);
      }
      return logits;
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
    // Prompt fill only needs the last position's logits for argmax.
    var logits = forward(
      promptT,
      startPos: 0,
      cache: cache,
      lastRowOnly: true,
    ).toList();
    out.add(_argmax(logits, 0, v).toDouble());

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

/// Per-forward stage timing accumulator. All fields are microseconds
/// summed across every layer / expert of a single `forward()` call.
class _StageProfile {
  int embed = 0;
  int swap = 0;
  int mha = 0;
  int router = 0;
  int sharedExpert = 0;
  int expertStream = 0;
  int expertCompute = 0;
  int head = 0;
  int expertsCounted = 0;

  void reset() {
    embed = 0;
    swap = 0;
    mha = 0;
    router = 0;
    sharedExpert = 0;
    expertStream = 0;
    expertCompute = 0;
    head = 0;
    expertsCounted = 0;
  }

  void report(int numLayers) {
    final total =
        embed +
        swap +
        mha +
        router +
        sharedExpert +
        expertStream +
        expertCompute +
        head;
    if (total == 0) return;
    String row(String name, int us) {
      final ms = us / 1000.0;
      final pct = 100 * us / total;
      return '  ${name.padRight(18)} ${ms.toStringAsFixed(0).padLeft(6)} ms  '
          '(${pct.toStringAsFixed(1).padLeft(4)} %)';
    }

    // ignore: avoid_print
    print('  ---- forward profile (numLayers=$numLayers) ----');
    // ignore: avoid_print
    print(row('embed', embed));
    // ignore: avoid_print
    print(row('layer swaps', swap));
    // ignore: avoid_print
    print(row('MHA total', mha));
    // ignore: avoid_print
    print(row('router+norms', router));
    // ignore: avoid_print
    print(row('shared expert', sharedExpert));
    // ignore: avoid_print
    print(row('expert stream I/O', expertStream));
    // ignore: avoid_print
    print(row('expert compute', expertCompute));
    // ignore: avoid_print
    print(row('head projection', head));
    // ignore: avoid_print
    print('  ---- $expertsCounted routed experts fired total ----');
  }
}
