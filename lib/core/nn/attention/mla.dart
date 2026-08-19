/// Multi-Head Latent Attention (MLA) — DeepSeek-V2 (Liu et al., 2024).
///
/// Standard MHA carries a `[seq, numHeads · headDim]` KV cache, which
/// dominates memory for long-context inference. MLA compresses K and V
/// into a **shared low-rank latent** `c_kv` of dim `kvLoraRank` (typically
/// 512 for `d_model = 5120`), so the cache shrinks by an
/// `H · d_head / kvLoraRank` factor — roughly 32× for DeepSeek-V2's
/// 128-head config. Q is likewise compressed to `qLoraRank` (~1536)
/// through a low-rank bottleneck.
///
/// To preserve rotary positional information under compression, MLA
/// **decouples** the RoPE component of Q/K from the content component:
///
///   q_h = concat(q_h_nope, q_h_rope)   per head
///   k_h = concat(k_h_nope, k_rope_shared)   shared k_rope across heads
///
/// `q_h_rope` and `k_rope_shared` are RoPE-rotated; `q_h_nope`,
/// `k_h_nope`, and `v_h` come from the compressed latents. Attention
/// is standard scaled dot-product on the concatenated `[nope | rope]`
/// vectors with scale `1 / √(qkNopeHeadDim + qkRopeHeadDim)`.
///
/// This implementation is a **straightforward** forward pass (no cache
/// yet, no absorption trick — DeepSeek's paper describes a matmul
/// "absorption" that folds `W_uk` into `W_q` at inference for further
/// speedup; that's a follow-on optimization). Weights are laid out to
/// match HuggingFace `deepseek-ai/DeepSeek-V2-Lite` and larger.
///
/// Single-sequence 2D convention (`[N, embedDim]`) matching
/// [MultiHeadAttention]. Batched can be added later.
library;

import '../../tensor/tensor.dart';
import '../linear.dart';
import '../module.dart';
import '../rms_norm.dart';
import '../rotary.dart';

class MLAConfig {
  final int embedDim;
  final int numHeads;
  /// When non-null, Q is passed through a `Linear(embedDim, qLoraRank)`
  /// + `RMSNorm(qLoraRank)` bottleneck before the per-head Q up-
  /// projections. When null (DeepSeek-V2-Lite), Q goes directly from
  /// `x` to the per-head projections at full embed dim — no bottleneck.
  final int? qLoraRank;
  final int kvLoraRank;
  final int qkNopeHeadDim;
  final int qkRopeHeadDim;
  final int vHeadDim;
  final double rmsNormEps;

  const MLAConfig({
    required this.embedDim,
    required this.numHeads,
    required this.qLoraRank,
    required this.kvLoraRank,
    required this.qkNopeHeadDim,
    required this.qkRopeHeadDim,
    required this.vHeadDim,
    this.rmsNormEps = 1e-6,
  });

  /// Combined Q/K head dim used inside the softmax (`√(nope + rope)`).
  int get qkHeadDim => qkNopeHeadDim + qkRopeHeadDim;

  /// Input dim for the per-head Q up-projections. `qLoraRank` when
  /// Q is compressed; `embedDim` when it isn't (DeepSeek-V2-Lite).
  int get qInDim => qLoraRank ?? embedDim;

  /// DeepSeek-V2 full (~236B, uses Q compression at `qLoraRank=1536`)
  /// attention shape numbers.
  static MLAConfig deepseekV2Config() => const MLAConfig(
    embedDim: 5120,
    numHeads: 128,
    qLoraRank: 1536,
    kvLoraRank: 512,
    qkNopeHeadDim: 128,
    qkRopeHeadDim: 64,
    vHeadDim: 128,
    rmsNormEps: 1e-6,
  );

  /// DeepSeek-V2-Lite (~16B, 27 layers, hidden 2048) attention config.
  /// **No Q compression** (`qLoraRank == null`) — matches HF
  /// `deepseek-ai/DeepSeek-V2-Lite` config.json exactly.
  static MLAConfig deepseekV2LiteConfig() => const MLAConfig(
    embedDim: 2048,
    numHeads: 16,
    qLoraRank: null,
    kvLoraRank: 512,
    qkNopeHeadDim: 128,
    qkRopeHeadDim: 64,
    vHeadDim: 128,
    rmsNormEps: 1e-6,
  );
}

class MultiHeadLatentAttention extends Module {
  final MLAConfig config;

  /// `x -> c_q` low-rank Q projection. **Null when Q is not compressed**
  /// (DeepSeek-V2-Lite / `MLAConfig.qLoraRank == null`).
  final Linear? qDown;
  final RMSNorm? qLn;

  /// Per-head "no-rope" and "rope" up-projections. Input dim is
  /// `config.qInDim` (`qLoraRank` when compressed, `embedDim` otherwise).
  final List<Linear> qUpNope;
  final List<Linear> qUpRope;

  /// `x -> c_kv` low-rank KV projection.
  final Linear kvDown;
  final RMSNorm kvLn;

  /// `x -> k_rope_shared` — the ONE shared rope key across all heads.
  final Linear kRope;

  /// Per-head "no-rope" K and V up-projections from `c_kv`.
  final List<Linear> kUpNope; // c_kv -> [N, qkNopeHeadDim] per head
  final List<Linear> vUp; // c_kv -> [N, vHeadDim] per head

  /// Output projection `[embedDim, numHeads * vHeadDim]`.
  final Linear oProj;

  /// Optional shared RoPE cache. Must be created with
  /// `headDim == qkRopeHeadDim` (rotary is applied to the rope part
  /// only, not to nope).
  RopeCache? rope;

  MultiHeadLatentAttention(
    this.config, {
    Device device = Device.CPU,
    int seed = 0,
  })  : qDown = config.qLoraRank == null
            ? null
            : Linear(
                config.embedDim,
                config.qLoraRank!,
                bias: false,
                device: device,
                seed: seed,
              ),
        qLn = config.qLoraRank == null
            ? null
            : RMSNorm(
                config.qLoraRank!,
                eps: config.rmsNormEps,
                device: device,
              ),
        qUpNope = List<Linear>.generate(
          config.numHeads,
          (h) => Linear(
            config.qLoraRank ?? config.embedDim,
            config.qkNopeHeadDim,
            bias: false,
            device: device,
            seed: seed + 100_000 + h,
          ),
        ),
        qUpRope = List<Linear>.generate(
          config.numHeads,
          (h) => Linear(
            config.qLoraRank ?? config.embedDim,
            config.qkRopeHeadDim,
            bias: false,
            device: device,
            seed: seed + 200_000 + h,
          ),
        ),
        kvDown = Linear(
          config.embedDim,
          config.kvLoraRank,
          bias: false,
          device: device,
          seed: seed + 300_000,
        ),
        kvLn = RMSNorm(
          config.kvLoraRank,
          eps: config.rmsNormEps,
          device: device,
        ),
        kRope = Linear(
          config.embedDim,
          config.qkRopeHeadDim,
          bias: false,
          device: device,
          seed: seed + 400_000,
        ),
        kUpNope = List<Linear>.generate(
          config.numHeads,
          (h) => Linear(
            config.kvLoraRank,
            config.qkNopeHeadDim,
            bias: false,
            device: device,
            seed: seed + 500_000 + h,
          ),
        ),
        vUp = List<Linear>.generate(
          config.numHeads,
          (h) => Linear(
            config.kvLoraRank,
            config.vHeadDim,
            bias: false,
            device: device,
            seed: seed + 600_000 + h,
          ),
        ),
        oProj = Linear(
          config.numHeads * config.vHeadDim,
          config.embedDim,
          bias: false,
          device: device,
          seed: seed + 700_000,
        );

  /// Forward pass. `x` is `[N, embedDim]`. Optional additive attention
  /// `mask` is broadcast to `[N, N]` (typical causal mask).
  ///
  /// `startPos` is the absolute position of `x[0]` — matters for
  /// RoPE (KV-cache generation would append single tokens at a
  /// growing `startPos`). No cache yet in this port.
  Tensor call(Tensor x, {Tensor? mask, int startPos = 0}) {
    if (x.shape.length != 2 || x.shape[1] != config.embedDim) {
      throw ArgumentError(
        'MLA: expected [N, ${config.embedDim}]; got ${x.shape}',
      );
    }
    if (rope != null && rope!.headDim != config.qkRopeHeadDim) {
      throw StateError(
        'MLA: attached RopeCache.headDim (${rope!.headDim}) must equal '
        'qkRopeHeadDim (${config.qkRopeHeadDim})',
      );
    }

    // ---- Q path ----
    // With compression: x -> qDown -> qLn -> cQ.
    // Without (Lite):    cQ = x directly (per-head Linears take embedDim).
    final cQ = qDown == null ? x : qLn!(qDown!(x));

    // ---- KV path ----
    final cKv = kvLn(kvDown(x));
    var kRopeShared = kRope(x);
    if (rope != null) {
      kRopeShared = rope!.apply(kRopeShared, startPos: startPos);
    }

    // ---- Per-head SDPA ----
    final headOuts = <Tensor>[];
    for (int h = 0; h < config.numHeads; h++) {
      final qNope = qUpNope[h](cQ);
      var qR = qUpRope[h](cQ);
      if (rope != null) {
        qR = rope!.apply(qR, startPos: startPos);
      }
      final qFull = TensorConcat.concat([qNope, qR], axis: 1);

      final kNope = kUpNope[h](cKv);
      final kFull = TensorConcat.concat([kNope, kRopeShared], axis: 1);

      final vH = vUp[h](cKv);

      headOuts.add(qFull.scaledDotProductAttention(kFull, vH, mask: mask));
    }
    final concat = TensorConcat.concat(headOuts, axis: 1);
    return oProj(concat);
  }

  @override
  List<Tensor> parameters() => [
    if (qDown != null) ...qDown!.parameters(),
    if (qLn != null) ...qLn!.parameters(),
    for (final l in qUpNope) ...l.parameters(),
    for (final l in qUpRope) ...l.parameters(),
    ...kvDown.parameters(),
    ...kvLn.parameters(),
    ...kRope.parameters(),
    for (final l in kUpNope) ...l.parameters(),
    for (final l in vUp) ...l.parameters(),
    ...oProj.parameters(),
  ];

  @override
  List<Module> submodules() => [
    if (qDown != null) qDown!,
    if (qLn != null) qLn!,
    ...qUpNope,
    ...qUpRope,
    kvDown,
    kvLn,
    kRope,
    ...kUpNope,
    ...vUp,
    oProj,
  ];
}
