/// DeepSeek-V2 causal language model — MLA + (dense FFN | MoE-64) blocks.
///
/// Implements the `DeepseekV2ForCausalLM` architecture from
/// `deepseek-ai/DeepSeek-V2` and `deepseek-ai/DeepSeek-V2-Lite`
/// (Liu et al. 2024). Each block is:
///
///   h = x + MLA( RMSNorm(x) )
///   y = h + FFN( RMSNorm(h) )       where FFN is:
///                                    * `SwiGluFfn` for layer i < firstKDenseReplace
///                                    * `MoEFeedForward` for i >= firstKDenseReplace
///
/// After the block stack a final `RMSNorm` + `lm_head` produces
/// `[N, vocabSize]` logits. Standard pre-LN + weight-tied embedding
/// setup (Lite has `tie_word_embeddings == false` — verify per model).
///
/// Weights are randomly initialised; a follow-up loader will bind
/// them to HF safetensors. Presets:
///
///   * [DeepSeekV2Config.lite] — `deepseek-ai/DeepSeek-V2-Lite`
///     (~16 B total, 2.4 B active per token, 27 layers, hidden 2048,
///     16 heads, KV-lora 512, **no Q compression**, dense FFN in
///     layer 0, MoE (64 routed + 2 shared, top-6) in layers 1..26).
///   * [DeepSeekV2Config.full] — `deepseek-ai/DeepSeek-V2` (~236 B
///     total, 21 B active, 60 layers, hidden 5120, 128 heads,
///     Q-lora 1536, KV-lora 512, dense FFN layers 0..0, MoE
///     (160 routed + 2 shared, top-6, 8 groups, group-top-3) 1..59).
library;

import '../tensor/tensor.dart';
import 'attention/mla.dart';
import 'embedding.dart';
import 'ffn/swiglu.dart';
import 'linear.dart';
import 'masks.dart';
import 'module.dart';
import 'moe.dart';
import 'rms_norm.dart';
import 'rotary.dart';

class DeepSeekV2Config {
  final int vocabSize;
  final int maxCtx;
  final int embedDim;
  final int numLayers;

  /// First `firstKDenseReplace` layers use a dense SwiGLU FFN with
  /// `denseFfnDim` intermediate; the remaining layers use MoE.
  final int firstKDenseReplace;
  final int denseFfnDim;

  final int moeExpertHiddenDim;
  final int numRoutedExperts;
  final int numSharedExperts;
  final int numExpertsPerTok;
  final int numExpertGroups;
  final int topKGroups;

  final MLAConfig mlaConfig;
  final double rmsNormEps;
  final double ropeBase;
  final bool tieWordEmbeddings;
  final Device device;
  final int seed;

  const DeepSeekV2Config({
    required this.vocabSize,
    required this.maxCtx,
    required this.embedDim,
    required this.numLayers,
    required this.firstKDenseReplace,
    required this.denseFfnDim,
    required this.moeExpertHiddenDim,
    required this.numRoutedExperts,
    required this.numSharedExperts,
    required this.numExpertsPerTok,
    required this.mlaConfig,
    this.numExpertGroups = 1,
    this.topKGroups = 1,
    this.rmsNormEps = 1e-6,
    this.ropeBase = 10000.0,
    this.tieWordEmbeddings = false,
    this.device = Device.CPU,
    this.seed = 0,
  });

  /// `deepseek-ai/DeepSeek-V2-Lite` config (16 B total / 2.4 B active).
  static DeepSeekV2Config lite({Device device = Device.CPU, int seed = 0}) =>
      DeepSeekV2Config(
        vocabSize: 102400,
        maxCtx: 163840,
        embedDim: 2048,
        numLayers: 27,
        firstKDenseReplace: 1,
        denseFfnDim: 10944,
        moeExpertHiddenDim: 1408,
        numRoutedExperts: 64,
        numSharedExperts: 2,
        numExpertsPerTok: 6,
        numExpertGroups: 1,
        topKGroups: 1,
        mlaConfig: MLAConfig.deepseekV2LiteConfig(),
        rmsNormEps: 1e-6,
        ropeBase: 10000.0,
        tieWordEmbeddings: false,
        device: device,
        seed: seed,
      );

  /// `deepseek-ai/DeepSeek-V2` full config (236 B total / 21 B active).
  static DeepSeekV2Config full({Device device = Device.CPU, int seed = 0}) =>
      DeepSeekV2Config(
        vocabSize: 102400,
        maxCtx: 163840,
        embedDim: 5120,
        numLayers: 60,
        firstKDenseReplace: 1,
        denseFfnDim: 12288,
        moeExpertHiddenDim: 1536,
        numRoutedExperts: 160,
        numSharedExperts: 2,
        numExpertsPerTok: 6,
        numExpertGroups: 8,
        topKGroups: 3,
        mlaConfig: MLAConfig.deepseekV2Config(),
        rmsNormEps: 1e-6,
        ropeBase: 10000.0,
        tieWordEmbeddings: false,
        device: device,
        seed: seed,
      );
}

class DeepSeekV2Block extends Module {
  final int layerIndex;
  final bool isMoE;

  final RMSNorm attnLn; // pre-attn RMSNorm
  final MultiHeadLatentAttention attn;
  final RMSNorm ffnLn; // pre-FFN RMSNorm

  /// Populated when `isMoE == false` (layer < firstKDenseReplace).
  final SwiGluFfn? denseFfn;

  /// Populated when `isMoE == true`.
  final MoEFeedForward? moeFfn;

  DeepSeekV2Block({
    required this.layerIndex,
    required DeepSeekV2Config config,
    required RopeCache rope,
  })  : isMoE = layerIndex >= config.firstKDenseReplace,
        attnLn = RMSNorm(
          config.embedDim,
          eps: config.rmsNormEps,
          device: config.device,
        ),
        attn = MultiHeadLatentAttention(
          config.mlaConfig,
          device: config.device,
          seed: config.seed + layerIndex * 1_000_000,
        ),
        ffnLn = RMSNorm(
          config.embedDim,
          eps: config.rmsNormEps,
          device: config.device,
        ),
        denseFfn = layerIndex >= config.firstKDenseReplace
            ? null
            : SwiGluFfn(
                config.embedDim,
                config.denseFfnDim,
                device: config.device,
                seed: config.seed + layerIndex * 1_000_000 + 900_000,
              ),
        moeFfn = layerIndex >= config.firstKDenseReplace
            ? MoEFeedForward(
                embedDim: config.embedDim,
                numRoutedExperts: config.numRoutedExperts,
                numSharedExperts: config.numSharedExperts,
                topK: config.numExpertsPerTok,
                expertHiddenDim: config.moeExpertHiddenDim,
                numExpertGroups: config.numExpertGroups,
                topKGroups: config.topKGroups,
                activation: ExpertActivation.silu,
                expertVariant: ExpertVariant.swiGlu,
                gateFunction: GateFunction.softmax,
                device: config.device,
                seed: config.seed + layerIndex * 1_000_000 + 950_000,
              )
            : null {
    attn.rope = rope;
  }

  Tensor call(Tensor x, {Tensor? mask, int startPos = 0}) {
    final a = attn(attnLn(x), mask: mask, startPos: startPos);
    final h = x + a;
    final ffnOut = isMoE ? moeFfn!(ffnLn(h)) : denseFfn!(ffnLn(h));
    return h + ffnOut;
  }

  @override
  List<Tensor> parameters() => [
        ...attnLn.parameters(),
        ...attn.parameters(),
        ...ffnLn.parameters(),
        if (denseFfn != null) ...denseFfn!.parameters(),
        if (moeFfn != null) ...moeFfn!.parameters(),
      ];

  @override
  List<Module> submodules() => [
        attnLn,
        attn,
        ffnLn,
        if (denseFfn != null) denseFfn!,
        if (moeFfn != null) moeFfn!,
      ];
}

class DeepSeekV2Model extends Module {
  final DeepSeekV2Config config;

  final Embedding embedIn;
  final List<DeepSeekV2Block> blocks;
  final RMSNorm finalNorm;
  final RopeCache rope;

  /// `[vocabSize, embedDim]`. `null` when `tieWordEmbeddings == true`;
  /// then the head is computed inline as `h @ embedIn.weight.T`.
  final Linear? untiedHead;

  DeepSeekV2Model(this.config)
      : embedIn = Embedding(
          config.vocabSize,
          config.embedDim,
          device: config.device,
          seed: config.seed,
        ),
        finalNorm = RMSNorm(
          config.embedDim,
          eps: config.rmsNormEps,
          device: config.device,
        ),
        rope = RopeCache(
          maxCtx: config.maxCtx,
          headDim: config.mlaConfig.qkRopeHeadDim,
          base: config.ropeBase,
          device: config.device,
        ),
        untiedHead = config.tieWordEmbeddings
            ? null
            : Linear(
                config.embedDim,
                config.vocabSize,
                bias: false,
                device: config.device,
                seed: config.seed + 900_000_000,
              ),
        blocks = <DeepSeekV2Block>[] {
    for (int i = 0; i < config.numLayers; i++) {
      blocks.add(DeepSeekV2Block(
        layerIndex: i,
        config: config,
        rope: rope,
      ));
    }
  }

  /// Forward pass. `tokens` is `[seqLen]` float32 ids. Output is
  /// `[seqLen, vocabSize]` logits.
  Tensor call(Tensor tokens) {
    if (tokens.shape.length != 1) {
      throw ArgumentError(
        'DeepSeekV2Model: tokens must be 1D [seqLen]; got ${tokens.shape}',
      );
    }
    final n = tokens.shape[0];
    if (n == 0) {
      throw ArgumentError('DeepSeekV2Model: empty sequence');
    }
    if (n > config.maxCtx) {
      throw ArgumentError(
        'DeepSeekV2Model: seqLen $n exceeds maxCtx ${config.maxCtx}',
      );
    }
    var h = embedIn(tokens);
    final mask = n > 1 ? causalMask(n, device: h.device) : null;
    for (final b in blocks) {
      h = b(h, mask: mask);
    }
    h = finalNorm(h);
    if (untiedHead != null) {
      return untiedHead!(h);
    }
    // Tied head: h @ embedIn.weight.T.
    return h.matmul(embedIn.weight.transpose());
  }

  @override
  List<Tensor> parameters() => [
        ...embedIn.parameters(),
        for (final b in blocks) ...b.parameters(),
        ...finalNorm.parameters(),
        if (untiedHead != null) ...untiedHead!.parameters(),
      ];

  @override
  List<Module> submodules() => [
        embedIn,
        ...blocks,
        finalNorm,
        if (untiedHead != null) untiedHead!,
      ];
}
