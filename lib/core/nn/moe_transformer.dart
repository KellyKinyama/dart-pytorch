/// Transformer stack with the dense FFN sublayer replaced by a
/// Mixture-of-Experts block.
///
/// Same pre-LN skeleton as [AFTBlock] / the vanilla `TransformerBlock`
/// — the only structural difference is `ffn -> MoEFeedForward`. The
/// self-attention path is standard multi-head causal attention so this
/// module drops in wherever `TransformerLM` / `GPT` would.
///
/// `MoELanguageModel` mirrors `TransformerLM`'s public surface
/// (`call(tokens) -> logits`) so training loops built for one can
/// train the other unchanged. Call
/// `model.updateRoutingBias()` once per epoch (or similar cadence) to
/// let the aux-loss-free load balancer nudge routing.
library;

import '../tensor/tensor.dart';
import 'attention/multi_head_attention.dart';
import 'dropout.dart';
import 'embedding.dart';
import 'layer_norm.dart';
import 'linear.dart';
import 'masks.dart';
import 'module.dart';
import 'moe.dart';
import 'positional.dart';

class MoEBlock extends Module {
  final int embedDim;
  final int numHeads;

  final LayerNorm ln1;
  final LayerNorm ln2;
  final MultiHeadAttention attn;
  final MoEFeedForward moe;
  final Dropout dropout;

  MoEBlock(
    this.embedDim,
    this.numHeads, {
    required int numRoutedExperts,
    required int numSharedExperts,
    required int topK,
    required int expertHiddenDim,
    double biasUpdateRate = 0.001,
    double dropoutP = 0.0,
    Device device = Device.CPU,
    int seed = 0,
    ExpertActivation activation = ExpertActivation.relu,
    ExpertVariant expertVariant = ExpertVariant.mlp,
    GateFunction gateFunction = GateFunction.softmax,
    BiasUpdateRule biasUpdateRule = BiasUpdateRule.sign,
    int numExpertGroups = 1,
    int? topKGroups,
    double routeScale = 1.0,
    bool sparseExecution = false,
    bool? renormalizeTopK,
  }) : ln1 = LayerNorm(embedDim, device: device),
       ln2 = LayerNorm(embedDim, device: device),
       attn = MultiHeadAttention(
         embedDim,
         numHeads,
         dropoutP: dropoutP,
         device: device,
         seed: seed,
       ),
       moe = MoEFeedForward(
         embedDim: embedDim,
         numRoutedExperts: numRoutedExperts,
         numSharedExperts: numSharedExperts,
         topK: topK,
         expertHiddenDim: expertHiddenDim,
         biasUpdateRate: biasUpdateRate,
         device: device,
         seed: seed + 4000,
         activation: activation,
         expertVariant: expertVariant,
         gateFunction: gateFunction,
         biasUpdateRule: biasUpdateRule,
         numExpertGroups: numExpertGroups,
         topKGroups: topKGroups,
         routeScale: routeScale,
         sparseExecution: sparseExecution,
         renormalizeTopK: renormalizeTopK,
       ),
       dropout = Dropout(dropoutP);

  /// Forward. `x` is `[T, embedDim]`; `mask` is an optional additive
  /// attention mask (`causalMask(T)` for a decoder-only LM).
  Tensor call(Tensor x, {Tensor? mask}) {
    final h = x + dropout(attn(ln1(x), mask: mask));
    return h + dropout(moe(ln2(h)));
  }

  @override
  List<Tensor> parameters() => [
    ...ln1.parameters(),
    ...ln2.parameters(),
    ...attn.parameters(),
    ...moe.parameters(),
  ];

  @override
  List<Module> submodules() => [ln1, ln2, attn, moe, dropout];
}

/// Decoder-only language model with MoE FFNs.
///
/// Public surface parallels [`TransformerLM`] / [`AFTLanguageModel`]:
/// call `model(tokens)` on a 1D `[seqLen]` tensor to get logits
/// `[seqLen, vocabSize]`. Causal masking is applied inside every
/// block.
class MoELanguageModel extends Module {
  final int vocabSize;
  final int embedDim;
  final int numLayers;
  final int numHeads;
  final int maxLen;

  final int numRoutedExperts;
  final int numSharedExperts;
  final int topK;
  final int expertHiddenDim;

  final Embedding tokenEmb;
  final SinusoidalPositionalEncoding posEnc;
  final List<MoEBlock> blocks;
  final LayerNorm finalLn;
  final Linear head;

  MoELanguageModel({
    required this.vocabSize,
    required this.embedDim,
    required this.numLayers,
    required this.numHeads,
    required this.maxLen,
    this.numRoutedExperts = 4,
    this.numSharedExperts = 1,
    this.topK = 2,
    int? expertHiddenDim,
    double biasUpdateRate = 0.001,
    double dropoutP = 0.0,
    Device device = Device.CPU,
    int seed = 0,
    ExpertActivation activation = ExpertActivation.relu,
    ExpertVariant expertVariant = ExpertVariant.mlp,
    GateFunction gateFunction = GateFunction.softmax,
    BiasUpdateRule biasUpdateRule = BiasUpdateRule.sign,
    int numExpertGroups = 1,
    int? topKGroups,
    double routeScale = 1.0,
    bool sparseExecution = false,
    bool? renormalizeTopK,
  }) : expertHiddenDim = expertHiddenDim ?? embedDim * 4,
       tokenEmb = Embedding(vocabSize, embedDim, device: device, seed: seed),
       posEnc = SinusoidalPositionalEncoding(embedDim),
       blocks = List<MoEBlock>.generate(
         numLayers,
         (i) => MoEBlock(
           embedDim,
           numHeads,
           numRoutedExperts: numRoutedExperts,
           numSharedExperts: numSharedExperts,
           topK: topK,
           expertHiddenDim: expertHiddenDim ?? embedDim * 4,
           biasUpdateRate: biasUpdateRate,
           dropoutP: dropoutP,
           device: device,
           seed: seed + 100000 + i * 10000,
           activation: activation,
           expertVariant: expertVariant,
           gateFunction: gateFunction,
           biasUpdateRule: biasUpdateRule,
           numExpertGroups: numExpertGroups,
           topKGroups: topKGroups,
           routeScale: routeScale,
           sparseExecution: sparseExecution,
           renormalizeTopK: renormalizeTopK,
         ),
       ),
       finalLn = LayerNorm(embedDim, device: device),
       head = Linear(embedDim, vocabSize, device: device, seed: seed + 900000);

  Tensor call(Tensor tokens) {
    if (tokens.shape.length != 1) {
      throw ArgumentError(
        'MoELanguageModel: tokens must be 1D [seqLen]; got ${tokens.shape}',
      );
    }
    final n = tokens.shape[0];
    if (n > maxLen) {
      throw ArgumentError('MoELanguageModel: seqLen $n exceeds maxLen $maxLen');
    }
    var x = tokenEmb(tokens);
    x = posEnc(x);
    final mask = causalMask(n, device: x.device);
    for (final b in blocks) {
      x = b(x, mask: mask);
    }
    x = finalLn(x);
    return head(x);
  }

  /// Applies the aux-loss-free bias update to every MoE block.
  void updateRoutingBias() {
    for (final b in blocks) {
      b.moe.updateRoutingBias();
    }
  }

  @override
  List<Tensor> parameters() => [
    ...tokenEmb.parameters(),
    ...posEnc.parameters(),
    for (final b in blocks) ...b.parameters(),
    ...finalLn.parameters(),
    ...head.parameters(),
  ];

  @override
  List<Module> submodules() => [tokenEmb, posEnc, ...blocks, finalLn, head];
}
