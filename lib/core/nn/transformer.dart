/// Pre-LayerNorm Transformer encoder block.
///
/// Layout (per Vaswani et al. with pre-norm; more stable to train):
///
///     h = x + dropout(mha(ln1(x)))
///     y = h + dropout(ffn(ln2(h)))
///
/// Where `ffn(z) = W2(activation(W1(z)))` — the classic 2-layer
/// feed-forward with an inner width of `ffnDim` (defaults to
/// `4 * embedDim`). The activation defaults to ReLU; pass
/// `activation: Activation.geluTanh` for the GPT-2 tanh-approx GELU.
library;

import '../tensor/tensor.dart';
import 'dropout.dart';
import 'kv_cache.dart';
import 'layer_norm.dart';
import 'linear.dart';
import 'module.dart';
import 'attention/multi_head_attention.dart';

/// Activation function for the FFN of a [TransformerBlock].
///
/// * `relu` — plain ReLU (default; matches the original Vaswani recipe
///   used by [TransformerLM]).
/// * `geluTanh` — GPT-2 style GELU with the tanh approximation:
///   `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
/// * `quickGelu` — OpenAI CLIP's fast GELU approximation:
///   `x * sigmoid(1.702 * x)`. Used by every CLIP vision / text
///   transformer, hence needed to load HuggingFace `openai/clip-*`
///   checkpoints without numerical drift.
enum Activation { relu, geluTanh, quickGelu }

class TransformerBlock extends Module {
  final int embedDim;
  final int numHeads;
  final int ffnDim;
  final Activation activation;

  final LayerNorm ln1;
  final LayerNorm ln2;
  final MultiHeadAttention mha;
  final Linear ffn1;
  final Linear ffn2;
  final Dropout dropout;

  TransformerBlock(
    this.embedDim,
    this.numHeads, {
    int? ffnDim,
    double dropoutP = 0.0,
    bool attnBias = false,
    this.activation = Activation.relu,
    Device device = Device.CPU,
    int seed = 0,
  }) : ffnDim = ffnDim ?? embedDim * 4,
       ln1 = LayerNorm(embedDim, device: device),
       ln2 = LayerNorm(embedDim, device: device),
       mha = MultiHeadAttention(
         embedDim,
         numHeads,
         bias: attnBias,
         dropoutP: dropoutP,
         device: device,
         seed: seed,
       ),
       ffn1 = Linear(
         embedDim,
         ffnDim ?? embedDim * 4,
         device: device,
         seed: seed + 4000,
       ),
       ffn2 = Linear(
         ffnDim ?? embedDim * 4,
         embedDim,
         device: device,
         seed: seed + 5000,
       ),
       dropout = Dropout(dropoutP);

  Tensor call(Tensor x, {Tensor? mask, MHACache? cache}) {
    final h = x + dropout(mha(ln1(x), mask: mask, cache: cache));
    final inner = ffn1(ln2(h));
    final activated = switch (activation) {
      Activation.relu => inner.relu(),
      Activation.geluTanh => _geluTanh(inner),
      Activation.quickGelu => _quickGelu(inner),
    };
    final ff = ffn2(activated);
    return h + dropout(ff);
  }

  /// GPT-2 tanh-approximation GELU:
  ///   `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
  ///
  /// Uses only elementwise tensor ops with autograd support.
  static Tensor _geluTanh(Tensor x) {
    const c = 0.7978845608028654; // sqrt(2 / pi)
    final inner = (x + x.pow(3.0) * 0.044715) * c;
    final t = inner.tanh();
    return x * (t + 1.0) * 0.5;
  }

  /// OpenAI CLIP QuickGELU: `x * sigmoid(1.702 * x)`. Cheap and used
  /// throughout the CLIP family — matching it is required to load
  /// HuggingFace CLIP safetensors without accuracy drift.
  static Tensor _quickGelu(Tensor x) {
    return x * (x * 1.702).sigmoid();
  }

  @override
  List<Tensor> parameters() => [
    ...ln1.parameters(),
    ...ln2.parameters(),
    ...mha.parameters(),
    ...ffn1.parameters(),
    ...ffn2.parameters(),
  ];

  @override
  List<Module> submodules() => [ln1, ln2, mha, ffn1, ffn2, dropout];
}
