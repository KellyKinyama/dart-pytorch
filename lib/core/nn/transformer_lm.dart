/// A minimal Transformer language model.
///
/// Architecture (single-sequence, 2D tensors throughout):
///
///     tokens: [N]                              # integer indices as float32
///     x = tokenEmb(tokens)                     # [N, embedDim]
///     x = x + posEnc(x)                        # [N, embedDim]
///     x = encoder(x, mask: causalMask)         # [N, embedDim]
///     logits = head(x)                         # [N, vocabSize]
///
/// The final `Linear` head is *not* weight-tied to `tokenEmb` — this
/// keeps the graph and gradient bookkeeping straightforward. Optimizer
/// consumers should pass `model.parameters()` (which flattens every
/// submodule's params in a stable order).
library;

import '../tensor/tensor.dart';
import 'embedding.dart';
import 'linear.dart';
import 'masks.dart';
import 'module.dart';
import 'positional.dart';
import 'transformer_encoder.dart';

class TransformerLM extends Module {
  final int vocabSize;
  final int embedDim;
  final int numLayers;
  final int numHeads;
  final int maxLen;

  final Embedding tokenEmb;
  final SinusoidalPositionalEncoding posEnc;
  final TransformerEncoder encoder;
  final Linear head;

  TransformerLM({
    required this.vocabSize,
    required this.embedDim,
    required this.numLayers,
    required this.numHeads,
    required this.maxLen,
    int? ffnDim,
    double dropoutP = 0.0,
    Device device = Device.CPU,
    int seed = 0,
  }) : tokenEmb = Embedding(vocabSize, embedDim, device: device, seed: seed),
       posEnc = SinusoidalPositionalEncoding(embedDim),
       encoder = TransformerEncoder(
         numLayers,
         embedDim,
         numHeads,
         ffnDim: ffnDim,
         dropoutP: dropoutP,
         device: device,
         seed: seed + 100000,
       ),
       head = Linear(embedDim, vocabSize, device: device, seed: seed + 900000);

  /// Forward pass returning `[seqLen, vocabSize]` logits.
  ///
  /// `tokens` must be a 1D `[seqLen]` tensor of integer class indices
  /// stored as float32 (same convention as `Embedding` / `crossEntropy`).
  /// A causal mask is applied inside the encoder so position `i` only
  /// attends to positions `<= i`.
  Tensor call(Tensor tokens) {
    if (tokens.shape.length != 1) {
      throw ArgumentError(
        'TransformerLM: tokens must be 1D [seqLen]; got ${tokens.shape}',
      );
    }
    final n = tokens.shape[0];
    if (n > maxLen) {
      throw ArgumentError('TransformerLM: seqLen $n exceeds maxLen $maxLen');
    }
    var x = tokenEmb(tokens);
    x = posEnc(x);
    final mask = causalMask(n, device: x.device);
    x = encoder(x, mask: mask);
    return head(x);
  }

  @override
  List<Tensor> parameters() => [
    ...tokenEmb.parameters(),
    ...posEnc.parameters(),
    ...encoder.parameters(),
    ...head.parameters(),
  ];

  @override
  List<Module> submodules() => [tokenEmb, posEnc, encoder, head];
}
