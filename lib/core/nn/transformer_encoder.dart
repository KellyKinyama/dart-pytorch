/// Stack of `TransformerBlock`s with an optional final `LayerNorm`.
///
/// The final norm (on by default) is standard in pre-LN transformers:
/// the residual pathway is not normalized on its way out of the last
/// block, so a trailing `LayerNorm` stabilizes the representation
/// before the head projection.
library;

import '../tensor/tensor.dart';
import 'layer_norm.dart';
import 'module.dart';
import 'transformer.dart';

class TransformerEncoder extends Module {
  final int numLayers;
  final int embedDim;
  final int numHeads;
  final int ffnDim;

  final List<TransformerBlock> blocks;
  final LayerNorm? finalNorm;

  TransformerEncoder(
    this.numLayers,
    this.embedDim,
    this.numHeads, {
    int? ffnDim,
    double dropoutP = 0.0,
    bool finalNorm = true,
    Device device = Device.CPU,
    int seed = 0,
  }) : ffnDim = ffnDim ?? embedDim * 4,
       blocks = List<TransformerBlock>.generate(
         numLayers,
         (i) => TransformerBlock(
           embedDim,
           numHeads,
           ffnDim: ffnDim,
           dropoutP: dropoutP,
           device: device,
           seed: seed + i * 10000,
         ),
       ),
       finalNorm = finalNorm ? LayerNorm(embedDim) : null;

  Tensor call(Tensor x, {Tensor? mask}) {
    var h = x;
    for (final b in blocks) {
      h = b(h, mask: mask);
    }
    if (finalNorm != null) {
      h = finalNorm!(h);
    }
    return h;
  }

  @override
  List<Tensor> parameters() => [
    for (final b in blocks) ...b.parameters(),
    if (finalNorm != null) ...finalNorm!.parameters(),
  ];

  @override
  List<Module> submodules() => [...blocks, if (finalNorm != null) finalNorm!];
}
