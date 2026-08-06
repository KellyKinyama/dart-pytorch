/// Small MLP that projects vision-encoder features into a language
/// model's embedding space.
///
/// Two `Linear` layers with SiLU in the middle:
///
///     y = downProj( silu(upProj(x)) )
///
/// Shape: `[N, inDim] → [N, outDim]`.
///
/// Used by [LlamaVision] to convert `ViTBackbone` per-patch outputs
/// (dim `vitEmbedDim`) into image tokens that can be prepended to
/// `Llama`'s text embeddings (dim `llamaEmbedDim`).
///
/// This is the same shape of module LLaVA calls "mm_projector" — two
/// dense layers, a nonlinearity, no bias tie. LLaVA uses GELU where
/// we use SiLU (matches the rest of the Llama stack).
library;

import '../tensor/tensor.dart';
import 'linear.dart';
import 'module.dart';

class VisionProjector extends Module {
  final int inDim;
  final int outDim;
  final int hiddenDim;
  final Linear upProj;
  final Linear downProj;

  VisionProjector(
    this.inDim,
    this.outDim, {
    int? hiddenDim,
    bool bias = true,
    Device device = Device.CPU,
    int seed = 0,
  }) : hiddenDim = hiddenDim ?? outDim,
       upProj = Linear(
         inDim,
         hiddenDim ?? outDim,
         bias: bias,
         device: device,
         seed: seed,
       ),
       downProj = Linear(
         hiddenDim ?? outDim,
         outDim,
         bias: bias,
         device: device,
         seed: seed + 1,
       );

  /// `x` is `[N, inDim]`. Returns `[N, outDim]`.
  Tensor call(Tensor x) {
    if (x.shape.length != 2 || x.shape[1] != inDim) {
      throw ArgumentError(
        'VisionProjector: expected [N, $inDim]; got ${x.shape}',
      );
    }
    final h = upProj(x);
    final silu = h * h.sigmoid();
    return downProj(silu);
  }

  @override
  List<Tensor> parameters() => [
    ...upProj.parameters(),
    ...downProj.parameters(),
  ];

  @override
  List<Module> submodules() => [upProj, downProj];
}
