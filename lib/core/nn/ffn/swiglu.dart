/// SwiGLU feed-forward block — the gated FFN used by Llama, Mistral,
/// Qwen, Phi-3 and DeepSeek. Formally:
///
///     y = downProj( silu(gateProj(x)) * upProj(x) )
///
/// with `silu(z) = z * sigmoid(z)`. Three bias-free `Linear` layers:
///
///   * `gateProj  : [dim -> hiddenDim]`
///   * `upProj    : [dim -> hiddenDim]`
///   * `downProj  : [hiddenDim -> dim]`
///
/// HuggingFace names the corresponding tensors `mlp.gate_proj.weight`,
/// `mlp.up_proj.weight`, `mlp.down_proj.weight` — the loader map is
/// therefore trivial. `hiddenDim` for Llama is `~8/3 * dim` rounded to
/// a multiple of 256; the model-specific value comes from `config.json`.
///
/// This is the standalone extraction of the same math already used
/// inside [Expert] (`ExpertVariant.swiGlu`) in `moe.dart`. Keeping the
/// two in sync is a manual concern for now.
library;

import '../../tensor/tensor.dart';
import '../linear.dart';
import '../module.dart';

class SwiGluFfn extends Module {
  final int dim;
  final int hiddenDim;
  final Linear gateProj;
  final Linear upProj;
  final Linear downProj;

  SwiGluFfn(
    this.dim,
    this.hiddenDim, {
    Device device = Device.CPU,
    int seed = 0,
  }) : gateProj = Linear(
         dim,
         hiddenDim,
         bias: false,
         device: device,
         seed: seed,
       ),
       upProj = Linear(
         dim,
         hiddenDim,
         bias: false,
         device: device,
         seed: seed + 1000,
       ),
       downProj = Linear(
         hiddenDim,
         dim,
         bias: false,
         device: device,
         seed: seed + 2000,
       );

  Tensor call(Tensor x) {
    final gate = gateProj(x);
    final up = upProj(x);
    final silu = gate * gate.sigmoid();
    return downProj(silu * up);
  }

  @override
  List<Tensor> parameters() => [
    ...gateProj.parameters(),
    ...upProj.parameters(),
    ...downProj.parameters(),
  ];
}
