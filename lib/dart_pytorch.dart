/// Public entry point for the `dart_pytorch` package.
///
/// Re-exports the GPU-backed [Tensor] and its `matmul` extension.
/// The CUDA library `libmat_mul.so` is loaded lazily by the first
/// tensor factory call; see the README for build instructions.
library;

export 'core/tensor/tensor.dart';
export 'core/nn/module.dart';
export 'core/nn/layer_norm.dart';
export 'core/nn/embedding.dart';
export 'core/nn/linear.dart';
export 'core/nn/dropout.dart';
export 'core/nn/attention/multi_head_attention.dart';
export 'core/nn/attention/multi_head_cross_attention.dart';
export 'core/nn/attention/aft_attention.dart';
export 'core/nn/transformer.dart';
export 'core/nn/positional.dart';
export 'core/nn/masks.dart';
export 'core/nn/transformer_encoder.dart';
export 'core/nn/transformer_decoder_block.dart';
export 'core/nn/transformer_decoder.dart';
export 'core/nn/encoder_decoder.dart';
export 'core/nn/transformer_lm.dart';
export 'core/nn/aft_transformer.dart';
export 'core/nn/moe.dart';
export 'core/nn/moe_transformer.dart';
export 'core/nn/kv_cache.dart';
export 'core/nn/gpt.dart';
export 'core/nn/serialize.dart';
export 'core/optim/optimizer.dart';
export 'core/optim/sgd.dart';
export 'core/optim/adam.dart';
export 'core/optim/grad_utils.dart';
export 'core/optim/lr_scheduler.dart';
export 'core/data/bpe_tokenizer.dart';
export 'core/data/char_tokenizer.dart';

/// Trivial sanity function used by the smoke test.
int calculate() {
  return 6 * 7;
}
