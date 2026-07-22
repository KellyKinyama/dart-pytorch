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
export 'core/nn/vision/vit_backbone.dart';
export 'core/nn/vision/vit_classifier.dart';
export 'core/nn/vision/vit_face_embedding.dart';
export 'core/nn/vision/vit_object_detector.dart';
export 'core/utils/hungarian_algorithm.dart';
export 'core/nn/modalities/audio_transformer.dart';
export 'core/nn/modalities/video_transformer.dart';
export 'core/nn/modalities/text_transformer.dart';
export 'core/nn/modalities/multi_modal_classifier.dart';
export 'core/nn/modalities/multi_modal_encoder.dart';
export 'core/nn/modalities/multi_modal_generator.dart';
export 'core/nn/modalities/multi_modal_lm.dart';
export 'core/nn/muzero_lm.dart';
export 'core/nn/serialize.dart';
export 'core/coop/param_avg.dart';
export 'core/coop/shard.dart';
export 'core/optim/optimizer.dart';
export 'core/optim/sgd.dart';
export 'core/optim/adam.dart';
export 'core/optim/grad_utils.dart';
export 'core/optim/lr_scheduler.dart';
export 'core/data/bpe_tokenizer.dart';
export 'core/data/char_tokenizer.dart';
export 'core/data/dataset.dart';
export 'core/data/image_folder_dataset.dart';
export 'core/data/text_token_dataset.dart';
export 'core/data/csv_dataset.dart';
export 'core/vector_store/index.dart';
export 'core/vector_store/index_flat.dart';
export 'core/vector_store/kmeans.dart';
export 'core/vector_store/index_ivf_flat.dart';
export 'core/vector_store/index_pq.dart';
export 'core/vector_store/index_ivf_pq.dart';
export 'core/vector_store/index_hnsw.dart';
export 'core/vector_store/index_id_map.dart';
export 'core/vector_store/index_scalar_quantizer.dart';
export 'core/vector_store/index_refine_flat.dart';
export 'core/vector_store/index_binary.dart';
export 'core/vector_store/index_binary_flat.dart';
export 'core/vector_store/index_binary_ivf.dart';
export 'core/vector_store/index_lsh.dart';
export 'core/vector_store/vector_transform.dart';
export 'core/vector_store/l2_norm_transform.dart';
export 'core/vector_store/random_rotation_transform.dart';
export 'core/vector_store/pca_transform.dart';
export 'core/vector_store/index_pre_transform.dart';
export 'core/vector_store/index_shards.dart';
export 'core/vector_store/index_replicas.dart';
export 'core/vector_store/index_factory.dart';
export 'core/vector_store/index_convert.dart';
export 'core/vector_store/bench.dart';
export 'core/vector_store/auto_tune.dart';
export 'core/vector_store/gpu_index_flat.dart';
export 'core/vector_store/index_io.dart';
export 'core/vector_store/faiss_io.dart';

/// Trivial sanity function used by the smoke test.
int calculate() {
  return 6 * 7;
}
