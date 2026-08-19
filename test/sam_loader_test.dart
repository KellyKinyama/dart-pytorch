@Timeout(Duration(minutes: 3))
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

/// Tiny SAM config trio for round-trip loader tests. Uses the same
/// image encoder config as sam_image_encoder_test.dart and a matching
/// prompt encoder + mask decoder.
const _imgCfg = SamImageEncoderConfig(
  imageSize: 32,
  patchSize: 8,
  embedDim: 32,
  numLayers: 2,
  numHeads: 4,
  mlpDim: 64,
  windowSize: 2,
  outChannels: 16,
  globalAttnIndices: [1],
);

const _decCfg = SamMaskDecoderConfig(
  embedDim: 16, // must equal image encoder's outChannels
  numHeads: 4,
  mlpDim: 32,
  transformerDepth: 2,
  numMultimaskOutputs: 3,
  iouHeadDepth: 3,
  iouHeadHiddenDim: 16,
);

Tensor _rand(List<int> shape, {int seed = 0}) {
  final rng = math.Random(seed);
  var n = 1;
  for (final d in shape) {
    n *= d;
  }
  final v = Float32List(n);
  for (int i = 0; i < n; i++) {
    v[i] = (rng.nextDouble() - 0.5) * 0.1;
  }
  return Tensor.fromFloat32List(shape, v);
}

/// Build a fully-populated synthetic state_dict matching everything
/// [SamHFLoader.loadMap] takes.
Map<String, Tensor> _synthState({int seed = 0}) {
  final rng = math.Random(seed);
  int nextSeed() => rng.nextInt(1 << 30);

  final state = <String, Tensor>{};
  final imgD = _imgCfg.embedDim;
  final grid = _imgCfg.gridSize;
  final numHeads = _imgCfg.numHeads;
  final headDim = imgD ~/ numHeads;

  // Image encoder.
  state['image_encoder.patch_embed.proj.weight'] = _rand([
    imgD,
    3,
    _imgCfg.patchSize,
    _imgCfg.patchSize,
  ], seed: nextSeed());
  state['image_encoder.patch_embed.proj.bias'] = _rand([
    imgD,
  ], seed: nextSeed());
  state['image_encoder.pos_embed'] = _rand([
    1,
    grid,
    grid,
    imgD,
  ], seed: nextSeed());

  for (int i = 0; i < _imgCfg.numLayers; i++) {
    final p = 'image_encoder.blocks.$i';
    state['$p.norm1.weight'] = _rand([imgD], seed: nextSeed());
    state['$p.norm1.bias'] = _rand([imgD], seed: nextSeed());
    state['$p.norm2.weight'] = _rand([imgD], seed: nextSeed());
    state['$p.norm2.bias'] = _rand([imgD], seed: nextSeed());
    state['$p.attn.qkv.weight'] = _rand([3 * imgD, imgD], seed: nextSeed());
    state['$p.attn.qkv.bias'] = _rand([3 * imgD], seed: nextSeed());
    state['$p.attn.proj.weight'] = _rand([imgD, imgD], seed: nextSeed());
    state['$p.attn.proj.bias'] = _rand([imgD], seed: nextSeed());
    final isGlobal = _imgCfg.globalAttnIndices.contains(i);
    final ws = isGlobal ? grid : _imgCfg.windowSize;
    state['$p.attn.rel_pos_h'] = _rand([2 * ws - 1, headDim], seed: nextSeed());
    state['$p.attn.rel_pos_w'] = _rand([2 * ws - 1, headDim], seed: nextSeed());
    state['$p.mlp.lin1.weight'] = _rand([
      _imgCfg.mlpDim,
      imgD,
    ], seed: nextSeed());
    state['$p.mlp.lin1.bias'] = _rand([_imgCfg.mlpDim], seed: nextSeed());
    state['$p.mlp.lin2.weight'] = _rand([
      imgD,
      _imgCfg.mlpDim,
    ], seed: nextSeed());
    state['$p.mlp.lin2.bias'] = _rand([imgD], seed: nextSeed());
  }

  state['image_encoder.neck.0.weight'] = _rand([
    _imgCfg.outChannels,
    imgD,
    1,
    1,
  ], seed: nextSeed());
  state['image_encoder.neck.1.weight'] = _rand([
    _imgCfg.outChannels,
  ], seed: nextSeed());
  state['image_encoder.neck.1.bias'] = _rand([
    _imgCfg.outChannels,
  ], seed: nextSeed());
  state['image_encoder.neck.2.weight'] = _rand([
    _imgCfg.outChannels,
    _imgCfg.outChannels,
    3,
    3,
  ], seed: nextSeed());
  state['image_encoder.neck.3.weight'] = _rand([
    _imgCfg.outChannels,
  ], seed: nextSeed());
  state['image_encoder.neck.3.bias'] = _rand([
    _imgCfg.outChannels,
  ], seed: nextSeed());

  // Prompt encoder.
  const promptD = 16;
  const numPosFeats = promptD ~/ 2;
  state['prompt_encoder.pe_layer.positional_encoding_gaussian_matrix'] = _rand([
    2,
    numPosFeats,
  ], seed: nextSeed());
  for (int i = 0; i < 4; i++) {
    state['prompt_encoder.point_embeddings.$i.weight'] = _rand([
      1,
      promptD,
    ], seed: nextSeed());
  }
  state['prompt_encoder.no_mask_embed.weight'] = _rand([
    1,
    promptD,
  ], seed: nextSeed());
  // mask_downscaling: use maskInputChannels=16
  const maskInCh = 16;
  state['prompt_encoder.mask_downscaling.0.weight'] = _rand([
    maskInCh ~/ 4,
    1,
    2,
    2,
  ], seed: nextSeed());
  state['prompt_encoder.mask_downscaling.0.bias'] = _rand([
    maskInCh ~/ 4,
  ], seed: nextSeed());
  state['prompt_encoder.mask_downscaling.1.weight'] = _rand([
    maskInCh ~/ 4,
  ], seed: nextSeed());
  state['prompt_encoder.mask_downscaling.1.bias'] = _rand([
    maskInCh ~/ 4,
  ], seed: nextSeed());
  state['prompt_encoder.mask_downscaling.3.weight'] = _rand([
    maskInCh,
    maskInCh ~/ 4,
    2,
    2,
  ], seed: nextSeed());
  state['prompt_encoder.mask_downscaling.3.bias'] = _rand([
    maskInCh,
  ], seed: nextSeed());
  state['prompt_encoder.mask_downscaling.4.weight'] = _rand([
    maskInCh,
  ], seed: nextSeed());
  state['prompt_encoder.mask_downscaling.4.bias'] = _rand([
    maskInCh,
  ], seed: nextSeed());
  state['prompt_encoder.mask_downscaling.6.weight'] = _rand([
    promptD,
    maskInCh,
    1,
    1,
  ], seed: nextSeed());
  state['prompt_encoder.mask_downscaling.6.bias'] = _rand([
    promptD,
  ], seed: nextSeed());

  // Mask decoder.
  final decD = _decCfg.embedDim;
  state['mask_decoder.iou_token.weight'] = _rand([1, decD], seed: nextSeed());
  state['mask_decoder.mask_tokens.weight'] = _rand([
    _decCfg.numMaskTokens,
    decD,
  ], seed: nextSeed());

  final internal = decD; // downsample_rate=1 for self-attn
  final internalDown = decD ~/ 2; // downsample_rate=2 for cross-attn
  for (int i = 0; i < _decCfg.transformerDepth; i++) {
    final p = 'mask_decoder.transformer.layers.$i';
    // self_attn: no downsample
    for (final proj in ['q_proj', 'k_proj', 'v_proj']) {
      state['$p.self_attn.$proj.weight'] = _rand([
        internal,
        decD,
      ], seed: nextSeed());
      state['$p.self_attn.$proj.bias'] = _rand([internal], seed: nextSeed());
    }
    state['$p.self_attn.out_proj.weight'] = _rand([
      decD,
      internal,
    ], seed: nextSeed());
    state['$p.self_attn.out_proj.bias'] = _rand([decD], seed: nextSeed());

    state['$p.norm1.weight'] = _rand([decD], seed: nextSeed());
    state['$p.norm1.bias'] = _rand([decD], seed: nextSeed());

    // cross_attn_token_to_image: downsample_rate=2
    for (final proj in ['q_proj', 'k_proj', 'v_proj']) {
      state['$p.cross_attn_token_to_image.$proj.weight'] = _rand([
        internalDown,
        decD,
      ], seed: nextSeed());
      state['$p.cross_attn_token_to_image.$proj.bias'] = _rand([
        internalDown,
      ], seed: nextSeed());
    }
    state['$p.cross_attn_token_to_image.out_proj.weight'] = _rand([
      decD,
      internalDown,
    ], seed: nextSeed());
    state['$p.cross_attn_token_to_image.out_proj.bias'] = _rand([
      decD,
    ], seed: nextSeed());
    state['$p.norm2.weight'] = _rand([decD], seed: nextSeed());
    state['$p.norm2.bias'] = _rand([decD], seed: nextSeed());

    // mlp
    state['$p.mlp.lin1.weight'] = _rand([
      _decCfg.mlpDim,
      decD,
    ], seed: nextSeed());
    state['$p.mlp.lin1.bias'] = _rand([_decCfg.mlpDim], seed: nextSeed());
    state['$p.mlp.lin2.weight'] = _rand([
      decD,
      _decCfg.mlpDim,
    ], seed: nextSeed());
    state['$p.mlp.lin2.bias'] = _rand([decD], seed: nextSeed());
    state['$p.norm3.weight'] = _rand([decD], seed: nextSeed());
    state['$p.norm3.bias'] = _rand([decD], seed: nextSeed());

    // cross_attn_image_to_token
    for (final proj in ['q_proj', 'k_proj', 'v_proj']) {
      state['$p.cross_attn_image_to_token.$proj.weight'] = _rand([
        internalDown,
        decD,
      ], seed: nextSeed());
      state['$p.cross_attn_image_to_token.$proj.bias'] = _rand([
        internalDown,
      ], seed: nextSeed());
    }
    state['$p.cross_attn_image_to_token.out_proj.weight'] = _rand([
      decD,
      internalDown,
    ], seed: nextSeed());
    state['$p.cross_attn_image_to_token.out_proj.bias'] = _rand([
      decD,
    ], seed: nextSeed());
    state['$p.norm4.weight'] = _rand([decD], seed: nextSeed());
    state['$p.norm4.bias'] = _rand([decD], seed: nextSeed());
  }

  // final_attn_token_to_image + norm_final
  for (final proj in ['q_proj', 'k_proj', 'v_proj']) {
    state['mask_decoder.transformer.final_attn_token_to_image.$proj.weight'] =
        _rand([internalDown, decD], seed: nextSeed());
    state['mask_decoder.transformer.final_attn_token_to_image.$proj.bias'] =
        _rand([internalDown], seed: nextSeed());
  }
  state['mask_decoder.transformer.final_attn_token_to_image.out_proj.weight'] =
      _rand([decD, internalDown], seed: nextSeed());
  state['mask_decoder.transformer.final_attn_token_to_image.out_proj.bias'] =
      _rand([decD], seed: nextSeed());
  state['mask_decoder.transformer.norm_final_attn.weight'] = _rand([
    decD,
  ], seed: nextSeed());
  state['mask_decoder.transformer.norm_final_attn.bias'] = _rand([
    decD,
  ], seed: nextSeed());

  // Output upscaling: ConvTranspose2d layers.
  state['mask_decoder.output_upscaling.0.weight'] = _rand([
    decD,
    decD ~/ 4,
    2,
    2,
  ], seed: nextSeed());
  state['mask_decoder.output_upscaling.0.bias'] = _rand([
    decD ~/ 4,
  ], seed: nextSeed());
  state['mask_decoder.output_upscaling.1.weight'] = _rand([
    decD ~/ 4,
  ], seed: nextSeed());
  state['mask_decoder.output_upscaling.1.bias'] = _rand([
    decD ~/ 4,
  ], seed: nextSeed());
  state['mask_decoder.output_upscaling.3.weight'] = _rand([
    decD ~/ 4,
    decD ~/ 8,
    2,
    2,
  ], seed: nextSeed());
  state['mask_decoder.output_upscaling.3.bias'] = _rand([
    decD ~/ 8,
  ], seed: nextSeed());

  // Per-mask hypernet MLPs (3 layers each).
  for (int i = 0; i < _decCfg.numMaskTokens; i++) {
    for (int j = 0; j < 3; j++) {
      final inD = j == 0 ? decD : decD;
      final outD = j == 2 ? decD ~/ 8 : decD;
      state['mask_decoder.output_hypernetworks_mlps.$i.layers.$j.weight'] =
          _rand([outD, inD], seed: nextSeed());
      state['mask_decoder.output_hypernetworks_mlps.$i.layers.$j.bias'] = _rand(
        [outD],
        seed: nextSeed(),
      );
    }
  }
  // IoU prediction head.
  for (int j = 0; j < _decCfg.iouHeadDepth; j++) {
    final inD = j == 0 ? decD : _decCfg.iouHeadHiddenDim;
    final outD = j == _decCfg.iouHeadDepth - 1
        ? _decCfg.numMaskTokens
        : _decCfg.iouHeadHiddenDim;
    state['mask_decoder.iou_prediction_head.layers.$j.weight'] = _rand([
      outD,
      inD,
    ], seed: nextSeed());
    state['mask_decoder.iou_prediction_head.layers.$j.bias'] = _rand([
      outD,
    ], seed: nextSeed());
  }

  return state;
}

void main() {
  group('SamHFLoader roundtrip', () {
    test('consumes a synthetic state_dict with no unused keys', () {
      final imgEnc = SamImageEncoder(_imgCfg);
      final promptEnc = SamPromptEncoder(
        embedDim: 16,
        imageEmbedH: 4,
        imageEmbedW: 4,
        imageSize: 32,
        maskInSize: 16,
        maskInputChannels: 16,
      );
      final maskDec = SamMaskDecoder(_decCfg);
      final state = _synthState(seed: 42);
      final report = SamHFLoader.loadMap(
        imageEncoder: imgEnc,
        promptEncoder: promptEnc,
        maskDecoder: maskDec,
        state: state,
      );
      expect(
        report.unusedKeys,
        isEmpty,
        reason: 'unused keys: ${report.unusedKeys.take(5).toList()}',
      );
    });

    test('rejects a missing tensor', () {
      final imgEnc = SamImageEncoder(_imgCfg);
      final promptEnc = SamPromptEncoder(
        embedDim: 16,
        imageEmbedH: 4,
        imageEmbedW: 4,
        imageSize: 32,
        maskInSize: 16,
        maskInputChannels: 16,
      );
      final maskDec = SamMaskDecoder(_decCfg);
      final state = _synthState(seed: 42);
      state.remove('image_encoder.pos_embed');
      expect(
        () => SamHFLoader.loadMap(
          imageEncoder: imgEnc,
          promptEncoder: promptEnc,
          maskDecoder: maskDec,
          state: state,
        ),
        throwsArgumentError,
      );
    });

    test('auto-detects sam. prefix', () {
      final imgEnc = SamImageEncoder(_imgCfg);
      final promptEnc = SamPromptEncoder(
        embedDim: 16,
        imageEmbedH: 4,
        imageEmbedW: 4,
        imageSize: 32,
        maskInSize: 16,
        maskInputChannels: 16,
      );
      final maskDec = SamMaskDecoder(_decCfg);
      final flat = _synthState(seed: 42);
      final prefixed = <String, Tensor>{
        for (final e in flat.entries) 'sam.${e.key}': e.value,
      };
      final report = SamHFLoader.loadMap(
        imageEncoder: imgEnc,
        promptEncoder: promptEnc,
        maskDecoder: maskDec,
        state: prefixed,
      );
      expect(report.unusedKeys, isEmpty);
    });
  });
}
