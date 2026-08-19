/// Loader for HuggingFace / Facebook Research **SAM** safetensors.
///
/// Reads a `.safetensors` file dumped from a torch checkpoint (either
/// `sam_vit_b_01ec64.pth` via `scripts/convert_sam_pt_to_safetensors.py`
/// or the HF `facebook/sam-vit-base` bundle) and binds it to a triple
/// of `SamImageEncoder` + `SamPromptEncoder` + `SamMaskDecoder`.
///
/// Prefixes vary across sources:
///   * Original `sam_vit_b_01ec64.pth` uses no prefix on the three
///     submodels (keys start with `image_encoder.`, `prompt_encoder.`,
///     `mask_decoder.`).
///   * The HF `facebook/sam-vit-base` bundle uses the same top-level
///     names; the wrapper module is called `sam.` in the joint model
///     but is dropped when the vision submodel is loaded alone.
///
/// The loader auto-detects the `sam.` prefix.
library;

import 'dart:typed_data';

import '../../tensor/tensor.dart';
import '../safetensors.dart';
import 'sam_image_encoder.dart';
import 'sam_mask_decoder.dart';
import 'sam_prompt_encoder.dart';

class SamLoadReport {
  final int consumedCount;
  final List<String> unusedKeys;
  const SamLoadReport({required this.consumedCount, required this.unusedKeys});

  @override
  String toString() =>
      'SamLoadReport(consumed=$consumedCount, unused=${unusedKeys.length})';
}

class SamHFLoader {
  /// Load all three submodules from a single safetensors file.
  static SamLoadReport loadFile({
    required SamImageEncoder imageEncoder,
    required SamPromptEncoder promptEncoder,
    required SamMaskDecoder maskDecoder,
    required String path,
  }) {
    final state = SafeTensors.loadFile(path);
    return loadMap(
      imageEncoder: imageEncoder,
      promptEncoder: promptEncoder,
      maskDecoder: maskDecoder,
      state: state,
    );
  }

  static SamLoadReport loadMap({
    required SamImageEncoder imageEncoder,
    required SamPromptEncoder promptEncoder,
    required SamMaskDecoder maskDecoder,
    required Map<String, Tensor> state,
  }) {
    final prefix = _detectPrefix(state);
    final consumed = <String>{};

    Tensor take(String name) {
      final key = '$prefix$name';
      final t = state[key];
      if (t == null) {
        throw ArgumentError('sam loader: missing tensor "$key"');
      }
      consumed.add(key);
      return t;
    }

    _loadImageEncoder(imageEncoder, take);
    _loadPromptEncoder(promptEncoder, take);
    _loadMaskDecoder(maskDecoder, take);

    final unused = state.keys.where((k) => !consumed.contains(k)).toList()
      ..sort();
    return SamLoadReport(consumedCount: consumed.length, unusedKeys: unused);
  }

  static String _detectPrefix(Map<String, Tensor> state) {
    const candidates = ['sam.', ''];
    for (final p in candidates) {
      if (state.containsKey('${p}image_encoder.pos_embed') ||
          state.containsKey('${p}image_encoder.patch_embed.proj.weight')) {
        return p;
      }
    }
    throw ArgumentError(
      'sam loader: could not detect prefix — expected image_encoder.pos_embed '
      'or image_encoder.patch_embed.proj.weight',
    );
  }

  // -----------------------------------------------------------------
  // Image encoder — SAM has: patch_embed, pos_embed, N blocks (each
  // with norm1/attn/norm2/mlp/rel_pos_h/rel_pos_w), neck (conv0 -> LN2d
  // -> conv2 -> LN2d).
  // -----------------------------------------------------------------
  static void _loadImageEncoder(
    SamImageEncoder m,
    Tensor Function(String) take,
  ) {
    final cfg = m.config;
    final d = cfg.embedDim;

    // Patch embed — SAM stores as `Conv2d.weight [D, 3, P, P]` +
    // `.bias [D]`. Our patchEmbed is also Conv2d, same layout.
    _copy(
      m.patchEmbed.weight,
      _expectShape(
        take('image_encoder.patch_embed.proj.weight'),
        [d, 3, cfg.patchSize, cfg.patchSize],
        'image_encoder.patch_embed.proj.weight',
      ),
    );
    _copy(
      m.patchEmbed.bias!,
      _reshapeVectorTo1xN(
        _expectShape(
          take('image_encoder.patch_embed.proj.bias'),
          [d],
          'image_encoder.patch_embed.proj.bias',
        ),
      ),
    );

    // 2-D positional embedding — SAM ships [1, grid, grid, D]; we
    // store [grid, grid, D]. Strip the batch axis.
    final pos = _expectShape(take('image_encoder.pos_embed'), [
      1,
      cfg.gridSize,
      cfg.gridSize,
      d,
    ], 'image_encoder.pos_embed');
    _copy(
      m.posEmbed,
      Tensor.fromList([cfg.gridSize, cfg.gridSize, d], pos.toList()),
    );

    // Blocks.
    final h = cfg.numHeads;
    final headDim = d ~/ h;
    for (int i = 0; i < cfg.numLayers; i++) {
      final block = m.blocks[i];
      final p = 'image_encoder.blocks.$i';

      _copy(
        block.norm1.gamma,
        _expectShape(take('$p.norm1.weight'), [d], '$p.norm1.weight'),
      );
      _copy(
        block.norm1.beta,
        _expectShape(take('$p.norm1.bias'), [d], '$p.norm1.bias'),
      );
      _copy(
        block.norm2.gamma,
        _expectShape(take('$p.norm2.weight'), [d], '$p.norm2.weight'),
      );
      _copy(
        block.norm2.beta,
        _expectShape(take('$p.norm2.bias'), [d], '$p.norm2.bias'),
      );

      // Attention: SAM uses a fused qkv Linear: [3D, D] weight, [3D]
      // bias. Split into per-head Q/K/V.
      final qkvW = _expectShape(take('$p.attn.qkv.weight'), [
        3 * d,
        d,
      ], '$p.attn.qkv.weight');
      final qkvB = _expectShape(take('$p.attn.qkv.bias'), [
        3 * d,
      ], '$p.attn.qkv.bias');
      // Split by rows: [0, D) = Q, [D, 2D) = K, [2D, 3D) = V.
      for (int hh = 0; hh < h; hh++) {
        final qRow = hh * headDim;
        _copy(block.attn.wq[hh].weight, _sliceRows(qkvW, qRow, qRow + headDim));
        _copy(
          block.attn.wq[hh].bias!,
          _reshapeVectorTo1xN(_slice1DVector(qkvB, qRow, qRow + headDim)),
        );
        final kRow = d + hh * headDim;
        _copy(block.attn.wk[hh].weight, _sliceRows(qkvW, kRow, kRow + headDim));
        _copy(
          block.attn.wk[hh].bias!,
          _reshapeVectorTo1xN(_slice1DVector(qkvB, kRow, kRow + headDim)),
        );
        final vRow = 2 * d + hh * headDim;
        _copy(block.attn.wv[hh].weight, _sliceRows(qkvW, vRow, vRow + headDim));
        _copy(
          block.attn.wv[hh].bias!,
          _reshapeVectorTo1xN(_slice1DVector(qkvB, vRow, vRow + headDim)),
        );
      }

      // Output projection.
      _copy(
        block.attn.wo.weight,
        _expectShape(take('$p.attn.proj.weight'), [
          d,
          d,
        ], '$p.attn.proj.weight'),
      );
      _copy(
        block.attn.wo.bias!,
        _reshapeVectorTo1xN(
          _expectShape(take('$p.attn.proj.bias'), [d], '$p.attn.proj.bias'),
        ),
      );

      // Relative-positional bias tables. SAM stores them as
      // [2·ws-1, headDim] per head; ours is [numHeads, 2·ws-1, headDim]
      // for both H and W. Load per head by row.
      final ws = block.attn.windowSize;
      final relH = _expectShape(take('$p.attn.rel_pos_h'), [
        2 * ws - 1,
        headDim,
      ], '$p.attn.rel_pos_h');
      final relW = _expectShape(take('$p.attn.rel_pos_w'), [
        2 * ws - 1,
        headDim,
      ], '$p.attn.rel_pos_w');
      // Our rel_pos is [numHeads, 2*ws-1, headDim] — all heads share
      // the same table in SAM. Broadcast: fill every head row.
      _copyBroadcastPerHead(block.attn.relPosH, relH, h);
      _copyBroadcastPerHead(block.attn.relPosW, relW, h);

      // MLP.
      _copy(
        block.fc1.weight,
        _expectShape(take('$p.mlp.lin1.weight'), [
          cfg.mlpDim,
          d,
        ], '$p.mlp.lin1.weight'),
      );
      _copy(
        block.fc1.bias!,
        _reshapeVectorTo1xN(
          _expectShape(take('$p.mlp.lin1.bias'), [
            cfg.mlpDim,
          ], '$p.mlp.lin1.bias'),
        ),
      );
      _copy(
        block.fc2.weight,
        _expectShape(take('$p.mlp.lin2.weight'), [
          d,
          cfg.mlpDim,
        ], '$p.mlp.lin2.weight'),
      );
      _copy(
        block.fc2.bias!,
        _reshapeVectorTo1xN(
          _expectShape(take('$p.mlp.lin2.bias'), [d], '$p.mlp.lin2.bias'),
        ),
      );
    }

    // Neck. SAM has `neck.0 = Conv2d(bias=False)`, `neck.1 = LN2d`,
    // `neck.2 = Conv2d(bias=False)`, `neck.3 = LN2d`.
    _copy(
      m.neck1.weight,
      _expectShape(take('image_encoder.neck.0.weight'), [
        cfg.outChannels,
        d,
        1,
        1,
      ], 'image_encoder.neck.0.weight'),
    );
    _copy(
      m.neckLn1.gamma,
      _expectShape(take('image_encoder.neck.1.weight'), [
        cfg.outChannels,
      ], 'image_encoder.neck.1.weight'),
    );
    _copy(
      m.neckLn1.beta,
      _expectShape(take('image_encoder.neck.1.bias'), [
        cfg.outChannels,
      ], 'image_encoder.neck.1.bias'),
    );
    _copy(
      m.neck2.weight,
      _expectShape(take('image_encoder.neck.2.weight'), [
        cfg.outChannels,
        cfg.outChannels,
        3,
        3,
      ], 'image_encoder.neck.2.weight'),
    );
    _copy(
      m.neckLn2.gamma,
      _expectShape(take('image_encoder.neck.3.weight'), [
        cfg.outChannels,
      ], 'image_encoder.neck.3.weight'),
    );
    _copy(
      m.neckLn2.beta,
      _expectShape(take('image_encoder.neck.3.bias'), [
        cfg.outChannels,
      ], 'image_encoder.neck.3.bias'),
    );
  }

  // -----------------------------------------------------------------
  // Prompt encoder — pe_layer (Gaussian frozen matrix),
  // point_embeddings.{0..3}.weight, not_a_point_embed.weight,
  // mask_downscaling.{0,1,2,3,4,5,6} sequence.
  // -----------------------------------------------------------------
  static void _loadPromptEncoder(
    SamPromptEncoder m,
    Tensor Function(String) take,
  ) {
    // Positional encoding Gaussian matrix — SAM stores as
    // `prompt_encoder.pe_layer.positional_encoding_gaussian_matrix`
    // shape [numPosFeats, 2] (input rows=2, output cols=numPosFeats).
    // Wait — SAM's shape is [2, numPosFeats]. Same as ours.
    final gm = _expectShape(
      take('prompt_encoder.pe_layer.positional_encoding_gaussian_matrix'),
      m.posEmbed.gaussianMatrix.shape,
      'prompt_encoder.pe_layer.positional_encoding_gaussian_matrix',
    );
    _copy(m.posEmbed.gaussianMatrix, gm);

    // Point embeddings: 4 separate [1, embedDim] rows. SAM uses
    // `nn.Embedding(1, embed_dim)` per type — key like
    // `point_embeddings.{i}.weight` shape [1, embed_dim].
    final ptData = List<double>.filled(4 * m.embedDim, 0);
    for (int i = 0; i < 4; i++) {
      final e = _expectShape(
        take('prompt_encoder.point_embeddings.$i.weight'),
        [1, m.embedDim],
        'prompt_encoder.point_embeddings.$i.weight',
      );
      final vals = e.toList();
      for (int j = 0; j < m.embedDim; j++) {
        ptData[i * m.embedDim + j] = vals[j];
      }
    }
    _copy(m.pointEmbeddings, Tensor.fromList([4, m.embedDim], ptData));

    // SAM has a `not_a_point_embed.weight` [1, embed_dim] for absent
    // point prompts. We don't currently model this — silently absorb.
    if (m.embedDim > 0) {
      // Try to consume the not_a_point_embed if present.
      try {
        take('prompt_encoder.not_a_point_embed.weight');
      } catch (_) {
        // Not fatal — some checkpoints omit it.
      }
    }

    // No-mask embedding.
    final nm = _expectShape(
      take('prompt_encoder.no_mask_embed.weight'),
      [1, m.embedDim],
      'prompt_encoder.no_mask_embed.weight',
    );
    _copy(m.noMaskEmbedding, Tensor.fromList([m.embedDim], nm.toList()));

    // Mask downscaling CNN — SAM uses:
    //   mask_downscaling.0: Conv2d(1, C/4, 2, 2)  bias
    //   mask_downscaling.1: LayerNorm2d(C/4)
    //   mask_downscaling.2: GELU (no params)
    //   mask_downscaling.3: Conv2d(C/4, C, 2, 2)  bias
    //   mask_downscaling.4: LayerNorm2d(C)
    //   mask_downscaling.5: GELU
    //   mask_downscaling.6: Conv2d(C, embed_dim, 1)  bias
    _copy(
      m.maskConv1.weight,
      _expectShape(
        take('prompt_encoder.mask_downscaling.0.weight'),
        [m.maskInputChannels ~/ 4, 1, 2, 2],
        'prompt_encoder.mask_downscaling.0.weight',
      ),
    );
    _copy(
      m.maskConv1.bias!,
      _reshapeVectorTo1xN(
        _expectShape(
          take('prompt_encoder.mask_downscaling.0.bias'),
          [m.maskInputChannels ~/ 4],
          'prompt_encoder.mask_downscaling.0.bias',
        ),
      ),
    );
    _copy(
      m.maskLn1.gamma,
      _expectShape(
        take('prompt_encoder.mask_downscaling.1.weight'),
        [m.maskInputChannels ~/ 4],
        'prompt_encoder.mask_downscaling.1.weight',
      ),
    );
    _copy(
      m.maskLn1.beta,
      _expectShape(
        take('prompt_encoder.mask_downscaling.1.bias'),
        [m.maskInputChannels ~/ 4],
        'prompt_encoder.mask_downscaling.1.bias',
      ),
    );

    _copy(
      m.maskConv2.weight,
      _expectShape(
        take('prompt_encoder.mask_downscaling.3.weight'),
        [m.maskInputChannels, m.maskInputChannels ~/ 4, 2, 2],
        'prompt_encoder.mask_downscaling.3.weight',
      ),
    );
    _copy(
      m.maskConv2.bias!,
      _reshapeVectorTo1xN(
        _expectShape(
          take('prompt_encoder.mask_downscaling.3.bias'),
          [m.maskInputChannels],
          'prompt_encoder.mask_downscaling.3.bias',
        ),
      ),
    );
    _copy(
      m.maskLn2.gamma,
      _expectShape(
        take('prompt_encoder.mask_downscaling.4.weight'),
        [m.maskInputChannels],
        'prompt_encoder.mask_downscaling.4.weight',
      ),
    );
    _copy(
      m.maskLn2.beta,
      _expectShape(
        take('prompt_encoder.mask_downscaling.4.bias'),
        [m.maskInputChannels],
        'prompt_encoder.mask_downscaling.4.bias',
      ),
    );

    _copy(
      m.maskConv3.weight,
      _expectShape(
        take('prompt_encoder.mask_downscaling.6.weight'),
        [m.embedDim, m.maskInputChannels, 1, 1],
        'prompt_encoder.mask_downscaling.6.weight',
      ),
    );
    _copy(
      m.maskConv3.bias!,
      _reshapeVectorTo1xN(
        _expectShape(
          take('prompt_encoder.mask_downscaling.6.bias'),
          [m.embedDim],
          'prompt_encoder.mask_downscaling.6.bias',
        ),
      ),
    );
  }

  // -----------------------------------------------------------------
  // Mask decoder — iou_token, mask_tokens, transformer.{layers, final},
  // output_upscaling.{0..4}, output_hypernetworks_mlps.{i}.layers.{j},
  // iou_prediction_head.layers.{j}.
  // -----------------------------------------------------------------
  static void _loadMaskDecoder(SamMaskDecoder m, Tensor Function(String) take) {
    final cfg = m.config;

    // IoU + mask tokens.
    final iouTok = _expectShape(take('mask_decoder.iou_token.weight'), [
      1,
      cfg.embedDim,
    ], 'mask_decoder.iou_token.weight');
    final maskTok = _expectShape(take('mask_decoder.mask_tokens.weight'), [
      cfg.numMaskTokens,
      cfg.embedDim,
    ], 'mask_decoder.mask_tokens.weight');
    // Assemble into our [numOutputTokens = 1 + numMaskTokens, embedDim].
    final iouData = iouTok.toList();
    final maskData = maskTok.toList();
    final combined = List<double>.filled(cfg.numOutputTokens * cfg.embedDim, 0);
    for (int i = 0; i < cfg.embedDim; i++) {
      combined[i] = iouData[i];
    }
    for (int i = 0; i < cfg.numMaskTokens * cfg.embedDim; i++) {
      combined[cfg.embedDim + i] = maskData[i];
    }
    _copy(
      m.tokenEmbeddings,
      Tensor.fromList([cfg.numOutputTokens, cfg.embedDim], combined),
    );

    // Transformer layers.
    for (int i = 0; i < cfg.transformerDepth; i++) {
      final block = m.transformer.blocks[i];
      final p = 'mask_decoder.transformer.layers.$i';
      _loadSamAttention(block.selfAttn, take, '$p.self_attn');
      _copy(
        block.norm1.gamma,
        _expectShape(take('$p.norm1.weight'), [
          cfg.embedDim,
        ], '$p.norm1.weight'),
      );
      _copy(
        block.norm1.beta,
        _expectShape(take('$p.norm1.bias'), [cfg.embedDim], '$p.norm1.bias'),
      );
      _loadSamAttention(
        block.crossAttnTokenToImage,
        take,
        '$p.cross_attn_token_to_image',
      );
      _copy(
        block.norm2.gamma,
        _expectShape(take('$p.norm2.weight'), [
          cfg.embedDim,
        ], '$p.norm2.weight'),
      );
      _copy(
        block.norm2.beta,
        _expectShape(take('$p.norm2.bias'), [cfg.embedDim], '$p.norm2.bias'),
      );
      _loadSamMlpBlock(block.mlp, take, '$p.mlp');
      _copy(
        block.norm3.gamma,
        _expectShape(take('$p.norm3.weight'), [
          cfg.embedDim,
        ], '$p.norm3.weight'),
      );
      _copy(
        block.norm3.beta,
        _expectShape(take('$p.norm3.bias'), [cfg.embedDim], '$p.norm3.bias'),
      );
      _loadSamAttention(
        block.crossAttnImageToToken,
        take,
        '$p.cross_attn_image_to_token',
      );
      _copy(
        block.norm4.gamma,
        _expectShape(take('$p.norm4.weight'), [
          cfg.embedDim,
        ], '$p.norm4.weight'),
      );
      _copy(
        block.norm4.beta,
        _expectShape(take('$p.norm4.bias'), [cfg.embedDim], '$p.norm4.bias'),
      );
    }

    // Final token-to-image attention + norm.
    _loadSamAttention(
      m.transformer.finalAttnTokenToImage,
      take,
      'mask_decoder.transformer.final_attn_token_to_image',
    );
    _copy(
      m.transformer.normFinal.gamma,
      _expectShape(
        take('mask_decoder.transformer.norm_final_attn.weight'),
        [cfg.embedDim],
        'mask_decoder.transformer.norm_final_attn.weight',
      ),
    );
    _copy(
      m.transformer.normFinal.beta,
      _expectShape(
        take('mask_decoder.transformer.norm_final_attn.bias'),
        [cfg.embedDim],
        'mask_decoder.transformer.norm_final_attn.bias',
      ),
    );

    // Output upscaling.
    // SAM: output_upscaling.0 = ConvTranspose2d(embed, embed/4, k=2, s=2)
    //      output_upscaling.1 = LayerNorm2d(embed/4)
    //      output_upscaling.2 = GELU
    //      output_upscaling.3 = ConvTranspose2d(embed/4, embed/8, k=2, s=2)
    //      output_upscaling.4 = GELU
    _copy(
      m.outputUpscaling1.weight,
      _expectShape(
        take('mask_decoder.output_upscaling.0.weight'),
        m.outputUpscaling1.weight.shape,
        'mask_decoder.output_upscaling.0.weight',
      ),
    );
    _copy(
      m.outputUpscaling1.bias!,
      _reshapeVectorTo1xN(
        _expectShape(
          take('mask_decoder.output_upscaling.0.bias'),
          [cfg.embedDim ~/ 4],
          'mask_decoder.output_upscaling.0.bias',
        ),
      ),
    );
    _copy(
      m.outputUpscalingLn.gamma,
      _expectShape(
        take('mask_decoder.output_upscaling.1.weight'),
        [cfg.embedDim ~/ 4],
        'mask_decoder.output_upscaling.1.weight',
      ),
    );
    _copy(
      m.outputUpscalingLn.beta,
      _expectShape(
        take('mask_decoder.output_upscaling.1.bias'),
        [cfg.embedDim ~/ 4],
        'mask_decoder.output_upscaling.1.bias',
      ),
    );
    _copy(
      m.outputUpscaling2.weight,
      _expectShape(
        take('mask_decoder.output_upscaling.3.weight'),
        m.outputUpscaling2.weight.shape,
        'mask_decoder.output_upscaling.3.weight',
      ),
    );
    _copy(
      m.outputUpscaling2.bias!,
      _reshapeVectorTo1xN(
        _expectShape(
          take('mask_decoder.output_upscaling.3.bias'),
          [cfg.embedDim ~/ 8],
          'mask_decoder.output_upscaling.3.bias',
        ),
      ),
    );

    // Per-mask hypernetwork MLPs.
    for (int i = 0; i < cfg.numMaskTokens; i++) {
      _loadSamMlp(
        m.outputHypernetworksMlps[i],
        take,
        'mask_decoder.output_hypernetworks_mlps.$i',
      );
    }

    // IoU prediction head.
    _loadSamMlp(m.iouPredictionHead, take, 'mask_decoder.iou_prediction_head');
  }

  static void _loadSamAttention(
    SamAttention a,
    Tensor Function(String) take,
    String prefix,
  ) {
    _copy(
      a.qProj.weight,
      _expectShape(take('$prefix.q_proj.weight'), [
        a.internalDim,
        a.embedDim,
      ], '$prefix.q_proj.weight'),
    );
    _copy(
      a.qProj.bias!,
      _reshapeVectorTo1xN(
        _expectShape(take('$prefix.q_proj.bias'), [
          a.internalDim,
        ], '$prefix.q_proj.bias'),
      ),
    );
    _copy(
      a.kProj.weight,
      _expectShape(take('$prefix.k_proj.weight'), [
        a.internalDim,
        a.embedDim,
      ], '$prefix.k_proj.weight'),
    );
    _copy(
      a.kProj.bias!,
      _reshapeVectorTo1xN(
        _expectShape(take('$prefix.k_proj.bias'), [
          a.internalDim,
        ], '$prefix.k_proj.bias'),
      ),
    );
    _copy(
      a.vProj.weight,
      _expectShape(take('$prefix.v_proj.weight'), [
        a.internalDim,
        a.embedDim,
      ], '$prefix.v_proj.weight'),
    );
    _copy(
      a.vProj.bias!,
      _reshapeVectorTo1xN(
        _expectShape(take('$prefix.v_proj.bias'), [
          a.internalDim,
        ], '$prefix.v_proj.bias'),
      ),
    );
    _copy(
      a.outProj.weight,
      _expectShape(take('$prefix.out_proj.weight'), [
        a.embedDim,
        a.internalDim,
      ], '$prefix.out_proj.weight'),
    );
    _copy(
      a.outProj.bias!,
      _reshapeVectorTo1xN(
        _expectShape(take('$prefix.out_proj.bias'), [
          a.embedDim,
        ], '$prefix.out_proj.bias'),
      ),
    );
  }

  static void _loadSamMlpBlock(
    SamMlpBlock m,
    Tensor Function(String) take,
    String prefix,
  ) {
    _copy(
      m.fc1.weight,
      _expectShape(
        take('$prefix.lin1.weight'),
        m.fc1.weight.shape,
        '$prefix.lin1.weight',
      ),
    );
    _copy(
      m.fc1.bias!,
      _reshapeVectorTo1xN(
        _expectShape(take('$prefix.lin1.bias'), [
          m.fc1.weight.shape[0],
        ], '$prefix.lin1.bias'),
      ),
    );
    _copy(
      m.fc2.weight,
      _expectShape(
        take('$prefix.lin2.weight'),
        m.fc2.weight.shape,
        '$prefix.lin2.weight',
      ),
    );
    _copy(
      m.fc2.bias!,
      _reshapeVectorTo1xN(
        _expectShape(take('$prefix.lin2.bias'), [
          m.fc2.weight.shape[0],
        ], '$prefix.lin2.bias'),
      ),
    );
  }

  static void _loadSamMlp(
    SamMlp m,
    Tensor Function(String) take,
    String prefix,
  ) {
    for (int i = 0; i < m.layers.length; i++) {
      _copy(
        m.layers[i].weight,
        _expectShape(
          take('$prefix.layers.$i.weight'),
          m.layers[i].weight.shape,
          '$prefix.layers.$i.weight',
        ),
      );
      _copy(
        m.layers[i].bias!,
        _reshapeVectorTo1xN(
          _expectShape(take('$prefix.layers.$i.bias'), [
            m.layers[i].weight.shape[0],
          ], '$prefix.layers.$i.bias'),
        ),
      );
    }
  }

  // -------------------- tensor helpers --------------------

  static Tensor _expectShape(Tensor t, List<int> expected, String name) {
    if (t.shape.length != expected.length || !_shapesEqual(t.shape, expected)) {
      throw ArgumentError(
        'sam loader: "$name" expected shape $expected, got ${t.shape}',
      );
    }
    return t;
  }

  static bool _shapesEqual(List<int> a, List<int> b) {
    for (int i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }

  static void _copy(Tensor dst, Tensor src) {
    if (dst.length != src.length) {
      throw ArgumentError(
        'sam loader: copy length mismatch — dst=${dst.shape} '
        '(${dst.length}), src=${src.shape} (${src.length})',
      );
    }
    final vals = src.toList();
    final matched = Tensor.fromList(dst.shape, vals, device: dst.device);
    dst.assign(matched);
  }

  static Tensor _sliceRows(Tensor t, int start, int end) =>
      t.sliceRows(start, end);

  static Tensor _slice1DVector(Tensor t, int start, int end) {
    final data = t.toList();
    final n = end - start;
    final out = Float32List(n);
    for (int i = 0; i < n; i++) {
      out[i] = data[start + i];
    }
    return Tensor.fromList([n], out, device: Device.CPU);
  }

  static Tensor _reshapeVectorTo1xN(Tensor v) {
    if (v.shape.length != 1) {
      throw ArgumentError(
        'sam loader: expected rank 1 for bias vector, got ${v.shape}',
      );
    }
    return Tensor.fromList([1, v.shape[0]], v.toList(), device: Device.CPU);
  }

  /// Broadcast a `[2·ws-1, headDim]` shared rel-pos table across the
  /// `[numHeads, 2·ws-1, headDim]` per-head storage we use.
  static void _copyBroadcastPerHead(Tensor dst, Tensor src, int numHeads) {
    // dst: [numHeads, 2*ws-1, headDim]  src: [2*ws-1, headDim]
    final srcVals = src.toList();
    final total = numHeads * srcVals.length;
    final buf = List<double>.filled(total, 0);
    for (int h = 0; h < numHeads; h++) {
      for (int i = 0; i < srcVals.length; i++) {
        buf[h * srcVals.length + i] = srcVals[i];
      }
    }
    final matched = Tensor.fromList(dst.shape, buf, device: dst.device);
    dst.assign(matched);
  }
}
