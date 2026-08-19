/// Loader for `facebook/dinov2-*` safetensors into [DinoV2Backbone].
///
/// Handles the important quirks:
///
///   * Patch projection: HF stores a Conv2d weight `[D, 3, P, P]`.
///     We reshape it into `[D, 3*P*P]` (already in the row-major order
///     our Linear expects) and copy it in with the associated bias.
///   * Position embeddings: checkpoint has `[1, 1370, D]` for
///     `image_size=518` (37×37 patches). When the target grid is
///     smaller we bilinearly resize the spatial portion and keep the
///     CLS-token row unchanged.
///   * Per-block `layer_scale{1,2}.lambda1` → block `layerScale{1,2}`.
///   * QKV split: HF stores separate `query.weight / key.weight /
///     value.weight` `[D, D]`; our `MultiHeadAttention` expects one
///     `Linear(D, headDim)` per head, so we row-split by head chunks
///     (same trick used by `bert_hf_loader` + `whisper_hf_loader`).
library;

import 'dart:math' as math;
import 'dart:typed_data';

import '../../tensor/tensor.dart';
import '../linear.dart';
import '../safetensors.dart';
import 'dinov2.dart';

class DinoV2LoadReport {
  final int consumedCount;
  final List<String> unusedKeys;
  const DinoV2LoadReport({
    required this.consumedCount,
    required this.unusedKeys,
  });

  @override
  String toString() =>
      'DinoV2LoadReport(consumed=$consumedCount, unused=${unusedKeys.length})';
}

class DinoV2Loader {
  static DinoV2LoadReport loadFile(DinoV2Backbone model, String path) {
    final state = SafeTensors.loadFile(path);
    return loadMap(model, state);
  }

  static DinoV2LoadReport loadMap(
    DinoV2Backbone model,
    Map<String, Tensor> state,
  ) {
    final consumed = <String>{};
    Tensor take(String name) {
      final t = state[name];
      if (t == null) {
        throw ArgumentError('dinov2 loader: missing "$name"');
      }
      consumed.add(name);
      return t;
    }

    // ---- CLS token ----
    final cls = take('embeddings.cls_token');
    // stored as [1, 1, D]; we hold [1, D].
    _assign1d(model.clsToken, cls);

    // ---- patch embed (Conv2d [D, C, P, P] -> Linear [D, C*P*P]) ----
    final peW = take('embeddings.patch_embeddings.projection.weight');
    final peB = take('embeddings.patch_embeddings.projection.bias');
    _loadPatchEmbed(model.patchProjection, peW, peB);

    // ---- positional embeddings (with grid-size interpolation) ----
    final pos = take('embeddings.position_embeddings');
    _loadPositionEmbeddings(model.positionEmbeddings, pos);

    // ---- per-block ----
    final headDim = model.embedDim ~/ model.numHeads;
    for (int i = 0; i < model.numLayers; i++) {
      final blk = model.blocks[i];
      final p = 'encoder.layer.$i';

      // norms
      _loadLayerNorm(
        blk.norm1,
        weight: take('$p.norm1.weight'),
        bias: take('$p.norm1.bias'),
      );
      _loadLayerNorm(
        blk.norm2,
        weight: take('$p.norm2.weight'),
        bias: take('$p.norm2.bias'),
      );

      // attention (per-head split)
      final qW = take('$p.attention.attention.query.weight');
      final qB = take('$p.attention.attention.query.bias');
      final kW = take('$p.attention.attention.key.weight');
      final kB = take('$p.attention.attention.key.bias');
      final vW = take('$p.attention.attention.value.weight');
      final vB = take('$p.attention.attention.value.bias');
      final outW = take('$p.attention.output.dense.weight');
      final outB = take('$p.attention.output.dense.bias');

      for (int h = 0; h < model.numHeads; h++) {
        _sliceHead(blk.attn.wq[h], qW, qB, h, headDim);
        _sliceHead(blk.attn.wk[h], kW, kB, h, headDim);
        _sliceHead(blk.attn.wv[h], vW, vB, h, headDim);
      }
      _assign2dPlain(blk.attn.wo.weight, outW);
      _assignLinearBias(blk.attn.wo, outB);

      // layer_scale
      _assign1d(blk.layerScale1, take('$p.layer_scale1.lambda1'));
      _assign1d(blk.layerScale2, take('$p.layer_scale2.lambda1'));

      // MLP
      _assign2dPlain(blk.mlp1.weight, take('$p.mlp.fc1.weight'));
      _assignLinearBias(blk.mlp1, take('$p.mlp.fc1.bias'));
      _assign2dPlain(blk.mlp2.weight, take('$p.mlp.fc2.weight'));
      _assignLinearBias(blk.mlp2, take('$p.mlp.fc2.bias'));
    }

    // ---- final LN ----
    _loadLayerNorm(
      model.norm,
      weight: take('layernorm.weight'),
      bias: take('layernorm.bias'),
    );

    final unused =
        state.keys
            .where(
              (k) =>
                  !consumed.contains(k) &&
                  k != 'embeddings.mask_token' /* DINOv2 pretrain-only */,
            )
            .toList()
          ..sort();
    return DinoV2LoadReport(consumedCount: consumed.length, unusedKeys: unused);
  }

  // ---------------- helpers ----------------

  static void _loadPatchEmbed(Linear proj, Tensor conv, Tensor bias) {
    // conv is [D, 3, P, P]; our Linear is [D, 3*P*P].
    final vals = conv.toList();
    final matched = Tensor.fromList(
      proj.weight.shape,
      vals,
      device: proj.weight.device,
      requiresGrad: proj.weight.requiresGrad,
    );
    proj.weight.assign(matched);
    _assignLinearBias(proj, bias);
  }

  static void _loadPositionEmbeddings(Tensor dst, Tensor src) {
    // src: [1, srcTokens, D]  (srcTokens = 1 + srcGrid²)
    // dst: [dstTokens, D]     (dstTokens = 1 + dstGrid²)
    final srcTokens = src.shape[1];
    final d = src.shape[2];
    if (dst.shape[0] == srcTokens && dst.shape[1] == d) {
      final vals = src.toList();
      final matched = Tensor.fromList(dst.shape, vals, device: dst.device);
      dst.assign(matched);
      return;
    }
    final srcGrid = _isqrt(srcTokens - 1);
    if (srcGrid * srcGrid != srcTokens - 1) {
      throw StateError(
        'dinov2 loader: position_embeddings has non-square patch grid '
        '(srcTokens=$srcTokens)',
      );
    }
    final dstTokens = dst.shape[0];
    final dstGrid = _isqrt(dstTokens - 1);
    if (dstGrid * dstGrid != dstTokens - 1) {
      throw StateError(
        'dinov2 loader: target position grid non-square (dstTokens=$dstTokens)',
      );
    }
    final data = src.toList();
    // Split CLS row + patch grid.
    final clsRow = Float32List(d);
    for (int i = 0; i < d; i++) {
      clsRow[i] = data[i];
    }
    // Reshape patch rows into [srcGrid, srcGrid, D].
    // src layout: token index 1 + y*srcGrid + x  (assumed row-major).
    // Bilinear interpolate to [dstGrid, dstGrid, D] then flatten.
    final resized = _bilinearGrid(
      Float32List.fromList(data.sublist(d)),
      srcGrid,
      srcGrid,
      dstGrid,
      dstGrid,
      d,
    );
    final out = Float32List(dstTokens * d);
    for (int i = 0; i < d; i++) {
      out[i] = clsRow[i];
    }
    for (int i = 0; i < dstGrid * dstGrid * d; i++) {
      out[d + i] = resized[i];
    }
    final tOut = Tensor.fromFloat32List(
      [dstTokens, d],
      out,
      device: dst.device,
    );
    dst.assign(tOut);
  }

  /// Bilinear resize a `[H*W*D]` flat grid (D = channels).
  static Float32List _bilinearGrid(
    Float32List src,
    int inH,
    int inW,
    int outH,
    int outW,
    int d,
  ) {
    final out = Float32List(outH * outW * d);
    for (int y = 0; y < outH; y++) {
      final srcY = (y + 0.5) * inH / outH - 0.5;
      final y0 = srcY.floor().clamp(0, inH - 1);
      final y1 = (y0 + 1).clamp(0, inH - 1);
      final wy = (srcY - y0).clamp(0.0, 1.0);
      for (int x = 0; x < outW; x++) {
        final srcX = (x + 0.5) * inW / outW - 0.5;
        final x0 = srcX.floor().clamp(0, inW - 1);
        final x1 = (x0 + 1).clamp(0, inW - 1);
        final wx = (srcX - x0).clamp(0.0, 1.0);
        final off00 = (y0 * inW + x0) * d;
        final off01 = (y0 * inW + x1) * d;
        final off10 = (y1 * inW + x0) * d;
        final off11 = (y1 * inW + x1) * d;
        final dst = (y * outW + x) * d;
        for (int c = 0; c < d; c++) {
          final v00 = src[off00 + c];
          final v01 = src[off01 + c];
          final v10 = src[off10 + c];
          final v11 = src[off11 + c];
          final v0 = v00 * (1 - wx) + v01 * wx;
          final v1 = v10 * (1 - wx) + v11 * wx;
          out[dst + c] = v0 * (1 - wy) + v1 * wy;
        }
      }
    }
    return out;
  }

  static void _sliceHead(
    Linear headLinear,
    Tensor fusedW,
    Tensor fusedB,
    int headIdx,
    int headDim,
  ) {
    // fusedW: [D, D]; fusedB: [D]. Rows [h*headDim, (h+1)*headDim)
    // go to head `h`. Bias is sliced identically.
    final wRows = fusedW.toList();
    final bAll = fusedB.toList();
    final embedDim = fusedW.shape[1];
    final rowStart = headIdx * headDim;
    final wChunk = Float32List(headDim * embedDim);
    for (int r = 0; r < headDim; r++) {
      for (int c = 0; c < embedDim; c++) {
        wChunk[r * embedDim + c] = wRows[(rowStart + r) * embedDim + c];
      }
    }
    final bChunk = Float32List(headDim);
    for (int r = 0; r < headDim; r++) {
      bChunk[r] = bAll[rowStart + r];
    }
    _assign2dPlain(
      headLinear.weight,
      Tensor.fromFloat32List([headDim, embedDim], wChunk),
    );
    if (headLinear.bias != null) {
      final wideB = Float32List(headDim);
      for (int i = 0; i < headDim; i++) {
        wideB[i] = bChunk[i];
      }
      final bT = Tensor.fromFloat32List(
        headLinear.bias!.shape,
        wideB,
        device: headLinear.bias!.device,
        requiresGrad: headLinear.bias!.requiresGrad,
      );
      headLinear.bias!.assign(bT);
    }
  }

  static void _loadLayerNorm(
    dynamic ln, {
    required Tensor weight,
    required Tensor bias,
  }) {
    _assign1d(ln.gamma, weight);
    _assign1d(ln.beta, bias);
  }

  static void _assign1d(Tensor dst, Tensor src) {
    if (dst.length != src.length) {
      throw ArgumentError(
        'dinov2 loader: length mismatch dst=${dst.shape} src=${src.shape}',
      );
    }
    final vals = src.toList();
    final t = Tensor.fromList(dst.shape, vals, device: dst.device);
    dst.assign(t);
  }

  static void _assign2dPlain(Tensor dst, Tensor src) {
    if (dst.length != src.length) {
      throw ArgumentError(
        'dinov2 loader: assign2d length mismatch '
        'dst=${dst.shape} src=${src.shape}',
      );
    }
    final vals = src.toList();
    final t = Tensor.fromList(dst.shape, vals, device: dst.device);
    dst.assign(t);
  }

  static void _assignLinearBias(Linear lin, Tensor src) {
    // Our Linear bias is [1, outF]; source is [outF].
    if (lin.bias == null) {
      throw StateError('dinov2 loader: bias assign into no-bias Linear');
    }
    final vals = src.toList();
    final wide = Float32List(vals.length);
    for (int i = 0; i < vals.length; i++) {
      wide[i] = vals[i];
    }
    final t = Tensor.fromFloat32List(
      lin.bias!.shape,
      wide,
      device: lin.bias!.device,
      requiresGrad: lin.bias!.requiresGrad,
    );
    lin.bias!.assign(t);
  }

  static int _isqrt(int n) {
    if (n < 0) return -1;
    var x = math.sqrt(n).round();
    // Guard against fp jitter.
    while (x * x > n) {
      x--;
    }
    while ((x + 1) * (x + 1) <= n) {
      x++;
    }
    return x;
  }
}
