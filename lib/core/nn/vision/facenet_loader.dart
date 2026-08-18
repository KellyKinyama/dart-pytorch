/// Loader for `facenet-pytorch`'s `InceptionResnetV1(pretrained='vggface2')`
/// converted to safetensors via [`scripts/convert_facenet_pt_to_safetensors.py`].
///
/// Every conv is folded with the following BN at load time
/// (`conv_bn_fold.dart`, eps = 1e-3) so at runtime the module is a
/// pure Conv2d + ReLU + concat + pool + Linear stack. `last_linear`
/// stays trainable; the folded conv weights also stay `requiresGrad`
/// (see the caveat about backpropagating *through* Conv2d in
/// `doc/facenet.md`). `last_bn` is affine (γ, β learned) and is
/// folded into `lastBnScale` / `lastBnOffset`.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import '../../tensor/tensor.dart';
import '../conv2d.dart';
import '../safetensors.dart';
import 'conv_bn_fold.dart';
import 'facenet.dart';

class FaceNetLoadReport {
  final int consumedCount;
  final List<String> unusedKeys;
  const FaceNetLoadReport({
    required this.consumedCount,
    required this.unusedKeys,
  });

  @override
  String toString() =>
      'FaceNetLoadReport(consumed=$consumedCount, unused=${unusedKeys.length})';
}

class FaceNetLoader {
  /// BN epsilon used across facenet-pytorch's InceptionResnetV1
  /// (both spatial BN and the final 1D BN on the embedding).
  static const double bnEps = 1e-3;

  static FaceNetLoadReport loadFile(
    InceptionResnetV1 model,
    String path, {
    bool keepFp16 = false,
  }) {
    final state = SafeTensors.loadFile(path, keepFp16: keepFp16);
    return loadMap(model, state);
  }

  static FaceNetLoadReport loadMap(
    InceptionResnetV1 model,
    Map<String, Tensor> state,
  ) {
    final consumed = <String>{};

    void loadBasic(BasicConv2d bc, String prefix) {
      final w = _take(state, consumed, '$prefix.conv.weight');
      final g = _take(state, consumed, '$prefix.bn.weight');
      final b = _take(state, consumed, '$prefix.bn.bias');
      final m = _take(state, consumed, '$prefix.bn.running_mean');
      final v = _take(state, consumed, '$prefix.bn.running_var');
      loadConvBnFolded(
        bc.conv,
        convW: _toF32(w),
        bnGamma: _toF32(g),
        bnBeta: _toF32(b),
        bnMean: _toF32(m),
        bnVar: _toF32(v),
        bnEps: bnEps,
      );
    }

    void loadPlainConv(Conv2d conv, String prefix) {
      // Bias-carrying 1×1 up-conv inside each Block35/17/8 residual.
      final w = _take(state, consumed, '$prefix.weight');
      final b = _take(state, consumed, '$prefix.bias');
      final wF = _toF32(w);
      final bF = _toF32(b);
      // Assign directly (no BN to fold).
      final wT = Tensor.fromFloat32List(
        conv.weight.shape,
        wF,
        device: conv.weight.device,
        requiresGrad: conv.weight.requiresGrad,
      );
      conv.weight.assign(wT);
      final bT = Tensor.fromFloat32List(
        conv.bias!.shape,
        bF,
        device: conv.bias!.device,
        requiresGrad: conv.bias!.requiresGrad,
      );
      conv.bias!.assign(bT);
    }

    // ---------- stem ----------
    loadBasic(model.conv2d1a, 'conv2d_1a');
    loadBasic(model.conv2d2a, 'conv2d_2a');
    loadBasic(model.conv2d2b, 'conv2d_2b');
    loadBasic(model.conv2d3b, 'conv2d_3b');
    loadBasic(model.conv2d4a, 'conv2d_4a');
    loadBasic(model.conv2d4b, 'conv2d_4b');

    // ---------- repeat_1: 5x Block35 ----------
    for (int i = 0; i < model.repeat1.length; i++) {
      final b = model.repeat1[i];
      final p = 'repeat_1.$i';
      loadBasic(b.branch0, '$p.branch0');
      loadBasic(b.branch1a, '$p.branch1.0');
      loadBasic(b.branch1b, '$p.branch1.1');
      loadBasic(b.branch2a, '$p.branch2.0');
      loadBasic(b.branch2b, '$p.branch2.1');
      loadBasic(b.branch2c, '$p.branch2.2');
      loadPlainConv(b.conv, '$p.conv2d');
    }

    // ---------- mixed_6a ----------
    loadBasic(model.mixed6a.branch0, 'mixed_6a.branch0');
    loadBasic(model.mixed6a.branch1a, 'mixed_6a.branch1.0');
    loadBasic(model.mixed6a.branch1b, 'mixed_6a.branch1.1');
    loadBasic(model.mixed6a.branch1c, 'mixed_6a.branch1.2');

    // ---------- repeat_2: 10x Block17 ----------
    for (int i = 0; i < model.repeat2.length; i++) {
      final b = model.repeat2[i];
      final p = 'repeat_2.$i';
      loadBasic(b.branch0, '$p.branch0');
      loadBasic(b.branch1a, '$p.branch1.0');
      loadBasic(b.branch1b, '$p.branch1.1');
      loadBasic(b.branch1c, '$p.branch1.2');
      loadPlainConv(b.conv, '$p.conv2d');
    }

    // ---------- mixed_7a ----------
    loadBasic(model.mixed7a.branch0a, 'mixed_7a.branch0.0');
    loadBasic(model.mixed7a.branch0b, 'mixed_7a.branch0.1');
    loadBasic(model.mixed7a.branch1a, 'mixed_7a.branch1.0');
    loadBasic(model.mixed7a.branch1b, 'mixed_7a.branch1.1');
    loadBasic(model.mixed7a.branch2a, 'mixed_7a.branch2.0');
    loadBasic(model.mixed7a.branch2b, 'mixed_7a.branch2.1');
    loadBasic(model.mixed7a.branch2c, 'mixed_7a.branch2.2');

    // ---------- repeat_3: 5x Block8 ----------
    for (int i = 0; i < model.repeat3.length; i++) {
      final b = model.repeat3[i];
      final p = 'repeat_3.$i';
      loadBasic(b.branch0, '$p.branch0');
      loadBasic(b.branch1a, '$p.branch1.0');
      loadBasic(b.branch1b, '$p.branch1.1');
      loadBasic(b.branch1c, '$p.branch1.2');
      loadPlainConv(b.conv, '$p.conv2d');
    }

    // ---------- final Block8 (no ReLU) ----------
    loadBasic(model.block8Final.branch0, 'block8.branch0');
    loadBasic(model.block8Final.branch1a, 'block8.branch1.0');
    loadBasic(model.block8Final.branch1b, 'block8.branch1.1');
    loadBasic(model.block8Final.branch1c, 'block8.branch1.2');
    loadPlainConv(model.block8Final.conv, 'block8.conv2d');

    // ---------- last_linear ----------
    final lw = _take(state, consumed, 'last_linear.weight');
    _assign(model.lastLinear.weight, lw);

    // ---------- last_bn (affine=True) folded into scale/offset ----------
    final bnW = _take(state, consumed, 'last_bn.weight'); // γ
    final bnB = _take(state, consumed, 'last_bn.bias'); // β
    final bnM = _take(state, consumed, 'last_bn.running_mean'); // μ
    final bnV = _take(state, consumed, 'last_bn.running_var'); // σ²
    _foldLastBn(
      gamma: _toF32(bnW),
      beta: _toF32(bnB),
      mean: _toF32(bnM),
      variance: _toF32(bnV),
      eps: bnEps,
      scaleDst: model.lastBnScale,
      offsetDst: model.lastBnOffset,
    );

    final unused = state.keys.where((k) => !consumed.contains(k)).toList()
      ..sort();
    return FaceNetLoadReport(
      consumedCount: consumed.length,
      unusedKeys: unused,
    );
  }

  // ---------------- helpers ----------------

  static Tensor _take(
    Map<String, Tensor> state,
    Set<String> consumed,
    String name,
  ) {
    final t = state[name];
    if (t == null) {
      throw ArgumentError('facenet loader: missing tensor "$name"');
    }
    consumed.add(name);
    return t;
  }

  static Float32List _toF32(Tensor t) {
    final data = t.toList();
    final out = Float32List(data.length);
    for (int i = 0; i < data.length; i++) {
      out[i] = data[i];
    }
    return out;
  }

  static void _assign(Tensor dst, Tensor src) {
    if (dst.length != src.length) {
      throw ArgumentError(
        'facenet loader: assign length mismatch — dst=${dst.shape}, '
        'src=${src.shape}',
      );
    }
    final vals = src.toList();
    final matched = Tensor.fromList(dst.shape, vals, device: dst.device);
    dst.assign(matched);
  }

  /// Convert `BatchNorm1d(affine=True)` running stats + affine into
  /// per-dim `scale` and `offset` on the 512-d embedding:
  ///   y = γ (x - μ) / √(σ² + ε) + β
  ///     = scale · x + offset
  /// where `scale = γ / √(σ² + ε)` and `offset = β − μ · scale`.
  static void _foldLastBn({
    required Float32List gamma,
    required Float32List beta,
    required Float32List mean,
    required Float32List variance,
    required double eps,
    required Tensor scaleDst,
    required Tensor offsetDst,
  }) {
    final n = gamma.length;
    if (beta.length != n || mean.length != n || variance.length != n) {
      throw ArgumentError('foldLastBn: length mismatch');
    }
    final scale = Float32List(n);
    final offset = Float32List(n);
    for (int i = 0; i < n; i++) {
      final s = gamma[i] / math.sqrt(variance[i] + eps);
      scale[i] = s;
      offset[i] = beta[i] - mean[i] * s;
    }
    _assignFlat(scaleDst, scale);
    _assignFlat(offsetDst, offset);
  }

  static void _assignFlat(Tensor dst, Float32List src) {
    final t = Tensor.fromFloat32List(dst.shape, src, device: dst.device);
    dst.assign(t);
  }
}
