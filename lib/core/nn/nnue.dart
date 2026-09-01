/// Stockfish NNUE forward pass (float port).
///
/// Assembles the raw quantised weights from `nnue_proto.dart` and the
/// sparse feature indices from `nnue_input.dart` into a centipawn
/// score. Everything is dequantised to `double` on load and the
/// forward pass is float-only — trades a couple of cp of accuracy vs
/// Stockfish's SIMD int8 kernels for a much shorter porting path.
///
/// Architecture (determined empirically on `nn-5af11540bbfe.nnue` —
/// see `nnue_proto.dart` for the byte-layout derivation):
///
///   * Feature transformer: 22528 sparse inputs per POV → ftDim=1536
///     (int16 accumulator + int32 PSQT accumulator)
///   * Element-wise combine of the two POV accumulators via
///     `combined[i] = clip(acc[stm][i]) * clip(acc[!stm][i])`
///     (Stockfish's SFNNv5-style "SqrClippedReLU on pair-product"
///     trick that halves L1 input width vs concat)
///   * L1: 1536 → 16
///   * Split L1 output into `[ClippedReLU(L1) | SqrClippedReLU(L1)]`
///     yielding a 32-dim vector
///   * L2: 32 → 32 (ClippedReLU)
///   * L3: 32 → 1
///   * Add PSQT contribution and scale to centipawns
///
/// Reference: `Stockfish/src/nnue/{network,layers/*}.h`.
library;

import 'dart:typed_data';

import 'nnue_input.dart';
import 'nnue_proto.dart';

/// Detailed result of a single evaluation. `cp` is the primary output;
/// the other fields expose intermediates so tests / demos can compare
/// against Stockfish's `d` or `eval` command breakdowns.
class NnueEvalResult {
  const NnueEvalResult({
    required this.cp,
    required this.bucket,
    required this.psqtContribution,
    required this.rawL3Output,
  });

  /// Centipawn score from the side-to-move's POV. Positive = STM better.
  final double cp;

  /// Which of the 8 material buckets was selected.
  final int bucket;

  /// PSQT-only contribution to the final cp (SF splits its eval into
  /// "positional" (network) and "material" (PSQT) components).
  final double psqtContribution;

  /// Raw L3 output before adding the PSQT contribution and scaling.
  final double rawL3Output;

  @override
  String toString() =>
      'NnueEvalResult(cp=${cp.toStringAsFixed(1)}, bucket=$bucket, '
      'psqt=${psqtContribution.toStringAsFixed(1)}, '
      'l3=${rawL3Output.toStringAsFixed(1)})';
}

/// Dequantised, evaluation-ready copy of an [NnueRaw] file.
class NnueNet {
  NnueNet._({
    required this.ftDim,
    required this.numInputs,
    required this.psqtBuckets,
    required this.ftBiases,
    required this.ftWeights,
    required this.ftPsqt,
    required this.buckets,
    required this.raw,
  });

  final int ftDim;
  final int numInputs;
  final int psqtBuckets;

  final Float32List ftBiases; // [ftDim]
  final Float32List ftWeights; // [numInputs, ftDim] row-major
  final Float32List ftPsqt; // [numInputs, psqtBuckets] row-major

  final List<_BucketFloat> buckets;

  /// Raw quantised weights kept alongside the floats so [evaluateInt]
  /// can run Stockfish's exact int8×int8 arithmetic without a second
  /// file parse.
  final NnueRaw raw;

  /// Dequantise the raw file into floats. See top-of-file for the
  /// scale factors used; they follow Stockfish's canonical quant
  /// scheme (`FT_QUANT = 64` for the transformer, `LayerScale = 64`
  /// for hidden layers).
  factory NnueNet.fromRaw(NnueRaw raw) {
    final ft = raw.featureTransformer;
    final ftBiases = Float32List(ft.ftDim);
    for (int i = 0; i < ft.ftDim; i++) {
      ftBiases[i] = ft.biasesI16[i] / _kFtQuant;
    }
    final ftWeights = Float32List(ft.numInputs * ft.ftDim);
    for (int i = 0; i < ftWeights.length; i++) {
      ftWeights[i] = ft.weightsI16[i] / _kFtQuant;
    }
    final ftPsqt = Float32List(ft.numInputs * ft.psqtBuckets);
    for (int i = 0; i < ftPsqt.length; i++) {
      ftPsqt[i] = ft.psqtI32[i] / _kPsqtScale;
    }

    final buckets = <_BucketFloat>[];
    for (final rb in raw.network.buckets) {
      final l1B = Float32List(rb.l1BiasesI32.length);
      for (int i = 0; i < l1B.length; i++) {
        l1B[i] = rb.l1BiasesI32[i] / _kBiasScale;
      }
      final l1W = Float32List(rb.l1WeightsI8.length);
      for (int i = 0; i < l1W.length; i++) {
        l1W[i] = rb.l1WeightsI8[i] / _kWeightScale;
      }
      final l2B = Float32List(rb.l2BiasesI32.length);
      for (int i = 0; i < l2B.length; i++) {
        l2B[i] = rb.l2BiasesI32[i] / _kBiasScale;
      }
      final l2W = Float32List(rb.l2WeightsI8.length);
      for (int i = 0; i < l2W.length; i++) {
        l2W[i] = rb.l2WeightsI8[i] / _kWeightScale;
      }
      final l3B = Float32List(rb.l3BiasesI32.length);
      for (int i = 0; i < l3B.length; i++) {
        l3B[i] = rb.l3BiasesI32[i] / _kBiasScale;
      }
      final l3W = Float32List(rb.l3WeightsI8.length);
      for (int i = 0; i < l3W.length; i++) {
        l3W[i] = rb.l3WeightsI8[i] / _kWeightScale;
      }
      buckets.add(
        _BucketFloat(
          l1Biases: l1B,
          l1Weights: l1W,
          l2Biases: l2B,
          l2Weights: l2W,
          l3Biases: l3B,
          l3Weights: l3W,
        ),
      );
    }

    return NnueNet._(
      ftDim: ft.ftDim,
      numInputs: ft.numInputs,
      psqtBuckets: ft.psqtBuckets,
      ftBiases: ftBiases,
      ftWeights: ftWeights,
      ftPsqt: ftPsqt,
      buckets: buckets,
      raw: raw,
    );
  }

  /// Evaluate a position given its pre-computed sparse features.
  NnueEvalResult evaluate(NnueFeatures features) {
    final accStm = Float32List.fromList(ftBiases);
    final accNstm = Float32List.fromList(ftBiases);

    final stmActive = features.stm == NnuePerspective.white
        ? features.whiteActive
        : features.blackActive;
    final nstmActive = features.stm == NnuePerspective.white
        ? features.blackActive
        : features.whiteActive;

    for (final idx in stmActive) {
      final rowBase = idx * ftDim;
      for (int j = 0; j < ftDim; j++) {
        accStm[j] += ftWeights[rowBase + j];
      }
    }
    for (final idx in nstmActive) {
      final rowBase = idx * ftDim;
      for (int j = 0; j < ftDim; j++) {
        accNstm[j] += ftWeights[rowBase + j];
      }
    }

    final bucket = ((features.pieceCount - 1) >> 2).clamp(0, psqtBuckets - 1);
    double psqtStm = 0.0;
    double psqtNstm = 0.0;
    for (final idx in stmActive) {
      psqtStm += ftPsqt[idx * psqtBuckets + bucket];
    }
    for (final idx in nstmActive) {
      psqtNstm += ftPsqt[idx * psqtBuckets + bucket];
    }
    final psqtDiff = (psqtStm - psqtNstm) * 0.5;

    // Element-wise combine POVs (SFNNv5-style). Each accumulator entry
    // is clipped to [0, 127/FT_QUANT] then paired with the other POV's
    // entry via multiplication — the same "SqrClippedReLU on pair-
    // product" trick Stockfish uses to halve the L1 input width vs a
    // concat.
    final combined = Float32List(ftDim);
    for (int i = 0; i < ftDim; i++) {
      final a = accStm[i] < 0
          ? 0.0
          : (accStm[i] > _kFtClip ? _kFtClip : accStm[i]);
      final b = accNstm[i] < 0
          ? 0.0
          : (accNstm[i] > _kFtClip ? _kFtClip : accNstm[i]);
      combined[i] = a * b / _kFtClip;
    }

    // L1: [ftDim] → [16].
    final bk = buckets[bucket];
    final l1Out = Float32List(16);
    for (int j = 0; j < 16; j++) {
      double s = bk.l1Biases[j];
      final rowBase = j * ftDim;
      for (int i = 0; i < ftDim; i++) {
        s += bk.l1Weights[rowBase + i] * combined[i];
      }
      l1Out[j] = s;
    }

    // Split into ClippedReLU + SqrClippedReLU branches, concat → 32.
    final l2In = Float32List(32);
    for (int j = 0; j < 16; j++) {
      final c = _clipReLU(l1Out[j]);
      l2In[j] = c;
      l2In[j + 16] = c * c;
    }

    final l2Out = Float32List(32);
    for (int j = 0; j < 32; j++) {
      double s = bk.l2Biases[j];
      final rowBase = j * 32;
      for (int i = 0; i < 32; i++) {
        s += bk.l2Weights[rowBase + i] * l2In[i];
      }
      l2Out[j] = _clipReLU(s);
    }

    double l3 = bk.l3Biases[0];
    for (int i = 0; i < 32; i++) {
      l3 += bk.l3Weights[i] * l2Out[i];
    }

    final cp = (l3 + psqtDiff) * _kOutputScale;
    return NnueEvalResult(
      cp: cp,
      bucket: bucket,
      psqtContribution: psqtDiff * _kOutputScale,
      rawL3Output: l3 * _kOutputScale,
    );
  }

  /// Stockfish-matched integer forward pass.
  ///
  /// Mirrors Stockfish's SIMD kernels bit-for-bit: int16 FT accumulator,
  /// int8 hidden weights, int32 biases, shifts by `WeightScaleBits = 6`
  /// between layers, ClippedReLU / SqrClippedReLU on uint8 outputs. The
  /// returned cp is the **raw network + PSQT output** in centipawn
  /// units. Stockfish's UCI cp is further scaled by phase, material
  /// drawishness, optimism and contempt — none of which are part of
  /// NNUE itself — so this value can be 2–5× larger than what a live
  /// Stockfish `d` command would print for the same position.
  NnueEvalResult evaluateInt(NnueFeatures features) {
    final ft = raw.featureTransformer;
    final ftDim = ft.ftDim;

    final accStm = Int32List(ftDim);
    final accNstm = Int32List(ftDim);
    for (int i = 0; i < ftDim; i++) {
      accStm[i] = ft.biasesI16[i];
      accNstm[i] = ft.biasesI16[i];
    }

    final stmActive = features.stm == NnuePerspective.white
        ? features.whiteActive
        : features.blackActive;
    final nstmActive = features.stm == NnuePerspective.white
        ? features.blackActive
        : features.whiteActive;

    for (final idx in stmActive) {
      final rowBase = idx * ftDim;
      for (int j = 0; j < ftDim; j++) {
        accStm[j] += ft.weightsI16[rowBase + j];
      }
    }
    for (final idx in nstmActive) {
      final rowBase = idx * ftDim;
      for (int j = 0; j < ftDim; j++) {
        accNstm[j] += ft.weightsI16[rowBase + j];
      }
    }

    final bucket = ((features.pieceCount - 1) >> 2).clamp(0, psqtBuckets - 1);
    int psqtStm = 0;
    int psqtNstm = 0;
    for (final idx in stmActive) {
      psqtStm += ft.psqtI32[idx * psqtBuckets + bucket];
    }
    for (final idx in nstmActive) {
      psqtNstm += ft.psqtI32[idx * psqtBuckets + bucket];
    }
    final psqtRaw = (psqtStm - psqtNstm) ~/ 2;

    // Combined uint8 vector (SFNNv5 pair-product).
    final combined = Uint8List(ftDim);
    for (int i = 0; i < ftDim; i++) {
      var a = accStm[i];
      if (a < 0) {
        a = 0;
      } else if (a > 127) {
        a = 127;
      }
      var b = accNstm[i];
      if (b < 0) {
        b = 0;
      } else if (b > 127) {
        b = 127;
      }
      combined[i] = (a * b) >> 7;
    }

    final rb = raw.network.buckets[bucket];

    // L1: [ftDim] uint8 → [16] int32 → shift → int8-clipped.
    final l1OutRaw = Int32List(16);
    for (int j = 0; j < 16; j++) {
      int s = rb.l1BiasesI32[j];
      final rowBase = j * ftDim;
      for (int i = 0; i < ftDim; i++) {
        s += rb.l1WeightsI8[rowBase + i] * combined[i];
      }
      l1OutRaw[j] = s >> _kWeightShift;
    }

    // Split into ClippedReLU + SqrClippedReLU branches, concat → 32 uint8.
    final l2In = Uint8List(32);
    for (int j = 0; j < 16; j++) {
      var c = l1OutRaw[j];
      if (c < 0) {
        c = 0;
      } else if (c > 127) {
        c = 127;
      }
      l2In[j] = c;
      l2In[j + 16] = (c * c) >> 7;
    }

    // L2: [32] uint8 → [32] int32 → shift → int8-clipped uint8.
    final l2Out = Uint8List(32);
    for (int j = 0; j < 32; j++) {
      int s = rb.l2BiasesI32[j];
      final rowBase = j * 32;
      for (int i = 0; i < 32; i++) {
        s += rb.l2WeightsI8[rowBase + i] * l2In[i];
      }
      var v = s >> _kWeightShift;
      if (v < 0) {
        v = 0;
      } else if (v > 127) {
        v = 127;
      }
      l2Out[j] = v;
    }

    // L3: [32] uint8 → [1] int32 (no shift/clip; single output).
    int l3 = rb.l3BiasesI32[0];
    for (int i = 0; i < 32; i++) {
      l3 += rb.l3WeightsI8[i] * l2Out[i];
    }

    // Final scaling. SF's cp = (nn_out + psqt) / OutputScale, in units of
    // Stockfish's Value where 1 pawn = 208. Our external cp uses "1 pawn
    // = 100", so multiply by 100/208. The returned cp is the raw NN+PSQT
    // signal; SF's UCI cp additionally applies phase / material / draw
    // scaling.
    final raw32 = l3 + psqtRaw;
    final cp = raw32 * 100.0 / (_kIntOutputScale * 208.0);
    return NnueEvalResult(
      cp: cp,
      bucket: bucket,
      psqtContribution: psqtRaw * 100.0 / (_kIntOutputScale * 208.0),
      rawL3Output: l3 * 100.0 / (_kIntOutputScale * 208.0),
    );
  }
}

// Stockfish's post-multiply shift for hidden layers (`WeightScaleBits`).
const int _kWeightShift = 6;

// SF `OutputScale` for SFNNv5. Divides the final int32 output from the
// last affine layer to normalise into Stockfish's internal Value units.
const double _kIntOutputScale = 16.0;

double _clipReLU(double x) => x < 0 ? 0 : (x > 1.0 ? 1.0 : x);

class _BucketFloat {
  const _BucketFloat({
    required this.l1Biases,
    required this.l1Weights,
    required this.l2Biases,
    required this.l2Weights,
    required this.l3Biases,
    required this.l3Weights,
  });

  final Float32List l1Biases; // [16]
  final Float32List l1Weights; // [16, ftDim]
  final Float32List l2Biases; // [32]
  final Float32List l2Weights; // [32, 32]
  final Float32List l3Biases; // [1]
  final Float32List l3Weights; // [32]
}

// Stockfish's canonical quant scales. FT accumulator uses int16 with
// weights and biases scaled by 64 (so /64 = "logical" float). The
// hidden stack uses int8 weights (scale 64) with int32 biases scaled by
// 64 * 64 = 4096 so that a bias-only pass through fc yields the same
// "logical" value that a purely quantised computation would. The final
// output scale converts a raw fc_2 float back to centipawns via
// Stockfish's `Value = fc_2_out / FV_SCALE * 100` heuristic; FV_SCALE
// is 16 in `nnue_common.h`.
const double _kFtQuant = 64.0;
const double _kFtClip = 127.0 / _kFtQuant;
const double _kWeightScale = 64.0;
const double _kBiasScale = 64.0 * 64.0;
const double _kPsqtScale = 64.0 * 64.0;
const double _kOutputScale = 100.0;
