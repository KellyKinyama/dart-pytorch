/// `PCATransform` — principal component analysis pre-transform.
///
/// Fits a `dIn × dOut` projection matrix `W` from the top-`dOut`
/// eigenvectors of the training covariance matrix, plus a bias
/// `-Wμ` so that outputs are mean-centred. Optionally the projection
/// can be scaled so each output dimension has unit variance
/// (`eigenPower = -0.5`), which matches FAISS' `eigen_power` knob.
///
/// Eigendecomposition uses the cyclic Jacobi rotation method on the
/// full `dIn × dIn` covariance matrix — `O(dIn³)` but branch-free
/// and numerically robust; runs in a few tens of milliseconds for
/// `dIn ≤ 512`.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'index_io.dart';
import 'vector_transform.dart';

class PCATransform extends VectorTransform {
  PCATransform({required int dIn, required int dOut, this.eigenPower = 0.0})
    : _w = Float32List(dOut * dIn),
      _mean = Float32List(dIn),
      super(dIn, dOut) {
    if (dOut < 1 || dOut > dIn) {
      throw ArgumentError('PCA: dOut ($dOut) must be in [1, $dIn]');
    }
    isTrained = false;
  }

  /// Exponent applied to eigenvalues when scaling `W` rows. `0` leaves
  /// the projection orthonormal (the default). `-0.5` whitens.
  final double eigenPower;

  final Float32List _w; // dOut × dIn, row-major
  final Float32List _mean; // dIn
  Float32List _eigenvalues = Float32List(0); // dOut

  Float32List get projection => _w;
  Float32List get mean => _mean;
  Float32List get eigenvalues => _eigenvalues;

  @override
  void train(List<Float32List> xs) {
    final n = xs.length;
    if (n < 2) {
      throw ArgumentError('PCA.train: need at least 2 vectors, got $n');
    }
    // 1. Sample mean.
    final mean = Float64List(dIn);
    for (var i = 0; i < n; i++) {
      final row = xs[i];
      if (row.length != dIn) {
        throw ArgumentError(
          'PCA.train: vector $i length ${row.length} != $dIn',
        );
      }
      for (var j = 0; j < dIn; j++) {
        mean[j] += row[j];
      }
    }
    for (var j = 0; j < dIn; j++) {
      mean[j] /= n;
      _mean[j] = mean[j];
    }
    // 2. Covariance matrix (dIn × dIn, symmetric, dense).
    final cov = List<Float64List>.generate(dIn, (_) => Float64List(dIn));
    final centred = Float64List(dIn);
    for (var i = 0; i < n; i++) {
      final row = xs[i];
      for (var j = 0; j < dIn; j++) {
        centred[j] = row[j] - mean[j];
      }
      for (var r = 0; r < dIn; r++) {
        final cr = centred[r];
        final covr = cov[r];
        for (var c = r; c < dIn; c++) {
          covr[c] += cr * centred[c];
        }
      }
    }
    final inv = 1.0 / (n - 1);
    for (var r = 0; r < dIn; r++) {
      for (var c = r; c < dIn; c++) {
        cov[r][c] *= inv;
        cov[c][r] = cov[r][c]; // symmetrize
      }
    }
    // 3. Cyclic Jacobi eigendecomposition. `V` accumulates rotations
    //    so that after convergence `Vᵀ cov V ≈ diag(eig)`.
    final v = List<Float64List>.generate(
      dIn,
      (i) => Float64List(dIn)..[i] = 1.0,
    );
    _jacobi(cov, v);
    // Extract diagonal eigenvalues + sort descending along with V.
    final eigPairs = List<({double val, int col})>.generate(
      dIn,
      (i) => (val: cov[i][i], col: i),
    );
    eigPairs.sort((a, b) => b.val.compareTo(a.val));
    // 4. Take top dOut eigenvectors as rows of _w, apply eigenPower.
    _eigenvalues = Float32List(dOut);
    for (var r = 0; r < dOut; r++) {
      final col = eigPairs[r].col;
      final ev = eigPairs[r].val;
      _eigenvalues[r] = ev;
      final scale = eigenPower == 0.0
          ? 1.0
          : math.pow(math.max(ev, 1e-12), eigenPower).toDouble();
      for (var c = 0; c < dIn; c++) {
        _w[r * dIn + c] = (v[c][col] * scale).toDouble();
      }
    }
    isTrained = true;
  }

  static void _jacobi(List<Float64List> a, List<Float64List> v) {
    final n = a.length;
    const maxSweeps = 50;
    const eps = 1e-10;
    for (var sweep = 0; sweep < maxSweeps; sweep++) {
      // Off-diagonal Frobenius norm.
      var off = 0.0;
      for (var p = 0; p < n - 1; p++) {
        for (var q = p + 1; q < n; q++) {
          off += a[p][q] * a[p][q];
        }
      }
      if (off < eps) return;
      for (var p = 0; p < n - 1; p++) {
        for (var q = p + 1; q < n; q++) {
          final apq = a[p][q];
          if (apq.abs() < 1e-16) continue;
          final app = a[p][p];
          final aqq = a[q][q];
          final theta = (aqq - app) / (2.0 * apq);
          final t = theta >= 0
              ? 1.0 / (theta + math.sqrt(1.0 + theta * theta))
              : 1.0 / (theta - math.sqrt(1.0 + theta * theta));
          final c = 1.0 / math.sqrt(1.0 + t * t);
          final s = t * c;
          // Update A: rotate rows/cols p and q.
          a[p][p] = app - t * apq;
          a[q][q] = aqq + t * apq;
          a[p][q] = 0.0;
          a[q][p] = 0.0;
          for (var i = 0; i < n; i++) {
            if (i != p && i != q) {
              final aip = a[i][p];
              final aiq = a[i][q];
              a[i][p] = c * aip - s * aiq;
              a[p][i] = a[i][p];
              a[i][q] = s * aip + c * aiq;
              a[q][i] = a[i][q];
            }
            // Update eigenvector matrix V.
            final vip = v[i][p];
            final viq = v[i][q];
            v[i][p] = c * vip - s * viq;
            v[i][q] = s * vip + c * viq;
          }
        }
      }
    }
  }

  @override
  List<Float32List> apply(List<Float32List> xs) {
    if (!isTrained) throw StateError('PCA.apply before train');
    return List<Float32List>.generate(xs.length, (i) {
      final row = xs[i];
      if (row.length != dIn) {
        throw ArgumentError(
          'PCA.apply: vector $i length ${row.length} != $dIn',
        );
      }
      final o = Float32List(dOut);
      for (var r = 0; r < dOut; r++) {
        var s = 0.0;
        final base = r * dIn;
        for (var c = 0; c < dIn; c++) {
          s += _w[base + c] * (row[c] - _mean[c]);
        }
        o[r] = s;
      }
      return o;
    });
  }

  @override
  void writeTo(IoWriter w) {
    w.writeU32(TransformKind.pca);
    w.writeU32(dIn);
    w.writeU32(dOut);
    // Store eigenPower as float32 for compactness.
    final ep = ByteData(4)..setFloat32(0, eigenPower, Endian.little);
    w.writeU32(ep.getUint32(0, Endian.little));
    w.writeU8(isTrained ? 1 : 0);
    w.writeF32List(_mean);
    w.writeF32List(_w);
    w.writeU32(_eigenvalues.length);
    if (_eigenvalues.isNotEmpty) w.writeF32List(_eigenvalues);
  }

  static PCATransform readFrom(IoReader r) {
    final dIn = r.readU32();
    final dOut = r.readU32();
    final epBits = r.readU32();
    final ep = ByteData(4)..setUint32(0, epBits, Endian.little);
    final eigenPower = ep.getFloat32(0, Endian.little);
    final trained = r.readU8() != 0;
    final mean = r.readF32List(dIn);
    final w = r.readF32List(dOut * dIn);
    final nEig = r.readU32();
    final eig = nEig > 0 ? r.readF32List(nEig) : Float32List(0);
    final t = PCATransform(dIn: dIn, dOut: dOut, eigenPower: eigenPower);
    for (var i = 0; i < mean.length; i++) {
      t._mean[i] = mean[i];
    }
    for (var i = 0; i < w.length; i++) {
      t._w[i] = w[i];
    }
    t._eigenvalues = eig;
    t.isTrained = trained;
    return t;
  }
}
