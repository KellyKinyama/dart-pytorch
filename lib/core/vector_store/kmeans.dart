/// Lloyd's k-means clustering used by IVF-family indexes.
///
/// This is a straight port of FAISS' `Clustering` object at the level
/// of detail needed for coarse-quantizer training: k-means++ init,
/// fixed number of iterations, empty-cluster split, no minibatching.
library;

import 'dart:math' as math;
import 'dart:typed_data';

/// Result of a [Kmeans.train] call.
class KmeansResult {
  KmeansResult(this.centroids, this.assignments, this.objective);

  /// `nlist × d` cluster centroids stored as a `List<Float32List>`.
  final List<Float32List> centroids;

  /// For each training vector, the id of its assigned centroid.
  final Int32List assignments;

  /// Sum of squared distances to the assigned centroid (Lloyd's cost).
  final double objective;
}

/// Simple Lloyd's k-means over `float32` vectors with k-means++ init.
class Kmeans {
  Kmeans({
    required this.d,
    required this.k,
    this.niter = 20,
    this.seed = 1234,
    this.minPointsPerCentroid = 1,
    this.verbose = false,
  });

  final int d;
  final int k;
  final int niter;
  final int seed;
  final int minPointsPerCentroid;
  final bool verbose;

  KmeansResult train(List<Float32List> xs) {
    final n = xs.length;
    if (n < k) {
      throw ArgumentError('k-means: need at least k=$k vectors, got $n');
    }
    final rng = math.Random(seed);

    // ---- k-means++ init ----------------------------------------------------
    final centroids = List<Float32List>.generate(k, (_) => Float32List(d));
    // First centroid = uniform random point.
    _copyInto(centroids[0], xs[rng.nextInt(n)]);
    final closestSq = Float32List(n);
    for (var i = 0; i < n; i++) {
      closestSq[i] = _l2sq(xs[i], centroids[0]);
    }
    for (var c = 1; c < k; c++) {
      var sum = 0.0;
      for (var i = 0; i < n; i++) {
        sum += closestSq[i];
      }
      double target;
      if (sum <= 0) {
        // All duplicates or degenerate — fall back to uniform random.
        _copyInto(centroids[c], xs[rng.nextInt(n)]);
      } else {
        target = rng.nextDouble() * sum;
        var acc = 0.0;
        var pick = n - 1;
        for (var i = 0; i < n; i++) {
          acc += closestSq[i];
          if (acc >= target) {
            pick = i;
            break;
          }
        }
        _copyInto(centroids[c], xs[pick]);
      }
      // Refresh closestSq to include the new centroid.
      for (var i = 0; i < n; i++) {
        final dsq = _l2sq(xs[i], centroids[c]);
        if (dsq < closestSq[i]) closestSq[i] = dsq;
      }
    }

    // ---- Lloyd iterations --------------------------------------------------
    final assign = Int32List(n);
    var obj = 0.0;
    for (var it = 0; it < niter; it++) {
      // Assignment step.
      obj = 0.0;
      for (var i = 0; i < n; i++) {
        var best = 0;
        var bestSq = _l2sq(xs[i], centroids[0]);
        for (var c = 1; c < k; c++) {
          final ds = _l2sq(xs[i], centroids[c]);
          if (ds < bestSq) {
            bestSq = ds;
            best = c;
          }
        }
        assign[i] = best;
        obj += bestSq;
      }

      // Update step.
      final sums = List<Float32List>.generate(k, (_) => Float32List(d));
      final counts = Int32List(k);
      for (var i = 0; i < n; i++) {
        final a = assign[i];
        counts[a]++;
        final row = xs[i];
        final s = sums[a];
        for (var j = 0; j < d; j++) {
          s[j] += row[j];
        }
      }
      for (var c = 0; c < k; c++) {
        if (counts[c] >= minPointsPerCentroid) {
          final inv = 1.0 / counts[c];
          final s = sums[c];
          final cen = centroids[c];
          for (var j = 0; j < d; j++) {
            cen[j] = s[j] * inv;
          }
        } else {
          // Empty cluster: steal the biggest one with a tiny perturbation.
          var big = 0;
          for (var cc = 1; cc < k; cc++) {
            if (counts[cc] > counts[big]) big = cc;
          }
          final src = centroids[big];
          final dst = centroids[c];
          for (var j = 0; j < d; j++) {
            final jitter = (rng.nextDouble() - 0.5) * 1e-4;
            dst[j] = src[j] + jitter;
          }
        }
      }
      if (verbose) {
        print('  k-means iter $it obj=${obj.toStringAsFixed(4)}');
      }
    }

    return KmeansResult(centroids, assign, obj);
  }

  static double _l2sq(Float32List a, Float32List b) {
    var s = 0.0;
    for (var i = 0; i < a.length; i++) {
      final diff = a[i] - b[i];
      s += diff * diff;
    }
    return s;
  }

  static void _copyInto(Float32List dst, Float32List src) {
    for (var i = 0; i < dst.length; i++) {
      dst[i] = src[i];
    }
  }
}
