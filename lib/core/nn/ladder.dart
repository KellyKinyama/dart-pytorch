/// Intelligence-ladder block selection, after Cactus Needle's ladder rule:
/// every depth from 1 to `numLayers` is a deployable sub-network, and the
/// blocks kept at a given depth are chosen by endpoint-preserving bisection so
/// the sub-networks are nested and spatially balanced.
library;

/// Block-selection order for an `n`-block model: start with the two endpoints,
/// then repeatedly add the midpoint of the largest remaining gap, breaking
/// ties toward the smaller left index. Depth `d` keeps the first `d` of these.
List<int> ladderLayerOrder(int n) {
  if (n <= 1) return [0];
  final selected = <int>[0, n - 1];
  final order = <int>[0, n - 1];
  while (order.length < n) {
    selected.sort();
    var bestGap = -1, bestLeft = 1 << 30, bestRight = 0;
    for (var i = 0; i < selected.length - 1; i++) {
      final l = selected[i], r = selected[i + 1];
      final g = r - l;
      if (g > 1 && (g > bestGap || (g == bestGap && l < bestLeft))) {
        bestGap = g;
        bestLeft = l;
        bestRight = r;
      }
    }
    final cand = (bestLeft + bestRight) ~/ 2;
    selected.add(cand);
    order.add(cand);
  }
  return order;
}

/// Original block indices (ascending) kept at [depth] for an `n`-block ladder.
List<int> ladderBlocks(int n, int depth) =>
    ladderLayerOrder(n).take(depth).toList()..sort();
