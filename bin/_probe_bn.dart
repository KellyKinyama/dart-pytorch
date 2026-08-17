import 'package:dart_pytorch/dart_pytorch.dart';

void main() {
  final w = Lc0Reader.readFile('models/lc0/744706.pb.gz');
  // Check ranges of a few raw and folded values.
  final v = w.input.bnStddivs.toList();
  final g = w.input.bnGammas.toList();
  final b = w.input.biases.toList();
  final rawW = w.input.weights.toList();
  double minW = rawW[0], maxW = rawW[0];
  for (final x in rawW) {
    if (x < minW) minW = x;
    if (x > maxW) maxW = x;
  }
  print('input.raw weight range: [$minW, $maxW]');
  print(
    'input.bias range: [${b.reduce((a, b) => a < b ? a : b)}, ${b.reduce((a, b) => a > b ? a : b)}]',
  );
  double sumG = 0, maxNewGamma = 0;
  for (int o = 0; o < g.length; o++) {
    final ng = g[o] / (v[o] + 1e-5) * (v[o] + 1e-5);
    // Actually let's compute: newGamma = gamma / sqrt(var + eps)
    final rng = g[o] / (v[o] < 0 ? 1 : (0.001 + v[o])).abs();
    sumG += rng;
    if (rng.abs() > maxNewGamma) maxNewGamma = rng.abs();
  }
  print(
    'input mean new_gamma abs: ${sumG.abs() / g.length}, max: $maxNewGamma',
  );

  // Check residual block 0 raw weight range.
  final r0w = w.residual[0].conv1.weights.toList();
  double minR = r0w[0], maxR = r0w[0];
  for (final x in r0w) {
    if (x < minR) minR = x;
    if (x > maxR) maxR = x;
  }
  print('res[0].conv1 raw weight range: [$minR, $maxR]');
  final r0var = w.residual[0].conv1.bnStddivs.toList();
  double minV = r0var[0], maxV = r0var[0];
  for (final x in r0var) {
    if (x < minV) minV = x;
    if (x > maxV) maxV = x;
  }
  print('res[0].conv1 var range: [$minV, $maxV]');
}
