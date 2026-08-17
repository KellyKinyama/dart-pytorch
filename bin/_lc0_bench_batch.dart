import 'package:dart_pytorch/dart_pytorch.dart';

void main() {
  final w = Lc0Reader.readFile('models/lc0/744706.pb.gz');
  final net = Lc0Net(w, device: Device.GPU);
  final start = Lc0Input.fromFen(startFen);
  final e4 = Lc0Input.fromFen(
    'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1',
  );

  // Single-batch reference.
  final swS = Stopwatch()..start();
  final s1 = net(start);
  final s2 = net(e4);
  final singleMs = swS.elapsedMilliseconds;

  // Batched.
  final swB = Stopwatch()..start();
  final batched = net.callBatch([start, e4]);
  final batchMs = swB.elapsedMilliseconds;

  double diff(List<double> a, List<double> b) {
    var d = 0.0;
    for (int i = 0; i < a.length; i++) {
      final x = (a[i] - b[i]).abs();
      if (x > d) d = x;
    }
    return d;
  }

  final vDiff0 = diff(s1.value.toList(), batched[0].value.toList());
  final pDiff0 = diff(
    s1.policyLogits.toList(),
    batched[0].policyLogits.toList(),
  );
  final vDiff1 = diff(s2.value.toList(), batched[1].value.toList());
  final pDiff1 = diff(
    s2.policyLogits.toList(),
    batched[1].policyLogits.toList(),
  );

  print('single: 2 x forward = $singleMs ms');
  print('batch : 1 x forward(B=2) = $batchMs ms');
  print('max abs diff wdl  #0: $vDiff0');
  print('max abs diff pol  #0: $pDiff0');
  print('max abs diff wdl  #1: $vDiff1');
  print('max abs diff pol  #1: $pDiff1');

  // Bigger batches.
  for (final b in [4, 8, 16, 32]) {
    final inputs = List<Tensor>.generate(b, (_) => Lc0Input.fromFen(startFen));
    final sw = Stopwatch()..start();
    net.callBatch(inputs);
    print(
      'B=$b: ${sw.elapsedMilliseconds} ms  '
      '(${(sw.elapsedMilliseconds / b).toStringAsFixed(1)} ms/position)',
    );
  }
}
