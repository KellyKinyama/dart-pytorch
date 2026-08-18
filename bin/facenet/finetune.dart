/// Fine-tune FaceNet's `last_linear` with triplet loss on a small
/// face gallery. Demonstrates that the folded backbone + trainable
/// `last_linear` autograd path from `doc/facenet.md` actually works.
///
///   dart run bin/facenet/finetune.dart              # CPU
///   LD_LIBRARY_PATH=/usr/lib/wsl/lib \
///     dart run bin/facenet/finetune.dart --gpu      # GPU
///
/// Options:
///   --gallery PATH   default: faces_gallery
///   --identities N   default: 4     (first N alphabetically)
///   --per-id N       default: 8     (limits samples per identity)
///   --steps N        default: 300
///   --lr F           default: 5e-4
///   --margin F       default: 0.4
///   --seed N         default: 0
///
/// Pipeline:
///   1. Load InceptionResnetV1(vggface2), `.eval()`, freeze everything.
///   2. Un-freeze `model.lastLinear.weight` and hand it to Adam.
///   3. Precompute 1792-d frozen `pooled` features for every gallery
///      image once (a single conv-stack forward).
///   4. Sample (anchor, positive, negative) triplets from the frozen
///      features, run only the trainable Linear + last_bn + L2 norm
///      per step, backprop triplet loss.
///   5. Report same-person / different-person cosine gaps before and
///      after training.
library;

import 'dart:io';
import 'dart:math' as math;

import 'package:dart_pytorch/core/nn/vision/facenet.dart';
import 'package:dart_pytorch/core/nn/vision/facenet_loader.dart';
import 'package:dart_pytorch/core/nn/vision/nchw.dart';
import 'package:dart_pytorch/core/nn/vision/pool2d.dart';
import 'package:dart_pytorch/core/optim/adam.dart';
import 'package:dart_pytorch/core/tensor/tensor.dart';

import '_common.dart';

const _weightsPathDefault = 'models/facenet-vggface2/model.safetensors';

Future<void> main(List<String> args) async {
  var weightsPath = _weightsPathDefault;
  var galleryPath = 'faces_gallery';
  var numIds = 4;
  var perId = 8;
  var steps = 300;
  var lr = 5e-4;
  var margin = 0.4;
  var seed = 0;
  var useGpu = false;

  for (int i = 0; i < args.length; i++) {
    final a = args[i];
    switch (a) {
      case '--gpu':
        useGpu = true;
        break;
      case '--weights':
        weightsPath = args[++i];
        break;
      case '--gallery':
        galleryPath = args[++i];
        break;
      case '--identities':
        numIds = int.parse(args[++i]);
        break;
      case '--per-id':
        perId = int.parse(args[++i]);
        break;
      case '--steps':
        steps = int.parse(args[++i]);
        break;
      case '--lr':
        lr = double.parse(args[++i]);
        break;
      case '--margin':
        margin = double.parse(args[++i]);
        break;
      case '--seed':
        seed = int.parse(args[++i]);
        break;
    }
  }
  if (!File(weightsPath).existsSync()) {
    stderr.writeln('missing weights: $weightsPath');
    exit(2);
  }
  if (!Directory(galleryPath).existsSync()) {
    stderr.writeln('missing gallery: $galleryPath');
    exit(2);
  }

  final device = useGpu ? Device.GPU : Device.CPU;
  final rng = math.Random(seed);
  final swTotal = Stopwatch()..start();

  // ---------------- 1. load + freeze ----------------

  print('== build + load ${useGpu ? '(GPU)' : '(CPU)'} ==');
  final swLoad = Stopwatch()..start();
  final model = InceptionResnetV1(device: device);
  FaceNetLoader.loadFile(model, weightsPath);
  model.eval();
  swLoad.stop();
  print('  loaded in ${swLoad.elapsedMilliseconds} ms');

  final totalParams = model.parameters().length;
  for (final p in model.parameters()) {
    p.requiresGrad = false;
  }
  model.lastLinear.weight.requiresGrad = true;
  final trainable = model.parameters().where((p) => p.requiresGrad).toList();
  print(
    '  froze $totalParams params, training ${trainable.length}: '
    'lastLinear.weight ${model.lastLinear.weight.shape}',
  );

  // ---------------- 2. discover gallery ----------------

  final classes = <String, List<String>>{};
  for (final e in Directory(galleryPath).listSync()) {
    if (e is Directory) {
      final files =
          e
              .listSync()
              .whereType<File>()
              .map((f) => f.path)
              .where((p) => p.toLowerCase().endsWith('.jpg'))
              .toList()
            ..sort();
      if (files.isNotEmpty) {
        classes[e.path.split(RegExp(r'[/\\]')).last] = files
            .take(perId)
            .toList();
      }
    }
  }
  final ids = classes.keys.toList()..sort();
  if (ids.length < numIds) {
    stderr.writeln('need $numIds identities; gallery has ${ids.length}');
    exit(2);
  }
  final chosen = ids.take(numIds).toList();
  print('');
  print('== gallery ==');
  for (final id in chosen) {
    print('  $id: ${classes[id]!.length} samples');
  }

  // ---------------- 3. precompute frozen [1792] features ----------------

  print('');
  print('== precompute 1792-d frozen features ==');
  final swFeat = Stopwatch()..start();
  // Per identity: List of frozen pooled tensors ([1, 1792]).
  final pooledByClass = <String, List<Tensor>>{};
  for (final id in chosen) {
    final list = <Tensor>[];
    for (final path in classes[id]!) {
      final x = decodeFaceJpeg(path, device: device);
      final pooled = _frozenBackbone(model, x);
      list.add(pooled);
    }
    pooledByClass[id] = list;
  }
  swFeat.stop();
  final totalFaces = chosen
      .map((id) => pooledByClass[id]!.length)
      .fold<int>(0, (a, b) => a + b);
  print(
    '  ${swFeat.elapsedMilliseconds} ms  ($totalFaces faces, '
    'shape ${pooledByClass[chosen.first]!.first.shape})',
  );

  // ---------------- 4. metrics BEFORE training ----------------

  print('');
  print('== metrics BEFORE ==');
  final gapBefore = _metrics(model, pooledByClass, chosen);

  // ---------------- 5. training loop ----------------

  print('');
  print(
    '== fine-tune lastLinear.weight ($steps steps, lr=$lr, margin=$margin) ==',
  );
  final opt = Adam(trainable, lr: lr);
  final swTrain = Stopwatch()..start();
  double lossSum = 0.0;
  int lossN = 0;
  final logEvery = math.max(1, steps ~/ 10);
  for (int s = 0; s < steps; s++) {
    final t = _sampleTriplet(pooledByClass, chosen, rng);
    final aEmb = _headForward(model, t.$1);
    final pEmb = _headForward(model, t.$2);
    final nEmb = _headForward(model, t.$3);

    // Triplet loss: relu(‖a − p‖² − ‖a − n‖² + margin).
    final dAP = _sqDist(aEmb, pEmb);
    final dAN = _sqDist(aEmb, nEmb);
    final raw = dAP - dAN + margin;
    final loss = raw.relu();

    opt.zeroGrad();
    loss.backward();
    opt.step();

    lossSum += loss.toList()[0];
    lossN++;
    if ((s + 1) % logEvery == 0 || s == steps - 1) {
      print(
        '  step ${(s + 1).toString().padLeft(4)}  '
        'loss(avg last $lossN)=${(lossSum / lossN).toStringAsFixed(4)}',
      );
      lossSum = 0.0;
      lossN = 0;
    }
  }
  swTrain.stop();
  print('  training wall: ${swTrain.elapsedMilliseconds} ms');

  // ---------------- 6. metrics AFTER training ----------------

  print('');
  print('== metrics AFTER ==');
  final gapAfter = _metrics(model, pooledByClass, chosen);

  print('');
  print('== summary ==');
  print(
    '  same-person mean cosine    ${gapBefore.$1.toStringAsFixed(4)} → '
    '${gapAfter.$1.toStringAsFixed(4)}  '
    '(Δ ${(gapAfter.$1 - gapBefore.$1).toStringAsFixed(4)})',
  );
  print(
    '  cross-person mean cosine   ${gapBefore.$2.toStringAsFixed(4)} → '
    '${gapAfter.$2.toStringAsFixed(4)}  '
    '(Δ ${(gapAfter.$2 - gapBefore.$2).toStringAsFixed(4)})',
  );
  print(
    '  gap (higher = better)      ${gapBefore.$3.toStringAsFixed(4)} → '
    '${gapAfter.$3.toStringAsFixed(4)}  '
    '(Δ ${(gapAfter.$3 - gapBefore.$3).toStringAsFixed(4)})',
  );

  swTotal.stop();
  print('');
  print('total wall = ${swTotal.elapsedMilliseconds} ms');
}

// ---------------- helpers ----------------

/// Frozen backbone: input [1,3,160,160] -> pooled [1,1792].
/// Inlines the stem + all blocks + globalAvgPool, but stops before
/// `lastLinear` so the caller can reuse the same 1792-d vector every
/// step without re-running the O(1 s) conv stack.
Tensor _frozenBackbone(InceptionResnetV1 m, Tensor x) {
  var h = m.conv2d1a(x);
  h = m.conv2d2a(h);
  h = m.conv2d2b(h);
  h = maxPool2d(h, kernel: 3, stride: 2);
  h = m.conv2d3b(h);
  h = m.conv2d4a(h);
  h = m.conv2d4b(h);
  for (final b in m.repeat1) {
    h = b(h);
  }
  h = m.mixed6a(h);
  for (final b in m.repeat2) {
    h = b(h);
  }
  h = m.mixed7a(h);
  for (final b in m.repeat3) {
    h = b(h);
  }
  h = m.block8Final(h);
  return globalAvgPool2d(h); // [1, 1792]
}

/// Trainable head only: pooled [1,1792] → 512-d L2-normalized emb.
Tensor _headForward(InceptionResnetV1 m, Tensor pooled) {
  var emb = m.lastLinear(pooled);
  emb = emb * m.lastBnScale + m.lastBnOffset;
  return l2NormalizeRows(emb);
}

/// Squared L2 distance between two [1, 512] L2-normalized embeddings.
///   ‖a − b‖² = 2 − 2 a·b   (fp32-cheap)
Tensor _sqDist(Tensor a, Tensor b) {
  final diff = a - b;
  final sq = diff * diff;
  // sum over columns via matmul with a ones-column.
  final ones = Tensor.fill([sq.shape[1], 1], 1.0, device: a.device);
  return sq.matmul(ones); // [1, 1]
}

(Tensor, Tensor, Tensor) _sampleTriplet(
  Map<String, List<Tensor>> byClass,
  List<String> ids,
  math.Random rng,
) {
  final ida = ids[rng.nextInt(ids.length)];
  String idn;
  do {
    idn = ids[rng.nextInt(ids.length)];
  } while (idn == ida);
  final a = byClass[ida]!;
  int ai = rng.nextInt(a.length);
  int pi = rng.nextInt(a.length);
  while (pi == ai && a.length > 1) {
    pi = rng.nextInt(a.length);
  }
  final n = byClass[idn]!;
  final ni = rng.nextInt(n.length);
  return (a[ai], a[pi], n[ni]);
}

/// Same-person mean cosine, cross-person mean cosine, gap.
(double, double, double) _metrics(
  InceptionResnetV1 m,
  Map<String, List<Tensor>> byClass,
  List<String> ids,
) {
  final emb = <String, List<List<double>>>{};
  for (final id in ids) {
    emb[id] = [for (final p in byClass[id]!) _headForward(m, p).toList()];
  }
  double same = 0.0;
  int sameN = 0;
  double cross = 0.0;
  int crossN = 0;
  for (int i = 0; i < ids.length; i++) {
    final aList = emb[ids[i]]!;
    for (int j = 0; j < aList.length; j++) {
      for (int k = j + 1; k < aList.length; k++) {
        same += cosine(aList[j], aList[k]);
        sameN++;
      }
    }
    for (int i2 = i + 1; i2 < ids.length; i2++) {
      final bList = emb[ids[i2]]!;
      for (int j = 0; j < aList.length; j++) {
        for (int k = 0; k < bList.length; k++) {
          cross += cosine(aList[j], bList[k]);
          crossN++;
        }
      }
    }
  }
  final sameMean = same / sameN;
  final crossMean = cross / crossN;
  print(
    '  same-person   pairs=$sameN   mean cosine=${sameMean.toStringAsFixed(4)}',
  );
  print(
    '  cross-person  pairs=$crossN  mean cosine=${crossMean.toStringAsFixed(4)}',
  );
  return (sameMean, crossMean, sameMean - crossMean);
}
