/// Minimal "vector database" demo — text encoder + triplet training.
///
/// Port of `dart_cuda/example/bin/vector_store.dart` to dart-pytorch.
///
/// Pipeline:
///   1. Word-hash tokenize each document into `[seqLen]` float indices.
///   2. Run a [TextTransformer] → mean-pool → L2-normalize → `[1, D]`.
///   3. Stack all doc vectors as an `[N, D]` index and cosine-search
///      via `query @ indexᵀ`.
///   4. Fine-tune on **triplet loss** — pull (anchor, positive) closer,
///      push (anchor, negative) apart — for a few epochs to sharpen
///      the ranking on a small set of semantic groups.
///
/// Even at this toy scale you should see the query "What is a safe
/// vehicle for kids?" go from mostly-random ordering to ranking
/// vehicle-related docs above cooking / physics docs.
///
/// Run:
///
///     dart run bin/vector_store_demo.dart          # CPU
///     dart run bin/vector_store_demo.dart --gpu    # GPU
library;

import 'dart:math';

import 'package:dart_pytorch/dart_pytorch.dart';

const int _vocabSize = 4096;
const int _maxSeqLen = 32;
const int _embedDim = 32;
const int _numLayers = 2;
const int _numHeads = 4;
const int _epochs = 20;
const double _lr = 1e-3;
const double _margin = 0.4;

// ---------------------------------------------------------------------

class VectorStore {
  final int dim;
  final _payloads = <String>[];
  final _rows = <List<double>>[];
  Tensor? _indexT; // [D, N], pre-transposed

  VectorStore(this.dim);

  void add(String payload, List<double> embedding) {
    if (embedding.length != dim) {
      throw ArgumentError('dim ${embedding.length} != $dim');
    }
    _payloads.add(payload);
    _rows.add(_l2norm(embedding));
    _indexT = null;
  }

  List<({String payload, double score})> search(
    List<double> query, {
    int k = 5,
  }) {
    if (_payloads.isEmpty) return const [];
    _indexT ??= _buildIndexT();

    final q = _l2norm(query);
    final qT = Tensor.fromList([1, dim], q, device: _indexT!.device);
    final sims = qT.matmul(_indexT!); // [1, N]
    final scores = sims.toList();

    final idx = List<int>.generate(_payloads.length, (i) => i)
      ..sort((a, b) => scores[b].compareTo(scores[a]));
    return [
      for (final i in idx.take(k))
        (payload: _payloads[i], score: scores[i].toDouble()),
    ];
  }

  Tensor _buildIndexT() {
    final flat = <double>[for (final v in _rows) ...v];
    final index = Tensor.fromList(
      [_rows.length, dim],
      flat,
      device: Device.CPU,
    );
    return index.transpose();
  }

  static List<double> _l2norm(List<double> v) {
    var s = 0.0;
    for (final x in v) {
      s += x * x;
    }
    final n = sqrt(s) + 1e-12;
    return [for (final x in v) x / n];
  }
}

// ---------------------------------------------------------------------

class TextEncoder {
  final int vocabSize;
  final int maxSeqLen;
  final int embedDim;
  final Device device;
  final TextTransformer model;

  TextEncoder({
    this.vocabSize = _vocabSize,
    this.maxSeqLen = _maxSeqLen,
    this.embedDim = _embedDim,
    this.device = Device.CPU,
    int seed = 0,
  }) : model = TextTransformer(
         vocabSize: vocabSize,
         maxSeqLen: maxSeqLen,
         embedDim: embedDim,
         numLayers: _numLayers,
         numHeads: _numHeads,
         device: device,
         seed: seed,
       );

  List<double> tokenize(String text) {
    final words = text
        .toLowerCase()
        .split(RegExp(r'[^a-z0-9]+'))
        .where((w) => w.isNotEmpty)
        .toList();
    var ids = <double>[
      for (final w in words) (w.hashCode.abs() % vocabSize).toDouble(),
    ];
    if (ids.isEmpty) ids = <double>[0.0];
    if (ids.length > maxSeqLen) ids = ids.sublist(0, maxSeqLen);
    return ids;
  }

  /// Autograd-live forward: [1, embedDim] L2-normalized.
  Tensor forwardEncode(String text) {
    final ids = tokenize(text);
    final tokens = Tensor.fromList([ids.length], ids, device: device);
    final pooled = model.poolMean(tokens); // [1, embedDim]
    // L2 normalize: y = x / sqrt(sum(x*x) + eps).
    final sq = pooled * pooled;
    final sumSq = sq.sum(); // [1, 1]
    final norm = (sumSq + 1e-10).pow(0.5);
    return pooled / norm;
  }

  /// Inference: encode + pull to host as plain doubles.
  List<double> encode(String text) {
    return Tensor.noGrad(() => forwardEncode(text).toList());
  }
}

// ---------------------------------------------------------------------

class Triplet {
  final String anchor;
  final String positive;
  final String negative;
  const Triplet(this.anchor, this.positive, this.negative);
}

/// Triplet loss: `relu(||a-p||² - ||a-n||² + margin)`. Because a/p/n
/// are unit vectors, `||a-b||² = 2 - 2*(a·b) ∈ [0, 4]`.
Tensor _tripletLoss(Tensor a, Tensor p, Tensor n, double margin) {
  final ap = a - p;
  final an = a - n;
  final dp = (ap * ap).sum();
  final dn = (an * an).sum();
  final device = a.device;
  final marginT = Tensor.fill([1, 1], margin, device: device);
  return (dp - dn + marginT).relu();
}

void _trainEncoder(TextEncoder enc, List<Triplet> triplets) {
  final params = enc.model.parameters();
  final opt = Adam(params, lr: _lr);
  final rng = Random(0);

  for (var epoch = 0; epoch < _epochs; epoch++) {
    final shuffled = [...triplets]..shuffle(rng);
    var epochLoss = 0.0;

    for (final t in shuffled) {
      opt.zeroGrad();
      final a = enc.forwardEncode(t.anchor);
      final p = enc.forwardEncode(t.positive);
      final n = enc.forwardEncode(t.negative);
      final loss = _tripletLoss(a, p, n, _margin);
      loss.backward();
      clipGradNorm(params, 1.0);
      opt.step();
      epochLoss += loss.toList()[0];
    }

    if (epoch == 0 || epoch % 5 == 0 || epoch == _epochs - 1) {
      final avg = epochLoss / triplets.length;
      print(
        '  epoch ${epoch.toString().padLeft(2)}  '
        'avg triplet loss=${avg.toStringAsFixed(4)}',
      );
    }
  }
}

void _runSearch(TextEncoder encoder, List<String> docs, String query) {
  final store = VectorStore(encoder.embedDim);
  for (final d in docs) {
    store.add(d, encoder.encode(d));
  }
  final hits = store.search(encoder.encode(query), k: docs.length);
  for (final h in hits) {
    print('  ${h.score.toStringAsFixed(4)}  ${h.payload}');
  }
}

void main(List<String> args) {
  final device = args.contains('--gpu') ? Device.GPU : Device.CPU;
  print('=== vector_store_demo (${device.name}) ===');
  print(
    'vocabSize=$_vocabSize, embedDim=$_embedDim, '
    'layers=$_numLayers, heads=$_numHeads',
  );

  final encoder = TextEncoder(device: device);

  const docs = [
    'SUV crash test ratings and family safety',
    'How to bake sourdough bread at home',
    'Minivan reliability comparison 2025',
    'Quantum entanglement explained simply',
    'Best toddler car seats reviewed',
  ];
  const query = 'What is a safe vehicle for kids?';

  const vehicles = [
    'SUV crash test ratings and family safety',
    'Minivan reliability comparison 2025',
    'Best toddler car seats reviewed',
    'safe family car for children',
    'kid friendly vehicle review',
    'child car seat safety guide',
  ];
  const cooking = [
    'How to bake sourdough bread at home',
    'beginner bread baking recipe',
    'sourdough starter tips',
  ];
  const physics = [
    'Quantum entanglement explained simply',
    'introduction to quantum physics',
    'entangled particles experiment',
  ];

  final groups = [vehicles, cooking, physics];
  final triplets = <Triplet>[];
  final rng = Random(1);
  for (var i = 0; i < groups.length; i++) {
    final pos = groups[i];
    final negPool = <String>[
      for (var j = 0; j < groups.length; j++)
        if (j != i) ...groups[j],
    ];
    for (final a in pos) {
      for (final p in pos) {
        if (a == p) continue;
        final n = negPool[rng.nextInt(negPool.length)];
        triplets.add(Triplet(a, p, n));
      }
    }
  }
  print(
    'mined ${triplets.length} triplets from '
    '${groups.length} groups\n',
  );

  print('Before training — query: "$query"');
  _runSearch(encoder, docs, query);

  print('\nTraining $_epochs epochs (margin=$_margin, lr=$_lr)...');
  _trainEncoder(encoder, triplets);

  print('\nAfter training — query: "$query"');
  _runSearch(encoder, docs, query);
}
