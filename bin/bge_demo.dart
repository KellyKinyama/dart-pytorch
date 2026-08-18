/// BGE-small-en-v1.5 sentence-embedding demo. SOTA text encoder in
/// the ~30M range, drops in wherever the repo currently uses
/// MiniLM-L6-v2 (see `bin/rag_gpu_demo.dart`, `bin/rag_qa_demo.dart`,
/// `bin/vector_*_demo.dart`).
///
///   dart run bin/bge_demo.dart              # CPU
///   LD_LIBRARY_PATH=/usr/lib/wsl/lib \
///     dart run bin/bge_demo.dart --gpu      # GPU
///
/// Prints per-sentence embed time and a small semantic-search demo
/// (query vs. corpus with cosine similarity).
///
/// Weights + vocab (one-time):
///   mkdir -p models/bge-small-en
///   curl -L -o models/bge-small-en/model.safetensors \
///     https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/model.safetensors
///   curl -L -o models/bge-small-en/vocab.txt \
///     https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/vocab.txt
library;

import 'dart:io';

import 'package:dart_pytorch/dart_pytorch.dart';

const _weightsPath = 'models/bge-small-en/model.safetensors';
const _vocabPath = 'models/bge-small-en/vocab.txt';

const List<String> _corpus = [
  'Ada Lovelace wrote the first computer program in the 1840s.',
  'The Great Wall of China stretches across northern China.',
  'Isaac Newton unified celestial and terrestrial mechanics.',
  'The Eiffel Tower was completed in 1889 in Paris.',
  'A pizza margherita is topped with tomato, mozzarella, and basil.',
  'Vincent van Gogh painted Starry Night in June 1889.',
];

const _query = 'Who wrote the first computer program?';

Future<void> main(List<String> args) async {
  var useGpu = false;
  var weightsPath = _weightsPath;
  var vocabPath = _vocabPath;
  for (int i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--gpu':
        useGpu = true;
        break;
      case '--weights':
        weightsPath = args[++i];
        break;
      case '--vocab':
        vocabPath = args[++i];
        break;
    }
  }
  for (final p in [weightsPath, vocabPath]) {
    if (!File(p).existsSync()) {
      stderr.writeln('missing: $p');
      exit(2);
    }
  }

  final device = useGpu ? Device.GPU : Device.CPU;
  final swTotal = Stopwatch()..start();

  print('== load ${useGpu ? '(GPU)' : '(CPU)'} ==');
  final swLoad = Stopwatch()..start();
  final tok = WordPieceTokenizer.fromVocabFile(vocabPath);
  final bert = BertModel(BertHFLoader.bgeSmallEnConfig(device: device));
  final report = BertHFLoader.loadFile(bert, weightsPath);
  final encoder = SentenceEncoder.wrap(bert, pooling: PoolingMode.cls);
  encoder.eval();
  swLoad.stop();
  print('  ${swLoad.elapsedMilliseconds} ms  $report');
  if (report.unusedKeys.isNotEmpty) {
    print('  unused (first 3):');
    for (final k in report.unusedKeys.take(3)) {
      print('    - $k');
    }
  }

  print('');
  print('== encode ${_corpus.length + 1} sentences ==');
  // Warm-up.
  encoder(_toTensor(tok, _corpus[0], device));
  final swE = Stopwatch()..start();
  final vecs = <List<double>>[];
  for (final s in _corpus) {
    final ids = _toTensor(tok, s, device);
    vecs.add(encoder(ids).toList());
  }
  final qVec = encoder(_toTensor(tok, _query, device)).toList();
  swE.stop();
  print('  ${swE.elapsedMilliseconds} ms  (dim=${qVec.length})');

  final scored = <MapEntry<String, double>>[];
  for (int i = 0; i < _corpus.length; i++) {
    scored.add(MapEntry(_corpus[i], _dot(qVec, vecs[i])));
  }
  scored.sort((a, b) => b.value.compareTo(a.value));

  print('');
  print('== query ==');
  print('  "$_query"');
  print('');
  print('== top matches ==');
  for (int i = 0; i < scored.length; i++) {
    final marker = i == 0 ? '★' : ' ';
    print('  $marker  cos=${scored[i].value.toStringAsFixed(4)}   '
        '${scored[i].key}');
  }

  swTotal.stop();
  print('');
  print('total wall = ${swTotal.elapsedMilliseconds} ms');
}

Tensor _toTensor(WordPieceTokenizer tok, String text, Device device) {
  final ids = tok.encode(text, maxLength: 128);
  final t = Tensor.fromList(
    [ids.length],
    ids.map((i) => i.toDouble()).toList(),
  );
  return device == Device.CPU ? t : t.to(device);
}

double _dot(List<double> a, List<double> b) {
  double s = 0.0;
  for (int i = 0; i < a.length; i++) {
    s += a[i] * b[i];
  }
  return s;
}
