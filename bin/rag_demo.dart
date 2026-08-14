/// End-to-end Retrieval-Augmented Generation demo.
///
/// Loads sentence-transformers/all-MiniLM-L6-v2 (BERT MiniLM, 6
/// layers, 384-d) from `models/minilm/`, indexes a small corpus of
/// factlets, and for each demo query prints the top-K passages by
/// cosine similarity, followed by a naive "answer" that just quotes
/// the top hit. Everything runs on CPU in pure Dart — no Python,
/// no external inference server.
///
/// Prerequisites:
///
/// ```bash
/// mkdir -p models/minilm && cd models/minilm
/// for f in config.json tokenizer.json tokenizer_config.json \
///          vocab.txt model.safetensors; do
///   curl -sSL -O \
///     "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/$f"
/// done
/// ```
///
/// Usage:
///
/// ```bash
/// dart run bin/rag_demo.dart
/// ```
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';

const _weightsPath = 'models/minilm/model.safetensors';
const _vocabPath = 'models/minilm/vocab.txt';

const _corpus = <String>[
  "Ada Lovelace was a nineteenth-century mathematician who wrote what is often "
      "considered the first computer program while working with Charles Babbage.",
  "Alan Turing formalized the concepts of algorithm and computation with the "
      "Turing machine, laying the theoretical foundation for modern computer "
      "science.",
  "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in "
      "Paris, France. It was completed in 1889 for the World's Fair.",
  "Photosynthesis is the process by which green plants convert sunlight, water "
      "and carbon dioxide into glucose and oxygen.",
  "Mount Everest, located in the Himalayas on the border of Nepal and Tibet, "
      "is the tallest mountain on Earth at 8,849 meters above sea level.",
  "Python is a high-level general-purpose programming language famous for its "
      "readability and its broad ecosystem of scientific libraries.",
  "The Great Wall of China is a series of fortifications built across northern "
      "China to protect Chinese states against nomadic groups from the Eurasian "
      "steppe.",
  "Insulin is a peptide hormone produced by beta cells of the pancreas. It "
      "regulates blood glucose by promoting cellular uptake of sugar.",
  "The Amazon rainforest is the largest tropical rainforest in the world, "
      "spanning much of northwestern Brazil and neighbouring countries.",
  "Isaac Newton formulated the laws of motion and universal gravitation, "
      "unifying celestial and terrestrial mechanics in his 1687 Principia.",
];

const _queries = <String>[
  "Who wrote the first computer program?",
  "How tall is Mount Everest?",
  "How do plants make food from sunlight?",
  "Which programming language is known for readability?",
];

Tensor _toTokens(List<int> ids) =>
    Tensor.fromList([ids.length], ids.map((i) => i.toDouble()).toList());

Float32List _rowVec(Tensor t) {
  final flat = t.toList();
  return Float32List.fromList(flat);
}

void main() {
  if (!File(_weightsPath).existsSync() || !File(_vocabPath).existsSync()) {
    stderr.writeln(
      'RAG demo: missing $_weightsPath or $_vocabPath.\n'
      'Download the model first:\n'
      '  mkdir -p models/minilm && cd models/minilm && \\\n'
      '  for f in config.json tokenizer.json tokenizer_config.json \\\n'
      '           vocab.txt model.safetensors; do \\\n'
      '    curl -sSL -O \\\n'
      '      "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/\$f"; \\\n'
      '  done',
    );
    exit(64);
  }

  stdout.writeln('Loading all-MiniLM-L6-v2 ...');
  final sw = Stopwatch()..start();

  final tok = WordPieceTokenizer.fromVocabFile(_vocabPath);
  final bert = BertModel(BertHFLoader.miniLmL6V2Config());
  final report = BertHFLoader.loadFile(bert, _weightsPath);
  final encoder = SentenceEncoder.wrap(bert);
  encoder.eval();

  stdout.writeln(
    '  loaded ${report.consumedCount} tensors '
    '(${report.unusedKeys.length} unused) in '
    '${sw.elapsedMilliseconds} ms.',
  );

  // Encode corpus (under noGrad so no autograd tape is retained).
  sw.reset();
  final corpusVecs = <Float32List>[];
  Tensor.noGrad(() {
    for (final passage in _corpus) {
      final ids = tok.encode(passage, maxLength: 256);
      corpusVecs.add(_rowVec(encoder(_toTokens(ids))));
    }
  });
  stdout.writeln(
    'Encoded ${_corpus.length} passages in ${sw.elapsedMilliseconds} ms.',
  );

  // Build a brute-force inner-product index. Because the sentence
  // embeddings are already L2-normalized, inner product == cosine
  // similarity, so top-K by IP == top-K by cosine.
  final index = IndexFlat(encoder.embedDim, Metric.innerProduct);
  index.add(corpusVecs);

  // ------- run queries -------
  for (final query in _queries) {
    stdout.writeln('');
    stdout.writeln('Q: $query');
    final ids = tok.encode(query, maxLength: 256);
    final qVec = Tensor.noGrad(() => _rowVec(encoder(_toTokens(ids))));
    final res = index.search([qVec], 3);
    for (int i = 0; i < 3; i++) {
      final rank = i + 1;
      final score = res.distances[0][i];
      final id = res.ids[0][i];
      if (id < 0) continue;
      stdout.writeln(
        '  #$rank  (cos=${score.toStringAsFixed(3)})  ${_corpus[id]}',
      );
    }
    final bestId = res.ids[0][0];
    if (bestId >= 0) {
      stdout.writeln('A: ${_corpus[bestId]}');
    }
  }
}
