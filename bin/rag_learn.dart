/// Learn-by-example fine-tuning of a sentence-transformer.
///
/// The pretrained `all-MiniLM-L6-v2` (loaded via `BertHFLoader`) is
/// already a solid encoder — but on domain-specific data a few
/// hundred training steps of in-batch contrastive learning can move
/// the needle. This demo shows the full loop:
///
///   1. Load MiniLM + WordPiece tokenizer.
///   2. Encode a small corpus and a set of labelled queries.
///   3. Report baseline retrieval accuracy (top-1) and the cosine
///      similarity to each query's true positive.
///   4. Fine-tune the encoder end-to-end with
///      [SentenceLosses.multipleNegativesRankingLoss] — the standard
///      sentence-transformers loss for `(query, passage)` pairs. All
///      other passages in the batch act as in-batch negatives.
///   5. Re-evaluate and print the improvement.
///
/// This is a tiny toy training run (Dart's CPU autograd + WSL make
/// full BERT backward slow), so numbers are illustrative — the point
/// is that the loss reliably drops and gradient signal flows all the
/// way through the pretrained backbone.
///
/// Prerequisites (one-time MiniLM download, ~87 MB):
///
///   mkdir -p models/minilm && cd models/minilm
///   for f in config.json tokenizer_config.json vocab.txt \
///            model.safetensors; do
///     curl -sSL -O \
///       "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/$f"
///   done
///
/// Usage:
///
///   dart run bin/rag_learn.dart               # 20 fine-tuning steps
///   dart run bin/rag_learn.dart --steps 5     # or configure however many
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';

const _weightsPath = 'models/minilm/model.safetensors';
const _vocabPath = 'models/minilm/vocab.txt';

// Corpus + labelled queries. `_queryPositive[i]` is the corpus index
// that answers `_queries[i]`, so we can measure retrieval@1 exactly
// and use the same passages as MNRL positives during fine-tuning.
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
  "Who invented the Turing machine?",
  "What is the Eiffel Tower?",
  "How do plants make food from sunlight?",
  "How tall is Mount Everest?",
  "Which programming language is known for readability?",
  "What is the Great Wall of China?",
  "What does insulin do in the body?",
  "Where is the largest tropical rainforest?",
  "Who formulated the laws of motion?",
];

const _queryPositive = <int>[0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

Tensor _toTokens(List<int> ids) =>
    Tensor.fromList([ids.length], ids.map((i) => i.toDouble()).toList());

Float32List _rowVec(Tensor t) => Float32List.fromList(t.toList());

int _parseSteps(List<String> args) {
  for (int i = 0; i < args.length - 1; i++) {
    if (args[i] == '--steps') return int.tryParse(args[i + 1]) ?? 20;
  }
  return 20;
}

class _EvalReport {
  final int correct;
  final double meanCosPositive;
  final double meanRank;
  _EvalReport(this.correct, this.meanCosPositive, this.meanRank);
}

_EvalReport _evaluate(
  SentenceEncoder encoder,
  WordPieceTokenizer tok,
  List<Float32List> corpusVecs,
  IndexFlat index,
) {
  var correct = 0;
  var sumPosSim = 0.0;
  var sumRank = 0.0;
  Tensor.noGrad(() {
    for (int q = 0; q < _queries.length; q++) {
      final ids = tok.encode(_queries[q], maxLength: 256);
      final qVec = _rowVec(encoder(_toTokens(ids)));

      final res = index.search([qVec], _corpus.length);
      if (res.ids[0][0] == _queryPositive[q]) correct++;

      for (int r = 0; r < _corpus.length; r++) {
        if (res.ids[0][r] == _queryPositive[q]) {
          sumRank += (r + 1);
          break;
        }
      }

      final pos = corpusVecs[_queryPositive[q]];
      var s = 0.0;
      for (int j = 0; j < qVec.length; j++) {
        s += qVec[j] * pos[j];
      }
      sumPosSim += s;
    }
  });
  return _EvalReport(
    correct,
    sumPosSim / _queries.length,
    sumRank / _queries.length,
  );
}

List<Float32List> _encodeCorpus(
  SentenceEncoder encoder,
  WordPieceTokenizer tok,
) {
  final out = <Float32List>[];
  Tensor.noGrad(() {
    for (final p in _corpus) {
      final ids = tok.encode(p, maxLength: 256);
      out.add(_rowVec(encoder(_toTokens(ids))));
    }
  });
  return out;
}

void _fineTune(SentenceEncoder encoder, WordPieceTokenizer tok, int steps) {
  final anchors = <Tensor>[
    for (final q in _queries) _toTokens(tok.encode(q, maxLength: 256)),
  ];
  final positives = <Tensor>[
    for (final i in _queryPositive)
      _toTokens(tok.encode(_corpus[i], maxLength: 256)),
  ];

  final opt = Adam(encoder.parameters(), lr: 5e-5);
  encoder.train();

  final order = List<int>.generate(anchors.length, (i) => i);
  final rng = math.Random(0);

  final sw = Stopwatch()..start();
  for (int step = 0; step < steps; step++) {
    order.shuffle(rng);
    final aBatch = [for (final i in order) anchors[i]];
    final pBatch = [for (final i in order) positives[i]];

    opt.zeroGrad();
    final a = encoder.encodeBatch(aBatch);
    final p = encoder.encodeBatch(pBatch);
    final loss = SentenceLosses.multipleNegativesRankingLoss(a, p);
    loss.backward();
    opt.step();

    final ms = sw.elapsedMilliseconds;
    stdout.writeln(
      '  step ${(step + 1).toString().padLeft(3)}: '
      'loss=${loss.toList()[0].toStringAsFixed(4)}   '
      '(${ms} ms elapsed, '
      '${(ms / (step + 1)).toStringAsFixed(0)} ms/step)',
    );
  }
  encoder.eval();
}

void main(List<String> args) {
  if (!File(_weightsPath).existsSync() || !File(_vocabPath).existsSync()) {
    stderr.writeln(
      'rag_learn: missing $_weightsPath or $_vocabPath.\n'
      'Download the model first:\n'
      '  mkdir -p models/minilm && cd models/minilm && \\\n'
      '  for f in config.json tokenizer_config.json vocab.txt \\\n'
      '           model.safetensors; do \\\n'
      '    curl -sSL -O \\\n'
      '      "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/\$f"; \\\n'
      '  done',
    );
    exit(64);
  }

  final steps = _parseSteps(args);

  stdout.writeln('Loading all-MiniLM-L6-v2 ...');
  final sw = Stopwatch()..start();
  final tok = WordPieceTokenizer.fromVocabFile(_vocabPath);
  final bert = BertModel(BertHFLoader.miniLmL6V2Config());
  final report = BertHFLoader.loadFile(bert, _weightsPath);
  final encoder = SentenceEncoder.wrap(bert);
  stdout.writeln(
    '  ${report.consumedCount} tensors loaded in '
    '${sw.elapsedMilliseconds} ms.',
  );

  // ---- baseline ----
  encoder.eval();
  stdout.writeln('\nEncoding corpus (baseline) ...');
  sw.reset();
  var corpusVecs = _encodeCorpus(encoder, tok);
  var index = IndexFlat(encoder.embedDim, Metric.innerProduct);
  index.add(corpusVecs);
  stdout.writeln(
    '  ${_corpus.length} passages in ${sw.elapsedMilliseconds} ms.',
  );

  stdout.writeln('\n== Baseline retrieval ==');
  final before = _evaluate(encoder, tok, corpusVecs, index);
  stdout.writeln(
    'top-1 accuracy: ${before.correct}/${_queries.length}   '
    'mean cos(q, pos+): ${before.meanCosPositive.toStringAsFixed(3)}   '
    'mean rank(pos+): ${before.meanRank.toStringAsFixed(2)}',
  );

  // ---- fine-tune ----
  stdout.writeln(
    '\nFine-tuning with MultipleNegativesRankingLoss '
    '($steps steps, batch=${_queries.length}, lr=5e-5) ...',
  );
  _fineTune(encoder, tok, steps);

  // ---- post-training ----
  stdout.writeln('\nRe-encoding corpus with fine-tuned encoder ...');
  sw.reset();
  corpusVecs = _encodeCorpus(encoder, tok);
  index = IndexFlat(encoder.embedDim, Metric.innerProduct);
  index.add(corpusVecs);
  stdout.writeln(
    '  ${_corpus.length} passages in ${sw.elapsedMilliseconds} ms.',
  );

  stdout.writeln('\n== Fine-tuned retrieval ==');
  final after = _evaluate(encoder, tok, corpusVecs, index);
  stdout.writeln(
    'top-1 accuracy: ${after.correct}/${_queries.length}   '
    'mean cos(q, pos+): ${after.meanCosPositive.toStringAsFixed(3)}   '
    'mean rank(pos+): ${after.meanRank.toStringAsFixed(2)}',
  );

  stdout.writeln('\n== Delta ==');
  final dTop1 = after.correct - before.correct;
  final dCos = after.meanCosPositive - before.meanCosPositive;
  stdout.writeln(
    'top-1: ${before.correct} -> ${after.correct} '
    '(${dTop1 >= 0 ? '+' : ''}$dTop1)   '
    'mean cos(q, pos+): '
    '${before.meanCosPositive.toStringAsFixed(3)} -> '
    '${after.meanCosPositive.toStringAsFixed(3)} '
    '(${dCos >= 0 ? '+' : ''}${dCos.toStringAsFixed(3)})',
  );
}
