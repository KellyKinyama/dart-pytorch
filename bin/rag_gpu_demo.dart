/// MiniLM sentence embeddings on GPU vs CPU — apples-to-apples timing.
///
/// Loads sentence-transformers/all-MiniLM-L6-v2 twice (once on CPU,
/// once on GPU), encodes the same small corpus + query with both,
/// prints per-sentence encode timings and the top-3 hits.
///
/// This is the "did GPU inference actually help?" demo for the
/// non-conv transformer stack in this repo. MiniLM is a 6-layer,
/// 384-hidden BERT — matmul-dominated (attention + FFN), so the
/// GPU tiled-32x32 matmul kernel should pay off.
///
/// Prerequisites (from repo root):
///
///   mkdir -p models/minilm && cd models/minilm
///   for f in config.json tokenizer_config.json vocab.txt \
///            model.safetensors; do
///     curl -sSL -O \
///       "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/$f"
///   done
///
/// Usage:  dart run bin/rag_gpu_demo.dart
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';

const _weightsPath = 'models/minilm/model.safetensors';
const _vocabPath = 'models/minilm/vocab.txt';

const _corpus = <String>[
  'Ada Lovelace wrote the first computer program.',
  'Photosynthesis converts sunlight into glucose in plants.',
  'Mount Everest is the tallest mountain on Earth.',
  'Python is famous for its readability.',
  'Alan Turing formalized the theory of computation.',
  'The Amazon is the largest tropical rainforest in the world.',
  'Insulin regulates blood sugar in the body.',
  'The Great Wall of China stretches across northern China.',
  'Isaac Newton unified celestial and terrestrial mechanics.',
  'The Eiffel Tower was completed in 1889 in Paris.',
];

const _query = 'Who wrote the first computer program?';

Tensor _tokens(WordPieceTokenizer tok, String text, Device device) {
  final ids = tok.encode(text, maxLength: 128);
  final t = Tensor.fromList(
    [ids.length],
    ids.map((i) => i.toDouble()).toList(),
    device: Device.CPU,
  );
  return device == Device.CPU ? t : t.to(device);
}

Float32List _rowVec(Tensor t) => Float32List.fromList(t.toList());

class _Timed {
  final SentenceEncoder enc;
  final WordPieceTokenizer tok;
  final Device device;
  _Timed(this.enc, this.tok, this.device);

  ({Float32List vec, int ms}) encodeOne(String text) {
    final ids = _tokens(tok, text, device);
    final sw = Stopwatch()..start();
    final vec = Tensor.noGrad(() => _rowVec(enc(ids)));
    return (vec: vec, ms: sw.elapsedMilliseconds);
  }
}

_Timed _buildEncoder(Device device) {
  final tok = WordPieceTokenizer.fromVocabFile(_vocabPath);
  final bert = BertModel(BertHFLoader.miniLmL6V2Config(device: device));
  BertHFLoader.loadFile(bert, _weightsPath);
  final enc = SentenceEncoder.wrap(bert);
  enc.eval();
  return _Timed(enc, tok, device);
}

void _report(String label, _Timed t) {
  stdout.writeln('\n== $label (device=${t.device.name.toUpperCase()}) ==');

  // Warm-up pass — the first forward on GPU is skewed by JIT / library
  // load, so we drop it before measuring.
  t.encodeOne(_corpus[0]);

  final sw = Stopwatch()..start();
  final corpusVecs = <Float32List>[];
  final perSentenceMs = <int>[];
  for (final passage in _corpus) {
    final r = t.encodeOne(passage);
    corpusVecs.add(r.vec);
    perSentenceMs.add(r.ms);
  }
  final corpusMs = sw.elapsedMilliseconds;

  final q = t.encodeOne(_query);
  stdout.writeln(
    'corpus encode  : ${_corpus.length} sentences in $corpusMs ms  '
    '(median ${_median(perSentenceMs)} ms/sentence)',
  );
  stdout.writeln('query encode   : ${q.ms} ms');

  // Top-3 by cosine (embeddings are L2-normalized, so IP = cosine).
  final index = IndexFlat(t.enc.embedDim, Metric.innerProduct);
  index.add(corpusVecs);
  final res = index.search([q.vec], 3);
  stdout.writeln('Q: $_query');
  for (int i = 0; i < 3; i++) {
    final id = res.ids[0][i];
    final score = res.distances[0][i];
    stdout.writeln(
      '  #${i + 1} (cos=${score.toStringAsFixed(3)}) '
      '${_corpus[id]}',
    );
  }
}

int _median(List<int> xs) {
  final sorted = [...xs]..sort();
  return sorted[sorted.length ~/ 2];
}

void main() {
  if (!File(_weightsPath).existsSync() || !File(_vocabPath).existsSync()) {
    stderr.writeln(
      'rag_gpu_demo: missing $_weightsPath or $_vocabPath. See the header '
      'for the one-time download command.',
    );
    exit(64);
  }

  final loadSw = Stopwatch()..start();
  final cpu = _buildEncoder(Device.CPU);
  stdout.writeln('CPU model load: ${loadSw.elapsedMilliseconds} ms');
  loadSw.reset();
  final gpu = _buildEncoder(Device.GPU);
  stdout.writeln('GPU model load: ${loadSw.elapsedMilliseconds} ms');

  _report('CPU baseline', cpu);
  _report('GPU inference', gpu);
}
