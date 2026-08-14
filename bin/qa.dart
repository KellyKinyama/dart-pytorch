/// General-purpose semantic-QA tool built on all-MiniLM-L6-v2 +
/// IndexFlat. Two subcommands:
///
///   dart run bin/qa.dart build --corpus <path.txt> --out <dir>
///     One passage per non-empty line. Encodes each with MiniLM,
///     saves the index to <dir>/index.bin and the raw passages to
///     <dir>/passages.txt so the two stay in sync.
///
///   dart run bin/qa.dart ask --index <dir> [--k 3] [--query "..."]
///     Loads the persisted index. Without --query drops into an
///     interactive REPL: type a question, hit Enter, see top-k.
///     With --query "..." prints the top-k for that one question
///     and exits.
///
/// Prerequisites (one-time model download, ~87 MB):
///
///   mkdir -p models/minilm && cd models/minilm
///   for f in config.json tokenizer_config.json vocab.txt \
///            model.safetensors; do
///     curl -sSL -O \
///       "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/$f"
///   done
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';

const _weightsPath = 'models/minilm/model.safetensors';
const _vocabPath = 'models/minilm/vocab.txt';

Tensor _toTokens(List<int> ids) =>
    Tensor.fromList([ids.length], ids.map((i) => i.toDouble()).toList());

Float32List _rowVec(Tensor t) => Float32List.fromList(t.toList());

class _Encoder {
  final WordPieceTokenizer tok;
  final SentenceEncoder enc;
  _Encoder(this.tok, this.enc);

  static _Encoder load() {
    if (!File(_weightsPath).existsSync() || !File(_vocabPath).existsSync()) {
      stderr.writeln(
        'qa: missing $_weightsPath or $_vocabPath. See the header of\n'
        'bin/qa.dart for the one-time download commands.',
      );
      exit(64);
    }
    final tok = WordPieceTokenizer.fromVocabFile(_vocabPath);
    final bert = BertModel(BertHFLoader.miniLmL6V2Config());
    BertHFLoader.loadFile(bert, _weightsPath);
    final enc = SentenceEncoder.wrap(bert);
    enc.eval();
    return _Encoder(tok, enc);
  }

  Float32List embed(String text, {int maxLength = 256}) {
    final ids = tok.encode(text, maxLength: maxLength);
    return Tensor.noGrad(() => _rowVec(enc(_toTokens(ids))));
  }
}

void _usage() {
  stderr.writeln('''
Usage:
  dart run bin/qa.dart build --corpus <path.txt> --out <dir>
  dart run bin/qa.dart ask   --index <dir> [--k 3] [--query "..."]
''');
}

Map<String, String> _parseArgs(List<String> args) {
  final out = <String, String>{};
  for (int i = 0; i < args.length; i++) {
    final a = args[i];
    if (!a.startsWith('--')) continue;
    final key = a.substring(2);
    if (i + 1 < args.length && !args[i + 1].startsWith('--')) {
      out[key] = args[i + 1];
      i++;
    } else {
      out[key] = 'true';
    }
  }
  return out;
}

Future<void> _build(Map<String, String> args) async {
  final corpusPath = args['corpus'];
  final outDir = args['out'];
  if (corpusPath == null || outDir == null) {
    _usage();
    exit(64);
  }
  final passages = File(corpusPath)
      .readAsLinesSync()
      .map((l) => l.trim())
      .where((l) => l.isNotEmpty)
      .toList();
  if (passages.isEmpty) {
    stderr.writeln('qa build: $corpusPath has no non-empty lines.');
    exit(1);
  }
  stdout.writeln(
    'qa build: ${passages.length} passages from $corpusPath',
  );

  final encoder = _Encoder.load();
  stdout.writeln('qa build: model loaded, encoding ...');

  final sw = Stopwatch()..start();
  final vecs = <Float32List>[];
  for (int i = 0; i < passages.length; i++) {
    vecs.add(encoder.embed(passages[i]));
    if ((i + 1) % 25 == 0 || i == passages.length - 1) {
      stdout.writeln(
        '  ${i + 1}/${passages.length}  '
        '(${sw.elapsedMilliseconds} ms elapsed)',
      );
    }
  }

  final index = IndexFlat(encoder.enc.embedDim, Metric.innerProduct);
  index.add(vecs);

  Directory(outDir).createSync(recursive: true);
  await saveIndex(index, '$outDir/index.bin');
  File(
    '$outDir/passages.txt',
  ).writeAsStringSync(passages.map((p) => p).join('\n'));
  File('$outDir/meta.json').writeAsStringSync(
    jsonEncode({
      'model': 'sentence-transformers/all-MiniLM-L6-v2',
      'embedDim': encoder.enc.embedDim,
      'passages': passages.length,
      'metric': 'innerProduct',
    }),
  );
  stdout.writeln(
    'qa build: saved $outDir/{index.bin,passages.txt,meta.json} '
    '(${sw.elapsedMilliseconds} ms total).',
  );
}

Future<void> _ask(Map<String, String> args) async {
  final indexDir = args['index'];
  if (indexDir == null) {
    _usage();
    exit(64);
  }
  final k = int.tryParse(args['k'] ?? '3') ?? 3;

  final passages = File('$indexDir/passages.txt').readAsLinesSync();
  final index = await loadIndex('$indexDir/index.bin');
  final encoder = _Encoder.load();
  stdout.writeln(
    'qa ask: ${passages.length} passages loaded, embedDim '
    '${encoder.enc.embedDim}.',
  );

  void answer(String q) {
    final qVec = encoder.embed(q);
    final res = index.search([qVec], k);
    for (int i = 0; i < k; i++) {
      final id = res.ids[0][i];
      if (id < 0) break;
      final score = res.distances[0][i];
      stdout.writeln(
        '  #${i + 1}  (cos=${score.toStringAsFixed(3)})  '
        '${passages[id]}',
      );
    }
    final best = res.ids[0][0];
    if (best >= 0) stdout.writeln('A: ${passages[best]}');
  }

  final oneShot = args['query'];
  if (oneShot != null && oneShot != 'true') {
    stdout.writeln('Q: $oneShot');
    answer(oneShot);
    return;
  }

  stdout.writeln('Interactive QA. Ctrl-D or blank line to quit.');
  while (true) {
    stdout.write('\nQ: ');
    final line = stdin.readLineSync();
    if (line == null) break;
    final q = line.trim();
    if (q.isEmpty) break;
    answer(q);
  }
}

Future<void> main(List<String> args) async {
  if (args.isEmpty) {
    _usage();
    exit(64);
  }
  final cmd = args.first;
  final rest = _parseArgs(args.sublist(1));
  switch (cmd) {
    case 'build':
      await _build(rest);
    case 'ask':
      await _ask(rest);
    default:
      _usage();
      exit(64);
  }
}
