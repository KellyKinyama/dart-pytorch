/// General-purpose semantic-QA tool built on all-MiniLM-L6-v2 +
/// IndexFlat. Two subcommands:
///
///   dart run bin/qa.dart build --corpus <path> --out <dir>
///                              [--chunk-words 120] [--overlap-words 20]
///     `--corpus` accepts either a single `.txt` / `.md` file or a
///     directory (walked recursively; `.txt` and `.md` picked up).
///     Each file is split on blank lines into paragraphs; any
///     paragraph longer than `--chunk-words` is further sliced into
///     overlapping windows. Encodes every chunk with MiniLM and
///     writes `<dir>/{index.bin, passages.jsonl, meta.json}`.
///
///   dart run bin/qa.dart ask --index <dir> [--k 3] [--query "..."]
///     Loads the persisted index. Without `--query` drops into an
///     interactive REPL: type a question, hit Enter, see top-k
///     annotated with source file. With `--query "..."` prints the
///     top-k for that one question and exits.
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
  dart run bin/qa.dart build --corpus <path>  --out <dir>
                             [--chunk-words 120] [--overlap-words 20]
    <path> is either a .txt/.md file or a directory of them.

  dart run bin/qa.dart ask   --index <dir>    [--k 3] [--query "..."]
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
  final chunkWords = int.tryParse(args['chunk-words'] ?? '120') ?? 120;
  final overlapWords = int.tryParse(args['overlap-words'] ?? '20') ?? 20;

  final passages = _gatherPassages(
    corpusPath,
    chunkWords: chunkWords,
    overlapWords: overlapWords,
  );
  if (passages.isEmpty) {
    stderr.writeln('qa build: no passages found under $corpusPath');
    exit(1);
  }
  stdout.writeln(
    'qa build: ${passages.length} passages from $corpusPath '
    '(chunk=$chunkWords words, overlap=$overlapWords).',
  );

  final encoder = _Encoder.load();
  stdout.writeln('qa build: model loaded, encoding ...');

  final sw = Stopwatch()..start();
  final vecs = <Float32List>[];
  for (int i = 0; i < passages.length; i++) {
    vecs.add(encoder.embed(passages[i].text));
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
  // JSONL keeps text-with-newlines safe and carries per-passage source
  // provenance without a second sidecar file.
  final jsonl = StringBuffer();
  for (final p in passages) {
    jsonl.writeln(jsonEncode({'text': p.text, 'source': p.source}));
  }
  File('$outDir/passages.jsonl').writeAsStringSync(jsonl.toString());
  File('$outDir/meta.json').writeAsStringSync(
    jsonEncode({
      'model': 'sentence-transformers/all-MiniLM-L6-v2',
      'embedDim': encoder.enc.embedDim,
      'passages': passages.length,
      'metric': 'innerProduct',
      'chunkWords': chunkWords,
      'overlapWords': overlapWords,
    }),
  );
  stdout.writeln(
    'qa build: saved $outDir/{index.bin,passages.jsonl,meta.json} '
    '(${sw.elapsedMilliseconds} ms total).',
  );
}

class _Passage {
  final String text;
  final String source;
  _Passage(this.text, this.source);
}

List<_Passage> _gatherPassages(
  String path, {
  required int chunkWords,
  required int overlapWords,
}) {
  final entity = FileSystemEntity.typeSync(path);
  final files = <String>[];
  if (entity == FileSystemEntityType.directory) {
    for (final f in Directory(path).listSync(recursive: true)) {
      if (f is File) {
        final p = f.path.toLowerCase();
        if (p.endsWith('.txt') || p.endsWith('.md')) {
          files.add(f.path);
        }
      }
    }
    files.sort();
  } else if (entity == FileSystemEntityType.file) {
    files.add(path);
  } else {
    stderr.writeln('qa build: $path not found');
    exit(1);
  }

  final passages = <_Passage>[];
  for (final f in files) {
    final raw = File(f).readAsStringSync();
    // Split on blank lines -> logical paragraphs; each paragraph is
    // then word-chunked if it exceeds `chunkWords`.
    final paragraphs = raw
        .split(RegExp(r'\n\s*\n'))
        .map((p) => p.replaceAll(RegExp(r'\s+'), ' ').trim())
        .where((p) => p.isNotEmpty)
        .toList();
    for (final para in paragraphs) {
      for (final chunk in _wordChunks(para, chunkWords, overlapWords)) {
        passages.add(_Passage(chunk, f));
      }
    }
  }
  return passages;
}

Iterable<String> _wordChunks(
  String text,
  int chunkWords,
  int overlapWords,
) sync* {
  final words = text.split(' ').where((w) => w.isNotEmpty).toList();
  if (words.length <= chunkWords) {
    yield text;
    return;
  }
  final step = (chunkWords - overlapWords).clamp(1, chunkWords);
  var i = 0;
  while (i < words.length) {
    final end = (i + chunkWords).clamp(0, words.length);
    yield words.sublist(i, end).join(' ');
    if (end == words.length) break;
    i += step;
  }
}


Future<void> _ask(Map<String, String> args) async {
  final indexDir = args['index'];
  if (indexDir == null) {
    _usage();
    exit(64);
  }
  final k = int.tryParse(args['k'] ?? '3') ?? 3;

  final passages = <_Passage>[];
  final jsonlPath = '$indexDir/passages.jsonl';
  final legacyPath = '$indexDir/passages.txt';
  if (File(jsonlPath).existsSync()) {
    for (final line in File(jsonlPath).readAsLinesSync()) {
      if (line.isEmpty) continue;
      final m = jsonDecode(line) as Map<String, dynamic>;
      passages.add(_Passage(m['text'] as String, m['source'] as String? ?? ''));
    }
  } else if (File(legacyPath).existsSync()) {
    for (final line in File(legacyPath).readAsLinesSync()) {
      if (line.isEmpty) continue;
      passages.add(_Passage(line, ''));
    }
  } else {
    stderr.writeln('qa ask: no passages.jsonl or passages.txt under $indexDir');
    exit(1);
  }
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
      final p = passages[id];
      final tag = p.source.isEmpty ? '' : ' [${p.source}]';
      stdout.writeln(
        '  #${i + 1}  (cos=${score.toStringAsFixed(3)})$tag  ${p.text}',
      );
    }
    final best = res.ids[0][0];
    if (best >= 0) stdout.writeln('A: ${passages[best].text}');
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
