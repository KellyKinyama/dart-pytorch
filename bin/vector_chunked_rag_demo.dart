/// Chunked long-document RAG.
///
/// The R1/R2 `rag_qa_demo` cheated by keeping each "document" to a
/// single short paragraph — small enough to embed as one vector and
/// small enough to stuff five of them into distilgpt2's 1024-token
/// context. Real corpora don't cooperate: a Wikipedia article, a paper
/// or a book chapter is 5-50k tokens long. Embedding it as one vector
/// throws away paragraph-level structure and blows past the context
/// window when retrieved.
///
/// This demo does what production RAG actually does:
///
///   1. **Chunk** each document into overlapping ~200-token windows
///      (stride 100 → 50% overlap so a fact straddling a boundary
///      still ends up whole in at least one chunk).
///   2. **Embed** each chunk with the same last-token + mean-centre
///      recipe from [bin/_lm_encoder.dart](_lm_encoder.dart) — but
///      passing token ids directly, no encode-decode roundtrip.
///   3. **Index** every chunk in a single `IndexFlatIP`, keeping a
///      side-table that maps chunk id → `(doc id, span, text)`.
///   4. **Retrieve** at chunk granularity: `top-K` chunks may all
///      come from one document or be spread across several.
///   5. **Group** hits back by parent doc for display, then stuff the
///      chunks into the prompt for `.generate(...)`.
///
/// Corpus below is three long-form articles (~500-800 tokens each)
/// on distinct topics with deliberate topic drift within each doc.
/// The `_questions` target material buried in the middle of a doc —
/// the exact case that doc-level embedding would miss.
///
/// Run:
///
/// ```sh
///   dart run bin/vector_chunked_rag_demo.dart
/// ```
///
/// GPU / larger checkpoint:
///
/// ```sh
///   LD_LIBRARY_PATH=/usr/lib/wsl/lib \
///     dart run bin/vector_chunked_rag_demo.dart \
///       --path models/gpt2-medium/model.safetensors \
///       --vocab models/gpt2-medium/tokenizer.json \
///       --preset medium --gpu
/// ```
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';

import '_lm_encoder.dart';

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const int _chunkTokens = 200;
const int _chunkStride = 100; // 50% overlap
const int _topKChunks = 4;
const int _maxNewTokens = 80;
const double _temperature = 0.7;
const int _generatorTopK = 40;

// ---------------------------------------------------------------------------
// Corpus — three long-form articles. Each has deliberate topic drift so
// facts near the end aren't findable by embedding the first paragraph.
// ---------------------------------------------------------------------------

const List<({String title, String body})> _corpus = [
  (
    title: 'Roman Empire',
    body:
        'The Roman Empire began in 27 BC when Octavian was granted the title '
        'Augustus by the Senate, marking the end of the Roman Republic. Rome '
        'itself, according to legend, was founded in 753 BC by the twins '
        'Romulus and Remus who had been suckled by a she-wolf. For centuries '
        'before the empire proper Rome was a republic governed by two annually '
        'elected consuls and a senate of patrician families. '
        'The Punic Wars, fought between 264 BC and 146 BC against Carthage in '
        'North Africa, decided which of the two powers would dominate the '
        'western Mediterranean. Rome won all three wars, most famously through '
        'the general Scipio Africanus who defeated Hannibal at the battle of '
        'Zama in 202 BC. The victory gave Rome control of Spain, Sicily and '
        'eventually North Africa itself. '
        'Under the Pax Romana, the roughly two centuries of relative peace '
        'that followed Augustus, the empire built roads, aqueducts and cities '
        'from Britain to Mesopotamia. Latin displaced Greek as the language '
        'of law and administration in the western provinces while remaining '
        'the language of the army throughout the empire. '
        'The traditional date for the fall of the western empire is 476 AD, '
        'when the Germanic chieftain Odoacer deposed the last western emperor '
        'Romulus Augustulus. The eastern half, centred on Constantinople and '
        'later called the Byzantine Empire, survived for another thousand '
        'years until Constantinople fell to the Ottomans in 1453.',
  ),
  (
    title: 'Neural Networks',
    body:
        'The perceptron, introduced by Frank Rosenblatt in 1958, was the '
        'first trainable artificial neural network. It was a single-layer '
        'linear classifier and could only separate linearly separable data — '
        'a limitation famously exposed by Minsky and Papert in 1969. '
        'Multi-layer perceptrons, trained by backpropagation as popularised '
        'by Rumelhart, Hinton and Williams in 1986, solved the linear-'
        'separability problem but were hard to train deep because gradients '
        'vanished or exploded through many layers. '
        'Convolutional neural networks, going back to Fukushima\'s Neocognitron '
        'and LeCun\'s LeNet in 1989, apply the same small filter across an '
        'input at different positions. This weight sharing gave them '
        'translation equivariance and drastically fewer parameters. '
        'AlexNet in 2012 showed that a deep CNN trained on a GPU could win '
        'ImageNet by a wide margin, kicking off the modern deep-learning boom. '
        'Recurrent neural networks including LSTMs handle sequences by '
        'passing hidden state along a time dimension, but their sequential '
        'update makes them slow to train and limited in the range of '
        'dependencies they can capture. '
        'The transformer, introduced by Vaswani et al in 2017 in the paper '
        'Attention Is All You Need, replaces recurrence with self-attention. '
        'Its key advantage over RNNs is parallelism: every position in a '
        'sequence attends to every other position in one matrix multiply, so '
        'training scales linearly with hardware while also modelling long-'
        'range dependencies directly.',
  ),
  (
    title: 'Apollo Program',
    body:
        'The Apollo program was announced by President Kennedy in a speech to '
        'a joint session of Congress in May 1961, setting the goal of landing '
        'a man on the Moon and returning him safely before the decade was out. '
        'The impetus was the Soviet lead in space: Yuri Gagarin had orbited '
        'the Earth the month before, and Sputnik had shocked the American '
        'public in 1957. '
        'Early Apollo missions tested hardware in low Earth orbit. Apollo 1 '
        'never flew — a cabin fire during a launch rehearsal in January 1967 '
        'killed astronauts Grissom, White and Chaffee. Apollo 7 in October '
        '1968 was the first crewed Apollo mission, testing the command module '
        'in Earth orbit. '
        'Apollo 8 was the first crewed mission to orbit the Moon, in December '
        '1968. The crew — Borman, Lovell and Anders — read from Genesis on '
        'Christmas Eve during a live TV broadcast, and Anders took the '
        'famous Earthrise photograph. Apollo 10 in May 1969 was a full dress '
        'rehearsal that took the lunar module down to within 15 kilometres '
        'of the surface. '
        'Apollo 11 landed Armstrong and Aldrin in the Sea of Tranquility on '
        'July 20 1969 while Collins orbited above. Five more successful '
        'landings followed — Apollo 12, 14, 15, 16 and 17. Apollo 13 famously '
        'aborted its landing after an oxygen tank explosion and returned safely '
        'to Earth. Apollo 17 in December 1972 was the last crewed mission to '
        'the Moon; no human has been back since.',
  ),
];

// ---------------------------------------------------------------------------
// Questions targeting facts buried in the MIDDLE of a doc — the case
// that doc-level embedding would miss.
// ---------------------------------------------------------------------------

const List<String> _questions = [
  'When did the Punic Wars happen and who won?',
  "What is a transformer's key advantage over recurrent networks?",
  'Which Apollo mission first orbited the Moon and when?',
  'Who took the Earthrise photograph?',
  'What year did Constantinople fall to the Ottomans?',
];

// ---------------------------------------------------------------------------

class _Chunk {
  _Chunk({
    required this.docId,
    required this.startTok,
    required this.endTok,
    required this.tokenIds,
    required this.text,
  });
  final int docId;
  final int startTok;
  final int endTok;
  final List<int> tokenIds;
  final String text;
}

void main(List<String> args) {
  final opts = parseEncoderArgs(args, programHelp: _help);
  final loaded = loadEncoder(opts);
  final model = loaded.model;
  final tokenizer = loaded.tokenizer;
  final cfg = loaded.config;

  // ---- 1. Chunk every document ----------------------------------

  stdout.writeln(
    '\nChunking ${_corpus.length} documents at '
    '$_chunkTokens tokens (stride $_chunkStride)...',
  );
  final chunks = <_Chunk>[];
  for (var d = 0; d < _corpus.length; d++) {
    final full = tokenizer.encode(_corpus[d].body);
    if (full.length <= _chunkTokens) {
      chunks.add(
        _Chunk(
          docId: d,
          startTok: 0,
          endTok: full.length,
          tokenIds: full,
          text: _corpus[d].body,
        ),
      );
      continue;
    }
    for (var start = 0; start < full.length; start += _chunkStride) {
      final end = math.min(start + _chunkTokens, full.length);
      final ids = full.sublist(start, end);
      chunks.add(
        _Chunk(
          docId: d,
          startTok: start,
          endTok: end,
          tokenIds: ids,
          text: tokenizer.decode(ids),
        ),
      );
      if (end == full.length) break;
    }
  }
  stdout.writeln(
    'Produced ${chunks.length} chunks from '
    '${_corpus.length} docs.',
  );
  for (var d = 0; d < _corpus.length; d++) {
    final n = chunks.where((c) => c.docId == d).length;
    stdout.writeln('  doc $d "${_corpus[d].title}"  -> $n chunk(s)');
  }

  // ---- 2. Embed every chunk (two-pass: raw -> mean -> centre) ---

  stdout.writeln('\nEmbedding ${chunks.length} chunks...');
  final sw = Stopwatch()..start();
  final raw = <Float32List>[];
  for (var i = 0; i < chunks.length; i++) {
    raw.add(lastTokenHidden(model, chunks[i].tokenIds));
    stdout.writeln(
      '  [${(i + 1).toString().padLeft(2)}/${chunks.length}] '
      'doc=${chunks[i].docId} span=${chunks[i].startTok}-${chunks[i].endTok}',
    );
  }
  final mean = meanVector(raw, cfg.embedDim);
  final index = IndexFlatIP(cfg.embedDim);
  for (final v in raw) {
    index.add([centerAndNormalize(v, mean)]);
  }
  sw.stop();
  stdout.writeln(
    'Indexed ${index.ntotal} chunk vectors in '
    '${sw.elapsed.inMilliseconds} ms.',
  );

  // ---- 3. Ask each question --------------------------------------

  for (final q in _questions) {
    stdout.writeln('\n=================================================');
    stdout.writeln('Q: $q');
    stdout.writeln('-------------------------------------------------');

    final qVec = embedQuery(model, tokenizer, q, mean);
    final res = index.search([qVec], _topKChunks);
    final ids = res.ids[0];
    final scores = res.distances[0];

    // Show which chunks (and thus which docs) fired.
    final grouped = <int, List<int>>{}; // docId -> chunk indices among ids
    for (var j = 0; j < ids.length; j++) {
      grouped.putIfAbsent(chunks[ids[j]].docId, () => <int>[]).add(j);
    }
    for (final entry in grouped.entries) {
      final docId = entry.key;
      final positions = entry.value;
      stdout.writeln(
        '  doc $docId "${_corpus[docId].title}"  '
        '${positions.length} chunk hit(s):',
      );
      for (final pos in positions) {
        final c = chunks[ids[pos]];
        final preview = c.text.length > 70
            ? '${c.text.substring(0, 70).replaceAll('\n', ' ')}...'
            : c.text.replaceAll('\n', ' ');
        stdout.writeln(
          '    #${pos + 1} span=${c.startTok}-${c.endTok}  '
          'cos=${scores[pos].toStringAsFixed(3)}  $preview',
        );
      }
    }

    // ---- 4. Build prompt from the retrieved chunks ---------------
    final ctxLines = <String>[];
    for (var j = 0; j < ids.length; j++) {
      ctxLines.add('[${j + 1}] ${chunks[ids[j]].text.trim()}');
    }
    final prompt = 'Context:\n${ctxLines.join('\n')}\n\nQuestion: $q\nAnswer:';
    final promptIds = tokenizer.encode(prompt);

    if (promptIds.length + _maxNewTokens > cfg.maxCtx) {
      // Drop lowest-scoring chunks until it fits.
      var trimmed = List<int>.from(ids);
      var trimmedScores = List<double>.from(scores);
      while (trimmed.length > 1) {
        trimmed.removeLast();
        trimmedScores.removeLast();
        final ctx = <String>[];
        for (var j = 0; j < trimmed.length; j++) {
          ctx.add('[${j + 1}] ${chunks[trimmed[j]].text.trim()}');
        }
        final p = 'Context:\n${ctx.join('\n')}\n\nQuestion: $q\nAnswer:';
        final pIds = tokenizer.encode(p);
        if (pIds.length + _maxNewTokens <= cfg.maxCtx) {
          _generateAndPrint(model, tokenizer, pIds);
          break;
        }
      }
    } else {
      _generateAndPrint(model, tokenizer, promptIds);
    }
  }
}

void _generateAndPrint(
  GPT model,
  HFBpeTokenizer tokenizer,
  List<int> promptIds,
) {
  final sw = Stopwatch()..start();
  final full = model.generate(
    promptIds.map((i) => i.toDouble()).toList(),
    maxNewTokens: _maxNewTokens,
    temperature: _temperature,
    topK: _generatorTopK,
  );
  sw.stop();

  final answerIds = full
      .sublist(promptIds.length)
      .map((d) => d.toInt())
      .toList();
  final answer = tokenizer.decode(answerIds).trim();
  stdout.writeln('A: $answer');
  stdout.writeln(
    '   (generated ${answerIds.length} tokens in '
    '${sw.elapsed.inMilliseconds} ms)',
  );
}

const String _help = '''
Chunked long-document RAG demo.

Usage:
  dart run bin/vector_chunked_rag_demo.dart [flags]

Flags:
  --path PATH      safetensors weights (default: models/distilgpt2/model.safetensors)
  --vocab PATH     tokenizer.json     (default: models/distilgpt2/tokenizer.json)
  --preset NAME    distilgpt2 | small | medium | large  (default: distilgpt2)
  --gpu            Run on CUDA (default: CPU).
  -h, --help       Print this message.
''';
