/// Semantic near-duplicate detection.
///
/// Vector embeddings turn "find the near-duplicates in this pile" from
/// an intractable pairwise-string-compare into an O(n) index scan.
/// Encode every document, index the embeddings, then for each doc
/// range-search the index for anything within a cosine threshold.
/// Group the resulting pairs into connected components — those are the
/// duplicate clusters.
///
/// The corpus below is intentionally seeded with:
///   * hard duplicates (identical rewordings)
///   * near-duplicates (same fact, different phrasing / detail)
///   * one-off originals that should end up in singleton clusters
///
/// so the output is easy to eyeball. Try changing `_threshold` to see
/// the precision/recall tradeoff: 0.65 catches near-duplicates, 0.90
/// catches only near-verbatim rewrites, 0.30 starts pulling in loosely
/// related passages.
///
/// Pipeline:
///   1. Load a pretrained GPT as an encoder.
///   2. `embedCorpus(...)` — last-token pool + corpus-mean centre +
///      L2-normalise, all as covered in
///      doc/vectors/12-RAG-NUTS-AND-BOLTS.md §12.3-12.4.
///   3. `IndexFlatIP(d).add(vecs)`.
///   4. `index.rangeSearch(vecs, radius=_threshold)` — for each doc
///      returns every other doc with cosine >= threshold.
///   5. Union-find on those pairs → duplicate clusters.
///
/// Run:
///
/// ```sh
///   dart run bin/vector_dedup_demo.dart
/// ```
///
/// GPU / larger checkpoint:
///
/// ```sh
///   LD_LIBRARY_PATH=/usr/lib/wsl/lib \
///     dart run bin/vector_dedup_demo.dart \
///       --path models/gpt2-medium/model.safetensors \
///       --vocab models/gpt2-medium/tokenizer.json \
///       --preset medium --gpu
/// ```
library;

import 'dart:io';

import 'package:dart_pytorch/dart_pytorch.dart';

import '_lm_encoder.dart';

// Cosine threshold above which two documents are considered "near
// duplicates". Distilgpt2 + mean-centred last-token cosines on this
// corpus land roughly in [-0.1, 0.9]; 0.55 empirically separates
// paraphrases from thematically related passages.
const double _threshold = 0.55;

const List<String> _corpus = [
  // 0-2 — three phrasings of "photosynthesis basics" (should cluster)
  'Photosynthesis is the process by which plants convert sunlight, water and '
      'carbon dioxide into glucose and oxygen using chlorophyll.',
  'Green plants use chlorophyll to turn light, water and CO2 into sugar and '
      'oxygen — a process called photosynthesis.',
  'In photosynthesis, chloroplasts absorb sunlight to produce glucose from '
      'water and carbon dioxide, releasing oxygen as a byproduct.',

  // 3-4 — two phrasings of "Eiffel Tower facts" (should cluster)
  'The Eiffel Tower in Paris is a 330-metre wrought-iron lattice tower '
      'completed in 1889 for the World Fair.',
  'Standing 330 metres tall in Paris, France, the wrought-iron Eiffel Tower '
      'was finished in 1889 as the entrance arch of the World Fair.',

  // 5-6 — two rewordings of "Insulin regulates blood sugar" (should cluster)
  'Insulin, produced by beta cells in the pancreas, regulates the uptake of '
      'glucose from the bloodstream into cells.',
  'The pancreas releases the peptide hormone insulin to control how cells '
      'absorb sugar from the blood.',

  // 7-9 — three genuinely distinct originals (should stay singletons)
  'The Great Wall of China stretches over 21000 kilometres and was built '
      'across many centuries to defend against nomadic invasions.',
  'Python is a high-level programming language first released in 1991 by '
      'Guido van Rossum, known for its clean syntax.',
  'Mount Everest, on the Nepal-Tibet border, is the highest mountain on '
      'Earth at 8848 metres above sea level.',
];

void main(List<String> args) {
  final opts = parseEncoderArgs(args, programHelp: _help);
  final loaded = loadEncoder(opts);
  final model = loaded.model;
  final tokenizer = loaded.tokenizer;
  final cfg = loaded.config;

  stdout.writeln('\nEmbedding ${_corpus.length} passages...');
  final sw = Stopwatch()..start();
  final emb = embedCorpus(
    model,
    tokenizer,
    _corpus,
    onProgress: (i, total, preview) {
      stdout.writeln('  [${(i + 1).toString().padLeft(2)}/$total] $preview');
    },
  );
  sw.stop();
  stdout.writeln('Embedded in ${sw.elapsed.inMilliseconds} ms.');

  final index = IndexFlatIP(cfg.embedDim);
  index.add(emb.vectors);

  // ---- range search for every doc --------------------------------

  stdout.writeln('\nRange-search cosine >= $_threshold ...');
  final result = index.rangeSearch(emb.vectors, _threshold).toRows();

  // ---- collect (a, b) pairs where a < b and score > threshold ----

  final pairs = <({int a, int b, double score})>[];
  for (var qi = 0; qi < _corpus.length; qi++) {
    final ids = result.ids[qi];
    final scores = result.distances[qi];
    for (var j = 0; j < ids.length; j++) {
      final other = ids[j];
      if (other <= qi) continue; // dedup (i,j) vs (j,i); skip self
      pairs.add((a: qi, b: other, score: scores[j]));
    }
  }
  pairs.sort((x, y) => y.score.compareTo(x.score));

  stdout.writeln('\nDuplicate pairs (cos >= $_threshold):');
  if (pairs.isEmpty) {
    stdout.writeln('  (none — try lowering --threshold)');
  } else {
    for (final p in pairs) {
      stdout.writeln('  ${p.a} <-> ${p.b}   cos=${p.score.toStringAsFixed(3)}');
    }
  }

  // ---- union-find to form clusters -------------------------------

  final parent = List<int>.generate(_corpus.length, (i) => i);
  int find(int x) {
    var r = x;
    while (parent[r] != r) {
      r = parent[r];
    }
    // path compression
    var cur = x;
    while (parent[cur] != r) {
      final next = parent[cur];
      parent[cur] = r;
      cur = next;
    }
    return r;
  }

  void union(int x, int y) {
    final rx = find(x);
    final ry = find(y);
    if (rx != ry) parent[rx] = ry;
  }

  for (final p in pairs) {
    union(p.a, p.b);
  }

  final clusters = <int, List<int>>{};
  for (var i = 0; i < _corpus.length; i++) {
    clusters.putIfAbsent(find(i), () => <int>[]).add(i);
  }

  stdout.writeln('\nClusters:');
  var cid = 0;
  final entries = clusters.values.toList()
    ..sort((a, b) => b.length.compareTo(a.length));
  for (final members in entries) {
    cid++;
    final tag = members.length > 1 ? 'DUPLICATES' : 'unique';
    stdout.writeln('  #$cid [$tag] members=${members.join(", ")}');
    for (final m in members) {
      final preview = _corpus[m].length > 90
          ? '${_corpus[m].substring(0, 90)}...'
          : _corpus[m];
      stdout.writeln('     [$m] $preview');
    }
  }

  final dupSize = pairs.isNotEmpty
      ? entries.where((c) => c.length > 1).length
      : 0;
  stdout.writeln('\n=================================================');
  stdout.writeln(
    '${_corpus.length} passages -> '
    '${entries.length} clusters ($dupSize with duplicates).',
  );
}

const String _help = '''
Semantic near-duplicate detection via range search.

Usage:
  dart run bin/vector_dedup_demo.dart [flags]

Flags:
  --path PATH      safetensors weights (default: models/distilgpt2/model.safetensors)
  --vocab PATH     tokenizer.json     (default: models/distilgpt2/tokenizer.json)
  --preset NAME    distilgpt2 | small | medium | large  (default: distilgpt2)
  --gpu            Run on CUDA (default: CPU).
  -h, --help       Print this message.
''';
