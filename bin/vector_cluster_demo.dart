/// Unsupervised topic discovery via k-means over sentence embeddings.
///
/// You don't need labels to find structure in a corpus. Embed every
/// document, run Lloyd's k-means on the vectors, and each cluster
/// centroid becomes a discovered "topic". This is the classic
/// unsupervised-clustering application of embeddings, popularised by
/// tools like BERTopic and top2vec.
///
/// The corpus below deliberately mixes ~24 short passages across four
/// distinct topics (astronomy, cooking, programming, sports) so with
/// `k=4` you can eyeball whether the clustering nailed it. The demo
/// also prints a rough **purity** score — the fraction of each
/// cluster's members that share the majority ground-truth topic. On
/// distilgpt2 you should see purity comfortably north of chance (25%);
/// bumping to gpt2-medium via `--preset medium` typically pushes it
/// above 85%.
///
/// Pipeline:
///   1. Load a pretrained GPT as an encoder.
///   2. `embedCorpus(...)` — last-token pool + mean-centre + L2-norm.
///   3. `Kmeans(d, k=4).train(vecs)` from
///      [lib/core/vector_store/kmeans.dart].
///   4. Print each cluster's members, then compute purity.
///
/// Run:
///
/// ```sh
///   dart run bin/vector_cluster_demo.dart
/// ```
///
/// GPU / larger checkpoint:
///
/// ```sh
///   LD_LIBRARY_PATH=/usr/lib/wsl/lib \
///     dart run bin/vector_cluster_demo.dart \
///       --path models/gpt2-medium/model.safetensors \
///       --vocab models/gpt2-medium/tokenizer.json \
///       --preset medium --gpu
/// ```
library;

import 'dart:io';

import 'package:dart_pytorch/dart_pytorch.dart';

import '_lm_encoder.dart';

// (text, ground-truth topic).
const List<({String text, String topic})> _corpus = [
  // ---- astronomy -----------------------------------------------------------
  (
    text: 'The Andromeda Galaxy is on a collision course with the Milky Way, '
        'expected to merge in about 4.5 billion years.',
    topic: 'astronomy',
  ),
  (
    text: 'A neutron star packs the mass of the Sun into a sphere only about '
        '20 kilometres across, giving it extreme surface gravity.',
    topic: 'astronomy',
  ),
  (
    text: 'The James Webb Space Telescope observes the universe in infrared '
        'from a stable orbit around the L2 Lagrange point.',
    topic: 'astronomy',
  ),
  (
    text: 'A black hole event horizon is the boundary beyond which not even '
        'light can escape the pull of the singularity.',
    topic: 'astronomy',
  ),
  (
    text: 'Jupiter has more than 90 known moons; the four largest were first '
        'observed by Galileo Galilei in 1610.',
    topic: 'astronomy',
  ),
  (
    text: 'The Cosmic Microwave Background is thermal radiation left over '
        'from the recombination epoch about 380,000 years after the Big Bang.',
    topic: 'astronomy',
  ),

  // ---- cooking -------------------------------------------------------------
  (
    text: 'To make proper carbonara you emulsify egg yolks and pecorino with '
        'starchy pasta water off the heat; adding cream is heresy in Rome.',
    topic: 'cooking',
  ),
  (
    text: 'Braise short ribs low and slow for four hours in red wine, '
        'aromatics and stock until the connective tissue turns to gelatin.',
    topic: 'cooking',
  ),
  (
    text: 'A good sourdough loaf uses only flour, water, salt and a wild-yeast '
        'starter, fermented slowly to develop flavour and structure.',
    topic: 'cooking',
  ),
  (
    text: 'Deglaze the pan with dry white wine after searing the shallots, '
        'scraping up the fond to build the base of the sauce.',
    topic: 'cooking',
  ),
  (
    text: 'Blanch green beans in heavily salted boiling water for two minutes, '
        'then shock them in ice water to lock in the bright colour.',
    topic: 'cooking',
  ),
  (
    text: 'Sear the steak over screaming-hot cast iron for a good crust, then '
        'rest it under foil for five minutes before slicing against the grain.',
    topic: 'cooking',
  ),

  // ---- programming ---------------------------------------------------------
  (
    text: 'A binary search tree lets you find an element in a sorted '
        'collection in O(log n) time on average when balanced.',
    topic: 'programming',
  ),
  (
    text: 'Rust enforces memory safety at compile time through its ownership '
        'and borrow-checker rules, eliminating whole classes of bugs.',
    topic: 'programming',
  ),
  (
    text: 'A hash table gives amortised O(1) lookup by mapping keys to bucket '
        'indices via a hash function, handling collisions with chaining.',
    topic: 'programming',
  ),
  (
    text: 'Immutability makes concurrent code easier to reason about because '
        'there is no shared mutable state that threads can race on.',
    topic: 'programming',
  ),
  (
    text: 'Dynamic programming solves overlapping-subproblem recurrences by '
        'memoising previously computed answers in a table.',
    topic: 'programming',
  ),
  (
    text: 'Compilers translate high-level source code into machine '
        'instructions after lexing, parsing, type checking and optimisation.',
    topic: 'programming',
  ),

  // ---- sports --------------------------------------------------------------
  (
    text: 'Real Madrid won the Champions League final on penalties after a '
        'tense 1-1 draw over 120 minutes of football at Wembley Stadium.',
    topic: 'sports',
  ),
  (
    text: 'She stuck the landing on a full-in, full-out double layout to '
        'secure the gold medal in the all-around gymnastics final.',
    topic: 'sports',
  ),
  (
    text: 'The marathon world record fell again this weekend as the runner '
        'crossed the finish line two seconds under the previous best.',
    topic: 'sports',
  ),
  (
    text: 'The starting pitcher struck out ten batters and gave up only two '
        'earned runs over seven innings on the mound.',
    topic: 'sports',
  ),
  (
    text: 'The point guard drove the lane and dished off a no-look pass for '
        'the alley-oop with three seconds left on the shot clock.',
    topic: 'sports',
  ),
  (
    text: 'The Grand Slam final went to a fifth-set tiebreak after four hours '
        'of grinding baseline tennis under the closed roof.',
    topic: 'sports',
  ),
];

const int _k = 4;

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
    _corpus.map((e) => e.text).toList(),
    onProgress: (i, total, preview) {
      stdout.writeln('  [${(i + 1).toString().padLeft(2)}/$total] '
          '(${_corpus[i].topic}) $preview');
    },
  );
  sw.stop();
  stdout.writeln('Embedded in ${sw.elapsed.inMilliseconds} ms.');

  // ---- k-means ---------------------------------------------------

  stdout.writeln('\nClustering with k=$_k (Lloyd, k-means++ init)...');
  final km = Kmeans(d: cfg.embedDim, k: _k, niter: 30, seed: 1234);
  final res = km.train(emb.vectors);
  stdout.writeln(
    'Done. objective (sum of squared L2 to centroid) = '
    '${res.objective.toStringAsFixed(3)}',
  );

  // ---- group members by cluster id -------------------------------

  final byCluster = <int, List<int>>{};
  for (var i = 0; i < _corpus.length; i++) {
    byCluster.putIfAbsent(res.assignments[i], () => <int>[]).add(i);
  }

  var totalCorrect = 0;
  for (var c = 0; c < _k; c++) {
    final members = byCluster[c] ?? const <int>[];
    if (members.isEmpty) {
      stdout.writeln('\nCluster #$c   (empty)');
      continue;
    }

    // Majority ground-truth topic in this cluster.
    final counts = <String, int>{};
    for (final m in members) {
      final t = _corpus[m].topic;
      counts[t] = (counts[t] ?? 0) + 1;
    }
    final sorted = counts.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));
    final majority = sorted.first.key;
    final purity = sorted.first.value / members.length;
    totalCorrect += sorted.first.value;

    stdout.writeln('\nCluster #$c   size=${members.length}   '
        'majority=$majority   purity=${(purity * 100).toStringAsFixed(0)}%');
    for (final m in members) {
      final t = _corpus[m].topic;
      final marker = t == majority ? '  ' : '! ';
      final preview = _corpus[m].text.length > 80
          ? '${_corpus[m].text.substring(0, 80)}...'
          : _corpus[m].text;
      stdout.writeln('  $marker[$m] ($t) $preview');
    }
  }

  final purity = totalCorrect / _corpus.length;
  stdout.writeln('\n=================================================');
  stdout.writeln(
    'Overall purity: $totalCorrect / ${_corpus.length} = '
    '${(purity * 100).toStringAsFixed(1)}%   (chance = '
    '${(100 / _k).toStringAsFixed(0)}%)',
  );
}

const String _help = '''
Unsupervised topic clustering via k-means over sentence embeddings.

Usage:
  dart run bin/vector_cluster_demo.dart [flags]

Flags:
  --path PATH      safetensors weights (default: models/distilgpt2/model.safetensors)
  --vocab PATH     tokenizer.json     (default: models/distilgpt2/tokenizer.json)
  --preset NAME    distilgpt2 | small | medium | large  (default: distilgpt2)
  --gpu            Run on CUDA (default: CPU).
  -h, --help       Print this message.
''';
