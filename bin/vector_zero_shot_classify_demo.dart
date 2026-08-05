/// Zero-shot text classification via prompt embeddings.
///
/// The trick: we don't train a classifier. We describe each candidate
/// **label** as a short natural-language sentence ("This text is about
/// astronomy."), embed every label the same way we embed documents,
/// index the label vectors in a flat cosine index, and for each
/// document take the argmax of its similarity to every label.
///
/// This is the sentence-embedding analogue of "prompt engineering as
/// classification" popularised by SBERT / instructor-xl. It works with
/// any encoder that produces meaningful cosine geometry — including
/// our recycled causal GPT (last-token pool + corpus-mean centring;
/// see docs/vectors/12-RAG-NUTS-AND-BOLTS.md §12.3-12.4).
///
/// Pipeline:
///   1. Load a pretrained GPT-2 family model as an encoder.
///   2. Embed a fixed set of `_labels` — one string per candidate class.
///   3. Compute the shared corpus mean from the label vectors
///      + a bootstrap sample of the docs (so both live in the same
///      centred space) and normalise.
///   4. `IndexFlatIP.add(labelVectors)`.
///   5. For each doc → embed → centre → normalise → `index.search(k=3)`
///      → print top-3 labels with cosine scores.
///
/// Run:
///
/// ```sh
///   dart run bin/vector_zero_shot_classify_demo.dart
/// ```
///
/// GPU / larger checkpoint:
///
/// ```sh
///   LD_LIBRARY_PATH=/usr/lib/wsl/lib \
///     dart run bin/vector_zero_shot_classify_demo.dart \
///       --path models/gpt2-medium/model.safetensors \
///       --vocab models/gpt2-medium/tokenizer.json \
///       --preset medium --gpu
/// ```
library;

import 'dart:io';

import 'package:dart_pytorch/dart_pytorch.dart';

import '_lm_encoder.dart';

// ---------------------------------------------------------------------------
// Label descriptions and test documents.
// ---------------------------------------------------------------------------

/// Each entry: (short label, natural-language description used for encoding).
/// The description matters much more than the short label — it's the
/// text the encoder sees. Making it a full sentence in the same
/// register as the docs gives noticeably better geometry than a bare
/// word.
const List<({String label, String description})> _labels = [
  (
    label: 'astronomy',
    description:
        'This passage is about stars, planets, galaxies, black holes '
        'and cosmology — the science of the universe beyond Earth.',
  ),
  (
    label: 'biology',
    description:
        'This passage is about living organisms, cells, genetics, '
        'evolution, physiology and how life works at every scale.',
  ),
  (
    label: 'cooking',
    description:
        'This passage is about recipes, ingredients, kitchen '
        'techniques and preparing food.',
  ),
  (
    label: 'finance',
    description:
        'This passage is about money, banking, markets, investing, '
        'interest rates and economic policy.',
  ),
  (
    label: 'programming',
    description:
        'This passage is about writing computer code, software '
        'engineering, programming languages, algorithms and data structures.',
  ),
  (
    label: 'sports',
    description:
        'This passage is about athletic competition, teams, matches, '
        'training and physical games.',
  ),
];

const List<String> _docs = [
  'The Andromeda Galaxy is on a collision course with the Milky Way, expected '
      'to merge in about 4.5 billion years to form a giant elliptical galaxy.',
  'Photosynthesis converts sunlight, carbon dioxide and water into glucose '
      'inside chloroplasts, releasing oxygen as a byproduct.',
  'To make a proper carbonara you emulsify egg yolks and pecorino with the '
      'starchy pasta water off the heat; adding cream is heresy in Rome.',
  'Central banks raise interest rates to cool inflation by making borrowing '
      'more expensive and encouraging saving over spending.',
  'A binary search tree lets you find an element in a sorted collection in '
      'O(log n) time on average, assuming the tree stays balanced.',
  'Real Madrid won the Champions League final on penalties after a tense '
      '1-1 draw over 120 minutes of football at Wembley Stadium.',
  'CRISPR-Cas9 is a bacterial immune system repurposed as a programmable '
      'gene editor that can cut DNA at a specified 20-base target sequence.',
  'A saddle-point regression tests whether an option pricing model overfits '
      'by looking at the tail behaviour of the residuals in stress scenarios.',
  'Braise short ribs low and slow for four hours in red wine, aromatics and '
      'stock until the connective tissue turns to gelatin.',
  'The Perseverance rover has been drilling core samples in Jezero crater '
      'since 2021, caching them for a future Mars sample return mission.',
  'Rust enforces memory safety at compile time through its ownership and '
      'borrow-checker rules, eliminating whole classes of runtime bugs.',
  'She stuck the landing on a full-in, full-out double layout to secure the '
      'gold medal in the all-around gymnastics final.',
];

// ---------------------------------------------------------------------------

void main(List<String> args) {
  final opts = parseEncoderArgs(args, programHelp: _help);
  final loaded = loadEncoder(opts);
  final model = loaded.model;
  final tokenizer = loaded.tokenizer;
  final cfg = loaded.config;

  stdout.writeln(
    '\nEmbedding ${_labels.length} label descriptions and '
    '${_docs.length} documents...',
  );

  final sw = Stopwatch()..start();
  // Encode labels + docs together so they live in the same centred
  // space. Otherwise you'd fight the label vectors clustering apart
  // from the doc vectors along the anisotropy axis.
  final joint = <String>[for (final l in _labels) l.description, ..._docs];
  final embedded = embedCorpus(
    model,
    tokenizer,
    joint,
    onProgress: (i, total, preview) {
      final tag = i < _labels.length ? 'label ${_labels[i].label}' : 'doc';
      stdout.writeln(
        '  [${(i + 1).toString().padLeft(2)}/$total] $tag: '
        '$preview',
      );
    },
  );
  sw.stop();
  stdout.writeln('Embedded in ${sw.elapsed.inMilliseconds} ms.');

  // Split back into label vs doc vectors, then index the label side.
  final labelVecs = embedded.vectors.sublist(0, _labels.length);
  final docVecs = embedded.vectors.sublist(_labels.length);

  final index = IndexFlatIP(cfg.embedDim);
  index.add(labelVecs);

  // ---- classify --------------------------------------------------

  const topK = 3;
  var correctTop1 = 0;
  final total = _docs.length;

  for (var i = 0; i < _docs.length; i++) {
    final res = index.search([docVecs[i]], topK);
    final ids = res.ids[0];
    final scores = res.distances[0];

    stdout.writeln('\n---');
    final preview = _docs[i].length > 90
        ? '${_docs[i].substring(0, 90)}...'
        : _docs[i];
    stdout.writeln('DOC: $preview');
    for (var j = 0; j < topK; j++) {
      final lbl = _labels[ids[j]].label;
      final s = scores[j].toStringAsFixed(3);
      final marker = j == 0 ? '=>' : '  ';
      stdout.writeln('  $marker $lbl  cos=$s');
    }

    // Quick ground truth: whichever label word literally appears in
    // the expected topic. Not a rigorous eval; just eyeball-friendly.
    final expected = _expectedLabel(i);
    if (_labels[ids[0]].label == expected) correctTop1++;
  }

  stdout.writeln('\n=================================================');
  stdout.writeln('Top-1 agreement with expected: $correctTop1 / $total');
}

String _expectedLabel(int docIdx) {
  // Hand-labelled ground truth aligned to `_docs` order.
  const truth = <String>[
    'astronomy',
    'biology',
    'cooking',
    'finance',
    'programming',
    'sports',
    'biology',
    'finance',
    'cooking',
    'astronomy',
    'programming',
    'sports',
  ];
  return truth[docIdx];
}

const String _help = '''
Zero-shot text classification via prompt embeddings.

Usage:
  dart run bin/vector_zero_shot_classify_demo.dart [flags]

Flags:
  --path PATH      safetensors weights (default: models/distilgpt2/model.safetensors)
  --vocab PATH     tokenizer.json     (default: models/distilgpt2/tokenizer.json)
  --preset NAME    distilgpt2 | small | medium | large  (default: distilgpt2)
  --gpu            Run on CUDA (default: CPU).
  -h, --help       Print this message.
''';
