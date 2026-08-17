/// Retrieval-Augmented Question Answering demo — document analysis
/// with a pretrained language model + a flat vector index.
///
/// Pipeline (all three ingredients from this repo):
///
///   1. Load `distilgpt2` (82M) as BOTH:
///        * the encoder — its final transformer hidden state at the
///          **last token** is captured, corpus-mean-centred, and
///          L2-normalised. That gives a 768-dim document / query
///          embedding. Two tricks stacked here:
///            - **Last-token pooling** — under a causal mask only
///              the final position has attended to every previous
///              token, so it holds the summary. Mean-pooling on a
///              causal LM suffers from the "anisotropy cone" — every
///              vector lands ~0.99 similar and retrieval collapses.
///            - **Corpus-mean centring** — even the last-token
///              vectors share a strong common direction. Subtracting
///              the mean across the corpus (BERT-flow style) restores
///              differentiation. In this demo it spreads cosine
///              scores from a flat ~0.996 to 0.09-0.69.
///        * the generator — same model, standard `.generate(...)`
///          call with a prompt that stuffs the retrieved passages
///          in front of the question.
///   2. Encode every document in `_corpus` (short factual snippets
///      about different topics) and add it to an `IndexFlatIP`
///      (cosine similarity via inner product on unit vectors).
///   3. For each entry in `_questions`:
///        a. Encode the question with the same encoder — including
///           subtracting the same corpus mean before normalising.
///        b. `index.search([q], k=_topK)` → retrieved passage ids.
///        c. Build the prompt
///              "Context:\n[1] {doc1}\n[2] {doc2}\n...\nQuestion:
///               {q}\nAnswer:"
///        d. `model.generate(...)` → decode the appended tokens.
///
/// Chapter 11 of [../doc/vectors/README.md](../doc/vectors/README.md)
/// walks through the theory; this file is the end-to-end wiring.
///
/// **Quality notes.** `distilgpt2` (6 layers) is a plain LM, not
/// instruction-tuned and not a purpose-built sentence encoder. With
/// mean-centring the top-3 retrieval routinely surfaces the correct
/// passage but not always at rank 1. Bumping to `gpt2-medium` (24
/// layers) via `--preset medium` gives a visible quality lift. For
/// production RAG use a real sentence encoder (SBERT-style) or fine-
/// tune with a triplet objective on labelled pairs — see
/// [bin/vector_store_demo.dart](vector_store_demo.dart) for a
/// self-contained triplet fine-tune example.
///
/// Run (default distilgpt2 on CPU — safest starting point):
///
/// ```sh
///   dart run bin/rag_qa_demo.dart
/// ```
///
/// On WSL2 with the CUDA FFI backend built:
///
/// ```sh
///   LD_LIBRARY_PATH=/usr/lib/wsl/lib \
///     dart run bin/rag_qa_demo.dart --gpu
/// ```
///
/// Or point at a bigger checkpoint that shares the GPT-2 architecture
/// (gpt2 small, gpt2-medium, gpt2-large — anything supported by
/// `GPT2HFLoader`):
///
/// ```sh
///   dart run bin/rag_qa_demo.dart \
///     --path models/gpt2-medium/model.safetensors \
///     --vocab models/gpt2-medium/tokenizer.json \
///     --preset medium
/// ```
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';

// ---------------------------------------------------------------------------
// Corpus — a handful of short factual passages on distinct topics so
// retrieval quality is easy to eyeball.
// ---------------------------------------------------------------------------

const List<String> _corpus = [
  'The Eiffel Tower is a 330-metre wrought-iron lattice tower in Paris, '
      'France. It was completed in 1889 as the entrance arch for the World Fair.',
  'Photosynthesis is the process by which green plants convert sunlight, '
      'water and carbon dioxide into glucose and oxygen using chlorophyll.',
  'The Great Wall of China stretches over 21000 kilometres and was built '
      'across many centuries to defend Chinese states against nomadic invasions.',
  'Python is a high-level, interpreted programming language known for its '
      'clean syntax and extensive standard library, first released in 1991 by '
      'Guido van Rossum.',
  'Mount Everest is the highest mountain on Earth, standing 8848 metres '
      'above sea level on the border between Nepal and Tibet.',
  'Insulin is a peptide hormone produced by beta cells in the pancreas that '
      'regulates the absorption of glucose from the bloodstream into cells.',
  'The Amazon rainforest covers roughly 5.5 million square kilometres across '
      'nine South American countries and produces about 6% of the world oxygen.',
  'Albert Einstein published the special theory of relativity in 1905, which '
      'introduced the equation E equals m c squared relating energy and mass.',
  'The Pacific Ocean is the largest and deepest of Earth oceans, covering '
      'about 63 million square miles, more than all land combined.',
  'Dart is a client-optimized programming language developed by Google, used '
      'primarily to build Flutter mobile and web applications.',
];

// ---------------------------------------------------------------------------
// Questions to ask against the corpus.
// ---------------------------------------------------------------------------

const List<String> _questions = [
  'Who designed the Eiffel Tower and when was it completed?',
  'What hormone regulates blood sugar?',
  'How tall is Mount Everest?',
  'Which language did Guido van Rossum create?',
  'What did Einstein publish in 1905?',
];

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const int _topK = 3; // how many passages to retrieve per question.
const int _maxNewTokens = 60;
const double _temperature = 0.7;
const int _generatorTopK = 40;

// ---------------------------------------------------------------------------

void main(List<String> args) {
  final opts = _parseArgs(args);

  // ---- 1. Load model + tokenizer ---------------------------------

  final device = opts.gpu ? Device.GPU : Device.CPU;
  final cfg = _configForPreset(opts.preset, device);

  stdout.writeln(
    'Building distilgpt2-style GPT (preset=${opts.preset}, device='
    '${opts.gpu ? "gpu" : "cpu"}, embed=${cfg.embedDim}, '
    'layers=${cfg.numLayers}, heads=${cfg.numHeads})',
  );
  final model = GPT(cfg);

  stdout.writeln('Loading safetensors from ${opts.path} ...');
  final report = GPT2HFLoader.loadFile(model, opts.path);
  stdout.writeln('Loaded. $report');

  stdout.writeln('Loading tokenizer from ${opts.vocabPath}');
  final tokenizer = HFBpeTokenizer.loadFile(opts.vocabPath);

  model.eval();

  // ---- 2. Encode + index the corpus ------------------------------
  //
  // Two-pass build:
  //   pass 1 — collect the raw last-token hidden state for every doc.
  //   pass 2 — subtract the corpus mean (BERT-flow style centering,
  //            removes the anisotropy "common direction" that makes
  //            every raw causal-LM vector look ~0.99 similar), then
  //            L2-normalise and add to the flat IP index.

  stdout.writeln('\nEmbedding ${_corpus.length} passages (raw)...');
  final embedSw = Stopwatch()..start();
  final rawDocVecs = <Float32List>[];
  for (var i = 0; i < _corpus.length; i++) {
    final ids = tokenizer.encode(_corpus[i]);
    rawDocVecs.add(_lastTokenHidden(model, ids));
    stdout.writeln(
      '  [${(i + 1).toString().padLeft(2)}/${_corpus.length}] '
      '${_corpus[i].substring(0, 60)}...',
    );
  }

  final corpusMean = _meanVector(rawDocVecs, cfg.embedDim);

  final index = IndexFlatIP(cfg.embedDim);
  for (final v in rawDocVecs) {
    index.add([_centerAndNormalize(v, corpusMean)]);
  }
  embedSw.stop();
  stdout.writeln(
    'Indexed ${index.ntotal} vectors in ${embedSw.elapsed.inMilliseconds} ms '
    '(${(embedSw.elapsed.inMilliseconds / _corpus.length).toStringAsFixed(0)}'
    ' ms/doc).',
  );

  // ---- 3. Retrieve + generate for each question ------------------

  for (final question in _questions) {
    stdout.writeln('\n=================================================');
    stdout.writeln('Q: $question');
    stdout.writeln('-------------------------------------------------');

    // Retrieval — same encoder, same centering.
    final qIds = tokenizer.encode(question);
    final qRaw = _lastTokenHidden(model, qIds);
    final qVec = _centerAndNormalize(qRaw, corpusMean);
    final result = index.search([qVec], _topK);

    final retrieved = <String>[];
    for (var j = 0; j < _topK; j++) {
      final id = result.ids[0][j];
      final score = result.distances[0][j];
      retrieved.add(_corpus[id]);
      stdout.writeln(
        '  hit ${j + 1}  score=${score.toStringAsFixed(3)}  '
        '${_corpus[id].substring(0, 70)}...',
      );
    }

    // Prompt assembly
    final ctxLines = <String>[];
    for (var j = 0; j < retrieved.length; j++) {
      ctxLines.add('[${j + 1}] ${retrieved[j]}');
    }
    final prompt =
        'Context:\n${ctxLines.join('\n')}\n\nQuestion: $question\nAnswer:';
    final promptIds = tokenizer.encode(prompt);

    // Guard against overflowing the 1024-ctx window on distilgpt2.
    if (promptIds.length + _maxNewTokens > cfg.maxCtx) {
      stdout.writeln(
        '  ! prompt (${promptIds.length} tok) + $_maxNewTokens > '
        '${cfg.maxCtx}; truncating to top hit only.',
      );
      final short =
          'Context:\n[1] ${retrieved.first}\n\n'
          'Question: $question\nAnswer:';
      final shortIds = tokenizer.encode(short);
      _generateAndPrint(model, tokenizer, shortIds);
    } else {
      _generateAndPrint(model, tokenizer, promptIds);
    }
  }
}

// ---------------------------------------------------------------------------
// Embedding: run the transformer stack (no LM head), take the LAST
// token's hidden state. Returns a raw (uncentered, unnormalised)
// Float32List of length embedDim. Callers apply corpus-mean centering
// and L2-normalisation before indexing / querying.
// ---------------------------------------------------------------------------

Float32List _lastTokenHidden(GPT model, List<int> tokenIds) {
  if (tokenIds.isEmpty) {
    return Float32List(model.config.embedDim);
  }
  // Clip to maxCtx — long docs get their first `maxCtx` tokens
  // embedded. Good enough for the tiny corpus in this demo.
  final clipped = tokenIds.length > model.config.maxCtx
      ? tokenIds.sublist(0, model.config.maxCtx)
      : tokenIds;

  return Tensor.noGrad(() {
    final tokens = Tensor.fromList(
      [clipped.length],
      clipped.map((i) => i.toDouble()).toList(),
      device: model.config.device,
    );

    var h = model.tokenEmb(tokens); // [N, D]
    h = model.posEmb(h); // [N, D]
    h = model.embedDrop(h);
    final mask = clipped.length > 1
        ? causalMask(clipped.length, device: h.device)
        : null;
    h = model.encoder(h, mask: mask); // [N, D]

    final flat = h.toList(); // length N*D, row-major
    final d = model.config.embedDim;
    final n = clipped.length;

    // Last-token pooling: take row `n-1` of the [N, D] hidden state.
    // The last position is the only one that has attended to every
    // preceding token under the causal mask.
    final out = Float32List(d);
    final base = (n - 1) * d;
    for (var j = 0; j < d; j++) {
      out[j] = flat[base + j].toDouble();
    }
    return out;
  });
}

Float32List _meanVector(List<Float32List> vecs, int d) {
  final mean = Float32List(d);
  if (vecs.isEmpty) return mean;
  for (final v in vecs) {
    for (var j = 0; j < d; j++) {
      mean[j] += v[j];
    }
  }
  final n = vecs.length.toDouble();
  for (var j = 0; j < d; j++) {
    mean[j] /= n;
  }
  return mean;
}

Float32List _centerAndNormalize(Float32List v, Float32List mean) {
  final d = v.length;
  final out = Float32List(d);
  for (var j = 0; j < d; j++) {
    out[j] = v[j] - mean[j];
  }
  var sq = 0.0;
  for (var j = 0; j < d; j++) {
    sq += out[j] * out[j];
  }
  final norm = 1.0 / (sq > 1e-24 ? _sqrt(sq) : 1.0);
  for (var j = 0; j < d; j++) {
    out[j] *= norm;
  }
  return out;
}

double _sqrt(double x) {
  // dart:math would be simpler, but this file is already large; keep the
  // dependency footprint minimal.
  var g = x;
  for (var i = 0; i < 20; i++) {
    g = 0.5 * (g + x / g);
  }
  return g;
}

// ---------------------------------------------------------------------------
// Generation helper: takes prompt token ids, runs generate(), decodes
// the appended tokens, prints them.
// ---------------------------------------------------------------------------

void _generateAndPrint(
  GPT model,
  HFBpeTokenizer tokenizer,
  List<int> promptIds,
) {
  final promptDouble = promptIds.map((i) => i.toDouble()).toList();
  final genSw = Stopwatch()..start();
  final full = model.generate(
    promptDouble,
    maxNewTokens: _maxNewTokens,
    temperature: _temperature,
    topK: _generatorTopK,
  );
  genSw.stop();

  final answerIds = full
      .sublist(promptIds.length)
      .map((d) => d.toInt())
      .toList(growable: false);
  final answer = tokenizer.decode(answerIds).trim();

  stdout.writeln('A: $answer');
  stdout.writeln(
    '   (generated ${answerIds.length} tokens in '
    '${genSw.elapsed.inMilliseconds} ms)',
  );
}

// ---------------------------------------------------------------------------
// CLI parsing
// ---------------------------------------------------------------------------

class _Opts {
  _Opts({
    required this.path,
    required this.vocabPath,
    required this.preset,
    required this.gpu,
  });
  final String path;
  final String vocabPath;
  final String preset; // 'distilgpt2', 'small', 'medium', 'large'
  final bool gpu;
}

_Opts _parseArgs(List<String> args) {
  var path = 'models/distilgpt2/model.safetensors';
  var vocab = 'models/distilgpt2/tokenizer.json';
  var preset = 'distilgpt2';
  var gpu = false;

  for (var i = 0; i < args.length; i++) {
    final a = args[i];
    String next() {
      if (i + 1 >= args.length) {
        stderr.writeln('missing value for $a');
        exit(64);
      }
      return args[++i];
    }

    switch (a) {
      case '--path':
        path = next();
        break;
      case '--vocab':
        vocab = next();
        break;
      case '--preset':
        preset = next();
        break;
      case '--gpu':
        gpu = true;
        break;
      case '-h':
      case '--help':
        stdout.writeln(_help);
        exit(0);
      default:
        stderr.writeln('unknown arg: $a');
        stderr.writeln(_help);
        exit(64);
    }
  }
  return _Opts(path: path, vocabPath: vocab, preset: preset, gpu: gpu);
}

const String _help = '''
Retrieval-Augmented Q&A demo.

Usage:
  dart run bin/rag_qa_demo.dart [flags]

Flags:
  --path PATH      safetensors weights (default: models/distilgpt2/model.safetensors)
  --vocab PATH     tokenizer.json     (default: models/distilgpt2/tokenizer.json)
  --preset NAME    distilgpt2 | small | medium | large  (default: distilgpt2)
                   Selects the GPTConfig factory; --path must match.
  --gpu            Run on CUDA (default: CPU).
  -h, --help       Print this message.
''';

GPTConfig _configForPreset(String preset, Device device) {
  switch (preset) {
    case 'distilgpt2':
      return GPT2HFLoader.distilGpt2Config(device: device);
    case 'small':
    case 'gpt2':
      return GPT2HFLoader.gpt2SmallConfig(device: device);
    case 'medium':
    case 'gpt2-medium':
      return GPT2HFLoader.gpt2MediumConfig(device: device);
    case 'large':
    case 'gpt2-large':
      return GPT2HFLoader.gpt2LargeConfig(device: device);
    default:
      stderr.writeln(
        'unknown preset "$preset"; use one of '
        'distilgpt2 | small | medium | large',
      );
      exit(64);
  }
}
