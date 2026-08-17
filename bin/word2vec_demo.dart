/// word2vec skip-gram + negative-sampling demo on the tiny-shakespeare
/// corpus, with nearest-neighbour queries backed by the vector-store
/// toolkit (IndexFlatIP over L2-normalised embeddings = cosine).
///
/// This is a Dart port of the TensorFlow tutorial at
/// https://www.tensorflow.org/text/tutorials/word2vec, adapted to the
/// primitives available in this package:
///
/// * two `Embedding(V, D)` tables — one for target words, one for
///   context words. The target table is what we keep after training.
/// * per-example logits are computed as `dot(target_vec, context_vec)`
///   over `1 + numNs` context slots (1 positive + numNs negatives),
///   yielding a `[batch, 1 + numNs]` matrix.
/// * loss = fused softmax + NLL cross-entropy (`crossEntropy`) with
///   integer targets of value `0` — semantically identical to the
///   `CategoricalCrossentropy(from_logits=True)` + one-hot label of
///   `[1, 0, 0, ..., 0]` that the TF tutorial uses.
/// * after training we pull the target embedding matrix to host,
///   L2-normalise, and add it to an `IndexFlatIP` — search returns the
///   top-k by cosine similarity for a handful of interesting query
///   words.
///
/// Run from the repository root:
///
///   dart run bin/word2vec_demo.dart
///
/// Optional flags:
///   --max-chars=N    truncate the corpus (default: 300_000)
///   --epochs=N       training epochs (default: 3)
///   --embed-dim=N    embedding dimension (default: 64)
///   --vocab-size=N   max vocab (default: 4096)
///   --batch=N        batch size (default: 512)
///   --lr=F           Adam learning rate (default: 5e-3)
///   --num-ns=N       negative samples per positive (default: 4)
///   --window=N       skip-gram window radius (default: 2)
///   --subsample-t=F  Mikolov subsample threshold (default: 1e-3;
///                    lower = drop frequent words more aggressively; use
///                    1e-5 only for gigabyte-scale corpora)
///   --queries=w,w,w  comma-separated words to query at the end
///   --device=cpu|gpu run everything on CPU (default) or CUDA GPU.
///                    GPU requires native/lib/libmat_mul.so — see
///                    scripts/setup_colab.sh + doc/colab.md for a
///                    free Colab / Kaggle GPU recipe.
///
/// Recipes (copy/paste ready):
///
///   # Default (fast smoke run — pipeline check, not meaningful clusters):
///   dart run bin/word2vec_demo.dart
///
///   # Bigger corpus + more epochs — decent semantic clusters:
///   dart run bin/word2vec_demo.dart --max-chars=1200000 --epochs=20 --window=4 --vocab-size=2048
///
///   # "Aggressive" small run — sharper discriminator, higher LR:
///   dart run bin/word2vec_demo.dart --max-chars=800000 --epochs=15 --window=5 --vocab-size=2048 --num-ns=8 --lr=0.01
///
///   # Full-corpus quality run (bigger dim, fewer epochs — avoids memorisation):
///   dart run bin/word2vec_demo.dart --max-chars=1200000 --epochs=8 --window=5 --vocab-size=4096 --embed-dim=128 --num-ns=8 --subsample-t=1e-3
///
///   # Run on a free Colab / Kaggle GPU (after scripts/setup_colab.sh):
///   dart run bin/word2vec_demo.dart --device=gpu --max-chars=1200000 --epochs=20 --embed-dim=128 --vocab-size=4096 --num-ns=8 --batch=1024
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const String _kCorpusPath = 'data/tiny_shakespeare.txt';
const int _kPadId = 0;
const int _kUnkId = 1;
const String _kPad = '<pad>';
const String _kUnk = '<unk>';

class _Config {
  int maxChars = 300000;
  int epochs = 3;
  int embedDim = 64;
  int vocabSize = 4096;
  int batchSize = 512;
  double lr = 5e-3;
  int numNs = 4;
  int windowSize = 2;
  double subsampleT = 1e-3;
  int seed = 42;
  Device device = Device.CPU;
  List<String> queries = const [
    'king',
    'queen',
    'love',
    'sword',
    'father',
    'night',
  ];
}

_Config _parseArgs(List<String> args) {
  final c = _Config();
  for (final raw in args) {
    final a = raw.startsWith('--') ? raw.substring(2) : raw;
    final eq = a.indexOf('=');
    if (eq < 0) continue;
    final k = a.substring(0, eq);
    final v = a.substring(eq + 1);
    switch (k) {
      case 'max-chars':
        c.maxChars = int.parse(v);
      case 'epochs':
        c.epochs = int.parse(v);
      case 'embed-dim':
        c.embedDim = int.parse(v);
      case 'vocab-size':
        c.vocabSize = int.parse(v);
      case 'batch':
        c.batchSize = int.parse(v);
      case 'lr':
        c.lr = double.parse(v);
      case 'num-ns':
        c.numNs = int.parse(v);
      case 'window':
        c.windowSize = int.parse(v);
      case 'subsample-t':
        c.subsampleT = double.parse(v);
      case 'queries':
        c.queries = v.split(',').where((s) => s.isNotEmpty).toList();
      case 'device':
        switch (v.toLowerCase()) {
          case 'cpu':
            c.device = Device.CPU;
          case 'gpu' || 'cuda':
            c.device = Device.GPU;
          default:
            stderr.writeln('word2vec_demo: --device must be cpu or gpu');
            exit(2);
        }
      default:
        stderr.writeln('word2vec_demo: unknown flag --$k');
        exit(2);
    }
  }
  return c;
}

// ---------------------------------------------------------------------------
// Corpus loading / tokenisation / vocab
// ---------------------------------------------------------------------------

String _loadCorpus(int maxChars) {
  final f = File(_kCorpusPath);
  if (!f.existsSync()) {
    stderr.writeln(
      'word2vec_demo: could not find $_kCorpusPath. '
      'Run from the repository root.',
    );
    exit(1);
  }
  final s = f.readAsStringSync();
  return s.length > maxChars ? s.substring(0, maxChars) : s;
}

/// Very small word tokeniser: lowercase, strip everything that is not a
/// letter or apostrophe, split on whitespace. Preserves line breaks so
/// callers can treat each line as an independent "sentence".
List<String> _tokenizeLine(String line) {
  final buf = StringBuffer();
  for (final rune in line.toLowerCase().runes) {
    final ch = String.fromCharCode(rune);
    final isLetter = (rune >= 0x61 && rune <= 0x7a); // a-z (already lower)
    final isApos = ch == "'";
    buf.write(isLetter || isApos ? ch : ' ');
  }
  return buf
      .toString()
      .split(RegExp(r'\s+'))
      .where((w) => w.isNotEmpty)
      .toList();
}

class _Vocab {
  _Vocab(this.itos, this.stoi, this.counts);
  final List<String> itos; // id -> word
  final Map<String, int> stoi; // word -> id
  final List<int> counts; // id -> raw token count in corpus

  int get size => itos.length;
}

_Vocab _buildVocab(List<List<String>> sentences, int vocabSize) {
  final freq = <String, int>{};
  for (final s in sentences) {
    for (final w in s) {
      freq[w] = (freq[w] ?? 0) + 1;
    }
  }
  final sorted = freq.entries.toList()
    ..sort((a, b) => b.value.compareTo(a.value));
  // Reserve 0 = <pad>, 1 = <unk>; take the top (vocabSize - 2).
  final take = math.min(sorted.length, vocabSize - 2);
  final itos = <String>[_kPad, _kUnk];
  final counts = <int>[0, 0];
  for (var i = 0; i < take; i++) {
    itos.add(sorted[i].key);
    counts.add(sorted[i].value);
  }
  final stoi = <String, int>{for (var i = 0; i < itos.length; i++) itos[i]: i};
  // Recount <unk> across everything that fell out of the top-K.
  var unk = 0;
  for (var i = take; i < sorted.length; i++) {
    unk += sorted[i].value;
  }
  counts[_kUnkId] = unk;
  return _Vocab(itos, stoi, counts);
}

List<List<int>> _encodeSentences(List<List<String>> sentences, _Vocab v) {
  final out = <List<int>>[];
  for (final s in sentences) {
    if (s.isEmpty) continue;
    out.add(s.map((w) => v.stoi[w] ?? _kUnkId).toList());
  }
  return out;
}

// ---------------------------------------------------------------------------
// Skip-gram + negative sampling
// ---------------------------------------------------------------------------

/// Mikolov subsampling: keep word `w` with probability
/// `sqrt(t / f(w)) + t / f(w)` where `f(w)` is the word's fraction of
/// the corpus. Very frequent words (the, and, ...) are dropped often,
/// rare words are almost always kept.
Float32List _keepProb(_Vocab v, double t) {
  final total = v.counts.fold<int>(0, (a, b) => a + b);
  final out = Float32List(v.size);
  for (var i = 0; i < v.size; i++) {
    if (i == _kPadId) {
      out[i] = 0.0;
      continue;
    }
    final c = v.counts[i];
    if (c == 0) {
      out[i] = 0.0;
      continue;
    }
    final f = c / total;
    final p = math.sqrt(t / f) + t / f;
    out[i] = p.clamp(0.0, 1.0).toDouble();
  }
  return out;
}

/// Log-uniform (Zipf) sampler over `[2, V)` — the id space excluding
/// `<pad>` and `<unk>`. `idx = 2 + floor(exp(u * ln(V - 2))) - 1`, clamped.
/// Frequency-sorted vocab means small ids = common words, so this
/// concentrates on common words the way `tf.random.log_uniform_candidate_sampler`
/// does.
int _sampleLogUniform(int vocabSize, math.Random rng) {
  final v = vocabSize - 2;
  if (v <= 1) return _kUnkId;
  final u = rng.nextDouble();
  final r = math.exp(u * math.log(v.toDouble())).floor() - 1;
  final id = 2 + r.clamp(0, v - 1);
  return id;
}

class _Triples {
  _Triples(this.targets, this.contexts);
  final List<int> targets; // length P
  final List<List<int>> contexts; // length P, each has 1 + numNs ids
}

_Triples _makeSkipGramData(
  List<List<int>> sequences,
  _Vocab vocab, {
  required int windowSize,
  required int numNs,
  required Float32List keepProb,
  required int seed,
}) {
  final rng = math.Random(seed);
  final targets = <int>[];
  final contexts = <List<int>>[];

  for (final seq in sequences) {
    // Subsample.
    final kept = <int>[];
    for (final id in seq) {
      if (id == _kPadId) continue;
      if (rng.nextDouble() <= keepProb[id]) kept.add(id);
    }
    if (kept.length < 2) continue;

    for (var i = 0; i < kept.length; i++) {
      final tgt = kept[i];
      // Dynamic window in [1, windowSize].
      final w = 1 + rng.nextInt(windowSize);
      final lo = math.max(0, i - w);
      final hi = math.min(kept.length, i + w + 1);
      for (var j = lo; j < hi; j++) {
        if (j == i) continue;
        final pos = kept[j];
        // Sample numNs negatives, rejecting collisions with `pos`.
        final row = <int>[pos];
        var tries = 0;
        while (row.length < 1 + numNs && tries < 32 * numNs) {
          final neg = _sampleLogUniform(vocab.size, rng);
          if (neg == pos || neg == tgt) {
            tries++;
            continue;
          }
          row.add(neg);
          tries++;
        }
        // In the unlikely event of rejection failures, pad with <unk>.
        while (row.length < 1 + numNs) {
          row.add(_kUnkId);
        }
        targets.add(tgt);
        contexts.add(row);
      }
    }
  }
  return _Triples(targets, contexts);
}

// ---------------------------------------------------------------------------
// Model: per-example dot products via `.embedding()` row-gather +
// element-wise multiply + row-sum-via-matmul-with-ones.
// ---------------------------------------------------------------------------

/// Given target `[B, D]` and context `[B, C, D]`, produce logits
/// `[B, C]` where `logits[b, c] = dot(target[b], context[b, c])`.
///
/// Implementation trick: this package has no batched matmul / einsum /
/// axis-sum. We instead:
///   1. Gather-repeat `target` from `[B, D]` to `[B*C, D]` by looking
///      it up as if it were an embedding table with indices
///      `[0,0,..,0, 1,1,..,1, ..., B-1,..,B-1]` (each id repeated C
///      times). `.embedding()` is a first-class Tensor op with full
///      autograd, so gradients flow back into the target table.
///   2. Flatten `context` to `[B*C, D]`.
///   3. Element-wise multiply -> `[B*C, D]`.
///   4. Row-sum via `matmul(ones[D, 1])` -> `[B*C, 1]`.
///   5. Reshape to `[B, C]`.
Tensor _batchLogits(Tensor target, Tensor context, Tensor onesD) {
  final b = target.shape[0];
  final d = target.shape[1];
  final c = context.shape[1];
  final repIdx = Tensor.fromList(
    [b * c],
    List<double>.generate(b * c, (bc) => (bc ~/ c).toDouble()),
    device: target.device,
  );
  final tgtRep = target.embedding(repIdx); // [B*C, D]
  final ctxFlat = context.reshape([b * c, d]); // [B*C, D]
  final prod = tgtRep * ctxFlat; // [B*C, D]
  final rowSum = prod.matmul(onesD); // [B*C, 1]
  return rowSum.reshape([b, c]);
}

// ---------------------------------------------------------------------------
// Post-training: pull embeddings, L2-normalise, build vector store.
// ---------------------------------------------------------------------------

List<Float32List> _pullNormalisedEmbeddings(Embedding e) {
  final flat = e.weight.toList();
  final v = e.numEmbeddings;
  final d = e.embeddingDim;
  final rows = <Float32List>[];
  for (var i = 0; i < v; i++) {
    final row = Float32List(d);
    var norm = 0.0;
    for (var j = 0; j < d; j++) {
      final x = flat[i * d + j];
      row[j] = x;
      norm += x * x;
    }
    norm = math.sqrt(norm);
    if (norm > 0) {
      for (var j = 0; j < d; j++) {
        row[j] /= norm;
      }
    }
    rows.add(row);
  }
  return rows;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

void main(List<String> args) {
  final cfg = _parseArgs(args);
  final stopwatch = Stopwatch()..start();

  // ---- Load + tokenise ----
  final text = _loadCorpus(cfg.maxChars);
  final lines = text.split('\n');
  final sentences = lines.map(_tokenizeLine).toList();
  final totalTokens = sentences.fold<int>(0, (a, s) => a + s.length);
  print(
    'Loaded ${text.length} chars, ${sentences.length} lines, '
    '$totalTokens tokens.',
  );

  // ---- Vocab ----
  final vocab = _buildVocab(sentences, cfg.vocabSize);
  print(
    'Vocab: ${vocab.size} words (top-10: '
    '${vocab.itos.sublist(2, math.min(12, vocab.size)).join(" ")})',
  );

  // ---- Encode + skip-gram + negative sampling ----
  final sequences = _encodeSentences(sentences, vocab);
  final keepProb = _keepProb(vocab, cfg.subsampleT);
  print(
    'Generating skip-gram pairs (window=${cfg.windowSize}, '
    'numNs=${cfg.numNs}, subsampleT=${cfg.subsampleT})...',
  );
  final triples = _makeSkipGramData(
    sequences,
    vocab,
    windowSize: cfg.windowSize,
    numNs: cfg.numNs,
    keepProb: keepProb,
    seed: cfg.seed,
  );
  final numPairs = triples.targets.length;
  print('  $numPairs pairs.');
  if (numPairs < cfg.batchSize) {
    stderr.writeln(
      'word2vec_demo: only $numPairs pairs generated, need >= batch size '
      '${cfg.batchSize}. Try --max-chars=... larger or --batch=... smaller.',
    );
    exit(1);
  }

  // ---- Model ----
  print('Device: ${cfg.device}');
  final targetEmb = Embedding(
    vocab.size,
    cfg.embedDim,
    seed: cfg.seed,
    device: cfg.device,
  );
  final ctxEmb = Embedding(
    vocab.size,
    cfg.embedDim,
    seed: cfg.seed + 1,
    device: cfg.device,
  );
  final params = [...targetEmb.parameters(), ...ctxEmb.parameters()];
  final opt = Adam(params, lr: cfg.lr);

  final onesD = Tensor.fill([cfg.embedDim, 1], 1.0, device: cfg.device);
  final zeroTargets = Tensor.fromList(
    [cfg.batchSize],
    List<double>.filled(cfg.batchSize, 0.0),
    device: cfg.device,
  );

  // ---- Train ----
  final rng = math.Random(cfg.seed);
  final numBatches = numPairs ~/ cfg.batchSize;
  print(
    'Training: ${cfg.epochs} epochs x $numBatches batches of '
    '${cfg.batchSize}, embedDim=${cfg.embedDim}, lr=${cfg.lr}',
  );

  for (var epoch = 0; epoch < cfg.epochs; epoch++) {
    final order = List<int>.generate(numPairs, (i) => i)..shuffle(rng);
    var epochLoss = 0.0;
    final epochWatch = Stopwatch()..start();
    for (var b = 0; b < numBatches; b++) {
      final baseIdx = b * cfg.batchSize;
      final tgtBuf = Float32List(cfg.batchSize);
      final ctxBuf = Float32List(cfg.batchSize * (1 + cfg.numNs));
      for (var i = 0; i < cfg.batchSize; i++) {
        final p = order[baseIdx + i];
        tgtBuf[i] = triples.targets[p].toDouble();
        final row = triples.contexts[p];
        for (var j = 0; j < row.length; j++) {
          ctxBuf[i * row.length + j] = row[j].toDouble();
        }
      }
      final tgtIds = Tensor.fromList(
        [cfg.batchSize],
        tgtBuf,
        device: cfg.device,
      );
      final ctxIds = Tensor.fromList(
        [cfg.batchSize, 1 + cfg.numNs],
        ctxBuf,
        device: cfg.device,
      );

      opt.zeroGrad();
      final tgtVec = targetEmb(tgtIds); // [B, D]
      final ctxVec = ctxEmb(ctxIds); // [B, C, D]
      final logits = _batchLogits(tgtVec, ctxVec, onesD); // [B, C]
      final lossPer = logits.crossEntropy(zeroTargets); // [B, 1]
      final loss = lossPer.mean(); // [1, 1]
      loss.backward();
      opt.step();
      epochLoss += loss.toList()[0];
    }
    print(
      'epoch ${epoch + 1}/${cfg.epochs}  '
      'avg loss=${(epochLoss / numBatches).toStringAsFixed(4)}  '
      '(${epochWatch.elapsedMilliseconds}ms)',
    );
  }

  // ---- Vector store: nearest neighbours by cosine similarity ----
  print('\nBuilding IndexFlatIP from L2-normalised target embeddings...');
  final embs = _pullNormalisedEmbeddings(targetEmb);
  final index = IndexFlatIP(cfg.embedDim);
  index.add(embs);

  final k = 10;
  print('Nearest neighbours (cosine sim) for query words:');
  for (final w in cfg.queries) {
    final id = vocab.stoi[w];
    if (id == null) {
      print('  "$w": not in vocab, skipping');
      continue;
    }
    final result = index.search([embs[id]], k + 1); // +1 to drop self
    final ids = result.ids[0];
    final dists = result.distances[0];
    final buf = StringBuffer('  ${w.padRight(10)} -> ');
    var shown = 0;
    for (var i = 0; i < ids.length && shown < k; i++) {
      if (ids[i] == id) continue;
      if (ids[i] < 0) break;
      buf.write('${vocab.itos[ids[i]]}(${dists[i].toStringAsFixed(3)}) ');
      shown++;
    }
    print(buf.toString());
  }

  stopwatch.stop();
  print(
    '\nDone in ${(stopwatch.elapsedMilliseconds / 1000).toStringAsFixed(1)}s.',
  );
}
