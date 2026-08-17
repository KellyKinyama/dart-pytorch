/// Shared helpers for "use a causal GPT as a sentence encoder" demos.
///
/// Not a `dart run` entry point — the leading underscore in the filename
/// signals this is a library used by neighbouring `vector_*_demo.dart`
/// files. See:
///   * bin/rag_qa_demo.dart          — retrieval-augmented Q&A
///   * bin/vector_zero_shot_classify_demo.dart
///   * bin/vector_dedup_demo.dart
///   * bin/vector_cluster_demo.dart
///
/// The API here is deliberately tiny:
///
///   * `lastTokenHidden(model, ids)` — run the transformer stack
///     (no LM head), return the last-position hidden state as a
///     `Float32List`. This is the "encoder-only" trick — see
///     doc/vectors/12-RAG-NUTS-AND-BOLTS.md §12.3.
///   * `meanVector(vecs, d)`         — compute the corpus mean.
///   * `centerAndNormalize(v, mean)` — subtract mean and L2-normalise,
///     turning inner product into cosine similarity. See §12.4-12.5.
///   * `embedCorpus(model, tok, corpus)` — two-pass shorthand: raw
///     encode, compute mean, centre + normalise, return
///     `(normalisedVecs, mean)`. Every demo does this.
///   * `parseArgs(args, defaultPath, defaultVocab)` — a shared minimal
///     CLI (`--path`, `--vocab`, `--preset`, `--gpu`, `-h`).
///   * `configForPreset(preset, device)` — GPTConfig factory.
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// Run `model` as an encoder over `tokenIds` and return the hidden
/// state at the last position. Skips the LM head. Under a causal mask
/// only the final position has attended to every preceding token, so
/// it's the natural "summary" slot on a decoder-only LM.
Float32List lastTokenHidden(GPT model, List<int> tokenIds) {
  if (tokenIds.isEmpty) {
    return Float32List(model.config.embedDim);
  }
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

    final flat = h.toList();
    final d = model.config.embedDim;
    final n = clipped.length;
    final out = Float32List(d);
    final base = (n - 1) * d;
    for (var j = 0; j < d; j++) {
      out[j] = flat[base + j].toDouble();
    }
    return out;
  });
}

/// Element-wise mean of `vecs` (assumed all length `d`).
Float32List meanVector(List<Float32List> vecs, int d) {
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

/// Subtract `mean` from `v` and L2-normalise the result. Under this
/// transform, `IndexFlatIP` reports cosine similarity directly.
Float32List centerAndNormalize(Float32List v, Float32List mean) {
  final d = v.length;
  final out = Float32List(d);
  var sq = 0.0;
  for (var j = 0; j < d; j++) {
    out[j] = v[j] - mean[j];
    sq += out[j] * out[j];
  }
  final norm = 1.0 / (sq > 1e-24 ? math.sqrt(sq) : 1.0);
  for (var j = 0; j < d; j++) {
    out[j] *= norm;
  }
  return out;
}

/// Two-pass shortcut: encode every string in `corpus`, compute the
/// corpus mean, centre + normalise each vector. Returns the ready-
/// to-index unit vectors plus the mean (so queries can be centred the
/// same way).
({List<Float32List> vectors, Float32List mean}) embedCorpus(
  GPT model,
  HFBpeTokenizer tokenizer,
  List<String> corpus, {
  void Function(int i, int total, String preview)? onProgress,
}) {
  final raw = <Float32List>[];
  for (var i = 0; i < corpus.length; i++) {
    final ids = tokenizer.encode(corpus[i]);
    raw.add(lastTokenHidden(model, ids));
    if (onProgress != null) {
      final preview = corpus[i].length > 60
          ? '${corpus[i].substring(0, 60)}...'
          : corpus[i];
      onProgress(i, corpus.length, preview);
    }
  }
  final mean = meanVector(raw, model.config.embedDim);
  final centered = raw.map((v) => centerAndNormalize(v, mean)).toList();
  return (vectors: centered, mean: mean);
}

/// Encode a single query the same way (raw last-token → centre against
/// the training `mean` → L2-normalise).
Float32List embedQuery(
  GPT model,
  HFBpeTokenizer tokenizer,
  String query,
  Float32List mean,
) {
  final ids = tokenizer.encode(query);
  final raw = lastTokenHidden(model, ids);
  return centerAndNormalize(raw, mean);
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/// Small CLI shared by the `vector_*_demo.dart` scripts.
class EncoderOpts {
  EncoderOpts({
    required this.path,
    required this.vocabPath,
    required this.preset,
    required this.gpu,
  });
  final String path;
  final String vocabPath;
  final String preset; // 'distilgpt2' | 'small' | 'medium' | 'large'
  final bool gpu;

  Device get device => gpu ? Device.GPU : Device.CPU;
}

const String _sharedHelp = '''
Flags:
  --path PATH      safetensors weights (default: models/distilgpt2/model.safetensors)
  --vocab PATH     tokenizer.json     (default: models/distilgpt2/tokenizer.json)
  --preset NAME    distilgpt2 | small | medium | large  (default: distilgpt2)
                   Selects the GPTConfig factory; --path must match.
  --gpu            Run on CUDA (default: CPU).
  -h, --help       Print this message.
''';

EncoderOpts parseEncoderArgs(
  List<String> args, {
  String defaultPath = 'models/distilgpt2/model.safetensors',
  String defaultVocab = 'models/distilgpt2/tokenizer.json',
  String defaultPreset = 'distilgpt2',
  String? programHelp,
}) {
  var path = defaultPath;
  var vocab = defaultVocab;
  var preset = defaultPreset;
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
        stdout.writeln(programHelp ?? _sharedHelp);
        exit(0);
      default:
        stderr.writeln('unknown arg: $a');
        stderr.writeln(programHelp ?? _sharedHelp);
        exit(64);
    }
  }
  return EncoderOpts(path: path, vocabPath: vocab, preset: preset, gpu: gpu);
}

GPTConfig configForPreset(String preset, Device device) {
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
        'unknown preset "$preset"; use distilgpt2 | small | medium | large',
      );
      exit(64);
  }
}

/// Convenience: build GPT, load weights, load tokenizer, set eval mode.
({GPT model, HFBpeTokenizer tokenizer, GPTConfig config}) loadEncoder(
  EncoderOpts opts, {
  void Function(String)? log,
}) {
  final say = log ?? stdout.writeln;
  final cfg = configForPreset(opts.preset, opts.device);
  say(
    'Building GPT (preset=${opts.preset}, device=${opts.gpu ? "gpu" : "cpu"},'
    ' embed=${cfg.embedDim}, layers=${cfg.numLayers}, heads=${cfg.numHeads})',
  );
  final model = GPT(cfg);
  say('Loading safetensors from ${opts.path} ...');
  final report = GPT2HFLoader.loadFile(model, opts.path);
  say('Loaded. $report');
  say('Loading tokenizer from ${opts.vocabPath}');
  final tokenizer = HFBpeTokenizer.loadFile(opts.vocabPath);
  model.eval();
  return (model: model, tokenizer: tokenizer, config: cfg);
}
