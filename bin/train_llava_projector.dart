/// LLaVA-style **projector-only** fine-tune.
///
/// Trains just the [VisionProjector] MLP that translates a frozen
/// CLIP-ViT's per-patch features into a frozen Llama-3 decoder's
/// embedding space. All CLIP weights and all Llama weights stay
/// fixed; only the projector receives gradient updates.
///
/// Dataset layout: a directory of `image.jpg` + `image.txt` (or
/// `.png` + `.caption`) sibling pairs. The `.txt` file is a plain
/// caption; empty files are skipped.
///
/// One training step:
///
///   1. Sample an (image, caption) pair.
///   2. Look up the CLIP features for this image (cached — CLIP is
///      frozen and deterministic, so we compute it once per epoch's
///      first touch and reuse).
///   3. Project → prepend to `<|begin_of_text|> <caption>` embeddings.
///   4. Full forward through Llama (grad-tape live: gradients flow
///      back through the frozen Llama blocks to the projector, but
///      Llama's parameters are not passed to the optimizer so its
///      weights don't change).
///   5. CE on the text-position rows, projector's parameters only
///      updated by Adam.
///
/// This is the LLaVA "stage 1 alignment" recipe in miniature. Real
/// LLaVA uses ~558k pairs; here you can point at whatever you have
/// locally (dozens to thousands of pairs). Loss goes down, the
/// resulting projector generalises weakly, but the *plumbing* is
/// exactly correct.
///
/// Usage:
///   dart run bin/train_llava_projector.dart --data DIR --out PATH [flags]
///
/// Flags:
///   --path PATH             Llama safetensors  (default: models/llama-3.2-1b-instruct/model.safetensors)
///   --vocab PATH            tokenizer.json     (default: models/llama-3.2-1b-instruct/tokenizer.json)
///   --preset NAME           llama-3.2-1b | llama-3.2-3b | llama-3.1-8b (default: llama-3.2-1b)
///   --gpu                   run on CUDA (default: CPU)
///   --clip PATH             CLIP safetensors (required)
///   --clip-preset NAME      base32 | base16 | large14 (default: base32)
///   --projector-hidden N    projector hidden width (default: llama.embedDim)
///
///   --data DIR              image + caption directory (required, recursive)
///   --out PATH              save projector checkpoint here (required)
///   --steps N               number of training steps (default: 500)
///   --lr F                  Adam learning rate (default: 1e-3)
///   --log-every N           log every N steps (default: 25)
///   --max-caption N         truncate captions to N tokens (default: 32)
///
/// Example:
///   LD_LIBRARY_PATH=/usr/lib/wsl/lib dart run bin/train_llava_projector.dart \\
///     --clip models/clip-vit-base-patch32/model.safetensors \\
///     --data data/coco_sample --out models/llava_projector.dptc \\
///     --gpu --steps 2000 --lr 5e-4
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:image/image.dart' as img;

import '_llama_encoder.dart';

// ---------------------------------------------------------------------------
// CLI parsing.
// ---------------------------------------------------------------------------

class _Opts {
  _Opts({
    required this.path,
    required this.vocabPath,
    required this.preset,
    required this.gpu,
    required this.clip,
    required this.clipPreset,
    required this.projectorHidden,
    required this.data,
    required this.out,
    required this.steps,
    required this.lr,
    required this.logEvery,
    required this.maxCaption,
  });

  final String path;
  final String vocabPath;
  final String preset;
  final bool gpu;
  final String? clip;
  final String clipPreset;
  final int? projectorHidden;
  final String? data;
  final String? out;
  final int steps;
  final double lr;
  final int logEvery;
  final int maxCaption;
}

_Opts _parseArgs(List<String> args) {
  var path = 'models/llama-3.2-1b-instruct/model.safetensors';
  var vocab = 'models/llama-3.2-1b-instruct/tokenizer.json';
  var preset = 'llama-3.2-1b';
  var gpu = false;
  String? clip;
  var clipPreset = 'base32';
  int? projectorHidden;
  String? data;
  String? out;
  var steps = 500;
  var lr = 1e-3;
  var logEvery = 25;
  var maxCaption = 32;

  for (var i = 0; i < args.length; i++) {
    final a = args[i];
    switch (a) {
      case '--path':
        path = args[++i];
      case '--vocab':
        vocab = args[++i];
      case '--preset':
        preset = args[++i];
      case '--gpu':
        gpu = true;
      case '--clip':
        clip = args[++i];
      case '--clip-preset':
        clipPreset = args[++i];
      case '--projector-hidden':
        projectorHidden = int.parse(args[++i]);
      case '--data':
        data = args[++i];
      case '--out':
        out = args[++i];
      case '--steps':
        steps = int.parse(args[++i]);
      case '--lr':
        lr = double.parse(args[++i]);
      case '--log-every':
        logEvery = int.parse(args[++i]);
      case '--max-caption':
        maxCaption = int.parse(args[++i]);
      case '-h' || '--help':
        stdout.writeln(_help);
        exit(0);
      default:
        stderr.writeln('unknown flag "$a" (see --help)');
        exit(64);
    }
  }
  return _Opts(
    path: path,
    vocabPath: vocab,
    preset: preset,
    gpu: gpu,
    clip: clip,
    clipPreset: clipPreset,
    projectorHidden: projectorHidden,
    data: data,
    out: out,
    steps: steps,
    lr: lr,
    logEvery: logEvery,
    maxCaption: maxCaption,
  );
}

const String _help = '''
LLaVA-style projector-only fine-tune.

Trains just the VisionProjector that maps a frozen CLIP-ViT into a
frozen Llama-3 decoder's embedding space. CLIP and Llama weights do
not change; only the small MLP does.

Usage:
  dart run bin/train_llava_projector.dart --clip PATH --data DIR --out PATH [flags]

Model:
  --path PATH             Llama safetensors  (default: models/llama-3.2-1b-instruct/model.safetensors)
  --vocab PATH            tokenizer.json     (default: models/llama-3.2-1b-instruct/tokenizer.json)
  --preset NAME           llama-3.2-1b | llama-3.2-3b | llama-3.1-8b (default: llama-3.2-1b)
  --gpu                   run on CUDA (default: CPU)

Vision:
  --clip PATH             CLIP safetensors (required)
  --clip-preset NAME      base32 | base16 | large14 (default: base32)
  --projector-hidden N    projector hidden width (default: llama.embedDim)

Data:
  --data DIR              image + caption directory (required, recursive).
                          Each image.jpg (or .png) must have a sibling
                          image.txt (or image.caption) with a non-empty caption.
  --out PATH              save projector checkpoint here (required, .dptc)

Training:
  --steps N               steps (default: 500)
  --lr F                  Adam lr (default: 1e-3)
  --log-every N           print every N steps (default: 25)
  --max-caption N         truncate captions to N tokens (default: 32)

Example:
  LD_LIBRARY_PATH=/usr/lib/wsl/lib dart run bin/train_llava_projector.dart \\
    --clip models/clip-vit-base-patch32/model.safetensors \\
    --data data/coco_sample --out models/llava_projector.dptc \\
    --gpu --steps 2000 --lr 5e-4
''';

// ---------------------------------------------------------------------------
// Image decoding (CLIP normalization) & patchify.
// ---------------------------------------------------------------------------

Tensor? _decodeAndPatchify(
  String path, {
  required int size,
  required int patchSize,
  required Device device,
}) {
  final f = File(path);
  if (!f.existsSync()) return null;
  try {
    final raw = img.decodeImage(f.readAsBytesSync());
    if (raw == null) return null;
    final resized = img.copyResize(
      raw,
      width: size,
      height: size,
      interpolation: img.Interpolation.linear,
    );
    const meanR = 0.48145466,
        meanG = 0.4578275,
        meanB = 0.40821073,
        stdR = 0.26862954,
        stdG = 0.26130258,
        stdB = 0.27577711;
    final flat = Float32List(size * size * 3);
    var i = 0;
    for (final p in resized) {
      flat[i++] = (p.r / 255.0 - meanR) / stdR;
      flat[i++] = (p.g / 255.0 - meanG) / stdG;
      flat[i++] = (p.b / 255.0 - meanB) / stdB;
    }
    final perPatch = patchSize * patchSize * 3;
    final side = size ~/ patchSize;
    final nP = side * side;
    final out = Float32List(nP * perPatch);
    for (var py = 0; py < side; py++) {
      for (var px = 0; px < side; px++) {
        final pIdx = py * side + px;
        final outBase = pIdx * perPatch;
        var w = 0;
        for (var dy = 0; dy < patchSize; dy++) {
          final y = py * patchSize + dy;
          for (var dx = 0; dx < patchSize; dx++) {
            final x = px * patchSize + dx;
            final inBase = (y * size + x) * 3;
            out[outBase + w++] = flat[inBase];
            out[outBase + w++] = flat[inBase + 1];
            out[outBase + w++] = flat[inBase + 2];
          }
        }
      }
    }
    return Tensor.fromList([nP, perPatch], out, device: device);
  } catch (_) {
    return null;
  }
}

CLIPVisionConfig _clipConfig(String preset, Device device) {
  switch (preset) {
    case 'base32':
      return ClipHFLoader.base32Config(device: device);
    case 'base16':
      return ClipHFLoader.base16Config(device: device);
    case 'large14':
      return ClipHFLoader.large14Config(device: device);
    default:
      stderr.writeln(
        'unknown --clip-preset "$preset"; use base32 | base16 | large14',
      );
      exit(64);
  }
}

// ---------------------------------------------------------------------------
// Dataset scan.
// ---------------------------------------------------------------------------

class _Pair {
  _Pair({required this.image, required this.caption});
  final String image;
  final String caption;
}

bool _isImage(String p) {
  final l = p.toLowerCase();
  return l.endsWith('.jpg') || l.endsWith('.jpeg') || l.endsWith('.png');
}

String? _siblingCaption(String imagePath) {
  final dot = imagePath.lastIndexOf('.');
  if (dot <= 0) return null;
  for (final ext in const ['.txt', '.caption']) {
    final f = File(imagePath.substring(0, dot) + ext);
    if (f.existsSync()) {
      final s = f.readAsStringSync().trim();
      if (s.isNotEmpty) return s;
    }
  }
  return null;
}

List<_Pair> _scanPairs(String dir) {
  final root = Directory(dir);
  if (!root.existsSync()) {
    throw ArgumentError('--data does not exist: $dir');
  }
  final out = <_Pair>[];
  for (final e in root.listSync(recursive: true, followLinks: false)) {
    if (e is! File || !_isImage(e.path)) continue;
    final cap = _siblingCaption(e.path);
    if (cap == null) continue;
    out.add(_Pair(image: e.path, caption: cap));
  }
  out.sort((a, b) => a.image.compareTo(b.image));
  return out;
}

// ---------------------------------------------------------------------------
// Text-logit slicing (differentiable).
// Same trick as bin/train_llama_vision_demo.dart: use `.embedding` on
// a float-index vector to pull out the rows that correspond to text
// positions in the full [imageTokens + textTokens] logit tensor.
// ---------------------------------------------------------------------------
Tensor _textLogits(Tensor fullLogits, int targetStart, int targetLen) {
  final idxData = Float32List(targetLen);
  for (int i = 0; i < targetLen; i++) {
    idxData[i] = (targetStart + i).toDouble();
  }
  final indices = Tensor.fromList(
    [targetLen],
    idxData,
    device: fullLogits.device,
  );
  return fullLogits.embedding(indices);
}

// ---------------------------------------------------------------------------
// Main.
// ---------------------------------------------------------------------------

void main(List<String> args) {
  final opts = _parseArgs(args);
  if (opts.clip == null) {
    stderr.writeln('--clip PATH is required (see --help)');
    exit(64);
  }
  if (opts.data == null) {
    stderr.writeln('--data DIR is required (see --help)');
    exit(64);
  }
  if (opts.out == null) {
    stderr.writeln('--out PATH is required (see --help)');
    exit(64);
  }
  final device = opts.gpu ? Device.GPU : Device.CPU;

  // ---- Llama ------------------------------------------------------
  final loaded = loadLlamaEncoder(
    path: opts.path,
    vocabPath: opts.vocabPath,
    preset: opts.preset,
    gpu: opts.gpu,
  );
  final llama = loaded.model;
  final tok = loaded.tokenizer;
  final cfg = loaded.config;

  // ---- CLIP -------------------------------------------------------
  final clipCfg = _clipConfig(opts.clipPreset, device);
  stdout.writeln(
    '[clip] building preset=${opts.clipPreset} '
    '(image=${clipCfg.imageSize}, patch=${clipCfg.patchSize})',
  );
  final clip = CLIPVisionModel(clipCfg);
  stdout.writeln('[clip] loading safetensors from ${opts.clip}');
  final rep = ClipHFLoader.loadFile(clip, opts.clip!);
  stdout.writeln('[clip] loaded. $rep');
  clip.eval();

  // ---- LlamaVision (projector) -----------------------------------
  final vision = LlamaVision.build(
    vit: clip,
    llama: llama,
    projectorHiddenDim: opts.projectorHidden,
    seed: 7,
  );
  vision.train();

  // ---- Data -------------------------------------------------------
  final pairs = _scanPairs(opts.data!);
  if (pairs.isEmpty) {
    stderr.writeln('[data] no image+caption pairs found under ${opts.data}');
    exit(1);
  }
  stdout.writeln('[data] ${pairs.length} image+caption pair(s)');

  // ---- Pre-compute CLIP features per image (CLIP is frozen, so
  //      this is a one-time cost). Cache on CPU as Float32List to
  //      avoid holding many GPU tensors alive. Skips pairs whose
  //      image fails to decode.
  final featureCache = <int, Float32List>{};
  final validIdx = <int>[];
  final numImageTokens = vision.numImageTokens;
  final clipDim = clipCfg.embedDim;
  stdout.writeln(
    '[cache] pre-computing CLIP features for ${pairs.length} pairs...',
  );
  for (int i = 0; i < pairs.length; i++) {
    final patches = _decodeAndPatchify(
      pairs[i].image,
      size: clipCfg.imageSize,
      patchSize: clipCfg.patchSize,
      device: device,
    );
    if (patches == null) {
      stderr.writeln('[cache] skipped (decode failed): ${pairs[i].image}');
      continue;
    }
    final feat = Tensor.noGrad(() {
      final v = clip(patches); // [P+1, clipDim]
      return Float32List.fromList(v.toList());
    });
    featureCache[i] = feat;
    validIdx.add(i);
    if ((i + 1) % 50 == 0 || i == pairs.length - 1) {
      stdout.writeln('[cache]   ${i + 1}/${pairs.length}');
    }
  }
  if (validIdx.isEmpty) {
    stderr.writeln('[cache] no images decoded — exiting');
    exit(1);
  }
  stdout.writeln('[cache] ${validIdx.length} usable pair(s)');

  // ---- Optimize projector params only ----------------------------
  final params = vision.projector.parameters();
  final opt = Adam(params, lr: opts.lr);
  final rng = math.Random(0);

  int paramCount = 0;
  for (final p in params) {
    paramCount += p.length;
  }
  stdout.writeln(
    '[opt] Adam lr=${opts.lr}, training '
    '$paramCount projector scalars (CLIP + Llama frozen)',
  );

  final sw = Stopwatch()..start();
  double lossSum = 0;
  var stepsRun = 0;
  for (int step = 1; step <= opts.steps; step++) {
    opt.zeroGrad();

    // Sample one (image_features, caption) pair.
    final pickIdx = validIdx[rng.nextInt(validIdx.length)];
    final feat = featureCache[pickIdx]!;
    final caption = pairs[pickIdx].caption;

    // Tokenize caption (with BOS in front, EOT at end for a clean signal).
    final capIds = tok.encode('<|begin_of_text|>$caption<|eot_id|>');
    if (capIds.length < 2) continue;
    final truncated = capIds.length > opts.maxCaption
        ? capIds.sublist(0, opts.maxCaption)
        : capIds;
    if (numImageTokens + truncated.length > cfg.maxCtx) {
      // Skip captions that don't fit — shouldn't happen at 32 tokens.
      continue;
    }

    // Rebuild the CLIP feature as a Tensor on `device` and run through
    // projector (only place gradients flow).
    final featTensor = Tensor.fromList(
      [numImageTokens, clipDim],
      feat,
      device: device,
    );
    final projected = vision.projector(featTensor); // [P+1, llamaEmbedDim]

    // Text: input = tokens[:-1], target = tokens[1:].
    final decIn = Tensor.fromList(
      [truncated.length - 1],
      truncated
          .sublist(0, truncated.length - 1)
          .map((v) => v.toDouble())
          .toList(),
      device: device,
    );
    final decTgt = Tensor.fromList(
      [truncated.length - 1],
      truncated.sublist(1).map((v) => v.toDouble()).toList(),
      device: device,
    );

    final textEmb = llama.embedIn(decIn); // [T-1, D]
    final full = TensorConcat.concat([projected, textEmb], axis: 0);
    final logits = llama.forwardFromEmbeddings(full);

    final tgtLen = decTgt.shape[0];
    final textLog = _textLogits(logits, numImageTokens, tgtLen);
    final loss = textLog.crossEntropy(decTgt).mean();
    loss.backward();
    clipGradNorm(params, 1.0);
    opt.step();

    final lossVal = loss.toList()[0];
    lossSum += lossVal;
    stepsRun++;

    if (step == 1 || step % opts.logEvery == 0 || step == opts.steps) {
      final ms = sw.elapsedMilliseconds / stepsRun;
      final avg = lossSum / stepsRun;
      stdout.writeln(
        '  step ${step.toString().padLeft(5)}  '
        'ce=${lossVal.toStringAsFixed(4)}  '
        'avg=${avg.toStringAsFixed(4)}  '
        '(${ms.toStringAsFixed(1)} ms/step)',
      );
    }
  }
  sw.stop();

  // ---- Save projector --------------------------------------------
  stdout.writeln('[save] writing projector checkpoint to ${opts.out}');
  final bytes = Checkpoint.saveBytes(vision.projector);
  File(opts.out!).writeAsBytesSync(bytes);
  stdout.writeln(
    '[done] $stepsRun steps, avg loss '
    '${(lossSum / stepsRun).toStringAsFixed(4)}',
  );
}
