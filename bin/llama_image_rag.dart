/// Interactive Llama-3 chat with **image retrieval** as the poor-man's
/// multi-modal path (option D from the LlamaVision arc plan).
///
/// The model itself is still a plain, text-only Llama-3 checkpoint —
/// it never sees pixels. What is new:
///
///  1. At startup we walk `--gallery DIR` for images. Two layout
///     conventions are supported:
///
///       gallery/                             gallery/
///         apple.jpg                            dog/
///         apple.txt   ← caption                  1.jpg
///         sunset.png                             2.jpg
///         sunset.txt                           cat/
///         ...                                    1.jpg
///                                                2.jpg
///
///     In the flat layout the sibling `.txt` (or `.caption`) file
///     holds the caption. In the class-folder layout the parent
///     directory name is the caption for every image beneath it.
///     Falls back to the filename stem (underscores → spaces) if
///     neither a sibling caption nor a class dir is available.
///
///  2. Each gallery image is decoded, resized to `--image-size`,
///     patchified and fed through a **ViT** (from-scratch, random
///     init unless you point `--vit-load` at a serialised checkpoint).
///     The CLS embedding is L2-normalised and packed into an
///     [IndexFlatIP], so an inner-product search *is* cosine
///     similarity.
///
///  3. In the REPL you type `:img PATH` before a message. The next
///     user turn embeds that image, retrieves the top-K nearest
///     gallery captions, and prepends them to the message as
///     "This image looks similar to: [1] caption1  [2] caption2  ..."
///     before Llama sees it. The `:img` attachment is consumed by
///     the next turn.
///
/// Model never sees pixels — just retrieved caption text. Good for
/// photo-lookup / gallery-lookup, poor for arbitrary VQA. Quality
/// tracks the ViT quality: with a random-init ViT it will still
/// distinguish images that differ dramatically in low-level colour
/// / composition (the first patch projection preserves the histogram
/// signal); to get real semantics train a ViT with
/// `bin/train_face_folder.dart` or similar and load it with
/// `--vit-load`.
///
/// Usage:
///   dart run bin/llama_image_rag.dart --gallery DIR [flags]
///
/// Common flags:
///   --path PATH        safetensors weights (default: models/llama-3.2-1b-instruct/model.safetensors)
///   --vocab PATH       tokenizer.json      (default: models/llama-3.2-1b-instruct/tokenizer.json)
///   --preset NAME      llama-3.2-1b | llama-3.2-3b | llama-3.1-8b  (default: llama-3.2-1b)
///   --gpu              run on CUDA (default: CPU)
///   --gallery DIR      directory of images (required)
///   --image-size N     resize target for gallery + query images (default: 64)
///   --patch-size N     ViT patch size (default: 16; must divide image-size)
///   --vit-embed-dim N  ViT embedding dim (default: 64)
///   --vit-layers N     ViT transformer layers (default: 2)
///   --vit-heads N      ViT attention heads (default: 4)
///   --vit-load PATH    load a serialised ViT checkpoint instead of random init
///   --top-k-imgs K     gallery captions retrieved per turn (default: 3)
///   --max-new N        max tokens per reply (default: 256)
///   --temperature F    sampling temperature (default: 0.7; 0.0 = greedy)
///   --top-k K          top-K sampling (default: 40, 0 disables)
///   --system "..."     initial system prompt
///
/// REPL:
///   :img PATH          attach an image for the NEXT user turn
///   :quit              exit
///   :reset             wipe conversation history
///   :sys <text>        replace system prompt and reset
///   :sources           show captions retrieved for the last image turn
///
/// Examples:
///   # Flat gallery, CPU
///   dart run bin/llama_image_rag.dart --gallery data/gallery
///
///   # Class-folder gallery on GPU (WSL2)
///   LD_LIBRARY_PATH=/usr/lib/wsl/lib dart run bin/llama_image_rag.dart \\
///     --gallery faces_gallery --gpu
///
///   # With a trained ViT checkpoint
///   dart run bin/llama_image_rag.dart --gallery faces_gallery \\
///     --vit-load models/vit_face.bin --image-size 64 --patch-size 8
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:image/image.dart' as img;

import '_llama_encoder.dart';
import '_lm_encoder.dart' show centerAndNormalize, meanVector;

// ---------------------------------------------------------------------------
// CLI parsing
// ---------------------------------------------------------------------------

class _Opts {
  _Opts({
    required this.path,
    required this.vocabPath,
    required this.preset,
    required this.gpu,
    required this.gallery,
    required this.imageSize,
    required this.patchSize,
    required this.vitEmbedDim,
    required this.vitLayers,
    required this.vitHeads,
    required this.vitLoad,
    required this.topKImgs,
    required this.system,
    required this.maxNew,
    required this.temperature,
    required this.topK,
  });

  final String path;
  final String vocabPath;
  final String preset;
  final bool gpu;
  final String? gallery;
  final int imageSize;
  final int patchSize;
  final int vitEmbedDim;
  final int vitLayers;
  final int vitHeads;
  final String? vitLoad;
  final int topKImgs;
  final String system;
  final int maxNew;
  final double temperature;
  final int topK;
}

_Opts _parseArgs(List<String> args) {
  var path = 'models/llama-3.2-1b-instruct/model.safetensors';
  var vocab = 'models/llama-3.2-1b-instruct/tokenizer.json';
  var preset = 'llama-3.2-1b';
  var gpu = false;
  String? gallery;
  var imageSize = 64;
  var patchSize = 16;
  var vitEmbedDim = 64;
  var vitLayers = 2;
  var vitHeads = 4;
  String? vitLoad;
  var topKImgs = 3;
  var system = 'You are a helpful, concise assistant.';
  var maxNew = 256;
  var temperature = 0.7;
  var topK = 40;

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
      case '--gallery':
        gallery = args[++i];
      case '--image-size':
        imageSize = int.parse(args[++i]);
      case '--patch-size':
        patchSize = int.parse(args[++i]);
      case '--vit-embed-dim':
        vitEmbedDim = int.parse(args[++i]);
      case '--vit-layers':
        vitLayers = int.parse(args[++i]);
      case '--vit-heads':
        vitHeads = int.parse(args[++i]);
      case '--vit-load':
        vitLoad = args[++i];
      case '--top-k-imgs':
        topKImgs = int.parse(args[++i]);
      case '--system':
        system = args[++i];
      case '--max-new':
        maxNew = int.parse(args[++i]);
      case '--temperature':
        temperature = double.parse(args[++i]);
      case '--top-k':
        topK = int.parse(args[++i]);
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
    gallery: gallery,
    imageSize: imageSize,
    patchSize: patchSize,
    vitEmbedDim: vitEmbedDim,
    vitLayers: vitLayers,
    vitHeads: vitHeads,
    vitLoad: vitLoad,
    topKImgs: topKImgs,
    system: system,
    maxNew: maxNew,
    temperature: temperature,
    topK: topK,
  );
}

const String _help = '''
Interactive Llama-3 chat with image-embedding retrieval.

Usage:
  dart run bin/llama_image_rag.dart --gallery DIR [flags]

Model:
  --path PATH        safetensors weights (default: models/llama-3.2-1b-instruct/model.safetensors)
  --vocab PATH       tokenizer.json      (default: models/llama-3.2-1b-instruct/tokenizer.json)
  --preset NAME      llama-3.2-1b | llama-3.2-3b | llama-3.1-8b  (default: llama-3.2-1b)
  --gpu              run on CUDA (default: CPU)

Vision:
  --gallery DIR      directory of images (required). Flat layout with sibling
                     .txt captions, or class-folder layout with folder-name captions.
  --image-size N     resize target for gallery + query images (default: 64)
  --patch-size N     ViT patch size (default: 16; must divide image-size)
  --vit-embed-dim N  ViT embedding dim (default: 64)
  --vit-layers N     ViT transformer layers (default: 2)
  --vit-heads N      ViT attention heads (default: 4)
  --vit-load PATH    load a serialised ViTBackbone checkpoint (default: random init)
  --top-k-imgs K     gallery captions retrieved per turn (default: 3)

Sampling:
  --system "..."     initial system prompt
  --max-new N        max tokens per reply    (default: 256)
  --temperature F    sampling temperature    (default: 0.7; 0.0 = greedy)
  --top-k K          top-K sampling          (default: 40, 0 disables)

REPL commands:
  :img PATH          attach an image for the NEXT user turn (consumed once)
  :quit              exit
  :reset             wipe conversation history (system prompt kept)
  :sys <text>        replace system prompt and reset
  :sources           print captions retrieved for the last image turn

Examples:
  # Flat gallery on CPU
  dart run bin/llama_image_rag.dart --gallery data/gallery

  # Class-folder gallery (folder name = caption) on GPU
  LD_LIBRARY_PATH=/usr/lib/wsl/lib dart run bin/llama_image_rag.dart \\
    --gallery faces_gallery --gpu

  # With a trained ViT
  dart run bin/llama_image_rag.dart --gallery faces_gallery \\
    --vit-load models/vit_face.bin --image-size 64 --patch-size 8
''';

// ---------------------------------------------------------------------------
// Image decoding / patchifying (single-image helper)
// ---------------------------------------------------------------------------

/// Decode an image file at `path`, resize to `size × size`, normalise
/// to `[0, 1]` channels-last RGB, and patchify into a Tensor of shape
/// `[numPatches, patchSize*patchSize*3]` on `device`. Returns null on
/// any decode / IO failure so the caller can skip the file gracefully.
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
    final flat = Float32List(size * size * 3);
    var i = 0;
    for (final p in resized) {
      flat[i++] = p.r / 255.0;
      flat[i++] = p.g / 255.0;
      flat[i++] = p.b / 255.0;
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

// ---------------------------------------------------------------------------
// Gallery loading & indexing
// ---------------------------------------------------------------------------

class _GalleryEntry {
  _GalleryEntry({required this.path, required this.caption});
  final String path;
  final String caption;
}

bool _isImage(String p) {
  final l = p.toLowerCase();
  return l.endsWith('.jpg') || l.endsWith('.jpeg') || l.endsWith('.png');
}

String _captionFromStem(String path) {
  final base = path.split(Platform.pathSeparator).last;
  final dot = base.lastIndexOf('.');
  final stem = dot > 0 ? base.substring(0, dot) : base;
  return stem.replaceAll('_', ' ').replaceAll('-', ' ');
}

String? _readSiblingCaption(String imagePath) {
  final dot = imagePath.lastIndexOf('.');
  if (dot <= 0) return null;
  for (final ext in const ['.txt', '.caption']) {
    final capPath = imagePath.substring(0, dot) + ext;
    final f = File(capPath);
    if (f.existsSync()) {
      final s = f.readAsStringSync().trim();
      if (s.isNotEmpty) return s;
    }
  }
  return null;
}

/// Walk `dir` (recursively) for image files. For each one, pick a
/// caption in this precedence order:
///   1. Sibling `<stem>.txt` / `<stem>.caption`
///   2. Immediate parent directory name (if it is a subdir of `dir`)
///   3. Filename stem with `_` / `-` → space
List<_GalleryEntry> _scanGallery(String dir) {
  final root = Directory(dir);
  if (!root.existsSync()) {
    throw ArgumentError('--gallery does not exist: $dir');
  }
  final out = <_GalleryEntry>[];
  final rootPath = root.absolute.path;
  for (final e in root.listSync(recursive: true, followLinks: false)) {
    if (e is! File) continue;
    if (!_isImage(e.path)) continue;
    final sibling = _readSiblingCaption(e.path);
    final parentDir = e.parent.absolute.path;
    String caption;
    if (sibling != null) {
      caption = sibling;
    } else if (parentDir != rootPath) {
      caption = parentDir.split(Platform.pathSeparator).last;
    } else {
      caption = _captionFromStem(e.path);
    }
    out.add(_GalleryEntry(path: e.path, caption: caption));
  }
  out.sort((a, b) => a.path.compareTo(b.path));
  return out;
}

/// Compute the CLS embedding for an image via a random-projected pixel
/// signature when the ViT is untrained. Uses [ViTFaceEmbedding] internals
/// but only wants the L2-normalised CLS row.
///
/// Returns a `Float32List` of length [ViTFaceEmbedding.outputDim].
Float32List _embedImage(ViTFaceEmbedding embedder, Tensor patchified) {
  return Tensor.noGrad(() {
    final v = embedder(patchified); // [1, outputDim] L2-normalised
    return Float32List.fromList(v.toList());
  });
}

({
  IndexFlat index,
  Float32List mean,
  List<_GalleryEntry> entries,
  int embedDim,
})?
_buildImageIndex({
  required List<_GalleryEntry> entries,
  required ViTFaceEmbedding embedder,
  required int imageSize,
  required int patchSize,
  required Device device,
}) {
  if (entries.isEmpty) return null;
  final d = embedder.outputDim;
  final raw = <Float32List>[];
  final kept = <_GalleryEntry>[];
  for (final e in entries) {
    final patches = _decodeAndPatchify(
      e.path,
      size: imageSize,
      patchSize: patchSize,
      device: device,
    );
    if (patches == null) {
      stderr.writeln('[gallery] skipped (decode failed): ${e.path}');
      continue;
    }
    raw.add(_embedImage(embedder, patches));
    kept.add(e);
  }
  if (raw.isEmpty) return null;
  final mean = meanVector(raw, d);
  final index = IndexFlatIP(d);
  for (final v in raw) {
    index.add([centerAndNormalize(v, mean)]);
  }
  return (index: index, mean: mean, entries: kept, embedDim: d);
}

// ---------------------------------------------------------------------------
// Retrieval + prompt augmentation
// ---------------------------------------------------------------------------

List<({_GalleryEntry entry, double score})> _retrieveByImage({
  required String imagePath,
  required ViTFaceEmbedding embedder,
  required IndexFlat index,
  required Float32List mean,
  required List<_GalleryEntry> entries,
  required int imageSize,
  required int patchSize,
  required Device device,
  required int k,
}) {
  final patches = _decodeAndPatchify(
    imagePath,
    size: imageSize,
    patchSize: patchSize,
    device: device,
  );
  if (patches == null) return const [];
  final raw = _embedImage(embedder, patches);
  final qVec = centerAndNormalize(raw, mean);
  final kEff = k < entries.length ? k : entries.length;
  final res = index.search([qVec], kEff);
  final out = <({_GalleryEntry entry, double score})>[];
  for (var i = 0; i < res.ids[0].length; i++) {
    out.add((entry: entries[res.ids[0][i]], score: res.distances[0][i]));
  }
  return out;
}

String _augmentWithImageContext(
  String message,
  String imagePath,
  List<({_GalleryEntry entry, double score})> hits,
) {
  if (hits.isEmpty) return message;
  final buf = StringBuffer();
  buf.writeln(
    'The user attached an image at "$imagePath". By visual similarity to '
    'the gallery, it most resembles the following captioned reference(s):',
  );
  buf.writeln();
  for (var i = 0; i < hits.length; i++) {
    final h = hits[i];
    buf.writeln(
      '  [${i + 1}] ${h.entry.caption} '
      '(cosine=${h.score.toStringAsFixed(3)})',
    );
  }
  buf.writeln();
  buf.writeln(
    'The model itself cannot see pixels — reason from these captions only. '
    'If the message does not depend on the image, ignore this block.',
  );
  buf.writeln();
  buf.writeln('Question: $message');
  return buf.toString();
}

String _systemPrefix(String system) {
  final s = system.trim();
  if (s.isEmpty) return '<|begin_of_text|>';
  return '<|begin_of_text|>'
      '<|start_header_id|>system<|end_header_id|>\n\n$s<|eot_id|>';
}

String _userTurn(String msg) =>
    '<|start_header_id|>user<|end_header_id|>\n\n$msg<|eot_id|>'
    '<|start_header_id|>assistant<|end_header_id|>\n\n';

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

void main(List<String> args) {
  final opts = _parseArgs(args);
  if (opts.gallery == null) {
    stderr.writeln('--gallery DIR is required (see --help)');
    exit(64);
  }
  if (opts.imageSize % opts.patchSize != 0) {
    stderr.writeln(
      '--image-size (${opts.imageSize}) must be divisible by '
      '--patch-size (${opts.patchSize})',
    );
    exit(64);
  }

  final loaded = loadLlamaEncoder(
    path: opts.path,
    vocabPath: opts.vocabPath,
    preset: opts.preset,
    gpu: opts.gpu,
  );
  final model = loaded.model;
  final tok = loaded.tokenizer;
  final cfg = loaded.config;
  final device = opts.gpu ? Device.GPU : Device.CPU;

  // ---- Build ViT embedder --------------------------------------------------
  final embedder = ViTFaceEmbedding(
    imageSize: opts.imageSize,
    patchSize: opts.patchSize,
    embedDim: opts.vitEmbedDim,
    outputDim: opts.vitEmbedDim,
    numLayers: opts.vitLayers,
    numHeads: opts.vitHeads,
    device: device,
    seed: 7,
  );
  if (opts.vitLoad != null) {
    try {
      Checkpoint.loadIntoFile(embedder, opts.vitLoad!);
      stdout.writeln('[vit] loaded checkpoint ${opts.vitLoad}');
    } catch (e) {
      stderr.writeln(
        '[vit] failed to load ${opts.vitLoad}: $e\n'
        '      falling back to random init (retrieval will be low-quality)',
      );
    }
  } else {
    stdout.writeln(
      '[vit] using random-init ViT '
      '(retrieval is coarse — pass --vit-load for a trained checkpoint)',
    );
  }
  embedder.eval();

  // ---- Build gallery index -------------------------------------------------
  final entries = _scanGallery(opts.gallery!);
  if (entries.isEmpty) {
    stderr.writeln('[gallery] no images under ${opts.gallery}; exiting');
    exit(1);
  }
  stdout.writeln('[gallery] scanning ${entries.length} image(s)...');
  final built = _buildImageIndex(
    entries: entries,
    embedder: embedder,
    imageSize: opts.imageSize,
    patchSize: opts.patchSize,
    device: device,
  );
  if (built == null) {
    stderr.writeln('[gallery] no images decoded successfully; exiting');
    exit(1);
  }
  stdout.writeln(
    '[gallery] indexed ${built.entries.length} image(s) '
    '(embedDim=${built.embedDim})',
  );

  final eot = tok.llamaEotId;
  final stopId = eot ?? tok.endOfTextId;

  var system = opts.system;
  var chatText = _systemPrefix(system);
  String? pendingImage;
  var lastHits = const <({_GalleryEntry entry, double score})>[];

  stdout.writeln(
    '\nReady. Attach an image with `:img PATH`, then send a text turn.\n'
    'Commands: :quit / :reset / :sys <text> / :sources. Ctrl+D to exit.\n',
  );

  while (true) {
    stdout.write('you> ');
    final line = stdin.readLineSync();
    if (line == null) {
      stdout.writeln();
      break;
    }
    final trimmed = line.trim();
    if (trimmed.isEmpty) continue;
    if (trimmed == ':quit') break;
    if (trimmed == ':reset') {
      chatText = _systemPrefix(system);
      pendingImage = null;
      stdout.writeln('[history cleared]');
      continue;
    }
    if (trimmed.startsWith(':sys ')) {
      system = trimmed.substring(5).trim();
      chatText = _systemPrefix(system);
      pendingImage = null;
      stdout.writeln('[system prompt updated; history cleared]');
      continue;
    }
    if (trimmed.startsWith(':img ')) {
      final p = trimmed.substring(5).trim();
      if (!File(p).existsSync()) {
        stdout.writeln('[image not found: $p]');
      } else {
        pendingImage = p;
        stdout.writeln('[attached "$p" — send a text turn now]');
      }
      continue;
    }
    if (trimmed == ':sources') {
      if (lastHits.isEmpty) {
        stdout.writeln('[no image turn has run yet]');
      } else {
        for (var i = 0; i < lastHits.length; i++) {
          final h = lastHits[i];
          stdout.writeln(
            '  [${i + 1}] ${h.entry.caption}  '
            'cos=${h.score.toStringAsFixed(3)}  '
            '(${h.entry.path})',
          );
        }
      }
      continue;
    }

    var userMsg = trimmed;
    if (pendingImage != null) {
      lastHits = _retrieveByImage(
        imagePath: pendingImage,
        embedder: embedder,
        index: built.index,
        mean: built.mean,
        entries: built.entries,
        imageSize: opts.imageSize,
        patchSize: opts.patchSize,
        device: device,
        k: opts.topKImgs,
      );
      userMsg = _augmentWithImageContext(trimmed, pendingImage, lastHits);
      pendingImage = null;
    }

    final prompt = chatText + _userTurn(userMsg);
    var promptIds = tok.encode(prompt);

    if (promptIds.length + opts.maxNew > cfg.maxCtx) {
      stdout.writeln(
        '[history exceeds context (${promptIds.length} + ${opts.maxNew} > '
        '${cfg.maxCtx}); resetting]',
      );
      chatText = _systemPrefix(system);
      promptIds = tok.encode(chatText + _userTurn(userMsg));
      if (promptIds.length + opts.maxNew > cfg.maxCtx) {
        stdout.writeln('[single turn still exceeds context — skipping]\n');
        continue;
      }
    }

    final full = model.generate(
      promptIds.map((i) => i.toDouble()).toList(),
      maxNewTokens: opts.maxNew,
      temperature: opts.temperature,
      topK: opts.topK == 0 ? null : opts.topK,
    );
    final newIds = full
        .sublist(promptIds.length)
        .map((d) => d.toInt())
        .toList();
    List<int> answerIds = newIds;
    if (stopId != null) {
      final idx = newIds.indexOf(stopId);
      if (idx >= 0) answerIds = newIds.sublist(0, idx);
    }
    final answer = tok.decode(answerIds).trim();
    stdout.writeln('\nbot> $answer\n');

    chatText = '$prompt$answer<|eot_id|>';
  }
}
