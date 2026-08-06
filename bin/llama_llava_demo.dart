/// Interactive **LLaVA-style** Llama-3 chat with real CLIP vision.
///
/// Loads a full CLIP-ViT (from HuggingFace safetensors) + a Llama-3
/// instruct checkpoint, wires them together via the
/// [VisionProjector], and lets you attach images to your turn with
/// `:img PATH`. The image is patchified, run through CLIP, projected
/// into Llama's embedding space, and prepended to the text token
/// sequence as image tokens (LLaVA convention).
///
/// **Important caveat.** The projector is randomly initialised at
/// startup unless you pass `--projector-load`. With a random
/// projector the model will produce nonsense — the point of this
/// CLI is to prove the wiring runs end-to-end. Train a real
/// projector with `bin/train_llava_projector.dart`, save the
/// checkpoint, then reload it here.
///
/// Usage:
///   dart run bin/llama_llava_demo.dart --clip PATH [flags]
///
/// Common flags:
///   --path PATH             Llama safetensors  (default: models/llama-3.2-1b-instruct/model.safetensors)
///   --vocab PATH            tokenizer.json     (default: models/llama-3.2-1b-instruct/tokenizer.json)
///   --preset NAME           llama-3.2-1b | llama-3.2-3b | llama-3.1-8b (default: llama-3.2-1b)
///   --gpu                   run on CUDA (default: CPU)
///   --clip PATH             CLIP safetensors (required)
///   --clip-preset NAME      base32 | base16 | large14 (default: base32)
///   --projector-load PATH   load a trained projector checkpoint (default: random init)
///   --projector-hidden N    projector hidden width (default: llama.embedDim)
///
///   --system "..."          initial system prompt
///   --max-new N             max tokens per reply    (default: 128)
///   --temperature F         0.0 = greedy            (default: 0.7)
///   --top-k K               top-K sampling          (default: 40, 0 disables)
///
/// REPL:
///   :img PATH   attach an image for the NEXT text turn (single-shot; consumed)
///   :quit       exit
///   :reset      wipe conversation history (system prompt kept)
///   :sys TEXT   replace system prompt and reset
///
/// Example:
///   LD_LIBRARY_PATH=/usr/lib/wsl/lib dart run bin/llama_llava_demo.dart \\
///     --clip models/clip-vit-base-patch32/model.safetensors \\
///     --projector-load models/llava_projector.dptc --gpu
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:image/image.dart' as img;

import '_llama_encoder.dart';

class _Opts {
  _Opts({
    required this.path,
    required this.vocabPath,
    required this.preset,
    required this.gpu,
    required this.clip,
    required this.clipPreset,
    required this.projectorLoad,
    required this.projectorHidden,
    required this.system,
    required this.maxNew,
    required this.temperature,
    required this.topK,
  });

  final String path;
  final String vocabPath;
  final String preset;
  final bool gpu;
  final String? clip;
  final String clipPreset;
  final String? projectorLoad;
  final int? projectorHidden;
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
  String? clip;
  var clipPreset = 'base32';
  String? projectorLoad;
  int? projectorHidden;
  var system = 'You are a helpful, concise assistant.';
  var maxNew = 128;
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
      case '--clip':
        clip = args[++i];
      case '--clip-preset':
        clipPreset = args[++i];
      case '--projector-load':
        projectorLoad = args[++i];
      case '--projector-hidden':
        projectorHidden = int.parse(args[++i]);
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
    clip: clip,
    clipPreset: clipPreset,
    projectorLoad: projectorLoad,
    projectorHidden: projectorHidden,
    system: system,
    maxNew: maxNew,
    temperature: temperature,
    topK: topK,
  );
}

const String _help = '''
LLaVA-style Llama-3 chat with real CLIP vision.

Usage:
  dart run bin/llama_llava_demo.dart --clip PATH [flags]

Model:
  --path PATH             Llama safetensors  (default: models/llama-3.2-1b-instruct/model.safetensors)
  --vocab PATH            tokenizer.json     (default: models/llama-3.2-1b-instruct/tokenizer.json)
  --preset NAME           llama-3.2-1b | llama-3.2-3b | llama-3.1-8b (default: llama-3.2-1b)
  --gpu                   run on CUDA (default: CPU)

Vision:
  --clip PATH             CLIP safetensors (required)
  --clip-preset NAME      base32 | base16 | large14 (default: base32)
  --projector-load PATH   load a trained projector checkpoint (default: random init)
  --projector-hidden N    projector hidden width (default: llama.embedDim)

Sampling:
  --system "..."          initial system prompt
  --max-new N             max tokens per reply (default: 128)
  --temperature F         0.0 = greedy         (default: 0.7)
  --top-k K               0 = disabled         (default: 40)

REPL:
  :img PATH   attach an image for the NEXT text turn (consumed once)
  :quit       exit
  :reset      wipe history (system prompt kept)
  :sys <text> replace system prompt and reset
''';

// ---------------------------------------------------------------------------
// Image decoding & patchifying (channels-last-per-pixel — matches the
// layout the ClipHFLoader permutes into).
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
    // CLIP-style ImageNet normalisation. Skipping the mean/std would
    // shift patch activations off the manifold CLIP was trained for.
    const meanR = 0.48145466;
    const meanG = 0.4578275;
    const meanB = 0.40821073;
    const stdR = 0.26862954;
    const stdG = 0.26130258;
    const stdB = 0.27577711;
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

String _systemPrefix(String system) {
  final s = system.trim();
  if (s.isEmpty) return '<|begin_of_text|>';
  return '<|begin_of_text|>'
      '<|start_header_id|>system<|end_header_id|>\n\n$s<|eot_id|>';
}

String _userTurn(String msg) =>
    '<|start_header_id|>user<|end_header_id|>\n\n$msg<|eot_id|>'
    '<|start_header_id|>assistant<|end_header_id|>\n\n';

void main(List<String> args) {
  final opts = _parseArgs(args);
  if (opts.clip == null) {
    stderr.writeln('--clip PATH is required (see --help)');
    exit(64);
  }
  final device = opts.gpu ? Device.GPU : Device.CPU;

  // ---- Llama -----------------------------------------------------
  final loaded = loadLlamaEncoder(
    path: opts.path,
    vocabPath: opts.vocabPath,
    preset: opts.preset,
    gpu: opts.gpu,
  );
  final llama = loaded.model;
  final tok = loaded.tokenizer;
  final cfg = loaded.config;

  // ---- CLIP ------------------------------------------------------
  final clipCfg = _clipConfig(opts.clipPreset, device);
  stdout.writeln(
    '[clip] building preset=${opts.clipPreset} '
    '(image=${clipCfg.imageSize}, patch=${clipCfg.patchSize}, '
    'embed=${clipCfg.embedDim}, layers=${clipCfg.numLayers})',
  );
  final clip = CLIPVisionModel(clipCfg);
  stdout.writeln('[clip] loading safetensors from ${opts.clip}');
  final rep = ClipHFLoader.loadFile(clip, opts.clip!);
  stdout.writeln('[clip] loaded. $rep');
  clip.eval();

  // ---- LlamaVision (projector + wiring) --------------------------
  final vision = LlamaVision.build(
    vit: clip,
    llama: llama,
    projectorHiddenDim: opts.projectorHidden,
    seed: 7,
  );
  if (opts.projectorLoad != null) {
    try {
      Checkpoint.loadIntoFile(vision.projector, opts.projectorLoad!);
      stdout.writeln('[projector] loaded ${opts.projectorLoad}');
    } catch (e) {
      stderr.writeln(
        '[projector] failed to load ${opts.projectorLoad}: $e\n'
        '            falling back to random init (output will be nonsense)',
      );
    }
  } else {
    stdout.writeln(
      '[projector] RANDOM INIT — output will be nonsense until you '
      'train and load one via --projector-load',
    );
  }
  vision.eval();

  final eot = tok.llamaEotId;
  final stopId = eot ?? tok.endOfTextId;

  var system = opts.system;
  var chatText = _systemPrefix(system);
  String? pendingImage;

  stdout.writeln(
    '\nReady. Attach an image with `:img PATH`, then send a text turn.\n'
    'Commands: :quit / :reset / :sys <text>. Ctrl+D to exit.\n',
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

    final prompt = chatText + _userTurn(trimmed);
    var promptIds = tok.encode(prompt);
    final imageTokenCount = pendingImage != null ? vision.numImageTokens : 0;
    if (promptIds.length + imageTokenCount + opts.maxNew > cfg.maxCtx) {
      stdout.writeln(
        '[history + image tokens + max-new exceed context '
        '(${promptIds.length} + $imageTokenCount + ${opts.maxNew} '
        '> ${cfg.maxCtx}); resetting]',
      );
      chatText = _systemPrefix(system);
      promptIds = tok.encode(chatText + _userTurn(trimmed));
      if (promptIds.length + imageTokenCount + opts.maxNew > cfg.maxCtx) {
        stdout.writeln('[single turn still too long — skipping]\n');
        pendingImage = null;
        continue;
      }
    }

    List<double> full;
    if (pendingImage != null) {
      final patches = _decodeAndPatchify(
        pendingImage,
        size: clipCfg.imageSize,
        patchSize: clipCfg.patchSize,
        device: device,
      );
      if (patches == null) {
        stdout.writeln('[image decode failed — sending text-only]');
        pendingImage = null;
        full = llama.generate(
          promptIds.map((i) => i.toDouble()).toList(),
          maxNewTokens: opts.maxNew,
          temperature: opts.temperature,
          topK: opts.topK == 0 ? null : opts.topK,
        );
      } else {
        full = vision.generate(
          patches,
          promptIds.map((i) => i.toDouble()).toList(),
          maxNewTokens: opts.maxNew,
          temperature: opts.temperature,
          topK: opts.topK == 0 ? null : opts.topK,
          stopId: stopId,
        );
        pendingImage = null;
      }
    } else {
      full = llama.generate(
        promptIds.map((i) => i.toDouble()).toList(),
        maxNewTokens: opts.maxNew,
        temperature: opts.temperature,
        topK: opts.topK == 0 ? null : opts.topK,
      );
    }

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
