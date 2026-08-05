/// Llama counterpart to `bin/_lm_encoder.dart`.
///
/// Provides the same three-ish helpers as `_lm_encoder.dart` but for
/// the [Llama] architecture (RMSNorm + GQA + SwiGLU + RoPE):
///
///   * `lastTokenHiddenLlama(model, ids)` — runs the transformer
///     stack (no LM head), returns the last-position hidden state
///     as a `Float32List`. Same "encoder-only" trick as the GPT one.
///   * `configForLlamaPreset(preset, device)` — LlamaConfig factory
///     mirroring the GPT `configForPreset` in `_lm_encoder.dart`.
///   * `loadLlamaEncoder(opts)` — build model, load safetensors,
///     load tokenizer.json, put into eval mode.
///
/// Not a `dart run` entry point — leading underscore signals it's
/// a library helper for `rag_chat_server.dart`.
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// Run `model` as an encoder over `tokenIds` and return the hidden
/// state at the last position (after `finalNorm`). Skips the LM head.
Float32List lastTokenHiddenLlama(Llama model, List<int> tokenIds) {
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

    var h = model.embedIn(tokens); // [N, D]
    final n = clipped.length;
    final mask = n > 1 ? causalMask(n, device: h.device) : null;
    for (int i = 0; i < model.blocks.length; i++) {
      h = model.blocks[i](h, mask: mask, startPos: 0);
    }
    h = model.finalNorm(h);

    final flat = h.toList();
    final d = model.config.embedDim;
    final out = Float32List(d);
    final base = (n - 1) * d;
    for (var j = 0; j < d; j++) {
      out[j] = flat[base + j].toDouble();
    }
    return out;
  });
}

// ---------------------------------------------------------------------------
// Config presets — mirror the naming in _lm_encoder.dart.
// ---------------------------------------------------------------------------

LlamaConfig configForLlamaPreset(String preset, Device device) {
  switch (preset) {
    case 'llama-3.2-1b':
    case 'llama-3.2-1b-instruct':
      return LlamaHFLoader.llama32_1BConfig(device: device);
    case 'llama-3.2-3b':
    case 'llama-3.2-3b-instruct':
      return LlamaHFLoader.llama32_3BConfig(device: device);
    case 'llama-3.1-8b':
    case 'llama-3.1-8b-instruct':
      return LlamaHFLoader.llama31_8BConfig(device: device);
    default:
      stderr.writeln(
        'unknown llama preset "$preset"; use '
        'llama-3.2-1b | llama-3.2-3b | llama-3.1-8b',
      );
      exit(64);
  }
}

/// Build a Llama model, load its safetensors, load its tokenizer.json,
/// eval-mode it. Same shape as `loadEncoder` in `_lm_encoder.dart` but
/// returns a Llama and a LlamaConfig.
({Llama model, HFBpeTokenizer tokenizer, LlamaConfig config})
loadLlamaEncoder({
  required String path,
  required String vocabPath,
  required String preset,
  required bool gpu,
  void Function(String)? log,
}) {
  final say = log ?? stdout.writeln;
  final device = gpu ? Device.GPU : Device.CPU;
  final cfg = configForLlamaPreset(preset, device);
  say(
    'Building Llama (preset=$preset, device=${gpu ? "gpu" : "cpu"}, '
    'embed=${cfg.embedDim}, layers=${cfg.numLayers}, '
    'heads=${cfg.numHeads}, kv=${cfg.numKvHeads})',
  );
  final model = Llama(cfg);
  say('Loading safetensors from $path ...');
  final report = LlamaHFLoader.loadFile(model, path);
  say('Loaded. $report');
  say('Loading tokenizer from $vocabPath');
  final tokenizer = HFBpeTokenizer.loadFile(vocabPath);
  model.eval();
  return (model: model, tokenizer: tokenizer, config: cfg);
}
