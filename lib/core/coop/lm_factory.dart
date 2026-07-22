/// Small factory that produces either a [GPT] (attention-based) or
/// an [AFTLanguageModel] (attention-free) language model behind a
/// uniform interface, so the coop_* training scripts can flip
/// architectures with a single `--arch=gpt|aft` flag without
/// duplicating the training loop.
///
/// Both models already accept a rank-1 `[seqLen]` token tensor and
/// return rank-2 `[seqLen, vocabSize]` logits — the training loops in
/// bin/coop_*.dart never batch beyond that, so the two are drop-in
/// interchangeable at the `.call(tokens)` level.
///
/// **DPTC compatibility:** [averageCheckpoints] requires that every
/// input describes the same architecture (byte-identical DPTC header).
/// Mixing a GPT checkpoint with an AFT checkpoint will therefore
/// raise a clean `ArgumentError` at aggregation time — the coop
/// coordinator (Phase 1) advertises the arch via GET /config and
/// workers refuse to join a fleet whose arch doesn't match theirs.
library;

import 'dart:io';

import '../nn/aft_transformer.dart';
import '../nn/gpt.dart';
import '../nn/module.dart';
import '../tensor/tensor.dart';

/// Which transformer family to build. Serialized as its lower-case
/// name in JSON so coordinator ↔ worker exchange is trivial.
enum Arch { gpt, aft }

Arch parseArch(String s) {
  switch (s.toLowerCase()) {
    case 'gpt':
      return Arch.gpt;
    case 'aft':
      return Arch.aft;
    default:
      throw ArgumentError('unknown arch "$s" (expected gpt|aft)');
  }
}

String archToString(Arch a) => a == Arch.gpt ? 'gpt' : 'aft';

/// A language model wrapped so callers can treat GPT and AFT
/// identically: forward with `.call(tokens)`, get parameters via
/// `.module.parameters()`, checkpoint via `Checkpoint.saveBytes(.module)`.
class CoopLM {
  final Arch arch;
  final String modelSize;
  final Module module;
  final Tensor Function(Tensor tokens) forward;
  final int maxLen;

  const CoopLM({
    required this.arch,
    required this.modelSize,
    required this.module,
    required this.forward,
    required this.maxLen,
  });

  /// Forward pass. `tokens` is rank-1 `[seqLen]`.
  Tensor call(Tensor tokens) => forward(tokens);

  /// Total number of scalar parameters.
  int get scalarCount =>
      module.parameters().fold<int>(0, (a, p) => a + p.length);
}

/// Build a fresh model. `modelSize` is `tiny` or `small` and picks
/// matching dims for both families (so a 3-peer fleet running
/// --arch=aft --model=tiny has the same param budget as one running
/// --arch=gpt --model=tiny, within ~10%).
///
/// AFT is CPU-only in this library. If `device: Device.GPU` is
/// requested with arch=aft, we log a warning and fall back to CPU so
/// the training loop doesn't blow up mid-round on a device-mismatch.
CoopLM buildCoopLM({
  required Arch arch,
  required String modelSize,
  required int vocabSize,
  required Device device,
  required int seed,
}) {
  switch (arch) {
    case Arch.gpt:
      final cfg = _gptConfig(modelSize, vocabSize, device, seed);
      final gpt = GPT(cfg);
      return CoopLM(
        arch: arch,
        modelSize: modelSize,
        module: gpt,
        forward: (t) => gpt(t),
        maxLen: cfg.maxCtx,
      );
    case Arch.aft:
      if (device == Device.GPU) {
        stderr.writeln(
          'coop: AFT is CPU-only in this library; falling back to CPU.',
        );
        device = Device.CPU;
      }
      final dims = _aftDims(modelSize);
      final aft = AFTLanguageModel(
        vocabSize: vocabSize,
        embedDim: dims.embedDim,
        numLayers: dims.numLayers,
        maxLen: dims.maxLen,
        ffnDim: dims.ffnDim,
        dropoutP: 0.0,
        device: device,
        seed: seed,
      );
      return CoopLM(
        arch: arch,
        modelSize: modelSize,
        module: aft,
        forward: (t) => aft(t),
        maxLen: dims.maxLen,
      );
  }
}

GPTConfig _gptConfig(String size, int vocabSize, Device device, int seed) {
  switch (size) {
    case 'tiny':
      return GPTConfig(
        vocabSize: vocabSize,
        maxCtx: 32,
        embedDim: 32,
        numLayers: 2,
        numHeads: 4,
        dropoutP: 0.0,
        tieWeights: true,
        device: device,
        seed: seed,
      );
    case 'small':
      return GPTConfig(
        vocabSize: vocabSize,
        maxCtx: 64,
        embedDim: 128,
        numLayers: 4,
        numHeads: 8,
        dropoutP: 0.0,
        tieWeights: true,
        device: device,
        seed: seed,
      );
    default:
      throw ArgumentError('unknown model size "$size" (expected tiny|small)');
  }
}

class _AftDims {
  final int maxLen;
  final int embedDim;
  final int numLayers;
  final int ffnDim;
  const _AftDims(this.maxLen, this.embedDim, this.numLayers, this.ffnDim);
}

_AftDims _aftDims(String size) {
  switch (size) {
    // Same width / depth / ctx as the GPT counterparts so a
    // side-by-side comparison is honest.
    case 'tiny':
      return const _AftDims(32, 32, 2, 128);
    case 'small':
      return const _AftDims(64, 128, 4, 512);
    default:
      throw ArgumentError('unknown model size "$size" (expected tiny|small)');
  }
}
