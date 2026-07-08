/// Learning-rate schedulers.
///
/// A [LRScheduler] wraps an [Optimizer] and adjusts its `lr` in place
/// via a `step()` call the training loop invokes once per optimizer
/// step. All schedulers here are stateful (they track their own step
/// counter) and idempotent under repeated `lastLr` reads.
///
/// The typical pattern:
///
/// ```dart
/// final opt = Adam(model.parameters(), lr: 1.0); // base lr = 1.0
/// final sched = LinearWarmupCosineDecay(
///   opt,
///   warmupSteps: 100,
///   totalSteps: 1000,
///   maxLr: 3e-4,
///   minLr: 3e-5,
/// );
/// for (int i = 0; i < 1000; i++) {
///   opt.zeroGrad();
///   loss(model, x, y).backward();
///   opt.step();
///   sched.step();
/// }
/// ```
///
/// Note: schedulers overwrite `optimizer.lr` on every `step()`; the
/// value the optimizer was constructed with is *not* used as the base
/// LR — the scheduler carries its own `maxLr` / `initialLr`.
library;

import 'dart:math' as math;

import 'optimizer.dart';

abstract class LRScheduler {
  final Optimizer optimizer;

  /// Steps taken so far. `0` before the first `step()` call. After
  /// the first `step()`, `stepCount == 1` and `optimizer.lr` reflects
  /// the LR *for* that first step.
  int stepCount = 0;

  LRScheduler(this.optimizer);

  /// The LR that will be applied on the next `step()` call.
  double lrAt(int step);

  /// Advance one step and write the new LR into the wrapped optimizer.
  /// Call this AFTER `optimizer.step()`.
  void step() {
    stepCount += 1;
    optimizer.lr = lrAt(stepCount);
  }

  /// The most recently applied LR (== `optimizer.lr` unless the caller
  /// mutated it externally). `null` before the first `step()`.
  double? get lastLr => stepCount == 0 ? null : optimizer.lr;
}

/// Multiplies the LR by [gamma] every [stepSize] steps.
///
///     lr(t) = initialLr * gamma^floor(t / stepSize)
class StepLR extends LRScheduler {
  final double initialLr;
  final int stepSize;
  final double gamma;

  StepLR(
    super.optimizer, {
    required this.initialLr,
    required this.stepSize,
    this.gamma = 0.1,
  }) {
    if (stepSize <= 0) {
      throw ArgumentError('StepLR: stepSize must be > 0');
    }
    optimizer.lr = initialLr;
  }

  @override
  double lrAt(int step) {
    final k = step ~/ stepSize;
    var lr = initialLr;
    for (int i = 0; i < k; i++) {
      lr *= gamma;
    }
    return lr;
  }
}

/// Linear warmup for the first [warmupSteps] steps, then cosine decay
/// from [maxLr] down to [minLr] across the remaining
/// `totalSteps - warmupSteps` steps. After [totalSteps] the LR stays
/// at [minLr].
///
/// Curve:
///
///     lr(t) = t / warmupSteps * maxLr                          for t <= warmupSteps
///           = minLr + 0.5 * (maxLr - minLr) *
///             (1 + cos(pi * (t - warmupSteps) /
///                            (totalSteps - warmupSteps)))       for warmupSteps < t <= totalSteps
///           = minLr                                              for t > totalSteps
class LinearWarmupCosineDecay extends LRScheduler {
  final int warmupSteps;
  final int totalSteps;
  final double maxLr;
  final double minLr;

  LinearWarmupCosineDecay(
    super.optimizer, {
    required this.warmupSteps,
    required this.totalSteps,
    required this.maxLr,
    this.minLr = 0.0,
  }) {
    if (warmupSteps < 0) {
      throw ArgumentError('LinearWarmupCosineDecay: warmupSteps must be >= 0');
    }
    if (totalSteps < warmupSteps) {
      throw ArgumentError(
        'LinearWarmupCosineDecay: totalSteps ($totalSteps) must be >= '
        'warmupSteps ($warmupSteps)',
      );
    }
    if (minLr > maxLr) {
      throw ArgumentError('LinearWarmupCosineDecay: minLr must be <= maxLr');
    }
    // Set the initial LR to what the first step will use.
    optimizer.lr = lrAt(1);
  }

  @override
  double lrAt(int step) {
    if (step <= 0) return 0.0;
    if (step <= warmupSteps) {
      // Linear warmup 0 -> maxLr across [1, warmupSteps].
      // At step == warmupSteps, lr == maxLr.
      return warmupSteps == 0 ? maxLr : maxLr * step / warmupSteps;
    }
    if (step >= totalSteps) return minLr;
    final progress = (step - warmupSteps) / (totalSteps - warmupSteps);
    return minLr + 0.5 * (maxLr - minLr) * (1.0 + math.cos(math.pi * progress));
  }
}
