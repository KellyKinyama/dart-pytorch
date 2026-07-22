/// coop_train_phase0 - single-process DiLoCo proof-of-concept.
///
/// Trains N GPT replicas on disjoint shards of the same corpus, averages
/// their parameters every K local steps, and shows that the averaged
/// model's loss is <= the best single-replica loss (and usually below
/// both). No sockets, no network, no coordinator - just the math.
///
/// This is the same paradigm as Google DeepMind's DiLoCo
/// (https://arxiv.org/abs/2311.08105) and Hivemind's local-SGD trainer:
/// each replica does K local Adam steps on its own data shard, then all
/// replicas' parameters are averaged and every replica adopts the
/// average as its new starting point.
///
/// Run:
///   dart run bin/coop_train_phase0.dart
///
/// Flags:
///   --replicas=N       number of parallel replicas (default: 3)
///   --rounds=N         averaging rounds (default: 8)
///   --local-steps=K    local Adam steps per round per replica (default: 30)
///   --seed=N           RNG seed (default: 42)
///   --device=cpu|gpu   default cpu
///   --model=tiny|small default tiny (~50k params)
///   --corpus=toy|shakespeare  default toy (built-in, self-contained)
library;

import 'dart:io';
import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

class _Config {
  int replicas = 3;
  int rounds = 8;
  int localSteps = 30;
  int seed = 42;
  Device device = Device.CPU;
  String model = 'tiny';
  String corpus = 'toy';
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
      case 'replicas':
        c.replicas = int.parse(v);
      case 'rounds':
        c.rounds = int.parse(v);
      case 'local-steps':
        c.localSteps = int.parse(v);
      case 'seed':
        c.seed = int.parse(v);
      case 'device':
        c.device = v.toLowerCase() == 'gpu' ? Device.GPU : Device.CPU;
      case 'model':
        c.model = v;
      case 'corpus':
        c.corpus = v;
      default:
        stderr.writeln('coop_train_phase0: unknown flag --$k');
        exit(2);
    }
  }
  return c;
}

GPTConfig _modelConfig(_Config cfg, int vocabSize) {
  switch (cfg.model) {
    case 'tiny':
      return GPTConfig(
        vocabSize: vocabSize,
        maxCtx: 32,
        embedDim: 32,
        numLayers: 2,
        numHeads: 4,
        dropoutP: 0.0,
        tieWeights: true,
        device: cfg.device,
        seed: cfg.seed,
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
        device: cfg.device,
        seed: cfg.seed,
      );
    default:
      stderr.writeln('coop_train_phase0: unknown --model=${cfg.model}');
      exit(2);
  }
}

String _loadCorpus(String kind) {
  switch (kind) {
    case 'toy':
      return kToyCorpus;
    case 'shakespeare':
      final f = File('data/tiny_shakespeare.txt');
      if (!f.existsSync()) {
        stderr.writeln(
          'coop_train_phase0: --corpus=shakespeare requires '
          'data/tiny_shakespeare.txt (run from repo root)',
        );
        exit(1);
      }
      return f.readAsStringSync().substring(0, 40000);
    default:
      stderr.writeln('coop_train_phase0: unknown --corpus=$kind');
      exit(2);
  }
}

double _evalLoss(
  GPT gpt,
  List<double> ids,
  int blockSize,
  math.Random rng, {
  int steps = 8,
  required Device device,
}) {
  var total = 0.0;
  for (var i = 0; i < steps; i++) {
    final (x, y) = sampleWindow(ids, blockSize, rng, device: device);
    final loss = gpt(x).crossEntropy(y).mean();
    total += loss.toList()[0];
  }
  return total / steps;
}

void main(List<String> args) {
  final cfg = _parseArgs(args);
  print(
    'coop_train_phase0: ${cfg.replicas} replicas x ${cfg.rounds} rounds '
    'x ${cfg.localSteps} local steps, model=${cfg.model}, '
    'device=${cfg.device}, corpus=${cfg.corpus}',
  );

  // ---- Corpus + vocab ----
  final text = _loadCorpus(cfg.corpus);
  final vocab = CharVocab.fromText(text);
  final ids = vocab.encode(text);
  print('Corpus: ${text.length} chars, vocab=${vocab.size}');

  // ---- Instantiate replicas ----
  final gptConfig = _modelConfig(cfg, vocab.size);
  final replicas = <GPT>[];
  final opts = <Adam>[];
  final shards = <List<double>>[];
  for (var r = 0; r < cfg.replicas; r++) {
    replicas.add(GPT(gptConfig));
    opts.add(Adam(replicas[r].parameters(), lr: 3e-3));
    shards.add(shardSlice(ids, r, cfg.replicas));
    print(
      '  replica $r: ${shards[r].length} tokens '
      '(${replicas[r].parameters().fold<int>(0, (a, p) => a + p.length)} '
      'scalars)',
    );
  }

  // Give every replica the same initial parameters (average of the
  // freshly-initialised replicas). Otherwise each replica has a
  // different random init and we cannot compare "same-start" runs.
  {
    final ckpts = [for (final g in replicas) Checkpoint.saveBytes(g)];
    final avg = averageCheckpoints(ckpts);
    for (final g in replicas) {
      Checkpoint.loadIntoBytes(g, avg);
    }
    print('Replicas synced to a common initial checkpoint');
  }

  // ---- Baseline: single replica trained on all data, same total steps ----
  final baseline = GPT(gptConfig);
  Checkpoint.loadIntoBytes(baseline, Checkpoint.saveBytes(replicas[0]));
  final baseOpt = Adam(baseline.parameters(), lr: 3e-3);
  final baseRng = math.Random(cfg.seed + 999);
  final blockSize = gptConfig.maxCtx - 1;

  // ---- Cooperative training ----
  final rngs = [
    for (var r = 0; r < cfg.replicas; r++) math.Random(cfg.seed + 100 + r),
  ];
  final evalRng = math.Random(cfg.seed + 12345);

  print('');
  final headerCols = <String>[
    'round',
    for (var r = 0; r < cfg.replicas; r++) 'replica_$r',
    'avg',
    'baseline',
  ];
  print(headerCols.map((s) => s.padLeft(12)).join(' '));

  for (var round = 1; round <= cfg.rounds; round++) {
    // Train each replica locally for K steps on its shard.
    for (var r = 0; r < cfg.replicas; r++) {
      for (var step = 0; step < cfg.localSteps; step++) {
        opts[r].zeroGrad();
        final (x, y) = sampleWindow(
          shards[r],
          blockSize,
          rngs[r],
          device: cfg.device,
        );
        final loss = replicas[r](x).crossEntropy(y).mean();
        loss.backward();
        clipGradNorm(replicas[r].parameters(), 1.0);
        opts[r].step();
      }
    }

    // Train the baseline for the same total-step budget (K * replicas)
    // on the full corpus. Ensures the comparison is honest.
    for (var step = 0; step < cfg.localSteps * cfg.replicas; step++) {
      baseOpt.zeroGrad();
      final (x, y) = sampleWindow(ids, blockSize, baseRng, device: cfg.device);
      final loss = baseline(x).crossEntropy(y).mean();
      loss.backward();
      clipGradNorm(baseline.parameters(), 1.0);
      baseOpt.step();
    }

    // Average the replica checkpoints and adopt in every replica.
    final ckpts = [for (final g in replicas) Checkpoint.saveBytes(g)];
    final avg = averageCheckpoints(ckpts);
    for (final g in replicas) {
      Checkpoint.loadIntoBytes(g, avg);
    }

    // Evaluate each replica *just after* the parameter sync (so they
    // all show the same number), and evaluate the baseline.
    final perReplicaLoss = <double>[
      for (final g in replicas)
        _evalLoss(g, ids, blockSize, evalRng, device: cfg.device),
    ];
    final avgLoss = perReplicaLoss.reduce((a, b) => a + b) / cfg.replicas;
    final baseLoss = _evalLoss(
      baseline,
      ids,
      blockSize,
      evalRng,
      device: cfg.device,
    );

    final row = <String>[
      '$round',
      for (final l in perReplicaLoss) l.toStringAsFixed(4),
      avgLoss.toStringAsFixed(4),
      baseLoss.toStringAsFixed(4),
    ];
    print(row.map((s) => s.padLeft(12)).join(' '));
  }

  print(
    '\ndone. If "avg" tracks "baseline" (or beats it), the DiLoCo '
    'math works and Phase 1/2 (add networking) is safe to build on.',
  );
}
