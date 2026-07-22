/// coop_coordinator - HTTP server for Phase 1 cooperative training.
///
/// A single coordinator process holds the "global" DPTC checkpoint.
/// Workers pull the checkpoint, train K local Adam steps on their own
/// shard, and POST the updated checkpoint back. When the coordinator
/// has accumulated N updates it averages them (weighted by local steps),
/// replaces the global checkpoint, bumps `round`, and clears the queue.
///
/// Protocol (all responses application/octet-stream unless noted):
///   GET  /config       200 - JSON {vocab, gptConfig, blockSize,
///                                   corpusPath (may be null), version}
///   GET  /corpus       200 - text/plain corpus that all workers use
///                             (only served when the coordinator has one)
///   GET  /checkpoint   200 - raw DPTC bytes of current global model
///                             + header X-Coop-Round: N
///   POST /submit       body = DPTC bytes from a worker
///                             headers:
///                               X-Coop-Worker-Id: string
///                               X-Coop-Round:     int (round they pulled)
///                               X-Coop-Local-Steps: int (weight)
///                             response: 200 {accepted, round, queue}
///                                       409 if X-Coop-Round is stale
///   GET  /status       200 JSON {round, queueSize, target, workersSeen}
///
/// Run:
///   dart run bin/coop_coordinator.dart --port=8080 --target=3
///
/// Flags:
///   --port=N          bind port (default 8080)
///   --host=HOST       bind host (default 127.0.0.1; use 0.0.0.0 for LAN)
///   --target=N        min updates to trigger an averaging round (default 2)
///   --model=tiny|small
///   --corpus=toy|shakespeare
///   --seed=N          initial-weights seed (default 42)
///   --device=cpu|gpu
///
/// SECURITY: this is a POC. There is no auth, no TLS, no worker
/// validation. Run on 127.0.0.1 or a private LAN only. Never expose
/// to the public internet without adding auth + input clamping.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';

class _CoordConfig {
  int port = 8080;
  String host = '127.0.0.1';
  int target = 2;
  int seed = 42;
  String model = 'tiny';
  String corpus = 'toy';
  Device device = Device.CPU;
}

_CoordConfig _parseArgs(List<String> args) {
  final c = _CoordConfig();
  for (final raw in args) {
    final a = raw.startsWith('--') ? raw.substring(2) : raw;
    final eq = a.indexOf('=');
    if (eq < 0) continue;
    final k = a.substring(0, eq);
    final v = a.substring(eq + 1);
    switch (k) {
      case 'port':
        c.port = int.parse(v);
      case 'host':
        c.host = v;
      case 'target':
        c.target = int.parse(v);
      case 'seed':
        c.seed = int.parse(v);
      case 'model':
        c.model = v;
      case 'corpus':
        c.corpus = v;
      case 'device':
        c.device = v.toLowerCase() == 'gpu' ? Device.GPU : Device.CPU;
      default:
        stderr.writeln('coop_coordinator: unknown flag --$k');
        exit(2);
    }
  }
  return c;
}

GPTConfig _makeGptConfig(String model, int vocabSize, Device device, int seed) {
  switch (model) {
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
      stderr.writeln('coop_coordinator: unknown --model=$model');
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
          'coop_coordinator: --corpus=shakespeare requires '
          'data/tiny_shakespeare.txt (run from repo root)',
        );
        exit(1);
      }
      return f.readAsStringSync().substring(0, 40000);
    default:
      stderr.writeln('coop_coordinator: unknown --corpus=$kind');
      exit(2);
  }
}

class _Submission {
  _Submission(this.workerId, this.localSteps, this.bytes);
  final String workerId;
  final int localSteps;
  final Uint8List bytes;
}

Future<void> main(List<String> args) async {
  final cfg = _parseArgs(args);
  final text = _loadCorpus(cfg.corpus);
  final vocab = CharVocab.fromText(text);
  final gptCfg = _makeGptConfig(cfg.model, vocab.size, cfg.device, cfg.seed);

  // Initial global model.
  final gpt = GPT(gptCfg);
  var globalBytes = Checkpoint.saveBytes(gpt);
  var round = 0;
  final workersSeen = <String>{};
  final queue = <_Submission>[];

  final configJson = jsonEncode({
    'version': 1,
    'model': cfg.model,
    'corpus': cfg.corpus,
    'vocab': vocab.itos,
    'gptConfig': {
      'vocabSize': gptCfg.vocabSize,
      'maxCtx': gptCfg.maxCtx,
      'embedDim': gptCfg.embedDim,
      'numLayers': gptCfg.numLayers,
      'numHeads': gptCfg.numHeads,
      'tieWeights': gptCfg.tieWeights,
      'seed': gptCfg.seed,
    },
    'target': cfg.target,
  });

  final server = await HttpServer.bind(cfg.host, cfg.port);
  print('coop_coordinator: listening on http://${cfg.host}:${cfg.port}');
  print(
    '  model=${cfg.model} corpus=${cfg.corpus} vocab=${vocab.size} '
    'target=${cfg.target}',
  );
  print('  initial round=$round, checkpoint=${globalBytes.length} bytes');

  await for (final req in server) {
    try {
      final path = req.uri.path;
      switch ('${req.method} $path') {
        case 'GET /config':
          req.response
            ..headers.contentType = ContentType.json
            ..write(configJson);

        case 'GET /corpus':
          req.response
            ..headers.contentType = ContentType.text
            ..write(text);

        case 'GET /checkpoint':
          req.response.headers
            ..contentType = ContentType.binary
            ..add('X-Coop-Round', '$round');
          req.response.add(globalBytes);

        case 'GET /status':
          req.response
            ..headers.contentType = ContentType.json
            ..write(
              jsonEncode({
                'round': round,
                'queueSize': queue.length,
                'target': cfg.target,
                'workersSeen': workersSeen.toList()..sort(),
              }),
            );

        case 'POST /submit':
          final workerId = req.headers.value('X-Coop-Worker-Id') ?? '?';
          final subRoundStr = req.headers.value('X-Coop-Round') ?? '-1';
          final localStepsStr = req.headers.value('X-Coop-Local-Steps') ?? '1';
          final subRound = int.tryParse(subRoundStr) ?? -1;
          final localSteps = int.tryParse(localStepsStr) ?? 1;

          final body = await _readAll(req);
          if (subRound != round) {
            req.response
              ..statusCode = HttpStatus.conflict
              ..headers.contentType = ContentType.json
              ..write(
                jsonEncode({
                  'accepted': false,
                  'reason': 'stale round',
                  'workerRound': subRound,
                  'currentRound': round,
                }),
              );
            break;
          }
          try {
            // Validate by loading into a scratch model.
            final scratch = GPT(gptCfg);
            Checkpoint.loadIntoBytes(scratch, body);
          } catch (e) {
            req.response
              ..statusCode = HttpStatus.badRequest
              ..headers.contentType = ContentType.json
              ..write(
                jsonEncode({
                  'accepted': false,
                  'reason': 'invalid checkpoint: $e',
                }),
              );
            break;
          }
          queue.add(_Submission(workerId, localSteps, body));
          workersSeen.add(workerId);

          var justAggregated = false;
          if (queue.length >= cfg.target) {
            final bytes = [for (final s in queue) s.bytes];
            final weights = [for (final s in queue) s.localSteps.toDouble()];
            // Include the current global in the average with weight
            // equal to the median submission (so a fresh coordinator
            // still counts and one crazy worker cannot yank the mean).
            bytes.add(globalBytes);
            final medianW = _median(weights);
            weights.add(medianW);
            globalBytes = averageCheckpoints(bytes, weights: weights);
            round += 1;
            print(
              'round $round aggregated ${queue.length} updates '
              '(workers: ${queue.map((s) => s.workerId).toSet().toList()..sort()})',
            );
            queue.clear();
            justAggregated = true;
          }
          req.response
            ..headers.contentType = ContentType.json
            ..write(
              jsonEncode({
                'accepted': true,
                'round': round,
                'queueSize': queue.length,
                'aggregated': justAggregated,
              }),
            );

        default:
          req.response
            ..statusCode = HttpStatus.notFound
            ..write('unknown route');
      }
    } catch (e, st) {
      stderr.writeln('coop_coordinator: error handling ${req.uri}: $e\n$st');
      req.response.statusCode = HttpStatus.internalServerError;
    } finally {
      await req.response.close();
    }
  }
}

Future<Uint8List> _readAll(HttpRequest req) async {
  final chunks = <List<int>>[];
  await for (final c in req) {
    chunks.add(c);
  }
  final total = chunks.fold<int>(0, (a, b) => a + b.length);
  final out = Uint8List(total);
  var off = 0;
  for (final c in chunks) {
    out.setRange(off, off + c.length, c);
    off += c.length;
  }
  return out;
}

double _median(List<double> xs) {
  if (xs.isEmpty) return 1.0;
  final sorted = List<double>.from(xs)..sort();
  final mid = sorted.length ~/ 2;
  return sorted.length.isOdd
      ? sorted[mid]
      : 0.5 * (sorted[mid - 1] + sorted[mid]);
}
