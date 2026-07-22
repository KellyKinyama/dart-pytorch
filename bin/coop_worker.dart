/// coop_worker - Phase 1 worker that talks to bin/coop_coordinator.dart.
///
/// Loop:
///   1. GET  /config      (once, at startup - build a matching GPT)
///   2. GET  /corpus      (once, at startup)
///   3. GET  /checkpoint  (fetch current global weights + round)
///   4. Load into local model
///   5. Train --local-steps K Adam steps on the assigned shard
///   6. POST /submit with the updated checkpoint bytes
///   7. Sleep briefly and loop to step 3
///
/// Run (two workers on one machine against a local coordinator):
///
///   # terminal A:  dart run bin/coop_coordinator.dart --target=2
///   # terminal B:  dart run bin/coop_worker.dart --id=alice --shard-id=0 --num-shards=2
///   # terminal C:  dart run bin/coop_worker.dart --id=bob   --shard-id=1 --num-shards=2
///
/// Flags:
///   --coordinator=URL   default http://127.0.0.1:8080
///   --id=NAME           worker id (default: `worker-<pid>`)
///   --shard-id=N        which corpus shard to train on (default 0)
///   --num-shards=N      total shards (default 2)
///   --local-steps=K     Adam steps per round (default 30)
///   --rounds=N          max rounds before exiting (default 20; -1 = forever)
///   --lr=F              Adam learning rate (default 3e-3)
///   --seed=N            local RNG seed (default: derived from --id)
///   --device=cpu|gpu    default cpu
///   --idle-ms=N         sleep between rounds (default 100)
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';

class _WorkerConfig {
  String coordinator = 'http://127.0.0.1:8080';
  String id = 'worker-$pid';
  int shardId = 0;
  int numShards = 2;
  int localSteps = 30;
  int rounds = 20;
  double lr = 3e-3;
  int? seed;
  Device device = Device.CPU;
  int idleMs = 100;
}

_WorkerConfig _parseArgs(List<String> args) {
  final c = _WorkerConfig();
  for (final raw in args) {
    final a = raw.startsWith('--') ? raw.substring(2) : raw;
    final eq = a.indexOf('=');
    if (eq < 0) continue;
    final k = a.substring(0, eq);
    final v = a.substring(eq + 1);
    switch (k) {
      case 'coordinator':
        c.coordinator = v;
      case 'id':
        c.id = v;
      case 'shard-id':
        c.shardId = int.parse(v);
      case 'num-shards':
        c.numShards = int.parse(v);
      case 'local-steps':
        c.localSteps = int.parse(v);
      case 'rounds':
        c.rounds = int.parse(v);
      case 'lr':
        c.lr = double.parse(v);
      case 'seed':
        c.seed = int.parse(v);
      case 'device':
        c.device = v.toLowerCase() == 'gpu' ? Device.GPU : Device.CPU;
      case 'idle-ms':
        c.idleMs = int.parse(v);
      default:
        stderr.writeln('coop_worker: unknown flag --$k');
        exit(2);
    }
  }
  return c;
}

Future<Map<String, dynamic>> _getJson(HttpClient client, Uri url) async {
  final req = await client.getUrl(url);
  final resp = await req.close();
  if (resp.statusCode != 200) {
    throw StateError('GET $url returned ${resp.statusCode}');
  }
  final body = await resp.transform(utf8.decoder).join();
  return jsonDecode(body) as Map<String, dynamic>;
}

Future<String> _getString(HttpClient client, Uri url) async {
  final req = await client.getUrl(url);
  final resp = await req.close();
  if (resp.statusCode != 200) {
    throw StateError('GET $url returned ${resp.statusCode}');
  }
  return resp.transform(utf8.decoder).join();
}

Future<(Uint8List, int)> _getCheckpoint(HttpClient client, Uri url) async {
  final req = await client.getUrl(url);
  final resp = await req.close();
  if (resp.statusCode != 200) {
    throw StateError('GET $url returned ${resp.statusCode}');
  }
  final round = int.parse(resp.headers.value('X-Coop-Round') ?? '-1');
  final chunks = <List<int>>[];
  await for (final c in resp) {
    chunks.add(c);
  }
  final total = chunks.fold<int>(0, (a, b) => a + b.length);
  final out = Uint8List(total);
  var off = 0;
  for (final c in chunks) {
    out.setRange(off, off + c.length, c);
    off += c.length;
  }
  return (out, round);
}

Future<Map<String, dynamic>> _postCheckpoint(
  HttpClient client,
  Uri url, {
  required String workerId,
  required int round,
  required int localSteps,
  required Uint8List body,
}) async {
  final req = await client.postUrl(url);
  req.headers
    ..contentType = ContentType.binary
    ..add('X-Coop-Worker-Id', workerId)
    ..add('X-Coop-Round', '$round')
    ..add('X-Coop-Local-Steps', '$localSteps');
  req.contentLength = body.length;
  req.add(body);
  final resp = await req.close();
  final text = await resp.transform(utf8.decoder).join();
  return {'status': resp.statusCode, ...jsonDecode(text) as Map};
}

GPTConfig _configFromJson(
  Map<String, dynamic> cfg,
  int vocabSize,
  Device device,
) {
  final g = cfg['gptConfig'] as Map<String, dynamic>;
  return GPTConfig(
    vocabSize: vocabSize,
    maxCtx: g['maxCtx'] as int,
    embedDim: g['embedDim'] as int,
    numLayers: g['numLayers'] as int,
    numHeads: g['numHeads'] as int,
    dropoutP: 0.0,
    tieWeights: g['tieWeights'] as bool,
    device: device,
    seed: g['seed'] as int,
  );
}

Future<void> main(List<String> args) async {
  final cfg = _parseArgs(args);
  final base = Uri.parse(cfg.coordinator);
  final client = HttpClient();
  print('coop_worker "${cfg.id}": talking to ${cfg.coordinator}');

  // Handshake.
  final serverCfg = await _getJson(client, base.resolve('/config'));
  final vocab = CharVocab.fromItos((serverCfg['vocab'] as List).cast<String>());
  final gptCfg = _configFromJson(serverCfg, vocab.size, cfg.device);
  final gpt = GPT(gptCfg);
  print(
    '  built local GPT: '
    '${gpt.parameters().fold<int>(0, (a, p) => a + p.length)} scalars',
  );

  final corpus = await _getString(client, base.resolve('/corpus'));
  final ids = vocab.encode(corpus);
  final shard = shardSlice(ids, cfg.shardId, cfg.numShards);
  print(
    '  corpus=${corpus.length} chars, shard=${cfg.shardId}/${cfg.numShards}'
    ' (${shard.length} tokens)',
  );

  final opt = Adam(gpt.parameters(), lr: cfg.lr);
  final rng = math.Random(cfg.seed ?? cfg.id.hashCode & 0x7fffffff);
  final blockSize = gptCfg.maxCtx - 1;

  var contributed = 0;
  while (cfg.rounds < 0 || contributed < cfg.rounds) {
    // Pull.
    final (bytes, serverRound) = await _getCheckpoint(
      client,
      base.resolve('/checkpoint'),
    );
    Checkpoint.loadIntoBytes(gpt, bytes);

    // Local train.
    var lossAccum = 0.0;
    for (var s = 0; s < cfg.localSteps; s++) {
      opt.zeroGrad();
      final (x, y) = sampleWindow(shard, blockSize, rng, device: cfg.device);
      final loss = gpt(x).crossEntropy(y).mean();
      loss.backward();
      clipGradNorm(gpt.parameters(), 1.0);
      opt.step();
      lossAccum += loss.toList()[0];
    }

    // Push.
    final out = Checkpoint.saveBytes(gpt);
    final result = await _postCheckpoint(
      client,
      base.resolve('/submit'),
      workerId: cfg.id,
      round: serverRound,
      localSteps: cfg.localSteps,
      body: out,
    );
    contributed += 1;
    final avgLoss = (lossAccum / cfg.localSteps).toStringAsFixed(4);
    if (result['status'] == 200 && result['accepted'] == true) {
      print(
        '  contrib #$contributed  from round=$serverRound '
        '-> now round=${result['round']}  local loss=$avgLoss'
        '${result['aggregated'] == true ? "  [aggregated]" : ""}',
      );
    } else {
      print('  contrib #$contributed  rejected: ${jsonEncode(result)}');
    }

    if (cfg.idleMs > 0) {
      await Future<void>.delayed(Duration(milliseconds: cfg.idleMs));
    }
  }
  print('coop_worker "${cfg.id}": done after $contributed contributions');
  client.close(force: true);
}
