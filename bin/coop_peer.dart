/// coop_peer - Phase 2 fully decentralized gossip trainer.
///
/// Each peer runs its own tiny HTTP server (exposing GET /checkpoint
/// and GET /health), knows about a bootstrap peer set via --peers, and
/// on each round:
///
///   1. Train --local-steps K Adam steps on its own shard.
///   2. Pick one random peer from --peers.
///   3. GET their /checkpoint, average with own (equal weights), and
///      adopt the average as the new local weights.
///
/// There is no coordinator, no single point of failure. This is
/// gossip-average SGD - a diffusion-based approximation to all-reduce,
/// used by e.g. Hivemind's DecentralizedSGD.
///
/// Convergence intuition: after enough gossip rounds every peer's
/// parameters converge to the fleet-wide average, so the whole fleet
/// behaves as if it were doing synchronous parameter averaging (Phase 1
/// coordinator style) but without any central node.
///
/// Bootstrap protocol: peers must agree on model config + vocab before
/// they can average. This demo assumes all peers are launched with the
/// same --model + --corpus flags. In production you would ship a
/// config manifest out-of-band (or use a bootstrap peer that serves
/// GET /config like the coordinator does).
///
/// Run three peers on one box:
///
///   dart run bin/coop_peer.dart --listen=9001 --id=p1 --shard-id=0 --num-shards=3 \
///       --peers=127.0.0.1:9002,127.0.0.1:9003 &
///   dart run bin/coop_peer.dart --listen=9002 --id=p2 --shard-id=1 --num-shards=3 \
///       --peers=127.0.0.1:9001,127.0.0.1:9003 &
///   dart run bin/coop_peer.dart --listen=9003 --id=p3 --shard-id=2 --num-shards=3 \
///       --peers=127.0.0.1:9001,127.0.0.1:9002
///
/// Flags:
///   --listen=PORT       TCP port this peer serves on (default 9001)
///   --host=HOST         bind host (default 127.0.0.1)
///   --peers=h:p,h:p     bootstrap peer list (comma-separated); empty
///                        means solo mode (train, but never gossip)
///   --id=NAME           this peer's id (default: `peer-<pid>`)
///   --shard-id=N        which corpus shard to train on (default 0)
///   --num-shards=N      total shards (default = number of peers + 1)
///   --local-steps=K     Adam steps between gossip rounds (default 30)
///   --gossip-every=R    gossip once every R rounds (default 1)
///   --rounds=N          max rounds before exiting (default 20; -1 = forever)
///   --arch=gpt|aft      transformer family (default: gpt). Every
///                        peer in the mesh MUST agree on this; a
///                        gpt peer trying to average a checkpoint
///                        pulled from an aft peer will fail loudly
///                        at `averageCheckpoints` header validation.
///   --model=tiny|small
///   --corpus=toy|shakespeare
///   --lr=F              Adam learning rate (default 3e-3)
///   --seed=N            model init seed (default 42; identical across
///                        peers so they start from the same weights)
///   --warmup-secs=N     wait N seconds after binding before starting
///                        training (default 0). Set to ~15s on a
///                        single box so all peers bind before anyone
///                        starts gossiping.
///   --device=cpu|gpu
///
/// SECURITY: same as Phase 1 - no auth, no TLS, no validation.
/// LAN/localhost only. See doc/coop-training.md.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';

class _PeerConfig {
  int listen = 9001;
  String host = '127.0.0.1';
  List<String> peers = const [];
  String id = 'peer-$pid';
  int shardId = 0;
  int? numShards;
  int localSteps = 30;
  int gossipEvery = 1;
  int rounds = 20;
  Arch arch = Arch.gpt;
  String model = 'tiny';
  String corpus = 'toy';
  double lr = 3e-3;
  int seed = 42;
  int warmupSecs = 0;
  Device device = Device.CPU;
}

_PeerConfig _parseArgs(List<String> args) {
  final c = _PeerConfig();
  for (final raw in args) {
    final a = raw.startsWith('--') ? raw.substring(2) : raw;
    final eq = a.indexOf('=');
    if (eq < 0) continue;
    final k = a.substring(0, eq);
    final v = a.substring(eq + 1);
    switch (k) {
      case 'listen':
        c.listen = int.parse(v);
      case 'host':
        c.host = v;
      case 'peers':
        c.peers = v
            .split(',')
            .map((s) => s.trim())
            .where((s) => s.isNotEmpty)
            .toList();
      case 'id':
        c.id = v;
      case 'shard-id':
        c.shardId = int.parse(v);
      case 'num-shards':
        c.numShards = int.parse(v);
      case 'local-steps':
        c.localSteps = int.parse(v);
      case 'gossip-every':
        c.gossipEvery = int.parse(v);
      case 'rounds':
        c.rounds = int.parse(v);
      case 'arch':
        try {
          c.arch = parseArch(v);
        } on ArgumentError catch (e) {
          stderr.writeln('coop_peer: $e');
          exit(2);
        }
      case 'model':
        c.model = v;
      case 'corpus':
        c.corpus = v;
      case 'lr':
        c.lr = double.parse(v);
      case 'seed':
        c.seed = int.parse(v);
      case 'warmup-secs':
        c.warmupSecs = int.parse(v);
      case 'device':
        c.device = v.toLowerCase() == 'gpu' ? Device.GPU : Device.CPU;
      default:
        stderr.writeln('coop_peer: unknown flag --$k');
        exit(2);
    }
  }
  return c;
}

String _loadCorpus(String kind) {
  switch (kind) {
    case 'toy':
      return kToyCorpus;
    case 'shakespeare':
      final f = File('data/tiny_shakespeare.txt');
      if (!f.existsSync()) {
        stderr.writeln(
          'coop_peer: --corpus=shakespeare requires data/tiny_shakespeare.txt',
        );
        exit(1);
      }
      return f.readAsStringSync().substring(0, 40000);
    default:
      stderr.writeln('coop_peer: unknown --corpus=$kind');
      exit(2);
  }
}

Future<Uint8List> _readAllBody(HttpClientResponse resp) async {
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
  return out;
}

Future<void> main(List<String> args) async {
  final cfg = _parseArgs(args);
  cfg.numShards ??= (cfg.peers.length + 1);

  // Model + vocab.
  final text = _loadCorpus(cfg.corpus);
  final vocab = CharVocab.fromText(text);
  final lm = buildCoopLM(
    arch: cfg.arch,
    modelSize: cfg.model,
    vocabSize: vocab.size,
    device: cfg.device,
    seed: cfg.seed,
  );
  final ids = vocab.encode(text);
  final shard = shardSlice(ids, cfg.shardId, cfg.numShards!);
  final opt = Adam(lm.module.parameters(), lr: cfg.lr);
  final rng = math.Random(cfg.seed + cfg.id.hashCode);
  final gossipRng = math.Random(cfg.seed + cfg.listen);
  final blockSize = lm.maxLen - 1;

  // Shared mutable state: current local checkpoint bytes, updated
  // after each local-train round and after each gossip average.
  // Serves as the reply body for GET /checkpoint.
  var localBytes = Checkpoint.saveBytes(lm.module);
  var localRound = 0;

  // ---- HTTP server for peers to pull from us ----
  final server = await HttpServer.bind(cfg.host, cfg.listen);
  print(
    'coop_peer "${cfg.id}": '
    'listening on http://${cfg.host}:${cfg.listen}',
  );
  print(
    '  arch=${archToString(cfg.arch)} model=${cfg.model} '
    'corpus=${cfg.corpus} vocab=${vocab.size} '
    'shard=${cfg.shardId}/${cfg.numShards} tokens=${shard.length} '
    '(${lm.scalarCount} scalars)',
  );
  print('  peers: ${cfg.peers.isEmpty ? "(solo)" : cfg.peers.join(", ")}');

  // Serve endpoints in the background.
  unawaited(
    Future(() async {
      await for (final req in server) {
        try {
          switch ('${req.method} ${req.uri.path}') {
            case 'GET /health':
              req.response
                ..headers.contentType = ContentType.json
                ..write(
                  jsonEncode({
                    'id': cfg.id,
                    'arch': archToString(cfg.arch),
                    'model': cfg.model,
                    'round': localRound,
                    'peers': cfg.peers,
                    'vocabSize': vocab.size,
                  }),
                );
            case 'GET /checkpoint':
              req.response.headers
                ..contentType = ContentType.binary
                ..add('X-Coop-Round', '$localRound')
                ..add('X-Coop-Peer-Id', cfg.id)
                ..add('X-Coop-Arch', archToString(cfg.arch));
              req.response.add(localBytes);
            default:
              req.response
                ..statusCode = HttpStatus.notFound
                ..write('unknown route');
          }
        } catch (e) {
          req.response.statusCode = HttpStatus.internalServerError;
        } finally {
          await req.response.close();
        }
      }
    }),
  );

  final httpClient = HttpClient();

  // Give other peers a chance to bind their ports before we start
  // hammering them with gossip requests. Without this, on a single
  // box where Dart cold-start is slow, the first N rounds all get
  // "Connection refused" because the target peer's HttpServer.bind
  // hasn't returned yet.
  if (cfg.warmupSecs > 0) {
    print('  warming up for ${cfg.warmupSecs}s to let peers bind...');
    await Future<void>.delayed(Duration(seconds: cfg.warmupSecs));
  }

  // ---- Training + gossip loop ----
  var round = 0;
  while (cfg.rounds < 0 || round < cfg.rounds) {
    round += 1;
    // Yield to the event loop so any pending HTTP requests from peers
    // get serviced before we go back to CPU-bound training. Dart is
    // single-threaded, so if we never await, the server handler
    // starves and every peer's GET /checkpoint times out.
    await Future<void>.delayed(Duration.zero);
    // Local train.
    var lossAccum = 0.0;
    for (var s = 0; s < cfg.localSteps; s++) {
      opt.zeroGrad();
      final (x, y) = sampleWindow(shard, blockSize, rng, device: cfg.device);
      final loss = lm(x).crossEntropy(y).mean();
      loss.backward();
      clipGradNorm(lm.module.parameters(), 1.0);
      opt.step();
      lossAccum += loss.toList()[0];
      // Drain the event loop periodically so incoming GET /checkpoint
      // requests from peers can be serviced mid-round.
      if (s % 4 == 3) await Future<void>.delayed(Duration.zero);
    }
    final avgLoss = (lossAccum / cfg.localSteps).toStringAsFixed(4);

    // Publish new local weights so peers see the post-train version.
    localBytes = Checkpoint.saveBytes(lm.module);
    localRound = round;

    // Gossip: pull one random peer's checkpoint, average with ours,
    // adopt. Do nothing if solo or if this isn't a gossip round.
    var gossipedWith = '';
    if (cfg.peers.isNotEmpty && round % cfg.gossipEvery == 0) {
      final target = cfg.peers[gossipRng.nextInt(cfg.peers.length)];
      final url = Uri.parse('http://$target/checkpoint');
      try {
        final req = await httpClient
            .getUrl(url)
            .timeout(const Duration(seconds: 5));
        final resp = await req.close().timeout(const Duration(seconds: 30));
        if (resp.statusCode == 200) {
          final theirBytes = await _readAllBody(resp);
          final avg = averageCheckpoints([localBytes, theirBytes]);
          Checkpoint.loadIntoBytes(lm.module, avg);
          localBytes = avg;
          gossipedWith = target;
        } else {
          gossipedWith = '$target(http ${resp.statusCode})';
        }
      } catch (e) {
        gossipedWith = '$target(err: $e)';
      }
    }

    print(
      '  round=$round  local-loss=$avgLoss'
      '${gossipedWith.isEmpty ? "" : "  gossip<-$gossipedWith"}',
    );
  }

  print('coop_peer "${cfg.id}": done after $round rounds');
  httpClient.close(force: true);
  await server.close(force: true);
}

/// Fire-and-forget helper to avoid `unawaited_futures` lint on the
/// background server loop.
void unawaited(Future<void> f) {}
