/// Per-node launcher — the multi-machine equivalent of `torchrun`.
///
/// Each node runs this once with its own `--node-rank`; it spawns
/// `--nproc-per-node` local ranks, computing the **global** rank as
/// `node_rank * nproc_per_node + local_rank` and `WORLD_SIZE = nnodes *
/// nproc_per_node`. All ranks rendezvous at `--master-addr:--master-port`
/// (which must be rank 0's host, i.e. node 0).
///
/// Two machines, 2 procs each (world_size 4):
///   # on node 0 (the master host, IP 10.0.0.1):
///   dart run bin/ddp_run.dart --nnodes 2 --node-rank 0 --nproc-per-node 2 \
///     --master-addr 10.0.0.1 --master-port 29500
///   # on node 1:
///   dart run bin/ddp_run.dart --nnodes 2 --node-rank 1 --nproc-per-node 2 \
///     --master-addr 10.0.0.1 --master-port 29500
///
/// Single host is just `--nnodes 1` (or use bin/ddp_launch.dart).
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';

Future<void> main(List<String> args) async {
  final opts = _parseArgs(args);
  final nnodes = _int(opts, 'nnodes', 1);
  final nproc = _int(opts, 'nproc-per-node', 1);
  final nodeRank = _int(opts, 'node-rank', 0);
  final masterAddr = opts['master-addr'] ?? '127.0.0.1';
  final masterPort = opts['master-port'] ?? '29500';
  final script =
      opts['script'] ?? '${Directory.current.path}/bin/ddp_train.dart';
  final worldSize = nnodes * nproc;

  if (nodeRank < 0 || nodeRank >= nnodes) {
    stderr.writeln('node-rank must be in [0, $nnodes)');
    exitCode = 2;
    return;
  }

  stdout.writeln('node $nodeRank/$nnodes: launching $nproc rank(s) '
      '(world_size=$worldSize, master=$masterAddr:$masterPort)');

  final procs = <Future<int>>[];
  for (var local = 0; local < nproc; local++) {
    final rank = nodeRank * nproc + local;
    final env = {
      ...Platform.environment,
      'RANK': '$rank',
      'WORLD_SIZE': '$worldSize',
      'LOCAL_RANK': '$local',
      'MASTER_ADDR': masterAddr,
      'MASTER_PORT': masterPort,
      // For a GPU run, pin one card per local rank:
      // 'CUDA_VISIBLE_DEVICES': '$local',
    };
    final p = await Process.start('dart', ['run', script], environment: env);
    _pipe(p.stdout, rank, stdout);
    _pipe(p.stderr, rank, stderr);
    procs.add(p.exitCode);
  }

  final codes = await Future.wait(procs);
  final failed = codes.where((c) => c != 0).length;
  stdout.writeln(failed == 0
      ? 'node $nodeRank: all $nproc rank(s) exited cleanly'
      : 'node $nodeRank: $failed/$nproc rank(s) failed');
  exitCode = failed == 0 ? 0 : 1;
}

void _pipe(Stream<List<int>> src, int rank, IOSink out) {
  src
      .transform(utf8.decoder)
      .transform(const LineSplitter())
      .listen((line) => out.writeln('[rank $rank] $line'));
}

Map<String, String> _parseArgs(List<String> args) {
  final out = <String, String>{};
  for (var i = 0; i < args.length; i++) {
    if (args[i].startsWith('--')) {
      final key = args[i].substring(2);
      final val = (i + 1 < args.length && !args[i + 1].startsWith('--'))
          ? args[++i]
          : 'true';
      out[key] = val;
    }
  }
  return out;
}

int _int(Map<String, String> opts, String key, int fallback) =>
    opts[key] != null ? int.parse(opts[key]!) : fallback;
