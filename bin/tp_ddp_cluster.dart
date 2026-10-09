/// Cluster launcher for 3D (data × tensor) parallelism — brings up one
/// rank per host with a per-host GPU set, over SSH, with one command.
///
/// Reads a hostfile (see scripts/tp_hostfile.example) where each line is:
///
///     host[:user@] [gpu-set]
///
/// e.g. `10.0.0.1 0,1,2,3`. Each host runs ONE data-parallel rank; within
/// that rank the model is tensor-parallel across the host's `gpu-set`
/// (passed as `TP_DEVICES`). Rank 0 (the first line) is the rendezvous
/// master by default.
///
///   dart run bin/tp_ddp_cluster.dart --hostfile scripts/tp_hostfile \
///     --workdir /opt/dart-pytorch --master-port 29500
///
/// Add --dry-run to print the exact ssh commands without executing them,
/// and --script to launch a different rank program (default the 3D demo).
///
/// Assumes passwordless SSH to each host and that the repo + Dart SDK are
/// present on every node at --workdir.
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';

Future<void> main(List<String> args) async {
  final opts = _parseArgs(args);
  final hostfile = opts['hostfile'];
  if (hostfile == null) {
    stderr.writeln('usage: tp_ddp_cluster.dart --hostfile PATH '
        '[--workdir DIR] [--master-addr HOST] [--master-port PORT] '
        '[--script bin/...dart] [--ssh "ssh -i key"] [--dry-run]');
    exitCode = 2;
    return;
  }

  final nodes = _parseHostfile(File(hostfile).readAsLinesSync());
  if (nodes.isEmpty) {
    stderr.writeln('no hosts found in $hostfile');
    exitCode = 2;
    return;
  }

  final workdir = opts['workdir'] ?? '.';
  final masterPort = opts['master-port'] ?? '29500';
  final masterAddr = opts['master-addr'] ?? _bareHost(nodes.first.host);
  final script = opts['script'] ?? 'bin/tensor_parallel_ddp_demo.dart';
  final sshCmd = (opts['ssh'] ?? 'ssh').split(' ');
  final dryRun = opts.containsKey('dry-run');
  final world = nodes.length;

  stdout.writeln('cluster: $world rank(s) (one per host), '
      'master=$masterAddr:$masterPort, workdir=$workdir');

  final running = <Future<int>>[];
  for (var rank = 0; rank < world; rank++) {
    final node = nodes[rank];
    final tpEnv = node.gpus.isEmpty ? '' : 'TP_DEVICES=${node.gpus} ';
    final remote = 'cd $workdir && '
        '${tpEnv}RANK=$rank WORLD_SIZE=$world LOCAL_RANK=0 '
        'MASTER_ADDR=$masterAddr MASTER_PORT=$masterPort '
        'dart run $script';
    final argv = [...sshCmd, node.host, remote];

    if (dryRun) {
      stdout.writeln('[rank $rank] ${argv.join(' ')}');
      continue;
    }

    final p = await Process.start(argv.first, argv.sublist(1));
    _pipe(p.stdout, rank, node.host, stdout);
    _pipe(p.stderr, rank, node.host, stderr);
    running.add(p.exitCode);
  }

  if (dryRun) return;

  final codes = await Future.wait(running);
  final failed = codes.where((c) => c != 0).length;
  stdout.writeln(failed == 0
      ? 'cluster: all $world rank(s) exited cleanly'
      : 'cluster: $failed/$world rank(s) failed');
  exitCode = failed == 0 ? 0 : 1;
}

class _Node {
  _Node(this.host, this.gpus);
  final String host;

  /// Comma-separated GPU ordinals for this rank's tensor-parallel set,
  /// e.g. "0,1". Empty means "use every visible GPU".
  final String gpus;
}

List<_Node> _parseHostfile(List<String> lines) {
  final out = <_Node>[];
  for (final raw in lines) {
    final line = raw.trim();
    if (line.isEmpty || line.startsWith('#')) continue;
    final parts = line.split(RegExp(r'\s+'));
    final gpus = parts.length > 1 ? parts[1] : '';
    out.add(_Node(parts.first, gpus));
  }
  return out;
}

/// Strips a leading `user@` so the master address is a bare host/IP.
String _bareHost(String host) {
  final at = host.indexOf('@');
  return at >= 0 ? host.substring(at + 1) : host;
}

void _pipe(Stream<List<int>> src, int rank, String host, IOSink out) {
  src
      .transform(utf8.decoder)
      .transform(const LineSplitter())
      .listen((line) => out.writeln('[rank $rank $host] $line'));
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
