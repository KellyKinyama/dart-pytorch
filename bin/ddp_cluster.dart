/// Cluster launcher — brings up a whole multi-node CPU run with one command.
///
/// Reads a hostfile (see scripts/hostfile.example), then SSHes into each node
/// and starts `bin/ddp_run.dart` there with the right `--node-rank`. Node 0
/// (the first hostfile line) is the rendezvous master by default.
///
///   dart run bin/ddp_cluster.dart --hostfile scripts/hostfile \
///     --workdir /opt/dart-pytorch --master-port 29500
///
/// Add --dry-run to print the exact ssh commands without executing them.
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
    stderr.writeln('usage: ddp_cluster.dart --hostfile PATH '
        '[--workdir DIR] [--master-addr HOST] [--master-port PORT] '
        '[--ssh "ssh -i key"] [--dry-run]');
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
  final sshCmd = (opts['ssh'] ?? 'ssh').split(' ');
  final dryRun = opts.containsKey('dry-run');
  final nnodes = nodes.length;

  stdout.writeln('cluster: $nnodes node(s), master=$masterAddr:$masterPort, '
      'workdir=$workdir');

  final running = <Future<int>>[];
  for (var nodeRank = 0; nodeRank < nnodes; nodeRank++) {
    final node = nodes[nodeRank];
    final remote = 'cd $workdir && '
        'dart run bin/ddp_run.dart '
        '--nnodes $nnodes --node-rank $nodeRank '
        '--nproc-per-node ${node.slots} '
        '--master-addr $masterAddr --master-port $masterPort';
    final argv = [...sshCmd, node.host, remote];

    if (dryRun) {
      stdout.writeln('[node $nodeRank] ${argv.join(' ')}');
      continue;
    }

    final p = await Process.start(argv.first, argv.sublist(1));
    _pipe(p.stdout, nodeRank, node.host, stdout);
    _pipe(p.stderr, nodeRank, node.host, stderr);
    running.add(p.exitCode);
  }

  if (dryRun) return;

  final codes = await Future.wait(running);
  final failed = codes.where((c) => c != 0).length;
  stdout.writeln(failed == 0
      ? 'cluster: all $nnodes node(s) exited cleanly'
      : 'cluster: $failed/$nnodes node(s) failed');
  exitCode = failed == 0 ? 0 : 1;
}

class _Node {
  _Node(this.host, this.slots);
  final String host;
  final int slots;
}

List<_Node> _parseHostfile(List<String> lines) {
  final out = <_Node>[];
  for (final raw in lines) {
    final line = raw.trim();
    if (line.isEmpty || line.startsWith('#')) continue;
    final parts = line.split(RegExp(r'\s+'));
    final slots = parts.length > 1 ? int.tryParse(parts[1]) ?? 1 : 1;
    out.add(_Node(parts.first, slots));
  }
  return out;
}

/// Strips a leading `user@` so the master address is a bare host/IP.
String _bareHost(String host) {
  final at = host.indexOf('@');
  return at >= 0 ? host.substring(at + 1) : host;
}

void _pipe(Stream<List<int>> src, int node, String host, IOSink out) {
  src
      .transform(utf8.decoder)
      .transform(const LineSplitter())
      .listen((line) => out.writeln('[node $node $host] $line'));
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
