/// Local multi-rank launcher — the single-host stand-in for `torchrun`.
/// Spawns N copies of `bin/ddp_train.dart`, one per rank, wired together via
/// env vars, and prefixes each line of output with its rank.
///
///   dart run bin/ddp_launch.dart 4
///
/// For GPU runs, each child would also get CUDA_VISIBLE_DEVICES=`<rank>` so its
/// default CUDA device is a distinct physical GPU. Across servers, skip this
/// and run bin/ddp_train.dart directly on each host with RANK/WORLD_SIZE/
/// MASTER_ADDR set (see ddp_train.dart header).
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';

Future<void> main(List<String> args) async {
  final world = args.isNotEmpty ? int.parse(args.first) : 2;
  const masterAddr = '127.0.0.1';
  final masterPort = Platform.environment['MASTER_PORT'] ?? '29500';
  final script = '${Directory.current.path}/bin/ddp_train.dart';

  stdout.writeln('launching $world ranks...');
  final procs = <Future<int>>[];
  for (var rank = 0; rank < world; rank++) {
    final env = {
      ...Platform.environment,
      'RANK': '$rank',
      'WORLD_SIZE': '$world',
      'LOCAL_RANK': '$rank',
      'MASTER_ADDR': masterAddr,
      'MASTER_PORT': masterPort,
      // For a real GPU run, uncomment to pin one card per rank:
      // 'CUDA_VISIBLE_DEVICES': '$rank',
    };
    final p = await Process.start('dart', ['run', script], environment: env);
    _pipe(p.stdout, rank, stdout);
    _pipe(p.stderr, rank, stderr);
    procs.add(p.exitCode);
  }

  final codes = await Future.wait(procs);
  final failed = codes.where((c) => c != 0).length;
  stdout.writeln(failed == 0
      ? 'all $world ranks exited cleanly'
      : '$failed/$world ranks failed');
  exitCode = failed == 0 ? 0 : 1;
}

void _pipe(Stream<List<int>> src, int rank, IOSink out) {
  src
      .transform(utf8.decoder)
      .transform(const LineSplitter())
      .listen((line) => out.writeln('[rank $rank] $line'));
}
