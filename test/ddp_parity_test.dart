@Timeout(Duration(minutes: 3))
library;

import 'dart:io';

import 'package:test/test.dart';

/// Integration parity test: launch a multi-rank DDP run over loopback and
/// assert every rank ends with the *same* parameter checksum — i.e. the
/// gradient all-reduce kept the replicas in sync.
void main() {
  test('DDP ranks stay in sync (identical param checksums)', () async {
    const world = 2;
    final result = await Process.run(
      'dart',
      ['run', 'bin/ddp_launch.dart', '$world'],
      environment: {...Platform.environment, 'MASTER_PORT': '29711'},
      workingDirectory: Directory.current.path,
    );

    expect(result.exitCode, 0, reason: 'launch failed:\n${result.stderr}');

    final out = result.stdout.toString();
    final checksums = RegExp(r'param_checksum=([-0-9.]+)')
        .allMatches(out)
        .map((m) => m.group(1)!)
        .toList();

    expect(checksums.length, world,
        reason: 'expected $world checksums, got ${checksums.length}:\n$out');
    expect(checksums.toSet().length, 1,
        reason: 'ranks diverged: $checksums\n$out');
  });
}
