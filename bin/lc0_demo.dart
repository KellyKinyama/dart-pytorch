/// End-to-end LC0 network inference demo.
///
/// Loads the classical `744706.pb.gz` net (128 filters, 10 residual
/// blocks) from `models/lc0/`, runs the forward pass on a supplied
/// FEN position, and prints:
///   * The WDL head (win / draw / loss probability from the side-
///     to-move POV) and a "score" = p_win - p_loss.
///   * The top 20 policy slots by raw logit + softmax probability.
///     The (plane, rank, file) index is the raw network output;
///     mapping it to a UCI move ("e2e4", ...) needs LC0's 1858-move
///     index table which is out of scope for this demo — the raw
///     signal is enough to prove the pipeline works end-to-end.
///
/// Prerequisites (one-time weights download, ~6 MB):
///
///   mkdir -p models/lc0 && cd models/lc0
///   curl -sSL -O \
///     "https://storage.lczero.org/files/networks-contrib/744706.pb.gz"
///
/// Usage:
///
///   dart run bin/lc0_demo.dart
///   dart run bin/lc0_demo.dart \
///     'rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3'
library;

import 'dart:io';
import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

const _weightsPath = 'models/lc0/744706.pb.gz';
const _weightsUrl =
    'https://storage.lczero.org/files/networks-contrib/744706.pb.gz';

Future<void> main(List<String> args) async {
  // Parse --cpu / --gpu / -y flags and remove them from the FEN args.
  final device = args.contains('--cpu') ? Device.CPU : Device.GPU;
  final noPrompt = args.contains('-y') || args.contains('--yes');
  final fenArgs = args
      .where((a) => a != '--cpu' && a != '--gpu' && a != '-y' && a != '--yes')
      .toList();
  final fen = fenArgs.isNotEmpty ? fenArgs.join(' ') : startFen;

  try {
    await _ensureWeights(prompt: !noPrompt);
  } catch (e) {
    stderr.writeln('lc0_demo: $e');
    exit(64);
  }

  stdout.writeln(
    'Loading LC0 network from $_weightsPath on ${device.name.toUpperCase()} ...',
  );
  final sw = Stopwatch()..start();
  final w = Lc0Reader.readFile(_weightsPath);
  final net = Lc0Net(w, device: device);
  stdout.writeln(
    '  filters=${w.filters}, blocks=${w.numBlocks}, '
    'policy planes=${w.policyOutputPlanes}, '
    'value FC=${w.valueFCUnits}, wdl=${w.wdl}   '
    '(load: ${sw.elapsedMilliseconds} ms)',
  );

  stdout.writeln('\nFEN: $fen');
  final input = Lc0Input.fromFen(fen);

  sw.reset();
  final out = net(input);
  stdout.writeln('\nForward pass: ${sw.elapsedMilliseconds} ms');

  // ---- value ----
  final v = out.value.toList();
  if (w.wdl == 3) {
    stdout.writeln(
      '\nValue (WDL, STM POV):\n'
      '  P(win)  = ${v[0].toStringAsFixed(3)}\n'
      '  P(draw) = ${v[1].toStringAsFixed(3)}\n'
      '  P(loss) = ${v[2].toStringAsFixed(3)}\n'
      '  score   = ${(v[0] - v[2]).toStringAsFixed(3)}  '
      '(p_win - p_loss)',
    );
  } else {
    stdout.writeln('\nValue (scalar, STM POV): ${v[0].toStringAsFixed(3)}');
  }

  // ---- policy ----
  final rawPolicy = out.policyLogits.toList();
  final n = rawPolicy.length;
  final idxs = List<int>.generate(n, (i) => i);
  idxs.sort((a, b) => rawPolicy[b].compareTo(rawPolicy[a]));

  final maxL = rawPolicy[idxs[0]];
  var norm = 0.0;
  for (int i = 0; i < n; i++) {
    norm += math.exp(rawPolicy[i] - maxL);
  }

  stdout.writeln('\nTop 20 policy slots (raw logits + softmax prob):');
  stdout.writeln(
    '  ${'plane'.padLeft(5)} ${'sq'.padLeft(3)}   '
    '${'logit'.padLeft(8)}   ${'prob'.padLeft(7)}',
  );
  for (int i = 0; i < 20; i++) {
    final idx = idxs[i];
    final plane = idx ~/ 64;
    final rank = (idx % 64) ~/ 8;
    final file = idx % 8;
    final logit = rawPolicy[idx];
    final prob = math.exp(logit - maxL) / norm;
    final sqName = String.fromCharCode(97 + file) + (rank + 1).toString();
    stdout.writeln(
      '  ${plane.toString().padLeft(5)} $sqName   '
      '${logit.toStringAsFixed(3).padLeft(8)}   '
      '${(prob * 100).toStringAsFixed(2).padLeft(6)}%',
    );
  }
}

Future<void> _ensureWeights({required bool prompt}) async {
  final file = File(_weightsPath);
  if (await file.exists()) return;

  stderr.writeln('lc0: weights not found at $_weightsPath');
  stderr.writeln('lc0: source     $_weightsUrl');
  if (prompt && stdin.hasTerminal) {
    stderr.write('lc0: download now? [Y/n] ');
    final line = stdin.readLineSync()?.trim().toLowerCase() ?? 'y';
    if (line.isNotEmpty && line != 'y' && line != 'yes') {
      throw StateError('user declined download');
    }
  }
  await file.parent.create(recursive: true);
  final client = HttpClient()..userAgent = 'dart_pytorch/lc0_demo';
  try {
    final req = await client.getUrl(Uri.parse(_weightsUrl));
    final resp = await req.close();
    if (resp.statusCode != 200) {
      throw HttpException(
        'HTTP ${resp.statusCode} fetching $_weightsUrl',
        uri: Uri.parse(_weightsUrl),
      );
    }
    final tmp = File('$_weightsPath.part');
    final sink = tmp.openWrite();
    final total = resp.contentLength;
    var got = 0;
    var lastPct = -1;
    await for (final chunk in resp) {
      sink.add(chunk);
      got += chunk.length;
      if (total > 0) {
        final pct = (got * 100 / total).floor();
        if (pct != lastPct && pct % 5 == 0) {
          stderr.write('\rlc0: downloading ${_mb(got)} / ${_mb(total)}  $pct%');
          lastPct = pct;
        }
      }
    }
    await sink.flush();
    await sink.close();
    stderr.writeln('\rlc0: downloaded ${_mb(got)}                    ');
    final head = await tmp.openRead(0, 2).expand((b) => b).toList();
    if (head.length < 2 || head[0] != 0x1f || head[1] != 0x8b) {
      await tmp.delete();
      throw StateError('downloaded file is not a gzip stream');
    }
    await tmp.rename(_weightsPath);
  } finally {
    client.close(force: true);
  }
}

String _mb(int bytes) => '${(bytes / 1024 / 1024).toStringAsFixed(1)} MB';
