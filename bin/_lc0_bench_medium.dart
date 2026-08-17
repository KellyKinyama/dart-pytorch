/// Try loading and benchmarking a "medium" LC0 network on GPU.
///
/// Usage:
///   dart run bin/_lc0_bench_medium.dart                           # default net
///   dart run bin/_lc0_bench_medium.dart --net URL --path PATH     # custom
///   dart run bin/_lc0_bench_medium.dart --batches 1,4,8,16,32
///
/// Downloads the network to [path] if missing, prints inferred
/// dimensions and estimated VRAM, then runs a forward pass at each
/// requested batch size. On WSL2 also runs `nvidia-smi` before /
/// after so peak VRAM can be eyeballed.
library;

import 'dart:io';

import 'package:dart_pytorch/dart_pytorch.dart';

const _defaultNetUrl =
    'https://storage.lczero.org/files/networks-contrib/t1-256x10-distilled-swa-2432500.pb.gz';
const _defaultNetPath = 'models/lc0/t1-256x10-distilled.pb.gz';

Future<void> main(List<String> args) async {
  final netUrl = _strArg(args, '--net') ?? _defaultNetUrl;
  final netPath = _strArg(args, '--path') ?? _defaultNetPath;
  final batches = (_strArg(args, '--batches') ?? '1,4,8,16,32')
      .split(',')
      .map(int.parse)
      .toList();

  // Reuse the LC0 weights downloader from lc0_dart? No — this bin is
  // in dart_pytorch, so inline the same tiny fetch.
  if (!File(netPath).existsSync()) {
    await _downloadPbGz(netUrl, netPath);
  }

  stdout.writeln('loading $netPath ...');
  final sw = Stopwatch()..start();
  final w = Lc0Reader.readFile(netPath);
  final loadMs = sw.elapsedMilliseconds;

  stdout.writeln('  filters       = ${w.filters}');
  stdout.writeln('  residual blks = ${w.numBlocks}');
  stdout.writeln('  policy planes = ${w.policyOutputPlanes}');
  stdout.writeln('  value filters = ${w.valueFilters}');
  stdout.writeln('  value FC      = ${w.valueFCUnits}');
  stdout.writeln('  wdl heads     = ${w.wdl}');
  stdout.writeln('  loaded in ${loadMs} ms');

  _printMemoryEstimate(w, batches);
  await _printNvidiaSmi('BEFORE net construction');

  sw.reset();
  final net = Lc0Net(w, device: Device.GPU);
  stdout.writeln('\nnet on GPU: ${sw.elapsedMilliseconds} ms');
  await _printNvidiaSmi('AFTER net construction');

  final startInput = Lc0Input.fromFen(startFen);
  // Warmup.
  net(startInput);

  for (final b in batches) {
    final inputs = List<Tensor>.generate(b, (_) => Lc0Input.fromFen(startFen));
    // Median of 3 to smooth first-run jitter.
    final samples = <int>[];
    for (int i = 0; i < 3; i++) {
      final swB = Stopwatch()..start();
      net.callBatch(inputs);
      samples.add(swB.elapsedMilliseconds);
    }
    samples.sort();
    final median = samples[1];
    stdout.writeln(
      'B=${b.toString().padLeft(2)}  '
      '${median.toString().padLeft(5)} ms   '
      '${(median / b).toStringAsFixed(2).padLeft(6)} ms/pos   '
      '(samples: ${samples.join(", ")})',
    );
  }

  await _printNvidiaSmi('AFTER forward passes');
}

void _printMemoryEstimate(Lc0Weights w, List<int> batches) {
  // Weights: 40*C^2*36 bytes for the res tower (approximating).
  //   input conv:  112 * C * 3*3
  //   res tower:   2 * numBlocks * C * C * 3*3
  //   policy:      C * C * 3*3   +   C * policyPlanes * 3*3
  //   value:       C * valueFilters * 1*1 + valueFilters*64*valueFCUnits +
  //                valueFCUnits*wdl
  final c = w.filters;
  final blocks = w.numBlocks;
  final weightBytes =
      (112 * c * 9 +
          2 * blocks * c * c * 9 +
          c * c * 9 +
          c * w.policyOutputPlanes * 9 +
          c * w.valueFilters +
          w.valueFilters * 64 * w.valueFCUnits +
          w.valueFCUnits * w.wdl) *
      4;

  stdout.writeln('\nestimated VRAM:');
  stdout.writeln('  weights (fp32) : ${_mb(weightBytes)}');

  for (final b in batches) {
    // Peak activation ≈ im2col scratch of the biggest conv layer.
    // im2col of one conv: [b*64, C*k*k] fp32.
    final im2col = b * 64 * c * 9 * 4;
    // Two live per-block tensors plus skip: 3 * [b*64, C] fp32 ≈ minor.
    final acts = 3 * b * 64 * c * 4;
    stdout.writeln(
      '  B=${b.toString().padLeft(2)} scratch      : '
      '${_mb(im2col + acts)}   '
      '(im2col ${_mb(im2col)}, acts ${_mb(acts)})',
    );
  }
}

String _mb(int bytes) => '${(bytes / 1024 / 1024).toStringAsFixed(1)} MB';

String? _strArg(List<String> args, String flag) {
  for (int i = 0; i < args.length - 1; i++) {
    if (args[i] == flag) return args[i + 1];
  }
  return null;
}

Future<void> _downloadPbGz(String url, String path) async {
  stderr.writeln('downloading $url');
  await File(path).parent.create(recursive: true);
  final client = HttpClient()..userAgent = 'dart_pytorch/bench';
  try {
    var current = Uri.parse(url);
    HttpClientResponse resp;
    for (var hop = 0; ; hop++) {
      final req = await client.getUrl(current);
      req.followRedirects = false;
      resp = await req.close();
      if (resp.isRedirect && hop < 5) {
        final loc = resp.headers.value(HttpHeaders.locationHeader);
        await resp.drain<void>();
        if (loc == null) throw StateError('redirect w/o Location');
        current = current.resolve(loc);
        continue;
      }
      break;
    }
    if (resp.statusCode != 200) {
      throw HttpException('HTTP ${resp.statusCode}', uri: current);
    }
    final tmp = File('$path.part');
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
          stderr.write('\r  ${_mb(got)} / ${_mb(total)}  $pct%');
          lastPct = pct;
        }
      }
    }
    await sink.flush();
    await sink.close();
    stderr.writeln('\r  ${_mb(got)}                    ');
    await tmp.rename(path);
  } finally {
    client.close(force: true);
  }
}

Future<void> _printNvidiaSmi(String tag) async {
  try {
    final r = await Process.run('nvidia-smi', [
      '--query-gpu=memory.used,memory.free',
      '--format=csv,noheader,nounits',
    ]);
    if (r.exitCode == 0) {
      final line = (r.stdout as String).trim();
      stdout.writeln('  nvidia-smi ($tag): used/free = $line MiB');
    }
  } catch (_) {
    // nvidia-smi not on PATH; ignore.
  }
}
