/// Auto-fetch the prebuilt native CUDA library
/// (`libmat_mul.so` / `mat_mul.dll` / `libmat_mul.dylib`).
///
/// Users normally don't call this — [engine] does it implicitly the
/// first time a GPU op runs. Call it manually from a top-level
/// `await` if you want to control download timing (or bail on an
/// unsupported platform) before the first tensor allocation.
library;

import 'dart:async';
import 'dart:io' as io;

/// Package version this build was published as. Used to construct
/// the GitHub Releases URL. **Bump this together with `pubspec.yaml`
/// on every release** so users pull binaries built from the same
/// source.
const kDartPytorchVersion = '0.1.1';

/// Prefix of the GitHub Releases URL the auto-download uses. Set
/// `DART_PYTORCH_NATIVE_URL` at runtime to point elsewhere (mirror,
/// LAN cache, air-gapped setup).
const kDefaultReleaseUrlBase =
    'https://github.com/KellyKinyama/dart-pytorch/releases/download';

/// Returns the absolute path of the native library on this host,
/// downloading a prebuilt from the [kDartPytorchVersion] GitHub
/// release into `<cwd>/native/lib/` if it is missing.
///
/// Set `DART_PYTORCH_AUTO_DOWNLOAD=0` to disable the network fetch —
/// callers that hit that branch must supply the lib themselves
/// (build from source, or set `DART_PYTORCH_NATIVE_LIB=<path>`).
///
/// Throws on unsupported platforms (macOS: no CUDA runtime exists
/// today) and on any HTTP / integrity failure.
Future<String> ensureNativeLib({
  String? version,
  String? urlBase,
  bool prompt = false,
  io.IOSink? log,
}) async {
  final err = log ?? io.stderr;
  final envOverride = io.Platform.environment['DART_PYTORCH_NATIVE_LIB'];
  if (envOverride != null && envOverride.isNotEmpty) {
    if (!io.File(envOverride).existsSync()) {
      throw StateError(
        'DART_PYTORCH_NATIVE_LIB=$envOverride does not exist',
      );
    }
    return envOverride;
  }

  final assetName = _releaseAssetName();
  final localName = _localLibName();
  final localPath =
      '${io.Directory.current.path}/native/lib/$localName';
  if (io.File(localPath).existsSync()) return localPath;

  // Also honor an existing lib in the executable directory (i.e.
  // shipped with a `dart compile exe` bundle) — no download needed.
  try {
    final exeDir = io.File(io.Platform.resolvedExecutable).parent.path;
    for (final candidate in [
      '$exeDir/native/lib/$localName',
      '$exeDir/$localName',
    ]) {
      if (io.File(candidate).existsSync()) return candidate;
    }
  } catch (_) {}

  if (io.Platform.environment['DART_PYTORCH_AUTO_DOWNLOAD'] == '0') {
    throw StateError(
      'dart_pytorch: native lib $localName missing at $localPath '
      'and DART_PYTORCH_AUTO_DOWNLOAD=0. Build it with '
      'scripts/build_native.sh or set DART_PYTORCH_NATIVE_LIB.',
    );
  }

  final v = version ?? kDartPytorchVersion;
  final base = urlBase ??
      io.Platform.environment['DART_PYTORCH_NATIVE_URL'] ??
      kDefaultReleaseUrlBase;
  final url = '$base/v$v/$assetName';

  err.writeln('dart_pytorch: native lib $localName missing');
  err.writeln('dart_pytorch: source     $url');
  if (prompt && io.stdin.hasTerminal) {
    err.write('dart_pytorch: download now? [Y/n] ');
    final line = io.stdin.readLineSync()?.trim().toLowerCase() ?? 'y';
    if (line.isNotEmpty && line != 'y' && line != 'yes') {
      throw StateError('user declined download');
    }
  }

  await io.Directory('${io.Directory.current.path}/native/lib')
      .create(recursive: true);
  final client = io.HttpClient()..userAgent = 'dart_pytorch/$v';
  try {
    var current = Uri.parse(url);
    io.HttpClientResponse resp;
    // GitHub Releases URLs redirect to a CDN; follow up to 5 hops.
    for (var hop = 0; ; hop++) {
      final req = await client.getUrl(current);
      req.followRedirects = false;
      resp = await req.close();
      if (resp.isRedirect && hop < 5) {
        final loc = resp.headers.value(io.HttpHeaders.locationHeader);
        await resp.drain<void>();
        if (loc == null) {
          throw io.HttpException('redirect without Location', uri: current);
        }
        current = current.resolve(loc);
        continue;
      }
      break;
    }
    if (resp.statusCode != 200) {
      throw io.HttpException(
        'HTTP ${resp.statusCode} fetching $current',
        uri: current,
      );
    }
    final total = resp.contentLength;
    var got = 0;
    var lastPct = -1;
    final tmp = io.File('$localPath.part');
    final sink = tmp.openWrite();
    await for (final chunk in resp) {
      sink.add(chunk);
      got += chunk.length;
      if (total > 0) {
        final pct = (got * 100 / total).floor();
        if (pct != lastPct && pct % 5 == 0) {
          err.write(
            '\rdart_pytorch: downloading ${_mb(got)} / ${_mb(total)}  $pct%',
          );
          lastPct = pct;
        }
      }
    }
    await sink.flush();
    await sink.close();
    err.writeln('\rdart_pytorch: downloaded ${_mb(got)}                    ');

    // Sanity-check ELF (0x7f 45 4c 46), PE ('MZ'), or Mach-O magic
    // so we don't rename an HTML "404" page into the native slot.
    final head = await tmp.openRead(0, 4).expand((b) => b).toList();
    final ok = head.length >= 2 &&
        ((head[0] == 0x7f && head[1] == 0x45 && head[2] == 0x4c &&
                head[3] == 0x46) ||
            (head[0] == 0x4d && head[1] == 0x5a) ||
            (head[0] == 0xcf && head[1] == 0xfa) ||
            (head[0] == 0xce && head[1] == 0xfa) ||
            (head[0] == 0xca && head[1] == 0xfe));
    if (!ok) {
      await tmp.delete();
      throw StateError(
        'downloaded file is not a native library '
        '(magic 0x${head.map((b) => b.toRadixString(16).padLeft(2, "0")).join()})',
      );
    }
    await tmp.rename(localPath);
  } finally {
    client.close(force: true);
  }
  return localPath;
}

String _releaseAssetName() {
  final arch = _hostArch();
  if (io.Platform.isWindows) return 'mat_mul-windows-$arch.dll';
  if (io.Platform.isMacOS) {
    throw StateError(
      'dart_pytorch: no CUDA binary is published for macOS (CUDA has '
      'been unsupported by NVIDIA on macOS since 2019). Use '
      '`Device.CPU` for tensor work.',
    );
  }
  return 'libmat_mul-linux-$arch.so';
}

String _localLibName() {
  if (io.Platform.isWindows) return 'mat_mul.dll';
  if (io.Platform.isMacOS) return 'libmat_mul.dylib';
  return 'libmat_mul.so';
}

String _hostArch() {
  final v = io.Platform.version.toLowerCase();
  if (v.contains('arm64') || v.contains('aarch64')) return 'arm64';
  return 'x86_64';
}

String _mb(int bytes) => '${(bytes / 1024 / 1024).toStringAsFixed(1)} MB';
