import 'dart:io';

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

void main() {
  group('ensureNativeLib', () {
    late Directory tmp;
    late String prevCwd;

    setUp(() async {
      tmp = await Directory.systemTemp.createTemp('nativelib_test_');
      prevCwd = Directory.current.path;
      Directory.current = tmp.path;
    });

    tearDown(() async {
      Directory.current = prevCwd;
      await tmp.delete(recursive: true);
    });

    test('downloads and installs prebuilt from a mock server', () async {
      final srv = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
      srv.listen((req) async {
        if (req.uri.path.endsWith(_expectedAsset())) {
          req.response.statusCode = 200;
          req.response.headers.contentType =
              ContentType('application', 'octet-stream');
          req.response.contentLength = _elfHeader.length;
          req.response.add(_elfHeader);
          await req.response.close();
        } else {
          req.response.statusCode = 404;
          await req.response.close();
        }
      });

      try {
        final path = await ensureNativeLib(
          urlBase: 'http://127.0.0.1:${srv.port}',
          version: 'test',
        );

        expect(File(path).existsSync(), isTrue);
        expect(path, contains('native/lib/'));
        expect(File(path).readAsBytesSync().sublist(0, 4), _elfHeader);
      } finally {
        await srv.close(force: true);
      }
    }, skip: Platform.isMacOS ? 'no CUDA on macOS' : null);
  });
}

final _elfHeader = List<int>.unmodifiable(<int>[0x7f, 0x45, 0x4c, 0x46]);

String _expectedAsset() {
  if (Platform.isWindows) return 'mat_mul-windows-x86_64.dll';
  final arch =
      Platform.version.toLowerCase().contains('arm64') ? 'arm64' : 'x86_64';
  return 'libmat_mul-linux-$arch.so';
}
