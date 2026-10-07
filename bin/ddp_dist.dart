/// Minimal, pure-Dart distributed layer — the piece `dart_pytorch` lacks.
///
/// Ports the essential mechanic of PyTorch DDP (as used by nanoGPT's
/// `train.py`): every rank computes gradients on its own data shard, then the
/// gradients are **averaged across all ranks** before the optimizer step, so
/// all replicas stay identical. PyTorch does this with NCCL; here we do it with
/// plain TCP sockets so it runs on CPU or GPU, on one box or across servers.
///
/// Topology: rank 0 is the master (binds a port); every other rank connects to
/// it. `allReduceMean` is a reduce-to-master + broadcast (O(world) per step) —
/// simple and correct, ideal for learning. Swap this for NCCL FFI bindings
/// later for GPU-direct speed; the training loop would not change.
///
/// Launched like `torchrun` via env vars: RANK, WORLD_SIZE, LOCAL_RANK,
/// MASTER_ADDR, MASTER_PORT.
library;

import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

class Dist {
  Dist._(this.rank, this.worldSize, this.localRank, this._workers, this._master);

  final int rank;
  final int worldSize;
  final int localRank;

  /// Master (rank 0) only: one connection per worker, indexed by rank
  /// (index 0 is null). Workers leave this empty.
  final List<_Conn?> _workers;

  /// Workers only: the single connection to the master.
  final _Conn? _master;

  bool get isMaster => rank == 0;

  static Future<Dist> init() async {
    final env = Platform.environment;
    final rank = int.parse(env['RANK'] ?? '0');
    final worldSize = int.parse(env['WORLD_SIZE'] ?? '1');
    final localRank = int.parse(env['LOCAL_RANK'] ?? '0');
    final addr = env['MASTER_ADDR'] ?? '127.0.0.1';
    final port = int.parse(env['MASTER_PORT'] ?? '29500');

    if (worldSize <= 1) {
      return Dist._(rank, worldSize, localRank, const [], null);
    }

    if (rank == 0) {
      final server =
          await ServerSocket.bind(InternetAddress.anyIPv4, port, shared: false);
      final conns = List<_Conn?>.filled(worldSize, null);
      final needed = worldSize - 1;
      final allConnected = Completer<void>();
      var got = 0;
      final sub = server.listen((sock) {
        sock.setOption(SocketOption.tcpNoDelay, true);
        final c = _Conn(sock);
        // Each worker's first 4 bytes are its rank.
        c.readExactly(4).then((b) {
          final r = ByteData.sublistView(b).getInt32(0, Endian.little);
          conns[r] = c;
          if (++got == needed && !allConnected.isCompleted) {
            allConnected.complete();
          }
        });
      });
      // Fail with a useful message if not everyone shows up in time.
      final initTimeoutMs =
          int.parse(env['DDP_INIT_TIMEOUT_MS'] ?? '120000');
      try {
        await allConnected.future
            .timeout(Duration(milliseconds: initTimeoutMs));
      } on TimeoutException {
        await sub.cancel();
        await server.close();
        throw StateError('rendezvous timed out: only $got/$needed worker(s) '
            'connected to master :$port within ${initTimeoutMs}ms');
      }
      await sub.cancel();
      await server.close();
      return Dist._(rank, worldSize, localRank, conns, null);
    }

    // Worker: connect to master (retry while it comes up).
    final connectTimeoutMs =
        int.parse(env['DDP_CONNECT_TIMEOUT_MS'] ?? '60000');
    final deadline = DateTime.now().add(Duration(milliseconds: connectTimeoutMs));
    Socket? sock;
    Object? lastErr;
    while (sock == null && DateTime.now().isBefore(deadline)) {
      try {
        sock = await Socket.connect(addr, port);
      } catch (e) {
        lastErr = e;
        await Future<void>.delayed(const Duration(milliseconds: 200));
      }
    }
    if (sock == null) {
      throw StateError('rank $rank could not reach master at $addr:$port '
          'within ${connectTimeoutMs}ms (last error: $lastErr)');
    }
    sock.setOption(SocketOption.tcpNoDelay, true);
    final c = _Conn(sock);
    final hdr = ByteData(4)..setInt32(0, rank, Endian.little);
    sock.add(hdr.buffer.asUint8List());
    return Dist._(rank, worldSize, localRank, const [], c);
  }

  /// Averages [buf] in place across all ranks (the DDP gradient all-reduce).
  Future<void> allReduceMean(Float32List buf) async {
    if (worldSize <= 1) return;
    if (isMaster) {
      for (var r = 1; r < worldSize; r++) {
        final bytes = await _workers[r]!.recvFrame();
        final incoming = bytes.buffer.asFloat32List(bytes.offsetInBytes, buf.length);
        for (var i = 0; i < buf.length; i++) {
          buf[i] += incoming[i];
        }
      }
      final inv = 1.0 / worldSize;
      for (var i = 0; i < buf.length; i++) {
        buf[i] *= inv;
      }
      final out = buf.buffer.asUint8List(buf.offsetInBytes, buf.lengthInBytes);
      for (var r = 1; r < worldSize; r++) {
        _workers[r]!.sendFrame(out);
      }
    } else {
      final out = buf.buffer.asUint8List(buf.offsetInBytes, buf.lengthInBytes);
      _master!.sendFrame(out);
      final bytes = await _master.recvFrame();
      final reduced = bytes.buffer.asFloat32List(bytes.offsetInBytes, buf.length);
      buf.setAll(0, reduced);
    }
  }

  /// Copies rank 0's [buf] to every other rank (used to sync initial weights).
  Future<void> broadcastFromMaster(Float32List buf) async {
    if (worldSize <= 1) return;
    if (isMaster) {
      final out = buf.buffer.asUint8List(buf.offsetInBytes, buf.lengthInBytes);
      for (var r = 1; r < worldSize; r++) {
        _workers[r]!.sendFrame(out);
      }
    } else {
      final bytes = await _master!.recvFrame();
      final src = bytes.buffer.asFloat32List(bytes.offsetInBytes, buf.length);
      buf.setAll(0, src);
    }
  }

  /// Blocks until every rank reaches this point (e.g. an epoch boundary or
  /// around checkpointing). Implemented as gather-at-master + release.
  Future<void> barrier() async {
    if (worldSize <= 1) return;
    final token = Uint8List(1);
    if (isMaster) {
      for (var r = 1; r < worldSize; r++) {
        await _workers[r]!.recvFrame();
      }
      for (var r = 1; r < worldSize; r++) {
        _workers[r]!.sendFrame(token);
      }
    } else {
      _master!.sendFrame(token);
      await _master.recvFrame();
    }
  }

  Future<void> close() async {
    for (final c in _workers) {
      await c?.close();
    }
    await _master?.close();
  }
}

/// A framed connection: length-prefixed byte payloads over one socket.
class _Conn {
  _Conn(this.socket) : _reader = _ByteReader(socket);

  final Socket socket;
  final _ByteReader _reader;

  /// Copies [payload] into a stable framed buffer before queueing it. The copy
  /// is essential: callers pass a view over a reused float buffer that may be
  /// mutated before the socket actually flushes.
  void sendFrame(Uint8List payload) {
    final frame = Uint8List(4 + payload.length);
    ByteData.sublistView(frame).setUint32(0, payload.length, Endian.little);
    frame.setRange(4, 4 + payload.length, payload);
    socket.add(frame);
  }

  Future<Uint8List> recvFrame() async {
    final header = await _reader.readExactly(4);
    final n = ByteData.sublistView(header).getUint32(0, Endian.little);
    return _reader.readExactly(n);
  }

  Future<Uint8List> readExactly(int n) => _reader.readExactly(n);

  Future<void> close() async {
    try {
      await socket.close();
    } catch (_) {}
  }
}

/// Pulls exact byte counts out of a socket's chunked stream.
class _ByteReader {
  _ByteReader(Stream<Uint8List> stream) {
    stream.listen(
      _onData,
      onDone: () => _waiter?.completeError(const SocketException('closed')),
      onError: (Object e) => _waiter?.completeError(e),
      cancelOnError: true,
    );
  }

  final BytesBuilder _buf = BytesBuilder(copy: false);
  Uint8List _pending = Uint8List(0);
  int _offset = 0;
  Completer<Uint8List>? _waiter;
  int _want = 0;

  void _onData(Uint8List chunk) {
    _buf.add(chunk);
    _tryComplete();
  }

  Future<Uint8List> readExactly(int n) {
    assert(_waiter == null, 'overlapping reads are not supported');
    final completer = Completer<Uint8List>();
    _want = n;
    _waiter = completer;
    _tryComplete();
    return completer.future;
  }

  int get _available => (_pending.length - _offset) + _buf.length;

  void _tryComplete() {
    final w = _waiter;
    if (w == null || _available < _want) return;
    // Consolidate the leftover + newly-buffered bytes, then slice.
    if (_buf.length > 0) {
      final merged = Uint8List(_pending.length - _offset + _buf.length)
        ..setAll(0, _pending.sublist(_offset))
        ..setAll(_pending.length - _offset, _buf.takeBytes());
      _pending = merged;
      _offset = 0;
    }
    final out = Uint8List.sublistView(_pending, _offset, _offset + _want);
    _offset += _want;
    _waiter = null;
    w.complete(out);
  }
}
