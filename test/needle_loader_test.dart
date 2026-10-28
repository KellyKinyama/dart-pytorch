import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/core/nn/needle/cact_format.dart';
import 'package:dart_pytorch/core/nn/needle/cact_quant.dart';
import 'package:dart_pytorch/core/nn/needle/needle_loader.dart';
import 'package:test/test.dart';

// --- tiny .cact writer (d=256, 2 layers, 2 lanes, no engram) -----------------

const int _d = 256;
const int _vocab = 64;
const int _layers = 2;
const int _lanes = 4; // matches NeedleConfig default numLanes
const int _nh = 12, _qk = 48, _nkv = 2, _vd = 64, _taps = 3;
const int _hn = 256; // hadamard size
const int _nd = _lanes * _d; // 1024

late math.Random _rng;
Float64List _rand(int n, [double s = 0.15]) =>
    Float64List.fromList(List.generate(n, (_) => (_rng.nextDouble() * 2 - 1) * s));

Float32List _codebooks() {
  final cb = Float32List(28);
  for (var i = 0; i < 4; i++) {
    cb[i] = (i - 1.5) / 6.0; // CQ2
  }
  for (var i = 0; i < 16; i++) {
    cb[12 + i] = (i - 7.5) / 32.0; // CQ4
  }
  return cb;
}

Uint8List _encF16(Float64List v) {
  final b = Uint8List(v.length * 2);
  for (var i = 0; i < v.length; i++) {
    final h = doubleToFp16(v[i]);
    b[i * 2] = h & 0xff;
    b[i * 2 + 1] = (h >> 8) & 0xff;
  }
  return b;
}

Uint8List _encF32(Float64List v) {
  final bd = ByteData(v.length * 4);
  for (var i = 0; i < v.length; i++) {
    bd.setFloat32(i * 4, v[i], Endian.little);
  }
  return bd.buffer.asUint8List();
}

Uint8List _encCq(Float64List v, int outDim, int inDim, int bits, Float32List cb) {
  return quantizeMatrix(v, outDim, inDim, bits, codebookForBits(cb, bits));
}

class _Builder {
  final List<CactTensor> recs = [];
  final BytesBuilder bb = BytesBuilder();
  int off = 0;
  final Float32List cb = _codebooks();

  void _align() {
    while (off % 64 != 0) {
      bb.addByte(0);
      off++;
    }
  }

  void _add(CactDType dt, List<int> shape, Uint8List blob, int bits) {
    _align();
    recs.add(CactTensor(
      index: recs.length,
      dtype: dt,
      ndim: shape.length,
      shape: shape,
      offset: off,
      nbytes: blob.length,
      group: 128,
      bits: bits,
    ));
    bb.add(blob);
    off += blob.length;
  }

  void f16(List<int> shape, Float64List v) =>
      _add(CactDType.fp16, shape, _encF16(v), 0);
  void f32(List<int> shape, Float64List v) =>
      _add(CactDType.fp32, shape, _encF32(v), 0);
  void cq(List<int> shape, int bits, Float64List v) {
    final inDim = shape.last;
    final outDim = shape.sublist(0, shape.length - 1).fold<int>(1, (a, b) => a * b);
    _add(CactDType.cq, shape, _encCq(v, outDim, inDim, bits, cb), bits);
  }

  void raw(List<int> shape, Uint8List v) => _add(CactDType.raw, shape, v, 0);
}

(CactFile, Float64List) _buildTinyCact() {
  _rng = math.Random(42);
  final b = _Builder();

  b.cq([_vocab, _d], 4, _rand(_vocab * _d, 1.0)); // embedding

  for (var l = 0; l < _layers; l++) {
    b.f16([_d], _rand(_d));
    b.cq([_nh * _qk, _d], 2, _rand(_nh * _qk * _d));
    b.cq([_nkv * _qk, _d], 2, _rand(_nkv * _qk * _d));
    b.cq([_nkv * _vd, _d], 2, _rand(_nkv * _vd * _d));
    b.f16([_taps, _nh * _qk], _rand(_taps * _nh * _qk));
    b.f16([_taps, _nkv * _qk], _rand(_taps * _nkv * _qk));
    b.f16([_taps, _nkv * _vd], _rand(_taps * _nkv * _vd));
    b.f16([_qk], _rand(_qk));
    b.f16([_qk], _rand(_qk));
    b.cq([_nh * _vd, _d], 2, _rand(_nh * _vd * _d));
    b.cq([_d, _nh * _vd], 2, _rand(_d * _nh * _vd));
    b.f16([_d], _rand(_d));
    b.f16([1], _rand(1));
    b.f16([_d], _rand(_d));
    for (var k = 0; k < 5; k++) {
      b.f16([_hn], _rand(_hn, k == 0 || k == 2 || k == 3 ? 1.0 : 0.3));
    }
    for (var k = 0; k < 6; k++) {
      b.f16([16, 16], _rand(256));
    }
    b.f16([_d, 8], _rand(_d * 8));
    b.f16([8, _hn], _rand(8 * _hn));
  }

  // mHC
  b.f16([_layers], _rand(_layers, 0.01));
  b.f16([_layers], _rand(_layers, 0.01));
  b.f16([_layers], _rand(_layers, 0.01));
  b.f16([_layers, _lanes], _rand(_layers * _lanes));
  b.f16([_layers, _lanes], _rand(_layers * _lanes));
  b.f16([_layers, _lanes, _lanes], _rand(_layers * _lanes * _lanes));
  b.cq([_layers * _lanes, _nd], 4, _rand(_layers * _lanes * _nd, 0.1));
  b.cq([_layers * _lanes, _nd], 4, _rand(_layers * _lanes * _nd, 0.1));
  b.cq([_layers * _lanes * _lanes, _nd], 4,
      _rand(_layers * _lanes * _lanes * _nd, 0.1));

  // Monarch shuffles (identity for the test)
  b.f32([_hn], Float64List.fromList(List.generate(_hn, (i) => i.toDouble())));
  b.f32([_hn], Float64List.fromList(List.generate(_hn, (i) => i.toDouble())));

  // final norm
  final finalNorm = _rand(_d);
  b.f16([_d], finalNorm);

  // heads: manifest (1 head) + 6 confidence-head tensors (unused by loader)
  b.f16([1], Float64List.fromList([2.0]));
  b.cq([4, _d], 4, _rand(4 * _d));
  b.f16([4], _rand(4));
  b.cq([4, _d], 4, _rand(4 * _d));
  b.f16([4], _rand(4));
  b.cq([_d, _d], 4, _rand(_d * _d));
  b.f16([_d], _rand(_d));

  b.raw([64], Uint8List(64)); // tokenizer stub

  final raw = List<int>.filled(49, 0);
  raw[0] = kCactTagNeedle3;
  raw[1] = b.recs.length;
  raw[2] = 28;
  raw[4] = 8;
  raw[5] = _vocab;
  raw[7] = _d;
  raw[10] = _layers;
  final header = CactHeader(rawFields: raw, ropeTheta: 100000.0);
  final file = CactFile(
    header: header,
    codebooks: b.cb,
    directory: b.recs,
    bytes: b.bb.toBytes(),
  );
  return (file, finalNorm);
}

void main() {
  group('loadNeedleModel from a tiny .cact', () {
    test('loads, dequantises and runs a finite forward', () {
      final (file, finalNorm) = _buildTinyCact();
      final model = loadNeedleModel(file);
      expect(model.config.dModel, _d);
      expect(model.blocks.length, _layers);
      expect(model.engram, isEmpty); // no sites below 2 layers
      expect(model.embedding.length, _vocab * _d);

      final logits = model.logits([1, 2, 3, 4]);
      expect(logits.length, 4 * _vocab);
      expect(logits.every((v) => v.isFinite), isTrue);
    });

    test('fp16 tensors round-trip through the loader', () {
      final (file, finalNorm) = _buildTinyCact();
      final model = loadNeedleModel(file);
      expect(model.finalNorm.length, _d);
      for (var i = 0; i < _d; i++) {
        // fp16 precision: relative tolerance ~2^-10
        expect(model.finalNorm[i], closeTo(finalNorm[i], 1e-2),
            reason: 'finalNorm[$i]');
      }
    });
  });
}
