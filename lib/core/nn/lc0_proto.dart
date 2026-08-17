/// Minimal LC0 weights.pb.gz reader.
///
/// LC0 stores its network as a gzipped protobuf ("weights.proto"
/// in the lczero-common repo). We don't want to depend on a full
/// protobuf runtime, so this file implements just enough of the
/// proto2 wire format to walk the `Net -> Weights` tree and extract
/// every named `Layer`, plus the LINEAR16 / FLOAT16 dequantization
/// that turns a Layer's raw bytes into a `Float32List`.
///
/// Currently supports the **classical CNN** flavour used by nets
/// like `744706.pb.gz`:
///
///   * `input` = 3x3 ConvBlock (Cin=112 planes, Cout=`filters`)
///   * `residual[N]` = two 3x3 ConvBlocks (no SE)
///   * `policy1` + `policy` = two-conv classical policy head
///   * `value` + fc + fc = classical value / WDL head
///   * `moves_left` + fc + fc = moves-left head (optional)
///
/// Attention-body / attention-policy nets (BT2, BT3, T78+, ...) share
/// the same proto schema but populate different fields; they are out
/// of scope for this reader.
library;

import 'dart:io';
import 'dart:typed_data';

import '../tensor/dtype.dart';
import '../tensor/tensor.dart';

class Lc0ConvBlock {
  final Tensor weights;
  final Tensor biases;
  final Tensor bnMeans;
  final Tensor bnStddivs;
  final Tensor bnGammas;
  final Tensor bnBetas;

  const Lc0ConvBlock({
    required this.weights,
    required this.biases,
    required this.bnMeans,
    required this.bnStddivs,
    required this.bnGammas,
    required this.bnBetas,
  });
}

class Lc0Residual {
  final Lc0ConvBlock conv1;
  final Lc0ConvBlock conv2;
  const Lc0Residual({required this.conv1, required this.conv2});
}

class Lc0Weights {
  final int filters;
  final int numBlocks;
  final int policyOutputPlanes;
  final int valueFilters;
  final int valueFCUnits;
  final int? movesLeftFilters;
  final int? movesLeftFCUnits;

  final Lc0ConvBlock input;
  final List<Lc0Residual> residual;

  final Lc0ConvBlock policy1;
  final Lc0ConvBlock policyOut;

  final Lc0ConvBlock valueConv;
  final Tensor ip1ValW;
  final Tensor ip1ValB;
  final Tensor ip2ValW;
  final Tensor ip2ValB;
  final int wdl;

  final Lc0ConvBlock? movesLeftConv;
  final Tensor? ip1MovW;
  final Tensor? ip1MovB;
  final Tensor? ip2MovW;
  final Tensor? ip2MovB;

  const Lc0Weights({
    required this.filters,
    required this.numBlocks,
    required this.policyOutputPlanes,
    required this.valueFilters,
    required this.valueFCUnits,
    required this.input,
    required this.residual,
    required this.policy1,
    required this.policyOut,
    required this.valueConv,
    required this.ip1ValW,
    required this.ip1ValB,
    required this.ip2ValW,
    required this.ip2ValB,
    required this.wdl,
    this.movesLeftFilters,
    this.movesLeftFCUnits,
    this.movesLeftConv,
    this.ip1MovW,
    this.ip1MovB,
    this.ip2MovW,
    this.ip2MovB,
  });
}

class _Field {
  final int number;
  final int wireType;
  final int start;
  final int end;
  const _Field(this.number, this.wireType, this.start, this.end);
}

class _RawLayer {
  double minVal = 0;
  double maxVal = 0;
  int encoding = 1;
  int paramsStart = 0;
  int paramsEnd = 0;
}

class Lc0Reader {
  static Lc0Weights readFile(String path) {
    final gz = File(path).readAsBytesSync();
    final raw = gzip.decode(gz);
    final bytes = Uint8List.fromList(raw);
    return _parseNet(bytes);
  }

  // ------- top-level parse -------

  static Lc0Weights _parseNet(Uint8List d) {
    // Walk the Net message and find field 10 = Weights.
    _Field? wField;
    var i = 0;
    while (i < d.length) {
      final f = _readField(d, i);
      i = f.end;
      if (f.number == 10) {
        wField = f;
      }
    }
    if (wField == null) {
      throw ArgumentError('lc0: no Weights (field 10) in Net message');
    }
    return _parseWeights(d, wField.start, wField.end);
  }

  static Lc0Weights _parseWeights(Uint8List d, int start, int end) {
    _Field? inputF;
    final residuals = <_Field>[];
    _Field? policyF, policy1F, valueF, movesLeftF;
    _Field? ip1ValW, ip1ValB, ip2ValW, ip2ValB;
    _Field? ip1MovW, ip1MovB, ip2MovW, ip2MovB;

    var i = start;
    while (i < end) {
      final f = _readField(d, i);
      i = f.end;
      switch (f.number) {
        case 1:
          inputF = f;
        case 2:
          residuals.add(f);
        case 3:
          policyF = f;
        case 6:
          valueF = f;
        case 7:
          ip1ValW = f;
        case 8:
          ip1ValB = f;
        case 9:
          ip2ValW = f;
        case 10:
          ip2ValB = f;
        case 11:
          policy1F = f;
        case 12:
          movesLeftF = f;
        case 13:
          ip1MovW = f;
        case 14:
          ip1MovB = f;
        case 15:
          ip2MovW = f;
        case 16:
          ip2MovB = f;
      }
    }

    if (inputF == null ||
        residuals.isEmpty ||
        policyF == null ||
        policy1F == null ||
        valueF == null ||
        ip1ValW == null ||
        ip1ValB == null ||
        ip2ValW == null ||
        ip2ValB == null) {
      throw ArgumentError(
        'lc0: missing required Weights fields (classical net expected)',
      );
    }

    // The input ConvBlock's `weights` layer has shape [F, 112, 3, 3].
    // We recover F (number of filters) from its param count / (112*3*3).
    final inputRaw = _rawConvWeights(d, inputF);
    final inputParamCount = _paramCount(inputRaw);
    final filters = inputParamCount ~/ (112 * 3 * 3);
    if (filters * 112 * 3 * 3 != inputParamCount) {
      throw ArgumentError(
        'lc0: input conv weights = $inputParamCount does not divide 112*3*3',
      );
    }

    final input = _parseConvBlock(d, inputF, [filters, 112, 3, 3]);
    final residual = <Lc0Residual>[
      for (final rf in residuals) _parseResidual(d, rf, filters),
    ];

    // Policy1 is [F, F, 3, 3]; final policy is [Pout, F, 3, 3] with
    // Pout usually 80 (5120 logits over the LC0 policy plane space).
    final policy1 = _parseConvBlock(d, policy1F, [filters, filters, 3, 3]);
    final policyOutRaw = _rawConvWeights(d, policyF);
    final pOutParams = _paramCount(policyOutRaw);
    final policyOutPlanes = pOutParams ~/ (filters * 3 * 3);
    final policyOut = _parseConvBlock(
      d,
      policyF,
      [policyOutPlanes, filters, 3, 3],
    );

    // Value ConvBlock is [Vf, F, 1, 1] (1x1 conv). Recover Vf from
    // params.
    final valueConvRaw = _rawConvWeights(d, valueF);
    final vParams = _paramCount(valueConvRaw);
    final valueFilters = vParams ~/ (filters * 1 * 1);
    final valueConv = _parseConvBlock(
      d,
      valueF,
      [valueFilters, filters, 1, 1],
    );

    // ip1_val_w is [FC, Vf*64]. Recover FC.
    final ip1ValWRaw = _readInnerLayer(d, ip1ValW);
    final ip1ValWParams = _paramCount(ip1ValWRaw);
    final valueFCUnits = ip1ValWParams ~/ (valueFilters * 64);
    final ip1ValWT = _dequantize(
      d,
      ip1ValWRaw,
      [valueFCUnits, valueFilters * 64],
    );
    final ip1ValBT = _dequantizeLayer(d, ip1ValB, [valueFCUnits]);

    // ip2_val_w is [WDL, FC]. WDL = 3 for WDL heads, 1 for scalar value.
    final ip2ValWRaw = _readInnerLayer(d, ip2ValW);
    final ip2ValWParams = _paramCount(ip2ValWRaw);
    final wdl = ip2ValWParams ~/ valueFCUnits;
    final ip2ValWT = _dequantize(d, ip2ValWRaw, [wdl, valueFCUnits]);
    final ip2ValBT = _dequantizeLayer(d, ip2ValB, [wdl]);

    // Moves-left head is optional.
    Lc0ConvBlock? movesLeftConv;
    Tensor? ip1MovWT, ip1MovBT, ip2MovWT, ip2MovBT;
    int? movesLeftFilters;
    int? movesLeftFCUnits;
    if (movesLeftF != null &&
        ip1MovW != null &&
        ip1MovB != null &&
        ip2MovW != null &&
        ip2MovB != null) {
      final mlRaw = _rawConvWeights(d, movesLeftF);
      final mlParams = _paramCount(mlRaw);
      movesLeftFilters = mlParams ~/ (filters * 1 * 1);
      movesLeftConv = _parseConvBlock(
        d,
        movesLeftF,
        [movesLeftFilters, filters, 1, 1],
      );
      final ip1MovWRaw = _readInnerLayer(d, ip1MovW);
      movesLeftFCUnits =
          _paramCount(ip1MovWRaw) ~/ (movesLeftFilters * 64);
      ip1MovWT = _dequantize(
        d,
        ip1MovWRaw,
        [movesLeftFCUnits, movesLeftFilters * 64],
      );
      ip1MovBT = _dequantizeLayer(d, ip1MovB, [movesLeftFCUnits]);
      final ip2MovWRaw = _readInnerLayer(d, ip2MovW);
      final mlOut = _paramCount(ip2MovWRaw) ~/ movesLeftFCUnits;
      ip2MovWT = _dequantize(d, ip2MovWRaw, [mlOut, movesLeftFCUnits]);
      ip2MovBT = _dequantizeLayer(d, ip2MovB, [mlOut]);
    }

    return Lc0Weights(
      filters: filters,
      numBlocks: residual.length,
      policyOutputPlanes: policyOutPlanes,
      valueFilters: valueFilters,
      valueFCUnits: valueFCUnits,
      input: input,
      residual: residual,
      policy1: policy1,
      policyOut: policyOut,
      valueConv: valueConv,
      ip1ValW: ip1ValWT,
      ip1ValB: ip1ValBT,
      ip2ValW: ip2ValWT,
      ip2ValB: ip2ValBT,
      wdl: wdl,
      movesLeftFilters: movesLeftFilters,
      movesLeftFCUnits: movesLeftFCUnits,
      movesLeftConv: movesLeftConv,
      ip1MovW: ip1MovWT,
      ip1MovB: ip1MovBT,
      ip2MovW: ip2MovWT,
      ip2MovB: ip2MovBT,
    );
  }

  static Lc0Residual _parseResidual(Uint8List d, _Field rf, int filters) {
    _Field? conv1F, conv2F;
    var i = rf.start;
    while (i < rf.end) {
      final f = _readField(d, i);
      i = f.end;
      if (f.number == 1) conv1F = f;
      if (f.number == 2) conv2F = f;
    }
    if (conv1F == null || conv2F == null) {
      throw ArgumentError('lc0: residual missing conv1 or conv2');
    }
    return Lc0Residual(
      conv1: _parseConvBlock(d, conv1F, [filters, filters, 3, 3]),
      conv2: _parseConvBlock(d, conv2F, [filters, filters, 3, 3]),
    );
  }

  static Lc0ConvBlock _parseConvBlock(
    Uint8List d,
    _Field cb,
    List<int> convShape,
  ) {
    final cOut = convShape[0];
    _Field? wF, bF, mF, sF, gF, bnF;
    var i = cb.start;
    while (i < cb.end) {
      final f = _readField(d, i);
      i = f.end;
      switch (f.number) {
        case 1:
          wF = f;
        case 2:
          bF = f;
        case 3:
          mF = f;
        case 4:
          sF = f;
        case 5:
          gF = f;
        case 6:
          bnF = f;
      }
    }
    if (wF == null) throw ArgumentError('lc0: ConvBlock missing weights');
    return Lc0ConvBlock(
      weights: _dequantizeLayer(d, wF, convShape),
      biases: bF != null
          ? _dequantizeLayer(d, bF, [cOut])
          : Tensor.fill([cOut], 0.0, device: Device.CPU),
      bnMeans: mF != null
          ? _dequantizeLayer(d, mF, [cOut])
          : Tensor.fill([cOut], 0.0, device: Device.CPU),
      bnStddivs: sF != null
          ? _dequantizeLayer(d, sF, [cOut])
          : Tensor.fill([cOut], 1.0, device: Device.CPU),
      bnGammas: gF != null
          ? _dequantizeLayer(d, gF, [cOut])
          : Tensor.fill([cOut], 1.0, device: Device.CPU),
      bnBetas: bnF != null
          ? _dequantizeLayer(d, bnF, [cOut])
          : Tensor.fill([cOut], 0.0, device: Device.CPU),
    );
  }

  // ------- Layer decoding -------

  // Reads the `weights` Layer (field 1) inside a ConvBlock message.
  static _RawLayer _rawConvWeights(Uint8List d, _Field cb) {
    _Field? wF;
    var i = cb.start;
    while (i < cb.end) {
      final f = _readField(d, i);
      i = f.end;
      if (f.number == 1) {
        wF = f;
        break;
      }
    }
    if (wF == null) throw ArgumentError('lc0: ConvBlock has no weights');
    return _readInnerLayer(d, wF);
  }

  static _RawLayer _readInnerLayer(Uint8List d, _Field lf) {
    final raw = _RawLayer();
    var i = lf.start;
    while (i < lf.end) {
      final f = _readField(d, i);
      i = f.end;
      switch (f.number) {
        case 1:
          raw.minVal = _asFloat32(d, f);
        case 2:
          raw.maxVal = _asFloat32(d, f);
        case 3:
          raw.paramsStart = f.start;
          raw.paramsEnd = f.end;
        case 4:
          raw.encoding = _asVarint(d, f);
      }
    }
    return raw;
  }

  static int _paramCount(_RawLayer r) {
    switch (r.encoding) {
      case 1: // LINEAR16
      case 2: // FLOAT16
      case 3: // BFLOAT16
        return (r.paramsEnd - r.paramsStart) ~/ 2;
    }
    throw ArgumentError('lc0: unsupported Layer encoding ${r.encoding}');
  }

  static Tensor _dequantizeLayer(Uint8List d, _Field lf, List<int> shape) {
    final raw = _readInnerLayer(d, lf);
    return _dequantize(d, raw, shape);
  }

  static Tensor _dequantize(Uint8List d, _RawLayer raw, List<int> shape) {
    final n = _paramCount(raw);
    var expected = 1;
    for (final v in shape) {
      expected *= v;
    }
    if (n != expected) {
      throw ArgumentError(
        'lc0: Layer param count $n does not match expected shape '
        '$shape ($expected)',
      );
    }
    final out = Float32List(n);
    final start = raw.paramsStart;
    final end = raw.paramsEnd;
    switch (raw.encoding) {
      case 1: // LINEAR16
        final range = raw.maxVal - raw.minVal;
        var j = 0;
        for (int p = start; p < end; p += 2) {
          final v = d[p] | (d[p + 1] << 8);
          out[j++] = raw.minVal + (v / 65535.0) * range;
        }
      case 2: // FLOAT16 (IEEE half)
        var j = 0;
        for (int p = start; p < end; p += 2) {
          final bits = d[p] | (d[p + 1] << 8);
          out[j++] = fp16BitsToFp32(bits);
        }
      case 3: // BFLOAT16
        var j = 0;
        for (int p = start; p < end; p += 2) {
          // bfloat16 is the top 16 bits of an fp32.
          final bits = (d[p] | (d[p + 1] << 8)) << 16;
          final buf = ByteData(4)..setUint32(0, bits, Endian.little);
          out[j++] = buf.getFloat32(0, Endian.little);
        }
    }
    return Tensor.fromList(shape, out, device: Device.CPU);
  }

  // ------- protobuf wire helpers -------

  static _Field _readField(Uint8List d, int i) {
    final (tag, iAfterTag) = _readVarintPair(d, i);
    final number = tag >> 3;
    final wire = tag & 7;
    switch (wire) {
      case 0: // varint
        final (_, end) = _readVarintPair(d, iAfterTag);
        return _Field(number, wire, iAfterTag, end);
      case 1: // i64
        return _Field(number, wire, iAfterTag, iAfterTag + 8);
      case 2: // length-delimited
        final (len, iLenEnd) = _readVarintPair(d, iAfterTag);
        return _Field(number, wire, iLenEnd, iLenEnd + len);
      case 5: // i32
        return _Field(number, wire, iAfterTag, iAfterTag + 4);
    }
    throw ArgumentError('lc0: unknown wire type $wire at offset $i');
  }

  static (int, int) _readVarintPair(Uint8List d, int i) {
    var v = 0;
    var shift = 0;
    while (true) {
      final b = d[i++];
      v |= (b & 0x7F) << shift;
      if ((b & 0x80) == 0) return (v, i);
      shift += 7;
    }
  }

  static double _asFloat32(Uint8List d, _Field f) {
    if (f.wireType != 5) {
      throw ArgumentError('lc0: expected fixed32 (float) at field ${f.number}');
    }
    final buf = ByteData.sublistView(d, f.start, f.end);
    return buf.getFloat32(0, Endian.little);
  }

  static int _asVarint(Uint8List d, _Field f) {
    if (f.wireType != 0) {
      throw ArgumentError('lc0: expected varint at field ${f.number}');
    }
    final (v, _) = _readVarintPair(d, f.start);
    return v;
  }
}
