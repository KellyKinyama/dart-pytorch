/// Reader for the Cactus `.cact` container (Needle 3 / Whistle weights).
///
/// The format is documented at https://cactuscompute.com/blog/cact-format
/// and is deliberately trivial to parse: a little-endian file in five
/// regions —
///
///   1. header     — 196 bytes, forty-nine 32-bit fields (geometry), the
///                   last field is `rope_theta` as a float32.
///   2. codebooks  — `codebookLen` float32s: the 4/8/16-entry Lloyd-Max
///                   codebooks for 2/3/4-bit weights, concatenated.
///   3. directory  — 44 bytes per tensor, positional (no names), in the
///                   order the forward pass reads them.
///   4. blobs      — each tensor's bytes at its absolute offset,
///                   64-byte aligned, matrices stored `[out, in]`.
///   5. tokenizer  — a RAW attachment: a SentencePiece BPE dump.
///
/// This is a *loader* only — it parses the header, codebooks, directory
/// and tokenizer region. Dequantisation of CQ blobs and the forward pass
/// live elsewhere.
library;

import 'dart:typed_data';

/// Magic tag in the first four bytes. `0x05E12A83` is Needle 2,
/// `0x05E12A84` is Needle 3.
const int kCactTagNeedle2 = 0x05E12A83;
const int kCactTagNeedle3 = 0x05E12A84;

/// Fixed header length in bytes (forty-nine 32-bit fields).
const int kCactHeaderBytes = 196;

/// Directory record size in bytes.
const int kCactDirRecordBytes = 44;

/// Blob storage dtype (directory `dtype` byte).
enum CactDType {
  fp16,
  fp32,
  cq, // Cactus Quant; width is in [CactTensor.bits]
  raw,
  unknown;

  static CactDType fromByte(int b) {
    switch (b) {
      case 1:
        return CactDType.fp16;
      case 2:
        return CactDType.fp32;
      case 3:
        return CactDType.cq;
      case 4:
        return CactDType.raw;
      default:
        return CactDType.unknown;
    }
  }

  String label(int bits) => this == CactDType.cq ? 'cq$bits' : name;
}

/// One directory record: `u8 dtype, u8 ndim, u16 pad, u32 shape[4],
/// u64 offset, u64 nbytes, u32 group, u32 bits`.
class CactTensor {
  final int index;
  final CactDType dtype;
  final int ndim;
  final List<int> shape; // length == ndim
  final int offset; // absolute byte offset of the blob
  final int nbytes;
  final int group; // quantisation group size (weights per group)
  final int bits; // CQ width (2/3/4), else 0/raw-defined

  const CactTensor({
    required this.index,
    required this.dtype,
    required this.ndim,
    required this.shape,
    required this.offset,
    required this.nbytes,
    required this.group,
    required this.bits,
  });

  String get dtypeLabel => dtype.label(bits);

  @override
  String toString() =>
      'T$index($dtypeLabel ${shape.join("x")} @$offset +$nbytes '
      'group=$group)';
}

/// Parsed header geometry. Only the fields documented with a known offset
/// are surfaced as named getters; [rawFields] holds all forty-nine 32-bit
/// values so nothing is lost for fields whose position is not yet pinned.
class CactHeader {
  /// All forty-nine 32-bit header fields, as read (little-endian).
  final List<int> rawFields;

  /// `rope_theta`, the final header field, reinterpreted as float32.
  final double ropeTheta;

  const CactHeader({required this.rawFields, required this.ropeTheta});

  int get formatTag => rawFields[0]; // offset 0
  int get numTensors => rawFields[1]; // offset 4
  int get codebookLen => rawFields[2]; // offset 8
  int get kvWindow => rawFields[3]; // offset 12
  int get kvBits => rawFields[4]; // offset 16
  int get vocab => rawFields[5]; // offset 20
  int get outVocab => rawFields[6]; // offset 24
  int get dModel => rawFields[7]; // offset 28
  int get numHeads => rawFields[8]; // offset 32
  int get numKvHeads => rawFields[9]; // offset 36
  int get numLayers => rawFields[10]; // offset 40
  int get qkHeadDim => rawFields[11]; // offset 44
  int get vHeadDim => rawFields[12]; // offset 48
  int get maxSeqLen => rawFields[13]; // offset 52
  int get hadaN => rawFields[14]; // offset 56
  int get mhcLanes => rawFields[15]; // offset 60
  int get slidingWindow => rawFields[16]; // offset 64
  int get globalMaskLo => rawFields[17]; // offset 68
  int get globalMaskHi => rawFields[18]; // offset 72
  int get qkvConvTaps => rawFields[19]; // offset 76
  int get engramSlots => rawFields[20]; // offset 80
  int get engramSubDim => rawFields[21]; // offset 84
  int get numEngramTables => rawFields[22]; // offset 88
  int get engramConvTaps => rawFields[23]; // offset 92
  int get engramConvDilation => rawFields[24]; // offset 96
  int get engramSeedHeads => rawFields[25]; // offset 100
  int get numEngramOrders => rawFields[26]; // offset 104
  List<int> get engramOrders =>
      [for (var i = 0; i < numEngramOrders; i++) rawFields[27 + i]];
  int get numEngramSites => rawFields[31]; // offset 124
  List<int> get engramSites =>
      [for (var i = 0; i < numEngramSites; i++) rawFields[32 + i]];

  bool get isNeedle3 => formatTag == kCactTagNeedle3;
  bool get isNeedle2 => formatTag == kCactTagNeedle2;
}

/// A parsed `.cact` file: header, codebooks, tensor directory, and the
/// byte span of the trailing tokenizer (RAW) attachment.
class CactFile {
  final CactHeader header;

  /// The concatenated Lloyd-Max codebook floats (length == codebookLen).
  final Float32List codebooks;

  /// Positional tensor directory, in forward-pass order.
  final List<CactTensor> directory;

  /// The raw file bytes, if available (needed to read tensor blobs).
  final Uint8List? bytes;

  const CactFile({
    required this.header,
    required this.codebooks,
    required this.directory,
    this.bytes,
  });

  /// The raw blob bytes for [t] (requires [bytes] to be present).
  Uint8List blob(CactTensor t) {
    final b = bytes;
    if (b == null) {
      throw StateError('CactFile has no bytes; cannot read tensor blobs');
    }
    return Uint8List.sublistView(b, t.offset, t.offset + t.nbytes);
  }

  /// Total bytes summed per dtype label (e.g. `cq2`, `cq4`, `fp16`, `raw`).
  Map<String, int> bytesByKind() {
    final m = <String, int>{};
    for (final t in directory) {
      m.update(t.dtypeLabel, (v) => v + t.nbytes, ifAbsent: () => t.nbytes);
    }
    return m;
  }

  /// Count of tensors per dtype label.
  Map<String, int> countByKind() {
    final m = <String, int>{};
    for (final t in directory) {
      m.update(t.dtypeLabel, (v) => v + 1, ifAbsent: () => 1);
    }
    return m;
  }

  /// The RAW tokenizer attachment is (by the format contract) the last
  /// RAW tensor; returns its blob span or `null` if absent.
  CactTensor? get tokenizerBlob {
    for (var i = directory.length - 1; i >= 0; i--) {
      if (directory[i].dtype == CactDType.raw) return directory[i];
    }
    return null;
  }

  /// Parse a `.cact` file from its raw bytes. Throws [FormatException] if
  /// the magic tag is not a recognised Cactus archive.
  static CactFile parse(Uint8List bytes) {
    if (bytes.length < kCactHeaderBytes) {
      throw const FormatException('file too small to be a .cact archive');
    }
    final view = ByteData.view(
      bytes.buffer,
      bytes.offsetInBytes,
      bytes.length,
    );

    final tag = view.getUint32(0, Endian.little);
    if (tag != kCactTagNeedle3 && tag != kCactTagNeedle2) {
      throw FormatException(
        'not a Cactus .cact archive: tag 0x${tag.toRadixString(16)}',
      );
    }

    // 49 little-endian uint32 header fields; the last is a float32.
    final raw = List<int>.generate(
      kCactHeaderBytes ~/ 4,
      (i) => view.getUint32(i * 4, Endian.little),
    );
    final ropeTheta = view.getFloat32(kCactHeaderBytes - 4, Endian.little);
    final header = CactHeader(rawFields: raw, ropeTheta: ropeTheta);

    final codebookLen = header.codebookLen;
    final cbBase = kCactHeaderBytes;
    final codebooks = Float32List(codebookLen);
    for (var i = 0; i < codebookLen; i++) {
      codebooks[i] = view.getFloat32(cbBase + i * 4, Endian.little);
    }

    var off = cbBase + codebookLen * 4;
    final n = header.numTensors;
    final dir = <CactTensor>[];
    for (var i = 0; i < n; i++) {
      final dtype = CactDType.fromByte(view.getUint8(off));
      final ndim = view.getUint8(off + 1);
      final shape = <int>[
        for (var s = 0; s < ndim; s++)
          view.getUint32(off + 4 + s * 4, Endian.little),
      ];
      final offset = view.getUint64(off + 20, Endian.little);
      final nbytes = view.getUint64(off + 28, Endian.little);
      final group = view.getUint32(off + 36, Endian.little);
      final bits = view.getUint32(off + 40, Endian.little);
      dir.add(CactTensor(
        index: i,
        dtype: dtype,
        ndim: ndim,
        shape: shape,
        offset: offset,
        nbytes: nbytes,
        group: group,
        bits: bits,
      ));
      off += kCactDirRecordBytes;
    }

    return CactFile(
        header: header, codebooks: codebooks, directory: dir, bytes: bytes);
  }
}
