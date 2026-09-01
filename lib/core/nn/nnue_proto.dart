/// Minimal Stockfish `.nnue` binary reader.
///
/// `.nnue` files are little-endian raw blobs. Every Stockfish NNUE net
/// starts with a 12-byte fixed header followed by a variable-length
/// description string; downstream come architecture-specific sections
/// (feature transformer + network) each prefixed by their own hash so
/// mismatches can be diagnosed before we start decoding weights as if
/// they were the wrong architecture.
///
/// This first pass targets the **HalfKAv2_hm / SFNNv8** family used
/// by Stockfish 16's default net (version tag `0x7AF32F18`). Later
/// architectures (SFNNv9 dual-net in SF 17+) reuse the same outer
/// framing but populate different section hashes and shapes; adding
/// them is a matter of extending [NnueArchitecture] and the section
/// readers below.
///
/// Reference: `Stockfish/src/nnue/*` and the `nnue-pytorch` decoder
/// (`serialize.py`). This reader is intentionally header-only: it
/// discovers what's in the file and pulls out raw byte spans, but
/// leaves quantised → float32 conversion and shape unpacking to
/// `nnue.dart` (kept separate so we can also feed the raw bytes to a
/// future int-quantised backend without re-parsing).
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

/// NNUE architecture families this reader knows about.
///
/// The value stored in the file's leading `u32 version` word maps to
/// one of these; unknown values are rejected early with a descriptive
/// error rather than being silently misinterpreted as a supported
/// arch.
enum NnueArchitecture {
  /// `0x7AF32F20` — HalfKAv2_hm, single-net SFNNv8. Stockfish 15/16.
  halfKAv2Hm,
}

const int _kVersionHalfKAv2Hm = 0x7AF32F20;

NnueArchitecture? _archFromVersion(int v) {
  switch (v) {
    case _kVersionHalfKAv2Hm:
      return NnueArchitecture.halfKAv2Hm;
    default:
      return null;
  }
}

/// The 12-byte fixed prefix common to every `.nnue` file, plus the
/// variable-length UTF-8 description that follows.
class NnueHeader {
  const NnueHeader({
    required this.version,
    required this.architecture,
    required this.hashValue,
    required this.description,
    required this.byteLength,
  });

  /// Raw `u32` version tag from the file (little-endian).
  final int version;

  /// Decoded architecture family, guaranteed non-null after
  /// [NnueReader.parseHeader] succeeds.
  final NnueArchitecture architecture;

  /// `u32` architecture / net hash. This is *not* the file's SHA;
  /// it's a compile-time constant baked into each Stockfish build
  /// that identifies the exact layer sizes expected by the engine.
  /// A mismatch here means the file was produced by an incompatible
  /// architecture even if its version tag matches.
  final int hashValue;

  /// Human-readable description string that trainers embed in nets
  /// (usually "Features=HalfKAv2_hm(...); Network=(...)"). Useful for
  /// logging; not consumed by the runtime.
  final String description;

  /// Total number of bytes consumed by version + hash + desc_len +
  /// desc. Section readers pick up from this offset.
  final int byteLength;

  @override
  String toString() =>
      'NnueHeader(version=0x${version.toRadixString(16)}, '
      'arch=$architecture, hashValue=0x${hashValue.toRadixString(16)}, '
      'descLen=${description.length}, headerBytes=$byteLength)';
}

/// Parsed feature-transformer section. Weights + biases are returned
/// as raw `Uint8List` byte spans (little-endian int16 for both) so
/// the eventual int backend can consume them without a float
/// round-trip; `nnue.dart` handles dequantisation for the float
/// backend.
class NnueFeatureTransformerBlob {
  const NnueFeatureTransformerBlob({
    required this.sectionHash,
    required this.biasesI16,
    required this.weightsI16,
    required this.psqtI32,
    required this.ftDim,
    required this.numInputs,
    required this.psqtBuckets,
    required this.byteEnd,
  });

  /// Per-section hash from the file. Currently informational; a full
  /// implementation would cross-check against a table of known hashes.
  final int sectionHash;

  /// FT bias vector, one int16 per FT dim. Length `ftDim`.
  final Int16List biasesI16;

  /// FT weight matrix, row-major `[numInputs, ftDim]` int16.
  final Int16List weightsI16;

  /// PSQT weight matrix, row-major `[numInputs, psqtBuckets]` int32.
  final Int32List psqtI32;

  /// Per-POV FT dimensionality (1024 for SFNNv8).
  final int ftDim;

  /// Number of sparse input features (41024 for HalfKAv2_hm).
  final int numInputs;

  /// PSQT bucket count (8 for SFNNv8).
  final int psqtBuckets;

  /// Absolute byte offset in the source file just past the last byte
  /// of this section; used by the network reader to pick up where FT
  /// decode left off.
  final int byteEnd;
}

/// One of the eight material-count buckets in Stockfish's layered
/// stack. Each bucket owns three fully-connected quantised layers:
///
///   L1: `[ftDim = 1536]` → `[16]` (int8 weights, int32 biases)
///   L2: `[32]`           → `[32]` (int8, int32)
///   L3: `[32]`           → `[1]`  (int8, int32)
///
/// The `[]` numbers above match the on-disk padding (Stockfish rounds
/// most dimensions up to a multiple of 16 or 32 so its SIMD kernels
/// can consume them without a shape check). Interpretation of "used
/// dim vs padded dim" is the runtime's problem — this class just
/// hands out the raw ints exactly as they appear in the file.
///
/// The trailing `bucketTail` field is one extra `int32` that appears
/// after L3 in every bucket and does not fit any obvious L1/L2/L3
/// slot. It's most likely a per-bucket cp offset ("psqt residual");
/// stored verbatim so downstream code can experiment.
class NnueNetworkBucket {
  const NnueNetworkBucket({
    required this.bucketHash,
    required this.l1BiasesI32,
    required this.l1WeightsI8,
    required this.l2BiasesI32,
    required this.l2WeightsI8,
    required this.l3BiasesI32,
    required this.l3WeightsI8,
  });

  /// Per-bucket architecture hash from Stockfish's
  /// `Network::hash_value()`. Currently informational — a full
  /// implementation would cross-check it against a table.
  final int bucketHash;

  final Int32List l1BiasesI32;
  final Int8List l1WeightsI8;
  final Int32List l2BiasesI32;
  final Int8List l2WeightsI8;
  final Int32List l3BiasesI32;
  final Int8List l3WeightsI8;
}

/// Parsed network-body section: `numBuckets` per-material-count stacks.
///
/// Note there is no outer network-section hash — Stockfish's
/// `LayerStacks::read_parameters` just iterates the buckets, each of
/// which prefixes its own `u32` hash (see [NnueNetworkBucket.bucketHash]).
class NnueNetworkBlob {
  const NnueNetworkBlob({required this.buckets, required this.byteEnd});

  final List<NnueNetworkBucket> buckets;
  final int byteEnd;
}

/// Everything a downstream module needs to run inference. Populated
/// section-by-section as the reader walks the file.
class NnueRaw {
  NnueRaw({
    required this.header,
    required this.featureTransformer,
    required this.network,
  });

  final NnueHeader header;
  final NnueFeatureTransformerBlob featureTransformer;
  final NnueNetworkBlob network;
}

class NnueReader {
  /// Load and fully parse a `.nnue` file from disk. Throws
  /// [FormatException] with a descriptive message if the version tag
  /// is unknown or a section is truncated.
  static NnueRaw loadFile(String path) {
    final bytes = File(path).readAsBytesSync();
    return parse(bytes);
  }

  /// Parse an in-memory `.nnue` byte buffer.
  static NnueRaw parse(Uint8List bytes) {
    final r = _ByteReader(bytes);
    final header = _parseHeader(r);
    final ft = _parseFeatureTransformerHalfKAv2Hm(r, header);
    final net = _parseNetworkHalfKAv2Hm(r, ft);
    return NnueRaw(header: header, featureTransformer: ft, network: net);
  }

  /// Header-only parse — cheap, does not touch weight sections. Useful
  /// for `bin/_nnue_inspect.dart`-style tooling.
  static NnueHeader parseHeader(Uint8List bytes) {
    return _parseHeader(_ByteReader(bytes));
  }
}

NnueHeader _parseHeader(_ByteReader r) {
  if (r.remaining < 12) {
    throw const FormatException('nnue: file shorter than 12-byte header');
  }
  final version = r.readU32();
  final hashValue = r.readU32();
  final descLen = r.readU32();
  if (descLen > 1 << 20) {
    throw FormatException(
      'nnue: implausible description length $descLen '
      '(possible endian or version mismatch, version=0x'
      '${version.toRadixString(16)})',
    );
  }
  if (r.remaining < descLen) {
    throw FormatException(
      'nnue: description truncated (need $descLen bytes, have ${r.remaining})',
    );
  }
  final descBytes = r.readBytes(descLen);
  final description = utf8.decode(descBytes, allowMalformed: true);
  final arch = _archFromVersion(version);
  if (arch == null) {
    throw FormatException(
      'nnue: unrecognised version tag 0x${version.toRadixString(16)} '
      '(supported: 0x${_kVersionHalfKAv2Hm.toRadixString(16)} '
      'HalfKAv2_hm / SFNNv8). File may be a newer format such as '
      'SFNNv9; extend NnueArchitecture to add support.',
    );
  }
  return NnueHeader(
    version: version,
    architecture: arch,
    hashValue: hashValue,
    description: description,
    byteLength: r.offset,
  );
}

// HalfKAv2_hm fixed dimensions. Per Stockfish/src/nnue/nnue_architecture.h
// the FT weight matrix is shared between the two POVs and applied to
// each side's own 22528-feature index set; the accumulator is then
// concatenated to produce a 2×ftDim vector for the hidden stack.
//
// `ftDim` (Stockfish's `TransformedFeatureDimensions`) is 1024 for
// SFNNv5 and 1536 for the SFNNv5.1 / SF15.1 / SF16 default nets. We
// auto-detect it below from the biases block size so the reader
// works for both variants.
const int _kHalfKAv2HmInputs = 22528;
const int _kHalfKAv2HmPsqtBuckets = 8;
const List<int> _kSupportedFtDims = [1024, 1536];

NnueFeatureTransformerBlob _parseFeatureTransformerHalfKAv2Hm(
  _ByteReader r,
  NnueHeader header,
) {
  if (header.architecture != NnueArchitecture.halfKAv2Hm) {
    throw StateError(
      'unreachable: header.architecture=${header.architecture} '
      'but SFNNv8 reader called',
    );
  }
  if (r.remaining < 4) {
    throw const FormatException(
      'nnue: file ends before feature-transformer section hash',
    );
  }
  final sectionHash = r.readU32();
  const numInputs = _kHalfKAv2HmInputs;
  const psqtBuckets = _kHalfKAv2HmPsqtBuckets;

  // Auto-detect ftDim from the biases LEB128 block: we don't know the
  // count up-front, so decode until the payload is exhausted.
  final biasesDynamic = _readLeb128Auto(r);
  final ftDim = biasesDynamic.length;
  if (!_kSupportedFtDims.contains(ftDim)) {
    throw FormatException(
      'nnue: unexpected feature-transformer dim $ftDim '
      '(supported: $_kSupportedFtDims)',
    );
  }
  final biases = Int16List.fromList(biasesDynamic);
  final weights = Int16List(numInputs * ftDim);
  _readLeb128Into(r, weights, numInputs * ftDim);
  final psqt = Int32List(numInputs * psqtBuckets);
  _readLeb128Into(r, psqt, numInputs * psqtBuckets);

  return NnueFeatureTransformerBlob(
    sectionHash: sectionHash,
    biasesI16: biases,
    weightsI16: weights,
    psqtI32: psqt,
    ftDim: ftDim,
    numInputs: numInputs,
    psqtBuckets: psqtBuckets,
    byteEnd: r.offset,
  );
}

// Post-FT hidden stack for SFNNv5 / SF16 nets. Determined empirically
// on `nn-5af11540bbfe.nnue`: after the FT section the file has 206,656
// bytes for the network body, which splits cleanly as 8 × 25,832 bytes
// per bucket. Layout per bucket:
//
//   [u32]         bucket hash (Stockfish `Network::hash_value()`) (4 B)
//   [16 × int32]  L1 biases                                       (64 B)
//   [16 × 1536 × int8] L1 weights                                 (24,576 B)
//   [32 × int32]  L2 biases                                       (128 B)
//   [32 × 32 × int8] L2 weights                                   (1,024 B)
//   [1 × int32]   L3 biases                                       (4 B)
//   [1 × 32 × int8] L3 weights                                    (32 B)
//
// There is no outer section hash — `LayerStacks::read_parameters`
// just iterates the buckets and each bucket's own `read_parameters`
// consumes its own u32 hash.
const int _kBucketCount = 8;
const int _kL1OutDim = 16;
const int _kL2InDim = 32;
const int _kL2OutDim = 32;
const int _kL3InDim = 32;
const int _kL3OutDim = 1;

NnueNetworkBlob _parseNetworkHalfKAv2Hm(
  _ByteReader r,
  NnueFeatureTransformerBlob ft,
) {
  final l1InDim = ft.ftDim;
  final bytesPerBucket = 4 +
      _kL1OutDim * 4 +
      _kL1OutDim * l1InDim +
      _kL2OutDim * 4 +
      _kL2OutDim * _kL2InDim +
      _kL3OutDim * 4 +
      _kL3OutDim * _kL3InDim;
  final needed = _kBucketCount * bytesPerBucket;
  if (r.remaining != needed) {
    throw FormatException(
      'nnue: network body size mismatch — expected $needed bytes '
      '($_kBucketCount × $bytesPerBucket, L1_IN=$l1InDim), '
      'file has ${r.remaining} remaining. Arch parameters may differ '
      'for this net.',
    );
  }

  final buckets = <NnueNetworkBucket>[];
  for (int b = 0; b < _kBucketCount; b++) {
    final bucketHash = r.readU32();
    final l1Biases = _readInt32ListCopy(r, _kL1OutDim);
    final l1WeightBytes = r.readBytes(_kL1OutDim * l1InDim);
    final l1Weights = Int8List.view(
      l1WeightBytes.buffer,
      l1WeightBytes.offsetInBytes,
      _kL1OutDim * l1InDim,
    );
    final l2Biases = _readInt32ListCopy(r, _kL2OutDim);
    final l2WeightBytes = r.readBytes(_kL2OutDim * _kL2InDim);
    final l2Weights = Int8List.view(
      l2WeightBytes.buffer,
      l2WeightBytes.offsetInBytes,
      _kL2OutDim * _kL2InDim,
    );
    final l3Biases = _readInt32ListCopy(r, _kL3OutDim);
    final l3WeightBytes = r.readBytes(_kL3OutDim * _kL3InDim);
    final l3Weights = Int8List.view(
      l3WeightBytes.buffer,
      l3WeightBytes.offsetInBytes,
      _kL3OutDim * _kL3InDim,
    );
    buckets.add(
      NnueNetworkBucket(
        bucketHash: bucketHash,
        l1BiasesI32: l1Biases,
        l1WeightsI8: l1Weights,
        l2BiasesI32: l2Biases,
        l2WeightsI8: l2Weights,
        l3BiasesI32: l3Biases,
        l3WeightsI8: l3Weights,
      ),
    );
  }

  return NnueNetworkBlob(buckets: buckets, byteEnd: r.offset);
}

// Copy `n` little-endian int32 values from the current reader position.
// A copy (rather than an `Int32List.view`) avoids the 4-byte alignment
// requirement that fails when the preceding LEB128 payload landed at an
// unaligned offset.
Int32List _readInt32ListCopy(_ByteReader r, int n) {
  final out = Int32List(n);
  for (int i = 0; i < n; i++) {
    out[i] = r.readU32AsI32();
  }
  return out;
}

// SF15+ stores feature-transformer weights as signed LEB128 varints
// prefixed by the 17-byte ASCII tag "COMPRESSED_LEB128" and a u32 byte
// count of the compressed payload. Matches
// `Stockfish/src/nnue/nnue_common.h::read_leb_128`.
const String _kLeb128Magic = 'COMPRESSED_LEB128';

void _readLeb128Into(_ByteReader r, dynamic out, int count) {
  const magicLen = 17;
  if (r.remaining < magicLen + 4) {
    throw const FormatException(
      'nnue: file ends inside LEB128 section prologue',
    );
  }
  final magicBytes = r.readBytes(magicLen);
  final magic = String.fromCharCodes(magicBytes);
  if (magic != _kLeb128Magic) {
    throw FormatException(
      'nnue: expected LEB128 magic "$_kLeb128Magic" but got "$magic" '
      '(section may be raw / unsupported)',
    );
  }
  final bufSize = r.readU32();
  if (bufSize > r.remaining) {
    throw FormatException(
      'nnue: LEB128 payload of $bufSize bytes exceeds file remainder '
      '${r.remaining}',
    );
  }
  final payload = r.readBytes(bufSize);
  int cursor = 0;
  for (int i = 0; i < count; i++) {
    int value = 0;
    int shift = 0;
    int byte;
    while (true) {
      if (cursor >= payload.length) {
        throw FormatException(
          'nnue: LEB128 payload exhausted at element $i / $count',
        );
      }
      byte = payload[cursor++];
      value |= (byte & 0x7f) << shift;
      shift += 7;
      if ((byte & 0x80) == 0) break;
    }
    if ((byte & 0x40) != 0 && shift < 64) {
      value |= -(1 << shift);
    }
    if (out is Int16List) {
      out[i] = value;
    } else if (out is Int32List) {
      out[i] = value;
    } else {
      throw StateError('unsupported LEB128 output type ${out.runtimeType}');
    }
  }
  if (cursor != payload.length) {
    throw FormatException(
      'nnue: LEB128 payload had ${payload.length - cursor} trailing bytes '
      'after decoding $count values (buf=${payload.length}, count=$count)',
    );
  }
}

/// Same LEB128 framing as [_readLeb128Into] but decodes values until
/// the payload buffer is exhausted, returning all of them. Used when
/// we don't know the section's element count up-front and want to
/// derive it from the encoded size (e.g. auto-detecting the feature
/// transformer output dimension).
List<int> _readLeb128Auto(_ByteReader r) {
  const magicLen = 17;
  if (r.remaining < magicLen + 4) {
    throw const FormatException(
      'nnue: file ends inside LEB128 section prologue',
    );
  }
  final magicBytes = r.readBytes(magicLen);
  final magic = String.fromCharCodes(magicBytes);
  if (magic != _kLeb128Magic) {
    throw FormatException(
      'nnue: expected LEB128 magic "$_kLeb128Magic" but got "$magic"',
    );
  }
  final bufSize = r.readU32();
  if (bufSize > r.remaining) {
    throw FormatException(
      'nnue: LEB128 payload of $bufSize bytes exceeds file remainder '
      '${r.remaining}',
    );
  }
  final payload = r.readBytes(bufSize);
  final out = <int>[];
  int cursor = 0;
  while (cursor < payload.length) {
    int value = 0;
    int shift = 0;
    int byte;
    while (true) {
      if (cursor >= payload.length) {
        throw const FormatException('nnue: LEB128 varint truncated mid-value');
      }
      byte = payload[cursor++];
      value |= (byte & 0x7f) << shift;
      shift += 7;
      if ((byte & 0x80) == 0) break;
    }
    if ((byte & 0x40) != 0 && shift < 64) {
      value |= -(1 << shift);
    }
    out.add(value);
  }
  return out;
}

/// Tiny little-endian byte-stream cursor. Not exposed; NNUE files are
/// small enough that we don't need anything fancier.
class _ByteReader {
  _ByteReader(this.bytes)
    : _view = ByteData.view(
        bytes.buffer,
        bytes.offsetInBytes,
        bytes.lengthInBytes,
      );

  final Uint8List bytes;
  final ByteData _view;
  int offset = 0;

  int get remaining => bytes.length - offset;

  int readU32() {
    final v = _view.getUint32(offset, Endian.little);
    offset += 4;
    return v;
  }

  int readU32AsI32() {
    final v = _view.getInt32(offset, Endian.little);
    offset += 4;
    return v;
  }

  Uint8List readBytes(int n) {
    final start = offset;
    offset += n;
    return Uint8List.view(bytes.buffer, bytes.offsetInBytes + start, n);
  }
}
