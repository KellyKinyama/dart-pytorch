/// Geometry for a Needle 3 model, derived from a `.cact` header plus the
/// architecture constants documented in the porting guide
/// (https://cactuscompute.com/blog/porting-needle).
///
/// Only a few header field offsets are public (vocab, width, layers,
/// kv_bits, rope_theta); the rest of the geometry is fixed by the
/// architecture and carried here as documented defaults. Sliced archives
/// (`--layers N`) change `numLayers` and the surviving engram/global sites,
/// which [ladderBlocks], [activeGlobalLayers] and [activeEngramSites]
/// reproduce from the ladder rule.
library;

import 'cact_format.dart';

class NeedleConfig {
  // --- from the header ---
  final int vocab; // 8192
  final int dModel; // 768
  final int numLayers; // 20 (or a sliced depth)
  final int kvBits; // int8 KV cache width
  final double ropeTheta; // 100000

  // --- fixed architecture constants (documented) ---
  final int numQHeads; // 12
  final int headDimQ; // 48 (q and k)
  final int numKvHeads; // 2
  final int headDimV; // 64
  final int convTaps; // 3 causal conv taps on q/k/v
  final int window; // 1024 local attention window
  final int numLanes; // 4 residual lanes (mHC)
  final int engramSlots; // 18432 rows per engram table
  final int engramSubDim; // 128
  final List<int> engramOrders; // {2, 3}-gram
  final int kvWindow; // 256 (KV cache training window)

  const NeedleConfig({
    required this.vocab,
    required this.dModel,
    required this.numLayers,
    required this.kvBits,
    required this.ropeTheta,
    this.numQHeads = 12,
    this.headDimQ = 48,
    this.numKvHeads = 2,
    this.headDimV = 64,
    this.convTaps = 3,
    this.window = 1024,
    this.numLanes = 4,
    this.engramSlots = 18432,
    this.engramSubDim = 128,
    this.engramOrders = const [2, 3],
    this.kvWindow = 256,
  });

  factory NeedleConfig.fromHeader(CactHeader h) {
    // Real files carry full geometry; synthetic test headers leave most
    // fields 0, so fall back to the shipped-model defaults when a field is 0.
    int or(int v, int d) => v != 0 ? v : d;
    return NeedleConfig(
      vocab: h.vocab,
      dModel: h.dModel,
      numLayers: h.numLayers,
      kvBits: h.kvBits,
      ropeTheta: h.ropeTheta != 0 ? h.ropeTheta : 100000.0,
      numQHeads: or(h.numHeads, 12),
      headDimQ: or(h.qkHeadDim, 48),
      numKvHeads: or(h.numKvHeads, 2),
      headDimV: or(h.vHeadDim, 64),
      convTaps: or(h.qkvConvTaps, 3),
      window: or(h.slidingWindow, 1024),
      numLanes: or(h.mhcLanes, 4),
      engramSlots: or(h.engramSlots, 18432),
      engramSubDim: or(h.engramSubDim, 128),
      engramOrders: h.numEngramOrders > 0 ? h.engramOrders : const [2, 3],
    );
  }

  /// Hadamard-MLP channel count: the next power of two ≥ [dModel]
  /// (1024 for the shipped 768-wide model). Matches the reference
  /// `n = 1 << (d_model - 1).bit_length()`.
  int get hadamardSize => dModel <= 1 ? 1 : (1 << (dModel - 1).bitLength);

  /// q_proj output dim: `numQHeads * headDimQ` (576 for the shipped model).
  int get qProjDim => numQHeads * headDimQ;

  /// k_proj output dim: `numKvHeads * headDimQ` (96).
  int get kProjDim => numKvHeads * headDimQ;

  /// v_proj output dim: `numKvHeads * headDimV` (128).
  int get vProjDim => numKvHeads * headDimV;

  /// The 16 reserved special-token surfaces, ids 0..15, in order.
  static const List<String> specialTokens = [
    '<pad>',
    '</s>',
    '<s>',
    '<unk>',
    '<|im_start|>',
    '<|im_end|>',
    '<think>',
    '</think>',
    '<tools>',
    '</tools>',
    '<tool_call>',
    '</tool_call>',
    '<tool_result>',
    '</tool_result>',
    '<context>',
    '</context>',
  ];

  /// The full 20-layer engram sites and global-attention layers.
  static const List<int> fullEngramSites = [3, 7, 11, 15, 19];
  static const List<int> fullGlobalLayers = [4, 9, 14, 19];

  /// Block-selection order for the depth ladder (endpoint-preserving
  /// bisection): depth N keeps the first N of these, then sorts ascending.
  static const List<int> ladderOrder = [
    0, 19, 9, 14, 4, 6, 11, 16, 2, 7, 12, 17, 1, 3, 5, 8, 10, 13, 15, 18, //
  ];

  /// The original (0..19) block indices kept at depth [n], ascending.
  static List<int> ladderBlocks(int n) {
    final kept = ladderOrder.take(n).toList()..sort();
    return kept;
  }

  /// Global-attention layers active at depth [numLayers]: whichever of
  /// {4, 9, 14, 19} are below the layer count.
  List<int> activeGlobalLayers() =>
      [for (final g in fullGlobalLayers) if (g < numLayers) g];

  /// Engram sites that fire at depth [numLayers]: whichever of
  /// {3, 7, 11, 15, 19} are below the layer count. (Sliced archives carry
  /// their remapped sites in the header; this covers the full and plain
  /// small-model cases.)
  List<int> activeEngramSites() =>
      [for (final s in fullEngramSites) if (s < numLayers) s];

  @override
  String toString() => 'NeedleConfig(vocab=$vocab, d=$dModel, '
      'layers=$numLayers, qh=$numQHeads x$headDimQ, kvh=$numKvHeads, '
      'vdim=$headDimV, rope=$ropeTheta, window=$window, lanes=$numLanes, '
      'engramSites=${activeEngramSites()})';
}
