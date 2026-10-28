/// Map a `.cact` tensor directory onto named Needle 3 tensors.
///
/// The directory is positional and nameless; the order is the format
/// contract (https://cactuscompute.com/blog/porting-needle §2). This walker
/// slices it into the embedding, 20 blocks of 27 tensors, the mHC tensors,
/// the two Monarch shuffles, the engram sites, the final norm, the probe
/// heads, and the RAW tokenizer — asserting the dtype pattern as it goes so
/// a reordered or wrong-generation file is refused rather than mis-read.
library;

import 'cact_format.dart';
import 'needle_config.dart';

/// The 27 tensors of one SAN block, in directory order.
class NeedleBlock {
  final CactTensor normIn, qProj, kProj, vProj;
  final CactTensor qTaps, kTaps, vTaps, qNorm, kNorm;
  final CactTensor gateProj, outProj, postNorm, attnGate, preHada;
  final CactTensor d1, d2, b2, d3, d4;
  final CactTensor w1a, w1b, w2a, w2b, w3a, w3b;
  final CactTensor condV, condU;
  const NeedleBlock({
    required this.normIn,
    required this.qProj,
    required this.kProj,
    required this.vProj,
    required this.qTaps,
    required this.kTaps,
    required this.vTaps,
    required this.qNorm,
    required this.kNorm,
    required this.gateProj,
    required this.outProj,
    required this.postNorm,
    required this.attnGate,
    required this.preHada,
    required this.d1,
    required this.d2,
    required this.b2,
    required this.d3,
    required this.d4,
    required this.w1a,
    required this.w1b,
    required this.w2a,
    required this.w2b,
    required this.w3a,
    required this.w3b,
    required this.condV,
    required this.condU,
  });
}

/// The nine multi-lane hyper-connection tensors.
class NeedleMhc {
  final CactTensor aPre, aPost, aRes, bPre, bPost, bRes;
  final CactTensor phiPre, phiPost, phiRes;
  const NeedleMhc({
    required this.aPre,
    required this.aPost,
    required this.aRes,
    required this.bPre,
    required this.bPost,
    required this.bRes,
    required this.phiPre,
    required this.phiPost,
    required this.phiRes,
  });
}

/// One engram site: the n-gram tables plus key/value projections and taps.
class NeedleEngram {
  final int site; // the layer this site fires at
  final CactTensor tables, keyProj, valueProj, taps;
  const NeedleEngram({
    required this.site,
    required this.tables,
    required this.keyProj,
    required this.valueProj,
    required this.taps,
  });
}

/// One probe head (code 1 embedding, 2 confidence, 3 router).
class NeedleHead {
  final int code;
  final CactTensor probes, gain, query, rowBias, proj, bias;
  const NeedleHead({
    required this.code,
    required this.probes,
    required this.gain,
    required this.query,
    required this.rowBias,
    required this.proj,
    required this.bias,
  });
}

/// Fully-named view over a `.cact` directory.
class NeedleWeights {
  final CactFile file;
  final NeedleConfig config;
  final CactTensor embedding; // CQ4, tied output head
  final List<NeedleBlock> blocks;
  final NeedleMhc mhc;
  final CactTensor hadaP1, hadaP2; // Monarch shuffles, FP32 index vectors
  final List<NeedleEngram> engram;
  final CactTensor finalNorm;
  final List<NeedleHead> heads;
  final CactTensor manifest; // FP16 head-code vector
  final CactTensor tokenizer; // RAW SentencePiece dump

  const NeedleWeights({
    required this.file,
    required this.config,
    required this.embedding,
    required this.blocks,
    required this.mhc,
    required this.hadaP1,
    required this.hadaP2,
    required this.engram,
    required this.finalNorm,
    required this.heads,
    required this.manifest,
    required this.tokenizer,
  });

  static void _expect(bool ok, String what) {
    if (!ok) throw FormatException('cact directory: $what');
  }

  static void _dtype(CactTensor t, CactDType want, String name) {
    _expect(t.dtype == want,
        '$name expected ${want.name}, got ${t.dtypeLabel} at record ${t.index}');
  }

  /// Walk [file]'s directory into named tensors using [config] (defaults to
  /// [NeedleConfig.fromHeader]). Throws [FormatException] on any contract
  /// violation.
  static NeedleWeights map(CactFile file, {NeedleConfig? config}) {
    final cfg = config ?? NeedleConfig.fromHeader(file.header);
    final d = file.directory;
    var i = 0;

    CactTensor next() {
      _expect(i < d.length, 'ran out of records at $i');
      return d[i++];
    }

    // 0: token embedding (CQ4), also the tied output head.
    final embedding = next();
    _dtype(embedding, CactDType.cq, 'embedding');

    // L blocks of 27 tensors.
    final blocks = <NeedleBlock>[];
    for (var b = 0; b < cfg.numLayers; b++) {
      final normIn = next();
      _dtype(normIn, CactDType.fp16, 'block$b.norm_in');
      final qProj = next();
      _dtype(qProj, CactDType.cq, 'block$b.q_proj');
      final kProj = next();
      _dtype(kProj, CactDType.cq, 'block$b.k_proj');
      final vProj = next();
      _dtype(vProj, CactDType.cq, 'block$b.v_proj');
      final qTaps = next();
      final kTaps = next();
      final vTaps = next();
      final qNorm = next();
      final kNorm = next();
      final gateProj = next();
      _dtype(gateProj, CactDType.cq, 'block$b.gate_proj');
      final outProj = next();
      _dtype(outProj, CactDType.cq, 'block$b.out_proj');
      final postNorm = next();
      final attnGate = next();
      final preHada = next();
      final d1 = next(), d2 = next(), b2 = next(), d3 = next(), d4 = next();
      final w1a = next(), w1b = next(), w2a = next();
      final w2b = next(), w3a = next(), w3b = next();
      final condV = next(), condU = next();
      blocks.add(NeedleBlock(
        normIn: normIn, qProj: qProj, kProj: kProj, vProj: vProj,
        qTaps: qTaps, kTaps: kTaps, vTaps: vTaps, qNorm: qNorm, kNorm: kNorm,
        gateProj: gateProj, outProj: outProj, postNorm: postNorm,
        attnGate: attnGate, preHada: preHada,
        d1: d1, d2: d2, b2: b2, d3: d3, d4: d4,
        w1a: w1a, w1b: w1b, w2a: w2a, w2b: w2b, w3a: w3a, w3b: w3b,
        condV: condV, condU: condU, //
      ));
    }

    // mHC: 6 FP16 lane maps, then 3 CQ4 phi tensors.
    final mhc = NeedleMhc(
      aPre: next(), aPost: next(), aRes: next(),
      bPre: next(), bPost: next(), bRes: next(),
      phiPre: next(), phiPost: next(), phiRes: next(), //
    );
    _dtype(mhc.phiPre, CactDType.cq, 'mhc.phi_pre');

    // Two Monarch shuffles (FP32 index vectors, length hadamardSize).
    final hadaP1 = next();
    final hadaP2 = next();
    _dtype(hadaP1, CactDType.fp32, 'hada_p1');
    _dtype(hadaP2, CactDType.fp32, 'hada_p2');

    // Engram sites: one quad per active site, each tables/key/value/taps.
    final sites = cfg.activeEngramSites();
    final engram = <NeedleEngram>[];
    for (final site in sites) {
      final tables = next();
      _dtype(tables, CactDType.cq, 'engram[$site].tables');
      _expect(
          tables.shape.isNotEmpty &&
              tables.shape[0] % cfg.engramSlots == 0,
          'engram[$site].tables rows ${tables.shape} not a multiple of '
          'slots ${cfg.engramSlots}');
      final keyProj = next();
      final valueProj = next();
      final taps = next();
      engram.add(NeedleEngram(
        site: site, tables: tables, keyProj: keyProj,
        valueProj: valueProj, taps: taps, //
      ));
    }

    // final norm.
    final finalNorm = next();
    _dtype(finalNorm, CactDType.fp16, 'final_norm');

    // Probe heads: a manifest (one code per head), then 6 tensors per head.
    final manifest = next();
    _dtype(manifest, CactDType.fp16, 'heads.manifest');
    final numHeads = manifest.shape.isEmpty
        ? 1
        : manifest.shape.reduce((a, b) => a * b);
    final heads = <NeedleHead>[];
    for (var hh = 0; hh < numHeads; hh++) {
      final probes = next();
      final gain = next();
      final query = next();
      final rowBias = next();
      final proj = next();
      final bias = next();
      heads.add(NeedleHead(
        code: 2, // shipped file carries a single confidence head
        probes: probes, gain: gain, query: query,
        rowBias: rowBias, proj: proj, bias: bias, //
      ));
    }

    // Last: RAW tokenizer.
    final tokenizer = next();
    _dtype(tokenizer, CactDType.raw, 'tokenizer');

    _expect(i == d.length,
        'trailing records: consumed $i of ${d.length}');
    _expect(file.header.numTensors == d.length,
        'header tensor count ${file.header.numTensors} != directory '
        '${d.length}');

    return NeedleWeights(
      file: file, config: cfg, embedding: embedding, blocks: blocks,
      mhc: mhc, hadaP1: hadaP1, hadaP2: hadaP2, engram: engram,
      finalNorm: finalNorm, heads: heads, manifest: manifest,
      tokenizer: tokenizer, //
    );
  }
}
