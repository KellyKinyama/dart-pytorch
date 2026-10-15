/// Top-level **ESMFold** assembly: ESM-2 stem → learned per-layer combination →
/// folding trunk (relative-position bias + triangular self-attention blocks,
/// with recycling) → structure module → backbone atoms and a PDB string.
///
/// Mirrors `EsmForProteinFolding.forward` in HuggingFace
/// `modeling_esmfold.py`. The ESM-2 language model runs on the `Tensor` core
/// (CPU or GPU); the trunk and structure module are the pure-Dart modules in
/// this directory. Weights load via [loadEsmFold] from a converted
/// `facebook/esmfold_v1` safetensors map.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import '../../tensor/tensor.dart';
import '../esm2.dart';
import 'geometry.dart';
import 'structure_module.dart';
import 'trunk.dart';

/// AlphaFold2 residue ordering (`restypes_with_x`); index == aatype.
const String restypeOrderWithX = 'ARNDCQEGHILKMFPSTWYVX';

/// One-letter → three-letter residue names for PDB output.
const Map<String, String> _threeLetter = {
  'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
  'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
  'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
  'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
  'X': 'UNK',
};

/// Configuration for the full model. Defaults are `facebook/esmfold_v1`.
class EsmFoldConfig {
  final FoldingTrunkConfig trunk; // cS=1024, cZ=128
  final IpaConfig structure; // sequence_dim=384, pairwise_dim=128
  final int esmFeats; // ESM-2 hidden size (2560 for 3B)
  final int esmLayers; // ESM-2 transformer layers (36) → 37 representations
  final int numBlocks; // trunk blocks (48)
  final int positionBins; // relative-position clamp (32)
  final int recycleBins; // distogram recycle bins (15)
  final int maxRecycles; // default recycles (4)
  final int nTokensEmbed; // aa embedding rows (23)
  final double transScale; // structure-module Å scaling (10)

  const EsmFoldConfig({
    this.trunk = const FoldingTrunkConfig(),
    this.structure = const IpaConfig(),
    this.esmFeats = 2560,
    this.esmLayers = 36,
    this.numBlocks = 48,
    this.positionBins = 32,
    this.recycleBins = 15,
    this.maxRecycles = 4,
    this.nTokensEmbed = 23,
    this.transScale = 10.0,
  });

  int get esmStack => esmLayers + 1;
}

Float32List _ones(int d) => Float32List(d)..fillRange(0, d, 1.0);

/// Result of a fold: per-residue backbone atoms plus the residue sequence.
class FoldResult {
  final String sequence;
  final List<({Vec3 n, Vec3 cAlpha, Vec3 c})> atoms;
  const FoldResult(this.sequence, this.atoms);
}

/// The ESMFold model. Construct with [EsmFold.build] (random init, for tests)
/// or load pretrained weights via `loadEsmFold` in `esmfold_loader.dart`.
class EsmFold {
  final EsmFoldConfig cfg;
  final ESM2Model esm; // ESM-2 stem
  final List<int> af2ToEsm; // AF2 aatype(+1) → ESM-2 token id

  // Top-level glue.
  final Float32List esmSCombine; // [esmStack] learned layer weights
  final Float32List mlpLnG, mlpLnB; // esm_s_mlp.0 LayerNorm(esmFeats)
  final Dense mlp1, mlp2; // esm_s_mlp.1 / .3
  final Float32List aaEmbedding; // [nTokensEmbed, cS]

  // Trunk glue.
  final Float32List relPos; // [(2*bins+2), cZ] relative-position embedding
  final Float32List recycleSg, recycleSb, recycleZg, recycleZb; // LayerNorms
  final Float32List recycleDisto; // [recycleBins, cZ]
  final Dense trunk2smS, trunk2smZ;
  final List<TriangularSelfAttentionBlock> blocks;

  final StructureModule structure;

  EsmFold(
    this.cfg,
    this.esm,
    this.af2ToEsm,
    this.esmSCombine,
    this.mlpLnG,
    this.mlpLnB,
    this.mlp1,
    this.mlp2,
    this.aaEmbedding,
    this.relPos,
    this.recycleSg,
    this.recycleSb,
    this.recycleZg,
    this.recycleZb,
    this.recycleDisto,
    this.trunk2smS,
    this.trunk2smZ,
    this.blocks,
    this.structure,
  );

  /// Random-init model for the given config and ESM-2 stem (used in tests).
  factory EsmFold.build(EsmFoldConfig cfg, ESM2Model esm, {int seed = 0}) {
    final t = cfg.trunk;
    final blocks = [
      for (var i = 0; i < cfg.numBlocks; i++)
        TriangularSelfAttentionBlock.random(t, seed: seed + 1000 + i * 37)
    ];
    return EsmFold(
      cfg,
      esm,
      _buildAf2ToEsm(),
      Float32List(cfg.esmStack),
      _ones(cfg.esmFeats),
      Float32List(cfg.esmFeats),
      Dense.random(cfg.esmFeats, t.cS, seed: seed + 1),
      Dense.random(t.cS, t.cS, seed: seed + 2),
      Float32List(cfg.nTokensEmbed * t.cS),
      Float32List((2 * cfg.positionBins + 2) * t.cZ),
      _ones(t.cS), Float32List(t.cS),
      _ones(t.cZ), Float32List(t.cZ),
      Float32List(cfg.recycleBins * t.cZ),
      Dense.random(t.cS, cfg.structure.cS, seed: seed + 3),
      Dense.random(t.cZ, cfg.structure.cZ, seed: seed + 4),
      blocks,
      StructureModule.random(cfg.structure,
          noBlocks: 8, seed: seed + 5),
    );
  }

  /// Map AF2 `aatype + 1` (0 = pad) to ESM-2 token ids, from [esm2Vocab].
  static List<int> _buildAf2ToEsm() {
    final out = <int>[esm2Vocab.indexOf('<pad>')];
    for (final r in restypeOrderWithX.split('')) {
      final idx = esm2Vocab.indexOf(r);
      out.add(idx < 0 ? esm2Vocab.indexOf('<unk>') : idx);
    }
    return out;
  }

  /// Fold [sequence] (one-letter amino acids) into backbone coordinates.
  /// [numRecycles] defaults to `cfg.maxRecycles`; 0 is a single pass.
  FoldResult fold(String sequence, {int? numRecycles}) {
    final seq = sequence.toUpperCase().replaceAll(RegExp(r'[^A-Z]'), '');
    final n = seq.length;
    if (n == 0) throw ArgumentError('empty sequence');
    final recycles = (numRecycles ?? cfg.maxRecycles).clamp(0, 16);

    final aa = [
      for (final ch in seq.split(''))
        restypeOrderWithX.indexOf(ch) < 0
            ? restypeOrderWithX.indexOf('X')
            : restypeOrderWithX.indexOf(ch)
    ];

    final sS0 = _languageModelFeatures(aa, n); // [n, cS]
    // Add the aa embedding.
    final cS = cfg.trunk.cS;
    for (var i = 0; i < n; i++) {
      final e = aa[i] * cS;
      for (var c = 0; c < cS; c++) {
        sS0[i * cS + c] += aaEmbedding[e + c];
      }
    }

    final cZ = cfg.trunk.cZ;
    final sZ0 = Float32List(n * n * cZ);
    final relposBias = _relativePosition(n); // [n, n, cZ]

    var recycleS = Float32List(n * cS);
    var recycleZ = Float32List(n * n * cZ);
    var recycleBins = Int32List(n * n); // all zeros initially
    late ({List<Rigid> frames, List<({Vec3 n, Vec3 cAlpha, Vec3 c})> atoms})
        sm;

    for (var it = 0; it <= recycles; it++) {
      // Recycling inputs.
      final rs = _layerNormRows(recycleS, n, cS, recycleSg, recycleSb);
      final rz = _layerNormRows(recycleZ, n * n, cZ, recycleZg, recycleZb);
      for (var k = 0; k < n * n; k++) {
        final db = recycleBins[k] * cZ;
        for (var c = 0; c < cZ; c++) {
          rz[k * cZ + c] += recycleDisto[db + c];
        }
      }

      var s = Float32List(n * cS);
      var z = Float32List(n * n * cZ);
      for (var i = 0; i < s.length; i++) {
        s[i] = sS0[i] + rs[i];
      }
      for (var i = 0; i < z.length; i++) {
        z[i] = sZ0[i] + rz[i] + relposBias[i];
      }

      for (final blk in blocks) {
        final out = blk.call(s, z, n);
        s = out.s;
        z = out.z;
      }

      // Project into the structure module and run it.
      final smS = trunk2smS.applyRows(s, n);
      final smZ = trunk2smZ.applyRows(z, n * n);
      sm = structure.call(smS, smZ);

      recycleS = s;
      recycleZ = z;
      recycleBins = _distogram(sm.atoms, n);
    }

    return FoldResult(seq, sm.atoms);
  }

  /// Combine the ESM-2 per-layer representations into trunk input `s_s_0`.
  Float32List _languageModelFeatures(List<int> aa, int n) {
    // Build ESM token ids: <cls> + af2ToEsm[aa+1] + <eos>.
    final ids = <double>[esm2Vocab.indexOf('<cls>').toDouble()];
    for (final a in aa) {
      ids.add(af2ToEsm[a + 1].toDouble());
    }
    ids.add(esm2Vocab.indexOf('<eos>').toDouble());

    final tokens = Tensor.fromList([ids.length], ids);
    final reps = Tensor.noGrad(() {
      final list = esm.representations(tokens);
      return [for (final r in list) r.toFloat32List()];
    });

    // Softmax over the learned per-layer combination weights.
    final w = _softmaxCopy(esmSCombine);
    final d = cfg.esmFeats;
    final esmS = Float32List(n * d);
    for (var l = 0; l < n; l++) {
      final src = (l + 1) * d; // +1 strips the <cls> row
      for (var k = 0; k < reps.length; k++) {
        final wk = w[k];
        final rk = reps[k];
        for (var c = 0; c < d; c++) {
          esmS[l * d + c] += wk * rk[src + c];
        }
      }
    }

    // esm_s_mlp: LayerNorm → Linear → ReLU → Linear.
    final ln = _layerNormRows(esmS, n, d, mlpLnG, mlpLnB);
    final h = mlp1.applyRows(ln, n);
    for (var i = 0; i < h.length; i++) {
      if (h[i] < 0) h[i] = 0;
    }
    return mlp2.applyRows(h, n);
  }

  /// Relative-position pair bias `[n, n, cZ]` (EsmFoldRelativePosition).
  Float32List _relativePosition(int n) {
    final cZ = cfg.trunk.cZ;
    final bins = cfg.positionBins;
    final out = Float32List(n * n * cZ);
    for (var i = 0; i < n; i++) {
      for (var j = 0; j < n; j++) {
        var diff = j - i; // residue_index[j] - residue_index[i]
        if (diff < -bins) diff = -bins;
        if (diff > bins) diff = bins;
        final row = (diff + bins + 1) * cZ; // +1 reserves 0 for masked pairs
        final o = (i * n + j) * cZ;
        for (var c = 0; c < cZ; c++) {
          out[o + c] = relPos[row + c];
        }
      }
    }
    return out;
  }

  /// Distogram recycle bins from backbone N/Cα/C (infers Cβ like AlphaFold).
  Int32List _distogram(List<({Vec3 n, Vec3 cAlpha, Vec3 c})> atoms, int nres) {
    const minBin = 3.375, maxBin = 21.375;
    final nb = cfg.recycleBins;
    final bounds = List<double>.generate(nb - 1, (i) {
      final v = minBin + (maxBin - minBin) * i / (nb - 2);
      return v * v;
    });
    final cb = [
      for (final a in atoms)
        () {
          final b = a.cAlpha - a.n;
          final c = a.c - a.cAlpha;
          final cross = b.cross(c);
          return cross.scale(-0.58273431) +
              b.scale(0.56802827) -
              c.scale(0.54067466) +
              a.cAlpha;
        }()
    ];
    final out = Int32List(nres * nres);
    for (var i = 0; i < nres; i++) {
      for (var j = 0; j < nres; j++) {
        final d = cb[i] - cb[j];
        final d2 = d.dot(d);
        var bin = 0;
        for (final b in bounds) {
          if (d2 > b) bin++;
        }
        out[i * nres + j] = bin;
      }
    }
    return out;
  }

  /// Render the fold as a PDB file string (backbone N, Cα, C atoms).
  static String toPdb(FoldResult r) {
    final sb = StringBuffer();
    var serial = 1;
    for (var i = 0; i < r.atoms.length; i++) {
      final res = _threeLetter[r.sequence[i]] ?? 'UNK';
      final a = r.atoms[i];
      void atom(String name, Vec3 p, String element) {
        sb.writeln(_pdbAtom(serial++, name, res, i + 1, p, element));
      }

      atom('N', a.n, 'N');
      atom('CA', a.cAlpha, 'C');
      atom('C', a.c, 'C');
    }
    sb.writeln('TER');
    sb.writeln('END');
    return sb.toString();
  }

  static String _pdbAtom(
      int serial, String name, String res, int resSeq, Vec3 p, String elem) {
    final nameField = name.length < 4 ? ' ${name.padRight(3)}' : name;
    return 'ATOM  '
        '${serial.toString().padLeft(5)} '
        '$nameField '
        '$res A'
        '${resSeq.toString().padLeft(4)}    '
        '${p.x.toStringAsFixed(3).padLeft(8)}'
        '${p.y.toStringAsFixed(3).padLeft(8)}'
        '${p.z.toStringAsFixed(3).padLeft(8)}'
        '  1.00  0.00          '
        '${elem.padLeft(2)}';
  }
}

// --------------------------------------------------------------------------
// Local numeric helpers (kept private to avoid cross-file coupling).
// --------------------------------------------------------------------------

Float32List _layerNormRows(
    Float32List x, int n, int d, Float32List gamma, Float32List beta,
    {double eps = 1e-5}) {
  final out = Float32List(n * d);
  for (var r = 0; r < n; r++) {
    final o = r * d;
    var mean = 0.0;
    for (var i = 0; i < d; i++) {
      mean += x[o + i];
    }
    mean /= d;
    var v = 0.0;
    for (var i = 0; i < d; i++) {
      final dv = x[o + i] - mean;
      v += dv * dv;
    }
    v /= d;
    final inv = 1.0 / math.sqrt(v + eps);
    for (var i = 0; i < d; i++) {
      out[o + i] = (x[o + i] - mean) * inv * gamma[i] + beta[i];
    }
  }
  return out;
}

Float32List _softmaxCopy(Float32List a) {
  var mx = a[0];
  for (final v in a) {
    if (v > mx) mx = v;
  }
  final out = Float32List(a.length);
  var sum = 0.0;
  for (var i = 0; i < a.length; i++) {
    final e = math.exp(a[i] - mx);
    out[i] = e;
    sum += e;
  }
  for (var i = 0; i < out.length; i++) {
    out[i] /= sum;
  }
  return out;
}
