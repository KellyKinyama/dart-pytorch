@Timeout(Duration(minutes: 3))
library;

import 'package:dart_pytorch/core/nn/esm2.dart';
import 'package:dart_pytorch/core/nn/fold/esmfold.dart';
import 'package:dart_pytorch/core/nn/fold/geometry.dart';
import 'package:dart_pytorch/core/nn/fold/structure_module.dart';
import 'package:dart_pytorch/core/nn/fold/trunk.dart';
import 'package:test/test.dart';

/// A tiny ESM-2 stem so the end-to-end pipeline runs fast without weights.
ESM2Model _tinyEsm() => ESM2Model(const ESM2Config(
      vocabSize: 33,
      maxCtx: 64,
      embedDim: 16,
      numLayers: 2,
      numHeads: 4,
      ffnDim: 32,
    ));

EsmFold _tinyModel() {
  const cfg = EsmFoldConfig(
    trunk: FoldingTrunkConfig(cS: 8, cZ: 4, seqHeadWidth: 4, pairHeadWidth: 2),
    structure: IpaConfig(
        cS: 12, cZ: 4, cHidden: 4, noHeads: 2, noQkPoints: 2, noVPoints: 2),
    esmFeats: 16,
    esmLayers: 2,
    numBlocks: 2,
    positionBins: 4,
    recycleBins: 5,
    maxRecycles: 1,
  );
  return EsmFold.build(cfg, _tinyEsm(), seed: 7);
}

void main() {
  test('af2ToEsm maps residues to valid ESM-2 token ids', () {
    final m = _tinyModel();
    // 22 entries: pad + 21 restypes_with_x.
    expect(m.af2ToEsm.length, 22);
    // Alanine (AF2 index 0) -> esm token for 'A'.
    expect(m.af2ToEsm[1], esm2Vocab.indexOf('A'));
    expect(m.af2ToEsm.every((t) => t >= 0 && t < 33), isTrue);
  });

  test('fold produces one backbone frame per residue (single pass)', () {
    final m = _tinyModel();
    final r = m.fold('MAGVSE', numRecycles: 0);
    expect(r.sequence, 'MAGVSE');
    expect(r.atoms.length, 6);
    for (final a in r.atoms) {
      for (final v in [a.n, a.cAlpha, a.c]) {
        expect(v.x.isFinite && v.y.isFinite && v.z.isFinite, isTrue);
      }
    }
  });

  test('ideal backbone bond lengths are preserved', () {
    final m = _tinyModel();
    final r = m.fold('ACDEFG', numRecycles: 0);
    for (final a in r.atoms) {
      expect((a.n - a.cAlpha).norm, closeTo(IdealBackbone.n.norm, 1e-3));
      expect((a.c - a.cAlpha).norm, closeTo(IdealBackbone.c.norm, 1e-3));
    }
  });

  test('recycling runs and stays finite', () {
    final m = _tinyModel();
    final r = m.fold('MKLV', numRecycles: 2);
    expect(r.atoms.length, 4);
    expect(r.atoms.every((a) => a.cAlpha.x.isFinite), isTrue);
  });

  test('toPdb emits well-formed ATOM records', () {
    final m = _tinyModel();
    final r = m.fold('MAG', numRecycles: 0);
    final pdb = EsmFold.toPdb(r);
    final atomLines =
        pdb.split('\n').where((l) => l.startsWith('ATOM')).toList();
    expect(atomLines.length, 9); // 3 residues × (N, CA, C)
    expect(atomLines.first.contains('MET'), isTrue);
    expect(atomLines.any((l) => l.contains(' CA ')), isTrue);
    expect(pdb.contains('END'), isTrue);
    // Fixed-column coordinate fields must parse as numbers.
    final x = double.tryParse(atomLines.first.substring(30, 38).trim());
    expect(x, isNotNull);
  });

  test('unknown residues fall back to X without crashing', () {
    final m = _tinyModel();
    final r = m.fold('MZB', numRecycles: 0); // Z, B not canonical
    expect(r.atoms.length, 3);
  });
}
