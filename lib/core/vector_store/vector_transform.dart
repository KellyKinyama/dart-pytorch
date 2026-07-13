/// Vector pre-transforms — reversible or non-reversible mappings applied
/// to database and query vectors before they reach an underlying
/// [Index]. Mirrors FAISS' `VectorTransform` hierarchy.
///
/// A [VectorTransform] maps `dIn`-dimensional inputs to `dOut`-
/// dimensional outputs. Some transforms (e.g. L2 normalization,
/// random rotation) have `dIn == dOut`; others (e.g. PCA — not yet
/// implemented) may reduce dimensionality.
///
/// Transforms compose via [IndexPreTransform].
library;

import 'dart:typed_data';

import 'index_io.dart';

/// Wire-level discriminators for [VectorTransform] subtypes. Never
/// change existing values — they are the persistence contract.
class TransformKind {
  static const int l2Norm = 0x01;
  static const int randomRotation = 0x02;
  static const int pca = 0x03;
}

abstract class VectorTransform {
  VectorTransform(this.dIn, this.dOut);

  final int dIn;
  final int dOut;
  bool isTrained = true;

  /// Optional training step. Default no-op.
  void train(List<Float32List> xs) {
    isTrained = true;
  }

  /// Apply the transform to a batch of vectors.
  List<Float32List> apply(List<Float32List> xs);

  /// Optional inverse. Throws by default.
  List<Float32List> reverseTransform(List<Float32List> xs) {
    throw UnsupportedError('$runtimeType does not implement reverseTransform');
  }

  /// Serialize this transform (subkind + payload) into [w].
  /// Called by [IndexPreTransform] when persisting a chain.
  void writeTo(IoWriter w);
}
