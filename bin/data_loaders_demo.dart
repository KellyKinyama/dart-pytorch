/// End-to-end walk-through of the dataloader stack.
///
/// Exercises the three concrete `Dataset`s and the shared
/// `DataLoader` wrapper:
///
///  1. Builds a synthetic image-folder classification dataset in a
///     temp directory (3 classes × 4 tiny PNGs), decodes+patchifies
///     it with [ImageFolderDataset], and iterates one shuffled epoch
///     via a `DataLoader(batchSize=4)`.
///  2. Trains a [CharTokenizer] on a short poem and streams
///     sliding-window `(input, target)` LM pairs from a
///     [TextTokenDataset].
///  3. Parses in-memory CSV rows into feature+label tensors with
///     [CsvDataset].
///
/// This is a *usage* demo — it does not train a model. It prints
/// shapes, a sample tensor slice, and a decoded next-token target
/// so you can see the wiring works.
///
/// Run:
///
///     dart run bin/data_loaders_demo.dart
library;

import 'dart:io';

import 'package:image/image.dart' as img;

import 'package:dart_pytorch/dart_pytorch.dart';

Directory _buildImageFolder({
  required int numClasses,
  required int perClass,
  int size = 16,
}) {
  final root = Directory.systemTemp.createTempSync('dp_imgfolder_');
  for (int c = 0; c < numClasses; c++) {
    final dir = Directory('${root.path}/class_$c')..createSync();
    for (int k = 0; k < perClass; k++) {
      final image = img.Image(width: size, height: size);
      final r = (60 * c + 20 * k) % 256;
      final g = (40 * c + 30 * k) % 256;
      final b = (80 * c + 10 * k) % 256;
      for (final p in image) {
        p
          ..r = r
          ..g = g
          ..b = b;
      }
      File('${dir.path}/img_$k.png')
          .writeAsBytesSync(img.encodePng(image));
    }
  }
  return root;
}

void _section(String title) {
  print('\n' + '=' * 60);
  print('  $title');
  print('=' * 60);
}

void main() {
  print('=== data_loaders_demo ===');
  print('demonstrates ImageFolderDataset, TextTokenDataset, CsvDataset');

  // -------------------------------------------------------------------
  // 1. ImageFolderDataset  (folder-per-class classification)
  // -------------------------------------------------------------------
  _section('1. ImageFolderDataset — synthetic 3-class image gallery');
  final root = _buildImageFolder(numClasses: 3, perClass: 4, size: 16);
  try {
    final ds = ImageFolderDataset(
      root.path,
      imageSize: 16,
      patchSize: 8,
      valSplit: 0.25,
      seed: 0,
    );
    print('root:        ${root.path}');
    print('classes:     ${ds.classes}  (numClasses=${ds.numClasses})');
    print('train / val: ${ds.numTrain} / ${ds.numVal}');
    print('per-item:    patches=[${ds.numPatches}, ${ds.patchPixels}]  '
        '+ int label');

    final dl = DataLoader(ds, batchSize: 4, shuffle: true, seed: 1);
    print('DataLoader:  batchSize=4, ${dl.length} batches / epoch');
    int b = 0;
    for (final batch in dl.batches()) {
      final labels = batch.map((s) => s.label).toList();
      final firstShape = batch.first.patches.shape;
      print('  batch ${b++}  size=${batch.length}  '
          'labels=$labels  first.patches=$firstShape');
    }

    print('\ntriplet sampling for face-recognition style training:');
    for (int i = 0; i < 3; i++) {
      final t = ds.sampleTriplet();
      print('  triplet $i  anchor/positive class=${t.anchorClass}, '
          'negative class=${t.negativeClass}  '
          '(shapes ${t.anchor.shape})');
    }
  } finally {
    root.deleteSync(recursive: true);
  }

  // -------------------------------------------------------------------
  // 2. TextTokenDataset  (sliding-window LM examples)
  // -------------------------------------------------------------------
  _section('2. TextTokenDataset — char LM over a short poem');
  const poem = '''
The fog comes on little cat feet.
It sits looking over harbor and city
on silent haunches
and then moves on.
''';
  final tok = CharTokenizer.fromText(poem);
  print('tokenizer:   CharTokenizer  vocabSize=${tok.vocabSize}');
  final lmDs = TextTokenDataset.fromText(
    poem,
    tokenizer: tok,
    blockSize: 16,
  );
  print('numTokens:   ${lmDs.numTokens}');
  print('length:      ${lmDs.length}  (windows of size ${16})');

  final lmDl = DataLoader(lmDs, batchSize: 8, shuffle: true, seed: 0);
  print('DataLoader:  batchSize=8, ${lmDl.length} batches / epoch');
  final firstBatch = lmDl.batches().first;
  final sample = firstBatch.first;
  final inIds = sample.input.toList().map((v) => v.toInt()).toList();
  final tgtIds = sample.target.toList().map((v) => v.toInt()).toList();
  print('sample:      input.shape=${sample.input.shape}  '
      'target.shape=${sample.target.shape}');
  print('             input decoded : ${_pretty(tok.decode(inIds))}');
  print('             target decoded: ${_pretty(tok.decode(tgtIds))}');

  // -------------------------------------------------------------------
  // 3. CsvDataset  (tabular)
  // -------------------------------------------------------------------
  _section('3. CsvDataset — iris-style tabular classification');
  final csv = [
    'sepal_len,sepal_wid,petal_len,petal_wid,species',
    '5.1,3.5,1.4,0.2,setosa',
    '4.9,3.0,1.4,0.2,setosa',
    '7.0,3.2,4.7,1.4,versicolor',
    '6.4,3.2,4.5,1.5,versicolor',
    '6.3,3.3,6.0,2.5,virginica',
    '5.8,2.7,5.1,1.9,virginica',
  ];
  final tab = CsvDataset.fromLines(
    csv,
    labelColumn: 4,
    classMap: {'setosa': 0, 'versicolor': 1, 'virginica': 2},
  );
  print('headers:     ${tab.headers}');
  print('numFeatures: ${tab.numFeatures}   numExamples: ${tab.numExamples}');
  final tabDl = DataLoader(tab, batchSize: 3, shuffle: true, seed: 0);
  for (final batch in tabDl.batches()) {
    for (final s in batch) {
      print('  features=${s.features.toList()}  label=${s.label.toInt()}');
    }
    print('  --');
  }

  print('\n✅ dataloader stack works end-to-end.');
}

/// Replace newlines with `\n` for a one-line preview.
String _pretty(String s) =>
    "'" + s.replaceAll('\n', r'\n').replaceAll('\r', r'\r') + "'";
