import 'dart:io';
import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'package:test/test.dart';

import 'package:dart_pytorch/dart_pytorch.dart';

/// Build a temporary folder-per-class dataset of tiny solid-color PNGs.
///
/// Returns the root directory (caller must delete when done).
Directory _buildImageFolder({
  required int numClasses,
  required int perClass,
  int size = 8,
}) {
  final root = Directory.systemTemp.createTempSync('imgfolder_');
  for (int c = 0; c < numClasses; c++) {
    final dir = Directory('${root.path}/class_$c')..createSync();
    for (int k = 0; k < perClass; k++) {
      // Each class gets its own base color; per-image variation via `k`.
      final image = img.Image(width: size, height: size);
      final r = (30 * c) % 256;
      final g = (50 * c + 40 * k) % 256;
      final b = (10 * k) % 256;
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

void main() {
  group('Dataset + DataLoader', () {
    test('ListDataset indexes and reports length', () {
      final ds = ListDataset<int>([10, 20, 30, 40]);
      expect(ds.length, 4);
      expect(ds[0], 10);
      expect(ds[3], 40);
    });

    test('DataLoader batches without shuffle keeps order', () {
      final ds = ListDataset<int>(List<int>.generate(7, (i) => i));
      final dl = DataLoader(ds, batchSize: 3);
      expect(dl.length, 3); // 3 + 3 + 1
      final batches = dl.batches().toList();
      expect(batches, [
        [0, 1, 2],
        [3, 4, 5],
        [6],
      ]);
    });

    test('DataLoader dropLast trims short trailing batch', () {
      final ds = ListDataset<int>(List<int>.generate(7, (i) => i));
      final dl = DataLoader(ds, batchSize: 3, dropLast: true);
      expect(dl.length, 2);
      final batches = dl.batches().toList();
      expect(batches, [
        [0, 1, 2],
        [3, 4, 5],
      ]);
    });

    test('DataLoader shuffle is deterministic given a seed', () {
      final ds = ListDataset<int>(List<int>.generate(10, (i) => i));
      final a = DataLoader(ds, batchSize: 4, shuffle: true, seed: 42)
          .batches()
          .expand((b) => b)
          .toList();
      final b = DataLoader(ds, batchSize: 4, shuffle: true, seed: 42)
          .batches()
          .expand((batch) => batch)
          .toList();
      final c = DataLoader(ds, batchSize: 4, shuffle: true, seed: 99)
          .batches()
          .expand((batch) => batch)
          .toList();
      expect(a, equals(b), reason: 'same seed → same order');
      expect(a, isNot(equals(c)), reason: 'different seed → different order');
      expect(a.toSet(), equals(Set.of(List<int>.generate(10, (i) => i))));
    });

    test('batchSize < 1 throws', () {
      final ds = ListDataset<int>([1, 2, 3]);
      expect(() => DataLoader(ds, batchSize: 0), throwsArgumentError);
    });
  });

  group('ImageFolderDataset', () {
    late Directory root;

    setUp(() {
      root = _buildImageFolder(numClasses: 3, perClass: 4, size: 8);
    });

    tearDown(() {
      root.deleteSync(recursive: true);
    });

    test('discovers classes and splits train/val', () {
      final ds = ImageFolderDataset(
        root.path,
        imageSize: 8,
        patchSize: 4,
        valSplit: 0.25,
        seed: 0,
      );
      expect(ds.classes, equals(['class_0', 'class_1', 'class_2']));
      expect(ds.numClasses, 3);
      // 4 per class * 3 classes = 12 total; val = round(4 * 0.25) = 1 each.
      expect(ds.numTrain, 9);
      expect(ds.numVal, 3);
      expect(ds.length, 9, reason: 'default indexes train split');
      expect(ds.valSplit().length, 3);
    });

    test('items are patchified tensors with correct shape', () {
      final ds = ImageFolderDataset(
        root.path,
        imageSize: 8,
        patchSize: 4,
        valSplit: 0.25,
        seed: 0,
      );
      final s = ds[0];
      // imageSize/patchSize = 2 → 4 patches; each patch = 4*4*3 = 48 pixels.
      expect(s.patches.shape, equals([4, 48]));
      expect(s.label, inInclusiveRange(0, 2));
      for (final v in s.patches.toList()) {
        expect(v, inInclusiveRange(0.0, 1.0));
      }
    });

    test('DataLoader over ImageFolderDataset yields batches of tensors', () {
      final ds = ImageFolderDataset(
        root.path,
        imageSize: 8,
        patchSize: 4,
        valSplit: 0.25,
        seed: 0,
      );
      final dl = DataLoader(ds, batchSize: 4, shuffle: true, seed: 3);
      final firstBatch = dl.batches().first;
      expect(firstBatch.length, 4);
      for (final item in firstBatch) {
        expect(item.patches.shape, equals([4, 48]));
      }
    });

    test('sampleTriplet returns three distinct-class tensors', () {
      final ds = ImageFolderDataset(
        root.path,
        imageSize: 8,
        patchSize: 4,
        valSplit: 0.25,
        seed: 0,
      );
      final t = ds.sampleTriplet();
      expect(t.anchor.shape, equals([4, 48]));
      expect(t.positive.shape, equals([4, 48]));
      expect(t.negative.shape, equals([4, 48]));
      expect(t.anchorClass, isNot(equals(t.negativeClass)));
    });
  });

  group('TextTokenDataset', () {
    test('sliding-window (input, target) pairs with a CharTokenizer', () {
      const text = 'hello world!';
      final tok = CharTokenizer.fromText(text);
      final ds = TextTokenDataset.fromText(
        text,
        tokenizer: tok,
        blockSize: 4,
      );
      // length = numTokens - blockSize.
      expect(ds.length, tok.encode(text).length - 4);

      final s = ds[0];
      expect(s.input.shape, equals([4]));
      expect(s.target.shape, equals([4]));

      // target is input shifted by one.
      final ids = tok.encode(text);
      expect(s.input.toList().map((v) => v.toInt()).toList(),
          equals(ids.sublist(0, 4)));
      expect(s.target.toList().map((v) => v.toInt()).toList(),
          equals(ids.sublist(1, 5)));
    });

    test('fromTokens accepts a pre-tokenized id list', () {
      final ds = TextTokenDataset.fromTokens(
        List<int>.generate(20, (i) => i),
        blockSize: 3,
      );
      expect(ds.length, 17);
      expect(ds[0].input.toList().map((v) => v.toInt()).toList(),
          equals([0, 1, 2]));
      expect(ds[0].target.toList().map((v) => v.toInt()).toList(),
          equals([1, 2, 3]));
    });

    test('rejects blockSize >= numTokens', () {
      expect(
        () => TextTokenDataset.fromTokens([1, 2, 3], blockSize: 3),
        throwsArgumentError,
      );
    });
  });

  group('CsvDataset', () {
    test('numeric features + numeric label from in-memory lines', () {
      final lines = [
        'x,y,z,label',
        '1.0, 2.0, 3.0, 0',
        '4.0, 5.0, 6.0, 1',
        '7.0, 8.0, 9.0, 0',
      ];
      final ds = CsvDataset.fromLines(
        lines,
        labelColumn: 3,
      );
      expect(ds.length, 3);
      expect(ds.numFeatures, 3);
      expect(ds.headers, equals(['x', 'y', 'z', 'label']));
      final s = ds[1];
      expect(s.features.shape, equals([3]));
      expect(s.features.toList(), equals([4.0, 5.0, 6.0]));
      expect(s.label, 1.0);
    });

    test('classMap maps categorical labels to ints', () {
      final lines = [
        'a,b,species',
        '1,2,cat',
        '3,4,dog',
        '5,6,cat',
      ];
      final ds = CsvDataset.fromLines(
        lines,
        labelColumn: 2,
        classMap: {'cat': 0, 'dog': 1},
      );
      expect(ds[0].label, 0.0);
      expect(ds[1].label, 1.0);
      expect(ds[2].label, 0.0);
    });

    test('no label column → label defaults to -1', () {
      final lines = ['a,b', '1,2', '3,4'];
      final ds = CsvDataset.fromLines(lines);
      expect(ds.numFeatures, 2);
      expect(ds[0].label, -1);
      expect(ds[0].features.toList(), equals([1.0, 2.0]));
    });

    test('mismatched column count throws', () {
      final lines = ['a,b', '1,2', '3'];
      expect(
        () => CsvDataset.fromLines(lines),
        throwsA(isA<FormatException>()),
      );
    });
  });
}
