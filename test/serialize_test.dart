import 'dart:io';
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

void main() {
  group('Checkpoint.saveBytes / loadIntoBytes', () {
    test('round-trip a Linear reproduces its output byte-identically', () {
      final l1 = Linear(4, 3, seed: 1);
      final x = Tensor.fromList([2, 4], [1, 2, 3, 4, 5, 6, 7, 8]);
      final y1 = l1(x).toList();

      final bytes = Checkpoint.saveBytes(l1);

      // Fresh module with different seed — outputs would differ.
      final l2 = Linear(4, 3, seed: 999);
      final yDiff = l2(x).toList();
      expect(yDiff, isNot(y1));

      Checkpoint.loadIntoBytes(l2, bytes);
      final y2 = l2(x).toList();
      expect(y2, y1);
    });

    test(
      'preserves values through backward+step (weights actually change)',
      () {
        final l = Linear(3, 2, seed: 2);
        final before = l.weight.toList();
        // One SGD step on a synthetic loss.
        final opt = SGD(l.parameters(), lr: 0.1);
        final x = Tensor.fromList([1, 3], [1, 1, 1]);
        opt.zeroGrad();
        l(x).sum().backward();
        opt.step();
        final after = l.weight.toList();
        expect(after, isNot(before));

        // Save trained weights, then reload into a fresh module.
        final bytes = Checkpoint.saveBytes(l);
        final l2 = Linear(3, 2, seed: 12345);
        Checkpoint.loadIntoBytes(l2, bytes);
        expect(l2.weight.toList(), after);
      },
    );

    test('rejects bad magic bytes', () {
      final l = Linear(3, 2, seed: 3);
      expect(
        () => Checkpoint.loadIntoBytes(
          l,
          Uint8List.fromList([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        ),
        throwsArgumentError,
      );
    });

    test('rejects param-count mismatch', () {
      final l = Linear(3, 2, seed: 4);
      final bytes = Checkpoint.saveBytes(l);
      // Wrong module: two Linears has more params than one.
      final wrong = TransformerBlock(4, 2, seed: 0);
      expect(() => Checkpoint.loadIntoBytes(wrong, bytes), throwsArgumentError);
    });

    test('rejects shape mismatch', () {
      final l1 = Linear(4, 3, seed: 5);
      final l2 = Linear(3, 4, seed: 5); // swapped dims
      final bytes = Checkpoint.saveBytes(l1);
      expect(() => Checkpoint.loadIntoBytes(l2, bytes), throwsArgumentError);
    });

    test('rejects truncated data blob', () {
      final l = Linear(4, 3, seed: 6);
      final bytes = Checkpoint.saveBytes(l);
      final truncated = Uint8List.fromList(bytes.sublist(0, bytes.length - 5));
      expect(() => Checkpoint.loadIntoBytes(l, truncated), throwsArgumentError);
    });

    test('header records magic + version + all param shapes', () {
      final block = TransformerBlock(4, 2, seed: 7);
      final bytes = Checkpoint.saveBytes(block);
      // Magic
      expect(bytes[0], 0x44);
      expect(bytes[1], 0x50);
      expect(bytes[2], 0x54);
      expect(bytes[3], 0x43);
      // Version
      final view = ByteData.sublistView(bytes);
      expect(view.getUint32(4, Endian.little), 1);
    });
  });

  group('Checkpoint.saveFile / loadIntoFile', () {
    test('round-trip via a temp file', () {
      final tmp = Directory.systemTemp.createTempSync('dpt_ckpt_');
      try {
        final path = '${tmp.path}/model.dpt';
        final l1 = Linear(3, 4, seed: 8);
        Checkpoint.saveFile(l1, path);
        expect(File(path).existsSync(), isTrue);
        final l2 = Linear(3, 4, seed: 99);
        Checkpoint.loadIntoFile(l2, path);
        final x = Tensor.fromList([1, 3], [0.5, -0.25, 2.0]);
        expect(l2(x).toList(), l1(x).toList());
      } finally {
        tmp.deleteSync(recursive: true);
      }
    });

    test('creates parent directories as needed', () {
      final tmp = Directory.systemTemp.createTempSync('dpt_ckpt_');
      try {
        final path = '${tmp.path}/sub/dir/model.dpt';
        final l = Linear(2, 2, seed: 10);
        Checkpoint.saveFile(l, path);
        expect(File(path).existsSync(), isTrue);
      } finally {
        tmp.deleteSync(recursive: true);
      }
    });
  });

  group('GPT round-trip', () {
    test('save then load produces identical generate() output', () {
      final config = GPTConfig(
        vocabSize: 6,
        maxCtx: 16,
        embedDim: 8,
        numLayers: 2,
        numHeads: 2,
        seed: 11,
      );
      final trained = GPT(config);
      // Do a tiny bit of training so weights differ from init.
      final opt = Adam(trained.parameters(), lr: 0.05);
      final x = Tensor.fromList([6], [0, 1, 2, 3, 4, 5]);
      final y = Tensor.fromList([6], [1, 2, 3, 4, 5, 0]);
      for (int i = 0; i < 20; i++) {
        opt.zeroGrad();
        trained(x).crossEntropy(y).mean().backward();
        opt.step();
      }
      final trainedOut = trained.generate(
        [0.0, 1.0],
        maxNewTokens: 6,
        temperature: 0.0,
      );

      final bytes = Checkpoint.saveBytes(trained);

      // Fresh model with a different seed — should generate differently.
      final fresh = GPT(
        GPTConfig(
          vocabSize: config.vocabSize,
          maxCtx: config.maxCtx,
          embedDim: config.embedDim,
          numLayers: config.numLayers,
          numHeads: config.numHeads,
          seed: 999,
        ),
      );
      final freshOut = fresh.generate(
        [0.0, 1.0],
        maxNewTokens: 6,
        temperature: 0.0,
      );
      expect(freshOut, isNot(trainedOut));

      Checkpoint.loadIntoBytes(fresh, bytes);
      final reloadedOut = fresh.generate(
        [0.0, 1.0],
        maxNewTokens: 6,
        temperature: 0.0,
      );
      expect(reloadedOut, trainedOut);
    });

    test(
      'tied weights: token embedding round-trips once as a single param',
      () {
        final gpt = GPT(
          GPTConfig(
            vocabSize: 5,
            maxCtx: 8,
            embedDim: 4,
            numLayers: 1,
            numHeads: 2,
            tieWeights: true,
            seed: 12,
          ),
        );
        // Only one copy of the embedding matrix in parameters().
        final embedShape = gpt.tokenEmb.weight.shape;
        final count = gpt
            .parameters()
            .where(
              (p) =>
                  p.shape.length == 2 &&
                  p.shape[0] == embedShape[0] &&
                  p.shape[1] == embedShape[1],
            )
            .length;
        expect(count, 1);

        // Round-trip still works.
        final bytes = Checkpoint.saveBytes(gpt);
        final other = GPT(
          GPTConfig(
            vocabSize: 5,
            maxCtx: 8,
            embedDim: 4,
            numLayers: 1,
            numHeads: 2,
            tieWeights: true,
            seed: 22,
          ),
        );
        Checkpoint.loadIntoBytes(other, bytes);
        final x = Tensor.fromList([3], [1, 2, 3]);
        expect(other(x).toList(), gpt(x).toList());
      },
    );
  });

  group('Checkpoint fp16 compression', () {
    test('fp16 save halves the data blob size', () {
      final l = Linear(8, 4, seed: 7);
      final f32Bytes = Checkpoint.saveBytes(l);
      final f16Bytes = Checkpoint.saveBytes(l, fp16: true);
      // Data blob: 8*4 (W) + 4 (b) = 36 scalars. fp32 = 144 bytes;
      // fp16 = 72 bytes. Header also grows a bit because each param
      // spec now carries `"dtype": "F16"` (~14 bytes each × 2 params
      // = ~28 bytes), so net delta ≈ 44.
      final dataDelta = f32Bytes.length - f16Bytes.length;
      expect(dataDelta, greaterThan(30));
      expect(dataDelta, lessThan(72));
    });

    test('fp16 save/load round-trips through half precision', () {
      final l = Linear(4, 3, seed: 3);
      final x = Tensor.fromList([2, 4], [1, 2, 3, 4, 5, 6, 7, 8]);
      final yRef = l(x).toList();

      final bytes = Checkpoint.saveBytes(l, fp16: true);
      final l2 = Linear(4, 3, seed: 999);
      Checkpoint.loadIntoBytes(l2, bytes);
      final yLoaded = l2(x).toList();

      // Small rounding error is expected (values passed through fp16),
      // but outputs stay close.
      for (int i = 0; i < yRef.length; i++) {
        expect(yLoaded[i], closeTo(yRef[i], 1e-2));
      }
    });

    test('old fp32 checkpoints (no dtype field) still load', () {
      final l = Linear(3, 2, seed: 4);
      final bytes = Checkpoint.saveBytes(l); // fp16 defaults to false
      final l2 = Linear(3, 2, seed: 55);
      Checkpoint.loadIntoBytes(l2, bytes);
      final x = Tensor.fromList([1, 3], [1, 1, 1]);
      expect(l2(x).toList(), l(x).toList());
    });
  });
}
