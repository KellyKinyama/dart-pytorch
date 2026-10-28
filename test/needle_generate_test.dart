import 'package:dart_pytorch/core/nn/needle/needle_generate.dart';
import 'package:test/test.dart';

void main() {
  group('generateGreedy', () {
    test('argmax picks the highest-logit token', () {
      // logits always favour token 3
      List<double> fn(List<int> ctx) =>
          [for (var i = 0; i < 5; i++) (i == 3 ? 10.0 : 0.0)];
      final out = generateGreedy(
        promptIds: [0],
        vocab: 5,
        lastLogits: fn,
        maxNewTokens: 4,
      );
      expect(out, [3, 3, 3, 3]);
    });

    test('forceFirst overrides the first token', () {
      List<double> fn(List<int> ctx) =>
          [for (var i = 0; i < 5; i++) (i == 3 ? 10.0 : 0.0)];
      final out = generateGreedy(
        promptIds: [0],
        vocab: 5,
        lastLogits: fn,
        maxNewTokens: 3,
        forceFirst: 1,
      );
      expect(out.first, 1);
      expect(out.skip(1), [3, 3]);
    });

    test('stops at a stop id', () {
      // emit token = context length, so sequence grows 1,2,3,...
      List<double> fn(List<int> ctx) {
        final v = List<double>.filled(10, 0);
        v[ctx.length] = 10.0;
        return v;
      }

      final out = generateGreedy(
        promptIds: [0],
        vocab: 10,
        lastLogits: fn,
        maxNewTokens: 20,
        stopIds: {4},
      );
      expect(out, [1, 2, 3, 4]); // stops when 4 is produced
    });

    test('respects maxNewTokens', () {
      List<double> fn(List<int> ctx) => List<double>.filled(4, 1.0);
      final out = generateGreedy(
        promptIds: [0],
        vocab: 4,
        lastLogits: fn,
        maxNewTokens: 6,
      );
      expect(out.length, 6);
    });
  });

  test('argmax helper', () {
    expect(argmax([1.0, 9.0, 3.0]), 1);
    expect(argmax([5.0, 2.0, 2.0]), 0);
  });
}
