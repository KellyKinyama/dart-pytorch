import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

void main() {
  group('ladder block selection', () {
    test('matches Needle\'s published 20-block selections', () {
      expect(ladderBlocks(20, 2), [0, 19]);
      expect(ladderBlocks(20, 4), [0, 9, 14, 19]);
      expect(ladderBlocks(20, 8), [0, 4, 6, 9, 11, 14, 16, 19]);
      expect(ladderBlocks(20, 20), [for (var i = 0; i < 20; i++) i]);
    });

    test('nested: each depth is a superset of the previous', () {
      for (var d = 2; d <= 20; d++) {
        final prev = ladderBlocks(20, d - 1).toSet();
        final cur = ladderBlocks(20, d).toSet();
        expect(prev.difference(cur), isEmpty, reason: 'depth $d not nested');
        expect(cur.length, d);
      }
    });

    test('always keeps both endpoints', () {
      for (final n in [4, 6, 12, 20]) {
        for (var d = 2; d <= n; d++) {
          final b = ladderBlocks(n, d);
          expect(b.first, 0);
          expect(b.last, n - 1);
        }
      }
    });

    test('small ladder (n=6)', () {
      expect(ladderBlocks(6, 1), [0]);
      expect(ladderBlocks(6, 2), [0, 5]);
      expect(ladderBlocks(6, 3), [0, 2, 5]);
      expect(ladderBlocks(6, 6), [0, 1, 2, 3, 4, 5]);
    });
  });
}
