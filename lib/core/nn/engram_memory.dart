/// Engram n-gram memory — a trainable lookup memory keyed by hashed n-grams
/// of the recent tokens, after Cactus Needle's engram (where such tables hold
/// most of the model's parameters as cheap, gather-read facts).
///
/// For each configured n-gram [orders] (e.g. 2- and 3-grams), the last `order`
/// token ids at every position are folded into an FNV-style 32-bit hash and
/// mapped into a learnable table of [slots] rows. The gathered rows are summed
/// across orders, giving a `[N, dim]` contribution the caller adds into the
/// residual stream. Only the tables learn; the hash is a constant function of
/// the tokens, so gradients flow to the looked-up rows exactly like an
/// [Embedding].
library;

import 'dart:typed_data';

import '../tensor/tensor.dart';
import 'embedding.dart';
import 'module.dart';

const int _fnvSeed = 0x9E3779B9;
const int _fnvPrime = 0x01000193;
const int _mask32 = 0xFFFFFFFF;

class EngramMemory extends Module {
  final int dim;
  final int slots;
  final List<int> orders;

  /// One learnable table per n-gram order, shape `[slots, dim]`.
  final List<Embedding> tables;

  EngramMemory({
    required this.dim,
    this.slots = 8192,
    this.orders = const [2, 3],
    Device device = Device.CPU,
    int seed = 0,
  }) : tables = [
          for (var i = 0; i < orders.length; i++)
            Embedding(slots, dim, device: device, seed: seed + 1000 * i),
        ];

  /// `ids` is a 1D `[N]` tensor of token ids; returns the `[N, dim]` memory
  /// contribution.
  Tensor call(Tensor ids) {
    final toks = [for (final v in ids.toList()) v.round()];
    final n = toks.length;
    Tensor? acc;
    for (var oi = 0; oi < orders.length; oi++) {
      final order = orders[oi];
      final idx = Float32List(n);
      for (var t = 0; t < n; t++) {
        var h = (_fnvSeed * (oi + 1)) & _mask32;
        for (var j = 0; j < order; j++) {
          final u = (t - j >= 0) ? (toks[t - j] & _mask32) : 0;
          h = ((h ^ u) * _fnvPrime) & _mask32;
        }
        h ^= h >> 15;
        idx[t] = (h % slots).toDouble();
      }
      final g = tables[oi](Tensor.fromList([n], idx, device: ids.device));
      acc = acc == null ? g : acc + g;
    }
    return acc!;
  }

  @override
  List<Tensor> parameters() => [for (final t in tables) ...t.parameters()];

  @override
  List<Module> submodules() => [...tables];
}
