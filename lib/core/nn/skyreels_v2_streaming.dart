/// Layer-streaming inference for [SkyReelsV2Model].
///
/// Same shape as [LlamaStreamingRunner]: one resident
/// [SkyReelsV2Block] whose weight storage is `adoptCpuStorageFrom`'d
/// per layer from a [ShardedSafeTensorsReader]. Everything else
/// (patch embed, text embed, time embed, output head) stays
/// resident.
///
/// Because this port is aspirational, this runner won't produce
/// real video — see [skyreels_v2.dart] for the list of unimplemented
/// pieces. Its job is to prove the streaming pattern works against a
/// video-DiT tensor layout: a full forward on random weights with
/// finite outputs.
library;

import 'dart:typed_data';

import '../tensor/tensor.dart';
import '../tensor/dtype.dart';
import 'skyreels_v2.dart';
import 'safetensors_reader.dart';

class SkyReelsV2StreamingRunner {
  final SkyReelsV2Config config;
  final ShardedSafeTensorsReader reader;
  final bool keepFp16;
  final bool profile;

  final SkyReelsV2Model model;

  SkyReelsV2StreamingRunner(
    this.config,
    this.reader, {
    this.keepFp16 = true,
    this.profile = false,
  }) : model = SkyReelsV2Model(config) {
    if (config.device != Device.CPU) {
      throw StateError(
        'SkyReelsV2StreamingRunner: only CPU is supported '
        '(got ${config.device})',
      );
    }
    // Load persistent (non-block) tensors once. The reader is
    // permissive: if a key is missing (e.g. we generated a random
    // ckpt), we skip and keep the fp32 random init.
    _loadPersistentIfPresent();
  }

  void _loadPersistentIfPresent() {
    void tryLoad(String name, Tensor dst) {
      if (!reader.contains(name)) return;
      final src = reader.readTensor(name, keepFp16: keepFp16);
      if (src.length != dst.length) return;
      if (src.dtype == DType.fp16 && dst.device == Device.CPU) {
        dst.adoptCpuStorageFrom(src);
      } else {
        dst.assign(
          Tensor.fromList(dst.shape, src.toList(), device: dst.device),
        );
      }
    }

    tryLoad('patch_embedding.weight', model.patchEmbed.weight);
    tryLoad('patch_embedding.bias', model.patchEmbed.bias!);
    tryLoad('text_embedding.0.weight', model.textEmbed0.weight);
    tryLoad('text_embedding.0.bias', model.textEmbed0.bias!);
    tryLoad('text_embedding.2.weight', model.textEmbed2.weight);
    tryLoad('text_embedding.2.bias', model.textEmbed2.bias!);
    tryLoad('time_embedding.0.weight', model.timeEmbed0.weight);
    tryLoad('time_embedding.0.bias', model.timeEmbed0.bias!);
    tryLoad('time_embedding.2.weight', model.timeEmbed2.weight);
    tryLoad('time_embedding.2.bias', model.timeEmbed2.bias!);
    tryLoad('time_projection.1.weight', model.timeProjection.weight);
    tryLoad('time_projection.1.bias', model.timeProjection.bias!);
    tryLoad('head.modulation', model.headModulation);
    tryLoad('head.head.weight', model.headOut.weight);
    tryLoad('head.head.bias', model.headOut.bias!);
  }

  /// Swap block [i]'s weights into the *first* resident block. When
  /// this runner is used for a real forward you must call this
  /// yourself before consuming that block — the streaming forward
  /// below does it.
  void _swapBlock(int i) {
    if (i < 0 || i >= config.numLayers) {
      throw ArgumentError('layer $i out of range [0, ${config.numLayers})');
    }
    final sw = profile ? (Stopwatch()..start()) : null;
    final block = model.blocks[0];
    final p = 'blocks.$i';

    _load('$p.modulation', block.modulation);

    _loadLinear('$p.self_attn', block.selfAttn);
    _loadLinear('$p.cross_attn', block.crossAttn);
    _load('$p.norm3.weight', block.norm3.gamma);
    _load('$p.norm3.bias', block.norm3.beta);
    _load('$p.ffn.0.weight', block.ffn1.weight);
    _load('$p.ffn.0.bias', block.ffn1.bias!);
    _load('$p.ffn.2.weight', block.ffn2.weight);
    _load('$p.ffn.2.bias', block.ffn2.bias!);

    if (sw != null) {
      sw.stop();
      // ignore: avoid_print
      print('  [block $i] swap ${sw.elapsedMilliseconds} ms');
    }
  }

  void _load(String name, Tensor dst) {
    if (!reader.contains(name)) return;
    final src = reader.readTensor(name, keepFp16: keepFp16);
    if (src.length != dst.length) return;
    if (src.dtype == DType.fp16 && dst.device == Device.CPU) {
      dst.adoptCpuStorageFrom(src);
    } else {
      dst.assign(Tensor.fromList(dst.shape, src.toList(), device: dst.device));
    }
  }

  void _loadLinear(String prefix, dynamic attn) {
    // MultiHeadAttention and MultiHeadCrossAttention both expose
    // per-head wq/wk/wv Linears and a single wo. In HF checkpoints
    // Q/K/V are typically stored as one Linear each (not per-head);
    // we don't attempt to split here because the aspirational demo
    // uses random weights, not a HF checkpoint. A real port would
    // slice `[dim, dim] -> [numHeads, [headDim, dim]]` here.
    for (int h = 0; h < attn.wq.length; h++) {
      _load('$prefix.q.weight_h$h', attn.wq[h].weight);
      if (attn.wq[h].bias != null) {
        _load('$prefix.q.bias_h$h', attn.wq[h].bias!);
      }
    }
    for (int h = 0; h < attn.wk.length; h++) {
      _load('$prefix.k.weight_h$h', attn.wk[h].weight);
      if (attn.wk[h].bias != null) {
        _load('$prefix.k.bias_h$h', attn.wk[h].bias!);
      }
    }
    for (int h = 0; h < attn.wv.length; h++) {
      _load('$prefix.v.weight_h$h', attn.wv[h].weight);
      if (attn.wv[h].bias != null) {
        _load('$prefix.v.bias_h$h', attn.wv[h].bias!);
      }
    }
    _load('$prefix.o.weight', attn.wo.weight);
    if (attn.wo.bias != null) {
      _load('$prefix.o.bias', attn.wo.bias!);
    }
  }

  /// One forward. `patches` is `[N, in_dim * prod(patch_size)]`,
  /// `text` is `[textLen, textDim]`, `step` is a diffusion timestep.
  /// Streams each block off disk before running it. The first block
  /// in `model.blocks` is the resident one; blocks[1..L-1] are only
  /// used as placeholders so `SkyReelsV2Model.call` can iterate the
  /// list. Their fp32 storage sits idle — that's a memory cost this
  /// port doesn't try to eliminate (see the streaming doc).
  Tensor forward(Tensor patches, Tensor text, double step) {
    return Tensor.noGrad(() {
      final ctx = model.textEmbed2(geluTanh(model.textEmbed0(text)));
      final tSin = model.sinusoidalTimestep(step);
      final tEmb = model.timeEmbed2(silu(model.timeEmbed0(tSin)));
      final e0 = model.timeProjection(silu(tEmb));

      var x = model.patchEmbed(patches);
      final resident = model.blocks[0];
      for (int i = 0; i < config.numLayers; i++) {
        _swapBlock(i);
        final e6 = _computeE6(e0, resident.modulation);
        x = resident(x, e6, ctx);
      }

      // Head modulation.
      final modList = model.headModulation.toList();
      final tList = tEmb.toList();
      final dim = config.dim;
      final shift = List<double>.filled(dim, 0.0);
      final scale = List<double>.filled(dim, 0.0);
      for (int i = 0; i < dim; i++) {
        shift[i] = modList[i] + tList[i];
        scale[i] = modList[dim + i] + tList[i];
      }
      final shiftT = Tensor.fromList([1, dim], shift, device: config.device);
      final scaleT = Tensor.fromList([1, dim], scale, device: config.device);
      final h = model.headNorm(x) * (scaleT + 1.0) + shiftT;
      return model.headOut(h);
    });
  }

  List<Tensor> _computeE6(Tensor e0, Tensor blockMod) {
    final e0List = e0.toList();
    final modList = blockMod.toList();
    final dim = config.dim;
    final out = <Tensor>[];
    for (int k = 0; k < 6; k++) {
      final slice = Float32List(dim);
      for (int i = 0; i < dim; i++) {
        slice[i] = e0List[k * dim + i] + modList[k * dim + i];
      }
      out.add(Tensor.fromList([1, dim], slice, device: config.device));
    }
    return out;
  }

  void close() => reader.close();
}
