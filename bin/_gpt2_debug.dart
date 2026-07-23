// Debug: load GPT-2 weights and trace intermediate stats to isolate the
// NaN. Not part of the public API.
library;

import 'dart:io';

import 'package:dart_pytorch/dart_pytorch.dart';

double _sum(List<double> xs) {
  var s = 0.0;
  for (final v in xs) {
    s += v;
  }
  return s;
}

double _mean(List<double> xs) => _sum(xs) / xs.length;

double _stddev(List<double> xs) {
  final m = _mean(xs);
  var s = 0.0;
  for (final v in xs) {
    s += (v - m) * (v - m);
  }
  return xs.length == 0 ? 0 : (s / xs.length);
}

int _nanCount(List<double> xs) {
  var c = 0;
  for (final v in xs) {
    if (v.isNaN || !v.isFinite) c++;
  }
  return c;
}

String _stats(String label, Tensor t) {
  final xs = t.toList();
  return '$label shape=${t.shape} n=${xs.length} '
      'mean=${_mean(xs).toStringAsFixed(6)} '
      'var=${_stddev(xs).toStringAsFixed(6)} '
      'nan=${_nanCount(xs)}';
}

void main(List<String> args) {
  final path = args.isEmpty ? 'models/gpt2/model.safetensors' : args.first;
  if (!File(path).existsSync()) {
    stderr.writeln('not found: $path');
    exit(1);
  }
  final cfg = GPT2HFLoader.gpt2SmallConfig();
  final gpt = GPT(cfg);
  stdout.writeln('loading ...');
  final r = GPT2HFLoader.loadFile(gpt, path);
  stdout.writeln('loaded: $r');

  // 1. Post-load param stats.
  stdout.writeln(_stats('wte           ', gpt.tokenEmb.weight));
  stdout.writeln(_stats('wpe           ', gpt.posEmb.table.weight));
  final b0 = gpt.encoder.blocks[0];
  stdout.writeln(_stats('block0.ln1.g  ', b0.ln1.gamma));
  stdout.writeln(_stats('block0.ln1.b  ', b0.ln1.beta));
  stdout.writeln(_stats('block0.wq[0].w', b0.mha.wq[0].weight));
  stdout.writeln(_stats('block0.wq[0].b', b0.mha.wq[0].bias!));
  stdout.writeln(_stats('block0.wo.w   ', b0.mha.wo.weight));
  stdout.writeln(_stats('block0.wo.b   ', b0.mha.wo.bias!));
  stdout.writeln(_stats('block0.ffn1.w ', b0.ffn1.weight));
  stdout.writeln(_stats('block0.ffn1.b ', b0.ffn1.bias!));
  stdout.writeln(_stats('block0.ffn2.w ', b0.ffn2.weight));
  stdout.writeln(_stats('block0.ffn2.b ', b0.ffn2.bias!));
  stdout.writeln(_stats('ln_f.gamma    ', gpt.encoder.finalNorm!.gamma));
  stdout.writeln(_stats('ln_f.beta     ', gpt.encoder.finalNorm!.beta));

  // 2. Trace forward step-by-step.
  gpt.eval();
  final tokens = Tensor.fromList([3], [464, 995, 318]);
  Tensor.noGrad(() {
    var h = gpt.tokenEmb(tokens);
    stdout.writeln(_stats('after tokenEmb', h));
    h = gpt.posEmb(h);
    stdout.writeln(_stats('after posEmb  ', h));

    // Feed through blocks one at a time, checking NaN.
    final mask = causalMask(3, device: h.device);
    for (int i = 0; i < cfg.numLayers; i++) {
      // Also trace sub-steps inside blocks 1..3 (where NaN appears).
      if (i >= 1 && i <= 3) {
        final blk = gpt.encoder.blocks[i];
        final normed1 = blk.ln1(h);
        final mhaOut = blk.mha(normed1, mask: mask);
        final afterMha = h + mhaOut;
        final normed2 = blk.ln2(afterMha);
        final ffnInner = blk.ffn1(normed2);
        // Manually apply GELU-tanh (same as inside block).
        const c = 0.7978845608028654;
        final geluOut =
            ffnInner *
                ((ffnInner + ffnInner.pow(3.0) * 0.044715) * c).tanh() *
                0.5 +
            ffnInner * 0.5;
        final ffnOut = blk.ffn2(geluOut);
        final blkOut = afterMha + ffnOut;
        final n1 = _nanCount(normed1.toList());
        final n2 = _nanCount(mhaOut.toList());
        final n3 = _nanCount(afterMha.toList());
        final n4 = _nanCount(ffnInner.toList());
        final n5 = _nanCount(geluOut.toList());
        final n6 = _nanCount(ffnOut.toList());
        final n7 = _nanCount(blkOut.toList());
        double _mx(List<double> xs) {
          var m = 0.0;
          for (final v in xs) {
            final a = v.isFinite ? v.abs() : double.infinity;
            if (a > m) m = a;
          }
          return m;
        }

        stdout.writeln(
          '  block $i sub-steps: ln1_nan=$n1 max=${_mx(normed1.toList()).toStringAsFixed(3)}, '
          'mha_nan=$n2 max=${_mx(mhaOut.toList()).toStringAsFixed(3)}, '
          'afterMha_nan=$n3 max=${_mx(afterMha.toList()).toStringAsFixed(3)}, '
          'ffn1_nan=$n4 max=${_mx(ffnInner.toList()).toStringAsFixed(3)}, '
          'gelu_nan=$n5 max=${_mx(geluOut.toList()).toStringAsFixed(3)}, '
          'ffn2_nan=$n6 max=${_mx(ffnOut.toList()).toStringAsFixed(3)}, '
          'total_nan=$n7 max=${_mx(blkOut.toList()).toStringAsFixed(3)}',
        );
        h = blkOut;
      } else {
        h = gpt.encoder.blocks[i](h, mask: mask);
      }
      final nan = _nanCount(h.toList());
      if (nan > 0) {
        stdout.writeln('  --> NaN at block $i. Stopping.');
        return;
      }
    }
    h = gpt.encoder.finalNorm!(h);
    stdout.writeln(_stats('after final LN', h));
    final logits = h.matmul(gpt.tokenEmb.weight.transpose());
    stdout.writeln(_stats('final logits  ', logits));
  });
}
