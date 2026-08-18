/// Loader for the `.dpt` weight bundles produced by
/// `scripts/extract_silero_vad.py`.
library;

import 'dart:io';
import 'dart:typed_data';

import 'silero_vad.dart';

class SileroVadReader {
  /// Load a `.dpt` file and populate every layer in [model].
  static void loadFile(SileroVad model, String path) {
    final bytes = File(path).readAsBytesSync();
    final bd = ByteData.sublistView(bytes);
    var off = 0;

    if (String.fromCharCodes(bytes.sublist(0, 4)) != 'SVAD') {
      throw StateError('$path: bad magic (expected SVAD)');
    }
    off += 4;
    final version = bd.getUint32(off, Endian.little);
    off += 4;
    if (version != 1) {
      throw StateError('$path: unsupported version $version');
    }
    final n = bd.getUint32(off, Endian.little);
    off += 4;

    final tensors = <String, Float32List>{};
    for (int i = 0; i < n; i++) {
      final nameLen = bd.getUint32(off, Endian.little);
      off += 4;
      final name = String.fromCharCodes(bytes.sublist(off, off + nameLen));
      off += nameLen;
      final ndim = bd.getUint32(off, Endian.little);
      off += 4;
      var count = 1;
      for (int d = 0; d < ndim; d++) {
        count *= bd.getInt32(off, Endian.little);
        off += 4;
      }
      final dtype = bd.getUint32(off, Endian.little);
      off += 4;
      if (dtype != 1) {
        throw StateError('$path: tensor $name has non-fp32 dtype $dtype');
      }
      final data = Float32List(count);
      for (int j = 0; j < count; j++) {
        data[j] = bd.getFloat32(off + j * 4, Endian.little);
      }
      tensors[name] = data;
      off += count * 4;
    }

    Float32List take(String key) {
      final t = tensors[key];
      if (t == null) throw StateError('$path: missing tensor $key');
      return t;
    }

    model.stftConv.loadFromPytorch(take('stft.forward_basis_buffer'), null);
    model.enc0.loadFromPytorch(
      take('encoder.0.reparam_conv.weight'),
      take('encoder.0.reparam_conv.bias'),
    );
    model.enc1.loadFromPytorch(
      take('encoder.1.reparam_conv.weight'),
      take('encoder.1.reparam_conv.bias'),
    );
    model.enc2.loadFromPytorch(
      take('encoder.2.reparam_conv.weight'),
      take('encoder.2.reparam_conv.bias'),
    );
    model.enc3.loadFromPytorch(
      take('encoder.3.reparam_conv.weight'),
      take('encoder.3.reparam_conv.bias'),
    );
    model.lstm.loadFromPytorch(
      weightIh: take('decoder.rnn.weight_ih'),
      weightHh: take('decoder.rnn.weight_hh'),
      biasIh: take('decoder.rnn.bias_ih'),
      biasHh: take('decoder.rnn.bias_hh'),
    );
    model.head.loadFromPytorch(
      take('decoder.decoder.2.weight'),
      take('decoder.decoder.2.bias'),
    );
  }
}
