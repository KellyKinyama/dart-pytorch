/// Quick sanity check that HFBpeTokenizer parses whisper-tiny.en's
/// tokenizer.json and can round-trip a few IDs.
library;

import 'package:dart_pytorch/core/data/hf_bpe_tokenizer.dart';

void main() {
  final tk = HFBpeTokenizer.loadFile(
    'models/whisper-tiny.en/tokenizer.json',
  );
  print('vocab size (base):  ${tk.vocab.length}');

  // Expected special IDs.
  const sot = 50257;
  const eot = 50256;
  const notimestamps = 50362;

  // Round-trip a few IDs.
  for (final id in [sot, eot, notimestamps]) {
    final s = tk.decode([id]);
    print('  $id -> ${s.runes.map((r) => 'U+${r.toRadixString(16)}').join(",")}  as string: $s');
  }

  // Decode a plausible sequence: "SOT NOTIMESTAMPS  hello world EOT"
  final encoded = tk.encode(' hello world');
  print('encode(" hello world") = $encoded');
  final decoded = tk.decode([sot, notimestamps, ...encoded, eot]);
  print('decode(SOT + NT + " hello world" + EOT) = ${jsonEscape(decoded)}');
}

String jsonEscape(String s) {
  final sb = StringBuffer();
  for (final c in s.runes) {
    if (c < 0x20 || c > 0x7e) {
      sb.write('\\u${c.toRadixString(16).padLeft(4, "0")}');
    } else {
      sb.writeCharCode(c);
    }
  }
  return '"$sb"';
}
