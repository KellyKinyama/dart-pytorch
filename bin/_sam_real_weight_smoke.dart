/// SAM real-weight smoke test.
///
/// Loads Meta's official SAM ViT-B weights (converted to safetensors)
/// into `SamImageEncoder + SamPromptEncoder + SamMaskDecoder` and
/// reports which HF keys were consumed and which are unused. Skips
/// the expensive 1024x1024 forward pass — the goal here is to verify
/// the loader path end-to-end against real weights.
///
///   dart run bin/_sam_real_weight_smoke.dart \
///     --weights models/sam-vit-b/sam_vit_b.safetensors
library;

import 'dart:io';

import 'package:dart_pytorch/dart_pytorch.dart';

Future<void> main(List<String> args) async {
  var weightsPath = 'models/sam-vit-b/sam_vit_b.safetensors';
  for (int i = 0; i < args.length; i++) {
    if (args[i] == '--weights') weightsPath = args[++i];
  }
  if (!File(weightsPath).existsSync()) {
    stderr.writeln('missing: $weightsPath');
    exit(2);
  }

  print('SAM ViT-B real-weight smoke test');
  print('weights: $weightsPath  (${File(weightsPath).lengthSync()} bytes)');
  print('');

  print('Building modules ...');
  final swBuild = Stopwatch()..start();
  final imgEnc = SamImageEncoder(SamImageEncoderConfig.vitB());
  final promptEnc = SamPromptEncoder(
    embedDim: 256,
    imageEmbedH: 64,
    imageEmbedW: 64,
    imageSize: 1024,
    maskInSize: 256,
    maskInputChannels: 16,
  );
  final maskDec = SamMaskDecoder(const SamMaskDecoderConfig());
  swBuild.stop();
  print('  ${swBuild.elapsedMilliseconds} ms');

  print('');
  print('Loading weights ...');
  final swLoad = Stopwatch()..start();
  final report = SamHFLoader.loadFile(
    imageEncoder: imgEnc,
    promptEncoder: promptEnc,
    maskDecoder: maskDec,
    path: weightsPath,
  );
  swLoad.stop();
  print('  ${swLoad.elapsedMilliseconds} ms');
  print('  $report');
  if (report.unusedKeys.isNotEmpty) {
    print('  unused (first 10):');
    for (final k in report.unusedKeys.take(10)) {
      print('    - $k');
    }
  }

  final params = [
    ...imgEnc.parameters(),
    ...promptEnc.parameters(),
    ...maskDec.parameters(),
  ];
  var totalScalars = 0;
  for (final p in params) {
    var n = 1;
    for (final d in p.shape) {
      n *= d;
    }
    totalScalars += n;
  }
  print('');
  print('module params: ${params.length}');
  print(
    'scalar count : $totalScalars '
    '(~${(totalScalars * 4 / 1e6).toStringAsFixed(1)} MB @ fp32)',
  );

  if (report.unusedKeys.isNotEmpty) {
    exit(1);
  }
  print('');
  print('OK — all keys consumed.');
}
