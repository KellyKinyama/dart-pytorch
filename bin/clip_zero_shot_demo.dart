/// CLIP zero-shot image classification demo.
///
/// Loads a joint OpenAI CLIP checkpoint (`openai/clip-vit-base-patch32`
/// or `-patch16`), encodes the input image through the vision tower
/// and each candidate label ("a photo of a {label}") through the text
/// tower, projects both towers into the shared 512-d contrastive
/// space, L2-normalizes, and picks the label with the highest cosine
/// similarity.
///
///   dart run bin/clip_zero_shot_demo.dart \
///       --image path/to/photo.jpg \
///       --labels "dog,cat,car,plane,face,sunset"
///
/// One-time weight download (~350 MB fp32):
///   mkdir -p models/clip-vit-base-patch32
///   for f in model.safetensors tokenizer.json; do
///     curl -L -o "models/clip-vit-base-patch32/$f" \
///       "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/$f"
///   done
///
/// The demo uses OpenAI's standard "a photo of a {label}" template.
/// For better accuracy on domain-specific tasks, wrap with a task
/// prompt (`--template "a satellite photo of a {}"`), or ensemble
/// multiple prompts by averaging their text embeddings.
library;

import 'dart:io';
import 'dart:math' as math;

import 'package:image/image.dart' as img_pkg;
import 'package:dart_pytorch/dart_pytorch.dart';

const _weightsDefault = 'models/clip-vit-base-patch32/model.safetensors';
const _tokenizerDefault = 'models/clip-vit-base-patch32/tokenizer.json';

// OpenAI CLIP BPE tokenizer specials.
const int _bos = 49406; // <|startoftext|>
const int _eos = 49407; // <|endoftext|>

Future<void> main(List<String> args) async {
  String? imagePath;
  var weightsPath = _weightsDefault;
  var tokenizerPath = _tokenizerDefault;
  var labelsCsv = 'dog,cat,car,plane,face,sunset,mountain,pizza';
  var template = 'a photo of a {}';
  var topK = 5;
  var maxCtx = 77;
  for (int i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--image':
        imagePath = args[++i];
        break;
      case '--weights':
        weightsPath = args[++i];
        break;
      case '--tokenizer':
        tokenizerPath = args[++i];
        break;
      case '--labels':
        labelsCsv = args[++i];
        break;
      case '--template':
        template = args[++i];
        break;
      case '--topk':
        topK = int.parse(args[++i]);
        break;
    }
  }
  if (imagePath == null) {
    stderr.writeln(
      'usage: dart run bin/clip_zero_shot_demo.dart '
      '--image PATH [--labels LABEL1,LABEL2,...] [--topk N]',
    );
    exit(64);
  }
  for (final p in [imagePath, weightsPath, tokenizerPath]) {
    if (!File(p).existsSync()) {
      stderr.writeln('missing: $p');
      exit(2);
    }
  }
  final labels = labelsCsv.split(',').map((s) => s.trim()).toList();
  if (topK > labels.length) topK = labels.length;

  // ---------- Build models ----------
  final visionCfg = ClipHFLoader.base32Config();
  final textCfg = ClipHFLoader.baseTextConfig();
  print(
    'Building CLIP-ViT-B/32 (vision hidden=${visionCfg.embedDim}, '
    'text hidden=${textCfg.embedDim})',
  );
  final visionModel = CLIPVisionModel(visionCfg);
  final textModel = CLIPTextModel(textCfg);

  print('Loading vision tower from $weightsPath');
  final visionRep = ClipHFLoader.loadFile(visionModel, weightsPath);
  print('  $visionRep');
  print('Loading text tower');
  final textRep = ClipHFLoader.loadTextFile(textModel, weightsPath);
  print('  $textRep');
  final proj = ClipHFLoader.loadProjections(weightsPath);
  if (proj == null) {
    stderr.writeln(
      'error: no visual_projection.weight / text_projection.weight '
      'in $weightsPath — need a joint CLIPModel checkpoint (not the '
      'vision-only file)',
    );
    exit(3);
  }
  print(
    '  projections: projDim=${proj.projDim} '
    'visionHidden=${proj.visionHidden} textHidden=${proj.textHidden}',
  );

  final tokenizer = HFBpeTokenizer.loadFile(tokenizerPath);

  // ---------- Encode image ----------
  print('');
  print('Preprocessing $imagePath');
  final patch = _patchify(imagePath, visionCfg);
  print('Vision forward');
  final swV = Stopwatch()..start();
  final visionSeq = visionModel(patch); // [1 + numPatches, D]
  swV.stop();
  final cls = _firstRow(visionSeq, visionCfg.embedDim);
  final imageEmbed = _project(cls, proj.visualProjection);
  final imageNorm = _l2Normalize(imageEmbed);
  print('  ${swV.elapsedMilliseconds} ms  → ${imageNorm.length}-d embedding');

  // ---------- Encode labels ----------
  print('');
  print('Encoding ${labels.length} labels via template "$template"');
  final labelEmbeds = <List<double>>[];
  final swT = Stopwatch()..start();
  for (final label in labels) {
    final prompt = template.replaceAll('{}', label).toLowerCase();
    final bpeIds = tokenizer.encode(prompt);
    final wrapped = <int>[_bos, ...bpeIds, _eos];
    if (wrapped.length > maxCtx) {
      throw StateError(
        'label "$label" tokenises to ${wrapped.length} '
        'ids (> maxCtx=$maxCtx)',
      );
    }
    final tokens = Tensor.fromList([
      wrapped.length,
    ], wrapped.map((i) => i.toDouble()).toList());
    final pooled = textModel.pooledEmbedding(tokens); // argmax token
    final projected = _project(pooled.toList(), proj.textProjection);
    labelEmbeds.add(_l2Normalize(projected));
  }
  swT.stop();
  print('  ${swT.elapsedMilliseconds} ms');

  // ---------- Similarity + softmax ----------
  final logitScaleVal = proj.logitScale == null
      ? math.exp(math.log(100.0))
      : math.exp(proj.logitScale!.toList()[0]);
  final rawSims = <double>[];
  for (final le in labelEmbeds) {
    double dot = 0;
    for (int i = 0; i < imageNorm.length; i++) {
      dot += imageNorm[i] * le[i];
    }
    rawSims.add(dot);
  }
  // Softmax with temperature.
  final scaled = rawSims.map((s) => s * logitScaleVal).toList();
  var maxVal = scaled.reduce(math.max);
  final expd = scaled.map((s) => math.exp(s - maxVal)).toList();
  final sumExp = expd.reduce((a, b) => a + b);
  final probs = expd.map((v) => v / sumExp).toList();

  // ---------- Top-K report ----------
  final ranked = List<(String, double, double)>.generate(
    labels.length,
    (i) => (labels[i], probs[i], rawSims[i]),
  )..sort((a, b) => b.$2.compareTo(a.$2));

  print('');
  print('== top-$topK ==');
  print('  ${'label'.padRight(20)}  ${'prob'.padLeft(8)}  ${'cos'.padLeft(8)}');
  for (int i = 0; i < topK; i++) {
    final (label, prob, cos) = ranked[i];
    print(
      '  ${label.padRight(20)}  '
      '${(prob * 100).toStringAsFixed(2).padLeft(7)}%  '
      '${cos.toStringAsFixed(4).padLeft(8)}',
    );
  }
}

Tensor _patchify(String path, CLIPVisionConfig cfg) {
  final bytes = File(path).readAsBytesSync();
  final decoded = img_pkg.decodeImage(bytes);
  if (decoded == null) {
    throw StateError('could not decode $path');
  }
  final resized = img_pkg.copyResize(
    decoded,
    width: cfg.imageSize,
    height: cfg.imageSize,
    interpolation: img_pkg.Interpolation.linear,
  );
  const meanR = 0.48145466, meanG = 0.4578275, meanB = 0.40821073;
  const stdR = 0.26862954, stdG = 0.26130258, stdB = 0.27577711;
  final p = cfg.patchSize;
  final rowStride = p * p * 3;
  final data = List<double>.filled(cfg.numPatches * rowStride, 0.0);
  final patchesX = cfg.imageSize ~/ p;
  final patchesY = cfg.imageSize ~/ p;
  for (int py = 0; py < patchesY; py++) {
    for (int px = 0; px < patchesX; px++) {
      final patchIdx = py * patchesX + px;
      final base = patchIdx * rowStride;
      for (int y = 0; y < p; y++) {
        for (int x = 0; x < p; x++) {
          final pix = resized.getPixel(px * p + x, py * p + y);
          final r = pix.rNormalized.toDouble();
          final g = pix.gNormalized.toDouble();
          final b = pix.bNormalized.toDouble();
          final off = base + (y * p + x) * 3;
          data[off + 0] = (r - meanR) / stdR;
          data[off + 1] = (g - meanG) / stdG;
          data[off + 2] = (b - meanB) / stdB;
        }
      }
    }
  }
  return Tensor.fromList([cfg.numPatches, rowStride], data);
}

List<double> _firstRow(Tensor t, int d) {
  final vals = t.toList();
  return vals.sublist(0, d);
}

/// `x @ W.T` — projects a `[D]` vector by a `[projDim, D]` matrix
/// into a `[projDim]` output. Done on host to avoid needing extra
/// Tensor manipulation for a one-off matmul.
List<double> _project(List<double> x, Tensor w) {
  final projDim = w.shape[0];
  final d = w.shape[1];
  if (x.length != d) {
    throw ArgumentError('_project: dim mismatch ${x.length} vs $d');
  }
  final wVals = w.toList();
  final out = List<double>.filled(projDim, 0.0);
  for (int i = 0; i < projDim; i++) {
    double acc = 0;
    for (int j = 0; j < d; j++) {
      acc += wVals[i * d + j] * x[j];
    }
    out[i] = acc;
  }
  return out;
}

List<double> _l2Normalize(List<double> v) {
  double sq = 0;
  for (final x in v) {
    sq += x * x;
  }
  final n = math.sqrt(sq).clamp(1e-12, double.infinity);
  return v.map((x) => x / n).toList();
}
