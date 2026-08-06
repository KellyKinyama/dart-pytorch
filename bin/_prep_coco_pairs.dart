/// Pull ~N (image, caption) pairs from COCO val2017 into a folder
/// layout consumable by `bin/train_llava_projector.dart`
/// (`img.jpg` + matching `img.txt`).
///
/// Usage:
///   dart run bin/_prep_coco_pairs.dart --n 200 \
///     --json data/coco_val_prep/annotations/captions_val2017.json \
///     --out data/coco_val200
library;

import 'dart:convert';
import 'dart:io';

Future<void> main(List<String> args) async {
  var n = 200;
  var jsonPath = 'data/coco_val_prep/annotations/captions_val2017.json';
  var outDir = 'data/coco_val200';
  for (var i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--n':
        n = int.parse(args[++i]);
      case '--json':
        jsonPath = args[++i];
      case '--out':
        outDir = args[++i];
    }
  }

  stdout.writeln('reading $jsonPath');
  final root =
      jsonDecode(File(jsonPath).readAsStringSync()) as Map<String, dynamic>;
  final imgs = (root['images'] as List).cast<Map<String, dynamic>>();
  final anns = (root['annotations'] as List).cast<Map<String, dynamic>>();

  // first caption per image_id
  final firstCap = <int, String>{};
  for (final a in anns) {
    final id = (a['image_id'] as num).toInt();
    if (firstCap.containsKey(id)) continue;
    firstCap[id] = (a['caption'] as String).trim();
  }

  final byId = {for (final im in imgs) (im['id'] as num).toInt(): im};

  // stable sample: sort by id, take first N with a caption
  final sortedIds = byId.keys.toList()..sort();
  final chosen = <int>[];
  for (final id in sortedIds) {
    if (chosen.length >= n) break;
    if (firstCap.containsKey(id)) chosen.add(id);
  }
  stdout.writeln('selected ${chosen.length} pairs from ${byId.length} images');

  final out = Directory(outDir);
  if (!out.existsSync()) out.createSync(recursive: true);

  final client = HttpClient();
  var ok = 0;
  var failed = 0;
  final sw = Stopwatch()..start();
  for (var i = 0; i < chosen.length; i++) {
    final id = chosen[i];
    final meta = byId[id]!;
    final fname = meta['file_name'] as String; // e.g. 000000000139.jpg
    final url = Uri.parse('http://images.cocodataset.org/val2017/$fname');
    final imgPath = '${out.path}/$fname';
    final txtPath = '${out.path}/${fname.replaceAll('.jpg', '.txt')}';
    if (File(imgPath).existsSync() && File(txtPath).existsSync()) {
      ok++;
      continue;
    }
    try {
      final req = await client.getUrl(url);
      final resp = await req.close();
      if (resp.statusCode != 200) {
        failed++;
        stdout.writeln('  [skip] $fname http ${resp.statusCode}');
        continue;
      }
      final bytes = <int>[];
      await for (final chunk in resp) {
        bytes.addAll(chunk);
      }
      File(imgPath).writeAsBytesSync(bytes);
      File(txtPath).writeAsStringSync(firstCap[id]!);
      ok++;
      if ((i + 1) % 20 == 0) {
        stdout.writeln(
          '  [$i/${chosen.length}] '
          'ok=$ok fail=$failed elapsed=${sw.elapsed.inSeconds}s',
        );
      }
    } catch (e) {
      failed++;
      stdout.writeln('  [fail] $fname $e');
    }
  }
  client.close();
  stdout.writeln('done. ok=$ok fail=$failed in ${sw.elapsed.inSeconds}s');
}
