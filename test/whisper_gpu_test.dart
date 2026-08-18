@Timeout(Duration(minutes: 5))
library;

import 'dart:io';

import 'package:dart_pytorch/core/audio/whisper_mel.dart';
import 'package:dart_pytorch/core/data/hf_bpe_tokenizer.dart';
import 'package:dart_pytorch/core/nn/whisper.dart';
import 'package:dart_pytorch/core/nn/whisper_decoder.dart';
import 'package:dart_pytorch/core/nn/whisper_hf_loader.dart';
import 'package:dart_pytorch/core/tensor/tensor.dart';
import 'package:test/test.dart';

const _weightsPath = 'models/whisper-tiny.en/model.safetensors';
const _tokenizerPath = 'models/whisper-tiny.en/tokenizer.json';
const _wavPath = 'data/jfk.wav';

void main() {
  group('Whisper tiny.en on GPU', () {
    final hasWeights = File(_weightsPath).existsSync();
    final hasTokenizer = File(_tokenizerPath).existsSync();
    final hasWav = File(_wavPath).existsSync();
    if (!hasWeights || !hasTokenizer || !hasWav) {
      test(
        'assets missing → skipped',
        () {},
        skip: 'Fetch tiny.en weights + tokenizer + data/jfk.wav first.',
      );
      return;
    }

    test('GPU pipeline reproduces HF token stream exactly', () async {
      const device = Device.GPU;
      const sot = 50257;
      const eot = 50256;
      const noTs = 50362;
      const spaceTok = 220;
      const refIds = <int>[
        843, 523, 616, 5891, 3399, 1265, 407, 644, 534, 1499, //
        460, 466, 329, 345, 11, 1265, 644, 345, 460, 466, //
        329, 534, 1499, 13,
      ];

      final encoder = WhisperEncoder(
        nMels: 80,
        embedDim: 384,
        numHeads: 6,
        numLayers: 4,
        nCtx: 1500,
        device: device,
      );
      final encReport = WhisperHFLoader.loadFile(encoder, _weightsPath);
      expect(encReport.unusedKeys, isEmpty);

      final decoder = WhisperDecoder(
        vocabSize: 51864,
        embedDim: 384,
        numHeads: 6,
        numLayers: 4,
        nCtx: 448,
        device: device,
      );
      final decReport = WhisperHFLoader.loadDecoderFile(decoder, _weightsPath);
      expect(decReport.unusedKeys, isEmpty);

      final tokenizer = HFBpeTokenizer.loadFile(_tokenizerPath);
      final mel = await WhisperMel().logMelFromFile(_wavPath);
      final melT = Tensor.fromFloat32List([1, 80, 3000], mel, device: device);
      final memory = encoder(melT);
      expect(memory.shape, equals([1, 1500, 384]));
      decoder.primeCrossAttn(memory);

      final tokens = decoder.greedyDecode(
        startTokens: [sot, noTs],
        eot: eot,
        maxLen: 100,
        initialSuppress: [spaceTok, eot],
      );
      final sampled = tokens.sublist(2);
      expect(sampled, equals(refIds));

      final text = tokenizer.decode(sampled);
      expect(
        text,
        equals(
          ' And so my fellow Americans ask not what your country can do '
          'for you, ask what you can do for your country.',
        ),
      );
    });
  });
}
