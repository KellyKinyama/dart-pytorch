@Timeout(Duration(minutes: 5))
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

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
  group('WhisperMel', () {
    test('logMelFromSamples produces [80, 3000] fp32', () {
      final samples = Float32List(16000);
      for (int i = 0; i < samples.length; i++) {
        samples[i] = 0.1 * math.sin(2 * math.pi * 440 * i / 16000.0);
      }
      final mel = WhisperMel().logMelFromSamples(samples);
      expect(mel.length, equals(80 * 3000));

      // Sanity range: Whisper log-mel is roughly bounded near [-1, 1.2]
      // after the (x + 4) / 4 rescale.
      double mn = double.infinity, mx = -double.infinity;
      for (final v in mel) {
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
      expect(mn, greaterThanOrEqualTo(-1.5));
      expect(mx, lessThanOrEqualTo(1.5));
    });
  });

  group('WhisperEncoder (structural)', () {
    test('parameters() has expected count for tiny.en', () {
      final enc = WhisperEncoder(
        nMels: 80,
        embedDim: 384,
        numHeads: 6,
        numLayers: 4,
        nCtx: 1500,
      );
      // Per block: attn_ln(2) + mlp_ln(2) + q/k/v heads (2+1+2=5 per head)
      // * 6 heads = 30 + outProj(2) + mlp0(2) + mlp2(2) = 40.
      // Plus conv1(2) + conv2(2) + ln_post(2) = 6.
      expect(enc.parameters().length, equals(4 * 40 + 6));
    });

    test('sinusoidal PE has canonical Whisper values (t=0)', () {
      final enc = WhisperEncoder(
        nMels: 80,
        embedDim: 384,
        numHeads: 6,
        numLayers: 4,
        nCtx: 1500,
      );
      final pe = enc.positionalEmbedding.toFloat32List();
      // Row 0: sin(0) = 0 in the first half, cos(0) = 1 in the second half.
      for (int i = 0; i < 192; i++) {
        expect(pe[i], closeTo(0.0, 1e-6), reason: 'sin(0) at i=$i');
      }
      for (int i = 192; i < 384; i++) {
        expect(pe[i], closeTo(1.0, 1e-6), reason: 'cos(0) at i=$i');
      }
    });

    test('large-v3 config: 128 mels + 32 layers builds and matches shapes', () {
      final enc = WhisperEncoder(
        nMels: 128,
        embedDim: 1280,
        numHeads: 20,
        numLayers: 32,
        nCtx: 1500,
      );
      // conv1 weight is [outC=1280, inC=128, k=3]; conv2 [1280, 1280, 3].
      expect(enc.nMels, equals(128));
      expect(enc.embedDim, equals(1280));
      expect(enc.blocks.length, equals(32));
      expect(enc.blocks.first.numHeads, equals(20));
      expect(enc.blocks.first.headDim, equals(64));
      // Per block: attn_ln(2) + mlp_ln(2) + q/k/v heads (2+1+2=5 per head)
      // * 20 heads = 100 + outProj(2) + mlp0(2) + mlp2(2) = 110.
      // Plus conv1(2) + conv2(2) + ln_post(2) = 6.
      expect(enc.parameters().length, equals(32 * 110 + 6));
    });
  });

  group('WhisperDecoder (structural)', () {
    test('primeCrossAttn required before forward', () {
      final dec = WhisperDecoder(
        vocabSize: 100,
        embedDim: 32,
        numHeads: 4,
        numLayers: 1,
        nCtx: 8,
      );
      final tokens = Tensor.fromList([1, 2], [0.0, 1.0], device: Device.CPU);
      expect(() => dec.forward(tokens), throwsA(isA<StateError>()));
    });

    test('forward + logitsLastToken produce expected shapes', () {
      final dec = WhisperDecoder(
        vocabSize: 100,
        embedDim: 32,
        numHeads: 4,
        numLayers: 2,
        nCtx: 16,
      );
      final xa = Tensor.fill([1, 20, 32], 0.0, device: Device.CPU);
      dec.primeCrossAttn(xa);

      final tokens = Tensor.fromList(
        [1, 4],
        [0.0, 1.0, 2.0, 3.0],
        device: Device.CPU,
      );
      final hidden = dec.forward(tokens);
      expect(hidden.shape, equals([1, 4, 32]));

      final logits = dec.logitsLastToken(hidden);
      expect(logits.shape, equals([1, 100]));
    });

    test('T > nCtx throws', () {
      final dec = WhisperDecoder(
        vocabSize: 100,
        embedDim: 32,
        numHeads: 4,
        numLayers: 1,
        nCtx: 4,
      );
      dec.primeCrossAttn(Tensor.fill([1, 10, 32], 0.0, device: Device.CPU));
      final tokens = Tensor.fromList(
        [1, 5],
        [0.0, 1.0, 2.0, 3.0, 4.0],
        device: Device.CPU,
      );
      expect(() => dec.forward(tokens), throwsA(isA<ArgumentError>()));
    });
  });

  group('Whisper tiny.en end-to-end', () {
    final hasWeights = File(_weightsPath).existsSync();
    final hasTokenizer = File(_tokenizerPath).existsSync();
    final hasWav = File(_wavPath).existsSync();
    if (!hasWeights || !hasTokenizer || !hasWav) {
      test(
        'assets missing → skipped',
        () {},
        skip:
            'Fetch tiny.en weights + tokenizer + a wav under models/ '
            '(see bin/whisper_demo.dart for the paths).',
      );
      return;
    }

    late WhisperEncoder encoder;
    late WhisperDecoder decoder;
    late HFBpeTokenizer tokenizer;
    late Tensor memory;

    setUpAll(() async {
      encoder = WhisperEncoder(
        nMels: 80,
        embedDim: 384,
        numHeads: 6,
        numLayers: 4,
        nCtx: 1500,
      );
      final encReport = WhisperHFLoader.loadFile(encoder, _weightsPath);
      expect(encReport.unusedKeys, isEmpty);
      expect(encReport.consumedCount, equals(67));

      decoder = WhisperDecoder(
        vocabSize: 51864,
        embedDim: 384,
        numHeads: 6,
        numLayers: 4,
        nCtx: 448,
      );
      final decReport = WhisperHFLoader.loadDecoderFile(decoder, _weightsPath);
      expect(decReport.unusedKeys, isEmpty);
      expect(decReport.consumedCount, equals(100));

      tokenizer = HFBpeTokenizer.loadFile(_tokenizerPath);

      final mel = await WhisperMel().logMelFromFile(_wavPath);
      final melT = Tensor.fromFloat32List(
        [1, 80, 3000],
        mel,
        device: Device.CPU,
      );
      memory = encoder(melT);
      decoder.primeCrossAttn(memory);
    });

    test('encoder output has correct shape', () {
      expect(memory.shape, equals([1, 1500, 384]));
    });

    test('greedy decode reproduces HF token stream exactly', () {
      const sot = 50257;
      const eot = 50256;
      const noTs = 50362;
      const spaceTok = 220;

      // Reference obtained from HuggingFace WhisperForConditionalGeneration
      // with num_beams=1, do_sample=False on data/jfk.wav (11 s JFK
      // inaugural excerpt from openai/whisper's test assets).
      const refIds = <int>[
        843, 523, 616, 5891, 3399, 1265, 407, 644, 534, 1499, //
        460, 466, 329, 345, 11, 1265, 644, 345, 460, 466, //
        329, 534, 1499, 13,
      ];

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
