@Timeout(Duration(minutes: 5))
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

/// Build a torchvision-style ResNet-50 state_dict populated with tiny
/// deterministic pseudo-random values. Enough to exercise the loader
/// end-to-end without downloading the 100 MB checkpoint.
Map<String, Tensor> _synthState({int seed = 0}) {
  final rng = math.Random(seed);
  Tensor rand(List<int> shape) {
    var n = 1;
    for (final d in shape) {
      n *= d;
    }
    final v = Float32List(n);
    for (int i = 0; i < n; i++) {
      v[i] = (rng.nextDouble() - 0.5) * 0.1;
    }
    return Tensor.fromFloat32List(shape, v);
  }

  Tensor ones(List<int> shape) {
    var n = 1;
    for (final d in shape) {
      n *= d;
    }
    return Tensor.fromList(shape, List<double>.filled(n, 1.0));
  }

  final s = <String, Tensor>{};
  s['conv1.weight'] = rand([64, 3, 7, 7]);
  s['bn1.weight'] = ones([64]);
  s['bn1.bias'] = rand([64]);
  s['bn1.running_mean'] = rand([64]);
  s['bn1.running_var'] = ones([64]);

  final stages = <(String, int, int, int, int)>[
    // (name, blocks, inCh, midCh, stride)
    ('layer1', 3, 64, 64, 1),
    ('layer2', 4, 256, 128, 2),
    ('layer3', 6, 512, 256, 2),
    ('layer4', 3, 1024, 512, 2),
  ];
  for (final (name, blocks, inCh, midCh, stride) in stages) {
    for (int i = 0; i < blocks; i++) {
      final actualIn = i == 0 ? inCh : midCh * 4;
      final actualStride = i == 0 ? stride : 1;
      final p = '$name.$i';
      s['$p.conv1.weight'] = rand([midCh, actualIn, 1, 1]);
      s['$p.bn1.weight'] = ones([midCh]);
      s['$p.bn1.bias'] = rand([midCh]);
      s['$p.bn1.running_mean'] = rand([midCh]);
      s['$p.bn1.running_var'] = ones([midCh]);
      s['$p.conv2.weight'] = rand([midCh, midCh, 3, 3]);
      s['$p.bn2.weight'] = ones([midCh]);
      s['$p.bn2.bias'] = rand([midCh]);
      s['$p.bn2.running_mean'] = rand([midCh]);
      s['$p.bn2.running_var'] = ones([midCh]);
      s['$p.conv3.weight'] = rand([midCh * 4, midCh, 1, 1]);
      s['$p.bn3.weight'] = ones([midCh * 4]);
      s['$p.bn3.bias'] = rand([midCh * 4]);
      s['$p.bn3.running_mean'] = rand([midCh * 4]);
      s['$p.bn3.running_var'] = ones([midCh * 4]);
      if (i == 0 && (actualStride != 1 || actualIn != midCh * 4)) {
        s['$p.downsample.0.weight'] = rand([midCh * 4, actualIn, 1, 1]);
        s['$p.downsample.1.weight'] = ones([midCh * 4]);
        s['$p.downsample.1.bias'] = rand([midCh * 4]);
        s['$p.downsample.1.running_mean'] = rand([midCh * 4]);
        s['$p.downsample.1.running_var'] = ones([midCh * 4]);
      }
    }
  }

  s['fc.weight'] = rand([1000, 2048]);
  s['fc.bias'] = rand([1000]);
  return s;
}

Tensor _fakeInput(List<int> shape, {int seed = 0, Device device = Device.CPU}) {
  final rng = math.Random(seed);
  var n = 1;
  for (final d in shape) {
    n *= d;
  }
  final v = Float32List(n);
  for (int i = 0; i < n; i++) {
    v[i] = (rng.nextDouble() - 0.5) * 2.0;
  }
  return Tensor.fromFloat32List(shape, v, device: device);
}

bool _gpuAvailable() {
  try {
    final probe = Tensor.fromList([2], [1.0, 2.0], device: Device.GPU);
    probe.toList();
    return true;
  } catch (_) {
    return false;
  }
}

void main() {
  group('ResNet-50 model', () {
    test('config defaults', () {
      const cfg = ResNetConfig.resnet50();
      expect(cfg.blocksPerStage, equals([3, 4, 6, 3]));
      expect(cfg.numClasses, 1000);
    });

    test('forward shape [1,3,64,64] -> [1,1000]', () {
      final model = ResNet(const ResNetConfig.resnet50());
      final x = _fakeInput([1, 3, 64, 64]);
      final y = model(x);
      expect(y.shape, equals([1, 1000]));
    });

    test('bottleneck stage counts match [3,4,6,3]', () {
      final model = ResNet(const ResNetConfig.resnet50());
      expect(model.layer1.length, 3);
      expect(model.layer2.length, 4);
      expect(model.layer3.length, 6);
      expect(model.layer4.length, 3);
    });

    test('first block of layer1 has downsample (64 != 256)', () {
      final model = ResNet(const ResNetConfig.resnet50());
      expect(model.layer1[0].downsampleRef, isNotNull);
      expect(model.layer1[1].downsampleRef, isNull);
    });

    test('first block of layer2..4 has stride-2 downsample', () {
      final model = ResNet(const ResNetConfig.resnet50());
      expect(model.layer2[0].downsampleRef, isNotNull);
      expect(model.layer3[0].downsampleRef, isNotNull);
      expect(model.layer4[0].downsampleRef, isNotNull);
    });
  });

  group('ResNetLoader', () {
    test('consumes a synthetic torchvision state_dict with no unused keys', () {
      final model = ResNet(const ResNetConfig.resnet50());
      final state = _synthState(seed: 1);
      final report = ResNetLoader.loadMap(model, state);
      expect(report.unusedKeys, isEmpty);
      // 1 stem conv + 1 stem bn (5 keys)
      //   = 5
      // + per bottleneck: 3 conv+bn (5 keys each) = 15
      //   + optional downsample (5 keys)
      // + fc (2 keys)
      final expectedConsumed =
          5 // stem
          +
          (3 * 15 + 5) // layer1: 3 blocks, first has downsample
          +
          (4 * 15 + 5) // layer2
          +
          (6 * 15 + 5) // layer3
          +
          (3 * 15 + 5) // layer4
          +
          2; // fc.weight, fc.bias
      expect(report.consumedCount, expectedConsumed);
    });

    test('rejects a missing tensor', () {
      final model = ResNet(const ResNetConfig.resnet50());
      final state = _synthState(seed: 2);
      state.remove('fc.weight');
      expect(() => ResNetLoader.loadMap(model, state), throwsArgumentError);
    });

    test('forward after load returns finite logits', () {
      final model = ResNet(const ResNetConfig.resnet50());
      final state = _synthState(seed: 3);
      ResNetLoader.loadMap(model, state);
      model.eval();
      final x = _fakeInput([1, 3, 64, 64], seed: 42);
      final y = model(x).toList();
      expect(y.length, 1000);
      for (int i = 0; i < y.length; i++) {
        expect(y[i].isFinite, isTrue, reason: 'logit $i is ${y[i]}');
      }
    });
  });

  group('ResNet-50 on GPU', () {
    final gpuOk = _gpuAvailable();
    if (!gpuOk) {
      test(
        'GPU unavailable → skipped',
        () {},
        skip: 'CUDA / native/lib/libmat_mul.so not usable in this env',
      );
      return;
    }

    test('GPU forward shape matches CPU (loaded synthetic weights)', () {
      final cpu = ResNet(const ResNetConfig.resnet50());
      final gpu = ResNet(const ResNetConfig.resnet50(device: Device.GPU));
      final state = _synthState(seed: 7);
      ResNetLoader.loadMap(cpu, state);
      ResNetLoader.loadMap(gpu, state);
      cpu.eval();
      gpu.eval();

      final xCpu = _fakeInput([1, 3, 64, 64], seed: 11);
      final xGpu = _fakeInput([1, 3, 64, 64], seed: 11, device: Device.GPU);
      final cpuLogits = cpu(xCpu).toList();
      final gpuLogits = gpu(xGpu).toList();
      expect(cpuLogits.length, gpuLogits.length);
      // Loose tolerance: 50 layers of accumulated fp32 differences.
      double maxDiff = 0;
      for (int i = 0; i < cpuLogits.length; i++) {
        final d = (cpuLogits[i] - gpuLogits[i]).abs();
        if (d > maxDiff) maxDiff = d;
      }
      expect(maxDiff, lessThan(5e-2), reason: 'max cpu/gpu diff = $maxDiff');
    });
  });
}
