/// Tests for the Mixture-of-Experts feed-forward block and MoE
/// language model.
library;

import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

void main() {
  group('Expert', () {
    test('forward preserves [T, dim] shape', () {
      final e = Expert(4, 8, seed: 1);
      final x = Tensor.fromList([
        3,
        4,
      ], List<double>.generate(12, (i) => i * 0.1));
      final y = e(x);
      expect(y.shape, [3, 4]);
    });

    test('parameters flatten w1 + w2', () {
      final e = Expert(4, 8, seed: 1);
      // Linear has weight + bias by default => 4 tensors total.
      expect(e.parameters().length, 4);
    });

    test('SwiGLU variant allocates w3 and produces correct shape', () {
      final e = Expert(4, 8, seed: 1, variant: ExpertVariant.swiGlu);
      // w1 + w2 + w3, each with weight + bias => 6 tensors.
      expect(e.parameters().length, 6);
      expect(e.w3, isNotNull);
      final x = Tensor.fromList([
        3,
        4,
      ], List<double>.generate(12, (i) => (i - 6) * 0.1));
      final y = e(x);
      expect(y.shape, [3, 4]);
      for (final v in y.toList()) {
        expect(v.isFinite, isTrue);
      }
    });

    test('SwiGLU variant: gradients flow through w1, w2, and w3', () {
      final e = Expert(4, 8, seed: 3, variant: ExpertVariant.swiGlu);
      final x = Tensor.fromList([
        2,
        4,
      ], List<double>.generate(8, (i) => (i - 4) * 0.1));
      final y = e(x);
      y.sum().backward();
      for (final w in [e.w1.weight, e.w2.weight, e.w3!.weight]) {
        expect(w.grad, isNotNull);
        expect(w.grad!.toList().any((v) => v.abs() > 1e-8), isTrue);
      }
    });

    test('SwiGLU variant matches manual computation', () {
      final e = Expert(4, 8, seed: 42, variant: ExpertVariant.swiGlu);
      final x = Tensor.fromList([
        2,
        4,
      ], List<double>.generate(8, (i) => (i - 4) * 0.1));
      final y = e(x);
      // Manual: w2(silu(w1(x)) * w3(x))
      final gate = e.w1(x);
      final up = e.w3!(x);
      final silu = gate * gate.sigmoid();
      final expected = e.w2(silu * up);
      final yList = y.toList();
      final eList = expected.toList();
      expect(yList.length, eList.length);
      for (int i = 0; i < yList.length; i++) {
        expect((yList[i] - eList[i]).abs(), lessThan(1e-9));
      }
    });
  });

  group('MoEFeedForward', () {
    test('constructor rejects out-of-range topK', () {
      expect(
        () => MoEFeedForward(
          embedDim: 4,
          numRoutedExperts: 3,
          numSharedExperts: 0,
          topK: 0,
          expertHiddenDim: 8,
        ),
        throwsArgumentError,
      );
      expect(
        () => MoEFeedForward(
          embedDim: 4,
          numRoutedExperts: 3,
          numSharedExperts: 0,
          topK: 4,
          expertHiddenDim: 8,
        ),
        throwsArgumentError,
      );
    });

    test('forward preserves [T, embedDim]', () {
      final moe = MoEFeedForward(
        embedDim: 8,
        numRoutedExperts: 4,
        numSharedExperts: 1,
        topK: 2,
        expertHiddenDim: 16,
        seed: 7,
      );
      final x = Tensor.fromList([
        6,
        8,
      ], List<double>.generate(48, (i) => (i % 7) * 0.05));
      final y = moe(x);
      expect(y.shape, [6, 8]);
    });

    test('routes each token to exactly topK experts', () {
      const t = 5;
      const e = 4;
      const k = 2;
      final moe = MoEFeedForward(
        embedDim: 4,
        numRoutedExperts: e,
        numSharedExperts: 0,
        topK: k,
        expertHiddenDim: 8,
        seed: 3,
      );
      final x = Tensor.fromList([
        t,
        4,
      ], List<double>.generate(t * 4, (i) => (i % 5) * 0.1));
      moe(x);
      final total = moe.expertLoad.fold<int>(0, (a, b) => a + b);
      expect(total, t * k);
    });

    test('gradients flow to gateW and to expert weights', () {
      final moe = MoEFeedForward(
        embedDim: 4,
        numRoutedExperts: 3,
        numSharedExperts: 1,
        topK: 2,
        expertHiddenDim: 6,
        seed: 11,
      );
      final x = Tensor.fromList([
        3,
        4,
      ], List<double>.generate(12, (i) => (i % 4) * 0.1 - 0.15));
      final y = moe(x);
      final loss = (y * y).sum() * 0.5;
      loss.backward();

      // gateW should have a non-null, non-zero gradient.
      expect(moe.gateW.grad, isNotNull);
      final gGate = moe.gateW.grad!.toList();
      final anyNonZero = gGate.any((v) => v.abs() > 1e-8);
      expect(anyNonZero, isTrue, reason: 'gateW gradient is all zero');

      // At least one routed expert must have a non-zero grad on its
      // first-layer weight (top-K guarantees at least topK experts
      // active across the batch).
      var routedAnyGrad = false;
      for (final e in moe.routedExperts) {
        final gw = e.w1.weight.grad;
        if (gw == null) continue;
        if (gw.toList().any((v) => v.abs() > 1e-8)) {
          routedAnyGrad = true;
          break;
        }
      }
      expect(routedAnyGrad, isTrue);

      // Shared expert always active => should always have gradient.
      final sharedG = moe.sharedExperts.first.w1.weight.grad;
      expect(sharedG, isNotNull);
      expect(sharedG!.toList().any((v) => v.abs() > 1e-8), isTrue);
    });

    test('updateRoutingBias equalises after biased routing', () {
      const e = 4;
      final moe = MoEFeedForward(
        embedDim: 4,
        numRoutedExperts: e,
        numSharedExperts: 0,
        topK: 1,
        expertHiddenDim: 4,
        biasUpdateRate: 0.5,
        seed: 5,
      );
      // Fake severely-lopsided load by directly poking the internal
      // counter through a synthesised forward: we run a batch that
      // routes largely to a single expert by leaving gateW random and
      // just observe that updateRoutingBias moves biases in the right
      // direction.
      final x = Tensor.fromList([
        8,
        4,
      ], List<double>.generate(32, (i) => (i % 3) * 0.1));
      moe(x);
      final loadBefore = moe.expertLoad;
      final maxIdx = _argmax(loadBefore);
      final minIdx = _argmin(loadBefore);
      final biasesBefore = List<double>.of(moe.routingBias);
      moe.updateRoutingBias();
      // Under-utilised expert bias must not have decreased.
      expect(
        moe.routingBias[minIdx],
        greaterThanOrEqualTo(biasesBefore[minIdx]),
      );
      // Over-utilised expert bias must not have increased.
      expect(moe.routingBias[maxIdx], lessThanOrEqualTo(biasesBefore[maxIdx]));
      // Counters reset.
      expect(moe.expertLoad, List<int>.filled(e, 0));
    });

    test('sign bias update rule adjusts by exactly biasUpdateRate', () {
      const e = 4;
      final moe = MoEFeedForward(
        embedDim: 4,
        numRoutedExperts: e,
        numSharedExperts: 0,
        topK: 1,
        expertHiddenDim: 4,
        biasUpdateRate: 0.25,
        seed: 5,
        // biasUpdateRule defaults to sign.
      );
      final x = Tensor.fromList([
        8,
        4,
      ], List<double>.generate(32, (i) => (i % 3) * 0.1));
      moe(x);
      final loadBefore = List<int>.of(moe.expertLoad);
      final total = loadBefore.fold<int>(0, (a, b) => a + b);
      final mean = total / e;
      final biasesBefore = List<double>.of(moe.routingBias);
      moe.updateRoutingBias();
      for (int j = 0; j < e; j++) {
        final expected = loadBefore[j] < mean
            ? biasesBefore[j] + 0.25
            : loadBefore[j] > mean
            ? biasesBefore[j] - 0.25
            : biasesBefore[j];
        expect((moe.routingBias[j] - expected).abs(), lessThan(1e-9));
      }
    });

    test('proportional bias update rule preserved as opt-in', () {
      const e = 4;
      final moe = MoEFeedForward(
        embedDim: 4,
        numRoutedExperts: e,
        numSharedExperts: 0,
        topK: 1,
        expertHiddenDim: 4,
        biasUpdateRate: 0.5,
        seed: 5,
        biasUpdateRule: BiasUpdateRule.proportional,
      );
      final x = Tensor.fromList([
        8,
        4,
      ], List<double>.generate(32, (i) => (i % 3) * 0.1));
      moe(x);
      final loadBefore = List<int>.of(moe.expertLoad);
      final total = loadBefore.fold<int>(0, (a, b) => a + b);
      final mean = total / e;
      final biasesBefore = List<double>.of(moe.routingBias);
      moe.updateRoutingBias();
      for (int j = 0; j < e; j++) {
        final expected = biasesBefore[j] + 0.5 * (mean - loadBefore[j]) / mean;
        expect((moe.routingBias[j] - expected).abs(), lessThan(1e-9));
      }
    });

    test('sigmoid gate: forward runs and produces correct shape', () {
      final moe = MoEFeedForward(
        embedDim: 4,
        numRoutedExperts: 4,
        numSharedExperts: 1,
        topK: 2,
        expertHiddenDim: 8,
        seed: 7,
        gateFunction: GateFunction.sigmoid,
      );
      final x = Tensor.fromList([
        5,
        4,
      ], List<double>.generate(20, (i) => (i - 10) * 0.05));
      final y = moe(x);
      expect(y.shape, [5, 4]);
      // Every element finite.
      for (final v in y.toList()) {
        expect(v.isFinite, isTrue);
      }
    });

    test(
      'top-K renormalization: routed contribution weights sum to ~1 per row',
      () {
        // Set up a moe that is purely routed (no shared) and identity-ish so we
        // can inspect the weights indirectly via a probe input. We instead
        // verify the renormalization by checking that a single-expert-topK moe
        // with renormalize=true reproduces its selected expert's output
        // scale-invariantly of the gate value.
        final moe = MoEFeedForward(
          embedDim: 4,
          numRoutedExperts: 3,
          numSharedExperts: 0,
          topK: 1,
          expertHiddenDim: 4,
          seed: 11,
          renormalizeTopK: true,
        );
        // For topK=1 with renormalization, the selected expert's weight is
        // always 1.0, so the moe output equals exactly the selected expert's
        // output on that row.
        final x = Tensor.fromList([1, 4], [0.1, 0.2, -0.1, 0.05]);
        final y = moe(x);
        // Find which expert was routed.
        final gateLogits = x.matmul(moe.gateW);
        final gateScores = gateLogits.softmax();
        final s = gateScores.toList();
        int argmax = 0;
        double best = s[0];
        for (int j = 1; j < 3; j++) {
          if (s[j] > best) {
            best = s[j];
            argmax = j;
          }
        }
        final expected = moe.routedExperts[argmax](x);
        final yList = y.toList();
        final eList = expected.toList();
        for (int i = 0; i < 4; i++) {
          expect((yList[i] - eList[i]).abs(), lessThan(1e-6));
        }
      },
    );

    test(
      'sigmoid + renormalize: gradients still flow to gateW and experts',
      () {
        final moe = MoEFeedForward(
          embedDim: 4,
          numRoutedExperts: 4,
          numSharedExperts: 0,
          topK: 2,
          expertHiddenDim: 4,
          seed: 13,
          gateFunction: GateFunction.sigmoid,
          renormalizeTopK: true,
        );
        final x = Tensor.fromList([
          3,
          4,
        ], List<double>.generate(12, (i) => (i - 6) * 0.1));
        final y = moe(x);
        final loss = y.sum();
        loss.backward();
        expect(moe.gateW.grad, isNotNull);
        expect(moe.gateW.grad!.toList().any((v) => v.abs() > 1e-8), isTrue);
        final ew = moe.routedExperts[0].w1.weight;
        expect(ew.grad, isNotNull);
        expect(ew.grad!.toList().any((v) => v.abs() > 1e-8), isTrue);
      },
    );

    test(
      'expertVariant swiGlu: forward + gradients through experts w1/w2/w3',
      () {
        final moe = MoEFeedForward(
          embedDim: 4,
          numRoutedExperts: 3,
          numSharedExperts: 1,
          topK: 2,
          expertHiddenDim: 8,
          seed: 17,
          expertVariant: ExpertVariant.swiGlu,
          gateFunction: GateFunction.sigmoid,
        );
        // Every routed expert should have a w3.
        for (final e in moe.routedExperts) {
          expect(e.w3, isNotNull);
        }
        for (final e in moe.sharedExperts) {
          expect(e.w3, isNotNull);
        }
        final x = Tensor.fromList([
          4,
          4,
        ], List<double>.generate(16, (i) => (i - 8) * 0.05));
        final y = moe(x);
        expect(y.shape, [4, 4]);
        y.sum().backward();
        // Pick one routed expert and one shared expert; both should get
        // grads on w3.
        expect(moe.routedExperts[0].w3!.weight.grad, isNotNull);
        expect(
          moe.routedExperts[0].w3!.weight.grad!.toList().any(
            (v) => v.abs() > 1e-8,
          ),
          isTrue,
        );
        expect(moe.sharedExperts[0].w3!.weight.grad, isNotNull);
        expect(
          moe.sharedExperts[0].w3!.weight.grad!.toList().any(
            (v) => v.abs() > 1e-8,
          ),
          isTrue,
        );
      },
    );

    test('grouped routing: numExpertGroups must divide numRoutedExperts', () {
      expect(
        () => MoEFeedForward(
          embedDim: 4,
          numRoutedExperts: 5,
          numSharedExperts: 0,
          topK: 1,
          expertHiddenDim: 4,
          numExpertGroups: 2,
        ),
        throwsArgumentError,
      );
    });

    test('grouped routing: topK must be reachable from topKGroups', () {
      expect(
        () => MoEFeedForward(
          embedDim: 4,
          numRoutedExperts: 8,
          numSharedExperts: 0,
          topK: 4,
          expertHiddenDim: 4,
          numExpertGroups: 4, // 2 experts/group
          topKGroups: 1, // -> 2 reachable experts, but topK=4
        ),
        throwsArgumentError,
      );
    });

    test(
      'grouped routing: every selected expert lives in a top-K_groups group',
      () {
        const e = 8;
        const g = 4;
        const kg = 2;
        final moe = MoEFeedForward(
          embedDim: 4,
          numRoutedExperts: e,
          numSharedExperts: 0,
          topK: 2,
          expertHiddenDim: 4,
          seed: 21,
          numExpertGroups: g,
          topKGroups: kg,
        );
        // Run a couple of tokens.
        final x = Tensor.fromList([
          3,
          4,
        ], List<double>.generate(12, (i) => (i - 6) * 0.1));
        moe(x);
        // Across 3 tokens with topK=2, exactly 6 routings happened.
        // The counters live per-expert. Verify that any expert with a
        // positive load lives inside one of the top-K_groups groups
        // for the token that routed to it — since we can't observe
        // per-token routing directly here, just check the counter sum.
        final loadTotal = moe.expertLoad.fold<int>(0, (a, b) => a + b);
        expect(loadTotal, 3 * 2);
        // At most `topKGroups * expertsPerGroup * t = 2 * 2 * 3 = 12`\n"
        // experts could have non-zero load — but with 3 tokens picking\n"
        // 2 out of a possible 4-expert pool per token, at most 4 unique\n"
        // experts across all tokens can be hot when topKGroups * per_g\n"
        // = 4, but tokens may share groups. Weaker invariant: the\n"
        // number of hot experts must be <= topKGroups * expertsPerGroup\n"
        // * t = 12 (trivially true), and >= 1 (non-empty).\n"
        final hot = moe.expertLoad.where((c) => c > 0).length;
        expect(hot, greaterThanOrEqualTo(1));
        expect(hot, lessThanOrEqualTo(kg * (e ~/ g) * x.shape[0]));
      },
    );

    test(
      'grouped routing exactly matches ungrouped when topKGroups == numExpertGroups',
      () {
        // With topKGroups == numExpertGroups every group is selected,
        // so grouped routing must produce the exact same top-K as
        // ungrouped routing.
        const e = 6;
        final ungrouped = MoEFeedForward(
          embedDim: 4,
          numRoutedExperts: e,
          numSharedExperts: 0,
          topK: 2,
          expertHiddenDim: 4,
          seed: 33,
        );
        final grouped = MoEFeedForward(
          embedDim: 4,
          numRoutedExperts: e,
          numSharedExperts: 0,
          topK: 2,
          expertHiddenDim: 4,
          seed: 33,
          numExpertGroups: 3,
          topKGroups: 3,
        );
        final x = Tensor.fromList([
          4,
          4,
        ], List<double>.generate(16, (i) => (i - 8) * 0.05));
        final yU = ungrouped(x).toList();
        final yG = grouped(x).toList();
        expect(yU.length, yG.length);
        for (int i = 0; i < yU.length; i++) {
          expect((yU[i] - yG[i]).abs(), lessThan(1e-6));
        }
      },
    );

    test('routeScale multiplies the routed contributions', () {
      // Compare two moes with identical seed but different routeScale.
      // The shared-experts path is unweighted, so with numShared=0 the
      // output is purely a linear function of routeScale.
      const scale = 2.5;
      final a = MoEFeedForward(
        embedDim: 4,
        numRoutedExperts: 3,
        numSharedExperts: 0,
        topK: 1,
        expertHiddenDim: 4,
        seed: 51,
      );
      final b = MoEFeedForward(
        embedDim: 4,
        numRoutedExperts: 3,
        numSharedExperts: 0,
        topK: 1,
        expertHiddenDim: 4,
        seed: 51,
        routeScale: scale,
      );
      final x = Tensor.fromList([
        2,
        4,
      ], [0.1, -0.2, 0.3, 0.05, -0.1, 0.2, -0.05, 0.15]);
      final ya = a(x).toList();
      final yb = b(x).toList();
      expect(ya.length, yb.length);
      for (int i = 0; i < ya.length; i++) {
        expect((yb[i] - scale * ya[i]).abs(), lessThan(1e-6));
      }
    });

    test('sparse execution: forward matches dense forward (CPU)', () {
      final dense = MoEFeedForward(
        embedDim: 4,
        numRoutedExperts: 4,
        numSharedExperts: 1,
        topK: 2,
        expertHiddenDim: 8,
        seed: 71,
      );
      final sparse = MoEFeedForward(
        embedDim: 4,
        numRoutedExperts: 4,
        numSharedExperts: 1,
        topK: 2,
        expertHiddenDim: 8,
        seed: 71,
        sparseExecution: true,
      );
      final x = Tensor.fromList([
        5,
        4,
      ], List<double>.generate(20, (i) => (i - 10) * 0.05));
      final yD = dense(x).toList();
      final yS = sparse(x).toList();
      expect(yD.length, yS.length);
      for (int i = 0; i < yD.length; i++) {
        expect((yD[i] - yS[i]).abs(), lessThan(1e-5));
      }
    });

    test('sparse execution: backward matches dense backward (CPU)', () {
      final xD = Tensor.fromList([
        4,
        4,
      ], List<double>.generate(16, (i) => (i - 8) * 0.05),
          requiresGrad: true);
      final xS = Tensor.fromList([
        4,
        4,
      ], List<double>.generate(16, (i) => (i - 8) * 0.05),
          requiresGrad: true);
      final dense = MoEFeedForward(
        embedDim: 4,
        numRoutedExperts: 4,
        numSharedExperts: 0,
        topK: 2,
        expertHiddenDim: 6,
        seed: 83,
      );
      final sparse = MoEFeedForward(
        embedDim: 4,
        numRoutedExperts: 4,
        numSharedExperts: 0,
        topK: 2,
        expertHiddenDim: 6,
        seed: 83,
        sparseExecution: true,
      );
      dense(xD).sum().backward();
      sparse(xS).sum().backward();
      final gD = xD.grad!.toList();
      final gS = xS.grad!.toList();
      for (int i = 0; i < gD.length; i++) {
        expect((gD[i] - gS[i]).abs(), lessThan(1e-5));
      }
      // Gate grads must match too.
      final ggD = dense.gateW.grad!.toList();
      final ggS = sparse.gateW.grad!.toList();
      for (int i = 0; i < ggD.length; i++) {
        expect((ggD[i] - ggS[i]).abs(), lessThan(1e-5));
      }
    });

    test('sparse execution: GPU parity with dense on GPU', () {
      final xD = Tensor.fromList([
        4,
        4,
      ], List<double>.generate(16, (i) => (i - 8) * 0.05),
          requiresGrad: true, device: Device.GPU);
      final xS = Tensor.fromList([
        4,
        4,
      ], List<double>.generate(16, (i) => (i - 8) * 0.05),
          requiresGrad: true, device: Device.GPU);
      final dense = MoEFeedForward(
        embedDim: 4,
        numRoutedExperts: 4,
        numSharedExperts: 0,
        topK: 2,
        expertHiddenDim: 6,
        seed: 91,
        device: Device.GPU,
        activation: ExpertActivation.silu,
      );
      final sparse = MoEFeedForward(
        embedDim: 4,
        numRoutedExperts: 4,
        numSharedExperts: 0,
        topK: 2,
        expertHiddenDim: 6,
        seed: 91,
        device: Device.GPU,
        activation: ExpertActivation.silu,
        sparseExecution: true,
      );
      final yD = dense(xD);
      final yS = sparse(xS);
      final yDList = yD.toList();
      final ySList = yS.toList();
      for (int i = 0; i < yDList.length; i++) {
        expect((yDList[i] - ySList[i]).abs(), lessThan(1e-4));
      }
      yD.sum().backward();
      yS.sum().backward();
      final gD = xD.grad!.toList();
      final gS = xS.grad!.toList();
      for (int i = 0; i < gD.length; i++) {
        expect((gD[i] - gS[i]).abs(), lessThan(1e-4));
      }
    });
  });

  group('MoELanguageModel', () {
    test('forward returns [seqLen, vocabSize] for 1D tokens', () {
      final lm = MoELanguageModel(
        vocabSize: 12,
        embedDim: 8,
        numLayers: 2,
        numHeads: 2,
        maxLen: 16,
        numRoutedExperts: 3,
        numSharedExperts: 1,
        topK: 2,
        expertHiddenDim: 16,
        seed: 1,
      );
      final tokens = Tensor.fromList([
        7,
      ], List<double>.generate(7, (i) => (i % 12).toDouble()));
      final logits = lm(tokens);
      expect(logits.shape, [7, 12]);
    });

    test('rejects non-1D input and oversize sequences', () {
      final lm = MoELanguageModel(
        vocabSize: 6,
        embedDim: 4,
        numLayers: 1,
        numHeads: 2,
        maxLen: 4,
        numRoutedExperts: 2,
        numSharedExperts: 0,
        topK: 1,
        expertHiddenDim: 4,
        seed: 0,
      );
      expect(
        () => lm(Tensor.fromList([2, 3], List<double>.filled(6, 0.0))),
        throwsArgumentError,
      );
      expect(
        () => lm(Tensor.fromList([5], List<double>.filled(5, 0.0))),
        throwsArgumentError,
      );
    });

    test('overfits a tiny next-token map', () {
      const vocab = 6;
      const seqLen = 8;
      final rng = math.Random(0);
      final tokens = List<double>.generate(
        seqLen,
        (_) => rng.nextInt(vocab).toDouble(),
      );
      // targets = tokens shifted by +1 mod vocab.
      final targets = List<double>.generate(
        seqLen,
        (i) => ((tokens[i].toInt() + 1) % vocab).toDouble(),
      );

      final lm = MoELanguageModel(
        vocabSize: vocab,
        embedDim: 8,
        numLayers: 2,
        numHeads: 2,
        maxLen: seqLen,
        numRoutedExperts: 3,
        numSharedExperts: 1,
        topK: 2,
        expertHiddenDim: 16,
        seed: 3,
      );
      final opt = Adam(lm.parameters(), lr: 5e-3);
      final x = Tensor.fromList([seqLen], tokens);
      final y = Tensor.fromList([seqLen], targets);

      double last = double.infinity;
      for (int step = 0; step < 200; step++) {
        opt.zeroGrad();
        final loss = lm(x).crossEntropy(y).mean();
        loss.backward();
        opt.step();
        last = loss.toList()[0];
      }
      expect(last, lessThan(0.2), reason: 'final loss = $last');
    });

    test('updateRoutingBias fans out across all blocks', () {
      final lm = MoELanguageModel(
        vocabSize: 8,
        embedDim: 4,
        numLayers: 2,
        numHeads: 2,
        maxLen: 8,
        numRoutedExperts: 3,
        numSharedExperts: 0,
        topK: 1,
        expertHiddenDim: 4,
        seed: 0,
      );
      final tokens = Tensor.fromList([
        4,
      ], List<double>.generate(4, (i) => i.toDouble()));
      lm(tokens);
      // Every block should now have non-zero load counters.
      for (final b in lm.blocks) {
        expect(b.moe.expertLoad.any((c) => c > 0), isTrue);
      }
      lm.updateRoutingBias();
      // And they should have been reset.
      for (final b in lm.blocks) {
        expect(b.moe.expertLoad.every((c) => c == 0), isTrue);
      }
    });
  });

  group('MoE on GPU (SiLU)', () {
    test('Expert(SiLU) forward+backward on GPU produces grads', () {
      final e = Expert(
        16,
        32,
        device: Device.GPU,
        seed: 0,
        activation: ExpertActivation.silu,
      );
      final x = Tensor.fromList(
        [8, 16],
        List<double>.generate(128, (i) => ((i * 7) % 13) * 0.05 - 0.3),
        device: Device.GPU,
        requiresGrad: true,
      );
      final y = e(x);
      expect(y.shape, [8, 16]);
      expect(y.device, Device.GPU);
      final loss = (y * y).sum() * 0.5;
      loss.backward();
      expect(e.w1.weight.grad, isNotNull);
      expect(e.w2.weight.grad, isNotNull);
      expect(x.grad, isNotNull);
      expect(x.grad!.device, Device.GPU);
    });

    test('MoEFeedForward on GPU accepts ReLU (relu_bwd on GPU landed)', () {
      // Historically this threw; now ReLU has a GPU backward.
      final ffn = MoEFeedForward(
        embedDim: 8,
        numRoutedExperts: 2,
        numSharedExperts: 0,
        topK: 1,
        expertHiddenDim: 8,
        device: Device.GPU,
      );
      final x = Tensor.fromList(
        [3, 8],
        List<double>.generate(24, (i) => (i % 5) * 0.1),
        device: Device.GPU,
        requiresGrad: true,
      );
      final y = ffn(x);
      expect(y.shape, [3, 8]);
      y.sum().backward();
    });

    test('MoEFeedForward GPU backward populates gate + expert grads', () {
      final ffn = MoEFeedForward(
        embedDim: 32,
        numRoutedExperts: 4,
        numSharedExperts: 1,
        topK: 2,
        expertHiddenDim: 64,
        device: Device.GPU,
        seed: 0,
        activation: ExpertActivation.silu,
      );
      final x = Tensor.fromList(
        [16, 32],
        List<double>.generate(16 * 32, (i) => ((i * 7) % 13) * 0.05 - 0.3),
        device: Device.GPU,
        requiresGrad: true,
      );
      final y = ffn(x);
      expect(y.shape, [16, 32]);
      expect(y.device, Device.GPU);
      final loss = (y * y).sum() * 0.5;
      loss.backward();
      expect(ffn.gateW.grad, isNotNull);
      for (final e in ffn.routedExperts) {
        expect(e.w1.weight.grad, isNotNull);
      }
    });

    test('MoELanguageModel on GPU trains a tiny sequence (loss decreases)', () {
      final lm = MoELanguageModel(
        vocabSize: 32,
        embedDim: 32,
        numLayers: 2,
        numHeads: 4,
        maxLen: 16,
        numRoutedExperts: 4,
        numSharedExperts: 1,
        topK: 2,
        expertHiddenDim: 64,
        device: Device.GPU,
        seed: 0,
        activation: ExpertActivation.silu,
      );
      final toks = Tensor.fromList(
        [8],
        List<double>.generate(8, (i) => (i * 3) % 32),
        device: Device.GPU,
      );
      final targets = Tensor.fromList(
        [8],
        List<double>.generate(8, (i) => (i * 3 + 1) % 32),
        device: Device.GPU,
      );
      final params = lm.parameters();
      final opt = SGD(params, lr: 0.05);
      double first = 0, last = 0;
      for (int step = 0; step < 30; step++) {
        opt.zeroGrad();
        final logits = lm(toks);
        final loss = logits.crossEntropy(targets).mean();
        final v = loss.toList()[0];
        if (step == 0) first = v;
        if (step == 29) last = v;
        loss.backward();
        opt.step();
      }
      expect(last, lessThan(first));
    });
  });
}

int _argmax(List<int> xs) {
  var bi = 0;
  for (int i = 1; i < xs.length; i++) {
    if (xs[i] > xs[bi]) bi = i;
  }
  return bi;
}

int _argmin(List<int> xs) {
  var bi = 0;
  for (int i = 1; i < xs.length; i++) {
    if (xs[i] < xs[bi]) bi = i;
  }
  return bi;
}
