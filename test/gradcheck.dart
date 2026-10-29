/// Numerical gradient-checking harness (A1 of the PyTorch parity roadmap,
/// see `doc/parity-plan.md`).
///
/// Verifies the analytic gradients produced by a tensor op's `backward`
/// pass against central finite differences, the same technique as
/// `torch.autograd.gradcheck`. This is the gate every new differentiable
/// op must pass.
library;

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

/// Per-input outcome of a [gradCheck] run.
class GradCheckResult {
  GradCheckResult(this.inputIndex, this.maxAbsError, this.analytic, this.numeric);

  final int inputIndex;
  final double maxAbsError;
  final List<double> analytic;
  final List<double> numeric;
}

/// Numerically verify the gradients of [fn] w.r.t. each checked input.
///
/// [fn] receives freshly-built input tensors (one per entry in [inputs],
/// shaped by [shapes]) and returns a tensor that is reduced to a scalar
/// via `.sum()` before differentiation — so inputs of any shape work.
///
/// For each checked input the analytic gradient (from one `backward`
/// pass) is compared elementwise against the central finite-difference
/// estimate `(f(x+eps) - f(x-eps)) / (2*eps)`. An element passes when
/// `|analytic - numeric| <= atol + rtol * |numeric|`.
///
/// Returns one [GradCheckResult] per input (null-grad inputs report a
/// zero-filled analytic vector).
List<GradCheckResult> gradCheck(
  Tensor Function(List<Tensor>) fn,
  List<List<double>> inputs,
  List<List<int>> shapes, {
  Device device = Device.CPU,
  double eps = 1e-3,
  double atol = 1e-2,
  double rtol = 1e-2,
  List<bool>? check,
}) {
  if (inputs.length != shapes.length) {
    throw ArgumentError(
      'gradCheck: ${inputs.length} inputs but ${shapes.length} shapes',
    );
  }
  final doCheck = check ?? List<bool>.filled(inputs.length, true);

  // --- Analytic gradients: one forward + backward pass. ---
  final analyticInputs = [
    for (int i = 0; i < inputs.length; i++)
      Tensor.fromList(
        shapes[i],
        inputs[i],
        device: device,
        requiresGrad: doCheck[i],
      ),
  ];
  final loss = fn(analyticInputs).sum();
  loss.backward();
  final analytic = [
    for (int i = 0; i < inputs.length; i++)
      (doCheck[i] && analyticInputs[i].grad != null)
          ? analyticInputs[i].grad!.toList()
          : List<double>.filled(inputs[i].length, 0.0),
  ];

  // --- Numeric gradients: central differences, autograd disabled. ---
  double evalLoss(List<List<double>> data) => Tensor.noGrad(() {
        final ins = [
          for (int i = 0; i < data.length; i++)
            Tensor.fromList(shapes[i], data[i], device: device),
        ];
        return fn(ins).sum().toList()[0];
      });

  final results = <GradCheckResult>[];
  for (int i = 0; i < inputs.length; i++) {
    if (!doCheck[i]) {
      results.add(GradCheckResult(i, 0.0, analytic[i], analytic[i]));
      continue;
    }
    final numeric = List<double>.filled(inputs[i].length, 0.0);
    for (int j = 0; j < inputs[i].length; j++) {
      final data = [for (final v in inputs) List<double>.of(v)];
      final orig = data[i][j];
      data[i][j] = orig + eps;
      final lp = evalLoss(data);
      data[i][j] = orig - eps;
      final lm = evalLoss(data);
      numeric[j] = (lp - lm) / (2 * eps);
    }
    double maxErr = 0.0;
    for (int j = 0; j < numeric.length; j++) {
      final err = (analytic[i][j] - numeric[j]).abs();
      if (err > maxErr) maxErr = err;
    }
    results.add(GradCheckResult(i, maxErr, analytic[i], numeric));
  }
  return results;
}

/// Run [gradCheck] and assert every checked input passes the tolerance.
void expectGradCheck(
  Tensor Function(List<Tensor>) fn,
  List<List<double>> inputs,
  List<List<int>> shapes, {
  Device device = Device.CPU,
  double eps = 1e-3,
  double atol = 1e-2,
  double rtol = 1e-2,
  List<bool>? check,
  String? reason,
}) {
  final results = gradCheck(
    fn,
    inputs,
    shapes,
    device: device,
    eps: eps,
    atol: atol,
    rtol: rtol,
    check: check,
  );
  for (final r in results) {
    for (int j = 0; j < r.analytic.length; j++) {
      final a = r.analytic[j];
      final n = r.numeric[j];
      final tol = atol + rtol * n.abs();
      expect(
        (a - n).abs() <= tol,
        isTrue,
        reason:
            '${reason ?? ''} input ${r.inputIndex} elem $j: '
            'analytic $a vs numeric $n (|diff| ${(a - n).abs()}, tol $tol)',
      );
    }
  }
}
