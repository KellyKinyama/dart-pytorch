# dart-eng-math — combined corpus

Generated from `/mnt/c/www/dart/dart-eng-math/docs` on 2026-09-14T08:31:16Z.
Every source file below is prefixed with a `## FILE: <relative-path>` marker so RAG chunkers can trace passages back to their origin. Blank lines are preserved so paragraph-based chunkers see one chunk per source paragraph.

## Manifest

- README.md
- bird-full-plan.md
- plain-english/README.md
- plain-english/topic-1-further-calculus.md
- plain-english/topic-2-series.md
- plain-english/topic-3-complex-numbers.md
- plain-english/topic-4-partial-fractions.md
- plain-english/topic-5-linear-algebra.md
- plain-english/topic-6-laplace.md
- plain-english/topic-g-statistics.md
- plain-english/topic-h-fourier.md
- plain-english/topic-i-ode-methods.md
- plain-english/topic-j-vectors.md
- plain-english/topic-k-trig-hyperbolic.md
- plain-english/topic-l-symbolic-diff.md
- plain-english/topic-m-applied-numerical.md
- plain-english/topic-n-symbolic-integration.md
- session-2026-09-12.md
- student-guide.md
- topic-1-further-calculus.md
- topic-2-series.md
- topic-3-complex-numbers.md
- topic-4-partial-fractions.md
- topic-5-linear-algebra.md
- topic-6-laplace.md
- topic-e-electronics.md
- topic-g-statistics.md
- topic-h-fourier.md
- topic-i-ode-methods.md
- topic-j-vectors.md
- topic-k-trig-hyperbolic.md
- topic-l-symbolic-diff.md
- topic-m-applied-numerical.md
- topic-n-symbolic-integration.md

---



---

## FILE: README.md

# dart-eng-math — documentation

Reference material for the `dart_eng_math` library. Content is organised
by syllabus topic (from
[A-advanced-mathematics-2.md](../A-advanced-mathematics-2.md)) and cross-referenced
to worked examples in *Higher Engineering Mathematics* (Bird).

**New here?** Start with the **[Student's guide](student-guide.md)** — a
practical walkthrough with setup instructions, one-liner recipes for
common tasks, and end-to-end worked examples.

## Topic map

| # | Topic | Doc | Status |
|---|---|---|---|
| 1 | Further calculus | [topic-1-further-calculus.md](topic-1-further-calculus.md) | ✅ Done |
| 2 | Series (AP, GP, binomial) | [topic-2-series.md](topic-2-series.md) | ✅ Done |
| 3 | Complex numbers, Euler, de Moivre | [topic-3-complex-numbers.md](topic-3-complex-numbers.md) | ✅ Done |
| 4 | Partial fractions | [topic-4-partial-fractions.md](topic-4-partial-fractions.md) | ✅ Done |
| 5 | Determinants, matrices, ODE systems | [topic-5-linear-algebra.md](topic-5-linear-algebra.md) | ✅ Done |
| 6 | Laplace transforms | [topic-6-laplace.md](topic-6-laplace.md) | ✅ Done |
| G | Statistics & probability (Bird Ch. 55–60) | [topic-g-statistics.md](topic-g-statistics.md) | ✅ Done |
| H | Fourier series & transforms (Bird Ch. 66–71 + CFT/DFT/FFT) | [topic-h-fourier.md](topic-h-fourier.md) | ✅ Done |
| I | ODE methods (Bird Ch. 49, 65) | [topic-i-ode-methods.md](topic-i-ode-methods.md) | ✅ Done |
| J | Vectors + partial derivatives (Bird Ch. 24, 26, 34–36) | [topic-j-vectors.md](topic-j-vectors.md) | ✅ Done |
| K | Trig, hyperbolic, waveforms (Bird Ch. 5, 11–14, 17, 25) | [topic-k-trig-hyperbolic.md](topic-k-trig-hyperbolic.md) | ✅ Done |
| M | Applied numerical methods (Bird Ch. 9, 19, 37–38, 45) | [topic-m-applied-numerical.md](topic-m-applied-numerical.md) | ✅ Done |
| L | Symbolic differentiation (Bird Ch. 27–33) | [topic-l-symbolic-diff.md](topic-l-symbolic-diff.md) | ✅ Done |
| N | Symbolic integration (Bird Ch. 37, 39, 43) | [topic-n-symbolic-integration.md](topic-n-symbolic-integration.md) | ✅ Done |
| E | Electronics engineering (impedance, phasors, AC power, Bode, filters, transmission lines, semiconductors) | [topic-e-electronics.md](topic-e-electronics.md) | ✅ Done |
| — | Acceptance suite: all 16 §Self-test questions | [test/syllabus_selftest_test.dart](../test/syllabus_selftest_test.dart) | ✅ Passing |
| — | Full-book Bird plan (Ch 1–71) | [bird-full-plan.md](bird-full-plan.md) | 🗺 Roadmap |
| — | Plain-English companion (no jargon) | [plain-english/README.md](plain-english/README.md) | ✅ Done |
| — | Session recap (2026-09-12) — Fourier transforms + Flutter app | [session-2026-09-12.md](session-2026-09-12.md) | 📝 Log |

## Getting started

```dart
import 'package:dart_eng_math/dart_eng_math.dart';

void main() {
  // Topic 1: Simpson's rule reproducing arctan(1) = π/4.
  final piOver4 = simpson((x) => 1 / (1 + x * x), 0, 1, 4);

  // Topic 2: sum to infinity of a decaying GP.
  final total = gpSumToInfinity(400, 0.9);

  // Topic 3: cube roots of 8j.
  final roots = complexRoots(Complex(0, 8), 3);

  // Topic 4: (11 − 3x) / [(x−1)(x+3)] → 2/(x−1) − 5/(x+3).
  final pf = partialFractions(
    Polynomial([11, -3]),
    [LinearFactor(1), LinearFactor(-3)],
  );

  // Topic 5: Cramer's rule for 3x + 2y = 12, 5x − y = 7.
  final xy = solveCramer(
    Matrix.fromRows([[3, 2], [5, -1]]),
    [12, 7],
  ); // [2, 3]

  // Topic 6: RL step response — i' + 2i = 12, i(0) = 0 → 6(1 − e^{-2t}).
  final i = solveOdeLaplace(
    coefficients: [2, 1],
    initialConditions: [0],
    forcing: TimeExpr.constant(12),
    denominatorFactors: [LinearFactor(-2), LinearFactor(0)],
  );
}
```

## Conventions

- **Syllabus reference**: chapter/section numbers from
  [A-advanced-mathematics-2.md](../A-advanced-mathematics-2.md).
- **Bird reference**: `Bird Ch. N §M Problem K` refers to worked
  examples in *Higher Engineering Mathematics* (John Bird).
- **Polynomial coefficients** are ascending: `coefficients[i]` is the
  coefficient of xⁱ.
- **Angles** in radians unless a name ends in `Deg` (e.g. `argumentDeg`,
  `Complex.polarDeg`).
- **`num`** is used for inputs so `int` and `double` are both accepted;
  internal storage is `double`.
- Every worked example that appears in a doc has a corresponding entry
  in [../test/](../test/).

## Repo layout

```
lib/
  dart_eng_math.dart            barrel (public API)
  src/                          per-topic modules
test/                           one file per module + Bird-anchored cases
docs/                           this folder
A-advanced-mathematics-2.md     syllabus
Higher Engineering Mathematics - John Bird.pdf
```


---

## FILE: bird-full-plan.md

# Bird full-book implementation plan

Roadmap for covering the **entirety** of Bird's *Higher Engineering
Mathematics* (71 chapters). Chapters already covered by the
027-A syllabus phases (A–F) are marked ✅.

## Legend

| Marker | Meaning |
|---|---|
| ✅ | Fully implemented and tested. |
| 🟢 | Implemented for the parts the 027-A syllabus needed; may extend later. |
| ⏳ | Planned for a future phase (specified). |
| 📄 | Paper-only / conceptual chapter — no code needed. |
| ➖ | Out of scope for this library (e.g. non-math or trivial with `dart:math`). |

## Chapter-by-chapter status

### Section: Number and Algebra

| Ch | Title | Status | Phase | Notes |
|---|---|---|---|---|
| 1 | Algebra (basic laws, poly division, factor theorem) | 🟢 | Phase A/D | `Polynomial.divmod` + factor theorem via roots. Basic laws are conceptual. |
| 2 | Partial fractions | ✅ | Phase D | All three cases + improper. |
| 3 | Logarithms | ✅ | Phase K | Fully covered by `dart:math` primitives; no wrapper needed. |
| 4 | Exponentials and Napierian logarithms | 🟢 | Phase A | Ch 4.2 power series done; growth/decay + reduction to linear form pending. |
| 5 | Hyperbolic functions | ✅ | Phase K | sinh/cosh/tanh/coth/sech/cosech + inverses + identities. |
| 6 | Arithmetic and geometric progressions | ✅ | Phase B | |
| 7 | Binomial series | ✅ | Phase B | |
| 8 | Maclaurin's series | ✅ | Phase A | |
| 9 | Iterative methods (bisection, Newton-Raphson) | ✅ | Phase M | `bisection`, `newtonRaphson`, `secant`. |
| 10 | Binary, octal, hexadecimal | ➖ | — | Dart `int.toRadixString`. |

### Section: Geometry and Trigonometry

| Ch | Title | Status | Phase | Notes |
|---|---|---|---|---|
| 11 | Introduction to trigonometry | 🟢 | Phase K | Sine + cosine rules, triangle areas (SAS + Heron). |
| 12 | Cartesian and polar co-ordinates | ✅ | Phase K | `polarToCartesian`, `cartesianToPolar`. |
| 13 | Circles and their properties | ✅ | Phase K | Arc length, sector area, linear/angular velocity. |
| 14 | Trigonometric waveforms | ✅ | Phase K | `SinusoidalWaveform` (amplitude, freq, period, phase). |
| 15 | Trigonometric identities and equations | 📄 | — | Symbolic — verify numerically as needed. |
| 16 | Trig and hyperbolic relationships | 📄 | — | Osborne's rule — identity-level. |
| 17 | Compound angle formulae | 🟢 | Phase K | `combineSinCos` implements the R·sin(ωt + α) conversion (§17.2). |

### Section: Graphs

| Ch | Title | Status | Phase | Notes |
|---|---|---|---|---|
| 18 | Functions and their curves | ⏳ | Phase K | Curve-sketching helpers; even/odd/periodic detection. |
| 19 | Irregular areas, volumes and mean values | ✅ | Phase M | `meanValue`, `rmsValue`, `volumeOfRevolution`. |

### Section: Complex numbers

| Ch | Title | Status | Phase | Notes |
|---|---|---|---|---|
| 20 | Complex numbers | ✅ | Phase C | |
| 21 | De Moivre's theorem | ✅ | Phase C | |

### Section: Matrices and Determinants

| Ch | Title | Status | Phase | Notes |
|---|---|---|---|---|
| 22 | The theory of matrices and determinants | ✅ | Phase E / P | Base matrix ops in E; LU-based determinant in P. |
| 23 | Simultaneous equations by matrices | ✅ | Phase E / P | Cramer + inverse + Gauss–Jordan (E); LU with partial pivoting + QR least-squares (P). |

### Section: Vector Geometry

| Ch | Title | Status | Phase | Notes |
|---|---|---|---|---|
| 24 | Vectors | ✅ | Phase J | `Vector` value type; arithmetic, dot/cross/angle/magnitude. |
| 25 | Methods of adding alternating waveforms (phasors) | ✅ | Phase K | `phasorSum` via complex addition. |
| 26 | Scalar and vector products | ✅ | Phase J | Dot and cross products, angle between vectors. |

### Section: Differential Calculus

| Ch | Title | Status | Phase | Notes |
|---|---|---|---|---|
| 27 | Methods of differentiation | ✅ | Phase L | Symbolic AST + differentiate() rules. |
| 28 | Some applications of differentiation | 📄 | Phase M | Numeric root-finding = turning points; rates/velocity via `differentiate` + `eval`. |
| 29 | Differentiation of parametric equations | 📄 | Phase L | dy/dx = (dy/dt) / (dx/dt) using differentiate twice. |
| 30 | Differentiation of implicit functions | 📄 | Phase L | Isolate dy/dx symbolically. |
| 31 | Logarithmic differentiation | 📄 | Phase L | `Ln(product)` → differentiate the sum. |
| 32 | Differentiation of hyperbolic functions | ✅ | Phase L | `d(sinh) = cosh` etc. |
| 33 | Differentiation of inverse trig / hyp functions | ✅ | Phase L | Full inverse-trig + inverse-hyp rules. |
| 34 | Partial differentiation | ✅ | Phase J | Numeric partials + gradient + Hessian. |
| 35 | Total differential, rates of change, small changes | ✅ | Phase J | `totalDifferential`. |
| 36 | Maxima, minima, saddle points of f(x, y) | ✅ | Phase J | `classifyCriticalPoint2D` via Hessian discriminant. |

### Section: Integral Calculus

| Ch | Title | Status | Phase | Notes |
|---|---|---|---|---|
| 37 | Standard integration | ✅ | Phase A/N | Numeric via `trapezium`/`simpson`; symbolic via `integrate`. |
| 38 | Some applications of integration | ✅ | Phase M | `curveLength`, `surfaceOfRevolution`, `volumeOfRevolution`, `centroid`. |
| 39 | Integration using algebraic substitutions | 🟢 | Phase N | Linear-inner substitution `∫ f(a·v + b) dv`. Other substitutions paper-only. |
| 40 | Integration using trig / hyp substitutions | ✅ | Phase N | Standard patterns: `∫ dv/(a²±v²)`, `∫ dv/√(a²±v²)`, `∫ dv/√(v²−a²)`. |
| 41 | Integration using partial fractions | ✅ | Phase D/N | Compose `partialFractions` + `integrate` term-by-term. |
| 42 | The `t = tan(θ/2)` substitution | � | Phase N | Closed-form patterns for `∫ dθ/(α + β cos θ)` and `∫ dθ/(α + β sin θ)` when `α > |β|`. General Weierstrass driver not implemented. |
| 43 | Integration by parts | ✅ | Phase N | LIATE, recursive; handles double by-parts. |
| 44 | Reduction formulae | � | Phase N | Sinⁿ / cosⁿ reduction for integer n ≥ 2 with linear inner. Tanⁿ, secⁿ, sinᵐ·cosⁿ not implemented. |
| 45 | Numerical integration (trapezium, Simpson, mid-ordinate) | ✅ | Phase A/M | Trapezium + Simpson (A) + mid-ordinate (M). |

### Section: Differential Equations

| Ch | Title | Status | Phase | Notes |
|---|---|---|---|---|
| 46 | 1st order ODE by separation of variables | 📄 | — | Symbolic — caller verifies with numerical solvers (Phase I) or Laplace (Phase F). |
| 47 | Homogeneous 1st order ODE | 📄 | — | Ditto. |
| 48 | Linear 1st order ODE | 📄 | — | Integrating-factor method — same story. |
| 49 | Numerical methods for 1st order ODE (Euler, RK) | ✅ | Phase I | Euler, improved Euler, RK4; scalar + system. |
| 50 | 2nd order homogeneous ODE | ✅ | Phase F | Solved via Laplace ODE. Add characteristic-poly path in Phase I. |
| 51 | 2nd order forced ODE | 🟢 | Phase F | Laplace path works for exp/poly/trig forcing. |
| 52 | Power series methods for ODE | ⏳ | *(later)* | Frobenius — needs symbolic Expr layer. |
| 53 | Introduction to partial differential equations | ✅ | Phase T | Heat (Crank–Nicolson), wave (leapfrog), Laplace (SOR) — see `solveHeatEquation`, `solveWaveEquation`, `solveLaplaceEquation`. |

### Section: Statistics and Probability

| Ch | Title | Status | Phase | Notes |
|---|---|---|---|---|
| 54 | Presentation of statistical data | 📄 | — | Histograms/ogives are display concerns; data structures live in Ch 55. |
| 55 | Measures of central tendency and dispersion | ✅ | Phase G | Mean, median, mode, variance, stdev, quartiles, grouped data. |
| 56 | Probability | ✅ | Phase G/O | Basic laws + expectation + `conditional` + `bayes` (Phase O). |
| 57 | Binomial and Poisson distributions | ✅ | Phase G | PMF, CDF, mean, variance. |
| 58 | Normal distribution | ✅ | Phase G/O | PDF/CDF via erf; Acklam quantile (Phase O). |
| 59 | Linear correlation | ✅ | Phase G | Pearson r. |
| 60 | Linear regression | ✅ | Phase G / P | Least-squares Y-on-X and X-on-Y (G); multi-predictor OLS via QR (P). |

Phase O extras (not numbered in this edition's chapter list but standard
Bird content): Student's t, chi-squared and F distributions; confidence
intervals (mean-z / mean-t / proportion); one- and two-sample t and z
tests; paired t; chi-square goodness-of-fit and independence; one-way
ANOVA; sign test; Wilcoxon signed-rank; Mann–Whitney U.
See [docs/topic-g-statistics.md](topic-g-statistics.md#phase-o--inferential-statistics).

Phase P — general linear algebra: LU with partial pivoting, Householder
QR, arbitrary-`n` least squares, and multiple linear regression via QR.
See [docs/topic-5-linear-algebra.md](topic-5-linear-algebra.md#lu-and-qr-decompositions-phase-p).

Phase Q — practical numerical methods: Cholesky factorisation for
symmetric-PD systems; Lagrange, Newton forward/backward and natural
cubic-spline interpolation; least-squares polynomial fit via
Vandermonde + QR. See
[docs/topic-m-applied-numerical.md](topic-m-applied-numerical.md#interpolation-and-polynomial-fitting-phase-q).

Phase R — general eigenvalue decomposition and PCA: cyclic-Jacobi
factorisation of any real symmetric matrix; power iteration for the
dominant eigenpair; Rayleigh quotient iteration for cubic convergence
near a target eigenvalue; principal-component analysis composed on top
of Jacobi. See
[docs/topic-5-linear-algebra.md](topic-5-linear-algebra.md#general-eigenvalue-decomposition-phase-r)
and [docs/topic-g-statistics.md](topic-g-statistics.md#phase-r--principal-component-analysis).

Phase S — non-linear least squares: Levenberg–Marquardt fits arbitrary
`y = f(x, β)` models to data with self-adjusting damping. Ships with
Gaussian, exponential-decay, and damped-sinusoid pre-baked models plus
their analytic Jacobians; falls back to central-difference numerical
Jacobians for user-defined models. Returns best-fit parameters,
covariance matrix, and per-parameter 1-σ standard errors. See
[docs/topic-m-applied-numerical.md](topic-m-applied-numerical.md#non-linear-curve-fitting-phase-s).

Phase T — canonical linear PDE solvers (Bird Ch. 53): heat equation
via Crank–Nicolson (unconditionally stable, tridiagonal Thomas solve
per timestep), wave equation via leapfrog (CFL-checked), Laplace on a
rectangle via SOR with the optimal-ω default. See
[docs/topic-i-ode-methods.md](topic-i-ode-methods.md#partial-differential-equations-phase-t).

Phase U — Singular Value Decomposition: `A = U · Σ · Vᵀ` via one-sided
cyclic Jacobi. Unlocks the Moore–Penrose pseudoinverse, effective
numerical rank, 2-norm condition number, and minimum-norm
least-squares for rank-deficient / under-determined systems where the
Phase P QR route throws. Not tied to any Bird chapter but the last big
gap in the general linear-algebra layer. See
[docs/topic-5-linear-algebra.md](topic-5-linear-algebra.md#singular-value-decomposition-phase-u).

### Section: Laplace Transforms

| Ch | Title | Status | Phase | Notes |
|---|---|---|---|---|
| 61 | Introduction to Laplace transforms | ✅ | Phase F | |
| 62 | Properties of Laplace transforms (first shift) | ✅ | Phase F | Also initial/final value theorems — deferred. |
| 63 | Inverse Laplace transforms | ✅ | Phase F | Includes poles/zeros — deferred. |
| 64 | Solution of ODE via Laplace | ✅ | Phase F | |
| 65 | Simultaneous ODE via Laplace | ✅ | Phase I | `solveLinear2x2OdeLaplace` for 2×2 systems. |

### Section: Fourier Series

| Ch | Title | Status | Phase | Notes |
|---|---|---|---|---|
| 66 | Fourier series for periodic functions of period 2π | ✅ | Phase H | `FourierSeries.fromFunction`. |
| 67 | Fourier series for non-periodic functions over range 2π | ✅ | Phase H | Same API — caller supplies interval. |
| 68 | Even/odd functions and half-range Fourier series | 🟢 | Phase H | Even/odd emerge naturally; half-range = build the extension by hand. |
| 69 | Fourier series over any range | ✅ | Phase H | `period` parameter. |
| 70 | Numerical harmonic analysis | ✅ | Phase H | `FourierSeries.fromSamples`. |
| 71 | Complex/exponential form of Fourier series | ✅ | Phase H | `ComplexFourierSeries`. |

## Proposed phase ordering

Phase G onward each has a clear scope and can be executed independently.
Each phase follows the established convention:
- One file per module under `lib/src/`.
- One test file per module in `test/`, Bird-anchored.
- One doc under `docs/topic-N-<name>.md`.

| Phase | Scope | Depends on | Est. new tests |
|---|---|---|---|
| **G — Statistics & probability** ✅ | Ch 54–60 | — | 67 delivered |
| **H — Fourier series** ✅ | Ch 66–71 | Phase A (integration), C (complex) | 23 delivered |
| **I — ODE methods** ✅ | Ch 49, 65 (46–48, 52 paper/deferred) | Phase F | 17 delivered |
| **J — Vectors + partial derivatives** ✅ | Ch 24, 26, 34–36 | Phase A, E | 40 delivered |
| **K — Trig / hyperbolic / waveforms** ✅ | Ch 3–5, 11–14, 17, 25 | Phase C | 51 delivered |
| **L — Symbolic differentiation** ✅ | Ch 27–33 | *(new: `Expr` AST)* | 33 delivered |
| **M — Applied numerical methods** ✅ | Ch 9, 19, 37–38, 45 | Phase A | 34 delivered |
| **N — Symbolic integration** ✅ | Ch 37, 39, 43 | Phase L | 28 delivered |
| **M — Applied numerical methods** | Ch 9, 19, 37–38 | Phase A | ~30 |

Total new tests projected (G–N): ~275; Phases G+H+I+J+K+L+M+N delivered
293. Suite currently at **505** tests.

## Foundational modules to add across phases

The following will be introduced early in whichever phase first needs them:

- **`RandomVariable` / probability distributions** (Phase G) — trait for
  PMF/PDF/CDF/quantile/expected value/variance.
- **`ExprLinear`** — polynomial or numeric expression on `x` used by
  symbolic-lite ODE Phase I; distinct from the general symbolic `Expr`.
- **`Expr` symbolic AST** (Phase L) — sum-of-products with variables,
  constants, standard functions. Differentiation via the classical rules
  (product, quotient, chain). Simplification rules kept minimal.
- **`Vector`** (Phase J) — flat `List<double>` value type for
  small dimensions; reuses `Matrix` where advantageous.

## Non-goals

- **Full CAS.** Bird uses hand-derived symbolic manipulation. We can add
  a limited symbolic layer (Phase L) but full CAS-quality integration is
  out of scope.
- **Perfect fidelity to Bird's presentation of trivial chapters.** Ch 3
  (logarithms), Ch 10 (number bases) etc. are conceptually covered by
  `dart:math` and language primitives.
- **PDE numerical solvers** (Ch 53). Conceptual mention only unless a
  specific application requires it.

## Status: all planned phases complete

All Bird chapters that map to a numeric or symbolic module have been
implemented. The remaining paper-only chapters (integration by
substitution / parts / reduction, PDEs, symbolic identity manipulation)
are well-served by the numerical machinery + the symbolic
differentiator; they do not warrant additional dedicated modules.

477 tests, all Bird-anchored where a numeric ground truth was
available. The library is ready for use across the entirety of *Higher
Engineering Mathematics*.


---

## FILE: plain-english/README.md

# Plain-English guide

Every topic in [../README.md](../README.md) explained without the jargon.

Each page here answers three questions:

1. **What is it, really?** — one paragraph, no symbols.
2. **Where does it show up in daily life?** — concrete examples.
3. **Why does an engineer care?** — the problem it solves at work.

Then it links to the formal doc for the actual formulas and code.

## Topic map

| # | Topic | Plain-English page | Formal doc |
|---|---|---|---|
| 1 | Further calculus | [topic-1-further-calculus.md](topic-1-further-calculus.md) | [../topic-1-further-calculus.md](../topic-1-further-calculus.md) |
| 2 | Series (AP, GP, binomial) | [topic-2-series.md](topic-2-series.md) | [../topic-2-series.md](../topic-2-series.md) |
| 3 | Complex numbers | [topic-3-complex-numbers.md](topic-3-complex-numbers.md) | [../topic-3-complex-numbers.md](../topic-3-complex-numbers.md) |
| 4 | Partial fractions | [topic-4-partial-fractions.md](topic-4-partial-fractions.md) | [../topic-4-partial-fractions.md](../topic-4-partial-fractions.md) |
| 5 | Linear algebra | [topic-5-linear-algebra.md](topic-5-linear-algebra.md) | [../topic-5-linear-algebra.md](../topic-5-linear-algebra.md) |
| 6 | Laplace transforms | [topic-6-laplace.md](topic-6-laplace.md) | [../topic-6-laplace.md](../topic-6-laplace.md) |
| G | Statistics & probability | [topic-g-statistics.md](topic-g-statistics.md) | [../topic-g-statistics.md](../topic-g-statistics.md) |
| H | Fourier series & transforms | [topic-h-fourier.md](topic-h-fourier.md) | [../topic-h-fourier.md](../topic-h-fourier.md) |
| I | ODE methods | [topic-i-ode-methods.md](topic-i-ode-methods.md) | [../topic-i-ode-methods.md](../topic-i-ode-methods.md) |
| J | Vectors | [topic-j-vectors.md](topic-j-vectors.md) | [../topic-j-vectors.md](../topic-j-vectors.md) |
| K | Trig, hyperbolic, waveforms | [topic-k-trig-hyperbolic.md](topic-k-trig-hyperbolic.md) | [../topic-k-trig-hyperbolic.md](../topic-k-trig-hyperbolic.md) |
| L | Symbolic differentiation | [topic-l-symbolic-diff.md](topic-l-symbolic-diff.md) | [../topic-l-symbolic-diff.md](../topic-l-symbolic-diff.md) |
| M | Applied numerical methods | [topic-m-applied-numerical.md](topic-m-applied-numerical.md) | [../topic-m-applied-numerical.md](../topic-m-applied-numerical.md) |
| N | Symbolic integration | [topic-n-symbolic-integration.md](topic-n-symbolic-integration.md) | [../topic-n-symbolic-integration.md](../topic-n-symbolic-integration.md) |


---

## FILE: plain-english/topic-1-further-calculus.md

# Topic 1 — Further calculus (plain English)

Formal doc: [../topic-1-further-calculus.md](../topic-1-further-calculus.md).

## What is it, really?

Calculus is the maths of **change** and **accumulation**.

- **Differentiation** — "how fast is this thing changing right now?"
  Speedometer vs odometer: the speedometer reads the derivative of the
  odometer.
- **Integration** — "add up a bunch of tiny changing bits to get a total."
  Fill level of a tank when the tap is being turned on and off.

"Further" calculus just means we use tricks (numerical rules like the
trapezoidal and Simpson's rule) when the shape is too messy to solve
with pen-and-paper formulas.

## Where does it show up?

- Working out the **area under a curve** on a graph when you only have
  measured points, not a formula.
- Estimating the **volume of a lake** from depth soundings.
- Fuel used on a journey when the throttle keeps changing.

## Why does an engineer care?

Real signals from sensors don't come with tidy equations — they come
as columns of numbers. Numerical integration (`trapezoid`, `simpson`)
lets you compute totals directly from the data.

## How the formulas were derived (plain English)

**Trapezoidal rule.** Cut the area under the curve into thin vertical
strips. Each strip is *almost* a rectangle, but the top is slanted.
A slanted-top strip is a **trapezium**, and its area is
"width × average of the two heights". Add up all those trapezium
areas and you have an estimate of the total area.

**Simpson's rule.** Same trick, but instead of joining the tops of
two neighbouring points with a straight line, you join **three**
points with a smooth curve (a **parabola**) that touches all three.
A parabola hugs a real curve much closer than a straight line does,
so the estimate is far more accurate for the same amount of work.
Simpson worked out that the area under any parabola through three
equally-spaced points is `(width / 3) × (first + 4·middle + last)`
— that's where the funny `1, 4, 2, 4, 2, …, 4, 1` weights come from.

**Derivatives from data.** "How fast is it changing?" ≈ "how much
did it change ÷ how long did that take". That's just rise over run
on tiny intervals — the definition of a derivative, applied to a
table of numbers.


---

## FILE: plain-english/topic-2-series.md

# Topic 2 — Series (plain English)

Formal doc: [../topic-2-series.md](../topic-2-series.md).

## What is it, really?

A **series** is what you get when you keep adding numbers that follow
a rule.

- **Arithmetic progression (AP)** — add the same amount each step:
  `2, 5, 8, 11, 14, …` (step of 3).
- **Geometric progression (GP)** — multiply by the same factor each
  step: `1000, 500, 250, 125, …` (halve each time).
- **Binomial series** — the pattern of coefficients you get when you
  expand something like `(a + b)⁵`. Same numbers as **Pascal's
  triangle**.

## Where does it show up?

- **AP** — a pay rise of £2,000 every year; stacking bricks so each
  row loses one brick.
- **GP** — compound interest, a bouncing ball losing a fixed
  percentage of height on each bounce, radioactive decay.
- **Binomial** — the number of ways to pick a team, coin-flip
  probabilities (`5 heads out of 10 tosses`), lottery odds.

## Why does an engineer care?

- Loan repayments and depreciation schedules are geometric.
- Signal levels going through a chain of amplifiers/attenuators
  multiply — that's geometric.
- Probability of `k` defective units in a batch of `n` uses the
  binomial coefficient `C(n, k)`.

## How the formulas were derived (plain English)

**AP sum — Gauss's schoolboy trick.** Write the sum forwards, then
write it again backwards underneath, and add column by column:

```
1 + 2 + 3 + … + 99 + 100
100 + 99 + 98 + … +  2 +   1
---------------------------------
101 + 101 + 101 + … + 101 + 101   ← 100 identical pairs
```

That gives `100 × 101`, but you added the sum twice, so divide by 2.
Generalise: `Sₙ = n/2 · (first + last)`. Every AP sum formula is
just this pairing trick.

**GP sum — the "multiply and subtract" trick.** Write
`S = a + ar + ar² + … + arⁿ⁻¹`. Multiply the whole line by `r`:
`rS = ar + ar² + … + arⁿ`. Subtract the two lines — nearly
everything cancels, leaving `S − rS = a − arⁿ`. Divide by `1 − r`
and you get the tidy `Sₙ = a(1 − rⁿ) / (1 − r)`. If `|r| < 1` and
you let `n` run to infinity, `rⁿ` shrinks to zero and the formula
collapses to `S∞ = a / (1 − r)`.

**Binomial coefficient — count then un-count the order.** How many
ways to pick `k` items from `n` if order **did** matter?
`n × (n − 1) × … × (n − k + 1)` — that's `n! / (n − k)!`. But the
same `k` items can be ordered `k!` different ways, and we don't
care about the order, so divide by `k!`. Result:
`C(n, k) = n! / (k!(n − k)!)`.

**Binomial theorem — where do the coefficients come from?** When
you expand `(a + x)(a + x)(a + x)…` (n copies), a typical term is
made by picking either `a` or `x` from each bracket. The number of
ways to end up with exactly `k` copies of `x` (and `n − k` copies of
`a`) is *exactly* `C(n, k)`. So each term is
`C(n, k) · aⁿ⁻ᵏ · xᵏ`.


---

## FILE: plain-english/topic-3-complex-numbers.md

# Topic 3 — Complex numbers (plain English)

Formal doc: [../topic-3-complex-numbers.md](../topic-3-complex-numbers.md).

## What is it, really?

A **complex number** is a number with two parts: a "left–right"
part (called *real*) and an "up–down" part (called *imaginary*).
Think of it as an **arrow on graph paper** with a length and a
direction.

- Length = **magnitude** ("how big").
- Direction = **phase / angle** ("which way it's pointing").

The letter `j` (or `i`) just means "turn 90° left". Multiplying by
`j` rotates the arrow a quarter turn.

## Where does it show up?

- **AC electricity** — voltage and current aren't just "how big",
  they're also "in-step or out-of-step". A complex number captures
  both in one go.
- **Rotating things** — motors, gears, compass bearings.
- **GPS and radio** — signals have amplitude *and* phase.

## Why does an engineer care?

Kirchhoff's laws for AC circuits become simple algebra when you use
complex numbers ("phasors"). Instead of chasing sines and cosines
through pages of trig, you multiply and add arrows. Euler's formula
(`e^{jθ} = cos θ + j sin θ`) and de Moivre's theorem are the shortcuts
that make it all click.

## How the formulas were derived (plain English)

**Why `j² = −1`.** Say `j` means "rotate 90° anti-clockwise".
Doing it once takes `+1` on the real line and points it up to `+j`.
Doing it **twice** takes `+1` all the way round to `−1`. So
`j × j = −1`. That's the whole magic — `j` isn't spooky, it's just a
quarter-turn operator.

**Multiplication = rotate + stretch.** If you write two arrows as
(length, angle), multiplying them **multiplies the lengths and adds
the angles**. That single rule explains why complex multiplication
is so useful for anything that spins.

**Euler's formula, without the calculus.** Everyone knows the Taylor
series for `eˣ`, `sin x`, `cos x` — infinite polynomials. Substitute
`x = jθ` into the `eˣ` series and split the terms into even and odd
powers of `jθ`. The even ones (using `j² = −1`) rebuild the `cos θ`
series exactly; the odd ones rebuild `j · sin θ`. So
`e^{jθ} = cos θ + j sin θ`. It's not a definition — it falls out of
the series when you're allowed to plug `jθ` in.

**De Moivre's theorem.** Once Euler's formula is in hand,
`(cos θ + j sin θ)ⁿ = (e^{jθ})ⁿ = e^{jnθ} = cos(nθ) + j sin(nθ)`.
Raising an arrow to the power `n` just multiplies its angle by `n`.


---

## FILE: plain-english/topic-4-partial-fractions.md

# Topic 4 — Partial fractions (plain English)

Formal doc: [../topic-4-partial-fractions.md](../topic-4-partial-fractions.md).

## What is it, really?

**Partial fractions** is the reverse of "putting fractions over a
common denominator".

You know how `1/2 + 1/3 = 5/6`? Partial fractions goes the other way:
given `5/6`, split it back into `1/2 + 1/3`. Same idea, but for
algebraic fractions like

$$\frac{11 - 3x}{(x-1)(x+3)} \;=\; \frac{2}{x-1} \;-\; \frac{5}{x+3}$$

The big messy fraction becomes a sum of small, tidy ones.

## Where does it show up?

Anywhere you have to **integrate** or **inverse-Laplace** a
complicated fraction. Splitting it up first turns one hard problem
into several easy ones you already know how to solve.

## Why does an engineer care?

- **Control systems** — every transfer function is a fraction of
  polynomials. Partial fractions turn it into a sum of standard
  building blocks (exponentials, sinusoids) whose response you can
  read off a table.
- **Circuit analysis** — same idea for Laplace-domain currents and
  voltages.

## How the formulas were derived (plain English)

**The "cover-up" trick.** Suppose you believe that

$$\frac{11 - 3x}{(x-1)(x+3)} = \frac{A}{x-1} + \frac{B}{x+3}.$$

To find `A`, multiply both sides by `(x − 1)` — that cancels the
`(x − 1)` on the left and turns the right into
`A + B(x−1)/(x+3)`. Now set `x = 1`, which kills the `B` term and
reads off `A` directly from the left-hand side. Repeat with
`x = −3` to get `B`. In practice you don't even multiply — you
**cover up** the `(x − 1)` factor on the left, plug `x = 1` into
what's left, and you have `A`. The formulas are just "compare
coefficients", but the cover-up is the fastest way to see them.

**Why it always works.** Two polynomials are equal only if their
coefficients match term-by-term. So if you *insist* that the sum on
the right equals the fraction on the left, you get one equation per
coefficient — always exactly enough equations to pin down the
unknowns `A`, `B`, `C`, … no matter how many factors the
denominator has.

**Repeated factors.** If `(x − 2)²` appears in the denominator, you
need **two** slots: `A/(x−2) + B/(x−2)²`. Why? Because one slot
alone can't reproduce every possible top when you put it over the
common denominator `(x − 2)²`. Same logic for higher powers.


---

## FILE: plain-english/topic-5-linear-algebra.md

# Topic 5 — Linear algebra (plain English)

Formal doc: [../topic-5-linear-algebra.md](../topic-5-linear-algebra.md).

## What is it, really?

**Linear algebra** is the maths of **many equations at once** and the
**grids of numbers** (matrices) that hold them.

- A **matrix** is just a rectangle of numbers — like a spreadsheet.
- A **determinant** is a single number that tells you whether a
  matrix is "well-behaved" (non-zero → solvable; zero → the equations
  are secretly the same or contradictory).
- Solving a system like *"3x + 2y = 12, 5x − y = 7"* by hand is
  tedious; matrices let you crank the handle mechanically.

## Where does it show up?

- **Balancing chemical equations.**
- **Circuit mesh/node analysis** — a 10-loop circuit is a 10×10
  system.
- **Structural engineering** — forces in a truss.
- **Computer graphics** — rotating and scaling 3D objects is matrix
  multiplication.
- **Machine learning and data science** — every dataset is a matrix.

## Why does an engineer care?

Real problems rarely have one unknown. Matrices let a computer solve
hundreds of coupled equations in a blink — Cramer's rule,
Gauss elimination, eigenvalues (natural resonance frequencies), and
SVD (finding the "main directions" in data) all live here.

## How the formulas were derived (plain English)

**Determinant of a 2×2 = area of a parallelogram.** Draw the two
columns of the matrix as arrows starting from the origin. They
span a parallelogram. Its area works out (from basic geometry)
to `ad − bc`. If the two arrows point the same way, they don't
span any area — determinant is zero, and the equations are
secretly the same line. That's why "det = 0 ⇒ no unique solution".
For 3×3, the determinant is the **volume** of the parallelepiped
formed by the three columns; same story in higher dimensions.

**Cramer's rule — ratios of areas.** For a 2×2 system
`ax + by = e`, `cx + dy = f`, replace the first column with
`(e, f)` and take the determinant — that gives you the *scaled*
parallelogram area "in the x direction". Divide by the original
determinant and the scale drops out, leaving `x`. Same trick for
`y` by replacing the second column. It's not magic — it's the
same area formula applied twice.

**Gauss elimination — the row-swap-and-subtract game.** Any
legal move on the equations (multiply a row by a constant, add
one row to another, swap two rows) doesn't change the solutions.
Gauss's recipe uses these moves to zero out entries below the
diagonal, one column at a time, until the last equation has just
one unknown. Then back-substitute upwards. It's the pen-and-paper
method, mechanised.

**Eigenvalues — the special directions.** For most vectors,
"multiply by the matrix" both rotates and stretches. But every
matrix has a few special directions where it **only stretches**,
no rotation — like the resonant frequencies of a bell. The stretch
factor is the **eigenvalue**, the direction is the **eigenvector**.
Solving `Ax = λx` is really asking "which direction survives
unchanged in orientation?".


---

## FILE: plain-english/topic-6-laplace.md

# Topic 6 — Laplace transforms (plain English)

Formal doc: [../topic-6-laplace.md](../topic-6-laplace.md).

## What is it, really?

The **Laplace transform** is a translator. It takes a problem written
in the language of **time** ("how does the current change every
millisecond?") and rewrites it in the language of **`s`** (a kind of
frequency), where the same problem becomes simple algebra instead of
a differential equation.

Rough analogy: multiplying big numbers is hard, but adding their
**logarithms** is easy. Laplace is the "logarithm trick" for
differential equations — take the transform, do easy algebra, then
transform back.

## Where does it show up?

- Every **RLC circuit** step or impulse response.
- **Control systems** — motors, thermostats, cruise control, drones.
- **Mechanical systems** — mass-spring-damper (car suspension).

## Why does an engineer care?

You avoid solving differential equations from scratch. You:

1. Transform the equation (`d/dt` becomes multiplication by `s`).
2. Rearrange like ordinary algebra.
3. Split with **partial fractions** (topic 4).
4. Look up each piece in a small table.

Out pops the time-domain answer — voltage, current, position — with
no calculus wrestling.

## How the formulas were derived (plain English)

**What the integral is doing.** The Laplace transform is
`F(s) = ∫₀^∞ e^{−st} · f(t) dt`. The `e^{−st}` bit is a "weighting
curtain" that fades exponentially with time. You're essentially
taking a **weighted total** of the whole signal `f(t)`, with recent
values counting more than distant ones (how quickly is set by `s`).
Different `s` values probe different "speeds" of the signal — the
complete set of totals over all `s` uniquely captures the signal.

**Why `d/dt` becomes multiplying by `s`.** Apply integration by
parts to `∫₀^∞ e^{−st} · f'(t) dt`. The boundary term at infinity
dies (because `e^{−st}` kills anything reasonable), the boundary
term at zero gives `−f(0)`, and the remaining integral is
`s · F(s)`. So the transform of `f'(t)` is `sF(s) − f(0)`. That one
line of algebra is why a whole differential equation collapses
into an algebraic one: every `d/dt` you had turns into a
multiplication by `s`.

**Why partial fractions come next.** After the algebra, `F(s)` is
usually some polynomial-over-polynomial mess. The Laplace table
only knows a small set of tidy building blocks (`1/(s+a)`,
`s/(s²+ω²)`, etc.). Partial fractions (topic 4) chop the mess into
exactly those building blocks, and then you look each one up.

**Inverse transform — the reverse dictionary.** In practice nobody
computes the inverse integral. You just recognise each partial
fraction in the table and write down its time-domain twin.


---

## FILE: plain-english/topic-g-statistics.md

# Topic G — Statistics & probability (plain English)

Formal doc: [../topic-g-statistics.md](../topic-g-statistics.md).

## What is it, really?

**Statistics** is how you make sense of a pile of measurements.
**Probability** is how you predict what's likely to happen next.

- **Mean, median, mode** — three different ways to say "typical".
- **Standard deviation** — how spread out the numbers are.
- **Distributions** (normal / binomial / Poisson) — recipes that say
  "here's the shape data usually takes when …".
- **Hypothesis test / confidence interval** — a way to say "I'm 95%
  sure the true value is between X and Y", so you're not fooled by
  random noise.

## Where does it show up?

- **Quality control** — is this batch of resistors within tolerance?
- **A/B testing** — did the new website design really convert better,
  or did we get lucky?
- **Reliability** — how many hours before this bearing is likely to
  fail?
- **Polls, forecasts, insurance premiums.**

## Why does an engineer care?

No measurement is exact. Statistics tells you how much of what you're
seeing is **signal** and how much is **noise**, so you don't ship a
product based on random luck.

## How the formulas were derived (plain English)

**Mean = fair share.** Add everyone's amount, divide by the number
of people. That's the amount each person would have if you
redistributed evenly. Nothing deeper going on.

**Why we square the deviations for variance.** Some measurements sit
above the mean, some below. If you just averaged the differences,
the positives and negatives cancel and you'd always get zero.
Squaring turns every distance into a positive contribution *and*
punishes big outliers more than small ones (a distance twice as far
counts four times as much). The square root at the end (standard
deviation) puts you back in the original units.

**Binomial distribution — coin flips.** Probability of exactly `k`
heads in `n` flips: pick **which** `k` flips are the heads
(`C(n, k)` ways — see topic 2), multiply by the probability of each
specific pattern (`pᵏ · (1−p)ⁿ⁻ᵏ`). Sum those and you have the
probability. That's `C(n,k) · pᵏ · (1−p)ⁿ⁻ᵏ`.

**Normal distribution — why the bell shape.** When you add up lots
of small independent random effects (measurement error, wind gusts,
tiny manufacturing variations), the total always ends up bell-shaped
regardless of what each individual effect looked like. This is the
**Central Limit Theorem** — it's the reason the same bell curve
appears everywhere in nature.

**Confidence intervals.** "There's a 95% chance the true value lies
within ±2 standard deviations of my measured average" — because 95%
of a bell curve's area sits between the mean ± 1.96σ. The magic
number `1.96` is just "how far do I need to walk from the top of
the bell to leave only 2.5% of the area on each side".


---

## FILE: plain-english/topic-h-fourier.md

# Topic H — Fourier series & transforms (plain English)

Formal doc: [../topic-h-fourier.md](../topic-h-fourier.md).

## What is it, really?

Fourier's big idea: **any shape can be built by stacking pure sine
waves** of different frequencies.

Think of a **music equaliser**: sliders labelled bass, mid, treble.
A Fourier decomposition tells you exactly how high each slider must
be to recreate a given sound — or any signal, no matter how jagged.

- A square wave = a fundamental sine + a bit of its 3rd harmonic +
  a bit of the 5th + …
- A sawtooth = fundamental + half the 2nd + a third of the 3rd + …

## Two flavours

There are two versions of the same idea, depending on the signal:

| Version | For signals that … | Ingredients list is … | This library |
|---|---|---|---|
| **Fourier series** | **repeat forever** (period known) | a **countable** set of harmonics `n = 1, 2, 3, …` | `FourierSeries`, `ComplexFourierSeries` |
| **Fourier transform (CFT)** | are **one-off** in continuous time (a pulse, a chirp) | a **continuous** function `F(ω)` of frequency | `fourierTransform` |
| **Discrete Fourier transform (DFT/FFT)** | are **sampled** on a computer (audio buffer, sensor log) | a **finite** list of complex bins `X[0..N-1]` | `dft`, `fft`, `ifft` |

Same "sines-and-cosines-as-ingredients" idea; three flavours of
book-keeping depending on whether time is periodic, continuous, or
sampled.

## Where does it show up?

- **Steady-state AC analysis** — a non-sinusoidal periodic voltage
  (square, sawtooth, half-wave rectified) broken into its harmonic
  ingredients so each can be pushed through the circuit separately.
- **Vibration analysis of periodic sources** — spot the exact
  harmonic of a rattling engine part running at a fixed speed.
- **Harmonic distortion (THD)** — measure how much of a "pure"
  50/60 Hz mains signal has leaked into the 2nd, 3rd, 5th
  harmonics.
- **Any repeating waveform** whose one-period shape you can describe.

## Why does an engineer care?

Once you know the ingredients (which frequencies, how much of each),
you can **filter** (throw away hum at 50 Hz), **compress** (drop the
tiny ingredients), or **detect** (a specific frequency spike = a
fault).

## How the formulas were derived (plain English)

**Sines and cosines don't interfere.** If you multiply `sin(nx)` by
`sin(mx)` and integrate over a full period, you get **zero** unless
`n = m`. Same for cosines, and for a sine times a cosine. That's the
key property — the harmonics are **orthogonal**, like perpendicular
axes in geometry.

**The extraction trick.** Suppose your signal is
`f(x) = a₁ sin x + a₂ sin 2x + a₃ sin 3x + …`. You want `a₂` on its
own. Multiply the whole equation by `sin 2x` and integrate over one
period. Every term on the right vanishes (orthogonality) **except**
the `sin 2x · sin 2x` one, which integrates to a known constant.
Divide by that constant and out pops `a₂` alone. Do this for each
harmonic and you have all the ingredients.

**Why 1/π appears in the coefficient formulas.** It's the value of
`∫ sin²(nx) dx` over one period. When you divide by it to isolate
each coefficient, that constant lands in the formula. Nothing more
mysterious than that.

**Half-range series (odd/even).** If your signal is a mirror image
around the y-axis (**even**), all the sines cancel and you keep
only cosines. If it's flipped upside-down through the origin
(**odd**), the cosines cancel and you keep only sines. Symmetry
saves you half the work.

**Complex form.** Instead of tracking sines and cosines separately,
Euler's formula (topic 3) rolls them into a single `e^{jnx}`. The
series becomes a sum of spinning arrows at each frequency — same
maths, half the bookkeeping. Implemented here as
`lib/src/fourier_complex.dart`.

## From series to transform (plain English)

**What if the signal doesn't repeat?** Take a periodic signal, make
its period longer and longer, and stretch the ruler on the frequency
axis to keep the picture the same. In the limit, the neighbouring
harmonics get infinitely close together — the "list of ingredients"
becomes a **smooth curve** `F(ω)`, and the sum becomes an integral.
That's the **continuous Fourier transform**:

`F(ω) = ∫ f(t) · e^{−jωt} dt`

Read it as: "at each frequency `ω`, multiply the signal by a probe
sine/cosine of that frequency and average — how much of that
frequency is inside?"

**What about sampled data on a computer?** You don't have a continuous
`f(t)`, just `N` numbers `x[0], x[1], …, x[N−1]`. Same probe idea,
but now the integral is a sum:

`X[k] = Σ x[n] · e^{−j 2π k n / N}`

That's the **DFT** — the discrete cousin. There are exactly `N` output
bins. Bin `k` says: "how much of a sine that fits `k` whole cycles
into the sample buffer is inside the signal?"

**Why "Fast" Fourier Transform (FFT)?** A naïve DFT takes `N²`
multiply-adds. Cooley and Tukey noticed you can split an `N`-point
DFT into two `N/2`-point DFTs (even-indexed samples and odd-indexed
samples), combine them, and recurse. Cost drops from `N²` to
`N log N`. For `N = 1024` that's a ~100× speed-up; for `N = 10⁶`,
~50,000×. Same answer, dramatically less arithmetic. The catch:
easiest form only works when `N` is a power of 2.

**Reading the FFT output.** For a real signal:

- **Bin 0** = DC (average value).
- **Bin k** = the amplitude/phase of the sinusoid that fits `k`
  whole cycles into the buffer, i.e. at frequency `k · f_s / N` Hz.
- **Bins above N/2** are the same real-signal information mirrored
  as "negative frequencies" — usually you throw them away and just
  plot bins `0..N/2`.
- **|X[k]|** = amplitude; **arg X[k]** = phase.

The library gives you both raw (`magnitudeSpectrum`, `phaseSpectrum`)
and a friendlier one-sided view (`oneSidedAmplitudeSpectrum`, scaled
so a pure `2·cos(2πf₀t)` shows a peak of exactly `2.0` at bin `f₀`).


---

## FILE: plain-english/topic-i-ode-methods.md

# Topic I — ODE methods (plain English)

Formal doc: [../topic-i-ode-methods.md](../topic-i-ode-methods.md).

## What is it, really?

An **ODE** (ordinary differential equation) is a rule that says
*"the way this thing is changing depends on where it is right now."*

- A hot cup of coffee cools **faster when it's hotter** — the rate
  of change depends on the current temperature. That's an ODE.
- A car accelerating: how fast the speed changes depends on the
  current speed (drag) and the throttle.

**ODE methods** are step-by-step recipes (Euler, Runge–Kutta) to
predict where the thing will be a moment from now, then a moment
after that, and so on — essentially "flip-book animation" for
physics.

## Where does it show up?

- **Cooling / heating** of anything (Newton's law of cooling).
- **Population growth**, epidemic spread (SIR model).
- **Charging capacitors, RC / RL / RLC circuits.**
- **Projectile with air drag**, satellite orbits.

## Why does an engineer care?

Most laws of physics are naturally written as ODEs. When there's no
neat pen-and-paper solution, numerical methods let a computer march
forward in tiny time steps and draw the answer as a curve.

## How the formulas were derived (plain English)

**Euler's method — take a tiny step in the current direction.** The
ODE tells you the slope `y' = f(x, y)` at any point. Rearrange the
definition of a derivative into words: *"next value ≈ current value
+ slope × tiny step"*. That's literally `yₙ₊₁ = yₙ + h · f(xₙ, yₙ)`.
Repeat and you've simulated the whole curve — like drawing a path
by always heading in whichever direction the local wind blows.

**Why Euler drifts.** The slope changes across the tiny step, but
Euler pretends it stays constant. Each step is a bit wrong; those
errors pile up. Halving the step size only halves the error, so
you pay a lot of steps for a small accuracy gain.

**Runge–Kutta — take several test-shots and average them.** Instead
of trusting the slope at the starting point alone, sample the slope
at several places inside the step (the start, two guesses in the
middle, and the tentative end), then take a **weighted average**
before stepping. RK4 uses four such samples with weights
`1 : 2 : 2 : 1` — those weights are chosen so the error terms
cancel out through order 4. It's like polling four
voters instead of one before deciding which way to walk. Halving
the step now cuts the error by roughly **16×** (not 2×) — hence its
popularity.

**Modified Euler / Heun — the two-shot compromise.** Take an Euler
step, look at the slope there, average with the slope at the start,
re-step with the average. Two slope evaluations, one order more
accurate than plain Euler. Halfway between simple and RK4 in both
cost and accuracy.

**Analytic tricks (integrating factor, separation of variables).**
Some ODEs have exact solutions. Separation is just "put all the `y`
stuff on one side and all the `x` stuff on the other, then
integrate both sides". The integrating factor multiplies both sides
by a cleverly chosen `e^{∫P(x)dx}` that makes the left-hand side
collapse into the derivative of a product — then you just integrate
back out.


---

## FILE: plain-english/topic-j-vectors.md

# Topic J — Vectors (plain English)

Formal doc: [../topic-j-vectors.md](../topic-j-vectors.md).

## What is it, really?

A **vector** is a quantity that has **both a size and a direction** —
an arrow.

- Speed alone is just a number: "50 km/h".
- **Velocity** is a vector: "50 km/h **north-east**".
- Force, wind, current, magnetic field — all vectors.

Two useful vector tricks:

- **Dot product** — measures **how much two arrows agree** (are they
  pointing the same way?). Zero means they're perpendicular.
- **Cross product** — gives an arrow **perpendicular** to both
  inputs, whose length is "how much they disagree in direction". This
  is how torque and magnetic force are computed.

**Partial derivatives** (also in this topic) mean: "if I nudge just
one input a tiny bit and hold the others still, how does the output
change?" Like turning one knob on a mixing desk at a time.

## Where does it show up?

- **Navigation**, **wind on an aircraft**, **currents on a boat**.
- **Forces on a bridge joint** — all the pull-arrows must cancel.
- **3D graphics and games** — lighting uses dot products, rotations
  use cross products.

## Why does an engineer care?

Almost every real-world quantity in mechanics and electromagnetics
has a direction. Vectors let you add them up correctly (parallelogram
rule) instead of just adding the sizes and getting the wrong answer.

## How the formulas were derived (plain English)

**Vector addition — parallelogram rule.** Slide the tail of the
second arrow to the head of the first. The single arrow from your
start point to the final head is the sum. Draw both arrows from
a common origin instead and complete the parallelogram — the
diagonal is the same answer. Both pictures give the coordinate
formula `(a₁+b₁, a₂+b₂, a₃+b₃)`.

**Dot product — the cosine rule in disguise.** Start with the
familiar cosine rule from a triangle formed by `a`, `b`, and
`a − b`. Expand `|a − b|² = |a|² + |b|² − 2|a||b|cos θ` using
coordinates and you'll find `a₁b₁ + a₂b₂ + a₃b₃ = |a||b|cos θ`.
That's the dot product. It measures **how much of `a` points along
`b`** — zero when perpendicular, maximum when parallel.

**Cross product — the parallelogram-area formula.** Two vectors
sprouting from the origin span a parallelogram. Its area is
`|a||b|sin θ`, and there's a unique direction perpendicular to both
(using the right-hand rule to pick which of the two). Packaging
both facts — length = area, direction = perpendicular — gives
`a × b`. The messy determinant formula with `i, j, k` is just the
bookkeeping to extract that arrow in coordinates.

**Partial derivatives — one knob at a time.** If a function depends
on `x`, `y`, `z`, freeze all but one variable and take the ordinary
derivative in the remaining one. Notation `∂f/∂x` just means
"derivative with respect to `x`, treating `y` and `z` as constants".
Nothing new — same rules as single-variable calculus.

**Gradient.** Stack all the partials into one vector: `∇f = (∂f/∂x,
∂f/∂y, ∂f/∂z)`. This arrow **always points in the direction of
steepest uphill**, and its length is how steep. That falls out of
the chain rule: the change in `f` for a small step `dr` is
`∇f · dr`, and (from the dot product) the biggest change happens
when `dr` is parallel to `∇f`.


---

## FILE: plain-english/topic-k-trig-hyperbolic.md

# Topic K — Trig, hyperbolic, waveforms (plain English)

Formal doc: [../topic-k-trig-hyperbolic.md](../topic-k-trig-hyperbolic.md).

## What is it, really?

- **Trigonometry** — the maths of triangles and, more importantly,
  of **things that go round in circles**. Sine and cosine describe
  the up–down and left–right position of a point spinning on a wheel.
- **Waveforms** — the wobbles you get when you plot that spinning
  point over time. Every AC voltage, every pure musical tone, every
  ripple on water is a waveform.
- **Hyperbolic functions** (`sinh`, `cosh`, `tanh`) — the "cousins"
  of sin and cos. They describe shapes that grow or decay, like a
  **hanging chain**, **cables between two pylons**, or the curve of
  a **suspension bridge**.

## Where does it show up?

- **AC mains** (50/60 Hz sine) — voltage `V = Vₘ sin(2π·50·t)`.
- **Sound and music** — every note = one or more sine waves.
- **Surveying and GPS** — bearings, distances, heights.
- **A hanging cable** — a `cosh` curve (called a *catenary*).

## Why does an engineer care?

Sines and cosines are the alphabet of vibrations, waves and rotating
machinery. Hyperbolics show up whenever something grows/decays
exponentially or hangs under gravity. Both are in every engineering
handbook because *most physics is either wavy or exponential*.

## How the formulas were derived (plain English)

**Sine and cosine from the unit circle.** Put a dot on the edge of a
circle of radius 1. As the dot spins anti-clockwise, its **height
above the x-axis** traces out `sin θ` and its **horizontal position**
traces out `cos θ`. Everything else — Pythagoras (`sin²+cos² = 1`),
angle-sum formulas, phase shifts — is squeezed out of that one
picture.

**Angle-sum identity (`sin(a+b)` etc.).** Draw two rotations, one
after another. Compute the final `(x, y)` two ways: as a single
rotation by `a+b`, and as "rotate by `a`, then by `b`". Match the
coordinates and out drop `sin(a+b) = sin a cos b + cos a sin b` and
its partner. Same geometric fact, two viewpoints.

**Hyperbolic functions — the exponential twins.** Any exponential
can be split into an even part and an odd part:
`e^x = cosh(x) + sinh(x)` where `cosh(x) = (eˣ + e⁻ˣ)/2` (even) and
`sinh(x) = (eˣ − e⁻ˣ)/2` (odd). Those are the **definitions** of the
hyperbolic functions — they're not new mystery objects, just tidy
repackagings of `eˣ` and `e⁻ˣ`.

**Why they parallel sin/cos.** Replace `θ` with `jθ` in the
definitions and use Euler (topic 3):
`cosh(jθ) = cos θ` and `sinh(jθ) = j sin θ`. So the hyperbolic
identities are the trig identities with a couple of sign flips.
One family, two flavours.

**Catenary (hanging cable).** A perfectly flexible cable hanging
under its own weight solves a small differential equation whose
solution turns out to be `y = a · cosh(x / a)`. That's why suspension
cables and power lines droop into a `cosh` curve — it's the shape
that balances tension and gravity at every point.


---

## FILE: plain-english/topic-l-symbolic-diff.md

# Topic L — Symbolic differentiation (plain English)

Formal doc: [../topic-l-symbolic-diff.md](../topic-l-symbolic-diff.md).

## What is it, really?

**Differentiation** answers: *"how fast is this changing?"*

**Symbolic** differentiation means the computer keeps working with
the actual **formula** — not just numbers.

- Type in `sin(x²)`.
- The computer gives you back the formula `2x·cos(x²)` — the exact
  rate of change, still as a formula.

Contrast that with numerical differentiation, which just spits out
a number at one particular point.

## Where does it show up?

- **Optimisation** — where does a curve reach its maximum? Set the
  derivative equal to zero and solve.
- **Sensitivity** — "if I change this resistor by 1%, how much does
  the output shift?"
- **Physics equations** — deriving velocity from position, force
  from potential energy.

## Why does an engineer care?

Once you have the derivative as a **formula**, you can plug in any
value, plot it, or feed it into further symbolic maths without
losing precision. This is what tools like Mathematica, SymPy, and
this library's `expr_calculus.dart` do behind the scenes.

## How the formulas were derived (plain English)

**The one rule everything else comes from.** A derivative is
"rise divided by run, as run shrinks to zero":
`f'(x) = lim_{h→0} (f(x+h) − f(x)) / h`. Every rule below is that
limit worked out for a specific shape of `f`.

**Power rule.** For `f(x) = xⁿ`, expand `(x+h)ⁿ` with the binomial
theorem (topic 2): `xⁿ + n·xⁿ⁻¹·h + (higher powers of h)`. Subtract
`xⁿ`, divide by `h`, and let `h → 0`. All the higher-order `h`
terms die, leaving `n·xⁿ⁻¹`.

**Sum rule.** The derivative of `f + g` is `f' + g'` because
rates of change add — if two accounts grow by £5 and £3 per day,
the combined balance grows by £8 per day.

**Product rule.** Picture the area of a rectangle whose sides are
`f(x)` and `g(x)`. Grow both sides a little. The area grows by:
(the thin strip on top, height `f`, width `g'·h`) **plus** (the
thin strip on the side, width `g`, height `f'·h`), plus a tiny
corner square that vanishes as `h → 0`. Total: `f·g' + g·f'`. The
famous "product rule" is really a picture of two growing strips.

**Chain rule.** If `y` depends on `u` and `u` depends on `x`, then
small changes multiply: `dy = (dy/du) · (du/dx) · dx`. Gears meshing:
turn `x` a little, `u` turns by its gear ratio, `y` turns by the
next gear ratio. Total ratio = product of the two.

**Quotient rule.** Once you have the product rule, the quotient
rule falls out of writing `f/g = f · g⁻¹` and combining product +
chain — no new idea, just algebra.


---

## FILE: plain-english/topic-m-applied-numerical.md

# Topic M — Applied numerical methods (plain English)

Formal doc: [../topic-m-applied-numerical.md](../topic-m-applied-numerical.md).

## What is it, really?

**Numerical methods** are what you use when a problem is too messy
for a neat pen-and-paper answer, so you get a computer to
**approximate** it very well instead.

Common jobs:

- **Root finding** — "at what x does `f(x) = 0`?" (bisection,
  Newton–Raphson). Like a game of hot-and-cold: try a value, see
  which side of zero you're on, and close in.
- **Interpolation** — you have measurements at 10 °C, 20 °C, 30 °C.
  What's the value at 24 °C? Fill in the gap sensibly.
- **Curve fitting** — draw the best-fit line (or curve) through
  scattered data points.
- **Numerical integration/differentiation** — do calculus on a
  table of numbers.

## Where does it show up?

- Solving `x² = 2` on a pocket calculator — that's Newton–Raphson
  under the hood.
- Weather-station data with missing hours — interpolation.
- Fitting a Beer–Lambert curve to a spectrophotometer's readings.
- Finding where a beam's deflection equals a safety limit.

## Why does an engineer care?

Most real problems don't have tidy exact answers. Numerical methods
give you an answer that's **accurate enough for the job**, along
with a way to check how accurate it is.

## How the formulas were derived (plain English)

**Bisection — the guessing game.** If `f(a)` is negative and `f(b)`
is positive, the graph must cross zero **somewhere between** `a`
and `b`. Test the midpoint: whichever half still has a sign change,
keep it. Halve again. Every step chops the search space in half —
after 20 steps you're within one part in a million. It's the
child's game "higher / lower", rigorously.

**Newton–Raphson — slide down the tangent.** Stand at your current
guess `xₙ`. Draw the **tangent** to the curve at that point. Follow
the tangent down to where it crosses zero — that intersection is
your next guess `xₙ₊₁ = xₙ − f(xₙ)/f'(xₙ)`. Near a root the tangent
is very close to the curve, so guesses converge alarmingly fast
(the number of correct digits roughly **doubles** every step).
Downside: needs the derivative, and can fly off if the curve has
nasty bumps.

**Linear interpolation.** You have `y₀` at `x₀` and `y₁` at `x₁`.
Assume the curve between them is a straight line and read off the
value at any intermediate `x`: `y = y₀ + (y₁−y₀)·(x−x₀)/(x₁−x₀)`.
That's just the equation of the line through the two points.

**Lagrange interpolation.** Extend the same idea to more points
by writing a polynomial that is `1` at exactly one data point and
`0` at all the others (built from products like `(x−xⱼ)/(xᵢ−xⱼ)`).
Multiply each such polynomial by its `y` value and add them up —
you get the unique polynomial passing through every data point.

**Least-squares curve fitting.** You have scattered data and want
the "best" line. Define "best" as: **the line that minimises the
sum of squared vertical distances** from data points to the line.
Differentiate that sum with respect to the slope and intercept,
set both derivatives to zero, and solve two simple equations.
Out come the classic slope-and-intercept formulas.


---

## FILE: plain-english/topic-n-symbolic-integration.md

# Topic N — Symbolic integration (plain English)

Formal doc: [../topic-n-symbolic-integration.md](../topic-n-symbolic-integration.md).

## What is it, really?

**Integration** answers: *"if I add up all these tiny changing
amounts, what's the total?"*

**Symbolic** integration means the computer produces the answer as
a **formula**, not just a number.

- Feed it `2x`.
- It hands back `x² + C` — the formula whose derivative is `2x`.

It's the reverse of symbolic differentiation (topic L): differentiation
finds the rate, integration undoes that and finds the total.

## Where does it show up?

- **Area under a curve** as a formula (e.g. exact area of a
  parabolic segment).
- **Work done** by a variable force: `W = ∫ F(x) dx`.
- **Charge stored** on a capacitor from a current profile.
- **Centre of gravity**, **moment of inertia** — all integrals of
  geometric shapes.

## Why does an engineer care?

A **symbolic** result stays exact — you can substitute different
numbers later, differentiate it again, or simplify it. Numerical
integration only gives one number at a time; symbolic integration
gives you the master formula.

## How the formulas were derived (plain English)

**Integration is anti-differentiation.** Every rule in the
integration table is just a differentiation rule read **right to
left**. You know `d/dx (x³) = 3x²`, so `∫ 3x² dx = x³ + C`. The
whole table is built this way. The `+ C` is because
differentiation loses constant offsets, so integration has to
admit "could be any constant".

**Power rule (in reverse).** Since `d/dx (xⁿ⁺¹/(n+1)) = xⁿ`, we
get `∫ xⁿ dx = xⁿ⁺¹/(n+1) + C` for any `n ≠ −1`. The exception
`n = −1` gives `∫ dx/x = ln|x| + C` — a special case because the
power rule would divide by zero.

**Substitution — the chain rule in reverse.** If you spot a
function and its derivative inside the integral (say, `f(g(x)) ·
g'(x)`), let `u = g(x)`, so `du = g'(x) dx`. The messy integral
collapses to a plain `∫ f(u) du`. It's the chain rule running
backwards: differentiation multiplies by `g'(x)`, integration
"un-multiplies" by making that factor become the new variable.

**Integration by parts — the product rule in reverse.** Start with
the product rule `(uv)' = u'v + uv'`, integrate both sides, and
rearrange to `∫ u dv = uv − ∫ v du`. Practical use: split a hard
integral into a piece you'll **differentiate** (`u`) and a piece
you'll **integrate** (`dv`), so the leftover integral is easier
than the one you started with. Mnemonic **LIATE** (Logarithms,
Inverse trig, Algebraic, Trig, Exponential) suggests which factor
to pick as `u`.

**Partial fractions (again).** Rational functions of `x` are
integrated by first splitting them with partial fractions
(topic 4), then each piece is a standard form (`1/(x−a)` →
`ln|x−a|`, `1/(x²+a²)` → `(1/a)·arctan(x/a)`, etc.). This is why
topic 4 keeps showing up — it's the pre-processing step that makes
integration and Laplace inversion tractable.

**Definite integrals — the shortcut.** The **fundamental theorem
of calculus** says: to get the value of `∫_a^b f(x) dx`, find any
antiderivative `F(x)` and compute `F(b) − F(a)`. All the tiny
strips add up neatly because the derivative and integral cancel.


---

## FILE: session-2026-09-12.md

# Session recap — 2026-09-12

A working log of what was built in this session across the two
repositories.

Both repos live under `c:\www\`:
- **Library** — `c:\www\dart\dart-eng-math\` (this repo).
- **Flutter companion app** — `c:\www\flutter\dart_eng_math_app\`
  (sibling repo, path-depends on this library).

## What we built, at a glance

1. Plain-English documentation companion for every topic doc.
2. Fourier **transforms** added to the library (continuous CFT + DFT / FFT).
3. Doc-comment refresh on `binomialCoefficient` in
   [lib/src/binomial.dart](../lib/src/binomial.dart).
4. A brand-new Flutter Windows-desktop app that consumes the
   library and presents each topic as a **Brilliant-style stepped
   lesson** with sliders, live plots, reveals, multiple-choice
   checkpoints and a confetti finish.

## 1 · Plain-English documentation

New folder: [docs/plain-english/](plain-english/README.md).

Fifteen files: a landing README plus one page per topic. Each page
answers three questions with no jargon:

- **What is it, really?**
- **Where does it show up in daily life?**
- **Why does an engineer care?**

Then links to the formal counterpart in [docs/](README.md).

A follow-up pass added **"How the formulas were derived (plain
English)"** to every page — Gauss's pairing trick for AP sums, `j²=−1`
from two quarter-turns, cover-up for partial fractions, parallelogram
= determinant, RK4 as "poll four voters", etc.

The Fourier page ([docs/plain-english/topic-h-fourier.md](plain-english/topic-h-fourier.md))
was then extended to cover **three flavours** (series / CFT / DFT+FFT)
once the transform code landed.

Commits (in this repo):

| Hash | What it did |
|---|---|
| `83a1099` | `docs(binomial)`: added the plain-English explanation to `binomialCoefficient` |
| `c0689de` | first pass of `docs/plain-english/` (14 pages + README) |
| `531530b` | added derivation sections to every page |
| `73342be` | scoped H page to series only (prior to CFT work) |

## 2 · Fourier transforms

The library previously covered **Fourier series only**. New modules
extend it to non-periodic and sampled signals.

New source files:

- [lib/src/fourier_transform.dart](../lib/src/fourier_transform.dart)
  — continuous transform via Simpson integration:
  `fourierTransform`, `fourierTransformSpectrum`,
  `inverseFourierTransform`.
- [lib/src/dft.dart](../lib/src/dft.dart) — discrete:
  `dft` / `idft` (O(N²), any N), `fft` / `ifft` (iterative radix-2
  Cooley–Tukey, power-of-two N), `dftReal`, `fftReal`,
  `magnitudeSpectrum`, `phaseSpectrum`, `oneSidedAmplitudeSpectrum`,
  `fftFrequencies`.

New tests: [test/fourier_transform_test.dart](../test/fourier_transform_test.dart)
— **15 tests, all green**. Verified against analytic pairs
(rectangular pulse ↔ sinc, Gaussian ↔ Gaussian, decaying exponential
↔ `1/(a+jω)`) and DFT identities (impulse → flat, constant → DC only,
`fft` ≡ `dft` on random inputs, round-trip `ifft(fft(x)) = x`).

Doc updates:

- [docs/topic-h-fourier.md](topic-h-fourier.md) — added "Continuous
  Fourier transform" and "Discrete Fourier transform (DFT) and FFT"
  sections with API tables and verified cases.
- [docs/plain-english/topic-h-fourier.md](plain-english/topic-h-fourier.md)
  — added a "From series to transform" section explaining CFT
  (period → ∞) and DFT (samples), plus how to read an FFT plot.
- [docs/README.md](README.md) and the plain-English README updated to
  reflect "series & transforms".

Commit: `88a0e0b` `feat(fourier): add continuous Fourier transform,
DFT and FFT (Topic H)`.

## 3 · Flutter companion app (`c:\www\flutter\dart_eng_math_app`)

Fresh Flutter Windows-desktop project scaffolded via `flutter create`,
consuming this library as a path dependency:

```yaml
dependencies:
  dart_eng_math:
    path: ../../dart/dart-eng-math
  fl_chart: ^0.68.0
```

### 3.1 Initial version

- `lib/main.dart` — home page grid with one card per topic (14 cards).
- `lib/topic_scaffold.dart` — shared blurb frame, `LabeledSlider` and
  a `SimpleLinePlot` `CustomPaint` widget.
- 14 topic pages under `lib/topics/`, each with a live plot and
  sliders. Custom-painted visuals for Argand plane, matrix-transformed
  unit square, unit circle, vectors, histogram.

Verified: `flutter analyze` clean, widget smoke test passes, Windows
debug build succeeds, executable launches.

Commit (in the app repo): `1a8a146`
`feat: initial dart_eng_math_app Flutter Windows app with 14 topic pages`
(41 files, +4,872 lines).

### 3.2 Brilliant-style rewrite

Every topic page then rewritten as a multi-step guided lesson.

New interactive primitives under `lib/interactive/`:

| File | Widget | Purpose |
|---|---|---|
| `lesson_flow.dart` | `LessonFlow` | `PageView` with top progress bar and Back/Continue buttons. Continue stays disabled until the current step calls `markComplete`. |
| `lesson_flow.dart` | `LessonStep`, `LessonCard`, `LessonText` | Building blocks used inside every lesson. |
| `multiple_choice.dart` | `MultipleChoice`, `MCOption` | Rounded chip-per-option UI with immediate feedback and per-option explanation. Only the correct pick fires `onCorrect`. |
| `reveal_box.dart` | `RevealBox` | "Show me why" button that expands into an amber derivation panel. |
| `reveal_box.dart` | `HintChip` | Collapsible "Need a nudge?" hint. |
| `confetti.dart` | `CompletionConfetti` | 80-particle physics-based confetti burst on lesson completion. |

Every topic follows a 5–6 step template:

1. **Intro** — plain-English what & why (auto-completes).
2. **Manipulate** — sliders drive a live plot; touching them completes.
3. **Reveal** — derivation hidden behind a "Show me why" button.
4. **Question** — `MultipleChoice` with explanations for every option.
5. **Wrap-up** — recap + confetti.

The Fourier page (flagship) was written first, verified, then the
pattern rolled to all other topics.

Verified: `flutter analyze` clean, widget test passes, Windows debug
build succeeds, executable launches cleanly.

Commit (in the app repo): `02e565f`
`feat(lessons): rewrite all 14 topics as Brilliant-style stepped
lessons with interactive primitives` (18 files, +3,055 / −1,540).

## Repo state at end of session

### `dart-eng-math` (library) — branch `main`

```
88a0e0b feat(fourier): add continuous Fourier transform, DFT and FFT (Topic H)
73342be docs(plain-english): scope Fourier page to series only (no FFT)
531530b docs(plain-english): add plain-English derivation section to each topic
c0689de docs: add plain-English companion guide for topic series
83a1099 docs(binomial): add plain-English explanation of C(n, k)
```

Nothing pushed to any remote.

### `dart_eng_math_app` (Flutter app) — branch `main`

```
02e565f feat(lessons): rewrite all 14 topics as Brilliant-style stepped lessons with interactive primitives
1a8a146 feat: initial dart_eng_math_app Flutter Windows app with 14 topic pages
```

Nothing pushed to any remote.

## How to run

Library tests:

```powershell
cd c:\www\dart\dart-eng-math
dart test
```

Flutter app (Windows desktop):

```powershell
cd c:\www\flutter\dart_eng_math_app
flutter run -d windows
```

## Open follow-ups (not done this session)

- **Push both repos** to a remote (GitHub or similar).
- **More manipulation primitives** (draggable point on a plot,
  drag-to-place vectors) — currently interaction is slider-based.
- **Theme refresh** (rounded/pastel palette, Google Fonts) — user
  chose to skip this in the pilot; can be added later.
- **Half-range Fourier series** convenience constructor
  (called out in [topic-h-fourier.md](topic-h-fourier.md) notes).
- **Web build target** for the Flutter app (currently Windows-only).


---

## FILE: student-guide.md

# Student's guide to `dart_eng_math`

A practical walkthrough for students working problems from *Higher
Engineering Mathematics* (Bird) or a similar syllabus. This guide
assumes no prior Dart experience — just enough terminal familiarity to
run a command.

## Contents

- [What this library is for](#what-this-library-is-for)
- [Setup](#setup)
- [Your first calculation](#your-first-calculation)
- [How do I …?  (task index)](#how-do-i---task-index)
- [Worked examples end-to-end](#worked-examples-end-to-end)
- [Gotchas](#gotchas)
- [Reading the tests as documentation](#reading-the-tests-as-documentation)
- [Where to look next](#where-to-look-next)

## What this library is for

Two use cases:

1. **Check your homework.** Solve the problem by hand, then verify with
   a few lines of Dart. If the numbers agree, you almost certainly got
   the method right.
2. **Skip the tedious arithmetic.** Once you understand the technique
   (partial fractions, Laplace transforms, matrix inversion, …), let
   the library grind through the algebra so you can focus on the
   concepts.

The library follows Bird's presentation conventions and worked examples
throughout, so results will match what you see in the textbook (modulo
4 significant figures of rounding).

## Setup

### 1. Install Dart

Get the Dart SDK from [dart.dev](https://dart.dev/get-dart). Version
≥ 3.11.5. Verify with:

```powershell
dart --version
```

### 2. Get the library

```powershell
git clone <this-repo> C:\src\dart-eng-math
cd C:\src\dart-eng-math
dart pub get
```

### 3. Confirm everything works

```powershell
dart test
```

You should see `+477: All tests passed!`. If a test fails on your
machine, that's a bug — please note the failing test name.

### 4. Editor setup (optional but recommended)

VS Code with the Dart extension gives you autocomplete, docstring
hover-help, and an in-editor test runner. Bird's problem numbers are
in the docstrings and test names, so the searches like
"Ch 63 Problem 5" hit the relevant code directly.

## Your first calculation

Create `bin/scratch.dart` in the project:

```dart
import 'dart:math' as math;
import 'package:dart_eng_math/dart_eng_math.dart';

void main() {
  // Simpson's rule: ∫₀¹ 1/(1 + x²) dx = π/4.
  final approx = simpson((x) => 1 / (1 + x * x), 0, 1, 4);
  print('Simpson approximation: $approx');
  print('π / 4              :   ${math.pi / 4}');
}
```

Run:

```powershell
dart run bin/scratch.dart
```

Output:

```
Simpson approximation: 0.7853921568627451
π / 4              :   0.7853981633974483
```

Match to 5 decimal places with only 4 subintervals — that's the strength
of Simpson.

## How do I ...?  (task index)

Each entry is one or two lines of code. See the linked topic doc for
the full API, more examples, and Bird problem-by-problem verification.

### … compute a definite integral?

Topic 1 — [numerical integration](topic-1-further-calculus.md).

```dart
simpson((x) => math.exp(-x * x), 0, 2, 100);   // most accurate, needs even n
trapezium((x) => math.log(x + 1), 0, 5, 20);   // baseline, O(h²)
midOrdinate((x) => math.sin(x), 0, math.pi, 50); // Bird Ch 45 alternative
```

### … find a root?

Phase M — [applied numerical methods](topic-m-applied-numerical.md).

```dart
// √2 three ways:
bisection((x) => x * x - 2, a: 1, b: 2);
newtonRaphson((x) => x * x - 2, (x) => 2 * x, x0: 1);
secant((x) => x * x - 2, x0: 1, x1: 2);
```

### … differentiate a function?

Two paths depending on what you need:

**Numeric** (Phase A) — you just want a number:

```dart
derivative((x) => math.exp(2 * x) * math.cos(3 * x), 0.5);
```

**Symbolic** (Phase L) — you want a formula:

```dart
final x = variable('x');
final f = Exp(Const(2) * x) * Cos(Const(3) * x);
final df = differentiate(f, 'x');
print(df);              // symbolic expression
print(df.eval({'x': 0.5}));  // numeric value
```

### … integrate by parts or by substitution?

Symbolic integration is out of scope. Do the technique on paper, then
verify the numeric answer with `simpson`:

```dart
// Verify ∫₁² x² · ln x dx = 8 ln 2 / 3 − 7/9.
final expected = (8 / 3) * math.log(2) - 7 / 9;
final approx = simpson((x) => x * x * math.log(x.toDouble()), 1, 2, 20);
print('expected: $expected');
print('approx:   $approx');
```

### … decompose a partial fraction?

Topic 4 — [partial fractions](topic-4-partial-fractions.md).

```dart
// (11 − 3x) / [(x − 1)(x + 3)]
final res = partialFractions(
  Polynomial([11, -3]),
  [LinearFactor(1), LinearFactor(-3)],
);
for (final t in res.terms) {
  print(t);   // e.g. "2 / (x − 1)"  and  "-5 / (x + 3)"
}
```

Repeated factors and irreducible quadratics use the same call — just
pass different `Factor`s:

```dart
partialFractions(
  Polynomial([-19, -2, 5]),
  [LinearFactor(-3), LinearFactor(1, multiplicity: 2)],
);
```

### … solve a 1st-order ODE?

Phase I — [ODE methods](topic-i-ode-methods.md).

```dart
// dy/dx = x + y, y(0) = 1.
final trajectory = solveRk4(
  (x, y) => x + y,
  x0: 0, y0: 1, xEnd: 2, steps: 200,
);
final yAt2 = trajectory.last.y;
```

### … solve a linear ODE with initial conditions (Laplace method)?

Topic 6 — [Laplace transforms](topic-6-laplace.md).

```dart
// y" + 3y' + 2y = 0, y(0) = 1, y'(0) = 0.
final y = solveOdeLaplace(
  coefficients: [2, 3, 1],        // ascending: c₀ + c₁·s + c₂·s²
  initialConditions: [1, 0],
  forcing: TimeExpr.zero(),
  denominatorFactors: [LinearFactor(-1), LinearFactor(-2)],
);
print(y.eval(0.5));   // ≈ 2·e^{-0.5} − e^{-1}
```

### … solve a coupled system of two ODEs?

```dart
// dx/dt = y, dy/dt = -x, x(0) = 1, y(0) = 0.
final r = solveLinear2x2OdeLaplace(
  a: Matrix.fromRows([[0, 1], [-1, 0]]),
  initial: [1, 0],
  denominatorFactors: [QuadraticFactor(1, 0, 1)],   // s² + 1
);
r.x.eval(0.5);    // ≈ cos(0.5)
r.y.eval(0.5);    // ≈ -sin(0.5)
```

### … solve a heat, wave, or Laplace PDE?

Phase T. All three take Dirichlet boundaries; the returned solution
container has `.at(t)` snapshots and bilinear `.eval(x, t)` / `.eval(x, y)`.

```dart
// Heat: cooling rod with sin(πx) initial condition.
final heat = solveHeatEquation(
  alpha: 0.5, length: 1, duration: 0.2,
  spatialSteps: 40, timeSteps: 200,
  initialCondition: (x) => math.sin(math.pi * x),
);

// Wave: plucked string.
final wave = solveWaveEquation(
  waveSpeed: 1, length: 1, duration: 0.5,
  spatialSteps: 40, timeSteps: 200,
  initialDisplacement: (x) => math.sin(math.pi * x),
);

// Laplace: steady heat on a plate.
final steady = solveLaplaceEquation(
  width: 1, height: 1, nx: 40, ny: 40,
  boundary: (x, y) => y >= 1 - 1e-12 ? math.sin(math.pi * x) : 0,
);
steady.eval(0.5, 0.5);
```

### … find a Fourier series?

Topic H — [Fourier series](topic-h-fourier.md).

```dart
// Sawtooth on [-π, π]: f(x) = x.
final fs = FourierSeries.fromFunction(
  (x) => x, period: 2 * math.pi, start: -math.pi, terms: 10,
);
fs.a0;     // ≈ 0
fs.b[0];   // ≈ 2       (b₁)
fs.b[1];   // ≈ -1      (b₂ = 2·(-1)²/2)
fs.eval(0.5);   // partial-sum approximation
```

### … convert `a·sin(ωt) + b·cos(ωt)` to `R·sin(ωt + α)`?

Topic K — [trig / waveforms](topic-k-trig-hyperbolic.md).

```dart
final r = combineSinCos(3, 4);
// r.amplitude = 5.0
// r.phase     = 0.9273 rad (≈ 53.13°)
```

### … add sinusoids of the same frequency?

```dart
final resultant = phasorSum([
  SinusoidalWaveform(amplitude: 3, angularFrequency: 100, phase: 0),
  SinusoidalWaveform(amplitude: 4, angularFrequency: 100, phase: math.pi / 2),
]);
// resultant.amplitude = 5, resultant.phase = atan2(4, 3)
```

### … use complex numbers?

Topic 3 — [complex numbers](topic-3-complex-numbers.md).

```dart
final z1 = Complex(1, 2);            // 1 + j2
final z2 = Complex.polar(3, math.pi / 4);
z1 + z2;
z1.conjugate;
z1.modulus;
z1.argumentDeg;

complexPow(Complex(1, 1), 8);        // (1 + j)⁸ = 16
complexRoots(Complex(0, 8), 3);      // three cube roots of 8j
```

### … solve a system of linear equations?

Topic 5 — [linear algebra](topic-5-linear-algebra.md).

```dart
// 3x + 2y = 12,  5x − y = 7.
solveCramer(
  Matrix.fromRows([[3, 2], [5, -1]]),
  [12, 7],
);   // returns [2, 3]

solveByInverse(matrix, rhs);
```

### … compute a determinant, inverse, or eigenvalue?

```dart
determinant(Matrix.fromRows([[3, 4, -1], [2, 0, 7], [1, -3, -2]])); // 113
inverse(matrix);
eigenvalues2x2(Matrix.fromRows([[2, 1], [1, 2]]));   // [3, 1]

// Full spectrum of any symmetric matrix (Phase R).
final eig = SymmetricEigenDecomposition.of(symmetricMatrix);
eig.eigenvalues;      // descending
eig.eigenvectors;     // orthonormal, columns match eigenvalues

// Dominant eigenpair via power iteration.
final r = powerIteration(matrix);
r.eigenvalue; r.eigenvector;

// Cubic-convergence refinement near a known shift.
rayleighQuotientIteration(matrix, r.eigenvalue, initial: r.eigenvector);
```

### … run PCA on a dataset?

Phase R composes `SymmetricEigenDecomposition` with covariance
computation.

```dart
final pca = PrincipalComponentAnalysis.fit(samples);
pca.explainedVarianceRatio;              // e.g. [0.92, 0.06, 0.02]
final k = pca.componentsFor(0.95);       // smallest k covering 95 %
final projected = pca.transform(samples, k: k);
final restored = pca.inverseTransform(projected);
```

### … solve a larger system, decompose a matrix, or fit multiple predictors?

For anything n ≥ 4, use the LU / QR routines from Phase P — they're
numerically stable and O(n³).

```dart
// One-shot solve of a 5×5 system.
solveLU(matrix5x5, [12, 20, 24, 22, 32]);

// Reusable factorisation.
final lu = LUDecomposition.of(matrix5x5);
lu.solve(b1);
lu.solve(b2);
lu.determinant();
lu.inverse();

// Symmetric positive-definite? Cholesky is ~2× faster than LU.
final ch = CholeskyDecomposition.of(spdMatrix);
ch.solve(b);

// Overdetermined systems (m > n) — least squares via QR.
linearLeastSquares(designMatrix, ys);

// Multiple linear regression: y = β₀ + β₁ x₁ + β₂ x₂ + … .
final fit = multipleLinearRegression(
  [[1, 1], [2, 3], [4, 5], [1, 4], [3, 2], [5, 1]],
  ys,
);
fit.intercept;
fit.slopes;
fit.predict([3, 4]);
fit.rSquared(xs, ys);
```

### … handle rank-deficient systems, compute the pseudoinverse, or check condition number?

Phase U — Singular Value Decomposition. Reach for this whenever the
matrix might be rank-deficient, under-determined, or numerically
ill-conditioned; the Phase P QR route throws in those cases, SVD
gives the minimum-norm least-squares answer.

```dart
final svd = SVD.of(anyMatrix);
svd.singularValues;       // sorted descending
svd.rank();               // effective numerical rank
svd.conditionNumber();    // σ_max / σ_min
svd.pseudoinverse();      // Moore–Penrose A⁺
svd.leastSquares(b);      // shortest x minimising ‖Ax − b‖
```

### … interpolate tabular data or fit a polynomial?

Phase Q. All four interpolators are callable — `interp(x)` gives ŷ.

```dart
final xs = [0.0, 1, 2, 4, 7, 10];
final ys = [1.0, 3, 2, 5, 4, 6];

// Arbitrary knots → Lagrange.
LagrangeInterpolator(xs, ys)(3.5);

// Equally-spaced knots → Newton differences.
NewtonForwardInterpolator([0.0, 1, 2, 3, 4], [1, 2, 5, 10, 17])(0.5);

// Smooth reconstruction with derivative → natural cubic spline.
final s = CubicSpline(xs, ys);
s(3.5);
s.derivative(3.5);

// Least-squares polynomial fit.
final p = polynomialFit(xs, ys, 3);
p.eval(3.5);
p.coefficients;    // ascending: [c0, c1, c2, c3]
```

### … fit a Gaussian peak, exponential decay, or arbitrary non-linear model?

Phase S — Levenberg–Marquardt for anything writable as
`double f(double x, List<double> params)`.

```dart
// Gaussian: y = A·exp(-(x-μ)²/(2σ²))
final peak = nonlinearLeastSquares(
  xs: xs, ys: ys,
  model: gaussianModel,
  jacobian: gaussianJacobian,
  initialParameters: [maxY, argmaxX, guessSpread],
);
peak.parameters;         // [A, μ, σ]
peak.standardErrors;     // 1-σ uncertainties per parameter
peak.rSquared(ys);

// Custom model — analytic Jacobian is optional.
final fit = nonlinearLeastSquares(
  xs: xs, ys: ys,
  model: (x, p) => p[0] / (1 + p[1] / x),   // Michaelis-Menten-ish
  initialParameters: [1.0, 1.0],
);
```

### … compute statistics or a probability?

Topic G — [statistics + probability](topic-g-statistics.md).

```dart
mean([1, 2, 3, 4, 5]);       // 3.0
median([2, 3, 7, 8]);        // 5.0
stdev([1, 2, 3, 4, 5]);      // 1.4142 (population)

BinomialDistribution(10, 0.3).pmf(3);   // 0.2668
PoissonDistribution(2.4).pmf(2);        // 0.2613
NormalDistribution(100, 15).cdf(115);   // 0.8413

pearsonCorrelation(xs, ys);
final fit = regressionYonX(xs, ys);
fit.predict(newX);
```

### … build a confidence interval or run a hypothesis test?

Also Topic G — the inferential-statistics stack from Phase O.

```dart
// 95 % CI for the mean when σ is unknown (t-based).
ciMeanT(sampleMean: 12.4, sampleStd: 1.8, n: 25);

// Two-sample comparison; returns a HypothesisTestResult with p-value.
final r = twoSampleT(a: sampleA, b: sampleB);
r.significantAt(0.05);

// Comparing three or more groups → one-way ANOVA.
oneWayAnova([groupA, groupB, groupC]);

// Non-parametric alternative when data is skewed or ordinal.
mannWhitneyU(a: sampleA, b: sampleB);
wilcoxonSignedRank(before: pre, after: post);

// Chi-square goodness-of-fit or independence.
chiSquareIndependence([[50, 10], [10, 50]]);

// Bayes' rule for discrete hypotheses.
bayes([0.01, 0.99], [0.99, 0.05]);   // → [0.167, 0.833]
```

### … work with 3-D vectors?

Topic J — [vectors + partial derivatives](topic-j-vectors.md).

```dart
final a = Vector.xyz(2, 3, -1);
final b = Vector.xyz(1, -4, 2);
a.dot(b);       // -12
a.cross(b);     // Vector(2, -5, -11)
a.magnitude;
a.angleTo(b);
```

### … compute partial derivatives or classify a stationary point?

```dart
double f(List<double> p) => p[0] * p[0] + p[1] * p[1] - 2 * p[0] - 4 * p[1] + 5;

partialDerivative(f, [1, 2], 0);  // ∂f/∂x
gradient(f, [1, 2]);              // [0, 0]  (this is a stationary point)
classifyCriticalPoint2D(f, [1, 2]);   // CriticalPointType.minimum
```

## Worked examples end-to-end

### Example A: verify a syllabus self-test question

Bird / syllabus Q7: express `z = −1 + j√3` in polar form and compute `z⁶`.

```dart
import 'dart:math' as math;
import 'package:dart_eng_math/dart_eng_math.dart';

void main() {
  final z = Complex(-1, math.sqrt(3));
  print('|z| = ${z.modulus}');           // 2
  print('arg z = ${z.argumentDeg}°');    // 120
  print('z⁶ = ${complexPow(z, 6)}');     // (64 + j·0)
}
```

### Example B: RL step response by Laplace

`L·di/dt + R·i = V`, `L = 1 H`, `R = 2 Ω`, `V = 12 V`, `i(0) = 0`.

Expected: `i(t) = 6·(1 − e^{-2t})`.

```dart
final i = solveOdeLaplace(
  coefficients: [2, 1],
  initialConditions: [0],
  forcing: TimeExpr.constant(12),
  denominatorFactors: [LinearFactor(-2), LinearFactor(0)],
);
for (final t in [0.0, 0.5, 1.0, 2.0]) {
  print('t = $t → i = ${i.eval(t)}');
}
```

### Example C: check a hand-computed derivative

You've computed `d/dx (e^{2x} cos 3x) = 2·e^{2x} cos 3x − 3·e^{2x} sin 3x`.
Verify with the symbolic differentiator:

```dart
final x = variable('x');
final f = Exp(Const(2) * x) * Cos(Const(3) * x);
final df = differentiate(f, 'x');

for (final xv in [0.0, 0.3, 0.7]) {
  final closedForm =
      2 * math.exp(2 * xv) * math.cos(3 * xv) -
      3 * math.exp(2 * xv) * math.sin(3 * xv);
  print('${df.eval({'x': xv})}  vs  $closedForm');
}
```

### Example D: fit a straight line and predict

```dart
// Bird Ch 60 Problem 1: frequency vs inductive reactance.
final freq = <num>[50, 100, 150, 200, 250, 300, 350];
final xL   = <num>[30,  65,  90, 130, 150, 190, 200];

final fit = regressionYonX(freq, xL);
print('Y = ${fit.intercept.toStringAsFixed(2)} + '
      '${fit.slope.toStringAsFixed(3)} · X');
print('X_L at 175 Hz ≈ ${fit.predict(175).toStringAsFixed(1)} Ω');
```

### Example E: sum three phasors (Bird Ch 25)

```dart
final resultant = phasorSum([
  SinusoidalWaveform(amplitude: 2, angularFrequency: 314, phase: math.pi / 6),
  SinusoidalWaveform(amplitude: 5, angularFrequency: 314, phase: -math.pi / 4),
  SinusoidalWaveform(amplitude: 4, angularFrequency: 314,
                     phase: 2 * math.pi / 3),
]);
print('R = ${resultant.amplitude.toStringAsFixed(3)}');
print('α = ${(resultant.phase * 180 / math.pi).toStringAsFixed(2)}°');
```

## Gotchas

1. **Simpson wants an even `n`.** Odd `n` → `ArgumentError`. Use even
   `n`. `n = 100` is a good default for smooth integrands.

2. **`inverseLaplace` needs a factored denominator.** There is no
   polynomial root-finder built into the Laplace pipeline. Factor
   `s² + 3s + 2 = (s + 1)(s + 2)` yourself and pass
   `[LinearFactor(-1), LinearFactor(-2)]`.

3. **`Polynomial` coefficients are ascending** — `Polynomial([1, 2, 3])`
   is `1 + 2x + 3x²`, not `x² + 2x + 3`. Same for `solveOdeLaplace`'s
   `coefficients` argument.

4. **`derivative` at a cusp** (e.g. `x^(3/2)` at `x = 0`) returns `NaN`
   because central-difference evaluates `f(-h)`. Reparameterise or
   shrink the domain.

5. **Bird's printed answers are 3–4 s.f.**  Our double-precision
   results often differ from the printed values by ~10⁻³. That's a
   printing artefact, not a bug.

6. **`combineSinCos` returns phase in `(−π, π]`.** Bird sometimes
   reports the equivalent angle in a different quadrant (e.g. 236.63°
   in the third quadrant vs our −123.37°). Both represent the same
   angle.

7. **`cross` product is 3-D only.** Throws `StateError` for other
   dimensions.

8. **`Vector` and `Matrix` are separate types.** Convert a vector to a
   column matrix with `Matrix.column([...])` when you need matrix
   arithmetic.

9. **`variable('x')` creates a new `Var` on every call.** Assign it to
   a local so you use the same instance consistently:
   ```dart
   final x = variable('x');
   final f = Sin(x) * Cos(x);
   final df = differentiate(f, 'x');
   ```

10. **Symbolic `simplify` is not a CAS.** It handles `x + 0`, `x · 1`,
    constant folding, and `−(−x)`, but not like-term collection. If
    output looks verbose, that's expected — the tree is mathematically
    correct.

## Reading the tests as documentation

Every module in `lib/src/` has a companion `test/<module>_test.dart`.
Test names are keyed to Bird problem numbers:

```
test('Bird Ch. 63 Problem 5(a): 3 / (s² − 4s + 13) = e^{2t} sin 3t', () { … });
```

If you're stuck on how to feed a specific kind of input to a function,
open the test file and look for the closest Bird problem. The test
does the tour: build the input, call the function, assert the answer.
It's often the shortest path from "I have this problem" to "here's the
Dart call".

Recommended reading order for a new user:

1. [test/syllabus_selftest_test.dart](../test/syllabus_selftest_test.dart)
   — all 16 syllabus §Self-test questions solved end-to-end.
2. Whichever topic-specific test file matches your homework problem.

## Where to look next

- **[docs/README.md](README.md)** — index of all topic docs.
- **[docs/bird-full-plan.md](bird-full-plan.md)** — Bird chapter →
  module map. Find the chapter you're studying and it will point you
  to the right doc and API.
- **[Per-topic docs](README.md#topic-map)** — API reference + full list
  of verified Bird problems per topic.
- **The tests** — see the previous section.

If you find a Bird problem the library doesn't handle, open an issue or
add a test — it's probably a five-minute change to extend an existing
module.


---

## FILE: topic-1-further-calculus.md

# Topic 1 — Further calculus

Syllabus reference: [A-advanced-mathematics-2.md §1.1–1.5](../A-advanced-mathematics-2.md).
Bird reference: Ch. 27 (Simpson's rule), Ch. 8 (Maclaurin), Ch. 27–29 (numerical integration).

## Contents

- [Numerical differentiation](#numerical-differentiation)
- [Numerical integration](#numerical-integration)
  - [Trapezium rule](#trapezium-rule)
  - [Simpson's 1/3 rule](#simpsons-13-rule)
- [Maclaurin series](#maclaurin-series)

## Numerical differentiation

File: [lib/src/numerical_derivative.dart](../lib/src/numerical_derivative.dart).

| Function | Description |
|---|---|
| `derivative(f, x, {h=1e-6})` | Central difference `(f(x+h) − f(x−h)) / (2h)`, error O(h²). |
| `forwardDerivative(f, x, {h=1e-6})` | Forward difference, error O(h). |
| `nthDerivative(f, x, n, {h=1e-4})` | n-th derivative by iterating central differences. |

### Example

```dart
derivative((x) => math.sin(x), 0);   // 1.0 (within ~1e-9)
nthDerivative((x) => x * x * x, 2, 2);  // 12.0
```

### Notes

- `nthDerivative` widens `h` slightly per nesting level to keep the
  outer difference numerically meaningful. Expect precision loss beyond
  `n = 4` on `double`.

## Numerical integration

### Trapezium rule

File: [lib/src/trapezium.dart](../lib/src/trapezium.dart).

`trapezium(f, a, b, n)` computes

$$\int_a^b f(x)\,dx \approx \frac{h}{2}\big[f(a)+f(b)+2\sum_{i=1}^{n-1} f(x_i)\big], \quad h=\frac{b-a}{n}.$$

Error is O(h²). Baseline; prefer `simpson` for smooth integrands.

### Simpson's 1/3 rule

File: [lib/src/simpson.dart](../lib/src/simpson.dart).

`simpson(f, a, b, n)` (n even) computes

$$\int_a^b f(x)\,dx \approx \frac{h}{3}\big[f(a)+f(b)+4\!\!\sum_{i\,\text{odd}}\!\!f(x_i)+2\!\!\sum_{i\,\text{even}\,>0}\!\!f(x_i)\big].$$

Error is O(h⁴); exact for polynomials up to degree 3.

### Verified worked examples

| Reference | Test |
|---|---|
| §1.3 Simpson demo: `∫₀¹ 1/(1+x²) dx ≈ π/4` with n=4 | [test/integration_test.dart](../test/integration_test.dart) |
| `∫₀^π sin x dx = 2` at n=10 (~1e-4) and n=100 (~1e-8) | same |
| `∫₀² x³ dx = 4` — exact for cubics | same |

## Maclaurin series

File: [lib/src/maclaurin_series.dart](../lib/src/maclaurin_series.dart).

$$f(x) \approx \sum_{k=0}^{N-1} \frac{f^{(k)}(0)}{k!}\,x^k$$

| Function | Description |
|---|---|
| `maclaurinSeries(a, x, {accuracy})` | `a[k] = f⁽ᵏ⁾(0)`. |
| `deltaMaclaurinSeries(deltas, x, {accuracy})` | `deltas[k]` is the k-th derivative as a function; evaluated at 0 on the fly. |

### Example

```dart
// e^x : all derivatives at 0 equal 1.
maclaurinSeries(List<num>.filled(10, 1), 0.5, accuracy: 10);
// ≈ 1.6487212707 (math.exp(0.5))

// sin x : 0, 1, 0, -1, 0, 1, ...
final sinDerivs = List<num>.generate(10,
    (n) => switch (n % 4) { 0 => 0, 1 => 1, 2 => 0, _ => -1 });
maclaurinSeries(sinDerivs, 0.5, accuracy: 10); // ≈ 0.4794255386
```

### Coverage

The demo `main()` reproduces the standard expansions from §1.5 —
`e^x`, `sin x`, `cos x`, `sinh x`, `cosh x`, `ln(1+x)`, `1/(1−x)`,
`arctan x` — and cross-checks each against `dart:math`.

## Not implemented (paper-only in this topic)

- **§1.1 Symbolic differentiation rules** — sum/product/quotient/chain
  rules. Left as paper exercises; the numerical derivative covers §1.5
  and the RLC/Laplace work in later topics.
- **§1.2 Symbolic integration techniques** — substitution, integration
  by parts. Ditto.


---

## FILE: topic-2-series.md

# Topic 2 — Series

Syllabus reference: [A-advanced-mathematics-2.md §2.1–2.2](../A-advanced-mathematics-2.md).
Bird reference: Ch. 6 (arithmetic + geometric progressions), Ch. 7 (binomial series).

## Contents

- [Arithmetic progression](#arithmetic-progression)
- [Geometric progression](#geometric-progression)
- [Binomial series](#binomial-series)

## Arithmetic progression

File: [lib/src/arithmetic.dart](../lib/src/arithmetic.dart).

$$u_n = a + (n-1)d, \qquad S_n = \tfrac{n}{2}[2a + (n-1)d]$$

| Function | Description |
|---|---|
| `apNthTerm(a, d, n)` | uₙ = a + (n − 1)d, n is 1-based. |
| `apSum(a, d, n)` | Sum of the first n terms. |

### Verified worked examples (Bird Ch. 6 §6.1–6.3)

| Reference | Result |
|---|---|
| Problem 1(a): 9th term of 2,7,12,…  | 42 |
| Problem 1(b): 16th term            | 77 |
| Problem 4: sum of first 12 of 5,9,13,17,… | 324 |
| Problem 5: sum of first 21 of 3.5,4.1,4.7,… | 199.5 |
| Problem 10: oil well 30,32,34,… (80 terms) | £8720 |

Tests: [test/arithmetic_test.dart](../test/arithmetic_test.dart).

## Geometric progression

File: [lib/src/geometric.dart](../lib/src/geometric.dart).

$$u_n = a\,r^{n-1}, \qquad S_n = \frac{a(1-r^n)}{1-r} \ (r\neq 1), \qquad S_\infty = \frac{a}{1-r}\ (|r|<1)$$

| Function | Description |
|---|---|
| `gpNthTerm(a, r, n)` | uₙ = a·rⁿ⁻¹, n is 1-based. |
| `gpSum(a, r, n)` | Sum of the first n terms; collapses to n·a when r = 1. |
| `gpSumToInfinity(a, r)` | Rejects `|r| ≥ 1`. |

### Example — compound interest (Bird Ch. 6 Problem 19)

```dart
// £100 at 8% p.a. after 10 years.
// Value after n years is the (n+1)th term with a = 100, r = 1.08.
gpNthTerm(100, 1.08, 11); // ≈ 215.89
```

### Verified worked examples (Bird Ch. 6 §6.4–6.6)

| Reference | Result |
|---|---|
| Problem 11: 10th term of 3,6,12,24,… | 1536 |
| Problem 12: sum of first 7 of 1/2, 3/2, 9/2,… (r=3) | 546.5 |
| Problem 15: sum of first 9 of 72.0, 57.6, 46.08 (r=0.8) | ≈ 311.7 |
| Problem 16: 3 + 1 + 1/3 + …                     | 4.5 |
| Problem 18: £400 + £400·0.9 + …                | £4000 |
| Problem 19: £100 at 8% p.a. after 10 years      | ≈ £215.89 |

Tests: [test/geometric_test.dart](../test/geometric_test.dart).

## Binomial series

File: [lib/src/binomial.dart](../lib/src/binomial.dart).

For non-negative integer n (finite polynomial):

$$(a+b)^n = \sum_{k=0}^{n}\binom{n}{k} a^{n-k} b^k, \qquad \binom{n}{k} = \frac{n!}{k!(n-k)!}$$

For real n (infinite series, converges for |x| < 1 when n ∉ ℤ⁺):

$$(1+x)^n = 1 + nx + \frac{n(n-1)}{2!}x^2 + \frac{n(n-1)(n-2)}{3!}x^3 + \cdots$$

| Function | Description |
|---|---|
| `binomialCoefficient(n, k)` | Exact `BigInt` result. |
| `pascalRow(n)` | `[C(n,0), C(n,1), …, C(n,n)]` — one row of Pascal's triangle. |
| `binomialExpansionTerms(a, b, n)` | Per-term contributions `C(n,k)·aⁿ⁻ᵏ·bᵏ`, k=0..n. |
| `binomialPow(a, b, n)` | Sum of the above — numeric `(a+b)ⁿ` for integer n ≥ 0. |
| `binomialSeries(n, x, {terms=5})` | `(1+x)ⁿ` truncated for real n. |

### Example — approximate a power without full multiplication

```dart
// Bird Ch. 7 Problem 8: (1.002)^9 ≈ 1.018145 (7 s.f.).
binomialPow(1, 0.002, 9); // ≈ 1.0181447

// Truncated real-n series: 1/(1 + 2x)^3 ≈ 1 − 6x + 24x² − 80x³.
binomialSeries(-3, 2 * 0.05, terms: 4); // ≈ 0.75; exact 1/1.1³ ≈ 0.7513
```

### Verified worked examples (Bird Ch. 7)

| Reference | Result |
|---|---|
| Problem 3: (2 + x)⁷ coefficients | 128, 448, 672, 560, 280, 84, 14, 1 |
| Problem 4: (2p − 3q)⁵ coefficients | 32, −240, 720, −1080, 810, −243 |
| Problem 8: (1.002)⁹ (7 s.f.) | ≈ 1.018145 |
| Problem 9: (0.97)⁶ (4 s.f.) | ≈ 0.8330 |
| Problem 11: 1/(1 + 2x)³ leading four terms | 1 − 6x + 24x² − 80x³ |
| Problem 13: √(1 + u) leading four terms | 1 + u/2 − u²/8 + u³/16 |
| Sanity: C(52, 5) | 2 598 960 |

Tests: [test/binomial_test.dart](../test/binomial_test.dart).

### Notes

- Convergence: the real-n series diverges for |x| ≥ 1 when n is not
  a non-negative integer. For non-negative integer n the series
  terminates and further terms are exactly zero.
- `binomialCoefficient` returns `BigInt` to preserve exactness for
  large n (e.g. `C(52, 26)`).
- 4-term truncations of the real-n series at |x| ≈ 0.1 typically have
  ~10⁻³ error — matches how Bird presents them as approximations.


---

## FILE: topic-3-complex-numbers.md

# Topic 3 — Complex numbers, Euler, de Moivre

Syllabus reference: [A-advanced-mathematics-2.md §3.1–3.3](../A-advanced-mathematics-2.md).
Bird reference: Ch. 20 (complex numbers), Ch. 21 (de Moivre's theorem).

## Contents

- [`Complex` value type](#complex-value-type)
- [Euler's formula and exponential form](#eulers-formula-and-exponential-form)
- [De Moivre — powers and roots](#de-moivre--powers-and-roots)

## `Complex` value type

File: [lib/src/complex.dart](../lib/src/complex.dart).

Immutable engineering-convention complex number `a + j·b` (using `j`,
not `i`, to avoid clashing with electric current).

| Member | Description |
|---|---|
| `Complex(re, im)` | Cartesian constructor. |
| `Complex.real(a)`, `Complex.imaginary(b)` | Convenience constructors. |
| `Complex.polar(r, θ)` | r∠θ (radians). |
| `Complex.polarDeg(r, θ°)` | r∠θ° (degrees). |
| `Complex.j`, `.zero`, `.one` | Constants. |
| `+ − * /`, unary `-`, `scale(k)` | Arithmetic. |
| `conjugate` | a − j·b. |
| `modulus` (aka `abs`) | √(a² + b²). |
| `argument` / `argumentDeg` | `atan2(im, re)`, ∈ (−π, π]. |
| `toPolar()` | `(r, θ)` record. |

Equality is strict double-equality (so `hashCode`/`Set` invariants
hold). Tests compare `.re` and `.im` with a tolerance.

### Example — a.c. impedance calculation

```dart
// Bird Ch. 20 Problem 16: V = 240 V, Z = (60 − j100) Ω.
final v = Complex.polarDeg(240, 0);
final z = Complex(60, -100);
final i = v / z; // ≈ 2.058 ∠ 59.04° A
```

### Verified worked examples (Bird Ch. 20)

| Reference | Result |
|---|---|
| Problem 5(a): Z₁·Z₂ where Z₁=1−j3, Z₂=−2+j5 | 13 + j11 |
| Problem 5(b): Z₁/Z₃ where Z₃=−3−j4 | 9/25 + j·13/25 |
| Problem 5(c): Z₁·Z₂/(Z₁+Z₂) | 1.8 − j7.4 |
| Problem 5(d): Z₁·Z₂·Z₃ | 5 − j85 |
| Problem 6(a): 2/(1+j)⁴ | −0.5 |
| Problem 6(b): j·((1+j3)/(1−j2))² | 2 |
| Problem 9: 2 + j3 → √13 ∠ 56.31° | ✓ |
| Problem 10: ±3 ± j4 four-quadrant check | ✓ |
| Problem 11: 4∠30° and 7∠−145° Cartesian conversions | ✓ |
| Problem 14: 2∠30° + 5∠−45° − 4∠120° = 9.425∠−39.54° | ✓ |

Tests: [test/complex_test.dart](../test/complex_test.dart).

## Euler's formula and exponential form

File: [lib/src/euler.dart](../lib/src/euler.dart).

$$e^{j\theta} = \cos\theta + j\sin\theta, \qquad e^{a+jb} = e^{a}(\cos b + j\sin b)$$

| Function | Description |
|---|---|
| `expJ(θ)` | e^{jθ}. |
| `complexExp(z)` | e^{z} for a Cartesian z. |
| `fromExponential(r, θ)` | Build z = r·e^{jθ}. |

### Verified

- **Euler's identity**: `expJ(π) + 1 = 0` within 1e-15.
- `complexExp(a + j·b) = e^{a}·(cos b + j·sin b)` (Ch. 21 §21.4
  derivation from the Maclaurin series of e^x, sin, cos).

## De Moivre — powers and roots

File: [lib/src/de_moivre.dart](../lib/src/de_moivre.dart).

$$(r\angle\theta)^n = r^n \angle (n\theta), \qquad z_k = r^{1/n}\angle\frac{\theta + 2\pi k}{n}, \quad k = 0, \dots, n-1$$

| Function | Description |
|---|---|
| `complexPow(z, n)` | z^n for any real n (integer, fractional, negative). |
| `complexRoots(z, n)` | All n distinct n-th roots, evenly spaced by 2π/n. |

### Example — cube roots of 8j (§3.3 syllabus)

```dart
final roots = complexRoots(Complex(0, 8), 3);
// z₀ = √3 + j, z₁ = −√3 + j, z₂ = −2j.
```

Every root has modulus 2 and cubing any one returns 8j.

### Verified worked examples (Bird Ch. 21)

| Reference | Result |
|---|---|
| Problem 1(a): [2∠35°]⁵ | 32∠175° |
| Problem 3: √(5 + j12) | ±(3 + j2) |
| Syllabus §3.3: (1 + j)⁸ | 16 |
| Syllabus §3.3: three cube roots of 8j | √3+j, −√3+j, −2j |
| Four 4th roots of 16 | ±2, ±2j |

Tests: [test/de_moivre_test.dart](../test/de_moivre_test.dart).

### Notes

- `complexRoots` returns roots in canonical order k = 0..n−1, starting
  from θ/n. Bird's "first root, then step 360°/n" ordering is preserved.
- `complexPow` accepts `num`, so `complexPow(z, -2/5)` is fine (this is
  the case that gives Bird Ch. 21 Problem 5 five distinct results).


---

## FILE: topic-4-partial-fractions.md

# Topic 4 — Partial fractions

Syllabus reference: [A-advanced-mathematics-2.md §4.1–4.4](../A-advanced-mathematics-2.md).
Bird reference: Ch. 2.

## Contents

- [`Polynomial` value type](#polynomial-value-type)
- [Partial-fraction decomposition](#partial-fraction-decomposition)

## `Polynomial` value type

File: [lib/src/polynomial.dart](../lib/src/polynomial.dart).

Immutable polynomial with real coefficients, stored in **ascending
order**: `coefficients[i]` is the coefficient of xⁱ. Trailing zeros
are trimmed so `degree` and `==` are well-defined.

| Member | Description |
|---|---|
| `Polynomial(coeffs)` | Ascending-order constructor. |
| `Polynomial.constant(c)`, `Polynomial.zero()` | Convenience. |
| `Polynomial.fromRoots([r₁, r₂, …])` | Expands `(x − r₁)(x − r₂)…`. |
| `+ − * −(unary) scale(k)` | Arithmetic. |
| `eval(x)` | Horner's method. |
| `divmod(divisor)` → `(q, r)` | Long division; `this = q·divisor + r`. |
| `degree`, `isZero`, `[]` | Introspection. |

### Example

```dart
// (x + 1)(x − 1) = x² − 1
final p = Polynomial([1, 1]) * Polynomial([-1, 1]);
p.coefficients; // [-1.0, 0.0, 1.0]

// Bird Ch. 2 Problem 3: (x² + 1) ÷ (x² − 3x + 2) → q = 1, r = 3x − 1.
final (q, r) = Polynomial([1, 0, 1]).divmod(Polynomial([2, -3, 1]));
```

## Partial-fraction decomposition

File: [lib/src/partial_fractions.dart](../lib/src/partial_fractions.dart).

Decomposes `P(x) / Q(x)` where the denominator is described as a list
of factors. Supports:

- distinct **linear factors** (x − r)
- **repeated linear factors** (x − r)^m
- irreducible **quadratic factors** (a x² + b x + c)^m
- **improper** rationals `deg P ≥ deg Q` — polynomial long division
  is performed first and the polynomial quotient is returned separately.

### Types

| Type | Purpose |
|---|---|
| `LinearFactor(root, {multiplicity})` | (x − root)^multiplicity |
| `QuadraticFactor(a, b, c, {multiplicity})` | (a x² + b x + c)^multiplicity, assumed irreducible |
| `LinearPfTerm(coefficient, factor, power)` | A / (x − r)^power |
| `QuadraticPfTerm(numeratorA, numeratorB, factor, power)` | (A x + B) / (…)^power |
| `PartialFractionResult(quotient, terms)` | Result — `quotient` is zero when the input is proper. |
| `partialFractions(numerator, factors)` | Top-level entry point. |

### How it works

Multiplying through by Q(x):

$$P(x) = \sum_i N_i(x)\,\frac{Q(x)}{D_i(x)}$$

where each `Dᵢ(x)` is a `(factor)^power` and each `Nᵢ(x)` is a
constant (linear factor) or `A·x + B` (quadratic factor). Equating
coefficients of `x⁰ … x^{deg Q − 1}` gives `deg Q` linear equations
in `deg Q` unknowns. Solved by Gaussian elimination with partial
pivoting inline in the module.

### Example

```dart
// Bird Ch. 2 Problem 6: (5x² − 2x − 19) / [(x + 3)(x − 1)²]
//   = 2/(x + 3) + 3/(x − 1) − 4/(x − 1)²
final res = partialFractions(
  Polynomial([-19, -2, 5]),
  [LinearFactor(-3), LinearFactor(1, multiplicity: 2)],
);
// res.terms contains three LinearPfTerm.
```

### Verified worked examples (Bird Ch. 2)

| Reference | Result |
|---|---|
| Problem 1: (11 − 3x) / [(x − 1)(x + 3)] | 2/(x − 1) − 5/(x + 3) |
| Problem 2: (2x² − 9x − 35) / [(x + 1)(x − 2)(x + 3)] | 4/(x + 1) − 3/(x − 2) + 1/(x + 3) |
| Problem 3: improper, (x² + 1) / (x² − 3x + 2) | 1 − 2/(x − 1) + 5/(x − 2) |
| Problem 4: improper cubic / quadratic | x − 3 + 4/(x + 2) − 3/(x − 1) |
| Problem 5: (2x + 3) / (x − 2)² | 2/(x − 2) + 7/(x − 2)² |
| Problem 6: repeated linear | 2/(x + 3) + 3/(x − 1) − 4/(x − 1)² |
| Problem 7: (3x² + 16x + 15) / (x + 3)³ | 3/(x + 3) − 2/(x + 3)² − 6/(x + 3)³ |
| Problem 8: quadratic + linear | (2x + 3)/(x² + 2) + 5/(x + 1) |
| Problem 9: x² + irreducible quadratic | 2/x + 1/x² + (−4x + 3)/(x² + 3) |
| Exercise 8-1: 12 / (x² − 9) | 2/(x − 3) − 2/(x + 3) |
| Round-trip: `Σ Aᵢ · Q/Dᵢ` reproduces original numerator | ✓ |

Tests: [test/partial_fractions_test.dart](../test/partial_fractions_test.dart),
[test/polynomial_test.dart](../test/polynomial_test.dart).

### Notes

- `QuadraticFactor` is trusted to be irreducible (i.e. `b² − 4ac < 0`).
  Not enforced.
- Repeated-linear-at-root-0 (i.e. `x^m`) is expressed as
  `LinearFactor(0, multiplicity: m)` — see Bird Problem 9.
- The inline Gaussian solver is small (~20 lines). Phase E will replace
  it with a call into the linear-algebra module.
- Feeds directly into Phase F (inverse Laplace transform), where the
  same decomposition reduces `F(s)` to a sum of table entries.


---

## FILE: topic-5-linear-algebra.md

# Topic 5 — Determinants, matrices, ODE systems

Syllabus reference: [A-advanced-mathematics-2.md §5.1–5.5](../A-advanced-mathematics-2.md).
Bird reference: Ch. 22 (matrix theory + determinants), Ch. 23 (systems by matrices, Cramer's rule).

## Contents

- [`Matrix` value type](#matrix-value-type)
- [Determinants, minors, cofactors, adjugate](#determinants-minors-cofactors-adjugate)
- [Inverse and equation solving](#inverse-and-equation-solving)
- [LU and QR decompositions (Phase P)](#lu-and-qr-decompositions-phase-p)
- [Least squares and multiple linear regression](#least-squares-and-multiple-linear-regression)
- [Eigenvalues and eigenvectors (2×2)](#eigenvalues-and-eigenvectors-22)
- [General eigenvalue decomposition (Phase R)](#general-eigenvalue-decomposition-phase-r)
- [Singular Value Decomposition (Phase U)](#singular-value-decomposition-phase-u)
- [Linear ODE systems](#linear-ode-systems)

## `Matrix` value type

File: [lib/src/matrix.dart](../lib/src/matrix.dart).

Immutable m × n real matrix, row-major. Construction copies input so
external mutation cannot leak in.

| Member | Description |
|---|---|
| `Matrix.fromRows([[…], …])` | Rectangular constructor. |
| `Matrix.zero(rows, cols)`, `Matrix.identity(n)` | Convenience constructors. |
| `Matrix.column([…])` | n × 1 column vector. |
| `.at(r, c)`, `.row(r)`, `.column(c)` | Element / row / column access. |
| `.shape`, `.isSquare`, `.rows`, `.cols` | Introspection. |
| `+ − * −(unary)`, `.scale(k)`, `.transpose` | Arithmetic. |
| `.submatrix(row, col)` | Delete a row + column (used for minors/cofactors). |
| `.replaceColumn(col, values)` | Used by Cramer's rule. |

Equality is strict double-equality per element.

### Example

```dart
final a = Matrix.fromRows([[1, 2], [3, 4]]);
a * Matrix.identity(2); // == a
a.transpose;             // [[1, 3], [2, 4]]
```

### Verified worked examples (Bird Ch. 22 §22.1–22.3)

| Reference | Result |
|---|---|
| Problem 1: 2×2 and 3×3 addition | ✓ |
| Problem 2: 2×2 subtraction | ✓ |
| Problem 4: `2A − 3B + 4C` | ✓ |
| Problem 5: 2×2 × 2×2 | `[[-19, 26], [7, -9]]` |
| Problem 6: 3×3 × 3×1 | `[26, 29, -7]` |
| Problem 7: 3×3 × 3×2 | `[[26, -39], [29, -5], [-7, -18]]` |
| Problem 8: 3×3 × 3×3 | `[[11, 8, 0], [11, 11, 2], [8, 13, 6]]` |
| Problem 9: `A·B ≠ B·A` | ✓ (both products computed and shown to differ) |
| Round-trip: `(Aᵀ)ᵀ = A` | ✓ |

Tests: [test/matrix_test.dart](../test/matrix_test.dart).

## Determinants, minors, cofactors, adjugate

File: [lib/src/determinant.dart](../lib/src/determinant.dart).

| Function | Description |
|---|---|
| `determinant(m)` | Cofactor closed form for n ≤ 3; LU row-reduction with partial pivoting for n ≥ 4. |
| `minor(m, row, col)` | Determinant of the submatrix after deleting `row`, `col`. |
| `cofactor(m, row, col)` | Signed minor: `(−1)^(row+col) · minor`. |
| `cofactorMatrix(m)` | Matrix of cofactors, same shape. |
| `adjugate(m)` | Transpose of the cofactor matrix. |

### Verified worked examples (Bird Ch. 22 §22.4 and §22.6)

| Reference | Result |
|---|---|
| Problem 10: `|3, −2; 7, 4|` | 26 |
| Problem 14: `|3,4,−1; 2,0,7; 1,−3,−2|` | 113 |
| Problem 15: `|1,4,−3; −5,2,6; −1,−4,2|` | −22 |
| Exercise 94-1..3, Exercise 96-3..4 | ✓ |
| `det(Aᵀ) = det(A)` | ✓ |
| `det(A·B) = det(A)·det(B)` | ✓ |
| `A · adj(A) = det(A) · I` | ✓ |
| Singular matrix → determinant is 0 | ✓ |
| 4×4 Vandermonde-like via row reduction (det = 12) | ✓ |

Tests: [test/determinant_test.dart](../test/determinant_test.dart).

## Inverse and equation solving

File: [lib/src/matrix_solve.dart](../lib/src/matrix_solve.dart).

| Function | Description |
|---|---|
| `inverse(m)` | Adjugate/det for n ≤ 3, Gauss–Jordan on `[A | I]` for n ≥ 4. |
| `solveByInverse(A, b)` | Returns `A⁻¹ b` as a `List<double>`. |
| `solveCramer(A, b)` | Cramer's rule: `xᵢ = det(Aᵢ)/det(A)`. |

Both `inverse` and `solveCramer` reject singular systems with a
`StateError` when `|det| < 1e-14`.

### Verified worked examples

Bird Ch. 22 §22.5 / §22.7 and Ch. 23 §23.1–23.3:

| Reference | Result |
|---|---|
| Problem 13: inverse of `[3,−2; 7,4]` | `(1/26)·[4, 2; −7, 3]` |
| Problem 18: inverse of `[1,5,−2; 3,−1,4; −3,6,−7]` | `[[8.5,−11.5,−9], [−4.5,6.5,5], [−7.5,10.5,8]]` |
| Ch. 23 Problem 1: `3x + 5y = 7, 4x − 3y = 19` | `x=4, y=−1` |
| §5.3 syllabus example: `3x + 2y = 12, 5x − y = 7` | `x=2, y=3` |
| Ch. 23 Problem 2: 3×3 system | `x=2, y=−3, z=5` |
| Ch. 23 Exercise 98-3 | `(1, −1, 2)` |
| Ch. 23 Exercise 99-6 (forces) | `F₁ = 1.5, F₂ = −4.5` |
| `A · A⁻¹ = I` round-trip (3×3 and 4×4) | ✓ |
| `solveByInverse` and `solveCramer` agree | ✓ |

Tests: [test/matrix_solve_test.dart](../test/matrix_solve_test.dart).

### Example — Cramer's rule

```dart
// §5.3 syllabus: 3x + 2y = 12,  5x − y = 7.
final a = Matrix.fromRows([[3, 2], [5, -1]]);
final x = solveCramer(a, [12, 7]); // [2, 3]
```

## Eigenvalues and eigenvectors (2×2)

File: [lib/src/eigen.dart](../lib/src/eigen.dart).

Only the real, distinct/repeated case is covered — the syllabus §5.5
worked example lives here. Complex eigenvalues throw a `StateError`;
they're deferred until we need them for damped-oscillator Laplace work.

| Function | Description |
|---|---|
| `eigenvalues2x2(A)` | Returns `[λ₁, λ₂]`. Throws if Δ < 0. |
| `eigenvector2x2(A, λ)` | Returns a representative eigenvector (not normalised). |

### Verified worked examples

| Reference | Result |
|---|---|
| §5.5 syllabus: `A = [[2, 1], [1, 2]]` → λ = 1, 3; eigenvectors `(1, −1)`, `(1, 1)` | ✓ |
| Bird Ex. 96 Problem 8(a): `A = [[2, 2], [−1, 5]]` | λ = 3, 4 |
| Identity → repeated eigenvalue 1 | ✓ |
| Rotation matrix `[[0, −1], [1, 0]]` → StateError (complex) | ✓ |
| `A · v = λ · v` cross-check for both eigenpairs | ✓ |

Tests: [test/eigen_test.dart](../test/eigen_test.dart).

## General eigenvalue decomposition (Phase R)

Also in [lib/src/eigen.dart](../lib/src/eigen.dart).

### `SymmetricEigenDecomposition.of(A)` — cyclic Jacobi rotations

Full spectral factorisation `A = V · diag(eigenvalues) · Vᵀ` for any
real symmetric `A`. Eigenvalues are returned in **descending** order;
eigenvector columns of `V` match. Converges in typically 5–10 sweeps.
Refuses non-symmetric input (`ArgumentError`).

```dart
final a = Matrix.fromRows([
  [4, 1, 2],
  [1, 3, 0],
  [2, 0, 5],
]);
final eig = SymmetricEigenDecomposition.of(a);
eig.eigenvalues;         // [~6.4, ~3.5, ~2.1] (descending)
eig.eigenvectors;        // 3 × 3 orthogonal, columns matching eigenvalues
```

### `powerIteration(A, initial: v0)` — dominant eigenpair

Linear convergence at rate `|λ₂/λ₁|`. Best when the spectral gap is
wide. Cheap: one mat-vec per iteration.

```dart
final r = powerIteration(a);
r.eigenvalue;             // ≈ dominant eigenvalue
r.eigenvector;            // ‖·‖ = 1
r.iterations;
```

### `rayleighQuotientIteration(A, shift, initial: v0)` — targeted

Cubic convergence to *an* eigenvalue determined jointly by the shift
and the initial vector. Provide an approximate eigenvector as `initial`
when you want to home in on a specific eigenpair.

```dart
rayleighQuotientIteration(a, 2.5);         // any eigenpair near 2.5
rayleighQuotientIteration(a, 5.9,
    initial: powerIteration(a).eigenvector); // refine dominant to 1e-12
```

Tests: [test/eigen_pca_test.dart](../test/eigen_pca_test.dart) — Jacobi
reconstructs `A = VDVᵀ` to 1e-9 on 3×3 and 4×4 examples, power-iteration
matches Jacobi's dominant eigenvalue, RQI cubic convergence verified.

## Linear ODE systems

File: [lib/src/ode_system.dart](../lib/src/ode_system.dart).

$$\frac{d\mathbf{x}}{dt} = A\mathbf{x}, \qquad \mathbf{x}(0) = \mathbf{x}_0$$

For 2 × 2 real A with real distinct eigenvalues:

$$\mathbf{x}(t) = c_1 \mathbf{v}_1 e^{\lambda_1 t} + c_2 \mathbf{v}_2 e^{\lambda_2 t}$$

with `(c₁, c₂)` obtained from the initial condition via Cramer.

| Function | Description |
|---|---|
| `solveLinear2x2System(A, x0)` | Returns a `List<double> Function(num t)` closure evaluating `x(t)`. |

### Example

```dart
// §5.5 syllabus: x'₁ = 2x₁ + x₂, x'₂ = x₁ + 2x₂, x(0) = (2, 0).
final a = Matrix.fromRows([[2, 1], [1, 2]]);
final x = solveLinear2x2System(a, [2, 0]);
x(0);   // ≈ [2, 0]
x(0.3); // ≈ [e^0.3 + e^0.9, -e^0.3 + e^0.9]
```

### Verified

- `x(0)` recovers the initial condition.
- Numerical central-difference test: `dx/dt` at t = 0.5 matches `A · x`
  within 1e-3.
- Closed-form check at t = 0.3 for the syllabus example.

Tests: [test/eigen_test.dart](../test/eigen_test.dart) (the ODE group).

## Notes and limitations

- **Complex eigenvalues** in the general non-symmetric case are still
  unimplemented. `SymmetricEigenDecomposition` guarantees real
  eigenvalues by construction (any real symmetric matrix has a real
  spectrum); non-symmetric general n × n eigenvalues would need a
  Hessenberg + implicit-shift QR path with complex arithmetic — future
  work if a use case demands it.
- **`inverse` for n ≥ 4** uses Gauss–Jordan; expect the usual
  double-precision loss on ill-conditioned matrices. No condition-number
  check is performed.
- The inline Gaussian solver still lives inside
  [partial_fractions.dart](../lib/src/partial_fractions.dart) (Phase D).
  It has been left unchanged because it works and has 12 tests around
  it; refactoring it to route through `matrix_solve.dart` is a
  cosmetic follow-up.

## LU and QR decompositions (Phase P)

File: [lib/src/linear_algebra.dart](../lib/src/linear_algebra.dart).

Numerically-stable decompositions for arbitrary `n`. The original
adjugate / Cramer / Gauss–Jordan routes still live in
[matrix_solve.dart](../lib/src/matrix_solve.dart) and are best for
hand-worked 2 × 2 / 3 × 3 problems; use these for anything larger or
where numerical conditioning matters.

### `LUDecomposition.of(A)` — PA = LU with partial pivoting

```dart
final a = Matrix.fromRows([
  [4.0, 1, 2, 3, 1],
  [1, 5, 1, 2, 3],
  [2, 1, 6, 1, 4],
  [3, 2, 1, 7, 2],
  [1, 3, 4, 2, 8],
]);
final lu = LUDecomposition.of(a);
lu.solve([12, 20, 24, 22, 32]);   // O(n²) re-use
lu.determinant();                   // ±∏ Uᵢᵢ
lu.inverse();                       // solves n right-hand-sides
```

Shortcut for a one-off solve: `solveLU(A, b)`.

### `QRDecomposition.of(A)` — Householder reflections

Works for rectangular `A` with `m ≥ n`. `Q` is `m × m` orthogonal
(`Qᵀ · Q = I`), `R` is `m × n` upper-triangular in the leading block.
Preferred for least-squares because it never forms `AᵀA`.

```dart
final qr = QRDecomposition.of(a);
qr.q; qr.r;
qr.leastSquares(b);   // returns the OLS coefficients
```

Shortcut: `linearLeastSquares(A, b)`.

### `CholeskyDecomposition.of(A)` — symmetric positive-definite

Half the flops of LU. Requires `A = Aᵀ` and all leading principal
minors positive — throws `ArgumentError` on asymmetry, `StateError` on
indefiniteness. Ideal for covariance matrices, normal equations, and
finite-element stiffness matrices.

```dart
final ch = CholeskyDecomposition.of(spdMatrix);
ch.l;                 // A = L · Lᵀ, lower-triangular
ch.solve(b);          // O(n²) per RHS
ch.determinant();     // ∏ Lᵢᵢ²
```

## Least squares and multiple linear regression

File: [lib/src/regression.dart](../lib/src/regression.dart).

`multipleLinearRegression(xs, ys)` fits
`Y = β₀ + β₁ X₁ + β₂ X₂ + … + β_p X_p` via QR — an order-of-magnitude
condition-number improvement over the normal-equations route. Returns a
`MultipleLinearRegression` with `intercept`, `slopes`, `predict(x)`,
and `rSquared(xs, ys)`.

```dart
final xs = [
  [1.0, 1.0], [2.0, 3.0], [4.0, 5.0], [1.0, 4.0],
  [3.0, 2.0], [5.0, 1.0], [2.0, 5.0], [4.0, 2.0],
];
final ys = <double>[
  for (final r in xs) 5 + 2 * r[0] - 3 * r[1],
];
final fit = multipleLinearRegression(xs, ys);
fit.intercept;    // 5
fit.slopes;       // [2, -3]
fit.predict([3, 4]);
fit.rSquared(xs, ys);
```

For a single predictor, `multipleLinearRegression([[x]], ys)` returns
the same fit as `regressionYonX(xs, ys)`; there's a test that confirms
this.

Tests: [test/linear_algebra_test.dart](../test/linear_algebra_test.dart)
— 14 tests exercising the decompositions, back-substitution correctness,
orthogonality of Q, R-upper-triangular structure, and rank-deficient
error cases.

## Singular Value Decomposition (Phase U)

File: [lib/src/svd.dart](../lib/src/svd.dart).

The workhorse of numerical linear algebra: `A = U · Σ · Vᵀ` for any
real m × n matrix. Implementation is **one-sided cyclic Jacobi** on
the columns of A — simpler than Golub–Reinsch bidiagonalisation, with
guaranteed convergence and fine performance for engineering-sized
matrices (a few hundred rows).

`SVD.of(A)` returns the "thin" (economy-sized) decomposition:
- `u` — m × k with orthonormal columns
- `singularValues` — length k, descending
- `v` — n × k with orthonormal columns

where `k = min(m, n)`. Tall (m > n), square, and wide (m < n)
matrices are all handled; the wide case is routed through the
transpose and U/V are swapped at the end.

### What SVD unlocks

| Method | Purpose |
|---|---|
| `svd.rank({tolerance})` | Effective numerical rank. Default tolerance follows LAPACK: `max(m, n) · ε_mach · σ_max`. |
| `svd.conditionNumber()` | `σ_max / σ_min` — 2-norm condition number. Returns `∞` when `σ_min = 0`. |
| `svd.pseudoinverse({tolerance})` | Moore–Penrose `A⁺`. For invertible `A`, matches `inverse(A)` to 1e-9. Satisfies `A · A⁺ · A = A` (verified in tests). |
| `svd.leastSquares(b, {tolerance})` | Minimum-norm LS solution to `A · x = b`. Works for **any** m × n, including rank-deficient and under-determined — cases where the Phase P QR route throws. |

### Example

```dart
final a = Matrix.fromRows([
  [1.0, 2, 3, 4],
  [2, 3, 4, 5],
  [3, 4, 5, 6],
]);
final svd = SVD.of(a);

svd.singularValues;      // [~13.2, ~0.8, ~1e-15] — rank 2
svd.rank();              // 2
svd.conditionNumber();   // ~1.6e16 (numerically infinite)

final aPlus = svd.pseudoinverse();     // 4 × 3
final x = svd.leastSquares([1, 2, 3]); // minimum-norm x
```

### When to reach for SVD vs QR

| Situation | Route |
|---|---|
| Full column rank, m ≥ n | `linearLeastSquares` (Phase P QR) — faster |
| Rank-deficient, under-determined, or unknown conditioning | `SVD.of(A).leastSquares(b)` |
| Need pseudoinverse | `SVD.of(A).pseudoinverse()` |
| Need rank / condition number | `SVD.of(A).rank()` / `.conditionNumber()` |

Tests: [test/svd_test.dart](../test/svd_test.dart) — reconstruction
`U · Σ · Vᵀ = A` to 1e-10 on square/tall/wide inputs; orthonormal
`U` and `V` verified via `QᵀQ = I`; rank-deficient matrix reports
rank 2 not 3; pseudoinverse matches `inverse(A)` for invertible A;
Moore–Penrose axiom holds; SVD-LS matches QR-LS on well-posed cases
and solves the rank-deficient cases QR refuses.


---

## FILE: topic-6-laplace.md

# Topic 6 — Laplace transforms

Syllabus reference: [A-advanced-mathematics-2.md §6.1–6.5](../A-advanced-mathematics-2.md).
Bird reference: Ch. 61 (definition + elementary transforms),
Ch. 62 (first shift theorem), Ch. 63 (inverse Laplace), Ch. 64 (ODE
solutions via Laplace).

## Contents

- [Rational functions in s](#rational-functions-in-s)
- [Time-domain expressions (AST)](#time-domain-expressions-ast)
- [Forward Laplace transform](#forward-laplace-transform)
- [Inverse Laplace transform](#inverse-laplace-transform)
- [ODE solver via Laplace](#ode-solver-via-laplace)

## Rational functions in s

File: [lib/src/rational_function.dart](../lib/src/rational_function.dart).

Used for s-domain values. Not automatically reduced — Laplace tables
map directly to unreduced rational forms and we cross-check by
evaluating at specific s, so GCD reduction is unnecessary here.

| Member | Description |
|---|---|
| `RationalFunction(num, den)` | Constructor. Rejects zero denominator. |
| `RationalFunction.constant(c)` | c/1. |
| `RationalFunction.zero` | 0/1 (const). |
| `+ − * scaled(k)` | Cross-multiplication arithmetic. |
| `.eval(s)` | Evaluate at a specific s. |

## Time-domain expressions (AST)

File: [lib/src/laplace_expression.dart](../lib/src/laplace_expression.dart).

Sealed hierarchy `TimeTerm` for the atoms that appear in Bird's
Table 61.1 (and Table 62.1 via `TShifted`). `TimeExpr(List<TimeTerm>)`
is a linear combination.

| Type | Represents |
|---|---|
| `TConst(k)` | `k` |
| `TPower(n, {scale})` | `scale · tⁿ`, n ≥ 1 |
| `TExp(a, {scale})` | `scale · e^{a·t}` |
| `TSin(ω, {scale})` | `scale · sin(ω·t)` |
| `TCos(ω, {scale})` | `scale · cos(ω·t)` |
| `TSinh(a, {scale})` | `scale · sinh(a·t)` |
| `TCosh(a, {scale})` | `scale · cosh(a·t)` |
| `TShifted(α, inner)` | `e^{α·t} · inner(t)` — first shift theorem |

`TimeExpr`:

| Member | Description |
|---|---|
| `TimeExpr(terms)`, `TimeExpr.zero()`, `TimeExpr.of(t)`, `TimeExpr.constant(c)` | Constructors. |
| `laplace()` | Forward transform, returns `RationalFunction`. |
| `eval(t)` | Numeric evaluation. |
| `+`, `-`, `scaled(k)` | Linearity. |

### Example

```dart
// f(t) = 6 sin 3t − 4 cos 5t.
final f = TimeExpr([TSin(3, scale: 6), TCos(5, scale: -4)]);
laplace(f).eval(2.0);
// = 18/(4+9) − 4·2/(4+25) = 18/13 − 8/29.
```

## Forward Laplace transform

File: [lib/src/laplace_transform.dart](../lib/src/laplace_transform.dart).

| Function | Description |
|---|---|
| `laplace(f)` | `f.laplace()` — returns `RationalFunction`. |

### Verified worked examples (Bird Ch. 61 and 62)

| Reference | Result |
|---|---|
| Table 61.1: L{1}, L{k}, L{tⁿ}, L{e^{at}}, L{sin at}, L{cos at}, L{sinh at}, L{cosh at} | All match at multiple s |
| Problem 1(a): `L{1 + 2t − t⁴/3}` = `1/s + 2/s² − 8/s⁵` | ✓ |
| Problem 1(b): `L{5e^{2t} − 3e^{−t}}` = `5/(s−2) − 3/(s+1)` | ✓ |
| Problem 2(a): `L{6 sin 3t − 4 cos 5t}` | ✓ |
| Problem 2(b): `L{2 cosh 2θ − sinh 3θ}` | ✓ |
| Table 62.1 (i): `L{2 t⁴ e^{3t}}` = `48/(s−3)⁵` | ✓ |
| Table 62.1 (iii): `L{4 e^{3t} cos 5t}` = `4(s−3)/[(s−3)²+25]` | ✓ |
| Table 62.1 (ii): `L{e^{−2t} sin 3t}` | ✓ |
| Table 62.1 (iv): `L{5 e^{−3t} sinh 2t}` | ✓ |
| Table 62.1 (v): `L{3 e^{θ} cosh 4θ}` | ✓ |

Tests: [test/laplace_forward_test.dart](../test/laplace_forward_test.dart).

## Inverse Laplace transform

File: [lib/src/laplace_transform.dart](../lib/src/laplace_transform.dart).

Given a rational F(s) and the factorization of its denominator,
`inverseLaplace` decomposes F(s) into partial fractions (Phase D)
and inverts each term via Bird's Table 63.1:

- `A/(s − r)^n`             → `A · t^{n−1} · e^{r·t} / (n−1)!`
- `(A s + B)/[(s − a)² + ω²]` → `A e^{a·t} cos(ω·t) + ((A a + B)/ω) e^{a·t} sin(ω·t)`
- `(A s + B)/[(s − a)² − ω²]` → `A e^{a·t} cosh(ω·t) + ((A a + B)/ω) e^{a·t} sinh(ω·t)`

Completion of the square is done automatically for `QuadraticFactor`s
based on the sign of `Δ = c − b²/4`.

### API

```dart
TimeExpr inverseLaplace(
  RationalFunction f, {
  required List<Factor> factors,
});
```

The user supplies the denominator's factorization (matches Bird's
pen-and-paper workflow). Numerator is auto-normalised for non-monic
denominators.

### Example

```dart
// Bird Ch. 63 Problem 5(a): L⁻¹{ 3 / (s² − 4s + 13) } = e^{2t} sin 3t.
final F = RationalFunction(
  Polynomial([3]),
  Polynomial([13, -4, 1]),
);
final y = inverseLaplace(F, factors: [QuadraticFactor(1, -4, 13)]);
y.eval(0.5); // ≈ e^{1.0} · sin 1.5
```

### Verified worked examples (Bird Ch. 63)

| Reference | Result |
|---|---|
| Problem 7 pattern: `(4s − 5)/(s² − s − 2)` | `e^{2t} + 3 e^{−t}` |
| Problem 5(a): `3/(s² − 4s + 13)` | `e^{2t} sin 3t` |
| Problem 5(b): `2(s + 1)/(s² + 2s + 10)` | `2 e^{−t} cos 3t` |
| Problem 6(a): `5/(s² + 2s − 3)` via quadratic factor | `2.5 e^{−t} sinh 2t` |
| Exercise 224-1: `(11 − 3s)/(s² + 2s − 3)` | `2 eᵗ − 5 e^{−3t}` |
| Exercise 224-2: `(2s² − 9s − 35) / [(s+1)(s−2)(s+3)]` | `4 e^{−t} − 3 e^{2t} + e^{−3t}` |
| Table 63.1 (xi): `2/(s − 3)⁵` | `t⁴ e^{3t} / 12` |
| Round-trip: `L(L⁻¹(F))(s) ≈ F(s)` | ✓ |

Tests: [test/laplace_inverse_test.dart](../test/laplace_inverse_test.dart).

## ODE solver via Laplace

File: [lib/src/laplace_ode.dart](../lib/src/laplace_ode.dart).

Solves any constant-coefficient linear ODE

$$\sum_i c_i\,y^{(i)}(t) = f(t),\qquad y(0),\,y'(0),\,\dots,\,y^{(n-1)}(0)$$

by transforming, solving for `L{y}` algebraically, and inverting.

### API

```dart
TimeExpr solveOdeLaplace({
  required List<num> coefficients,       // ascending (c_i multiplies y^(i))
  required List<num> initialConditions,  // [y(0), y'(0), ..., y^(n-1)(0)]
  required TimeExpr forcing,             // f(t)
  required List<Factor> denominatorFactors, // factorization of L{y} denominator
});
```

Bird's method is pen-and-paper: the user manually factors the s-domain
denominator (characteristic polynomial × forcing's denominator) and
supplies it as `denominatorFactors`.

### Example — RL step response (syllabus §6.5)

```dart
// L·di/dt + R·i = V,  L = 1, R = 2, V = 12, i(0) = 0.
// → i(t) = 6(1 − e^{−2t}).
final i = solveOdeLaplace(
  coefficients: [2, 1],           // 2·i + 1·i'
  initialConditions: [0],
  forcing: TimeExpr.constant(12),
  denominatorFactors: [LinearFactor(-2), LinearFactor(0)],
);
i.eval(0.5); // ≈ 6 · (1 − e^{-1}) ≈ 3.792
```

### Verified worked examples

| Reference | Result |
|---|---|
| Syllabus §6.5 RL step: `i' + 2i = 12, i(0) = 0` | `6(1 − e^{−2t})` |
| Syllabus §6.5 2nd order: `y'' + 3y' + 2y = 0, y(0)=1, y'(0)=0` | `2 e^{−t} − e^{−2t}` |
| Bird Ch. 64 Problem 1: `2y'' + 5y' − 3y = 0, y(0)=4, y'(0)=9` | `6 e^{t/2} − 2 e^{−3t}` |
| Bird Ch. 64 Problem 2: `y'' + 6y' + 13y = 0, y(0)=3, y'(0)=7` (complex roots) | `e^{−3t}(3 cos 2t + 8 sin 2t)` |
| Initial-condition sanity: `y(0)` reproduces input | ✓ |
| Rejects mismatched initial-condition count | ✓ |
| Rejects zero leading coefficient | ✓ |

Tests: [test/laplace_ode_test.dart](../test/laplace_ode_test.dart).

## Notes and limitations

- **Unit step / Dirac delta** not implemented. Bird's later chapters
  (65 onwards) cover them; not required for §6.1–6.5.
- **Repeated quadratic factors** rejected with `UnimplementedError`.
  Bird's syllabus examples never exercise this case.
- **Improper input** to `inverseLaplace` (`deg P ≥ deg Q`) rejected
  with `ArgumentError`. Every ODE that arises from the syllabus is
  automatically proper because the boundary polynomial from initial
  conditions has degree < characteristic polynomial's degree.
- **Non-monic quadratic factors** (`a ≠ 1`) are rescaled internally,
  but the syllabus examples always use monic `QuadraticFactor`.
- No **factoring** of the characteristic polynomial is performed — the
  caller supplies the factorization. Bird's syllabus problems come with
  their factorizations spelled out.


---

## FILE: topic-e-electronics.md

# Topic E — Electronics engineering

New in the library: named APIs for the day-to-day mathematics of
electronics — impedance, phasors, AC power, transfer functions, Bode
plots. All of it sits on top of the existing `Complex`, `Polynomial`,
`Vector`, and Laplace machinery; these modules just add
electronics-engineering vocabulary and unit conventions.

Bird reference: Ch. 20 (complex numbers → phasors), Ch. 41 (Laplace →
transfer functions).

## Contents

- [Unit conversions](#unit-conversions)
- [Impedance](#impedance)
- [Phasors](#phasors)
- [AC power](#ac-power)
- [Transfer functions](#transfer-functions)
- [Bode plots](#bode-plots)
- [Filter design](#filter-design)
- [Transmission lines](#transmission-lines)
- [Semiconductors](#semiconductors)
- [Worked examples](#worked-examples)

## Unit conversions

File: [lib/src/electronics/units.dart](../lib/src/electronics/units.dart).

| Function | Meaning |
|---|---|
| `omegaFromHz(f)` | ω = 2π · f  (rad/s from Hz) |
| `hzFromOmega(ω)` | f = ω / (2π) |
| `peakToRms(pk)`  | V_rms = V_pk / √2   (sinusoid only) |
| `rmsToPeak(rms)` | V_pk = V_rms · √2 |
| `dbToRatio(dB)`  | 10^{dB/20} (voltage-ratio convention) |
| `ratioToDb(r)`   | 20 · log₁₀ r |

## Impedance

File: [lib/src/electronics/impedance.dart](../lib/src/electronics/impedance.dart).

Impedances are stored as `Complex` values in ohms. At angular frequency
ω:

$$Z_R = R, \qquad Z_L = j\omega L, \qquad Z_C = \frac{1}{j\omega C} = -\frac{j}{\omega C}.$$

| Function | Description |
|---|---|
| `impedanceR(R)` | Z = R + j·0 |
| `impedanceL(L, ω)` | Z = jωL |
| `impedanceC(C, ω)` | Z = −j/(ωC) |
| `seriesImpedance([Z…])` | Σ Zₖ |
| `parallelImpedance([Z…])` | (Σ 1/Zₖ)⁻¹ — a short in ANY branch collapses to 0 |
| `admittance(Z)` | Y = 1/Z |
| `voltageDivider(V, Z₁, Z₂)` | V · Z₂ / (Z₁ + Z₂) |
| `currentDivider(I, Zbranch, Zother)` | I · Zother / (Zbranch + Zother) |

### Example — RC low-pass at cutoff

```dart
const R = 1000.0;
const C = 1e-6;
final fc = 1 / (2 * math.pi * R * C);          // ≈ 159.15 Hz
final vout = voltageDivider(
  Complex.real(1),                              // V_in = 1 V
  impedanceR(R),                                // Z₁ = R
  impedanceC(C, omegaFromHz(fc)),               // Z₂ = 1/(jωC)
);
vout.modulus;      // ≈ 1/√2  (−3 dB)
vout.argumentDeg;  // ≈ −45°
```

## Phasors

File: [lib/src/electronics/phasor.dart](../lib/src/electronics/phasor.dart).

A phasor is a complex number carrying (magnitude, phase) of a sinusoid.
Depending on convention the magnitude is either the peak or the RMS
value; the library does not enforce a choice — pass whichever you're
working with.

| Function | Description |
|---|---|
| `phasorRad(M, φ)` | Complex from (magnitude, phase-in-radians) |
| `phasorDeg(M, φ°)` | (magnitude, phase-in-degrees) |
| `instantaneousValue(P, t, ω)` | v(t) = \|P\|·cos(ωt + arg P) for a PEAK phasor |
| `ohmVoltage(I, Z)` | V = I·Z |
| `ohmCurrent(V, Z)` | I = V/Z |
| `phaseDifferenceDeg(a, b)` | arg a − arg b wrapped to (−180°, 180°] |

## AC power

File: [lib/src/electronics/ac_power.dart](../lib/src/electronics/ac_power.dart).

$$S = V \cdot \overline{I}, \qquad P = |V|\,|I|\,\cos\phi, \qquad Q = |V|\,|I|\,\sin\phi.$$

| Function | Description |
|---|---|
| `complexPower(V, I)` | S = V·conj(I) in VA (V, I RMS phasors) |
| `apparentPower(vRms, iRms)` | \|S\| (VA) |
| `realPower(vRms, iRms, φ)` | P (W) |
| `reactivePower(vRms, iRms, φ)` | Q (VAR) |
| `powerFactor(φ)` | cos φ |
| `rmsFromFourierCoefficients(a0, a, b)` | Parseval: RMS of a Fourier series |
| `totalHarmonicDistortion(a, b)` | √(Σ Vₙ² for n≥2) / V₁ |

### Example — 230 V, 10 A, 0.8 PF lagging

```dart
final phi = math.acos(0.8);
realPower(230, 10, phi);       // 1840 W
reactivePower(230, 10, phi);   // 1380 VAR
apparentPower(230, 10);        // 2300 VA
```

## Transfer functions

File: [lib/src/electronics/transfer_function.dart](../lib/src/electronics/transfer_function.dart).

A `TransferFunction` holds `numerator` and `denominator` polynomials.
Coefficients are ascending, matching [`Polynomial`](../lib/src/polynomial.dart):
`[c₀, c₁, c₂, …]` means `c₀ + c₁·s + c₂·s² + …`.

| Member | Description |
|---|---|
| `TransferFunction(N, D)` | Direct constructor from polynomials |
| `TransferFunction.fromCoefficients(numerator:, denominator:)` | From ascending-coefficient lists |
| `.evaluate(s)` | H(s) at a Complex point |
| `.frequencyResponse(ω)` | H(jω) at angular frequency ω |
| `.dcGain()` | H(0) |
| `tf1 * tf2` | Series (cascade) composition |
| `.scale(k)` | Multiply the numerator by scalar k |

### Example — 1st-order low-pass  H(s) = 1 / (1 + s/ω₀)

```dart
const omega0 = 2 * math.pi * 1000;              // 1 kHz cutoff
final tf = TransferFunction.fromCoefficients(
  numerator: [1],
  denominator: [1, 1 / omega0],                 // 1 + s/ω₀
);
tf.frequencyResponse(omega0).modulus;      // 1/√2  (−3.0103 dB)
tf.frequencyResponse(omega0).argumentDeg;  // −45°
```

## Bode plots

File: [lib/src/electronics/bode.dart](../lib/src/electronics/bode.dart).

| Function | Description |
|---|---|
| `BodePoint(f, magDb, phaseDeg)` | One sample of a Bode plot |
| `bodeSweep(tf, fMin:, fMax:, points:, log:)` | Sweep H(j2πf); default logarithmic |
| `unwrapDegrees(list)` | Remove ±360° jumps from a phase array |
| `cutoffFrequency3dB(tf, fMin:, fMax:)` | Bisect the sweep for the −3 dB point |

### Example — Bode sweep of the 1 kHz LPF above

```dart
final sweep = bodeSweep(tf,
    fMin: 10, fMax: 100000, points: 200);
for (final p in sweep) {
  // p.frequencyHz, p.magnitudeDb, p.phaseDeg
}
cutoffFrequency3dB(tf, fMin: 10, fMax: 100000); // ≈ 1000 Hz
```

## Filter design

File: [lib/src/electronics/filter_design.dart](../lib/src/electronics/filter_design.dart).

| Function | Description |
|---|---|
| `rcLowPass(R, C)` | H(s) = 1 / (1 + sRC) — first-order LPF |
| `rcHighPass(R, C)` | H(s) = sRC / (1 + sRC) — first-order HPF |
| `rlcBandPass(R, L, C)` | Series RLC, output across R |
| `naturalFrequencyLC(L, C)` | ω₀ = 1/√(LC) |
| `qualityFactorSeriesRLC(R, L, C)` | Q = √(L/C) / R |
| `dampingRatioSeriesRLC(R, L, C)` | ζ = R/2 · √(C/L) = 1/(2Q) |
| `butterworthPoles(order)` | Unit-circle prototype poles in the left half-plane |
| `butterworthLowPass(order:, omegaC:)` | n-th order LPF, `|H(jω_c)| = 1/√2`, DC gain 1 |

Verified across orders 1–5: exactly `−3 dB` at `ω_c` and rolls off at
`20n dB/decade`. See [test/ee_filter_test.dart](../test/ee_filter_test.dart).

## Transmission lines

File: [lib/src/electronics/transmission_line.dart](../lib/src/electronics/transmission_line.dart).

$$\Gamma = \frac{Z_L - Z_0}{Z_L + Z_0}, \quad \text{VSWR} = \frac{1 + |\Gamma|}{1 - |\Gamma|}, \quad Z_{\text{in}} = Z_0 \cdot \frac{Z_L + j Z_0 \tan\beta\ell}{Z_0 + j Z_L \tan\beta\ell}.$$

| Function | Description |
|---|---|
| `characteristicImpedanceLC(L', C')` | Z₀ = √(L'/C') for a lossless line |
| `reflectionCoefficient(zLoad, z0)` | Γ (complex) at the load |
| `vswr(gamma)` | VSWR from Γ |
| `returnLossDb(gamma)` | Return loss in dB |
| `inputImpedance(zLoad:, z0:, beta:, ell:)` | Impedance seen at distance ℓ from the load |
| `phaseConstant(frequencyHz:, velocity:)` | β = 2π·f/v |
| `distanceToVoltageMaximum(gamma, beta)` | Nearest voltage max from the load (m) |

Verified: matched load → Γ = 0 & VSWR = 1; open → |Γ| = 1;
short → Γ = −1; quarter-wave transformer inverts Z_L about Z₀;
half-wave line reproduces Z_L. See
[test/ee_transmission_line_test.dart](../test/ee_transmission_line_test.dart).

## Semiconductors

File: [lib/src/electronics/semiconductor.dart](../lib/src/electronics/semiconductor.dart).

$$V_t = \frac{kT}{q}, \qquad I_D = I_s\left(e^{V_d / (n V_t)} - 1\right), \qquad g_m = I_C / V_t.$$

| Function | Description |
|---|---|
| `kBoltzmann`, `qElementary` | Physical constants (SI) |
| `thermalVoltage(T)` | V_t at temperature T (kelvin) |
| `diodeCurrent(Vd, {Is, n, Vt})` | Shockley I–V |
| `diodeVoltage(I, {Is, n, Vt})` | Inverse — closed-form for I ≫ I_s, Newton for small |I| |
| `diodeLoadLine(vSupply:, rSeries:, …)` | Series-R + diode operating point via Newton |
| `bjtTransconductance(iC)` | g_m = I_C / V_t |
| `bjtInputResistance(beta:, gm:)` | r_π = β / g_m |
| `mosfetSaturationCurrent(vgs:, vth:, vds:, kn:, lambda:)` | Square-law saturation model with channel-length modulation |
| `mosfetTransconductance(vgs:, vth:, kn:)` | g_m in saturation |

Verified: V_t ≈ 25.85 mV at 300 K; diode round-trip `V(I(V)) = V`;
standard textbook operating point (5 V, 1 kΩ → ≈ 4.28 mA);
quadratic MOSFET scaling in overdrive; g_m identities. See
[test/ee_semiconductor_test.dart](../test/ee_semiconductor_test.dart).

## Worked examples

```dart
const R = 100.0;
const L = 0.1;                          // 100 mH
final omega = omegaFromHz(50);
final z = seriesImpedance([impedanceR(R), impedanceL(L, omega)]);
z.modulus;      // ≈ 105.0 Ω
z.argumentDeg;  // ≈ 17.44°  (current lags V by this much)
```

### 2. Cascade of two 1st-order LPFs

```dart
const w0 = 100.0;
final one = TransferFunction.fromCoefficients(
  numerator: [1],
  denominator: [1, 1 / w0],
);
final two = one * one;  // each stage 1/√2 at ω₀
two.frequencyResponse(w0).modulus;  // 0.5
```

### 3. THD of a signal with a 30% third-harmonic content

```dart
totalHarmonicDistortion([1.0, 0.0, 0.3], [0.0, 0.0, 0.0]); // 0.30
```

### 4. 4th-order Butterworth LPF, 1 kHz cutoff

```dart
final tf = butterworthLowPass(order: 4, omegaC: 2 * math.pi * 1000);
tf.frequencyResponse(2 * math.pi * 1000).modulus;  // 1/√2  (−3 dB)
tf.frequencyResponse(2 * math.pi * 10000).modulus; // ≈ 1e-4 (−80 dB/dec)
```

### 5. 100 Ω load on a 50 Ω line

```dart
final gamma = reflectionCoefficient(Complex.real(100), 50);
gamma.re;               // 1/3
vswr(gamma);            // 2
returnLossDb(gamma);    // ≈ 9.54 dB
```

### 6. Diode operating point at 5 V through 1 kΩ

```dart
final op = diodeLoadLine(vSupply: 5, rSeries: 1000);
op.vD;    // ≈ 0.72 V
op.iD;    // ≈ 4.28 mA
```

## Verified against

| Test file | Coverage |
|---|---|
| [test/ee_impedance_test.dart](../test/ee_impedance_test.dart) | Unit conversions, R/L/C impedances, series/parallel combinators, voltage divider at −3 dB |
| [test/ee_phasor_test.dart](../test/ee_phasor_test.dart) | Phasor factories, instantaneous value, Ohm's law, phase difference (wrapped) |
| [test/ee_power_test.dart](../test/ee_power_test.dart) | Real/reactive/apparent power, complex power V·conj(I), Parseval RMS, THD |
| [test/ee_bode_test.dart](../test/ee_bode_test.dart) | 1st-order LPF, cascade, cutoff detection, phase unwrap |
| [test/ee_filter_test.dart](../test/ee_filter_test.dart) | RC LPF/HPF, RLC BPF, Butterworth pole placement & roll-off orders 1–5 |
| [test/ee_transmission_line_test.dart](../test/ee_transmission_line_test.dart) | Γ, VSWR, return loss, quarter-wave transformer, half-wave transparency |
| [test/ee_semiconductor_test.dart](../test/ee_semiconductor_test.dart) | Thermal voltage, Shockley diode round-trip, load-line operating point, MOSFET quadratic, BJT g_m / rπ |

All 66 tests passing.

## Coming next

The library foundation is now complete for typical analog EE work.
Future extensions might add:

- Two-port network parameters (Z, Y, ABCD, S) and cascade helpers
- Op-amp closed-loop transfer functions (inverting, non-inverting,
  Sallen–Key)
- Chebyshev / elliptic filter design
- Small-signal BJT/MOSFET amplifier gain topologies (common-emitter,
  common-source, differential)
- Companion Flutter lessons in `dart_eng_math_app` for each of the
  seven modules above.


---

## FILE: topic-g-statistics.md

# Topic G — Statistics and probability

Bird reference: Ch. 55 (central tendency + dispersion),
Ch. 56 (probability laws), Ch. 57 (binomial + Poisson),
Ch. 58 (normal), Ch. 59 (linear correlation), Ch. 60 (linear regression).
Phase O extends this with inferential statistics:
Student's t distribution, chi-squared distribution, confidence intervals,
one- and two-sample tests, and chi-square goodness-of-fit / independence.

## Contents

- [Descriptive statistics](#descriptive-statistics)
- [Probability helpers](#probability-helpers)
- [Distributions](#distributions)
- [Linear correlation](#linear-correlation)
- [Linear regression](#linear-regression)

## Descriptive statistics

File: [lib/src/descriptive_stats.dart](../lib/src/descriptive_stats.dart).

| Function | Description |
|---|---|
| `mean(data)` | Arithmetic mean. |
| `median(data)` | Median; even-count → mean of the two middle values. |
| `modes(data)` | List of the most-frequent value(s); empty when every value is unique. |
| `variance(data, {population = true})` | Divide by n (population) or n−1 (sample). |
| `stdev(data, {population = true})` | √variance. |
| `meanGrouped(classes)`, `varianceGrouped`, `stdevGrouped` | For grouped-frequency data. |
| `quartiles(data)` | Q₁, Q₂ (= median), Q₃ using Tukey's hinges (matches Bird's convention). |
| `semiInterquartileRange(data)` | (Q₃ − Q₁) / 2. |

Grouped data is expressed as `List<GroupedClass>` where each class is a
record `(midpoint, frequency)`.

### Example

```dart
mean([2, 3, 7, 5, 5, 13, 1, 7, 4, 8, 3, 4, 3]);   // 5.0
median([27.90, 34.70, 54.40, 18.92, 47.60, 39.68]); // 37.19
modes([2, 3, 7, 5, 5, 13, 1, 7, 4, 8, 3, 4, 3]);   // [3]
quartiles([2, 3, 4, 5, 5, 7, 9, 11, 13, 14, 17]);
// (q1: 4, q2: 7, q3: 13)

// Grouped: 48 resistors from Bird Ch. 55 Problem 3.
final classes = <GroupedClass>[
  (midpoint: 20.7, frequency: 3),
  (midpoint: 21.2, frequency: 10),
  (midpoint: 21.7, frequency: 11),
  (midpoint: 22.2, frequency: 13),
  (midpoint: 22.7, frequency: 9),
  (midpoint: 23.2, frequency: 2),
];
meanGrouped(classes);   // 21.919
stdevGrouped(classes);  // 0.645
```

### Verified worked examples (Bird Ch. 55)

| Reference | Result |
|---|---|
| Problem 1 (discrete) | mean 5, median 4, mode 3 |
| Problem 2 (news vendor) | mean £37.20, median £37.19, no mode |
| Problem 3 (grouped, resistors) | mean 21.919 Ω |
| Problem 5 (σ, discrete) | σ ≈ 2.380 |
| Problem 6 (σ, grouped) | σ ≈ 0.645 |
| Preamble set §55.5 | Q₁ = 4, Q₂ = 7, Q₃ = 13 |
| Exercise 210-1 (even n) | Q₁ = 25.5, Q₂ = 30, Q₃ = 33.5 |

Tests: [test/descriptive_stats_test.dart](../test/descriptive_stats_test.dart).

## Probability helpers

File: [lib/src/probability.dart](../lib/src/probability.dart).

| Function | Description |
|---|---|
| `complement(p)` | 1 − p (probability of not happening). |
| `expectation(p, trials)` | E = p · n. |
| `independentAnd([p₁, p₂, …])` | Π pᵢ — all independent events happen. |
| `mutuallyExclusiveOr([p₁, p₂, …])` | Σ pᵢ — one of mutually exclusive events. |
| `atLeastOne([p₁, p₂, …])` | 1 − Π(1 − pᵢ) — union of independent events. |

### Verified worked examples (Bird Ch. 56)

| Reference | Result |
|---|---|
| Problem 2: expectation of 4 upwards in 3 dice throws | 1/2 |
| Problem 4(a): temp ∧ vibration failure | 1/500 |
| Problem 4(b): vibration ∨ humidity failure | 3/50 |
| Problem 4(c): no-temp ∧ no-humidity | 931/1000 |
| Problem 6(b): both good, no replacement (7/8·34/39) | ≈ 0.7628 |
| Problem 8: three steel washers (86/200·85/199·84/198) | ≈ 0.0779 |
| Problem 9: no aluminium in three draws | ≈ 0.5101 |

Tests: [test/probability_test.dart](../test/probability_test.dart).

## Distributions

File: [lib/src/distributions.dart](../lib/src/distributions.dart).

Interfaces:

```dart
abstract class DiscreteDistribution {
  double pmf(int k);
  double cdf(int k);
  double get mean;
  double get variance;
  double get stdev;
}

abstract class ContinuousDistribution {
  double pdf(num x);
  double cdf(num x);
  double get mean;
  double get variance;
  double get stdev;
}
```

Concrete types:

| Type | Formula | E[X] | Var(X) |
|---|---|---|---|
| `BinomialDistribution(n, p)` | `C(n, k) p^k q^{n-k}` | `n·p` | `n·p·q` |
| `PoissonDistribution(λ)` | `λ^k · e^{-λ} / k!` | λ | λ |
| `NormalDistribution(μ, σ)` | `1/(σ√2π) · exp(−(x−μ)²/(2σ²))` | μ | σ² |

Extras:

- `NormalDistribution.standard` — cached N(0, 1).
- `standardNormalPartialArea(z)` — the area between 0 and z on N(0, 1),
  i.e. `Φ(z) − 0.5`. This is the value Bird's Table 58.1 tabulates.
- Internal `erf` uses Abramowitz & Stegun 7.1.26 rational approximation
  (max absolute error ≤ 1.5 × 10⁻⁷).

### Example

```dart
// Bird Ch. 57 Problem 3: 7 bolts, p = 0.05, P(exactly 2 defective).
BinomialDistribution(7, 0.05).pmf(2); // ≈ 0.0406

// Bird Ch. 57 Problem 6: gearwheels, λ = 2.4, P(exactly 2 defective).
PoissonDistribution(2.4).pmf(2);      // ≈ 0.2613

// Bird Ch. 58 Problem 1: N(170, 9), P(150 ≤ h ≤ 195).
final heights = NormalDistribution(170, 9);
heights.cdf(195) - heights.cdf(150);  // ≈ 0.9841
```

### Verified worked examples

Bird Ch. 57 (Binomial + Poisson):

| Reference | Result |
|---|---|
| Problem 1: P(≥ 1 girl in 4 children) | 0.9375 |
| Problem 2(a): three 4s in 9 dice throws | ≈ 0.1302 |
| Problem 2(b): fewer than four 4s | ≈ 0.9520 |
| Problem 3(a): 2 defective bolts out of 7 (p = 0.05) | ≈ 0.0406 |
| Problem 3(b): more than 2 defective | ≈ 0.0038 |
| Problem 6: gearwheels λ = 2.4, P(k = 2) | ≈ 0.2613 |
| Problem 6: P(k > 2) | ≈ 0.4303 |
| Problem 7: milling λ = 2.1, P(k = 1) | ≈ 0.2572 |

Bird Ch. 58 (Normal):

| Reference | Result |
|---|---|
| Table 58.1 headline z-values (0.5, 1.0, 1.5, 2.0, 2.22, 2.67, 2.78, 3.0) | All within 1e-3 of table |
| Problem 1: N(170, 9), P(150 ≤ h ≤ 195) | ≈ 0.9841 |
| Problem 2: P(h < 165) | ≈ 0.2877 (table-rounded; precise 0.2893) |
| Problem 3: P(h > 194) → 500·p ≈ 2 people | ✓ |
| Problem 4(a): N(753, 1.8), P(v < 750) | ≈ 0.0475 |
| Problem 4(b): P(751 ≤ v ≤ 754) | ≈ 0.5788 |
| Exercise 215-1: N(75, 2.8), defect rate → ~6/350 | ✓ |

Tests: [test/distributions_test.dart](../test/distributions_test.dart).

## Linear correlation

File: [lib/src/correlation.dart](../lib/src/correlation.dart).

`pearsonCorrelation(xs, ys)` computes the Pearson product-moment
correlation coefficient

$$r = \frac{\sum x y}{\sqrt{\sum x^2 \cdot \sum y^2}}$$

where `x = X − X̄` and `y = Y − Ȳ`. Returns a value in [−1, 1].

### Verified worked examples (Bird Ch. 59)

| Reference | Result |
|---|---|
| Problem 1: force vs extension | r ≈ 0.996 |
| Problem 2: welfare spend vs days lost | r ≈ −0.830 |
| Problem 3: cars vs petrol income | r ≈ 0.667 |
| Exercise 217-1 | r ≈ 0.999 |
| Exercise 217-5: pressure vs volume | r ≈ −0.962 |

Tests: [test/correlation_regression_test.dart](../test/correlation_regression_test.dart).

## Linear regression

File: [lib/src/regression.dart](../lib/src/regression.dart).

Least-squares fit `Y = a₀ + a₁ X`. Bird Ch. 60 uses the normal equations

$$\sum Y = a_0 N + a_1 \sum X, \qquad \sum XY = a_0 \sum X + a_1 \sum X^2$$

closed-form solved to

$$a_1 = \frac{N \sum XY - \sum X \sum Y}{N \sum X^2 - (\sum X)^2}, \qquad a_0 = \frac{\sum Y - a_1 \sum X}{N}.$$

| Function | Description |
|---|---|
| `LinearRegression` | `intercept` (a₀), `slope` (a₁), `predict(x)`. |
| `regressionYonX(xs, ys)` | Fit Y on X — use to predict Y from X. |
| `regressionXonY(xs, ys)` | Fit X on Y — use to predict X from Y. |

### Example

```dart
// Bird Ch. 60 Problem 1: frequency vs inductive reactance.
final freq  = <num>[50, 100, 150, 200, 250, 300, 350];
final xL    = <num>[30,  65,  90, 130, 150, 190, 200];
final fit = regressionYonX(freq, xL);
fit.predict(175);  // ≈ 107.5 Ω
```

### Verified worked examples (Bird Ch. 60)

| Reference | Result |
|---|---|
| Problem 1: Y on X | `Y = 4.94 + 0.586·X` (Bird 3-s.f.; precise intercept ≈ 5.0) |
| Problem 2: X on Y | `X = −6.15 + 1.69·Y` |
| Problem 3(a): predict inductance at 175 Hz | ≈ 107.5 Ω |
| Problem 3(b): predict frequency at 250 Ω | ≈ 416.4 Hz |
| Problem 4: force vs radius | `Y = 33.7 − 0.617·X` |
| Problem 4: predict force at r = 40 | ≈ 9.02 N |
| Problem 4: predict radius at F = 32 | ≈ 7.08 cm |
| Exercise 218-1 | `Y = −256 + 80.6·X` |

Tests: [test/correlation_regression_test.dart](../test/correlation_regression_test.dart).

## Notes and limitations

- **Tukey's-hinges quartiles** (`quartiles(...)`) match Bird's discrete
  examples exactly. Bird's grouped-data quartiles use the ogive (linear
  interpolation of the cumulative-frequency curve); a
  `quartilesGrouped(...)` for that will be added if needed.
- **Bird uses table-lookup rounding** for the normal distribution.
  Our tests accept ~2×10⁻³ deviation on those cases because the
  double-precision values are more accurate than the table-rounded
  answers Bird prints.
- **`erf` accuracy**: A&S 7.1.26 rational approx, sufficient for the
  4-dp Table 58.1 and downstream Bird problems. A higher-order variant
  can be swapped in later if required.

## Phase O — Inferential statistics

Files: [lib/src/special_functions.dart](../lib/src/special_functions.dart),
[lib/src/inference.dart](../lib/src/inference.dart), plus the
`StudentTDistribution` and `ChiSquaredDistribution` classes in
[lib/src/distributions.dart](../lib/src/distributions.dart).

### New distributions

| Class | Purpose |
|---|---|
| `NormalDistribution.quantile(p)` | Inverse CDF via Acklam's approximation (accuracy ≈ 1.15 × 10⁻⁹). |
| `StudentTDistribution(df)` | pdf / cdf via regularised incomplete beta, quantile via bisection. |
| `ChiSquaredDistribution(df)` | pdf / cdf via regularised incomplete gamma, quantile via bisection. |
| `FDistribution(df1, df2)` | pdf / cdf via regularised incomplete beta, quantile via bisection; used by ANOVA. |

### Bayes and conditional probability

```dart
conditional(0.2, 0.5);      // P(A | B) = 0.4
bayes([0.01, 0.99], [0.99, 0.05]);   // → [0.167, 0.833]
```

### Confidence intervals

```dart
ciMeanZ(sampleMean: 20, sigma: 4, n: 60);          // known σ
ciMeanT(sampleMean: 10, sampleStd: 2, n: 25);      // unknown σ, t-based
ciProportion(pHat: 0.4, n: 200);                    // Wald 95 % CI
```

All three return a `ConfidenceInterval(lower, upper, confidence)` with
`width` and `halfWidth` getters.

### Hypothesis tests

| Function | H₀ |
|---|---|
| `oneSampleZ({sample, mu0, sigma})` | μ = μ₀ (σ known) |
| `oneSampleT({sample, mu0})` | μ = μ₀ (σ unknown) |
| `twoSampleT({a, b, pooled = true})` | μ_a = μ_b (Welch when `pooled: false`) |
| `pairedT({before, after})` | mean difference = 0 |
| `oneWayAnova(groups)` | μ₁ = μ₂ = … = μ_k across k independent groups (F-test) |
| `chiSquareGoodnessOfFit({observed, expected})` | observed matches expected |
| `chiSquareIndependence(table)` | rows and columns are independent |
| `signTest({before, after})` | median difference = 0 (paired, exact binomial null) |
| `wilcoxonSignedRank({before, after})` | median difference = 0 (paired, rank-based) |
| `mannWhitneyU({a, b})` | two samples come from the same distribution (rank-based) |

Every test returns a `HypothesisTestResult` with `statistic`, `df`,
`df2` (populated for F-tests), `pValue` (two-tailed), and
`significantAt(alpha)`.

```dart
final r = oneSampleT(sample: [98, 100, 102, 99, 101], mu0: 100);
r.significantAt(0.05);   // false — sample is consistent with μ = 100

final anova = oneWayAnova([
  [6, 8, 4, 5, 3, 4],       // fertiliser A
  [8, 12, 9, 11, 6, 8],     // fertiliser B
  [13, 9, 11, 8, 7, 12],    // fertiliser C
]);
// F ≈ 9.27, df = (2, 15), p ≈ 0.0025 → treatment matters.

final u = mannWhitneyU(a: [12, 15, 14], b: [22, 24, 23]);
// Rank-based; robust to non-normal samples.
```

Non-parametric tests use midrank tie-handling; the p-values for
`wilcoxonSignedRank` and `mannWhitneyU` come from the normal
approximation with tie correction, which is accurate once
`min(n₁, n₂) ≥ ~8`. For smaller samples the approximation is mildly
conservative.

### Special functions

Exposed as building blocks for users who need them directly:

- `logGamma(x)` — Lanczos approximation.
- `regularisedIncompleteGammaP(a, x)` — for custom chi²-family work.
- `regularisedIncompleteBeta(x, a, b)` — for custom beta-family work.

Tests: [test/inference_test.dart](../test/inference_test.dart) — Bird
table critical values (t, χ² and F) verified; ANOVA hand-checked
against a canonical three-fertiliser worked example (F ≈ 9.27,
p ≈ 0.0025).

## Phase R — Principal Component Analysis

File: [lib/src/pca.dart](../lib/src/pca.dart).

Standard engineering-lab dimensionality reduction. Mean-centres the
data, forms the unbiased sample covariance `(n − 1)⁻¹ · XᵀX`, and
diagonalises it via
[`SymmetricEigenDecomposition`](topic-5-linear-algebra.md#general-eigenvalue-decomposition-phase-r).

```dart
final samples = [
  [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0],
  [2.0, 4.0, 8.0], [3.0, 6.0, 12.0],
];
final pca = PrincipalComponentAnalysis.fit(samples);

pca.mean;                      // per-feature means
pca.eigenvalues;               // variance captured by each component (descending)
pca.explainedVarianceRatio;    // sums to 1
pca.components;                // rows = principal directions

// "How many components do I need for 95 % of the variance?"
final k = pca.componentsFor(0.95);

// Project to k dims and reconstruct.
final projected = pca.transform(samples, k: k);
final reconstructed = pca.inverseTransform(projected);
```

`transform` + `inverseTransform` with all components round-trips to
1e-9. `componentsFor(target)` finds the smallest `k` whose cumulative
explained variance first exceeds `target`.

Tests: [test/eigen_pca_test.dart](../test/eigen_pca_test.dart).


---

## FILE: topic-h-fourier.md

# Topic H — Fourier series & transforms

Bird reference: Ch. 66 (period-2π), Ch. 67 (non-periodic over 2π),
Ch. 68 (even/odd + half-range), Ch. 69 (any range), Ch. 70 (numerical
harmonic analysis), Ch. 71 (complex/exponential form). Continuous
Fourier transform + DFT/FFT extend the syllabus to non-periodic
signals and sampled data.

## Contents

- [Real Fourier series](#real-fourier-series)
- [Numerical harmonic analysis](#numerical-harmonic-analysis)
- [Complex Fourier series](#complex-fourier-series)
- [Continuous Fourier transform](#continuous-fourier-transform)
- [Discrete Fourier transform (DFT) and FFT](#discrete-fourier-transform-dft-and-fft)

## Real Fourier series

File: [lib/src/fourier_series.dart](../lib/src/fourier_series.dart).

For a period-L function f,

$$f(x) \approx \tfrac{a_0}{2} + \sum_{k=1}^{N}\bigl[a_k \cos(k\omega x) + b_k \sin(k\omega x)\bigr], \qquad \omega = \tfrac{2\pi}{L},$$

with

$$a_0 = \tfrac{2}{L}\!\int_{s}^{s+L}\!\!f(x)\,dx, \quad a_k = \tfrac{2}{L}\!\int f(x)\cos(k\omega x)\,dx, \quad b_k = \tfrac{2}{L}\!\int f(x)\sin(k\omega x)\,dx.$$

### `FourierSeries` API

| Member | Description |
|---|---|
| `.period` | The period L. |
| `.omega` | Fundamental frequency `2π/L`. |
| `.a0` | The `a₀` constant; the DC term in the sum is `a₀/2`. |
| `.a`, `.b` | Lists of harmonic coefficients: `a[k-1] = aₖ`, `b[k-1] = bₖ`. |
| `.terms` | Number of non-DC harmonics stored. |
| `.eval(x)` | Partial-sum evaluation. |

### Constructors

| Constructor | Purpose |
|---|---|
| `FourierSeries(period, a0, a, b)` | Direct construction from known coefficients. |
| `FourierSeries.fromFunction(f, period, {start, terms, samplesPerPeriod})` | Numerically compute coefficients of `f` via Simpson's rule over `[start, start + period]`. |
| `FourierSeries.fromSamples(samples, period, {terms})` | Ch. 70 numerical harmonic analysis from equally-spaced samples. |

### Example — sawtooth on [−π, π]

Analytic: `bₙ = 2·(−1)^{n+1}/n`, `aₙ = 0`, `a₀ = 0`.

```dart
final fs = FourierSeries.fromFunction(
  (x) => x,
  period: 2 * math.pi,
  start: -math.pi,
  terms: 6,
);

fs.a0;      // ≈ 0
fs.a[0];    // ≈ 0    (a₁)
fs.b[0];    // ≈ 2    (b₁ = 2)
fs.b[1];    // ≈ -1   (b₂ = -1)
fs.eval(1); // ≈ 1 for enough terms, away from x = ±π
```

### Verified worked examples

| Signal | Verified against analytic |
|---|---|
| Constant `f(x) = 5` | `a₀ = 10`, all other coefficients 0 |
| Sawtooth `f(x) = x` on `[−π, π]` | `bₙ = 2·(−1)^{n+1}/n`, all aₙ ≈ 0 |
| Triangle `f(x) = |x|` on `[−π, π]` | `a₀ = π`, `a_odd = −4/(π n²)`, `a_even = 0`, all bₙ ≈ 0 |
| Parabola `f(x) = x²` on `[−π, π]` | `a₀ = 2π²/3`, `aₙ = 4·(−1)^n/n²`, all bₙ ≈ 0 |
| `cos(2π x / L)` on `[0, L]` (L = 4) | `a₁ = 1`, all other coefficients 0 |
| 12-term triangle reconstruction | within 0.05 of `|x|` in interior |
| 20-term sawtooth reconstruction | within 0.1 of `x` for `|x| ≤ 1` |
| 20-term parabola reconstruction | within 0.02 of `x²` |

Tests: [test/fourier_test.dart](../test/fourier_test.dart).

## Numerical harmonic analysis

Bird Ch. 70: given `K` equally-spaced samples of one period,

$$a_n \approx \tfrac{2}{K}\sum_{k=0}^{K-1} y_k \cos(2\pi n k / K), \qquad b_n \approx \tfrac{2}{K}\sum_{k=0}^{K-1} y_k \sin(2\pi n k / K).$$

`FourierSeries.fromSamples(samples, period: L, terms: N)` computes these
directly. The sample at `x = start + L` is not included (equals
`samples[0]` by periodicity). `N` is capped by the Nyquist limit
`K / 2`.

### Verified

| Signal | Result |
|---|---|
| `cos(2π k / 8)` at K = 8 samples | `a₁ = 1` exactly, all others 0 |
| `sin(2π · 2 k / 16)` at K = 16 samples | `b₂ = 1` exactly, all others 0 |
| Too few samples | `ArgumentError` |
| Requesting `terms > K/2` | `ArgumentError` (Nyquist limit) |

## Complex Fourier series

File: [lib/src/fourier_complex.dart](../lib/src/fourier_complex.dart).

Bird Ch. 71 exponential form:

$$f(x) = \sum_{n=-\infty}^{\infty} c_n\,e^{jn\omega x}, \qquad c_n = \tfrac{1}{L}\!\int_{s}^{s+L} f(x)\,e^{-jn\omega x}\,dx.$$

Real-form relations:

- `c₀ = a₀ / 2`
- `cₙ = (aₙ − j·bₙ) / 2` for n ≥ 1
- `c₋ₙ = conjugate(cₙ)` (reality of f)

### `ComplexFourierSeries` API

| Member | Description |
|---|---|
| `.c0` | c₀ as `Complex`. |
| `.cPositive` | List of positive-index cₙ; `cPositive[k-1] = cₖ`. |
| `.c(n)` | Coefficient for any integer n; c₋ₙ auto-conjugated. |
| `.evalComplex(x)` | Full complex partial sum. |
| `.eval(x)` | Real part (for real signals). |

### Constructors

| Constructor | Purpose |
|---|---|
| `ComplexFourierSeries(period, c0, cPositive)` | Direct. |
| `ComplexFourierSeries.fromReal(fs)` | Convert an existing `FourierSeries`. |
| `ComplexFourierSeries.fromFunction(f, period, {start, terms, samplesPerPeriod})` | Direct Simpson integration of the complex integrand. |

### Verified

| Case | Result |
|---|---|
| `c₀ = a₀/2`, `cₙ = (aₙ − j bₙ)/2` (sawtooth) | ✓ |
| Reality condition `c₋ₙ = conjugate(cₙ)` | ✓ |
| `fromReal` and `fromFunction` agree on `x²` | Within 1e-3 |
| Partial-sum recovers `x²` (Re within 0.02, Im within 1e-9) | ✓ |

## Continuous Fourier transform

File: [lib/src/fourier_transform.dart](../lib/src/fourier_transform.dart).

For a non-periodic signal `f(t)`, the (engineering-convention)
Fourier transform pair is

$$F(\omega) = \int_{-\infty}^{\infty} f(t)\, e^{-j\omega t}\, dt, \qquad f(t) = \tfrac{1}{2\pi}\int_{-\infty}^{\infty} F(\omega)\, e^{+j\omega t}\, d\omega.$$

Numerical evaluation uses Simpson's rule over a caller-supplied
truncation window `[tMin, tMax]` (for the forward transform) or
`[omegaMin, omegaMax]` (for the inverse). The window must be wide
enough that the integrand is effectively zero at the endpoints.

### API

| Function | Description |
|---|---|
| `fourierTransform(f, ω, {tMin, tMax, samples})` | `F(ω)` at a single frequency; returns `Complex`. |
| `fourierTransformSpectrum(f, omegas, {tMin, tMax, samples})` | List of `F(ω)` values across many frequencies. |
| `inverseFourierTransform(F, t, {omegaMin, omegaMax, samples})` | `f(t)` at a single time; returns `Complex`. |

### Verified transform pairs

| Signal `f(t)` | `F(ω)` (analytic) | Verified |
|---|---|---|
| `rect(t/T)` (T = 2) — pulse | `2 sin ω / ω` (sinc) | Re within 5e-3, Im ≈ 0 |
| `e^{-a t²}` Gaussian (a = 1) | `√(π/a)·e^{-ω²/4a}` | Within 1e-4 |
| `u(t)·e^{-a t}` (a = 1.5) | `1 / (a + jω)` | Within 1e-3 |
| Inverse of `2 sinω/ω` reconstructs `rect` at t = 0, 0.5 | ✓ (5% near discontinuity, Gibbs) |

Tests: [test/fourier_transform_test.dart](../test/fourier_transform_test.dart).

## Discrete Fourier transform (DFT) and FFT

File: [lib/src/dft.dart](../lib/src/dft.dart).

For a length-`N` sample sequence,

$$X[k] = \sum_{n=0}^{N-1} x[n]\, e^{-j 2\pi k n / N}, \qquad x[n] = \tfrac{1}{N}\sum_{k=0}^{N-1} X[k]\, e^{+j 2\pi k n / N}.$$

Bin `k` corresponds to physical frequency `k · f_s / N` Hz (for
`k ≤ N/2`), where `f_s` is the sample rate. Bins above `N/2` are
negative (aliased) frequencies.

### API

| Function | Complexity | Description |
|---|---|---|
| `dft(x)` / `idft(X)` | O(N²) | Direct sum for arbitrary N. |
| `dftReal(x)` | O(N²) | DFT of a real-valued list. |
| `fft(x)` / `ifft(X)` | O(N log N) | Radix-2 iterative Cooley–Tukey; length must be a power of 2. |
| `fftReal(x)` | O(N log N) | FFT of a real-valued list. |
| `magnitudeSpectrum(X)` | O(N) | `|X[k]|`. |
| `phaseSpectrum(X)` | O(N) | `arg X[k]` in radians. |
| `oneSidedAmplitudeSpectrum(X)` | O(N) | Positive-frequency amplitudes, scaled so peaks equal sinusoid amplitude. |
| `fftFrequencies(N, sampleRate)` | O(N) | Frequency (Hz) of each bin, with negative frequencies for bins > N/2. |

### Verified identities

| Case | Result |
|---|---|
| `dft([1,0,0,0])` | `[1,1,1,1]` (impulse → flat) |
| `dft([3,3,3,3])` | `[12,0,0,0]` (constant → DC only) |
| `dft(cos(2π·2n/8))` | magnitude N/2 at bins 2 and N−2, 0 elsewhere |
| `idft(dft(x)) = x` (length 6) | Within 1e-10 |
| `fft` vs `dft` for random length-16 complex signal | Within 1e-10 per bin |
| `ifft(fft(x)) = x` (length 32) | Within 1e-10 |
| Non-power-of-two → `fft` throws | ✓ |
| One-sided amplitude spectrum of `2·cos(2π·4t)` sampled at 32 Hz | Peak of 2.0 at bin 4 |
| `fftFrequencies(8, 8)` | `[0,1,2,3,4,-3,-2,-1]` |

Tests: [test/fourier_transform_test.dart](../test/fourier_transform_test.dart).

## Notes and limitations

- **Gibbs phenomenon**: partial sums overshoot ~9% at each discontinuity
  (e.g. the sawtooth at `x = ±π`). Tests focus on continuous interior
  points, where convergence is uniform-ish.
- **Simpson integration** at 1024 subintervals gives Fourier coefficients
  accurate to ~1e-3 for typical Bird signals. The
  `samplesPerPeriod` parameter is tunable.
- **`fromSamples`** enforces the Nyquist limit (`terms ≤ K/2`). Above
  that, the DFT aliases and results are meaningless.
- **`ComplexFourierSeries.c(n)`** throws `RangeError` outside the stored
  positive-harmonics window `[-terms, +terms]`.
- **Half-range series** (Bird Ch. 68) are not a separate module — build
  the appropriate even/odd extension by hand and call
  `FourierSeries.fromFunction`. If this pattern comes up frequently a
  convenience constructor can be added.
- **Continuous transform truncation**: `fourierTransform` /
  `inverseFourierTransform` integrate over a finite window. Compact-
  support signals (rectangular pulse) are exact up to Simpson error;
  decaying signals (Gaussian, one-sided exponential) need the window
  wide enough that the tails are negligible.
- **DFT vs FFT**: `dft` accepts any N (including primes) at O(N²) cost;
  `fft` is O(N log N) but requires N a power of 2. Non-power-of-two
  sizes should either be zero-padded up to a power of 2 or routed
  through `dft`.
- **One-sided vs two-sided spectra**: `magnitudeSpectrum` returns the
  full length-N two-sided spectrum. For real signals prefer
  `oneSidedAmplitudeSpectrum`, which reads bin heights directly as
  sinusoid amplitudes.


---

## FILE: topic-i-ode-methods.md

# Topic I — ODE methods

Bird reference: Ch. 46 (separation of variables), Ch. 47 (homogeneous),
Ch. 48 (linear + integrating factor), Ch. 49 (Euler, improved Euler,
Runge–Kutta), Ch. 52 (power series — deferred), Ch. 53 (introduction
to PDEs — heat/wave/Laplace), Ch. 65 (simultaneous ODE via Laplace).

## Contents

- [Numerical solvers (scalar)](#numerical-solvers-scalar)
- [Numerical solvers (system)](#numerical-solvers-system)
- [2×2 simultaneous ODE via Laplace](#22-simultaneous-ode-via-laplace)
- [Partial differential equations (Phase T)](#partial-differential-equations-phase-t)

## Numerical solvers (scalar)

File: [lib/src/ode_numerical.dart](../lib/src/ode_numerical.dart).

All three solve `dy/dx = f(x, y)`, `y(x₀) = y₀`, and return the
trajectory as `List<({double x, double y})>` with `steps + 1` entries
(the initial point plus one per step).

| Function | Order | Formula |
|---|---|---|
| `solveEuler(f, {x0, y0, xEnd, steps})` | O(h) | `y_{n+1} = y_n + h·f(x_n, y_n)` |
| `solveImprovedEuler(...)` (Heun) | O(h²) | `y_{n+1} = y_n + (h/2)·[f(x_n, y_n) + f(x_n + h, y_n + h·f(x_n, y_n))]` |
| `solveRk4(...)` | O(h⁴) | Classical RK4 |

### Example — Bird Ch. 49 style

`dy/dx = 3(1 + x) − y`, `y(1) = 4`, find `y(1.5)`.

Analytical solution: `y(x) = 3x + e^{1−x}`, so `y(1.5) = 4.5 + e^{−0.5} ≈ 5.1065`.

```dart
final traj = solveRk4(
  (x, y) => 3 * (1 + x) - y,
  x0: 1, y0: 4, xEnd: 1.5, steps: 50,
);
traj.last.y; // ≈ 5.10653066 (< 1e-8 error)
```

### Accuracy comparison at equal step counts (100 steps, xEnd = 1)

Test problem `dy/dx = y`, `y(0) = 1`, exact answer `y(1) = e ≈ 2.71828`.

| Method | Final y | Absolute error |
|---|---|---|
| `solveEuler` | ≈ 2.7048 | ~1.35 × 10⁻² |
| `solveImprovedEuler` | ≈ 2.7181 | ~2 × 10⁻³ |
| `solveRk4` | ≈ 2.71828 | ≲ 1 × 10⁻⁷ |

Tests: [test/ode_numerical_test.dart](../test/ode_numerical_test.dart).

## Numerical solvers (system)

`solveRk4System(f, {x0, y0, xEnd, steps})` handles

$$\frac{d\mathbf{y}}{dx} = f(x, \mathbf{y}), \qquad \mathbf{y}(x_0) = \mathbf{y}_0$$

where `y` is a `List<double>` and `f` returns a `List<double>` of the
same length. Trajectory type is
`List<({double x, List<double> y})>`.

### Example — harmonic oscillator

`dx₁/dt = x₂`, `dx₂/dt = −x₁`, `x(0) = (1, 0)` → `x₁(t) = cos t`,
`x₂(t) = −sin t`.

```dart
final traj = solveRk4System(
  (t, y) => [y[1], -y[0]],
  x0: 0, y0: [1, 0], xEnd: 2 * math.pi, steps: 2000,
);
traj.last.y; // ≈ [1, 0] within 1e-6 after one full period
```

### Verified

- **One period returns to initial condition** within 1e-6.
- **Energy conservation** `x₁² + x₂²` drifts less than 1e-4 over 5
  periods with 5000 steps.
- **`t = π/2`**: `(cos π/2, −sin π/2) = (0, −1)` within 1e-6.

## 2×2 simultaneous ODE via Laplace

File: [lib/src/laplace_ode_system.dart](../lib/src/laplace_ode_system.dart).

Solves

$$\frac{d\mathbf{x}}{dt} = A\mathbf{x} + f(t), \qquad \mathbf{x}(0) = \mathbf{x}_0$$

for `A ∈ ℝ^{2×2}` via Laplace, following Bird's Ch. 65 procedure.
Applies

$$(sI - A) X(s) = \mathbf{x}_0 + F(s),$$

then inverts

$$(sI - A)^{-1} = \frac{1}{\det(sI - A)}\begin{pmatrix} s - a_{22} & a_{12} \\ a_{21} & s - a_{11} \end{pmatrix}.$$

Each component of `X(s)` is inverted via [inverseLaplace](topic-6-laplace.md#inverse-laplace-transform), so the
caller supplies the factorization of the combined denominator
`det(sI − A) × (product of forcing denominators)`.

### API

```dart
({TimeExpr x, TimeExpr y}) solveLinear2x2OdeLaplace({
  required Matrix a,               // 2×2
  required List<num> initial,      // [x(0), y(0)]
  TimeExpr? forcingX,              // f₁(t); default zero
  TimeExpr? forcingY,              // f₂(t); default zero
  required List<Factor> denominatorFactors,
});
```

### Example — decoupled forced system

`dx/dt = −x + 1`, `dy/dt = −y`, `x(0) = 0`, `y(0) = 1` → `x(t) = 1 − e^{-t}`,
`y(t) = e^{-t}`.

```dart
final r = solveLinear2x2OdeLaplace(
  a: Matrix.fromRows([[-1, 0], [0, -1]]),
  initial: [0, 1],
  forcingX: TimeExpr.constant(1),
  // det(sI − A) = (s + 1)², forcing denominator = s.
  denominatorFactors: [
    LinearFactor(0),
    LinearFactor(-1, multiplicity: 2),
  ],
);
r.x.eval(0.5); // ≈ 1 − e^{-0.5} ≈ 0.3935
r.y.eval(0.5); // ≈ e^{-0.5} ≈ 0.6065
```

### Verified

| Case | Result |
|---|---|
| Decoupled forced (above) | `1 − e^{-t}, e^{-t}` within 1e-9 |
| Coupled homogeneous: `A = [[0, 1], [−1, 0]]` (rotation) | `cos t, −sin t` within 1e-9 |
| Coupled real-eigenvalue: `A = [[3, 1], [0, 2]]`, `x(0) = (1, 0)` | pure `e^{3t}` mode within 1e-9 |
| Non-2×2 `A` rejected | ArgumentError |
| Wrong-length `initial` | ArgumentError |

Tests: [test/ode_numerical_test.dart](../test/ode_numerical_test.dart) (last group).

## Notes and limitations

- **Ch. 46–48 symbolic methods** (separation of variables, homogeneous
  substitution, integrating factor) are not implemented as separate
  modules. Each of them yields an analytic solution the caller can
  verify with the numerical solvers here or with the Laplace-based
  solver for linear ODEs (Phase F).
- **Ch. 52 power series** (Frobenius) is out of scope for this phase —
  it needs a symbolic Expr layer.
- **`solveLinear2x2OdeLaplace`** requires the caller to hand-factor
  the combined s-domain denominator, consistent with Bird's
  pen-and-paper workflow and with `solveOdeLaplace` from Phase F.
- **Stiff systems** need implicit methods (backward Euler, BDF, Rosenbrock).
  Not needed for Bird's problems. Add if a use case appears.
- **Adaptive step size**: no built-in adaptive control. Fine for
  Bird's exercises; a Dormand–Prince or Cash–Karp variant can be added
  later.

## Partial differential equations (Phase T)

File: [lib/src/pde.dart](../lib/src/pde.dart).

Finite-difference solvers for the three canonical linear PDEs from
Bird Ch. 53. Each takes uniform grids and Dirichlet boundary
conditions; each returns a solution container with grid arrays and a
bilinear `eval(...)` for off-grid queries.

### Heat equation — `solveHeatEquation`

$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2},\qquad u(0, t) = u_L,\; u(L, t) = u_R,\; u(x, 0) = u_0(x).$$

Default scheme is **Crank–Nicolson** — unconditionally stable and
second-order accurate in both time and space. Each timestep solves a
tridiagonal system via Thomas's algorithm. Set `useCrankNicolson: false`
for the classic explicit **FTCS** scheme; it will throw
`ArgumentError` if `r = α·Δt/Δx² > 0.5` (violating the FTCS stability
condition).

```dart
final sol = solveHeatEquation(
  alpha: 0.5,
  length: 1.0,
  duration: 0.2,
  spatialSteps: 40,
  timeSteps: 200,
  initialCondition: (x) => math.sin(math.pi * x),
  leftBoundary: 0,
  rightBoundary: 0,
);
sol.at(0.1);            // snapshot near t = 0.1
sol.eval(0.4, 0.15);    // bilinearly interpolated value
```

Verified: `u(x, 0) = sin(πx/L)` on `[0, L]` with zero-boundary decays
as `exp(−α π² t / L²) · sin(πx/L)`; the solver matches this analytic
solution to 1e-3 at t = 0.2.

### Wave equation — `solveWaveEquation`

$$\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}.$$

Explicit **leapfrog** scheme. Requires the CFL condition
`r = c · Δt / Δx ≤ 1` — throws otherwise. Second time step is
bootstrapped from a Taylor expansion so both `u(x, 0)` and
`∂u/∂t(x, 0)` are honoured to second order.

```dart
final sol = solveWaveEquation(
  waveSpeed: 1,
  length: 1,
  duration: 0.5,
  spatialSteps: 40,
  timeSteps: 200,
  initialDisplacement: (x) => math.sin(math.pi * x),   // plucked string
  initialVelocity: (x) => 0,                            // released from rest
);
```

Verified: plucked-string case matches `cos(πct/L) · sin(πx/L)` to 5e-3
over a full half-period; peak displacement stays bounded across five
periods of simulation.

### Laplace equation — `solveLaplaceEquation`

$$\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0 \text{ on } [0, W] \times [0, H].$$

**SOR** (successive over-relaxation) on a 5-point stencil. The
`omega` relaxation parameter defaults to the theoretical optimum
`2 / (1 + sin(π·h_min))` for a rectangular grid.

```dart
final sol = solveLaplaceEquation(
  width: 1, height: 1,
  nx: 40, ny: 40,
  boundary: (x, y) => y >= 1 - 1e-12 ? math.sin(math.pi * x) : 0,
  tolerance: 1e-8,
);
sol.converged;
sol.iterations;
sol.eval(0.5, 0.5);
```

Verified: with `u(x, 1) = sin(πx)` and zero on the other three sides,
the interior matches the analytic separation-of-variables solution
`sin(πx) · sinh(πy) / sinh(π)` to 1e-3 on a 40×40 grid. Constant BC
gives a constant interior; linear BC (`u = x`) gives an exactly linear
interior — harmonic.

Tests: [test/pde_test.dart](../test/pde_test.dart).


---

## FILE: topic-j-vectors.md

# Topic J — Vectors and partial derivatives

Bird reference: Ch. 24 (vectors), Ch. 26 (scalar + vector products),
Ch. 34 (partial derivatives), Ch. 35 (total differential),
Ch. 36 (max/min/saddle for f(x, y)).

## Contents

- [`Vector` value type](#vector-value-type)
- [Partial derivatives + gradient + Hessian](#partial-derivatives--gradient--hessian)
- [Total differential](#total-differential)
- [Critical-point classification (2-D)](#critical-point-classification-2-d)

## `Vector` value type

File: [lib/src/vector.dart](../lib/src/vector.dart).

Immutable n-dimensional real vector with 3-D convenience accessors.

| Member | Description |
|---|---|
| `Vector([...])` | General constructor from a `List<num>`. |
| `Vector.xyz(x, y, z)` | Convenience 3-D constructor (i, j, k). |
| `Vector.zero(dim)` | All-zero vector of the given dimension. |
| `.dimension`, `.components` | Introspection. |
| `.x`, `.y`, `.z` | 3-D accessors — throw for n ≠ 3. |
| `+ − −(unary)`, `.scale(k)` | Arithmetic (`+` and `−` require matching dimension). |
| `.dot(other)` | `Σ aᵢ bᵢ`. |
| `.cross(other)` | `a × b` — 3-D only, throws otherwise. |
| `.magnitude` / `.norm` | Euclidean length. |
| `.normalized()` | Unit vector in the same direction; throws on zero. |
| `.angleTo(other)` | Angle in radians `∈ [0, π]`. |

### Example

```dart
final a = Vector.xyz(2, 3, -1);
final b = Vector.xyz(1, -4, 2);

a.dot(b);      // -12
a.cross(b);    // Vector(2, -5, -11)
a.magnitude;   // √14 ≈ 3.742
a.angleTo(b);  // ≈ 2.014 rad
```

### Verified worked examples (Bird Ch. 24 + 26)

| Reference | Result |
|---|---|
| Dot product `a·b` for `a = 2i + 3j − k`, `b = i − 4j + 2k` | `−12` |
| Cross product for the same vectors | `2i − 5j − 11k` |
| Anticommutativity: `a × b = −(b × a)` | ✓ |
| `i × j = k`, `j × k = i`, `k × i = j` | ✓ |
| `(3, 4, 12).magnitude = 13` | ✓ |
| `‖(2, 3, −1).normalized()‖ = 1` | ✓ |
| `(1, 0, 0).angleTo((0, 1, 0)) = π/2` | ✓ |
| `(1, 1, 0).angleTo((1, 0, 0)) = π/4` | ✓ |

Tests: [test/vector_test.dart](../test/vector_test.dart).

## Partial derivatives + gradient + Hessian

File: [lib/src/partial_derivative.dart](../lib/src/partial_derivative.dart).

All numeric via central-difference formulae.

| Function | Description |
|---|---|
| `partialDerivative(f, point, i, {h})` | `∂f/∂xᵢ` (O(h²)). |
| `gradient(f, point, {h})` | Vector of first partials. |
| `secondPartialDerivative(f, point, i, j, {h})` | `∂²f/(∂xᵢ ∂xⱼ)` via 3-point or 4-point difference. |
| `hessian(f, point, {h})` | Full Hessian as a [`Matrix`](topic-5-linear-algebra.md#matrix-value-type). |
| `totalDifferential(f, point, delta, {h})` | `df ≈ Σ (∂f/∂xᵢ) δxᵢ` (Ch. 35). |

Scalar field convention: `double Function(List<double> point)`.

### Example — Bird Ch. 34

```dart
double f(List<double> p) {
  final x = p[0], y = p[1];
  return 5 * math.pow(x, 4) + 2 * math.pow(x, 3) * y * y - 3 * y;
}

partialDerivative(f, [1, 2], 0); // ∂z/∂x = 20·1 + 6·1·4 = 44
partialDerivative(f, [1, 2], 1); // ∂z/∂y = 4·1·2 − 3 = 5
```

### Verified

| Reference | Result |
|---|---|
| `z = 5x⁴ + 2x³y² − 3y` at (1, 2): partials | 44, 5 |
| Gradient of `x²y + yz` at (1, 2, 3) | (4, 4, 2) |
| Second partials of `x³y² + xy` at (1, 2) | fₓₓ = 24, fᵧᵧ = 2 |
| Clairaut's theorem: `fₓᵧ = fᵧₓ` | ✓ (both ≈ 13) |
| Hessian symmetry | ✓ |

Tests: [test/multivariable_calculus_test.dart](../test/multivariable_calculus_test.dart).

## Total differential

Bird Ch. 35: for small increments `δx, δy, …`

$$df \approx \sum_i \frac{\partial f}{\partial x_i}\,\delta x_i.$$

Useful for linear error propagation.

### Verified

| Case | Result |
|---|---|
| `f(x, y) = xy` at `(2, 3)`, `δ = (0.1, 0.05)` → `df ≈ 3·0.1 + 2·0.05 = 0.4` | ✓ |
| Mismatched point/delta lengths → `ArgumentError` | ✓ |

## Critical-point classification (2-D)

File: [lib/src/multivariable_extrema.dart](../lib/src/multivariable_extrema.dart).

`classifyCriticalPoint2D(f, point)` uses the Hessian discriminant
`D = fₓₓ fᵧᵧ − fₓᵧ²`:

| Condition | Type |
|---|---|
| `D > 0` and `fₓₓ > 0` | `CriticalPointType.minimum` |
| `D > 0` and `fₓₓ < 0` | `CriticalPointType.maximum` |
| `D < 0` | `CriticalPointType.saddle` |
| `D ≈ 0` | `CriticalPointType.inconclusive` |

The caller supplies a genuine stationary point (∇f = 0). Gradient
verification is not built in — call `gradient(f, point)` to double-check.

### Example

```dart
// Bird-style: f(x, y) = x² + y² − 2x − 4y + 5, critical point (1, 2).
double f(List<double> p) =>
    p[0] * p[0] + p[1] * p[1] - 2 * p[0] - 4 * p[1] + 5;

gradient(f, [1, 2]);              // ≈ (0, 0)
classifyCriticalPoint2D(f, [1, 2]); // CriticalPointType.minimum
```

### Verified

| Function | Critical point | Type |
|---|---|---|
| `x² + y² − 2x − 4y + 5` | `(1, 2)` | minimum |
| `−(x² + y²)` | `(0, 0)` | maximum |
| `x² − y²` | `(0, 0)` | saddle |
| `x²y²` | `(0, 0)` | inconclusive (`D = 0`) |

## Notes and limitations

- **Numerical partials lose ~1 dp per differentiation order**. Second
  partials use `h = 1e-4` by default to keep the outer subtraction
  meaningful; third and higher orders are not implemented.
- **Critical-point discovery** is *not* included. Standard approach:
  set `∇f = 0`, solve — you can do this with the numerical root-finder
  from Phase M once it lands, or by algebra outside the library.
- **`classifyCriticalPoint2D`** handles the 2-variable case. Higher-dim
  extension via Hessian eigenvalue signs is straightforward once needed
  and reuses `hessian(...)` + Phase E's eigen APIs.
- **Ch. 24 vector drawing / graphical addition** is out of scope
  (display concern). Vector arithmetic is fully covered numerically.
- **Vector equation of a line** (Bird Ch. 26 §26.4) is a use of the
  primitives above — no separate abstraction is warranted.


---

## FILE: topic-k-trig-hyperbolic.md

# Topic K — Trigonometry, hyperbolic, waveforms

Bird reference: Ch. 3 (logarithms — subsumed by `dart:math`), Ch. 5
(hyperbolic functions and identities), Ch. 11 §11.7–11.8 (sine + cosine
rule, triangle area), Ch. 12 (polar coordinates), Ch. 13 (circles,
arc length, sector area), Ch. 14 (sinusoidal form `A sin(ωt ± α)`),
Ch. 17 §17.2 (`a sin ωt + b cos ωt → R sin(ωt + α)`), Ch. 25 (phasor
addition).

## Contents

- [Hyperbolic functions](#hyperbolic-functions)
- [Sinusoidal waveforms and phasors](#sinusoidal-waveforms-and-phasors)
- [Polar / Cartesian and circles](#polar--cartesian-and-circles)
- [Triangles](#triangles)

## Hyperbolic functions

File: [lib/src/hyperbolic.dart](../lib/src/hyperbolic.dart).

Dart's `dart:math` does not expose hyperbolic functions, so we provide
them from the exponential definitions.

| Function | Definition |
|---|---|
| `sinh(x)` | `(eˣ − e⁻ˣ) / 2` |
| `cosh(x)` | `(eˣ + e⁻ˣ) / 2` |
| `tanh(x)` | `(e²ˣ − 1) / (e²ˣ + 1)` (numerically stable for large `|x|`) |
| `coth(x)` | `1 / tanh(x)`, throws at 0 |
| `sech(x)` | `1 / cosh(x)` |
| `cosech(x)` | `1 / sinh(x)`, throws at 0 |
| `arsinh(x)` | `ln(x + √(x² + 1))`, defined for all real x |
| `arcosh(x)` | `ln(x + √(x² − 1))`, requires `x ≥ 1` |
| `artanh(x)` | `½ · ln((1 + x) / (1 − x))`, requires `|x| < 1` |

### Example

```dart
sinh(5.4);   // ≈ 110.7   (Bird Ch. 5 Problem 1)
cosh(1.86);  // ≈ 3.290
tanh(50);    // ≈ 1.0     (no NaN — numerically stable)

// arsinh(sinh(x)) round-trip.
arsinh(sinh(2.5)); // = 2.5
```

### Verified identities (Bird §5.3)

For every value tested:

- `cosh²x − sinh²x = 1`
- `1 − tanh²x = sech²x`
- `coth²x − 1 = cosech²x`
- `cosh 2x = cosh²x + sinh²x` (Osborne's rule applied to cos 2x)
- `sinh 2x = 2·sinh x·cosh x`

Plus:

- **Parity**: sinh, tanh odd; cosh even.
- **Round-trips**: `arsinh(sinh(x))`, `arcosh(cosh(x))`, `artanh(tanh(x))`.
- **Cross-consistency**: `TSinh(1).eval(x) = sinh(x)`, `TCosh(1).eval(x) = cosh(x)` — matches Phase F's Laplace atoms.

Tests: [test/hyperbolic_test.dart](../test/hyperbolic_test.dart).

## Sinusoidal waveforms and phasors

File: [lib/src/waveform.dart](../lib/src/waveform.dart).

### `SinusoidalWaveform`

Represents `A · sin(ω·t + α)`.

| Member | Description |
|---|---|
| `SinusoidalWaveform({amplitude, angularFrequency, phase})` | Primary constructor. |
| `.fromFrequencyHz({amplitude, frequencyHz, phase})` | Set ω = 2π·f. |
| `.amplitude`, `.angularFrequency`, `.phase` | Fields. |
| `.frequency`, `.periodicTime`, `.phaseDeg` | Derived. |
| `.eval(t)` | `A · sin(ωt + α)`. |

### `combineSinCos(a, b)` — Bird Ch. 17 §17.2

Combines `a · sin(ω t) + b · cos(ω t)` into a single `R · sin(ω t + α)`:

$$R = \sqrt{a^2 + b^2}, \qquad \alpha = \operatorname{atan2}(b, a) \in (-\pi, \pi].$$

Returns a record `(amplitude: R, phase: α)`.

### `phasorSum(phasors)` — Bird Ch. 25

Adds a list of same-frequency `SinusoidalWaveform`s via complex-phasor
addition. Throws if the frequencies don't match or the list is empty.

### Example

```dart
// Bird Ch. 14 Problem 14: i = 30·sin(100π t + 0.27).
final i = SinusoidalWaveform(
  amplitude: 30,
  angularFrequency: 100 * math.pi,
  phase: 0.27,
);
i.frequency;    // 50 Hz
i.periodicTime; // 0.02 s
i.phaseDeg;     // 15.47°

// Bird Ch. 17 Problem 6: 3 sin ωt + 4 cos ωt → 5·sin(ωt + 0.927).
final r = combineSinCos(3, 4);
r.amplitude; // 5
r.phase;     // 0.927 rad ≈ 53.13°

// Bird Ch. 25: 3 sin(ω t) + 4 sin(ω t + π/2) → 5·sin(ω t + 53.13°).
phasorSum([
  SinusoidalWaveform(amplitude: 3, angularFrequency: 1, phase: 0),
  SinusoidalWaveform(amplitude: 4, angularFrequency: 1, phase: math.pi / 2),
]); // (amplitude: 5, phase: 0.927)
```

### Verified worked examples

| Reference | Result |
|---|---|
| Ch. 14 Problem 14 | f = 50 Hz, T = 0.02 s, phase 15.47° |
| Ch. 17 Problem 6: `3 sin + 4 cos` | `5·sin(ω t + 0.927)` |
| Ch. 17 Problem 7: `4.6 sin − 7.3 cos` | `8.628·sin(ω t − 1.008)` |
| Ch. 17 Problem 8: `−2.7 sin − 4.1 cos` | `4.909·sin(ω t − 2.153)` (principal value) |
| Ch. 25: in-phase 3 + 4 | 7·sin |
| Ch. 25: orthogonal 3 + 4 | 5·sin at 53.13° |
| Ch. 25: opposing 5 + 5 | 0 |

Tests: [test/waveform_test.dart](../test/waveform_test.dart).

## Polar / Cartesian and circles

File: [lib/src/polar.dart](../lib/src/polar.dart).

| Function | Description |
|---|---|
| `polarToCartesian(r, θ)` | Returns `(x: r cos θ, y: r sin θ)`. |
| `cartesianToPolar(x, y)` | Returns `(r: √(x² + y²), θ: atan2(y, x))`, θ ∈ (−π, π]. |
| `arcLength(r, θ)` | `s = r · θ` (Bird §13.4). |
| `sectorArea(r, θ)` | `A = ½ · r² · θ`. |
| `linearVelocity(r, ω)` | `v = r · ω`. |

### Verified

| Case | Result |
|---|---|
| Polar → Cartesian → Polar round trip | Preserves both coordinates |
| `polarToCartesian(4, 30°)` | `(3.464, 2.000)` |
| `cartesianToPolar(3, 4)` | `(5, atan2(4, 3))` |
| `arcLength(3, 2π) = 6π` | ✓ (full circle) |
| `sectorArea(3, 2π) = 9π` | ✓ (full circle area) |

Tests: [test/polar_test.dart](../test/polar_test.dart).

## Triangles

File: [lib/src/triangle.dart](../lib/src/triangle.dart).

Bird Ch. 11 §11.7 sine + cosine rule, §11.8 triangle area. All angles
in radians.

| Function | Purpose |
|---|---|
| `sineRuleSide({knownSide, knownAngle, otherAngle})` | Given a side + its opposite angle + one more angle, return the second side. |
| `cosineRuleSide({sideB, sideC, includedAngle})` | Two sides + included angle → third side. |
| `cosineRuleAngle({sideA, sideB, sideC})` | Three sides → angle opposite `sideA`. |
| `triangleAreaSAS({sideB, sideC, includedAngle})` | `½ b c sin A`. |
| `triangleAreaSSS({sideA, sideB, sideC})` | Heron's formula. |

### Verified

| Case | Result |
|---|---|
| Sine rule: `sin A / A ↔ sin B / B` | ✓ |
| 30-60-90 hypotenuse 2 → opposite of 30° = 1 | ✓ |
| Cosine rule: sides 5, 7, angle 60° → `√39` | ✓ |
| 3-4-5 triangle: angle opposite 5 = π/2 | ✓ |
| Equilateral: all angles = π/3 | ✓ |
| SAS area of right 3-4 → 6 | ✓ |
| Heron's area of 3-4-5 → 6 | ✓ |
| SAS and SSS agree on the same triangle | ✓ |
| Impossible triangle (1, 1, 10) → ArgumentError | ✓ |

Tests: [test/triangle_test.dart](../test/triangle_test.dart).

## Notes and limitations

- **Ch. 3 (logarithms)** — Dart's `math.log` (natural), `math.exp`, and
  arbitrary bases via `math.log(x) / math.log(base)` are sufficient. No
  wrapper module.
- **Trig identity manipulation (Ch. 15, 16, 17 §17.1)** is symbolic and
  belongs to a future symbolic-algebra layer, not this phase. Numeric
  verification of any identity is a trivial `expect` against `dart:math`.
- **Compound-angle formulae (Ch. 17 §17.1)** are similarly pen-and-paper
  identities; verify numerically as needed.
- **Angle input convention**: `phase`, `angleTo`, etc. are all in
  **radians**. Where Bird uses degrees, our results are consistent
  (Ch. 17 Problem 7: −57.78° = −1.008 rad).
- **Principal-value angle**: `combineSinCos` and `cartesianToPolar`
  return α ∈ (−π, π] via `atan2`. Bird sometimes reports the
  supplementary angle in the third quadrant (Problem 8: 236.63° vs our
  −2.153 rad ≈ −123.37°); the two are equivalent modulo 2π.


---

## FILE: topic-l-symbolic-diff.md

# Topic L — Symbolic differentiation

Bird reference: Ch. 27 (differentiation rules), Ch. 28 (rates,
turning points), Ch. 29 (parametric), Ch. 30 (implicit),
Ch. 31 (logarithmic), Ch. 32 (hyperbolic), Ch. 33 (inverse trig +
hyperbolic).

This phase introduces a small symbolic algebra layer — the only
non-numeric module in the library. Scope is limited to what's needed
for exact analytic differentiation of Bird's problem set.

## Contents

- [`Expr` AST](#expr-ast)
- [Building expressions](#building-expressions)
- [`differentiate` and `simplify`](#differentiate-and-simplify)
- [Verified rules](#verified-rules)

## `Expr` AST

File: [lib/src/expr.dart](../lib/src/expr.dart).

Sealed hierarchy of expression nodes.

| Node | Meaning |
|---|---|
| `Const(v)` | numeric constant |
| `Var(name)` | free variable |
| `Add(a, b)` | `a + b` |
| `Neg(a)` | `−a` |
| `Mul(a, b)` | `a · b` |
| `Div(a, b)` | `a / b` |
| `Pow(base, exp)` | `base^exp` |
| `Sin`, `Cos`, `Tan`, `Exp`, `Ln` | Standard functions |
| `Sinh`, `Cosh`, `Tanh` | Hyperbolic |
| `Asin`, `Acos`, `Atan` | Inverse trig |
| `Asinh`, `Acosh`, `Atanh` | Inverse hyperbolic |

Every node implements `eval(Map<String, num> bindings) → double`.

## Building expressions

- Numeric literals auto-lift via operator overloads: `x + 3`, `2 * x`, `x / 4`.
- `variable('x')` (top-level helper) or `Var('x')` for named variables.
- Standard functions are concrete constructors: `Sin(x)`, `Ln(Sin(x))`, etc.
- `Pow(base, exponent)` for powers (both numeric and symbolic exponents).

### Example

```dart
final x = variable('x');

// f(x) = e^{2x} · cos(3x).
final f = Exp(Const(2) * x) * Cos(Const(3) * x);

// d/dx f — Bird self-test Q1.
final df = differentiate(f, 'x');
df.eval({'x': 0.5});
// = 2·e^1·cos(1.5) − 3·e^1·sin(1.5)
```

## `differentiate` and `simplify`

File: [lib/src/expr_calculus.dart](../lib/src/expr_calculus.dart).

### `differentiate(e, v)`

Applies the classical rules mechanically via pattern matching on the
sealed hierarchy:

- Constant → 0
- Variable → 1 if it matches the target, else 0
- Sum, negation → distribute over children
- **Product rule**: `(fg)' = f'g + fg'`
- **Quotient rule**: `(f/g)' = (f'g − fg') / g²`
- **Power rule**: `d/dx bⁿ = n·bⁿ⁻¹·db/dx` when `n` is a numeric constant
- **General power**: `d/dx b^e = b^e · (e'·ln b + e·b'/b)`
- Chain rule embedded in every standard function (multiplies by the
  inner derivative).
- Standard trig, hyperbolic, and inverse forms per Bird's tables.

### `simplify(e)`

Best-effort readability pass — not a canonicalisation engine.

- Constant folding of `+ − × ÷ ^`
- `x + 0 = x`, `0 + x = x`, `x · 0 = 0`, `x · 1 = x`, `x / 1 = x`
- `x⁰ = 1`, `x¹ = x`, `0^n = 0` (n ≠ 0), `1^n = 1`
- `−(−x) = x`, `Neg(Const c) = Const(−c)`
- `a + (−b)` → `a − b`
- No like-term collection, no distributive expansion.

## Verified rules

Correctness is checked two ways:

1. Simple rules (constants, variables, elementary powers) via
   structural comparison of `simplify(differentiate(...)).toString()`
   against the expected form.
2. Complex composites via numerical evaluation of the symbolic
   derivative against a closed-form answer at multiple test points.

| Rule | Test |
|---|---|
| `d/dx (const) = 0` | ✓ |
| `d/dx (x) = 1`; `d/dx (y) w.r.t. x = 0` | ✓ |
| `d/dx (x²) = 2x` and `d/dx (x³ + 2x² − 5x + 3) = 3x² + 4x − 5` | ✓ |
| Product: `d/dx (x·sin x) = sin x + x·cos x` | ✓ |
| Product: `d/dx (x²·e^x)` | ✓ |
| Quotient: `d/dx (sin x / x) = (x cos x − sin x)/x²` | ✓ |
| Quotient: `d/dx (1/x) = −1/x²` | ✓ |
| Chain: `d/dx (sin(x²)) = 2x·cos(x²)` | ✓ |
| Chain: `d/dx (e^{2x}) = 2·e^{2x}` | ✓ |
| Chain: `d/dx (ln(sin x)) = cos x / sin x` | ✓ |
| **Bird Q1**: `d/dx (e^{2x} · cos(3x)) = 2·e^{2x}·cos 3x − 3·e^{2x}·sin 3x` | ✓ |
| `d(sin) = cos`, `d(cos) = −sin`, `d(tan) = sec²` | ✓ |
| `d(sinh) = cosh`, `d(cosh) = sinh`, `d(tanh) = 1 − tanh²` | ✓ |
| `d(asin) = 1/√(1−x²)`, `d(acos) = −1/√(1−x²)`, `d(atan) = 1/(1+x²)` | ✓ |
| `d(asinh) = 1/√(1+x²)`, `d(acosh) = 1/√(x²−1)`, `d(atanh) = 1/(1−x²)` | ✓ |
| Generalised power: `d/dx (x^x) = x^x·(ln x + 1)` | ✓ |

Simplification:
- `x + 0`, `0 · x`, `x^0`, `x^1`, `−(−x)` and constant folding all
  produce the expected canonical form.

Tests: [test/expr_test.dart](../test/expr_test.dart).

## Notes and limitations

- **Not a CAS.** `simplify` deliberately does not expand, factor,
  or collect like terms. Output can be verbose (e.g. `Neg(Neg(x))`
  reduces to `x`, but `2x + 3x` does *not* collapse to `5x`).
- **Numerical verification.** For anything beyond hand-simplifiable
  forms, verify via `eval` against a closed-form answer or against the
  numerical `derivative` from Phase A. Tests use this strategy
  consistently.
- **Implicit differentiation (Ch. 30)** is not a separate module — the
  standard approach is to isolate `dy/dx` symbolically from an equation
  and pass the resulting expression through the differentiator.
- **Parametric differentiation (Ch. 29)** is `dy/dx = (dy/dt) / (dx/dt)`
  — a two-step application of `differentiate`.
- **Logarithmic differentiation (Ch. 31)** — use `Ln(...)` around a
  product-of-powers expression, then differentiate the sum.
- **Symbolic integration** would be a much larger effort (Risch or
  pattern-based) and is out of scope. Numerical integration from
  Phase A / Phase M covers the practical needs.


---

## FILE: topic-m-applied-numerical.md

# Topic M — Applied numerical methods

Bird reference: Ch. 9 (iterative root-finding), Ch. 19 (irregular
areas, volumes, mean value of a waveform), Ch. 37 (mean value, RMS),
Ch. 38 (arc length, surfaces + volumes of revolution, centroids),
Ch. 45 (mid-ordinate rule).

## Contents

- [Root finding](#root-finding)
- [Mid-ordinate rule](#mid-ordinate-rule)
- [Mean value + RMS](#mean-value--rms)
- [Arc length](#arc-length)
- [Volumes and surfaces of revolution](#volumes-and-surfaces-of-revolution)
- [Centroids](#centroids)
- [Interpolation and polynomial fitting (Phase Q)](#interpolation-and-polynomial-fitting-phase-q)
- [Non-linear curve fitting (Phase S)](#non-linear-curve-fitting-phase-s)

## Root finding

File: [lib/src/root_finding.dart](../lib/src/root_finding.dart).

Three iterative methods for `f(x) = 0`:

| Function | Convergence | Notes |
|---|---|---|
| `bisection(f, {a, b, tolerance, maxIterations})` | Linear (each iteration halves the bracket) | Robust — requires only a sign-changing bracket. |
| `newtonRaphson(f, fPrime, {x0, tolerance, maxIterations})` | Quadratic near a simple root | Needs analytic `f'`; may diverge. |
| `secant(f, {x0, x1, tolerance, maxIterations})` | Super-linear (order ≈ 1.618) | Newton-Raphson without the analytic derivative. |

### Example

```dart
// √2 via three different methods.
bisection((x) => x * x - 2, a: 1, b: 2);                       // 1.41421356…
newtonRaphson((x) => x*x - 2, (x) => 2*x, x0: 1);              // same
secant((x) => x * x - 2, x0: 1, x1: 2);                        // same

// The Dottie number, root of cos x = x.
newtonRaphson(
  (x) => math.cos(x) - x,
  (x) => -math.sin(x) - 1,
  x0: 0.5,
); // 0.7390851332151607
```

### Verified

| Case | Result |
|---|---|
| `bisection` on `x² − 2` in `[1, 2]` | `√2` within 1e-10 |
| `bisection` on `cos x − x` in `[0, 1]` | Dottie number within 1e-10 |
| `bisection` on `x³ − 5x + 1` in `(0, 1)` | residual < 1e-9 |
| Sign of `f(a)·f(b)` not opposite → `ArgumentError` | ✓ |
| `a ≥ b` → `ArgumentError` | ✓ |
| `newtonRaphson` on `x² − 2` from `x0 = 1` | `√2` within 1e-12 |
| Zero derivative during iteration → `StateError` | ✓ |
| `secant` on `√2` and Dottie number | Within 1e-12 |
| All three agree on the root of `x³ − x − 2` in `[1, 2]` | ✓ |

Tests: [test/root_finding_test.dart](../test/root_finding_test.dart).

## Mid-ordinate rule

File: [lib/src/integration_applications.dart](../lib/src/integration_applications.dart).

`midOrdinate(f, a, b, n)` computes

$$\int_a^b f(x)\,dx \;\approx\; h \sum_{k=0}^{n-1} f\!\left(a + h\!\left(k + \tfrac{1}{2}\right)\right), \quad h = \tfrac{b-a}{n}.$$

Bird Ch. 45. Same O(h²) order as trapezium, marginally more accurate
for equal `n`. **Exact for linear integrands.**

## Mean value + RMS

Bird Ch. 19 and Ch. 37.

| Function | Formula |
|---|---|
| `meanValue(f, a, b, {n})` | `(1/(b − a)) · ∫ f dx` (via Simpson) |
| `rmsValue(f, a, b, {n})` | `√(mean of f²)` |

### Verified

| Case | Result |
|---|---|
| Mean of `sin x` over `[0, π]` | `2/π` (within 1e-6) |
| Mean of a constant | The constant |
| Mean of `x` over `[0, 10]` | Midpoint 5 |
| RMS of `sin x` (and `cos x`) over `[0, 2π]` | `1/√2` |
| RMS of a constant `k` | `|k|` |

## Arc length

`curveLength(f, a, b, {n, h})` computes

$$L = \int_a^b \sqrt{1 + f'(x)^2}\,dx$$

using Simpson for the outer integral and the [derivative](topic-1-further-calculus.md) helper for `f'`.

### Verified

| Case | Analytic | Note |
|---|---|---|
| `y = x` on `[0, 1]` | `√2` | Straight-line hypotenuse. |
| `y = 2x` on `[0, 3]` | `3√5` | Same idea. |
| `y = cosh x` on `[0, 1]` | `sinh(1)` | Uses `1 + sinh²x = cosh²x`. |

**Endpoint singularity warning.** `f(x) = x^(3/2)` has a cusp at 0 and
`derivative` returns `NaN` there (evaluating `math.pow(-h, 1.5)`).
For such integrands either start the interval slightly inside the
singularity or supply a smooth reparameterisation.

## Volumes and surfaces of revolution

Bird Ch. 38.

| Function | Formula |
|---|---|
| `volumeOfRevolution(f, a, b, {n})` | `π · ∫ f² dx` (disc method about the x-axis) |
| `surfaceOfRevolution(f, a, b, {n, h})` | `2π · ∫ f · √(1 + f'²) dx` |

### Verified

| Case | Analytic |
|---|---|
| Cone from `y = x` on `[0, 1]` | `V = π/3` |
| Cylinder from `y = 3` on `[0, 5]` | `V = 45π` |
| Sphere of radius 2 from `y = √(4 − x²)` | `V = (32/3) π` (within 1e-3 at `n = 400`) |
| Cone lateral surface from `y = x` on `[0, 1]` | `π · √2` |
| Cylinder lateral surface from `y = 3` on `[0, 5]` | `30π` |

Note: the sphere test integrates over `[-r, r]` with `n = 400` and
accepts ~1e-3 error — Simpson underperforms slightly near the sharp
`√(r² − x²)` endpoints.

## Centroids

`centroid(f, a, b, {n})` returns `(xBar, yBar)` for the planar region
bounded by `y = f(x)`, the x-axis, and `x ∈ [a, b]`:

$$\bar{x} = \frac{\int x\,f\,dx}{\int f\,dx}, \qquad \bar{y} = \frac{\int \tfrac{1}{2} f^2\,dx}{\int f\,dx}.$$

Requires `f ≥ 0` on the interval and non-zero area (throws otherwise).

### Verified

| Region | Centroid |
|---|---|
| Right triangle under `y = x` on `[0, 1]` | `(2/3, 1/3)` |
| Rectangle `y = 4` on `[0, 6]` | `(3, 2)` |
| Semicircle radius 3, `y = √(9 − x²)` on `[-3, 3]` | `x̄ = 0`, `ȳ ≈ 4·3/(3π)` ≈ 1.273 |
| Zero-area region | `StateError` |

## Notes and limitations

- **Endpoint singularities** in `curveLength` / `surfaceOfRevolution`
  can produce `NaN` because central-difference `derivative` evaluates
  `f(x − h)` for `x ≈ a`. Recommended workarounds:
  1. Shrink the integration interval by a small ε.
  2. Reparameterise (e.g. angle for a circle).
  3. Provide an analytic `fPrime` variant (a follow-up API if the
     use case shows up).
- **Simpson at `n = 100`** gives ~1e-6 accuracy for smooth integrands.
  Interior singularities or sharp endpoints hurt this — bump `n` if
  needed.
- **`newtonRaphson` divergence**: the classifier throws when `f'` hits
  zero. A more sophisticated variant might switch to bisection on
  poorly conditioned steps; not implemented.

## Interpolation and polynomial fitting (Phase Q)

File: [lib/src/interpolation.dart](../lib/src/interpolation.dart) and
`polynomialFit` in [lib/src/regression.dart](../lib/src/regression.dart).

Four interchangeable interpolators, all callable as `interp(x)`:

| Class | Best for | Complexity |
|---|---|---|
| `LagrangeInterpolator(xs, ys)` | Arbitrary knots, few points | O(n²) per query |
| `NewtonForwardInterpolator(xs, ys)` | Equally-spaced knots, target near table start | O(n) per query after O(n²) setup |
| `NewtonBackwardInterpolator(xs, ys)` | Equally-spaced knots, target near table end | Same |
| `CubicSpline(xs, ys)` | Smooth reconstruction, moderate-to-many knots | O(log n) per query after O(n) setup |

`CubicSpline` uses natural boundary conditions (second derivative zero
at endpoints). It also exposes `.derivative(x)` for the pointwise first
derivative — useful when the tabular data is a smooth function and you
want a differentiable interpolant.

### Example

```dart
// A calibration curve tabulated at 6 measurements.
final xs = [0.0, 1.0, 2.0, 4.0, 7.0, 10.0];
final ys = [1.0, 3.0, 2.0, 5.0, 4.0, 6.0];

final l = LagrangeInterpolator(xs, ys);
final s = CubicSpline(xs, ys);

l(3.5);           // Lagrange estimate
s(3.5);           // spline estimate
s.derivative(3.5);
```

### `polynomialFit(xs, ys, degree)`

Least-squares polynomial fit via QR (Vandermonde design). Returns a
`Polynomial` (ascending coefficients, the library-wide convention) so
you can immediately `.eval(x)`, differentiate, or add to other
polynomials.

```dart
// Fit y ≈ c₀ + c₁ x + c₂ x² to noisy data.
final p = polynomialFit(xs, ys, 2);
p.coefficients;    // [c0, c1, c2]
p.eval(3.5);
p.degree;
```

Numerically stable through moderate degrees; for very high degrees
(≥ ~15) the Vandermonde design becomes catastrophically ill-conditioned
regardless of the QR route — use orthogonal polynomial bases (Chebyshev,
Legendre) or lower the degree.

Tests: [test/interpolation_test.dart](../test/interpolation_test.dart)
— exact recovery on polynomial data (Lagrange and Newton); spline
reproduces sin(x) to 1e-4 on a 20-knot grid; polynomial fit recovers a
degree-3 polynomial to 1e-8; degree-1 fit matches simple linear
regression to 1e-9.

## Non-linear curve fitting (Phase S)

File: [lib/src/nonlinear_fit.dart](../lib/src/nonlinear_fit.dart).

Levenberg–Marquardt fits any model `y = f(x, β)` to data `(xᵢ, yᵢ)`
by minimising `Σ (yᵢ − f(xᵢ, β))²`. The step direction blends
Gauss–Newton (fast near the minimum) with steepest descent (safe far
from it) via a damping parameter λ that self-adjusts.

```dart
final fit = nonlinearLeastSquares(
  xs: xs,
  ys: ys,
  model: (x, p) => p[0] * math.exp(-p[1] * x) + p[2],   // your model
  initialParameters: [1.0, 1.0, 0.0],
);

fit.parameters;         // best-fit β
fit.sumOfSquares;       // Σ residuals²
fit.standardErrors;     // 1-σ per-parameter uncertainty
fit.covariance;         // full p × p covariance matrix
fit.rSquared(ys);       // coefficient of determination
fit.converged;          // did convergence trigger, or did we hit maxIterations?
```

`model` is any `double Function(double x, List<double> params)`.
Optionally supply an analytic `jacobian` returning
`[∂f/∂β₀, ∂f/∂β₁, …]` — recommended for high-dimensional or
badly-scaled problems, but omitting it makes the routine transparently
fall back to central-difference numerical derivatives.

Ready-made models with matching analytic Jacobians:

| Model | Signature | Parameters |
|---|---|---|
| `gaussianModel` | `A · exp(−(x − μ)² / (2σ²))` | `[A, μ, σ]` |
| `exponentialDecayModel` | `A · exp(−k · x) + c` | `[A, k, c]` |
| `dampedSinusoidModel` | `A · exp(−k · x) · sin(ω x + φ) + c` | `[A, k, ω, φ, c]` |

```dart
final peak = nonlinearLeastSquares(
  xs: xs, ys: ys,
  model: gaussianModel,
  jacobian: gaussianJacobian,
  initialParameters: [maxY, argmaxX, spread],
);
// peak.parameters → [A, μ, σ] with peak.standardErrors on each.
```

**Uncertainty estimates** come from `(JᵀJ)⁻¹ · σ²_residual` where
`σ²_residual = χ² / (n − p)`. These are asymptotic (large-n, small-noise)
1-σ estimates — fine for well-identified fits; for tight uncertainty
quantification on marginal models use bootstrap resampling instead.

Tests: [test/nonlinear_fit_test.dart](../test/nonlinear_fit_test.dart)
— noise-free recovery on Gaussian / exponential-decay / damped-sinusoid
models to 1e-4 to 1e-6; analytic and numerical Jacobians agree to 1e-4;
noisy Gaussian peak recovered within 5 % with R² > 0.99.


---

## FILE: topic-n-symbolic-integration.md

# Topic N — Symbolic integration

Bird reference: Ch. 37 (standard integrals), Ch. 39 §39.2 (linear-inner
substitution), Ch. 40 (trig / hyperbolic substitution), Ch. 42
(Weierstrass `t = tan(θ/2)` — closed forms), Ch. 43 (integration by
parts), Ch. 44 (reduction formulae for sin/cos powers).

Companion to Phase L (symbolic differentiation). Same `Expr` AST;
`integrate(e, v)` produces an antiderivative when the expression is
within the covered patterns and throws `IntegrationFailure` otherwise.

## Contents

- [`integrate` and `IntegrationFailure`](#integrate-and-integrationfailure)
- [What's covered](#whats-covered)
- [What's not](#whats-not)
- [Verification strategy](#verification-strategy)

## `integrate` and `IntegrationFailure`

File: [lib/src/expr_integrate.dart](../lib/src/expr_integrate.dart).

```dart
Expr integrate(Expr e, String v);          // may throw IntegrationFailure
class IntegrationFailure implements Exception { … }
```

The constant of integration is **omitted** — it's arbitrary and would
only clutter the output. Definite integrals are computed as
`F(b) − F(a)`.

## What's covered

### Linearity

Sum, difference, constant multiplier — all lift through the integrator
automatically.

```dart
final x = variable('x');
integrate(Const(2) * Sin(x) - Const(3) * Cos(x) + Pow(x, Const(2)), 'x');
// → −2·cos(x) − 3·sin(x) + x³/3   (after simplify)
```

### Power rule (Bird Ch. 37)

For any linear inner `a·v + b`:

$$\int (a v + b)^n\,dv = \frac{(a v + b)^{n+1}}{a(n+1)} \quad (n\neq -1), \qquad \int \frac{dv}{a v + b} = \frac{\ln|a v + b|}{a}.$$

```dart
integrate(Pow(x, Const(5)), 'x');                    // x⁶/6
integrate(Pow(x, Const(-1)), 'x');                   // ln(x)
integrate(Pow(x, Const(0.5)), 'x');                  // (2/3)·x^{3/2}
integrate(Pow(Const(2) * x + Const(3), Const(4)), 'x');
integrate(Pow(Const(3) * x + Const(5), Const(-1)), 'x'); // ln(3x+5)/3
```

### Standard integrals with linear inner (Bird Ch. 39)

For `f ∈ {sin, cos, sinh, cosh, exp}` and inner `a·v + b`:

$$\int f(av+b)\,dv = \frac{F(av+b)}{a}.$$

```dart
integrate(Sin(Const(3) * x), 'x');                   // −cos(3x)/3
integrate(Cos(Const(2) * x + Const(1)), 'x');
integrate(Exp(Const(2) * x), 'x');                   // e^{2x}/2
integrate(Sinh(x / Const(2)), 'x');                  // 2·cosh(x/2)
```

### Trig / hyperbolic substitution (Bird Ch. 40)

Pattern coverage for the standard forms:

$$\int \frac{dv}{a^2 + v^2} = \frac{1}{a}\,\operatorname{atan}\!\Big(\frac{v}{a}\Big), \qquad \int \frac{dv}{a^2 - v^2} = \frac{1}{a}\,\operatorname{artanh}\!\Big(\frac{v}{a}\Big),$$

$$\int \frac{dv}{\sqrt{a^2 - v^2}} = \operatorname{asin}\!\Big(\frac{v}{a}\Big), \quad \int \frac{dv}{\sqrt{a^2 + v^2}} = \operatorname{arsinh}\!\Big(\frac{v}{a}\Big), \quad \int \frac{dv}{\sqrt{v^2 - a^2}} = \operatorname{arcosh}\!\Big(\frac{v}{a}\Big).$$

```dart
integrate(Const(1) / (Const(4) + Pow(x, Const(2))), 'x');   // (1/2)·atan(x/2)
integrate(Const(1) / (Const(9) - Pow(x, Const(2))), 'x');   // (1/3)·artanh(x/3)
integrate(Const(1) / Pow(Const(9) - Pow(x, Const(2)), Const(0.5)), 'x');
// → asin(x/3)
integrate(Const(1) / Pow(Const(4) + Pow(x, Const(2)), Const(0.5)), 'x');
// → arsinh(x/2)
integrate(Const(1) / Pow(Pow(x, Const(2)) - Const(4), Const(0.5)), 'x');
// → arcosh(x/2)   (valid for x > a)
```

The legacy shorthand `∫ dv/(1 + v²) = atan v` still works as the a = 1
special case.

### Integration by parts (Bird Ch. 43)

Applied to `polynomial(v) · standard-function(v)` products, choosing
the polynomial as `u` and the standard function as `dv`, then
recursively integrating. Also picks `Ln` as `u` when it appears in a
product (LIATE rule).

```dart
integrate(x * Exp(x), 'x');                          // (x−1)·e^x
integrate(x * Sin(x), 'x');                          // sin(x) − x·cos(x)
integrate(Pow(x, Const(2)) * Exp(x), 'x');           // (x² − 2x + 2)·e^x
integrate(Pow(x, Const(2)) * Sin(x), 'x');           // 2x sin x + (2 − x²) cos x
integrate(Pow(x, Const(2)) * Ln(x), 'x');            // (x³/3) ln x − x³/9
integrate(Ln(x), 'x');                               // x·ln x − x
```

Double by-parts (`x²·sin x`, `x²·e^x`) works via recursion — the inner
`x·sin x` integral terminates in one more by-parts step.

### Reduction formulae for sin/cos powers (Bird Ch. 44)

For integer `n ≥ 2` and linear inner `a·v + b`:

$$\int \sin^n(av+b)\,dv = -\frac{\sin^{n-1}(av+b)\cos(av+b)}{n\,a} + \frac{n-1}{n}\int \sin^{n-2}(av+b)\,dv,$$

$$\int \cos^n(av+b)\,dv = \frac{\cos^{n-1}(av+b)\sin(av+b)}{n\,a} + \frac{n-1}{n}\int \cos^{n-2}(av+b)\,dv,$$

terminating at $I_0 = v$ or $I_1 = -\cos(av+b)/a$ (resp. $\sin(av+b)/a$).

```dart
integrate(Pow(Sin(x), Const(2)), 'x');
integrate(Pow(Sin(x), Const(4)), 'x');
integrate(Pow(Cos(Const(2) * x), Const(3)), 'x');
// Definite: ∫₀^{π/2} sin²(x) dx = π/4
```

### Weierstrass `t = tan(θ/2)` substitution (Bird Ch. 42)

Closed-form patterns for the two most common textbook denominators,
valid when `α > |β|`:

$$\int \frac{d\theta}{\alpha + \beta\cos\theta} = \frac{2}{\sqrt{\alpha^2-\beta^2}}\;\operatorname{atan}\!\left(\sqrt{\frac{\alpha-\beta}{\alpha+\beta}}\;\tan(\theta/2)\right),$$

$$\int \frac{d\theta}{\alpha + \beta\sin\theta} = \frac{2}{\sqrt{\alpha^2-\beta^2}}\;\operatorname{atan}\!\left(\frac{\alpha\tan(\theta/2) + \beta}{\sqrt{\alpha^2-\beta^2}}\right).$$

```dart
integrate(Const(1) / (Const(2) + Cos(x)), 'x');
integrate(Const(1) / (Const(5) + Const(4) * Cos(x)), 'x');
integrate(Const(1) / (Const(3) - Cos(x)), 'x');   // β = −1
integrate(Const(1) / (Const(2) + Sin(x)), 'x');
```

The result is only valid on branches where `tan(θ/2)` is finite, i.e.
away from odd multiples of `π`. Definite integrals across those
discontinuities need to be split by hand.

## What's not

- **Nonlinear inner arguments** — `∫ sin(x²) dx` throws (not
  elementary anyway).
- **General Weierstrass driver** — only `α + β·cos θ` and
  `α + β·sin θ` are recognised. Denominators like
  `α·sin θ + β·cos θ` or `α + β·cos θ + γ·sin θ` are not
  matched — do the substitution by hand into a rational integrand in
  `t` and use `partialFractions`.
- **General reduction formulae beyond sin/cos powers (Ch. 44)** —
  `tanⁿ`, `secⁿ`, `sinᵐ · cosⁿ` reductions are not implemented.
- **Partial-fractions integration (Ch. 41)** — compose it manually:
  call `partialFractions(...)`, integrate each term, sum.

If you hit an `IntegrationFailure`, either transform the integrand by
hand into a pattern the library recognises, or drop back to numerical
integration via [`simpson`](topic-1-further-calculus.md#simpsons-13-rule).

## Verification strategy

Symbolic-integration tests **don't check the resulting `Expr` tree
structurally** — different but equivalent forms would spuriously fail.
Instead, they:

1. Compute `F = integrate(f, x)`.
2. Compute `F' = differentiate(F, x)`.
3. Evaluate `F'` at several test points and compare to `f` evaluated at
   the same points.

Any correct antiderivative satisfies this check (the constant of
integration drops out under differentiation). This is the same
strategy Phase L uses for verifying `differentiate`, run in reverse.

For definite integrals, tests compare `F(b) − F(a)` against a
closed-form answer:

| Case | Result |
|---|---|
| `∫₀¹ x² dx` | `1/3` |
| `∫₀^π sin x dx` | `2` |
| `∫₀^{π/2} x sin x dx` | `1` |
| `∫₁² x² ln x dx` (syllabus Q2) | `8 ln 2 / 3 − 7/9` |

Tests: [test/expr_integrate_test.dart](../test/expr_integrate_test.dart).

## Interaction with `simplify`

Phase N required extending `simplify` with:

- Structural equality (`exprEqual(a, b)`) between expressions.
- Same-base power combination: `xᵐ · xⁿ → xᵐ⁺ⁿ` (also with bare `x`).
- Same-base division reduction: `xᵐ / xⁿ → xᵐ⁻ⁿ`.
- `1/x → x^{-1}` normalisation.
- Constant-denominator normalisation: `a / c → (1/c) · a`.
- Associativity distribution through `Mul` so that `(xᵐ · c) · xⁿ`
  collapses correctly.

These simplification rules also improve the readability of Phase L's
symbolic derivatives — no dedicated tests were added for them, but
they're exercised end-to-end whenever a by-parts recursion runs.

## Notes and limitations

- **`integrate` composes with `differentiate`** for verification, so any
  new integrand you throw at the library can be sanity-checked
  end-to-end: differentiate the result, compare to the input.
- **Combining like terms** (e.g. `2x + 3x → 5x`) is still not
  implemented. Output can be verbose but numerically correct.
- **Full symbolic Weierstrass driver** (rewrite `R(sin θ, cos θ)` into a
  rational integrand in `t`, then partial-fractions-integrate) is the
  natural next extension — it would subsume the current two closed-form
  cases and cover the remaining Bird Ch. 42 problems.
