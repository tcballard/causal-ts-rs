# Validation

Validated on Linux x86-64, Rust 1.99.1, using Cargo.lock. Source revision: Bloomberg Causal-TS `0d1fae061b2445a51d0c8b75fc565d897dd54fb7`.

## Test coverage

The CPU suite has **100 tests**, with two additional hardware tests explicitly ignored by default:

| Suite | Tests | Evidence |
| --- | ---: | --- |
| CEDAR upstream | 26 | 25 discovery configurations plus raw CI statistics |
| Numeric/core | 15 | Analytical statistics, BH, dcor versus independent quadratic definition, constraints, missing data and determinism |
| Prepared ParCorr | 5 | Complete-data scalar/batch parity, per-test missing rows, singular/constant fallbacks, validation and independent discovery windows |
| CDNOTS | 21 | 18 upstream graphs, kernel scores, configuration/exports and constraints/sparse-lag invariants |
| CDNOTS+ | 17 | Upstream endpoint graphs, marks, binary graphs and skeleton p-values |
| GRACE/tensor CI | 10 | Three neural oracle fixtures, helpers, training reproducibility, cross-validation, stability, validation and CPU tensor CI, CI skeleton parity and C-model output |
| CLI | 6 | CSV discovery, configuration, graph export, evaluation, algorithm routing and protection of existing output on errors |
| Hardware | 2 ignored | CUDA/Metal forward, backward, Adam and CI checks; require target hardware |

The standard CPU suite, the statistical-only feature build, formatting, strict Clippy and the release build are checked before packaging. The Linux executable is exercised on each algorithm. No macOS, Windows, CUDA or Metal runtime validation was performed. Included GitHub workflow files have not been run remotely in this session.

## Upstream fixtures

Fixtures are generated from the unmodified pinned Python implementation using CPU PyTorch. All numerical inputs and expected results are retained under `tests/fixtures`, so running Rust tests requires neither Python nor PyTorch.

CEDAR coverage includes defaults, lags one through three, no pruning, single/multiple lags, all-lag sweeps, four lag statistics, three normalization settings, lag zero, BIC, no autoregression, target-only discovery, both child filters, AR-relative filtering, required/forbidden edges, missing observations, single/multiple C covariates and GCMI.

CDNOTS coverage includes default and extended lag windows, no C, only contemporaneous variables, bounded depth, unlimited combinations, sepset search, all collider priorities, lag verification, sparse lags, trends, multiple C columns, missing data, GCMI, sequential removal and a nonstationarity case that executes the kernel sink-orientation step.

CDNOTS+ coverage includes default and extended windows, no C, lag zero, limited screening depth, all priorities, lag verification, trends, multiple C columns, missing data, GCMI, parent caps, the legacy conditioning option and alternate alpha.

| Comparison | Tolerance |
| --- | --- |
| Discovery graphs, CDNOTS endpoint matrices and marks | Exact equality on fixtures |
| CEDAR selected lags, AR orders and test counts | Exact equality |
| CEDAR lag importance / p-values | 1e-10 / 1e-9 absolute error |
| CEDAR retained ParCorr / GCMI evidence | 1e-8 / 1e-5 |
| Raw CPU partial-correlation statistic / p-value | 1e-12 / 1e-10 |
| CDNOTS skeleton p-values | 2e-6 (includes f32 missing-data/GCMI reference paths) |
| CDNOTS+ skeleton p-values | 1e-5 |
| Nonstationarity score | 1e-6 |
| GRACE predictions, loss and Adam parameters | 2e-5 |
| GRACE gradients | 5e-5 |
| GRACE deterministic gates | 1e-6 |
| Tensor CPU CI versus native CPU CI | 1e-10 |

CEDAR synthetic C-target lag scores are excluded from numeric comparisons because residualizing deterministic time leaves implementation-dependent rounding noise; these targets are unused. Full graphs, selected lags and test counts are still compared.

## GRACE oracle details

`grace_reference.json` includes basis, linear and nonlinear encoders. Each fixture transfers every trainable parameter, an input batch, a binary skeleton and explicit uniform gate draws from upstream. Rust verifies deterministic predictions, sampled predictions, Gaussian/L0/lag-group loss, deterministic gate extraction, every parameter gradient and one coupled-weight-decay Adam update.

Training tests separately cover seeded reproducibility, decreasing training loss, permanent masking of excluded edges, cross-validation selection, stability sweeps and per-pair lag limits. They establish execution and reproducibility, not equality of final graphs from unrelated Rust/PyTorch random streams.

Optional hardware checks:

```sh
cargo test --locked --features cuda --test accelerator cuda_forward_backward_adam_and_ci -- --ignored
cargo test --locked --features metal --test accelerator metal_forward_backward_adam_and_ci -- --ignored
```

They intentionally fail if the requested device is unavailable. Default tests verify those errors in CPU-only builds. Neither GPU backend was compiled or exercised here because this environment has no CUDA toolkit/device or Apple SDK/device.

## CEDAR CPU baseline

The v0.1 CEDAR implementation, retained in v0.2, measured a median **1.184 ms** in Rust against **11.932 ms** upstream on seven warm single-thread runs of the 300-row/four-variable fixture, maximum lag two. Both returned six edges. This historical approximately 10.1x result applies only to that workload, not CDNOTS, GRACE, larger inputs or GPU execution.

Timing excluded parsing, imports and serialization. CPU: AMD EPYC 9V74. Raw measurements are in `benchmarks/single-cpu.json`; `examples/benchmark.rs` repeats the native workload.

## ParCorr covariance reuse and GRACE profiling

The 2026-10-24 follow-up on an Intel Xeon Platinum 8573C passed 100 CPU tests,
89 tests without default features, strict Clippy and formatting. All 24
before/after discovery workloads retained every discrete result field; the
largest absolute floating-point difference was `2.42e-12`. Complete-data CDNOTS
on 5,000 rows/four variables at lag two improved from 69.306 ms to 1.423 ms.
Missing-data tests retain their own complete-row subsets and the existing
SVD fallback.

GRACE model and optimizer source are unchanged. A separate profiling example
splits forward, loss, backward and Adam timing, and verifies five instrumented
updates exactly against the public implementation before measuring. The full
report, raw measurements and reproduction instructions are in
[optimization measurements](benchmarks/optimization.md).

## Scope of evidence

These are regression and numerical comparisons, not a guarantee of causal recovery or bitwise equivalence on every dataset. Conditioning budgets, near-singular inputs, rank ties, ordering and random streams can affect decisions. See COMPATIBILITY.md for precise differences and remaining upstream subsystems.
