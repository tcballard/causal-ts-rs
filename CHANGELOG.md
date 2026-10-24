# Changelog

## Unreleased

- Reuse native ParCorr covariance within immutable CDNOTS/CDNOTS+ windows and multi-test batches, preserving per-test missing-row selection and singular-matrix fallbacks.
- Add prepared-CI regression coverage and a GRACE profiling example that separates forward, loss, backward and Adam without changing the production neural implementation.
- Record before/after discovery measurements and GRACE phase profiles in the accompanying optimization report.

## 0.2.0

Added CDNOTS, then CDNOTS+, then GRACE, with independent upstream comparisons for each stage.

- CDNOTS window skeletons, temporal constraints, collider voting/conflicts, cycle handling, Meek rules, nonstationarity orientation and marked graph exports.
- CDNOTS+ per-variable screening, MCI pruning, extended parent windows and majority orientation.
- GRACE trainable Hard Concrete gates, three encoder variants, Gaussian decoders, Adam, early stopping, lambda selection, stability sweeps and model state APIs.
- Native tensor backends for CPU, optional CUDA and optional Metal; tensor covariance CI can be selected separately from neural computation.
- `--algorithm`, algorithm-specific configuration, GRACE training CLI and explicit device-selection errors.
- 95 CPU tests, 60 upstream discovery comparisons, three transferred-weight neural fixtures, and two opt-in GPU hardware tests.
- Statistical-only builds remain available with `--no-default-features`.

CUDA/Metal hardware, platform builds outside Linux, and final trained-graph equality across Rust/PyTorch random streams remain unverified. See COMPATIBILITY.md for remaining upstream subsystems and numerical differences.

## 0.1.0

Initial native CEDAR library/CLI, partial correlation and GCMI, lag selection, synthetic data and graph evaluation; 45 CPU tests and 25 upstream discovery comparisons.
