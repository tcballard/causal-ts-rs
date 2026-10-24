# Optimization measurements

Recorded on 24 October 2026 on an Intel Xeon Platinum 8573C, one CPU thread. These are historical measurements from the supplied optimization bundle, not GitHub Actions timings.

Native ParCorr covariance reuse reduced the 5,000-row, four-variable, lag-two CDNOTS case from 69.306 ms to 1.423 ms (48.71 times faster than the original Rust implementation). CDNOTS+ reduced from 12.472 ms to 1.062 ms. Construction of the cache is included. Each median uses 11 calls after two warmups; input parsing and serialization are excluded.

All 24 recorded discovery comparisons preserve discrete outputs. Across 110,141 numeric fields, maximum absolute difference is 2.4119595209981526e-12. Independent review confirmed these comparisons and passed all 100 CPU tests; two GPU tests were skipped.

GRACE model and optimizer code are unchanged. In phase profiles, backward propagation accounts for 64–69 percent of native training-step time and is 3.3–3.8 times slower than PyTorch on the tested shapes. Adam is already 3.1–3.7 times faster. Phase timings do not establish an end-to-end GRACE speed improvement.

The CSV files retain the supplied discovery and phase summaries. Complete raw outputs, baseline source, profiling inputs and reproduction scripts remain in the supplied causal-ts-optimized-2026-10-24.zip bundle. This repository imports the source and summary evidence, not that complete benchmark bundle.

Known regression: batch_test prepares all data columns even when queries use only a few. A separate review probe with 512 rows and 1,024 columns took 26.4 ms for two queries versus 0.025 ms through the old scalar path. See FOLLOW_UP.md before extending batch caching.

These results describe selected CPU workloads on one machine. GPU execution remains unverified.
