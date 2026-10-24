# Completed CI batch and GRACE optimization bundle

[Download the ZIP](https://github.com/tcballard/causal-ts-rs/raw/refs/heads/delivery/ci-batch-grace-2026-10-24/downloads/causal-ts-source-results.zip)

The archive contains the completed source, tests, golden fixtures, before/after measurements, raw profiles, reproduction scripts and a Linux CPU executable. Start with `causal-ts-batch-grace/README.md` and `iteration2/REPORT.md` inside the archive. The completed implementation is under its `causal-ts-rs/` directory.

The archive is based on `causal-ts-optimized-2026-10-24.zip`. The imported source in this repository remains the earlier baseline; this branch delivers the completed work as a downloadable artifact.

- Required CI batch: 23.980151 ms to 0.022637 ms; requested allocation volume 19,008,616 to 70,664 bytes. Discovery covariance gains are retained.
- GRACE backward: 2.1–2.5 times faster, with 30–32% fewer requested allocation bytes. Complete discovery: 2.01–2.07 times faster.
- 104 CPU tests and 91 statistical-only tests passed. Prediction, gradient and update reference checks passed. Adam is unchanged.
- CUDA/Metal feature builds and hardware tests were not run; the environment lacks the required hardware and SDKs.

Duplicate benchmark executables are omitted from this compact download. Their source, rebuild scripts and recorded hashes are included. The current CLI is included.

SHA-256: `146e8ed5ffe8317da23b4bd8489514c48134f4256d49211690296662fa5ebed7`
