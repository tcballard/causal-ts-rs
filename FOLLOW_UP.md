# Development handoff

## 1. Fix CI batch performance

CiEngine::batch_test prepares covariance for all data columns when multiple queries exist. Prepare only columns used by the queries. Use the scalar path when preparation costs more than it saves.

Preserve query order, per-test missing-row selection, singular-matrix fallback, validation and existing discovery covariance reuse. Add no code comments.

Run the CPU tests. Compare scalar and cached results. Measure two queries using three columns from 512 rows and 1,024 columns. Confirm the discovery speed gains remain.

## 2. Reduce GRACE backward cost

Start after task 1 passes. Profile operations and allocations inside Candle backward. Identify the largest costs. Change one cause at a time and measure each change.

Preserve model, loss, gradients and Adam behavior. Retain CPU, CUDA and Metal support. Avoid Adam changes unless new measurements justify them. Add no code comments.

Run neural reference and CPU tests. Compare predictions, gradients and updates using identical weights and gate noise. Repeat phase profiles and measure complete discovery time. State which GPU checks were run.

For each task, deliver source, tests and before/after measurements. Neither task is implemented in this import.
