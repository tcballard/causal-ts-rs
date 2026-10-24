# causal-ts-rs

An independent native Rust library and CLI for **CEDAR, CDNOTS, CDNOTS+ and GRACE** causal discovery in time series. Version 0.2 ports the remaining three discovery algorithms and adds optional CUDA and Metal backends. Python and PyTorch are not runtime dependencies.

The reference is Bloomberg's [Causal-TS](https://github.com/bloomberg/causal-ts), pinned to commit `0d1fae061b2445a51d0c8b75fc565d897dd54fb7`. The port includes numeric CSV input, JSON/CSV/DOT output, conditional-independence tests, graph evaluation, synthetic data and reproducible training.

## Run

Build with Rust 1.88 or newer; the delivered build was validated with Rust 1.99.1:

```sh
cargo build --release --locked
cargo install --path . --locked
causal-ts discover --algorithm cedar --input tests/fixtures/var.csv --max-lag 2 --output cedar.json
causal-ts discover --algorithm cdnots --input tests/fixtures/var.csv --max-lag 2 --output cdnots.json
causal-ts discover --algorithm cdnots+ --input tests/fixtures/var.csv --max-lag 2 --output cdnots-plus.json
causal-ts discover --algorithm grace --input tests/fixtures/var.csv --max-lag 2 --seed 42 --output grace.json
```

`cedar` remains the default. CPU builds need no external BLAS, Python, PyTorch or GPU installation. For a smaller build containing the statistical discovery algorithms only, use `--no-default-features`.

Each CSV row is one time step; headers are unique variable names. Remove textual date columns before discovery. Blank, `NA`, `NaN` and `null` values are handled per test by the statistical algorithms. GRACE requires finite values. Infinite values and ragged rows are rejected.

## Algorithms

| Algorithm | Implemented behaviour |
| --- | --- |
| CEDAR | Lag selection, SyPI conditions, autoregressive augmentation, multiple lags, MCI pruning, C covariates, constraints and target discovery |
| CDNOTS | Stable and sequential PC skeleton search, temporal constraints, majority/classic collider rules, conflict resolution, Meek rules, kernel nonstationarity orientation and lag verification |
| CDNOTS+ | Per-variable PC screening, deeper-window MCI pruning, shifted-parent majority voting and the CDNOTS orientation pipeline |
| GRACE gated | CDNOTS or supplied skeleton, Hard Concrete gates, linear/basis/nonlinear encoders, independent Gaussian decoders, Adam, early stopping and optional lambda cross-validation |
| GRACE stability | Lambda sweeps, subsampling, warm starts, informative-run scoring and per-pair lag budgets |

## Configuration and output

```sh
causal-ts config --algorithm cdnots+ > cdnots-plus-config.json
causal-ts discover --algorithm cdnots+ --input data.csv --config cdnots-plus-config.json
causal-ts config --algorithm grace > grace-config.json
causal-ts discover --algorithm grace --input data.csv --config grace-config.json --epochs 150
causal-ts discover --algorithm grace --mode stability --input data.csv --config examples/grace-stability.json
causal-ts discover --algorithm cdnots --input data.csv --no-c --format dot --output graph.dot
causal-ts discover --algorithm cedar --input data.csv --test gcmi --estimate-ar-order --max-lag 3
causal-ts discover --algorithm cedar --input data.csv --covariates covariates.csv
```

Explicit CLI options override JSON. Unknown configuration fields and algorithm-incompatible options fail with an error. CDNOTS+ shares CDNOTS options under its `discovery` field; omitted fields use CDNOTS+ defaults:

```json
{
  "discovery": {"max_lag": 3, "include_c": false, "alpha": 0.01},
  "max_conds_py": 5,
  "max_conds_px": 3
}
```

GRACE configuration has a `model` object for network dimensions, encoder choice, gate distribution, learning rate, weight decay and lag penalties. `lambda_l0` overrides the gated mode's formula, or `lambda_cv: true` selects it on a temporal validation split. Stability mode uses `lambda_grid`. `--skeleton graph.json` supplies an existing binary graph with matching ordered variable names and lags.

All dense output arrays use `[cause][effect][lag]`, with lag zero meaning the current time. With an explicit `lag_list`, absent lag slots are empty/zero and `lags` records the tested lags.

**For CDNOTS and CDNOTS+, use `marks` as the authoritative graph.** `-->` is directed, `<--` is its contemporaneous mirror, `o-o` is unresolved, `x-x` is conflicting, and an empty string is absent. `endpoint_matrix` preserves the entire embedded window. The binary `graph` follows upstream rendering: CDNOTS keeps unresolved adjacency as symmetric ones; CDNOTS+ drops it. Both drop conflicts. `CdnotsResult::to_binary` lets callers choose these policies explicitly. DOT and edge CSV retain the uncertainty marks.

CDNOTS p-value matrices aggregate the minimum recorded skeleton p-value, as upstream does; they are not adjusted edge confidence levels. GRACE gate values and stability frequencies are scores, not p-values. CEDAR retains its edge-level `origin`, statistic and p-value evidence.

## CPU and GPU backends

The CPU implementation is the tested default. Neural computation uses [Candle](https://github.com/huggingface/candle). CUDA requires its toolkit and driver; Metal requires macOS and the Apple SDK.

```sh
cargo build --release --locked --features cuda
causal-ts discover --algorithm grace --input data.csv --backend cuda
causal-ts discover --algorithm cdnots+ --input data.csv --ci-backend cuda

cargo build --release --locked --features metal
causal-ts discover --algorithm grace --input data.csv --backend metal
causal-ts ci-test --input data.csv --x X0 --y X1 --given X2 --ci-backend metal
```

`--backend` selects GRACE's tensor device. `--ci-backend` independently selects tensor covariance computation for partial correlation/GCMI in any algorithm. Ranking, complete-case filtering and small precision-matrix inversions remain on the CPU; ill-conditioned CI tests use the native SVD fallback. CUDA/CPU tensor CI uses f64, Metal uses f32 with a stricter condition guard. Lag scoring and CDNOTS kernel orientation currently remain on the CPU.

An explicit unavailable device produces an error. `auto` permits fallback. **CUDA and Metal hardware execution has not been validated in this environment.** The CPU tensor path, unavailable-device errors and the standard CPU algorithms have been tested. See `tests/accelerator.rs` for opt-in hardware checks.

## Rust API

```rust
use causal_ts::{run_cdnots_plus, CdnotsPlusConfig, CiEngine, TimeSeries};
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = TimeSeries::from_csv(File::open("data.csv")?)?;
    let mut config = CdnotsPlusConfig::default();
    config.discovery.max_lag = 2;
    let result = run_cdnots_plus(&data, &CiEngine::default(), &config)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
```

`run_cedar`, `run_cdnots`, `run_cdnots_plus` and `grace::run_grace` expose the four algorithms. Implement `ConditionalIndependence` for a custom CI test. `CiEngine::test_indices`/`batch_test` expose CPU CI tests. `accelerated::TensorCiEngine` implements the same trait on a selected tensor backend.

Native ParCorr reuses a centered, normalized covariance matrix within each CDNOTS/CDNOTS+ data window, including GRACE's CI skeleton search. `batch_test` also prepares one matrix for multiple tests. A test uses this cache only when all its selected columns are complete; otherwise it retains per-test complete-row filtering. Ill-conditioned subsets retain the existing SVD fallback. CEDAR's varying windows, GCMI and tensor CI keep their existing paths. Custom CI engines can opt into the window-scoped `ConditionalIndependence::prepare` hook; its default preserves scalar testing.

`grace::GatedModel` exposes forward computation, deterministic or sampled gates, loss, gradients, and validated state export/import. `grace::model::Adam` exposes training updates. Parameter names and layouts match the upstream neural model; the test fixtures transfer identical parameters between implementations.

## Verification and scope

```sh
cargo test --locked
cargo test --locked --no-default-features
cargo clippy --locked --all-targets -- -D warnings
cargo fmt --check
```

CEDAR and both CDNOTS algorithms are compared with unmodified upstream outputs. GRACE forward results, gradients and Adam updates are compared using identical weights and random gate samples. Full training uses Rust's ChaCha8 random streams, so a shared numeric seed does not promise the same final graph as PyTorch.

This release ports the discovery algorithms above, not every upstream utility. Additional CI families, VAR-EM imputation, ancestor-path constraints, LassoCV selection, effect estimation, regime analysis, confounder discovery and plotting integrations remain outside the port. [COMPATIBILITY.md](COMPATIBILITY.md) and [VALIDATION.md](VALIDATION.md) describe the tested scope and remaining differences.

## Performance and follow-up work

See [optimization measurements](benchmarks/optimization.md) and [the development handoff](FOLLOW_UP.md). This is an independent project and is not affiliated with or endorsed by Bloomberg.
