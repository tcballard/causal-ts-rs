# Compatibility contract

Reference: `bloomberg/causal-ts` at `0d1fae061b2445a51d0c8b75fc565d897dd54fb7`, package 0.27.0. Rust port version: 0.2.0.

## Discovery coverage

| Subsystem | Status |
| --- | --- |
| CEDAR, including MCI pruning | Implemented; 25 upstream discovery comparisons |
| CDNOTS | Implemented; 18 upstream endpoint/binary/mark comparisons |
| CDNOTS+ | Implemented; 17 upstream endpoint/binary/mark comparisons |
| GRACE gated refinement | Implemented, including three encoder choices, Gaussian decoders, L0/lag penalties and Adam |
| GRACE cross-validation and stability selection | Implemented, with temporal validation splits, subsamples, warm starts and lag budgets |
| CPU partial correlation and GCMI | Implemented |
| Tensor CI backends | CPU/CUDA/Metal covariance computation; CPU filtering, ranking and precision solves |
| GPU neural training | Optional Candle CUDA/Metal backends; hardware execution unverified |
| KCI, SplitKCI, RCOT, k-NN, residual-score and signature tests | Not implemented |
| LassoCV lag selection, VAR-EM imputation, ancestral/path constraints, discrete-test wrappers | Not implemented |
| Effects, regimes, confounder discovery, notebooks and plotting adapters | Not implemented; DOT export is available |

The `ConditionalIndependence` trait is the extension point for further CI families. JSON rejects unknown fields instead of silently accepting unsupported upstream arguments.

## CEDAR

Preserves window alignment, partial-distance-correlation target residualization, lag significance, BH adjustment, descending candidate importance, both SyPI conditions, AR augmentation, required-edge protection and stable MCI removal. C-node exclusions and the optional target-only behaviour follow upstream. Target-only discovery can differ from filtering a full discovery result because it does not estimate every target's parents before MCI.

The default AR(1) self-loops are assumptions; `assume_ar1=false` selects order using BIC. Distance correlation uses an O(n log n) scalar distance-product implementation with O(n) workspace per pair.

## CDNOTS and CDNOTS+

CDNOTS performs the full-window PC search with stable or immediate removals. Neighbor-strength ordering, bounded conditioning combinations, recorded separating-set unions, lag copies and the default full majority vote follow upstream. It includes classic/sepset alternatives, collider conflict priorities 0–2, cycle-aware collider reselection, all three Meek rules, kernel sink orientation, and optional lag verification.

CDNOTS+ uses a separate per-variable screen (one combination per depth), stable MCI pruning with cause-parent shifts out to twice the maximum lag, and MCI majority voting. Its default alpha is 0.01, collider priority 1 and orientation margin 0.1. Plain CDNOTS defaults to 0.05, priority 2 and margin zero.

Results retain the entire endpoint matrix plus `[cause,effect,lag]` marks. Binary rendering is deliberately lossy: plain CDNOTS keeps unresolved lag-zero pairs as symmetric adjacency; CDNOTS+ drops them. Both drop conflicting pairs. DOT and CSV exports preserve marks.

The p-value matrix aggregates the **minimum** skeleton p-value, following upstream. `ci_tests` contains test records; `n_tests` counts test requests including orientation votes. Rust completes each collider vote, whereas upstream can stop a vote early, so orientation test counts need not match. Required edges remain protected; direct forbidden/required constraints are supported. Indirect ancestor/path knowledge is not implemented.

With a gapped CDNOTS+ lag list, Rust constructs the deeper MCI array using actual lag values and keeps all sample rows aligned. The pinned upstream MCI implementation extends by the number of slices instead, which can misidentify gapped columns. Gapped CDNOTS+ has invariant tests, not an upstream-equality claim. Sparse plain-CDNOTS output matches the included upstream fixture. Lag-copy replication in the window follows upstream slice replication.

The nonstationarity calculation uses f64 Cholesky solves of the regularized kernels. Upstream uses f32 Torch Hermitian pseudoinverses. Well-conditioned fixture scores agree within 1e-6; near-equal sink scores can still change the chosen orientation. Non-finite/constant score inputs cause abstention. Kernel orientation remains CPU-only and has quadratic storage and cubic dense-solve cost.

## GRACE

Preserves chronological window layout (internal lag order L..0), per-variable standardization, Hard Concrete sampling/stretch/clipping, deterministic gates, skeleton initialization/masking, self-loop exclusion, encoder aggregation and independent SiLU decoders. Encoders can use a linear scalar, fixed `[x,x²,|x|]` basis, or a shared MLP with cause/lag embeddings. Gaussian NLL, normalized L0 penalty, lag-group penalty, coupled Adam weight decay and stopping on epoch loss are implemented.

The gated entry point uses the CDNOTS binary skeleton and the upstream lambda formula, a supplied lambda, or a temporal holdout search. Stability mode includes lambda sweeps, random window subsamples, warm starts, early termination after zero-edge runs, informative-run scoring, per-pair lag pruning and the upstream expected-false-positive expression. That expression is exposed as an upstream diagnostic, not a calibrated guarantee for arbitrary data.

Parameter layouts and names match upstream. Three transferred-weight fixtures compare deterministic and sampled predictions, losses, gate values, every parameter gradient and one Adam update. Initialization, shuffling and gate draws use ChaCha8; full training is reproducible within Rust, but matching numeric seeds across Rust/PyTorch do not imply identical parameters, stopping epochs or learned graphs. Identical weights and noise are used for numerical parity tests.

GRACE rejects missing/infinite input. It does not silently impute. `include_c_in_model` is supported for gated refinement; C variables are removed from its final discovery graph. The neural mask still follows `include_instantaneous`, including for C inputs. Low-level model state export/import is available; high-level discovery returns graphs and diagnostics, not a Python Lightning trainer/checkpoint.

## Numerical and API differences

| Area | Rust behaviour |
| --- | --- |
| Ill-conditioned partial correlation | Correlation guard at 1e-10, then f64 SVD residualization; upstream falls back to f32 least squares |
| Constant/residual-degenerate inputs | Statistic zero, p-value one and explicit `degenerate` status |
| Insufficient complete observations | Null p-value and `insufficient_samples`; CEDAR does not accept the test, CDNOTS retains untestable candidates and reports a warning |
| GCMI ties | Deterministic average ranks; upstream usually jitters and uses ordinal ranks; oracle comparisons disable jitter |
| Randomization | Seeded ChaCha8, rather than NumPy/PyTorch random streams |
| Missing data in permutation tests | Complete pairs are selected first; circular shifts no longer preserve the spacing of missing original rows |
| Tail probabilities | Student-t survival function avoids cancellation from 1-CDF |
| Tensor CI | Covariance accumulation on the chosen device, small solves on CPU; CUDA uses f64, Metal f32; condition-guard failures use native CPU CI |
| Explicit unavailable GPU | Returns an error; only `auto` permits fallback |
| Required/forbidden conflicts | Invalid or contradictory direct constraints are rejected |
| CEDAR edge evidence | Assumed/required edges have null edge-level statistical evidence |
| Evaluation | Directed lag-specific adjacency errors; reversed edge counts as removal plus addition; not CPDAG-equivalence scoring |

GPU CI can have transfer overhead and is not assumed faster for small conditioning sets. No GPU speedup or hardware parity is claimed. The independent source and hardware tests are included so that these can be checked on the target machine.
