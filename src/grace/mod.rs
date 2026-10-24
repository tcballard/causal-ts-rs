pub mod model;

use crate::{
    backend::Backend,
    cdnots::{run_cdnots, CdnotsConfig},
    ci::ConditionalIndependence,
    data::TimeBasis,
    graph::{Cube, Edge, Graph},
    Error, Result, TimeSeries, UPSTREAM_REVISION,
};
use candle_core::Tensor;
pub use model::{prepare_data, GatedModel, ModelConfig, TensorData};
use rand::{seq::SliceRandom, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum GraceMode {
    #[default]
    Gated,
    Stability,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GraceConfig {
    pub max_lag: usize,
    pub mode: GraceMode,
    pub backend: Backend,
    pub normalize: bool,
    pub include_c: bool,
    pub include_c_in_model: bool,
    pub c_preset: TimeBasis,
    pub ci_alpha: f64,
    pub ci_max_degree: Option<usize>,
    pub gate_threshold: f64,
    pub lambda_l0: Option<f64>,
    pub lambda_cv: bool,
    pub cv_lambdas: usize,
    pub cv_epochs: usize,
    pub cv_warm_start_epochs: usize,
    pub cv_val_fraction: f64,
    pub max_epochs: usize,
    pub batch_size: Option<usize>,
    pub patience: usize,
    pub seed: u64,
    pub model: ModelConfig,
    pub lambda_grid: Option<Vec<f64>>,
    pub n_subsamples: usize,
    pub subsample_fraction: f64,
    pub stability_threshold: f64,
    pub stability_use_ci_skeleton: bool,
    pub use_skeleton_mask: bool,
    pub warm_start: bool,
    pub warm_start_epochs: usize,
    pub early_terminate_zero: usize,
}

impl Default for GraceConfig {
    fn default() -> Self {
        Self {
            max_lag: 1,
            mode: GraceMode::Gated,
            backend: Backend::Cpu,
            normalize: true,
            include_c: true,
            include_c_in_model: false,
            c_preset: TimeBasis::Linear,
            ci_alpha: 0.05,
            ci_max_degree: None,
            gate_threshold: 0.5,
            lambda_l0: None,
            lambda_cv: false,
            cv_lambdas: 8,
            cv_epochs: 100,
            cv_warm_start_epochs: 75,
            cv_val_fraction: 0.2,
            max_epochs: 150,
            batch_size: None,
            patience: 20,
            seed: 42,
            model: ModelConfig::default(),
            lambda_grid: None,
            n_subsamples: 1,
            subsample_fraction: 0.8,
            stability_threshold: 0.6,
            stability_use_ci_skeleton: false,
            use_skeleton_mask: false,
            warm_start: true,
            warm_start_epochs: 75,
            early_terminate_zero: 3,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingRun {
    pub lambda: f64,
    pub epochs: usize,
    pub training_loss: f64,
    pub validation_loss: Option<f64>,
    pub active_edges: usize,
    pub loss_history: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraceResult {
    pub schema_version: u32,
    pub algorithm: String,
    pub upstream_revision: String,
    pub backend: String,
    pub n_observations: usize,
    #[serde(flatten)]
    pub causal_graph: Graph,
    pub edges: Vec<Edge>,
    pub gate_values: Vec<Vec<Vec<f64>>>,
    pub stability_scores: Option<Vec<Vec<Vec<f64>>>>,
    pub skeleton: Option<Graph>,
    pub lambda_l0: Option<f64>,
    pub training_runs: Vec<TrainingRun>,
    pub cross_validation: Vec<TrainingRun>,
    pub informative_runs: usize,
    pub expected_false_positives_bound: Option<f64>,
    pub config: GraceConfig,
}

pub fn compute_lambda(d: usize, samples: usize, density: f64) -> Result<f64> {
    if d == 0 || samples == 0 || !(0.0..=1.0).contains(&density) {
        return Err(Error::InvalidConfig(
            "lambda formula requires positive dimensions and density in [0,1]".into(),
        ));
    }
    Ok((0.007 + 0.16 * density)
        .max(1.0 / d as f64 + 4.0 / samples as f64 - 0.4 * density.powf(0.8)))
}

fn linspace(low: f64, high: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![low];
    }
    (0..n)
        .map(|i| low + (high - low) * i as f64 / (n - 1) as f64)
        .collect()
}

pub fn smart_lambda_grid(n: usize) -> Result<Vec<f64>> {
    if n == 0 {
        return Err(Error::InvalidConfig("lambda grid must be nonempty".into()));
    }
    if n <= 3 {
        return Ok(linspace(-3.0, 0.0, n)
            .iter()
            .map(|v| 10f64.powf(*v))
            .collect());
    }
    let nt = ((n as f64 * 0.6) as usize).max(2);
    let explore = linspace(-3.0, 0.0, n - nt + 2);
    let mut all = vec![-3.0, 0.0];
    all.extend_from_slice(&explore[1..explore.len() - 1]);
    all.extend(linspace(-2.0, -0.5, nt));
    all.sort_by(f64::total_cmp);
    all.dedup();
    let values: Vec<_> = all.into_iter().map(|v| 10f64.powf(v)).collect();
    if values.len() <= n {
        Ok(values)
    } else {
        Ok(linspace(0.0, (values.len() - 1) as f64, n)
            .iter()
            .map(|v| values[v.round_ties_even() as usize])
            .collect())
    }
}

struct TrainSettings {
    epochs: usize,
    batch: usize,
    patience: usize,
    lambda: f64,
}

fn train(
    model: &GatedModel,
    data: &Tensor,
    validation: Option<&Tensor>,
    settings: &TrainSettings,
    rng: &mut ChaCha8Rng,
) -> Result<TrainingRun> {
    let n = data.dim(0)?;
    let steps = n.div_ceil(settings.batch);
    let mut opt = model::Adam::new();
    let mut indices: Vec<u32> = (0..n as u32).collect();
    let mut history = vec![];
    let mut best = f64::INFINITY;
    let mut stale = 0;
    let mut val_loss = None;
    for _ in 0..settings.epochs {
        indices.shuffle(rng);
        let mut total = 0.0;
        for batch in indices.chunks(settings.batch) {
            let idx = Tensor::from_slice(batch, batch.len(), &model.device)?;
            let x = data.index_select(&idx, 0)?;
            let u = model::uniform_noise(model.num_vars, model.max_lag + 1, rng, &model.device)?;
            let (mu, sigma) = model.forward(&x, Some(&u))?;
            let loss = model.loss(&x, &mu, &sigma, settings.lambda, steps, true)?;
            let value = loss.to_scalar::<f32>()? as f64;
            if !value.is_finite() {
                return Err(Error::Numerical(
                    "GRACE training produced a non-finite loss".into(),
                ));
            }
            total += value * batch.len() as f64;
            opt.step(model, &loss)?;
        }
        let avg = total / n as f64;
        history.push(avg);
        if let Some(v) = validation {
            let (mu, sigma) = model.forward(v, None)?;
            val_loss = Some(
                model
                    .loss(v, &mu, &sigma, settings.lambda, steps, false)?
                    .to_scalar::<f32>()? as f64,
            );
        }
        let monitored = val_loss.unwrap_or(avg);
        if monitored < best {
            best = monitored;
            stale = 0;
        } else {
            stale += 1;
        }
        if stale >= settings.patience.max(1) {
            break;
        }
    }
    let active = model
        .gate_values()?
        .values
        .iter()
        .filter(|&&v| v > 0.5)
        .count();
    Ok(TrainingRun {
        lambda: settings.lambda,
        epochs: history.len(),
        training_loss: *history.last().unwrap(),
        validation_loss: val_loss,
        active_edges: active,
        loss_history: history,
    })
}

fn prune_lags(active: &mut Cube<u8>, values: &Cube<f64>, config: &ModelConfig, reverse_ties: bool) {
    for c in 0..active.dimension {
        for e in 0..active.dimension {
            let k = config
                .per_pair_lag_budget
                .as_ref()
                .map_or(config.max_lags_per_pair, |b| b[c][e]);
            let mut lags: Vec<_> = (0..active.lag_count)
                .filter(|&l| active[(c, e, l)] != 0)
                .collect();
            lags.sort_by(|&a, &b| {
                values[(c, e, b)]
                    .total_cmp(&values[(c, e, a)])
                    .then_with(|| if reverse_ties { b.cmp(&a) } else { a.cmp(&b) })
            });
            for lag in lags.into_iter().skip(k) {
                active[(c, e, lag)] = 0;
            }
        }
    }
}

pub fn run_grace<T: ConditionalIndependence + ?Sized>(
    data: &TimeSeries,
    ci: &T,
    config: &GraceConfig,
) -> Result<GraceResult> {
    run_grace_with_skeleton(data, None, ci, config)
}

pub fn run_grace_with_skeleton<T: ConditionalIndependence + ?Sized>(
    data: &TimeSeries,
    provided: Option<&Graph>,
    ci: &T,
    config: &GraceConfig,
) -> Result<GraceResult> {
    let cfg = config;
    if cfg.max_lag >= data.nrows()
        || data.nrows() > u32::MAX as usize
        || data.nrows() - cfg.max_lag < 10
        || cfg.max_epochs == 0
        || cfg.batch_size == Some(0)
        || cfg.n_subsamples == 0
        || !(cfg.subsample_fraction > 0.0 && cfg.subsample_fraction <= 1.0)
        || !(cfg.gate_threshold > 0.0 && cfg.gate_threshold <= 1.0)
        || !(cfg.stability_threshold > 0.0 && cfg.stability_threshold <= 1.0)
        || cfg.lambda_l0.is_some_and(|v| v < 0.0 || !v.is_finite())
        || cfg.warm_start_epochs == 0
        || cfg.cv_epochs == 0
        || cfg.cv_warm_start_epochs == 0
        || cfg.cv_lambdas == 0
        || !(0.0..1.0).contains(&cfg.cv_val_fraction)
        || cfg.cv_val_fraction == 0.0
        || !(cfg.ci_alpha > 0.0 && cfg.ci_alpha < 1.0)
        || cfg
            .lambda_grid
            .as_ref()
            .is_some_and(|v| v.is_empty() || v.iter().any(|v| *v < 0.0 || !v.is_finite()))
    {
        return Err(Error::InvalidConfig(
            "invalid GRACE data length, training schedule, penalty or threshold".into(),
        ));
    }
    if cfg.include_c_in_model && (!cfg.include_c || matches!(cfg.mode, GraceMode::Stability)) {
        return Err(Error::InvalidConfig(
            "C in the neural model requires gated mode and include_c".into(),
        ));
    }
    if matches!(cfg.mode, GraceMode::Stability) && (cfg.lambda_cv || cfg.lambda_l0.is_some()) {
        return Err(Error::InvalidConfig(
            "stability mode uses lambda_grid; lambda_cv and lambda_l0 apply to gated mode".into(),
        ));
    }
    if matches!(cfg.mode, GraceMode::Gated) && cfg.lambda_grid.is_some() {
        return Err(Error::InvalidConfig(
            "lambda_grid applies to stability mode".into(),
        ));
    }
    let device = cfg.backend.device()?;
    let frame = if cfg.include_c_in_model {
        data.with_time_basis(cfg.c_preset)?
    } else {
        data.clone()
    };
    let d = frame.ncols();
    let observed = data.ncols();
    let windows = prepare_data(&frame, cfg.max_lag, cfg.normalize, &device)?;
    let n = windows.dim(0)?;
    let batch = cfg.batch_size.unwrap_or_else(|| {
        if cfg.model.normalize_l0 {
            (n / 8).clamp(32, 256)
        } else {
            (n / 15).clamp(16, 64)
        }
    });
    let mut max_pvalues = None;
    let skeleton = if let Some(g) = provided {
        g.validate()?;
        if g.var_names != frame.names() || g.max_lag != cfg.max_lag {
            return Err(Error::InvalidConfig(
                "skeleton variables and lags must match the neural model".into(),
            ));
        }
        Some(g.clone())
    } else if matches!(cfg.mode, GraceMode::Gated) {
        let cd = run_cdnots(
            data,
            ci,
            &CdnotsConfig {
                max_lag: cfg.max_lag,
                alpha: cfg.ci_alpha,
                include_c: cfg.include_c,
                c_preset: cfg.c_preset,
                max_degree: cfg.ci_max_degree,
                ..Default::default()
            },
        )?;
        Some(Graph {
            var_names: frame.names().to_vec(),
            max_lag: cfg.max_lag,
            graph: cd
                .causal_graph
                .graph
                .into_iter()
                .take(d)
                .map(|r| r.into_iter().take(d).collect())
                .collect(),
        })
    } else if cfg.stability_use_ci_skeleton {
        let (s, p) = crate::cdnots::ci_skeleton(
            data,
            ci,
            &CdnotsConfig {
                max_lag: cfg.max_lag,
                alpha: cfg.ci_alpha,
                include_c: cfg.include_c,
                c_preset: cfg.c_preset,
                max_degree: Some(cfg.ci_max_degree.unwrap_or(1)),
                ..Default::default()
            },
        )?;
        max_pvalues = Some(p);
        Some(s)
    } else {
        None
    };
    if cfg.use_skeleton_mask && skeleton.is_none() {
        return Err(Error::InvalidConfig(
            "use_skeleton_mask requires a skeleton".into(),
        ));
    }
    let mut model_cfg = cfg.model.clone();
    if let Some(p) = max_pvalues {
        if model_cfg.lambda_lag_group > 0.0 && model_cfg.per_pair_lag_budget.is_none() {
            model_cfg.per_pair_lag_budget = Some(
                (0..d)
                    .map(|c| {
                        (0..d)
                            .map(|e| {
                                (0..=cfg.max_lag)
                                    .filter(|&l| p[(c, e, l)] <= cfg.ci_alpha / 10.0)
                                    .count()
                                    .max(1)
                                    .min(model_cfg.max_lags_per_pair)
                            })
                            .collect()
                    })
                    .collect(),
            );
        }
    }
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
    let mut cv = vec![];
    let mut runs = vec![];
    let mut informative = 0;
    let mut bound = None;
    let mut selected_lambda = None;
    let initialize = |seed| -> Result<GatedModel> {
        let mut m = GatedModel::new(d, cfg.max_lag, &model_cfg, &device, seed)?;
        if let Some(s) = &skeleton {
            m.initialize_skeleton(
                &s.graph,
                matches!(cfg.mode, GraceMode::Gated) || cfg.use_skeleton_mask,
            )?;
        }
        Ok(m)
    };
    let (gates, scores, binary) = match cfg.mode {
        GraceMode::Gated => {
            let skeleton = skeleton.as_ref().unwrap();
            let candidates = d * d * (cfg.max_lag + 1) - d;
            let density = if candidates == 0 {
                0.0
            } else {
                skeleton
                    .graph
                    .iter()
                    .flatten()
                    .flatten()
                    .map(|&v| v as usize)
                    .sum::<usize>() as f64
                    / candidates as f64
            };
            let mut lambda = cfg
                .lambda_l0
                .unwrap_or(compute_lambda(d, data.nrows(), density)?);
            if cfg.lambda_cv && cfg.lambda_l0.is_none() {
                let val_n = ((n as f64 * cfg.cv_val_fraction) as usize)
                    .max(1)
                    .min(n - 1);
                let training = windows.narrow(0, 0, n - val_n)?;
                let validation = windows.narrow(0, n - val_n, val_n)?;
                let mut previous = None;
                let mut best = f64::INFINITY;
                for (i, log) in linspace(-2.5, -0.3, cfg.cv_lambdas).iter().enumerate() {
                    let lam = 10f64.powf(*log);
                    let m = initialize(cfg.seed.wrapping_add(i as u64))?;
                    if let Some(state) = &previous {
                        m.load_state(state)?;
                    }
                    let report = train(
                        &m,
                        &training,
                        Some(&validation),
                        &TrainSettings {
                            epochs: if i == 0 {
                                cfg.cv_epochs
                            } else {
                                cfg.cv_warm_start_epochs
                            },
                            batch,
                            patience: cfg.patience.min(15),
                            lambda: lam,
                        },
                        &mut rng,
                    )?;
                    if report.validation_loss.unwrap() < best {
                        best = report.validation_loss.unwrap();
                        lambda = lam;
                    }
                    cv.push(report);
                    previous = Some(m.state()?);
                }
            }
            let m = initialize(cfg.seed)?;
            runs.push(train(
                &m,
                &windows,
                None,
                &TrainSettings {
                    epochs: cfg.max_epochs,
                    batch,
                    patience: cfg.patience,
                    lambda,
                },
                &mut rng,
            )?);
            let gates = m.gate_values()?;
            let mut binary = Cube::new(d, cfg.max_lag + 1, 0u8);
            for (b, &v) in binary.values.iter_mut().zip(&gates.values) {
                *b = u8::from(v >= cfg.gate_threshold);
            }
            selected_lambda = Some(lambda);
            (gates, None, binary)
        }
        GraceMode::Stability => {
            let mut grid = cfg.lambda_grid.clone().unwrap_or_else(|| {
                linspace(-3.0, 0.0, 20)
                    .iter()
                    .map(|v| 10f64.powf(*v))
                    .collect()
            });
            grid.sort_by(f64::total_cmp);
            let n_sub = ((n as f64 * cfg.subsample_fraction) as usize)
                .max(cfg.max_lag + 2)
                .min(n);
            let mut previous = None;
            let mut selections = vec![];
            let mut last_gates = Cube::new(d, cfg.max_lag + 1, 0.0);
            let mut zeros = 0;
            for lambda in grid {
                if cfg.early_terminate_zero > 0 && zeros >= cfg.early_terminate_zero {
                    break;
                }
                let mut all_zero = true;
                for _ in 0..cfg.n_subsamples {
                    let mut indices: Vec<u32> = (0..n as u32).collect();
                    if cfg.n_subsamples > 1 || cfg.subsample_fraction < 1.0 {
                        indices.shuffle(&mut rng);
                        indices.truncate(n_sub);
                    }
                    let subset = windows.index_select(
                        &Tensor::from_vec(indices.clone(), indices.len(), &device)?,
                        0,
                    )?;
                    let m = initialize(cfg.seed.wrapping_add(runs.len() as u64))?;
                    let warm = cfg.warm_start && previous.is_some();
                    if warm {
                        m.load_state(previous.as_ref().unwrap())?;
                    }
                    let mut report = train(
                        &m,
                        &subset,
                        None,
                        &TrainSettings {
                            epochs: if warm {
                                cfg.warm_start_epochs
                            } else {
                                cfg.max_epochs
                            },
                            batch,
                            patience: cfg.patience,
                            lambda,
                        },
                        &mut rng,
                    )?;
                    last_gates = m.gate_values()?;
                    let mut active = Cube::new(d, cfg.max_lag + 1, 0u8);
                    for (b, &v) in active.values.iter_mut().zip(&last_gates.values) {
                        *b = u8::from(v > 0.5);
                    }
                    prune_lags(&mut active, &last_gates, &model_cfg, true);
                    report.active_edges = active.values.iter().map(|&v| v as usize).sum();
                    if report.active_edges > 0 {
                        informative += 1;
                        all_zero = false;
                    }
                    selections.push(active);
                    runs.push(report);
                    previous = Some(m.state()?);
                }
                if all_zero {
                    zeros += 1;
                } else {
                    zeros = 0;
                }
            }
            let mut scores = Cube::new(d, cfg.max_lag + 1, 0.0);
            let mut total_selected = 0usize;
            for s in &selections {
                if s.values.iter().any(|&v| v != 0) {
                    for (value, &v) in scores.values.iter_mut().zip(&s.values) {
                        *value += v as f64;
                        total_selected += v as usize;
                    }
                }
            }
            if informative > 0 {
                for value in &mut scores.values {
                    *value /= informative as f64;
                }
            }
            bound = if informative == 0 {
                Some(0.0)
            } else if cfg.stability_threshold > 0.5 {
                Some(
                    (total_selected as f64 / informative as f64).powi(2)
                        / ((2.0 * cfg.stability_threshold - 1.0) * informative as f64),
                )
            } else {
                None
            };
            let mut binary = Cube::new(d, cfg.max_lag + 1, 0u8);
            for (b, &v) in binary.values.iter_mut().zip(&scores.values) {
                *b = u8::from(v >= cfg.stability_threshold);
            }
            prune_lags(&mut binary, &scores, &model_cfg, false);
            (last_gates, Some(scores), binary)
        }
    };
    let trim = |v: &Cube<f64>| -> Vec<Vec<Vec<f64>>> {
        (0..observed)
            .map(|c| {
                (0..observed)
                    .map(|e| (0..=cfg.max_lag).map(|l| v[(c, e, l)]).collect())
                    .collect()
            })
            .collect()
    };
    let mut edges = vec![];
    for c in 0..observed {
        for e in 0..observed {
            for l in 0..=cfg.max_lag {
                if binary[(c, e, l)] != 0 {
                    edges.push(Edge {
                        cause: data.names()[c].clone(),
                        effect: data.names()[e].clone(),
                        lag: l,
                        statistic: Some(scores.as_ref().unwrap_or(&gates)[(c, e, l)]),
                        p_value: None,
                        origin: if scores.is_some() {
                            "stability"
                        } else {
                            "gate"
                        }
                        .into(),
                    });
                }
            }
        }
    }
    let graph = (0..observed)
        .map(|c| {
            (0..observed)
                .map(|e| (0..=cfg.max_lag).map(|l| binary[(c, e, l)]).collect())
                .collect()
        })
        .collect();
    Ok(GraceResult {
        schema_version: 1,
        algorithm: "grace".into(),
        upstream_revision: UPSTREAM_REVISION.into(),
        backend: Backend::name(&device).into(),
        n_observations: data.nrows(),
        causal_graph: Graph {
            var_names: data.names().to_vec(),
            max_lag: cfg.max_lag,
            graph,
        },
        edges,
        gate_values: trim(&gates),
        stability_scores: scores.as_ref().map(trim),
        skeleton,
        lambda_l0: selected_lambda,
        training_runs: runs,
        cross_validation: cv,
        informative_runs: informative,
        expected_false_positives_bound: bound,
        config: cfg.clone(),
    })
}
