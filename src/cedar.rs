use crate::{
    ci::{CiResult, ConditionalIndependence},
    data::{Normalization, TimeBasis},
    graph::{Cube, Edge, EdgeConstraint, Graph},
    lag::{self, LagConfig, LagMethod, LagPvalueMethod, LagSelection},
    stats, Error, Result, TimeSeries, UPSTREAM_REVISION,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum MultiLagKeep {
    #[default]
    First,
    All,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum DirectionMethod {
    #[default]
    Dcor,
    CrossLag,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CedarConfig {
    pub max_lag: usize,
    pub alpha_cond1: f64,
    pub alpha_cond2: f64,
    pub lag_method: LagMethod,
    pub lag_pvalue_method: LagPvalueMethod,
    pub lag_alpha: f64,
    pub n_permutations: usize,
    pub seed: u64,
    pub normalize: Normalization,
    pub filter_children: bool,
    pub direction_threshold: f64,
    pub direction_method: DirectionMethod,
    pub prune: bool,
    pub alpha_mci: Option<f64>,
    pub sweep_all_lags: bool,
    pub include_lag0: bool,
    pub multi_lag: bool,
    pub multi_lag_keep: MultiLagKeep,
    pub target_var: Option<String>,
    pub forbidden: Vec<EdgeConstraint>,
    pub required: Vec<EdgeConstraint>,
    pub include_c: bool,
    pub c_preset: TimeBasis,
    pub include_autoreg: bool,
    pub assume_ar1: bool,
    pub lag_filter_frac: Option<f64>,
    pub fdr: bool,
}

impl Default for CedarConfig {
    fn default() -> Self {
        Self {
            max_lag: 1,
            alpha_cond1: 0.01,
            alpha_cond2: 0.01,
            lag_method: LagMethod::PartialDcor,
            lag_pvalue_method: LagPvalueMethod::TTest,
            lag_alpha: 0.05,
            n_permutations: 200,
            seed: 42,
            normalize: Normalization::Minmax,
            filter_children: false,
            direction_threshold: 1.2,
            direction_method: DirectionMethod::Dcor,
            prune: true,
            alpha_mci: None,
            sweep_all_lags: false,
            include_lag0: false,
            multi_lag: true,
            multi_lag_keep: MultiLagKeep::First,
            target_var: None,
            forbidden: Vec::new(),
            required: Vec::new(),
            include_c: false,
            c_preset: TimeBasis::Linear,
            include_autoreg: true,
            assume_ar1: true,
            lag_filter_frac: None,
            fdr: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PruneDecision {
    pub cause: String,
    pub effect: String,
    pub lag: usize,
    pub p_value: f64,
    pub removed: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CedarResult {
    pub schema_version: u32,
    pub algorithm: String,
    pub ci_test: String,
    pub n_observations: usize,
    pub upstream_revision: String,
    #[serde(flatten)]
    pub causal_graph: Graph,
    pub edges: Vec<Edge>,
    pub val_matrix: Vec<Vec<Vec<f64>>>,
    pub pvalue_matrix: Vec<Vec<Vec<f64>>>,
    pub lag_importance: Vec<Vec<Vec<f64>>>,
    pub lag_pvalues: Vec<Vec<Vec<f64>>>,
    pub selected_lags: Vec<Vec<Option<usize>>>,
    pub ar_order: Vec<usize>,
    pub n_tests: usize,
    pub pruned: bool,
    pub pruning: Vec<PruneDecision>,
    pub c_node_names: Vec<String>,
    pub warnings: Vec<String>,
    pub config: CedarConfig,
}

#[derive(Clone, Copy)]
struct Constraint {
    cause: usize,
    effect: usize,
    lag: Option<usize>,
}

fn matches(edges: &[Constraint], c: usize, e: usize, l: usize) -> bool {
    edges
        .iter()
        .any(|edge| edge.cause == c && edge.effect == e && edge.lag.is_none_or(|lag| lag == l))
}

fn resolve_constraints(
    edges: &[EdgeConstraint],
    data: &TimeSeries,
    original_d: usize,
    max_lag: usize,
) -> Result<Vec<Constraint>> {
    edges
        .iter()
        .map(|edge| {
            let cause = data.index(&edge.cause)?;
            let effect = data.index(&edge.effect)?;
            if edge.lag.is_some_and(|l| l > max_lag) || (cause == effect && edge.lag == Some(0)) {
                return Err(Error::InvalidConfig("constraint lag is invalid".into()));
            }
            if effect >= original_d {
                return Err(Error::InvalidConfig(
                    "time covariates cannot be effects".into(),
                ));
            }
            if cause >= original_d && edge.lag != Some(0) {
                return Err(Error::InvalidConfig(
                    "time covariate constraints must specify lag zero".into(),
                ));
            }
            Ok(Constraint {
                cause,
                effect,
                lag: edge.lag,
            })
        })
        .collect()
}

pub fn run_cedar(
    data: &TimeSeries,
    ci: &impl ConditionalIndependence,
    config: &CedarConfig,
) -> Result<CedarResult> {
    run_cedar_with_covariates(data, None, ci, config)
}

pub fn run_cedar_with_covariates(
    raw: &TimeSeries,
    covariates: Option<&TimeSeries>,
    ci: &impl ConditionalIndependence,
    config: &CedarConfig,
) -> Result<CedarResult> {
    validate(config, raw)?;
    let original_d = raw.ncols();
    let expanded = if let Some(c) = covariates {
        raw.with_covariates(c)?
    } else if config.include_c {
        raw.with_time_basis(config.c_preset)?
    } else {
        raw.clone()
    };
    let d = expanded.ncols();
    let include_c = d > original_d;
    let include_lag0 = config.include_lag0 || include_c;
    let data = expanded.normalized(config.normalize);
    let target = config
        .target_var
        .as_ref()
        .map(|s| raw.index(s))
        .transpose()?;
    let required = resolve_constraints(&config.required, &expanded, original_d, config.max_lag)?;
    let forbidden = resolve_constraints(&config.forbidden, &expanded, original_d, config.max_lag)?;
    for req in &required {
        if forbidden.iter().any(|f| {
            req.cause == f.cause
                && req.effect == f.effect
                && (req.lag.is_none() || f.lag.is_none() || req.lag == f.lag)
        }) {
            return Err(Error::InvalidConfig(
                "an edge is both required and forbidden".into(),
            ));
        }
    }
    let lag_config = LagConfig {
        max_lag: config.max_lag,
        method: config.lag_method,
        pvalue_method: config.lag_pvalue_method,
        alpha: config.lag_alpha,
        include_lag0,
        fdr: config.fdr,
        n_permutations: config.n_permutations,
        seed: config.seed,
        target,
    };
    let mut selection = lag::select_lags(&data, &lag_config)?;
    if let Some(frac) = config.lag_filter_frac {
        for e in 0..original_d {
            let peak = (usize::from(!include_lag0)..=config.max_lag)
                .map(|l| selection.importance[(e, e, l)])
                .fold(0.0, f64::max);
            for c in 0..d {
                if c != e
                    && peak > 0.0
                    && (usize::from(!include_lag0)..=config.max_lag)
                        .map(|l| selection.importance[(c, e, l)])
                        .fold(0.0, f64::max)
                        < frac * peak
                {
                    selection.selected[c][e] = None;
                    for l in 0..=config.max_lag {
                        selection.significant[(c, e, l)] = false;
                        selection.importance[(c, e, l)] = 0.0;
                        selection.pvalues[(c, e, l)] = 1.0;
                    }
                }
            }
        }
        selection = lag::reselect(selection.importance, selection.pvalues, &lag_config);
    }
    if include_c {
        for c in original_d..d {
            for e in 0..d {
                selection.selected[c][e] = Some(0);
                for l in 1..=config.max_lag {
                    selection.significant[(c, e, l)] = false;
                }
                if e < original_d {
                    selection.significant[(c, e, 0)] = true;
                }
            }
        }
        if !config.include_lag0 {
            for c in 0..original_d {
                for e in 0..original_d {
                    if selection.selected[c][e] == Some(0) {
                        selection.selected[c][e] = None;
                    }
                    selection.significant[(c, e, 0)] = false;
                }
            }
        }
    }
    if config.sweep_all_lags {
        for c in 0..d {
            for e in 0..d {
                if selection.selected[c][e].is_some() {
                    selection.selected[c][e] = Some(config.max_lag);
                }
                for l in 0..=config.max_lag {
                    selection.significant[(c, e, l)] =
                        (l > 0 || include_lag0) && (c < original_d || l == 0);
                }
            }
        }
    }
    let children = children_mask(&data, &selection, config, include_lag0);
    let st = config.max_lag + 1;
    let end = data.nrows() - config.max_lag - 1;
    let ar_order: Vec<_> = (0..original_d)
        .map(|col| {
            if config.assume_ar1 {
                1
            } else {
                estimate_ar_order(&data, col, st, end, config.max_lag)
            }
        })
        .collect();
    let targets: Vec<_> = target.map_or_else(|| (0..original_d).collect(), |t| vec![t]);
    let discovery = Discovery {
        data: &data,
        ci,
        config,
        selection: &selection,
        original_d,
        required: &required,
        forbidden: &forbidden,
        children: &children,
        ar_order: &ar_order,
    };
    let results: Vec<_> = targets
        .par_iter()
        .map(|&e| discovery.discover_target(e))
        .collect::<Result<_>>()?;
    let mut graph = Cube::new(d, config.max_lag + 1, 0u8);
    let mut values = Cube::new(d, config.max_lag + 1, 0.0);
    let mut pvalues = Cube::new(d, config.max_lag + 1, 1.0);
    let mut origins = Cube::new(d, config.max_lag + 1, "discovered");
    let mut n_tests = 0;
    for result in results {
        n_tests += result.n_tests;
        for edge in result.edges {
            let idx = (edge.cause, edge.effect, edge.lag);
            graph[idx] = 1;
            values[idx] = edge.statistic;
            pvalues[idx] = edge.p_value;
        }
    }
    if config.include_autoreg && end > st + 10 {
        for i in 0..original_d {
            for l in 1..=ar_order[i].min(config.max_lag) {
                if !matches(&forbidden, i, i, l) {
                    graph[(i, i, l)] = 1;
                    origins[(i, i, l)] = if config.assume_ar1 {
                        "assumed_ar1"
                    } else {
                        "bic_autoregression"
                    };
                }
            }
        }
    }
    for req in &required {
        let lags: Vec<_> = req
            .lag
            .map_or_else(|| (1..=config.max_lag).collect(), |l| vec![l]);
        for l in lags {
            graph[(req.cause, req.effect, l)] = 1;
            origins[(req.cause, req.effect, l)] = "required";
        }
    }
    let pruning = if config.prune {
        let (decisions, tests) = discovery.prune(&graph)?;
        n_tests += tests;
        for (c, e, l, p) in &decisions {
            if *p >= config.alpha_mci.unwrap_or(config.alpha_cond2) {
                graph[(*c, *e, *l)] = 0;
                values[(*c, *e, *l)] = 0.0;
            }
        }
        decisions
            .into_iter()
            .map(|(c, e, l, p)| PruneDecision {
                cause: data.names()[c].clone(),
                effect: data.names()[e].clone(),
                lag: l,
                p_value: p,
                removed: graph[(c, e, l)] == 0,
            })
            .collect()
    } else {
        Vec::new()
    };
    let mut edges = Vec::new();
    for c in 0..d {
        for e in 0..d {
            for l in 0..=config.max_lag {
                if graph[(c, e, l)] == 0 {
                    continue;
                }
                let tested = origins[(c, e, l)] == "discovered";
                edges.push(Edge {
                    cause: data.names()[c].clone(),
                    effect: data.names()[e].clone(),
                    lag: l,
                    statistic: tested.then_some(values[(c, e, l)]),
                    p_value: tested.then_some(pvalues[(c, e, l)]),
                    origin: origins[(c, e, l)].into(),
                });
            }
        }
    }
    let mut warnings = Vec::new();
    if raw.columns().iter().flatten().any(|v| v.is_nan()) {
        warnings
            .push("Missing observations were removed separately for each statistical test.".into());
    }
    if config.assume_ar1 && config.include_autoreg {
        warnings.push("AR(1) self-edges are assumed, not tested; use assume_ar1=false for BIC order selection.".into());
    }
    if config.target_var.is_some() && config.prune {
        warnings.push(
            "Target-only pruning has fewer source-parent estimates than full-graph discovery."
                .into(),
        );
    }
    Ok(CedarResult {
        schema_version: 1,
        algorithm: "cedar".into(),
        ci_test: ci.name().into(),
        n_observations: raw.nrows(),
        upstream_revision: UPSTREAM_REVISION.into(),
        causal_graph: Graph {
            var_names: data.names().to_vec(),
            max_lag: config.max_lag,
            graph: graph.nested(),
        },
        edges,
        val_matrix: values.nested(),
        pvalue_matrix: pvalues.nested(),
        lag_importance: selection.importance.nested(),
        lag_pvalues: selection.pvalues.nested(),
        selected_lags: selection.selected,
        ar_order,
        n_tests,
        pruned: config.prune,
        pruning,
        c_node_names: data.names()[original_d..].to_vec(),
        warnings,
        config: config.clone(),
    })
}

fn validate(config: &CedarConfig, data: &TimeSeries) -> Result<()> {
    if config.max_lag == 0
        || config.max_lag >= data.nrows() / 2
        || data.nrows() - 2 * (config.max_lag + 1) <= 10
    {
        return Err(Error::InvalidConfig(
            "CEDAR requires more than 2 * (max_lag + 1) + 10 observations".into(),
        ));
    }
    for a in [
        Some(config.alpha_cond1),
        Some(config.alpha_cond2),
        Some(config.lag_alpha),
        config.alpha_mci,
    ]
    .into_iter()
    .flatten()
    {
        if !a.is_finite() || a <= 0.0 || a >= 1.0 {
            return Err(Error::InvalidConfig(
                "significance thresholds must be between zero and one".into(),
            ));
        }
    }
    if !config.direction_threshold.is_finite()
        || config.direction_threshold <= 0.0
        || config
            .lag_filter_frac
            .is_some_and(|x| !x.is_finite() || x < 0.0)
    {
        return Err(Error::InvalidConfig(
            "invalid direction threshold or lag filter fraction".into(),
        ));
    }
    Ok(())
}

fn children_mask(
    data: &TimeSeries,
    selection: &LagSelection,
    config: &CedarConfig,
    include_lag0: bool,
) -> Vec<Vec<bool>> {
    let d = data.ncols();
    let mut mask = vec![vec![false; d]; d];
    if !config.filter_children {
        return mask;
    }
    let mut score = vec![vec![0.0; d]; d];
    for (c, row) in score.iter_mut().enumerate() {
        for (e, value) in row.iter_mut().enumerate() {
            if c == e {
                continue;
            }
            *value = match config.direction_method {
                DirectionMethod::Dcor => (usize::from(!include_lag0)..=config.max_lag)
                    .map(|l| selection.importance[(c, e, l)])
                    .fold(f64::NEG_INFINITY, f64::max),
                DirectionMethod::CrossLag => {
                    stats::pearson(&data.column(c)[..data.nrows() - 1], &data.column(e)[1..]).abs()
                }
            };
        }
    }
    for i in 0..d {
        for j in 0..d {
            if score[j][i] > 0.02
                && score[i][j] > 0.02
                && score[j][i] / score[i][j] > config.direction_threshold
            {
                mask[i][j] = true;
            }
        }
    }
    mask
}

fn estimate_ar_order(
    data: &TimeSeries,
    col: usize,
    start: usize,
    end: usize,
    max_lag: usize,
) -> usize {
    if end <= start + 9 {
        return 0;
    }
    let y = &data.column(col)[start..end];
    let mean = y.iter().sum::<f64>() / y.len() as f64;
    let rss = y.iter().map(|v| (v - mean).powi(2)).sum::<f64>();
    let mut best_bic = y.len() as f64 * (rss.max(1e-12) / y.len() as f64).ln();
    let mut best = 0;
    for k in 1..=max_lag {
        if k > start {
            break;
        }
        let rows: Vec<_> = (start..end)
            .filter(|&i| (0..=k).all(|l| data.get(i - l, col).is_finite()))
            .collect();
        if rows.len() < k + 2 {
            break;
        }
        let x: Vec<_> = (1..=k)
            .map(|l| rows.iter().map(|&i| data.get(i - l, col)).collect())
            .collect();
        let y: Vec<_> = rows.iter().map(|&i| data.get(i, col)).collect();
        let rss = stats::residuals(&x, &y, false)
            .iter()
            .map(|r| r * r)
            .sum::<f64>();
        let n = rows.len() as f64;
        let bic = n * (rss.max(1e-12) / n).ln() + k as f64 * n.ln();
        if bic < best_bic {
            best_bic = bic;
            best = k;
        } else {
            break;
        }
    }
    best
}

struct FoundEdge {
    cause: usize,
    effect: usize,
    lag: usize,
    statistic: f64,
    p_value: f64,
}
struct TargetResult {
    edges: Vec<FoundEdge>,
    n_tests: usize,
}

type IndexedPruneDecision = (usize, usize, usize, f64);

struct Discovery<'a, C> {
    data: &'a TimeSeries,
    ci: &'a C,
    config: &'a CedarConfig,
    selection: &'a LagSelection,
    original_d: usize,
    required: &'a [Constraint],
    forbidden: &'a [Constraint],
    children: &'a [Vec<bool>],
    ar_order: &'a [usize],
}

impl<C: ConditionalIndependence> Discovery<'_, C> {
    fn discover_target(&self, target: usize) -> Result<TargetResult> {
        let candidates: Vec<_> = (0..self.data.ncols())
            .filter(|&c| {
                c != target
                    && self.selection.selected[c][target].is_some()
                    && (!self.children[c][target]
                        || self
                            .required
                            .iter()
                            .any(|r| r.cause == c && r.effect == target))
            })
            .collect();
        let mut result = TargetResult {
            edges: Vec::new(),
            n_tests: 0,
        };
        let Some(max_lag) = candidates
            .iter()
            .filter_map(|&c| self.selection.selected[c][target])
            .max()
        else {
            return Ok(result);
        };
        let start = max_lag + 1;
        let end = self.data.nrows() - max_lag - 1;
        if end <= start + 10 {
            return Ok(result);
        }
        for &cause in &candidates {
            let mut lags: Vec<_> = if self.config.multi_lag || self.config.sweep_all_lags {
                (0..=self.config.max_lag)
                    .filter(|&l| self.selection.significant[(cause, target, l)])
                    .collect()
            } else {
                self.selection.selected[cause][target].into_iter().collect()
            };
            lags.sort_by(|&a, &b| {
                self.selection.importance[(cause, target, b)]
                    .total_cmp(&self.selection.importance[(cause, target, a)])
                    .then(a.cmp(&b))
            });
            let ar = self.ar_order.get(cause).copied().unwrap_or_else(|| {
                estimate_ar_order(self.data, cause, start, end, self.config.max_lag)
            });
            for l in lags {
                if matches(self.forbidden, cause, target, l) {
                    continue;
                }
                let (c1, c2, tests) =
                    self.test_at_lag(cause, target, l, &candidates, (start, end), ar)?;
                result.n_tests += tests;
                if let (Some(c1), Some(c2)) = (c1, c2) {
                    if c1.p_value.is_some_and(|p| p <= self.config.alpha_cond1)
                        && c2.p_value.is_some_and(|p| p >= self.config.alpha_cond2)
                    {
                        result.edges.push(FoundEdge {
                            cause,
                            effect: target,
                            lag: l,
                            statistic: c1.statistic,
                            p_value: c1.p_value.unwrap(),
                        });
                        if matches!(self.config.multi_lag_keep, MultiLagKeep::First) {
                            break;
                        }
                    }
                }
            }
        }
        Ok(result)
    }

    fn test_at_lag(
        &self,
        cause: usize,
        target: usize,
        l: usize,
        candidates: &[usize],
        window: (usize, usize),
        ar: usize,
    ) -> Result<(Option<CiResult>, Option<CiResult>, usize)> {
        let (start, end) = window;
        let t = self.data.nrows();
        if start + l >= t {
            return Ok((None, None, 0));
        }
        let mut x = &self.data.column(cause)[start..end];
        let mut y = &self.data.column(target)[start + l..(end + l).min(t)];
        let is_c = cause >= self.original_d;
        let mut conditions = Vec::new();
        if !is_c {
            conditions.push(&self.data.column(target)[start + l - 1..(end + l - 1).min(t)]);
        }
        for &other in candidates {
            if other == cause || (is_c && other >= self.original_d) {
                continue;
            }
            if let Some(w) = self.selection.selected[other][target] {
                let shift = l as isize - w as isize - 1;
                let s = start as isize + shift;
                let e = end as isize + shift;
                if s >= 0 && e <= t as isize {
                    conditions.push(&self.data.column(other)[s as usize..e as usize]);
                }
            }
        }
        let n = conditions
            .iter()
            .map(|c| c.len())
            .chain([x.len(), y.len()])
            .min()
            .unwrap();
        if n < 10 {
            return Ok((None, None, 0));
        }
        x = &x[..n];
        y = &y[..n];
        for col in &mut conditions {
            *col = &col[..n];
        }
        let mut columns = vec![x, y];
        columns.extend(conditions.iter().copied());
        let c1 = self.ci.test(&columns)?;
        if !c1.p_value.is_some_and(|p| p <= self.config.alpha_cond1) {
            return Ok((Some(c1), None, 1));
        }
        let p = ar.max(1).min(start);
        let mut columns2 = vec![&self.data.column(cause)[start - p..start - p + n], y];
        columns2.extend(conditions.iter().copied());
        for j in 1..p {
            columns2.push(&self.data.column(cause)[start - j..start - j + n]);
        }
        columns2.push(x);
        let c2 = self.ci.test(&columns2)?;
        Ok((Some(c1), Some(c2), 2))
    }

    fn prune(&self, graph: &Cube<u8>) -> Result<(Vec<IndexedPruneDecision>, usize)> {
        let start = self.config.max_lag + 1;
        let end = self.data.nrows() - self.config.max_lag - 1;
        let mut decisions = Vec::new();
        let mut tests = 0;
        for target in 0..self.original_d {
            for cause in 0..self.data.ncols() {
                for l in 0..=self.config.max_lag {
                    if graph[(cause, target, l)] == 0
                        || cause == target
                        || matches(self.required, cause, target, l)
                    {
                        continue;
                    }
                    let is_c = cause >= self.original_d;
                    let mut columns = vec![
                        &self.data.column(cause)[start..end],
                        &self.data.column(target)[start + l..end + l],
                    ];
                    for tau in 0..=self.config.max_lag {
                        for c in 0..self.data.ncols() {
                            if graph[(c, target, tau)] == 1
                                && c != cause
                                && c != target
                                && !(is_c && c >= self.original_d)
                            {
                                columns.push(&self.data.column(c)[start + l - tau..end + l - tau]);
                            }
                        }
                    }
                    columns.push(&self.data.column(target)[start + l - 1..end + l - 1]);
                    if !is_c {
                        for tau in 0..=self.config.max_lag {
                            for c in 0..self.data.ncols() {
                                if graph[(c, cause, tau)] == 1 && c != cause {
                                    columns.push(&self.data.column(c)[start - tau..end - tau]);
                                }
                            }
                        }
                        columns.push(&self.data.column(cause)[start - 1..end - 1]);
                    }
                    let result = self.ci.test(&columns)?;
                    tests += 1;
                    if let Some(p) = result.p_value {
                        decisions.push((cause, target, l, p));
                    }
                }
            }
        }
        Ok((decisions, tests))
    }
}
