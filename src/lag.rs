use crate::{graph::Cube, stats, Error, Result, TimeSeries};
use rand::{seq::SliceRandom, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum LagMethod {
    #[default]
    PartialDcor,
    Dcor,
    DcorBiased,
    Pearson,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum LagPvalueMethod {
    #[default]
    TTest,
    CircularShift,
    Permutation,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LagConfig {
    pub max_lag: usize,
    pub method: LagMethod,
    pub pvalue_method: LagPvalueMethod,
    pub alpha: f64,
    pub include_lag0: bool,
    pub fdr: bool,
    pub n_permutations: usize,
    pub seed: u64,
    pub target: Option<usize>,
}

impl Default for LagConfig {
    fn default() -> Self {
        Self {
            max_lag: 1,
            method: LagMethod::PartialDcor,
            pvalue_method: LagPvalueMethod::TTest,
            alpha: 0.05,
            include_lag0: false,
            fdr: true,
            n_permutations: 200,
            seed: 42,
            target: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LagSelection {
    pub importance: Cube<f64>,
    pub pvalues: Cube<f64>,
    pub adjusted_pvalues: Cube<f64>,
    pub selected: Vec<Vec<Option<usize>>>,
    pub significant: Cube<bool>,
}

pub fn select_lags(data: &TimeSeries, config: &LagConfig) -> Result<LagSelection> {
    let d = data.ncols();
    let t = data.nrows();
    let lag = config.max_lag;
    if lag == 0 || lag >= t || t - lag < 10 {
        return Err(Error::InvalidConfig(
            "maximum lag must be positive and leave at least ten observations".into(),
        ));
    }
    if !config.alpha.is_finite() || config.alpha <= 0.0 || config.alpha >= 1.0 {
        return Err(Error::InvalidConfig(
            "lag alpha must be between zero and one".into(),
        ));
    }
    if config.target.is_some_and(|v| v >= d) {
        return Err(Error::InvalidConfig("target index is out of bounds".into()));
    }
    if !matches!(config.pvalue_method, LagPvalueMethod::TTest) && config.n_permutations == 0 {
        return Err(Error::InvalidConfig(
            "permutation tests need at least one permutation".into(),
        ));
    }
    let start_lag = usize::from(!config.include_lag0);
    let n = t - lag;
    let residual_targets: Vec<Vec<f64>> = if matches!(config.method, LagMethod::PartialDcor) {
        (0..d)
            .map(|e| {
                let col = data.column(e);
                let rows: Vec<_> = (lag..t)
                    .filter(|&i| col[i].is_finite() && col[i - 1].is_finite())
                    .collect();
                let mut output = vec![f64::NAN; n];
                if rows.len() >= 10 {
                    let x = rows.iter().map(|&i| col[i - 1]).collect::<Vec<_>>();
                    let y = rows.iter().map(|&i| col[i]).collect::<Vec<_>>();
                    let r = stats::residuals(&[x], &y, true);
                    for (&i, &v) in rows.iter().zip(&r) {
                        output[i - lag] = v;
                    }
                }
                output
            })
            .collect()
    } else {
        Vec::new()
    };
    let pairs: Vec<_> = (0..d)
        .flat_map(|c| {
            (0..d).filter_map(move |e| {
                if config
                    .target
                    .is_none_or(|target| c == target || e == target)
                {
                    Some((c, e))
                } else {
                    None
                }
            })
        })
        .collect();
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let permutations: Vec<Vec<usize>> =
        if matches!(config.pvalue_method, LagPvalueMethod::Permutation) {
            (0..config.n_permutations)
                .map(|_| {
                    let mut p: Vec<_> = (0..n).collect();
                    p.shuffle(&mut rng);
                    p
                })
                .collect()
        } else {
            Vec::new()
        };
    let scores: Vec<_> = pairs
        .par_iter()
        .map(|&(c, e)| {
            let mut importance = vec![0.0; lag + 1];
            let mut pvalues = vec![1.0; lag + 1];
            let seed = config.seed.wrapping_add((c * d + e) as u64 * 0x9e3779b9);
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            for l in start_lag..=lag {
                let source = &data.column(c)[lag - l..t - l];
                let raw_target = &data.column(e)[lag..];
                let target = if matches!(config.method, LagMethod::PartialDcor) {
                    &residual_targets[e][..]
                } else {
                    raw_target
                };
                let (x, y) = complete_pairs(source, target);
                if x.len() < 10 {
                    continue;
                }
                importance[l] = match config.method {
                    LagMethod::Pearson => stats::pearson(&x, &y).abs(),
                    LagMethod::DcorBiased => stats::distance_correlation(&x, &y, false),
                    _ => stats::distance_correlation(&x, &y, true),
                };
                if c == e || importance[l] <= 0.0 {
                    continue;
                }
                pvalues[l] = match config.pvalue_method {
                    LagPvalueMethod::TTest => {
                        let df = (n * (n - 3)) as f64 / 2.0 - 1.0;
                        let r = importance[l];
                        if r >= 1.0 {
                            0.0
                        } else {
                            stats::student_survival(df.sqrt() * r / (1.0 - r * r).sqrt(), df)
                        }
                    }
                    method => {
                        let (x, y) = complete_pairs(source, raw_target);
                        if x.len() < 10 {
                            continue;
                        }
                        let observed = stats::distance_correlation(&x, &y, false);
                        let mut count = 0;
                        for permutation in (0..config.n_permutations).map(|p| permutations.get(p)) {
                            let permuted: Vec<_> = match method {
                                LagPvalueMethod::CircularShift => {
                                    let shift = rng.gen_range(1..y.len());
                                    (0..y.len())
                                        .map(|i| y[(i + y.len() - shift) % y.len()])
                                        .collect()
                                }
                                _ if y.len() == n => {
                                    permutation.unwrap().iter().map(|&i| y[i]).collect()
                                }
                                _ => {
                                    let mut yp = y.clone();
                                    yp.shuffle(&mut rng);
                                    yp
                                }
                            };
                            if stats::distance_correlation(&x, &permuted, false) >= observed {
                                count += 1;
                            }
                        }
                        (count + 1) as f64 / (config.n_permutations + 1) as f64
                    }
                };
            }
            if c == e {
                importance[0] = 1.0;
            }
            (c, e, importance, pvalues)
        })
        .collect();
    let mut importance = Cube::new(d, lag + 1, 0.0);
    let mut pvalues = Cube::new(d, lag + 1, 1.0);
    for (c, e, imp, p) in scores {
        for l in 0..=lag {
            importance[(c, e, l)] = imp[l];
            pvalues[(c, e, l)] = p[l];
        }
    }
    for i in 0..d {
        importance[(i, i, 0)] = 1.0;
    }
    Ok(reselect(importance, pvalues, config))
}

pub(crate) fn reselect(
    importance: Cube<f64>,
    pvalues: Cube<f64>,
    config: &LagConfig,
) -> LagSelection {
    let d = importance.dimension;
    let lag = config.max_lag;
    let start_lag = usize::from(!config.include_lag0);
    let mut adjusted_pvalues = pvalues.clone();
    if config.fdr {
        let mut indices = Vec::new();
        for c in 0..d {
            for e in 0..d {
                if c != e {
                    for l in start_lag..=lag {
                        indices.push((c, e, l));
                    }
                }
            }
        }
        let adjusted =
            stats::bh_adjust(&indices.iter().map(|&idx| pvalues[idx]).collect::<Vec<_>>());
        for (&idx, &p) in indices.iter().zip(&adjusted) {
            adjusted_pvalues[idx] = p;
        }
    }
    let mut selected = vec![vec![None; d]; d];
    let mut significant = Cube::new(d, lag + 1, false);
    for c in 0..d {
        for e in 0..d {
            if c == e {
                continue;
            }
            let mut best = f64::NEG_INFINITY;
            for l in start_lag..=lag {
                if adjusted_pvalues[(c, e, l)] < config.alpha {
                    significant[(c, e, l)] = true;
                    if importance[(c, e, l)] > best {
                        best = importance[(c, e, l)];
                        selected[c][e] = Some(l);
                    }
                }
            }
        }
    }
    LagSelection {
        importance,
        pvalues,
        adjusted_pvalues,
        selected,
        significant,
    }
}

fn complete_pairs(x: &[f64], y: &[f64]) -> (Vec<f64>, Vec<f64>) {
    x.iter()
        .copied()
        .zip(y.iter().copied())
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .unzip()
}
