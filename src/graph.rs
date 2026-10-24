use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::ops::{Index, IndexMut};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Cube<T> {
    pub dimension: usize,
    pub lag_count: usize,
    pub values: Vec<T>,
}

impl<T: Clone> Cube<T> {
    pub fn new(dimension: usize, lag_count: usize, value: T) -> Self {
        Self {
            dimension,
            lag_count,
            values: vec![value; dimension * dimension * lag_count],
        }
    }
    pub fn nested(&self) -> Vec<Vec<Vec<T>>> {
        (0..self.dimension)
            .map(|c| {
                (0..self.dimension)
                    .map(|e| {
                        (0..self.lag_count)
                            .map(|l| self[(c, e, l)].clone())
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }
}

impl<T> Index<(usize, usize, usize)> for Cube<T> {
    type Output = T;
    fn index(&self, (c, e, l): (usize, usize, usize)) -> &T {
        assert!(c < self.dimension && e < self.dimension && l < self.lag_count);
        &self.values[(c * self.dimension + e) * self.lag_count + l]
    }
}

impl<T> IndexMut<(usize, usize, usize)> for Cube<T> {
    fn index_mut(&mut self, (c, e, l): (usize, usize, usize)) -> &mut T {
        assert!(c < self.dimension && e < self.dimension && l < self.lag_count);
        &mut self.values[(c * self.dimension + e) * self.lag_count + l]
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EdgeConstraint {
    pub cause: String,
    pub effect: String,
    pub lag: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Edge {
    pub cause: String,
    pub effect: String,
    pub lag: usize,
    pub statistic: Option<f64>,
    pub p_value: Option<f64>,
    pub origin: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Graph {
    pub var_names: Vec<String>,
    pub max_lag: usize,
    pub graph: Vec<Vec<Vec<u8>>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct GraphMetrics {
    pub true_positive: usize,
    pub false_positive: usize,
    pub false_negative: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub structural_hamming_distance: usize,
}

impl Graph {
    pub fn validate(&self) -> Result<()> {
        let d = self.var_names.len();
        if d == 0
            || self.max_lag == usize::MAX
            || self.graph.len() != d
            || self.graph.iter().any(|row| {
                row.len() != d
                    || row
                        .iter()
                        .any(|lags| lags.len() != self.max_lag + 1 || lags.iter().any(|&v| v > 1))
            })
        {
            return Err(Error::InvalidData(
                "graph shape or edge values are invalid".into(),
            ));
        }
        let mut names = self.var_names.clone();
        names.sort();
        names.dedup();
        if names.len() != d {
            return Err(Error::InvalidData(
                "graph variable names must be unique".into(),
            ));
        }
        Ok(())
    }

    pub fn evaluate(&self, truth: &Self) -> Result<GraphMetrics> {
        self.validate()?;
        truth.validate()?;
        if self.var_names != truth.var_names || self.max_lag != truth.max_lag {
            return Err(Error::InvalidData(
                "graphs must have the same ordered variables and maximum lag".into(),
            ));
        }
        let (mut tp, mut fp, mut fn_) = (0, 0, 0);
        for c in 0..self.var_names.len() {
            for e in 0..self.var_names.len() {
                for l in 0..=self.max_lag {
                    match (self.graph[c][e][l] != 0, truth.graph[c][e][l] != 0) {
                        (true, true) => tp += 1,
                        (true, false) => fp += 1,
                        (false, true) => fn_ += 1,
                        _ => (),
                    }
                }
            }
        }
        let precision = if tp + fp == 0 {
            0.0
        } else {
            tp as f64 / (tp + fp) as f64
        };
        let recall = if tp + fn_ == 0 {
            0.0
        } else {
            tp as f64 / (tp + fn_) as f64
        };
        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };
        Ok(GraphMetrics {
            true_positive: tp,
            false_positive: fp,
            false_negative: fn_,
            precision,
            recall,
            f1,
            structural_hamming_distance: fp + fn_,
        })
    }

    pub fn to_dot(&self) -> Result<String> {
        self.validate()?;
        let mut out = String::from("digraph causal_ts {\n  rankdir=LR;\n");
        for (i, name) in self.var_names.iter().enumerate() {
            out.push_str(&format!(
                "  n{i} [label={}];\n",
                serde_json::to_string(name)?
            ));
        }
        for c in 0..self.var_names.len() {
            for e in 0..self.var_names.len() {
                for l in 0..=self.max_lag {
                    if self.graph[c][e][l] != 0 {
                        out.push_str(&format!("  n{c} -> n{e} [label=\"lag {l}\"];\n"));
                    }
                }
            }
        }
        out.push_str("}\n");
        Ok(out)
    }
}
