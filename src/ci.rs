use crate::{stats, Error, Result, TimeSeries};
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Normal};

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum CiKind {
    #[default]
    #[value(name = "parcorr", alias = "par-corr")]
    ParCorr,
    Gcmi,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CiStatus {
    Valid,
    Degenerate,
    InsufficientSamples,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CiResult {
    pub p_value: Option<f64>,
    pub statistic: f64,
    pub n_samples: usize,
    pub conditioning_size: usize,
    pub status: CiStatus,
}

pub trait ConditionalIndependence: Sync {
    fn name(&self) -> &str {
        "custom"
    }
    fn test(&self, columns: &[&[f64]]) -> Result<CiResult>;
    /// Optionally prepare repeated tests over one immutable, aligned data window.
    /// Returning `None` keeps the scalar `test` path for existing/custom engines.
    fn prepare<'a>(
        &self,
        _columns: &'a [Vec<f64>],
    ) -> Option<Box<dyn PreparedConditionalIndependence + 'a>> {
        None
    }
}

/// A window-scoped CI context. Implementations must preserve per-test row selection.
pub trait PreparedConditionalIndependence: Sync {
    fn test_indices(&self, x: usize, y: usize, z: &[usize]) -> Result<CiResult>;
}

#[derive(Clone, Debug, Default)]
pub struct CiEngine {
    pub kind: CiKind,
}

impl CiEngine {
    pub fn new(kind: CiKind) -> Self {
        Self { kind }
    }

    pub fn test_indices(
        &self,
        data: &TimeSeries,
        x: usize,
        y: usize,
        z: &[usize],
    ) -> Result<CiResult> {
        let indices = test_indices(data.ncols(), x, y, z)?;
        let columns: Vec<_> = indices.iter().map(|&i| data.column(i)).collect();
        self.test(&columns)
    }

    pub fn batch_test(
        &self,
        data: &TimeSeries,
        tests: &[(usize, usize, Vec<usize>)],
    ) -> Result<Vec<CiResult>> {
        if tests.len() > 1 {
            if let Some(prepared) = self.prepare(data.columns()) {
                return tests
                    .par_iter()
                    .map(|(x, y, z)| prepared.test_indices(*x, *y, z))
                    .collect();
            }
        }
        tests
            .par_iter()
            .map(|(x, y, z)| self.test_indices(data, *x, *y, z))
            .collect()
    }
}

impl ConditionalIndependence for CiEngine {
    fn name(&self) -> &str {
        match self.kind {
            CiKind::ParCorr => "parcorr",
            CiKind::Gcmi => "gcmi",
        }
    }

    fn prepare<'a>(
        &self,
        columns: &'a [Vec<f64>],
    ) -> Option<Box<dyn PreparedConditionalIndependence + 'a>> {
        if !matches!(self.kind, CiKind::ParCorr) {
            return None;
        }
        PreparedParCorr::new(columns)
            .map(|prepared| Box::new(prepared) as Box<dyn PreparedConditionalIndependence>)
    }

    fn test(&self, columns: &[&[f64]]) -> Result<CiResult> {
        if columns.len() < 2 || columns.iter().any(|c| c.len() != columns[0].len()) {
            return Err(Error::InvalidData(
                "CI tests require X, Y and equally sized conditioning columns".into(),
            ));
        }
        if columns
            .iter()
            .flat_map(|c| c.iter())
            .any(|v| v.is_infinite())
        {
            return Err(Error::InvalidData("CI inputs contain infinity".into()));
        }
        let complete: Vec<_> = (0..columns[0].len())
            .filter(|&i| columns.iter().all(|c| c[i].is_finite()))
            .collect();
        let n = complete.len();
        let k = columns.len() - 2;
        if n < 10 || n <= k + 2 {
            return Ok(CiResult {
                p_value: None,
                statistic: 0.0,
                n_samples: n,
                conditioning_size: k,
                status: CiStatus::InsufficientSamples,
            });
        }
        let mut matrix = DMatrix::from_fn(n, columns.len(), |i, j| columns[j][complete[i]]);
        if matches!(self.kind, CiKind::Gcmi) {
            copula_normalize(&mut matrix);
        }
        let (r, degenerate) = match self.kind {
            CiKind::ParCorr => partial_correlation(&matrix),
            CiKind::Gcmi => copula_correlation(&matrix),
        };
        if !r.is_finite() {
            return Err(Error::Numerical("non-finite partial correlation".into()));
        }
        let statistic = match self.kind {
            CiKind::ParCorr => r,
            CiKind::Gcmi => -0.5 * (1.0 - r.clamp(-1.0 + 1e-15, 1.0 - 1e-15).powi(2)).ln(),
        };
        Ok(CiResult {
            p_value: stats::correlation_pvalue(r, n, k),
            statistic,
            n_samples: n,
            conditioning_size: k,
            status: if degenerate {
                CiStatus::Degenerate
            } else {
                CiStatus::Valid
            },
        })
    }
}

fn test_indices(d: usize, x: usize, y: usize, z: &[usize]) -> Result<Vec<usize>> {
    if x >= d || y >= d || z.iter().any(|&i| i >= d) {
        return Err(Error::InvalidConfig(
            "CI variable index is out of bounds".into(),
        ));
    }
    if x == y || z.contains(&x) || z.contains(&y) {
        return Err(Error::InvalidConfig(
            "X, Y and conditioning variables must be distinct".into(),
        ));
    }
    let mut z = z.to_vec();
    z.sort_unstable();
    z.dedup();
    let mut indices = vec![x.min(y), x.max(y)];
    indices.extend(z);
    Ok(indices)
}

struct PreparedParCorr<'a> {
    columns: &'a [Vec<f64>],
    correlation: DMatrix<f64>,
    usable: Vec<bool>,
    constant: Vec<bool>,
    n: usize,
}

impl<'a> PreparedParCorr<'a> {
    fn new(columns: &'a [Vec<f64>]) -> Option<Self> {
        let n = columns.first()?.len();
        if n < 10 || columns.len() < 2 || columns.iter().any(|c| c.len() != n) {
            return None;
        }
        let mut usable: Vec<_> = columns
            .iter()
            .map(|c| c.iter().all(|v| v.is_finite()))
            .collect();
        if usable.iter().filter(|&&v| v).count() < 2 {
            return None;
        }
        let mut values = DMatrix::from_fn(n, columns.len(), |i, j| {
            if usable[j] {
                columns[j][i]
            } else {
                0.0
            }
        });
        let mut constant = vec![false; columns.len()];
        for j in 0..columns.len() {
            if !usable[j] {
                continue;
            }
            let mean = values.column(j).sum() / n as f64;
            for i in 0..n {
                values[(i, j)] -= mean;
            }
            let norm = values.column(j).norm();
            constant[j] = norm == 0.0;
            if !norm.is_finite() {
                usable[j] = false;
                values.column_mut(j).fill(0.0);
            } else if norm > 0.0 {
                values.column_mut(j).scale_mut(1.0 / norm);
            }
        }
        // Only tests whose selected columns are all complete may use this matrix.
        // Missing columns must not change the row set of unrelated tests.
        let correlation = values.transpose() * &values;
        Some(Self {
            columns,
            correlation,
            usable,
            constant,
            n,
        })
    }
}

impl PreparedConditionalIndependence for PreparedParCorr<'_> {
    fn test_indices(&self, x: usize, y: usize, z: &[usize]) -> Result<CiResult> {
        let indices = test_indices(self.columns.len(), x, y, z)?;
        let k = indices.len() - 2;
        let fallback = || {
            let cols: Vec<_> = indices
                .iter()
                .map(|&i| self.columns[i].as_slice())
                .collect();
            CiEngine::default().test(&cols)
        };
        if self.n <= k + 2 || indices.iter().any(|&i| !self.usable[i]) {
            return fallback();
        }
        let degenerate = self.constant[x] || self.constant[y];
        let statistic = if degenerate {
            0.0
        } else if indices.len() == 2 {
            self.correlation[(x, y)].clamp(-1.0, 1.0)
        } else {
            let correlation = DMatrix::from_fn(indices.len(), indices.len(), |i, j| {
                self.correlation[(indices[i], indices[j])]
            });
            let Some(r) = precision_correlation(correlation) else {
                return fallback();
            };
            r
        };
        if !statistic.is_finite() {
            return fallback();
        }
        Ok(CiResult {
            p_value: stats::correlation_pvalue(statistic, self.n, k),
            statistic,
            n_samples: self.n,
            conditioning_size: k,
            status: if degenerate {
                CiStatus::Degenerate
            } else {
                CiStatus::Valid
            },
        })
    }
}

fn centered(data: &DMatrix<f64>) -> DMatrix<f64> {
    let mut result = data.clone();
    for j in 0..data.ncols() {
        let mean = data.column(j).sum() / data.nrows() as f64;
        for i in 0..data.nrows() {
            result[(i, j)] -= mean;
        }
    }
    result
}

fn partial_correlation(data: &DMatrix<f64>) -> (f64, bool) {
    let mut values = centered(data);
    for j in 0..values.ncols() {
        let norm = values.column(j).norm();
        if norm == 0.0 && j < 2 {
            return (0.0, true);
        }
        if norm > 0.0 {
            values.column_mut(j).scale_mut(1.0 / norm);
        }
    }
    if values.ncols() == 2 {
        return (
            values.column(0).dot(&values.column(1)).clamp(-1.0, 1.0),
            false,
        );
    }
    let correlation = values.transpose() * &values;
    if let Some(r) = precision_correlation(correlation) {
        return (r, false);
    }
    let z = values.columns(2, values.ncols() - 2).into_owned();
    let svd = z.clone().svd(true, true);
    let tolerance =
        f64::EPSILON * values.nrows().max(values.ncols()) as f64 * svd.singular_values.max();
    let residual = |col| {
        let target: DVector<f64> = values.column(col).into_owned();
        svd.solve(&target, tolerance).map(|beta| target - &z * beta)
    };
    match (residual(0), residual(1)) {
        (Ok(x), Ok(y)) => {
            if x.norm() <= 32.0 * f64::EPSILON || y.norm() <= 32.0 * f64::EPSILON {
                (0.0, true)
            } else {
                (stats::pearson(x.as_slice(), y.as_slice()), false)
            }
        }
        _ => (0.0, true),
    }
}

fn precision_correlation(correlation: DMatrix<f64>) -> Option<f64> {
    let eigen = correlation.clone().symmetric_eigen().eigenvalues;
    let max = eigen.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min = eigen.iter().copied().fold(f64::INFINITY, f64::min);
    if min > 0.0 && min / max > 1e-10 {
        if let Some(precision) = correlation.try_inverse() {
            let denom = precision[(0, 0)] * precision[(1, 1)];
            if precision[(0, 0)] > 0.0 && precision[(1, 1)] > 0.0 && denom.is_finite() {
                let r = -precision[(0, 1)] / denom.sqrt();
                if r.abs() <= 1.0 + 1e-8 {
                    return Some(r.clamp(-1.0, 1.0));
                }
            }
        }
    }
    None
}

pub(crate) fn copula_normalize(data: &mut DMatrix<f64>) {
    let normal = Normal::new(0.0, 1.0).unwrap();
    for j in 0..data.ncols() {
        let mut order: Vec<_> = (0..data.nrows()).collect();
        order.sort_by(|&a, &b| data[(a, j)].total_cmp(&data[(b, j)]));
        let mut start = 0;
        while start < order.len() {
            let mut end = start + 1;
            while end < order.len() && data[(order[start], j)] == data[(order[end], j)] {
                end += 1;
            }
            let rank = (start + end - 1) as f64 / 2.0;
            let value = normal.inverse_cdf((rank + 0.5) / data.nrows() as f64);
            for &i in &order[start..end] {
                data[(i, j)] = value;
            }
            start = end;
        }
    }
}

fn copula_correlation(data: &DMatrix<f64>) -> (f64, bool) {
    let values = centered(data);
    if values.column(0).norm() == 0.0 || values.column(1).norm() == 0.0 {
        return (0.0, true);
    }
    let mut covariance = values.transpose() * &values / (data.nrows() - 1) as f64;
    for j in 0..covariance.ncols() {
        covariance[(j, j)] += 1e-10;
    }
    if let Some(p) = covariance.try_inverse() {
        let denom = p[(0, 0)] * p[(1, 1)];
        if denom > 0.0 {
            return (
                (-p[(0, 1)] / denom.sqrt()).clamp(-1.0 + 1e-15, 1.0 - 1e-15),
                false,
            );
        }
    }
    (0.0, true)
}
