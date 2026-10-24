use crate::{
    backend::Backend,
    ci::{copula_normalize, CiEngine, CiKind, CiResult, CiStatus, ConditionalIndependence},
    stats, Error, Result,
};
use candle_core::{Device, Tensor};
use nalgebra::DMatrix;

pub struct TensorCiEngine {
    pub kind: CiKind,
    device: Device,
    name: String,
}

impl TensorCiEngine {
    pub fn new(kind: CiKind, backend: Backend) -> Result<Self> {
        let device = backend.device()?;
        let name = format!(
            "{}-{}",
            match kind {
                CiKind::ParCorr => "parcorr",
                CiKind::Gcmi => "gcmi",
            },
            Backend::name(&device)
        );
        Ok(Self { kind, device, name })
    }
    pub fn backend(&self) -> &str {
        Backend::name(&self.device)
    }
}

impl ConditionalIndependence for TensorCiEngine {
    fn name(&self) -> &str {
        &self.name
    }
    fn test(&self, columns: &[&[f64]]) -> Result<CiResult> {
        let fallback = || CiEngine::new(self.kind).test(columns);
        if columns.len() < 2
            || columns.iter().any(|c| c.len() != columns[0].len())
            || columns
                .iter()
                .flat_map(|c| c.iter())
                .any(|v| v.is_infinite())
        {
            return fallback();
        }
        let rows: Vec<usize> = (0..columns[0].len())
            .filter(|&i| columns.iter().all(|c| c[i].is_finite()))
            .collect();
        let n = rows.len();
        let k = columns.len() - 2;
        let p = columns.len();
        if n < 10 || n <= k + 2 {
            return fallback();
        }
        let mut matrix = DMatrix::from_fn(n, p, |i, j| columns[j][rows[i]]);
        if matches!(self.kind, CiKind::Gcmi) {
            copula_normalize(&mut matrix);
        }
        for j in 0..p {
            let mean = matrix.column(j).sum() / n as f64;
            for i in 0..n {
                matrix[(i, j)] -= mean;
            }
            let norm = matrix.column(j).norm();
            if norm == 0.0 {
                return fallback();
            }
            let scale = if matches!(self.kind, CiKind::ParCorr) {
                norm
            } else {
                ((n - 1) as f64).sqrt()
            };
            matrix.column_mut(j).scale_mut(1.0 / scale);
        }
        if matrix.iter().any(|v| !v.is_finite()) {
            return Err(Error::Numerical("non-finite covariance input".into()));
        }
        let low_precision = matches!(self.device, Device::Metal(_));
        let flat = if low_precision {
            let x = Tensor::from_vec(
                matrix.iter().map(|v| *v as f32).collect(),
                (p, n),
                &self.device,
            )?;
            x.matmul(&x.t()?)?
                .flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .map(f64::from)
                .collect()
        } else {
            let x = Tensor::from_vec(matrix.as_slice().to_vec(), (p, n), &self.device)?;
            x.matmul(&x.t()?)?.flatten_all()?.to_vec1::<f64>()?
        };
        let mut covariance = DMatrix::from_row_slice(p, p, &flat);
        if matches!(self.kind, CiKind::Gcmi) {
            for j in 0..p {
                covariance[(j, j)] += 1e-10;
            }
        }
        let eigen = covariance.clone().symmetric_eigen().eigenvalues;
        let min = eigen.min();
        let max = eigen.max();
        if min <= 0.0 || min / max < if low_precision { 1e-5 } else { 1e-10 } {
            return fallback();
        }
        let Some(precision) = covariance.try_inverse() else {
            return fallback();
        };
        let denom = precision[(0, 0)] * precision[(1, 1)];
        if denom <= 0.0 || !denom.is_finite() {
            return fallback();
        }
        let raw = -precision[(0, 1)] / denom.sqrt();
        if !raw.is_finite() || raw.abs() > 1.0 + 1e-8 {
            return fallback();
        }
        let r = raw.clamp(-1.0, 1.0);
        let statistic = match self.kind {
            CiKind::ParCorr => r,
            CiKind::Gcmi => -0.5 * (1.0 - r.clamp(-1.0 + 1e-15, 1.0 - 1e-15).powi(2)).ln(),
        };
        Ok(CiResult {
            p_value: stats::correlation_pvalue(r, n, k),
            statistic,
            n_samples: n,
            conditioning_size: k,
            status: CiStatus::Valid,
        })
    }
}
