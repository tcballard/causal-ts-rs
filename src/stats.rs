use nalgebra::{DMatrix, DVector};
use statrs::distribution::{ContinuousCDF, StudentsT};

pub fn pearson(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    if x.is_empty() {
        return 0.0;
    }
    let mx = x.iter().sum::<f64>() / x.len() as f64;
    let my = y.iter().sum::<f64>() / y.len() as f64;
    let (mut xy, mut xx, mut yy) = (0.0, 0.0, 0.0);
    for (&a, &b) in x.iter().zip(y) {
        let a = a - mx;
        let b = b - my;
        xy += a * b;
        xx += a * a;
        yy += b * b;
    }
    let denom = xx.sqrt() * yy.sqrt();
    if denom > 0.0 {
        (xy / denom).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

pub fn student_survival(t: f64, df: f64) -> f64 {
    if df <= 0.0 || !df.is_finite() {
        return f64::NAN;
    }
    StudentsT::new(0.0, 1.0, df).unwrap().sf(t)
}

pub fn correlation_pvalue(r: f64, n: usize, conditioning: usize) -> Option<f64> {
    let df = n.checked_sub(conditioning + 2)?;
    if df == 0 {
        return None;
    }
    if r.abs() >= 1.0 {
        return Some(0.0);
    }
    let r = r.clamp(-1.0 + 1e-15, 1.0 - 1e-15);
    Some(
        (2.0 * student_survival(r.abs() * (df as f64 / (1.0 - r * r)).sqrt(), df as f64))
            .clamp(0.0, 1.0),
    )
}

pub fn bh_adjust(pvalues: &[f64]) -> Vec<f64> {
    let mut indices: Vec<_> = (0..pvalues.len()).collect();
    indices.sort_by(|&a, &b| pvalues[a].total_cmp(&pvalues[b]));
    let mut result = vec![1.0; pvalues.len()];
    let mut minimum = 1.0_f64;
    for (rank, &index) in indices.iter().enumerate().rev() {
        minimum = minimum.min(pvalues[index] * pvalues.len() as f64 / (rank + 1) as f64);
        result[index] = minimum.clamp(0.0, 1.0);
    }
    result
}

pub fn distance_correlation(x: &[f64], y: &[f64], unbiased: bool) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    if n < 4 {
        return 0.0;
    }
    if x.iter().chain(y).any(|v| !v.is_finite()) {
        return f64::NAN;
    }
    let normalize = |values: &[f64]| {
        let magnitude = values.iter().map(|x| x.abs()).fold(0.0, f64::max);
        if magnitude == 0.0 {
            return vec![0.0; n];
        }
        let mean = values.iter().map(|v| v / magnitude).sum::<f64>() / n as f64;
        values
            .iter()
            .map(|v| v / magnitude - mean)
            .collect::<Vec<_>>()
    };
    let x = normalize(x);
    let y = normalize(y);
    let rx = distance_row_sums(&x);
    let ry = distance_row_sums(&y);
    let sum_x = rx.iter().sum::<f64>();
    let sum_y = ry.iter().sum::<f64>();
    let f = n as f64;
    let (row_denom, total_denom) = if unbiased {
        (f - 2.0, (f - 1.0) * (f - 2.0))
    } else {
        (f, f * f)
    };
    let xy = distance_product_sum(&x, &y)
        - 2.0 * rx.iter().zip(&ry).map(|(a, b)| a * b).sum::<f64>() / row_denom
        + sum_x * sum_y / total_denom;
    let squared_sum = |values: &[f64]| {
        2.0 * f * values.iter().map(|v| v * v).sum::<f64>()
            - 2.0 * values.iter().sum::<f64>().powi(2)
    };
    let xx = squared_sum(&x) - 2.0 * rx.iter().map(|v| v * v).sum::<f64>() / row_denom
        + sum_x * sum_x / total_denom;
    let yy = squared_sum(&y) - 2.0 * ry.iter().map(|v| v * v).sum::<f64>() / row_denom
        + sum_y * sum_y / total_denom;
    if xx <= 0.0 || yy <= 0.0 {
        return 0.0;
    }
    let denom = xx.sqrt() * yy.sqrt();
    let squared = (xy / denom).clamp(-1.0, 1.0);
    if unbiased {
        squared
    } else {
        squared.max(0.0).sqrt()
    }
}

fn distance_row_sums(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<_> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let total = values.iter().sum::<f64>();
    let mut prefix = 0.0;
    let mut sums = vec![0.0; values.len()];
    for (rank, &i) in order.iter().enumerate() {
        sums[i] = (2.0 * rank as f64 - values.len() as f64) * values[i] + total - 2.0 * prefix;
        prefix += values[i];
    }
    sums
}

fn distance_product_sum(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    let mut order: Vec<_> = (0..n).collect();
    order.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let mut y_order: Vec<_> = (0..n).collect();
    y_order.sort_by(|&a, &b| y[a].total_cmp(&y[b]));
    let mut rank = vec![0; n];
    for (r, &i) in y_order.iter().enumerate() {
        rank[i] = r + 1;
    }
    let mut tree = vec![[0.0; 4]; n + 1];
    let mut totals = [0.0; 4];
    let mut sum = 0.0;
    for i in order {
        let mut left = [0.0; 4];
        let mut r = rank[i];
        while r > 0 {
            for k in 0..4 {
                left[k] += tree[r][k];
            }
            r &= r - 1;
        }
        let right: [f64; 4] = std::array::from_fn(|k| totals[k] - left[k]);
        sum += x[i] * y[i] * left[0] - y[i] * left[1] - x[i] * left[2] + left[3];
        sum += x[i] * right[2] - x[i] * y[i] * right[0] - right[3] + y[i] * right[1];
        let add = [1.0, x[i], y[i], x[i] * y[i]];
        let mut r = rank[i];
        while r <= n {
            for k in 0..4 {
                tree[r][k] += add[k];
            }
            r += r & r.wrapping_neg();
        }
        for k in 0..4 {
            totals[k] += add[k];
        }
    }
    2.0 * sum
}

pub fn residuals(columns: &[Vec<f64>], y: &[f64], intercept: bool) -> Vec<f64> {
    if columns.is_empty() && !intercept {
        return y.to_vec();
    }
    let k = columns.len() + usize::from(intercept);
    let matrix = DMatrix::from_fn(y.len(), k, |i, j| {
        if intercept && j == 0 {
            1.0
        } else {
            columns[j - usize::from(intercept)][i]
        }
    });
    let target = DVector::from_column_slice(y);
    let svd = matrix.clone().svd(true, true);
    let threshold = f64::EPSILON
        * y.len().max(k) as f64
        * svd.singular_values.iter().copied().fold(0.0, f64::max);
    match svd.solve(&target, threshold) {
        Ok(beta) => (target - matrix * beta).as_slice().to_vec(),
        Err(_) => vec![f64::NAN; y.len()],
    }
}
