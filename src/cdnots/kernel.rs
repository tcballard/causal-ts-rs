use super::*;
use nalgebra::DMatrix;

fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let middle = (values.len() - 1) / 2;
    let (_, v, _) = values.select_nth_unstable_by(middle, f64::total_cmp);
    Some(*v)
}

fn kernel(columns: &[Vec<f64>], fixed_width: Option<f64>) -> DMatrix<f64> {
    let n = columns[0].len();
    let distance = |a: usize, b: usize| columns.iter().map(|c| (c[a] - c[b]).powi(2)).sum::<f64>();
    let width = fixed_width.unwrap_or_else(|| {
        let mut distances = vec![];
        for i in 0..n.min(500) {
            for j in i + 1..n.min(500) {
                let v = distance(i, j).sqrt();
                if v > 0.0 {
                    distances.push(v);
                }
            }
        }
        median(distances).map_or(1.0, |v| v * (columns.len() as f64).sqrt())
    });
    DMatrix::from_fn(n, n, |i, j| (-distance(i, j) / width.powi(2)).exp())
}

fn centered_gaussian(m: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let n = m.nrows();
    let d = DMatrix::from_fn(n, n, |i, j| m[(i, i)] + m[(j, j)] - 2.0 * m[(i, j)]);
    let mut distances = vec![];
    for i in 0..n {
        for j in 0..i {
            if d[(i, j)] > 0.0 {
                distances.push(d[(i, j)]);
            }
        }
    }
    let scale = median(distances)?;
    let k = d.map(|v| (-v / scale / 2.0).exp());
    let sums: Vec<f64> = (0..n).map(|i| k.column(i).sum()).collect();
    let total = sums.iter().sum::<f64>();
    Some(DMatrix::from_fn(n, n, |i, j| {
        k[(i, j)] - (sums[i] + sums[j]) / n as f64 + total / (n * n) as f64
    }))
}

pub fn nonstationarity_score(x: &[Vec<f64>], y: &[f64], c: &[Vec<f64>]) -> Result<Option<f64>> {
    if x.is_empty() || c.is_empty() || y.len() < 3 || x.iter().chain(c).any(|v| v.len() != y.len())
    {
        return Err(Error::InvalidData(
            "nonstationarity score requires aligned X, Y and C columns".into(),
        ));
    }
    let rows: Vec<_> = (0..y.len())
        .filter(|&i| y[i].is_finite() && x.iter().chain(c).all(|v| v[i].is_finite()))
        .collect();
    let n = rows.len();
    if n < 3 {
        return Ok(None);
    }
    let normalize = |col: &[f64]| -> Option<Vec<f64>> {
        let mean = rows.iter().map(|&i| col[i]).sum::<f64>() / n as f64;
        let sd =
            (rows.iter().map(|&i| (col[i] - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt();
        if sd <= f64::EPSILON {
            return None;
        }
        Some(rows.iter().map(|&i| (col[i] - mean) / sd).collect())
    };
    let Some(x) = x.iter().map(|v| normalize(v)).collect::<Option<Vec<_>>>() else {
        return Ok(None);
    };
    let Some(y) = normalize(y) else {
        return Ok(None);
    };
    let c: Vec<Vec<_>> = c
        .iter()
        .map(|col| rows.iter().map(|&i| col[i]).collect())
        .collect();
    let kx = kernel(&x, None);
    let ky = kernel(&[y], None);
    let kt = kernel(&c, Some(1.0));
    let eye = DMatrix::identity(n, n) * 2.0;
    let inverse = |m: DMatrix<f64>| {
        m.cholesky()
            .map(|v| v.inverse())
            .ok_or_else(|| Error::Numerical("regularized kernel is not positive definite".into()))
    };
    let inv = inverse(kx.component_mul(&kt) + &eye)?;
    let prod = &inv * ky * &inv;
    let ml = kt
        .component_mul(&kx.map(|v| v.powi(3)))
        .component_mul(&prod)
        .component_mul(&kt)
        / (n * n) as f64;
    let inv2 = inverse(&kt + eye)?;
    let ml2 = &kt * &inv2 * &kx * &inv2 * &kt;
    let Some(mg) = centered_gaussian(&ml) else {
        return Ok(None);
    };
    let Some(mg2) = centered_gaussian(&ml2) else {
        return Ok(None);
    };
    Ok(Some(
        mg.transpose().component_mul(&mg2).sum() / (n * n) as f64,
    ))
}

pub(super) fn phase_three(
    g: &mut WindowGraph,
    data: &[Vec<f64>],
    cfg: &CdnotsConfig,
) -> Result<()> {
    if !cfg.include_c {
        orientation::meek(g);
        return Ok(());
    }
    let mut ns: Vec<_> = (0..g.observed)
        .filter(|&x| {
            (g.observed..g.d).any(|c| g.directed(c, x))
                && (0..g.observed).any(|y| g.undirected(x, y))
        })
        .collect();
    while ns.len() > 1 {
        let mut scored = vec![];
        for &effect in &ns {
            let causes: Vec<_> = (0..g.observed)
                .filter(|&c| g.undirected(c, effect) || g.directed(c, effect))
                .collect();
            let x: Vec<_> = causes.iter().map(|&i| data[i].clone()).collect();
            if let Some(score) = nonstationarity_score(&x, &data[effect], &data[g.observed..g.d])? {
                scored.push((score, effect, causes));
            }
        }
        scored.sort_by(|a, b| a.0.total_cmp(&b.0));
        if scored.is_empty()
            || (cfg.orient_margin > 0.0
                && (scored.len() < 2
                    || (scored[1].0 - scored[0].0)
                        / scored[1].0.abs().max(scored[0].0.abs()).max(1e-300)
                        < cfg.orient_margin))
        {
            break;
        }
        let (_, sink, causes) = &scored[0];
        for &c in causes {
            if !g.conflicts.contains(&(c, *sink))
                && (g.undirected(c, *sink) || g.directed(c, *sink))
            {
                g.orient_copies(c, *sink);
            }
        }
        ns.retain(|&v| v != *sink);
    }
    for v in ns {
        for c in 0..g.observed {
            if g.directed(c, v) {
                g.orient_copies(c, v);
            }
        }
    }
    orientation::meek(g);
    Ok(())
}
