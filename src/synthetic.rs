use crate::{
    graph::{Cube, Graph},
    Error, Result, TimeSeries,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralLink {
    pub cause: usize,
    pub effect: usize,
    pub lag: usize,
    pub coefficient: f64,
}

pub fn autoregressive(
    samples: usize,
    variables: usize,
    links: &[StructuralLink],
    noise: f64,
    burn_in: usize,
    seed: u64,
) -> Result<(TimeSeries, Graph)> {
    if samples == 0 || variables == 0 || !noise.is_finite() || noise <= 0.0 {
        return Err(Error::InvalidConfig(
            "samples, variables and noise must be positive".into(),
        ));
    }
    if links.iter().any(|l| {
        l.cause >= variables || l.effect >= variables || l.lag == 0 || !l.coefficient.is_finite()
    }) {
        return Err(Error::InvalidConfig("structural links require valid variable indices, positive lags and finite coefficients".into()));
    }
    let max_lag = links.iter().map(|l| l.lag).max().unwrap_or(1);
    let total = samples
        .checked_add(burn_in)
        .and_then(|t| t.checked_add(max_lag))
        .ok_or_else(|| Error::InvalidConfig("sample count is too large".into()))?;
    let mut values = vec![vec![0.0; total]; variables];
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    for t in max_lag..total {
        for col in &mut values {
            let draw: f64 = StandardNormal.sample(&mut rng);
            col[t] = draw * noise;
        }
        for link in links {
            values[link.effect][t] += link.coefficient * values[link.cause][t - link.lag];
        }
        if values.iter().any(|col| !col[t].is_finite()) {
            return Err(Error::Numerical(
                "synthetic process diverged; reduce coefficients".into(),
            ));
        }
    }
    let columns = values
        .into_iter()
        .map(|col| col[total - samples..].to_vec())
        .collect();
    let names = (0..variables).map(|i| format!("X{i}")).collect::<Vec<_>>();
    let mut graph = Cube::new(variables, max_lag + 1, 0u8);
    for l in links {
        if l.coefficient != 0.0 {
            graph[(l.cause, l.effect, l.lag)] = 1;
        }
    }
    Ok((
        TimeSeries::new(names.clone(), columns)?,
        Graph {
            var_names: names,
            max_lag,
            graph: graph.nested(),
        },
    ))
}

pub fn chain(samples: usize, variables: usize, seed: u64) -> Result<(TimeSeries, Graph)> {
    let mut links = Vec::new();
    for i in 0..variables {
        links.push(StructuralLink {
            cause: i,
            effect: i,
            lag: 1,
            coefficient: 0.35,
        });
        if i > 0 {
            links.push(StructuralLink {
                cause: i - 1,
                effect: i,
                lag: 1,
                coefficient: 0.65,
            });
        }
    }
    autoregressive(samples, variables, &links, 0.5, 200, seed)
}
