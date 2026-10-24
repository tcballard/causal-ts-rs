use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CdnotsPlusConfig {
    #[serde(deserialize_with = "deserialize_discovery")]
    pub discovery: CdnotsConfig,
    pub max_conds_py: Option<usize>,
    pub max_conds_px: Option<usize>,
    pub legacy_mci_conds: bool,
}

fn deserialize_discovery<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<CdnotsConfig, D::Error> {
    let value = serde_json::Value::deserialize(deserializer)?;
    let obj = value
        .as_object()
        .ok_or_else(|| serde::de::Error::custom("discovery must be an object"))?;
    let mut base = serde_json::to_value(CdnotsPlusConfig::default().discovery)
        .map_err(serde::de::Error::custom)?;
    for (key, value) in obj {
        base[key] = value.clone();
    }
    serde_json::from_value(base).map_err(serde::de::Error::custom)
}

impl Default for CdnotsPlusConfig {
    fn default() -> Self {
        Self {
            discovery: CdnotsConfig {
                alpha: 0.01,
                priority: 1,
                orient_margin: 0.1,
                ..Default::default()
            },
            max_conds_py: None,
            max_conds_px: None,
            legacy_mci_conds: false,
        }
    }
}

fn per_variable<T: ConditionalIndependence + ?Sized>(
    g: &mut WindowGraph,
    t: &mut Tester<T>,
    cfg: &CdnotsConfig,
) -> Result<()> {
    let mut removals = vec![];
    for j in 0..g.d {
        let mut initial: Vec<_> = g.neighbors(j).into_iter().filter(|&p| p >= g.d).collect();
        if j < g.observed {
            initial.extend(
                g.neighbors(j)
                    .into_iter()
                    .filter(|&p| p >= g.observed && p < g.d),
            );
        }
        let mut parents = initial.clone();
        let mut strength: BTreeMap<usize, f64> =
            parents.iter().map(|&p| (p, f64::INFINITY)).collect();
        for depth in 0..=cfg.max_degree.unwrap_or(initial.len()) {
            if parents.len() <= depth {
                break;
            }
            let mut nonsignificant = BTreeSet::new();
            for &p in &parents {
                if g.required_pair(p, j) {
                    continue;
                }
                let z: Vec<_> = parents
                    .iter()
                    .copied()
                    .filter(|&q| q != p)
                    .take(depth)
                    .collect();
                let r = t.test(j, p, &z, true)?;
                strength
                    .entry(p)
                    .and_modify(|v| *v = v.min(r.statistic.abs()));
                if r.p_value.is_some_and(|p| p > cfg.alpha) {
                    nonsignificant.insert(p);
                    g.separate(j, p, &z);
                }
            }
            parents.retain(|p| !nonsignificant.contains(p));
            parents.sort_by(|a, b| strength[b].total_cmp(&strength[a]));
        }
        removals.extend(
            initial
                .into_iter()
                .filter(|p| !parents.contains(p))
                .map(|p| (j, p)),
        );
    }
    for (j, p) in removals {
        g.remove_copies(j, p);
    }
    Ok(())
}

fn mci<T: ConditionalIndependence + ?Sized>(
    g: &mut WindowGraph,
    t: &mut Tester<T>,
    cfg: &CdnotsPlusConfig,
    lagged: &[Vec<usize>],
    extended_lags: &[usize],
) -> Result<()> {
    let mut all_parents: Vec<Vec<usize>> = (0..g.observed)
        .map(|j| {
            g.neighbors(j)
                .into_iter()
                .filter(|&p| p % g.d < g.observed || (p < g.d && p >= g.observed))
                .collect()
        })
        .collect();
    let mut all_contemp: Vec<Vec<usize>> = all_parents
        .iter()
        .map(|ps| ps.iter().copied().filter(|&p| p < g.observed).collect())
        .collect();
    let mut strengths: Vec<BTreeMap<usize, f64>> = all_parents
        .iter()
        .map(|ps| ps.iter().map(|&p| (p, f64::INFINITY)).collect())
        .collect();
    let mut bases: Vec<Vec<usize>> = (0..g.observed)
        .map(|j| {
            let mut base: Vec<_> = lagged[j]
                .iter()
                .copied()
                .filter(|&p| p % g.d < g.observed)
                .take(cfg.max_conds_py.unwrap_or(usize::MAX))
                .collect();
            base.extend(
                g.neighbors(j)
                    .into_iter()
                    .filter(|&p| p % g.d >= g.observed),
            );
            base
        })
        .collect();
    for depth in 0..g.d {
        let mut tested = false;
        let mut remove = vec![];
        for j in 0..g.observed {
            for &parent in &all_parents[j] {
                let other: Vec<_> = all_contemp[j]
                    .iter()
                    .copied()
                    .filter(|&p| p != parent)
                    .collect();
                if depth > other.len() {
                    continue;
                }
                tested = true;
                let mut base: Vec<_> = bases[j]
                    .iter()
                    .copied()
                    .filter(|&p| {
                        if cfg.legacy_mci_conds {
                            p % g.d != parent % g.d
                        } else {
                            p != parent
                        }
                    })
                    .collect();
                let mut shifted = vec![];
                for &p in &lagged[parent % g.d] {
                    if p % g.d >= g.observed || (cfg.legacy_mci_conds && p % g.d == j) {
                        continue;
                    }
                    let lag = g.lags[p / g.d] + g.lags[parent / g.d];
                    if let Some(s) = extended_lags.iter().position(|&l| l == lag) {
                        let idx = p % g.d + s * g.d;
                        if idx != j {
                            shifted.push(idx);
                        }
                    }
                }
                shifted.truncate(cfg.max_conds_px.unwrap_or(usize::MAX));
                base.extend(shifted);
                for s in combinations(&other, depth, None) {
                    let mut z = s;
                    z.extend(&base);
                    let r = t.test(j, parent, &z, true)?;
                    strengths[j]
                        .entry(parent)
                        .and_modify(|v| *v = v.min(r.statistic.abs()));
                    if r.p_value.is_some_and(|p| p > cfg.discovery.alpha) {
                        g.separate(j, parent, &z);
                        remove.push((j, parent));
                        break;
                    }
                }
            }
        }
        for (j, p) in remove {
            if g.required_pair(j, p) {
                continue;
            }
            all_parents[j].retain(|&v| v != p);
            all_contemp[j].retain(|&v| v != p);
            if p % g.d >= g.observed {
                bases[j].retain(|&v| v != p);
            }
            if p < g.observed {
                all_parents[p].retain(|&v| v != j);
                all_contemp[p].retain(|&v| v != j);
            }
            g.remove_copies(j, p);
        }
        for j in 0..g.observed {
            all_parents[j].sort_by(|a, b| strengths[j][b].total_cmp(&strengths[j][a]));
            all_contemp[j].sort_by(|a, b| strengths[j][b].total_cmp(&strengths[j][a]));
        }
        if !tested {
            break;
        }
    }
    Ok(())
}

pub fn run_cdnots_plus<T: ConditionalIndependence + ?Sized>(
    data: &TimeSeries,
    ci: &T,
    config: &CdnotsPlusConfig,
) -> Result<CdnotsResult> {
    run_cdnots_plus_with_covariates(data, None, ci, config)
}

pub fn run_cdnots_plus_with_covariates<T: ConditionalIndependence + ?Sized>(
    data: &TimeSeries,
    covariates: Option<&TimeSeries>,
    ci: &T,
    config: &CdnotsPlusConfig,
) -> Result<CdnotsResult> {
    let cfg = &config.discovery;
    if !cfg.stable
        || !matches!(cfg.collider_search, ColliderSearch::Full)
        || cfg.max_combinations != Some(20)
    {
        return Err(Error::InvalidConfig("CDNOTS+ uses stable per-variable screening with one combination and MCI majority orientation; plain CDNOTS search overrides do not apply".into()));
    }
    let (frame, lags, mut columns) = prepare(data, covariates, cfg)?;
    let mut g = WindowGraph::new(&frame, data.ncols(), lags.clone(), cfg)?;
    let mut extended_lags = lags.clone();
    let start = 2 * lags.last().unwrap();
    for lag in 0..=start {
        if !extended_lags.contains(&lag) {
            extended_lags.push(lag);
            for col in frame.columns() {
                columns.push(col[start - lag..frame.nrows() - lag].to_vec());
            }
        }
    }
    let mut t = Tester::new(&columns, ci);
    per_variable(&mut g, &mut t, cfg)?;
    let lagged: Vec<Vec<usize>> = (0..g.d)
        .map(|j| g.neighbors(j).into_iter().filter(|&p| p >= g.d).collect())
        .collect();
    mci(&mut g, &mut t, config, &lagged, &extended_lags)?;
    if cfg.lag_verify {
        lag_verify(&mut g, &mut t, cfg)?;
    }
    orientation::temporal(&mut g);
    let parents = orientation::MajorityParents {
        parents: lagged,
        lags: extended_lags,
        max_py: config.max_conds_py,
        max_px: config.max_conds_px,
    };
    orientation::colliders(&mut g, &mut t, cfg, Some(&parents))?;
    orientation::meek(&mut g);
    kernel::phase_three(&mut g, t.data, cfg)?;
    let mut result = finish(&frame, g, t, cfg, true);
    result.plus_config = Some(config.clone());
    Ok(result)
}
