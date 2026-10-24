mod kernel;
mod orientation;
mod plus;
pub use kernel::nonstationarity_score;
pub use plus::{run_cdnots_plus, run_cdnots_plus_with_covariates, CdnotsPlusConfig};

use crate::{
    ci::{CiResult, ConditionalIndependence, PreparedConditionalIndependence},
    data::TimeBasis,
    graph::{Cube, EdgeConstraint, Graph},
    Error, Result, TimeSeries, UPSTREAM_REVISION,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum ColliderSearch {
    #[default]
    Full,
    Sepset,
    Classic,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CdnotsConfig {
    pub max_lag: usize,
    pub lag_list: Option<Vec<usize>>,
    pub include_c: bool,
    pub c_preset: TimeBasis,
    pub alpha: f64,
    pub stable: bool,
    pub max_degree: Option<usize>,
    pub max_combinations: Option<usize>,
    pub priority: u8,
    pub collider_search: ColliderSearch,
    pub collider_max_combinations: usize,
    pub orient_margin: f64,
    pub lag_verify: bool,
    pub alpha_lv: Option<f64>,
    pub forbidden: Vec<EdgeConstraint>,
    pub required: Vec<EdgeConstraint>,
}

impl Default for CdnotsConfig {
    fn default() -> Self {
        Self {
            max_lag: 1,
            lag_list: None,
            include_c: true,
            c_preset: TimeBasis::Linear,
            alpha: 0.05,
            stable: true,
            max_degree: None,
            max_combinations: Some(20),
            priority: 2,
            collider_search: ColliderSearch::Full,
            collider_max_combinations: 1000,
            orient_margin: 0.0,
            lag_verify: false,
            alpha_lv: None,
            forbidden: vec![],
            required: vec![],
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MarkedEdge {
    pub cause: String,
    pub effect: String,
    pub lag: usize,
    pub mark: String,
    pub p_value: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CiRecord {
    pub effect_node: usize,
    pub cause_node: usize,
    pub conditioning_set: Vec<usize>,
    pub p_value: Option<f64>,
    pub statistic: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CdnotsResult {
    pub schema_version: u32,
    pub algorithm: String,
    pub ci_test: String,
    pub upstream_revision: String,
    pub n_observations: usize,
    pub effective_observations: usize,
    #[serde(flatten)]
    pub causal_graph: Graph,
    pub marks: Vec<Vec<Vec<String>>>,
    pub endpoint_matrix: Vec<Vec<i8>>,
    pub lags: Vec<usize>,
    pub edges: Vec<MarkedEdge>,
    pub pvalue_matrix: Vec<Vec<Vec<Option<f64>>>>,
    pub ci_tests: Vec<CiRecord>,
    pub n_tests: usize,
    pub ambiguous_triples: Vec<(usize, usize, usize)>,
    pub conflict_edges: Vec<(usize, usize)>,
    pub c_node_names: Vec<String>,
    pub warnings: Vec<String>,
    pub config: CdnotsConfig,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub plus_config: Option<CdnotsPlusConfig>,
}

impl CdnotsResult {
    pub fn to_binary(&self, keep_undirected: bool, keep_conflicts: bool) -> Graph {
        let mut graph = self.causal_graph.clone();
        for c in 0..self.marks.len() {
            for e in 0..self.marks.len() {
                for l in 0..self.marks[c][e].len() {
                    graph.graph[c][e][l] = u8::from(match self.marks[c][e][l].as_str() {
                        "-->" => true,
                        "o-o" => keep_undirected,
                        "x-x" => keep_conflicts,
                        _ => false,
                    });
                }
            }
        }
        graph
    }
    pub fn to_dot(&self) -> Result<String> {
        let mut out = String::from("digraph causal_ts {\n  rankdir=LR;\n");
        for (i, name) in self.causal_graph.var_names.iter().enumerate() {
            out.push_str(&format!(
                "  n{i} [label={}];\n",
                serde_json::to_string(name)?
            ));
        }
        for c in 0..self.marks.len() {
            for e in 0..self.marks.len() {
                for l in 0..self.marks[c][e].len() {
                    let mark = self.marks[c][e][l].as_str();
                    let style = match mark {
                        "-->" => "",
                        "o-o" if c < e => ",dir=none,style=dashed",
                        "x-x" if c < e => ",dir=both,arrowhead=tee,arrowtail=tee",
                        _ => continue,
                    };
                    out.push_str(&format!("  n{c} -> n{e} [label=\"lag {l}\"{style}];\n"));
                }
            }
        }
        out.push_str("}\n");
        Ok(out)
    }
}

#[derive(Clone)]
struct WindowGraph {
    g: Vec<Vec<i8>>,
    d: usize,
    observed: usize,
    lags: Vec<usize>,
    sep: Vec<Vec<BTreeSet<usize>>>,
    sep_known: Vec<Vec<bool>>,
    ambiguous: BTreeSet<(usize, usize, usize)>,
    conflicts: BTreeSet<(usize, usize)>,
    forbidden: BTreeSet<(usize, usize)>,
    required: BTreeSet<(usize, usize)>,
}

impl WindowGraph {
    fn new(
        data: &TimeSeries,
        observed: usize,
        lags: Vec<usize>,
        cfg: &CdnotsConfig,
    ) -> Result<Self> {
        let d = data.ncols();
        let n = d * lags.len();
        let mut out = Self {
            g: vec![vec![-1; n]; n],
            d,
            observed,
            lags,
            sep: vec![vec![BTreeSet::new(); n]; n],
            sep_known: vec![vec![false; n]; n],
            ambiguous: BTreeSet::new(),
            conflicts: BTreeSet::new(),
            forbidden: BTreeSet::new(),
            required: BTreeSet::new(),
        };
        for x in 0..n {
            for y in 0..n {
                let cx = x % d >= observed;
                let cy = y % d >= observed;
                if x == y || (cx && cy) || (cx != cy && x / d != y / d) {
                    out.g[x][y] = 0;
                }
            }
        }
        for (required, entries) in [(false, &cfg.forbidden), (true, &cfg.required)] {
            for entry in entries {
                let c = data.index(&entry.cause)?;
                let e = data.index(&entry.effect)?;
                if c == e && entry.lag == Some(0) && required {
                    return Err(Error::InvalidConfig(
                        "a contemporaneous self-loop cannot be required".into(),
                    ));
                }
                if entry.lag.is_some_and(|l| !out.lags.contains(&l)) {
                    return Err(Error::InvalidConfig(
                        "constraint lag is outside the lag list".into(),
                    ));
                }
                for a in 0..out.lags.len() {
                    for b in 0..=a {
                        let lag = out.lags[a] - out.lags[b];
                        if !out.lags.contains(&lag)
                            || entry.lag.is_some_and(|v| v != lag)
                            || (c == e && a == b)
                        {
                            continue;
                        }
                        let pair = (c + a * d, e + b * d);
                        if required {
                            if !out.adj(pair.0, pair.1) {
                                return Err(Error::InvalidConfig(
                                    "required edge violates temporal or C constraints".into(),
                                ));
                            }
                            out.required.insert(pair);
                        } else {
                            out.forbidden.insert(pair);
                        }
                    }
                }
            }
        }
        if out
            .required
            .iter()
            .any(|&(c, e)| out.forbidden.contains(&(c, e)) || out.required.contains(&(e, c)))
        {
            return Err(Error::InvalidConfig("conflicting edge constraints".into()));
        }
        for x in 0..n {
            for y in 0..n {
                if x / d < y / d {
                    out.forbidden.insert((x, y));
                }
                if x % d < observed && y % d >= observed {
                    out.forbidden.insert((x, y));
                }
            }
        }
        if out.required.iter().any(|pair| out.forbidden.contains(pair)) {
            return Err(Error::InvalidConfig(
                "required edge violates temporal or C constraints".into(),
            ));
        }
        for x in 0..n {
            for y in x + 1..n {
                if out.forbidden.contains(&(x, y)) && out.forbidden.contains(&(y, x)) {
                    out.remove(x, y);
                }
            }
        }
        Ok(out)
    }
    fn adj(&self, x: usize, y: usize) -> bool {
        self.g[x][y] != 0 || self.g[y][x] != 0
    }
    fn directed(&self, x: usize, y: usize) -> bool {
        self.g[x][y] == -1 && self.g[y][x] == 1
    }
    fn undirected(&self, x: usize, y: usize) -> bool {
        self.g[x][y] == -1 && self.g[y][x] == -1
    }
    fn neighbors(&self, x: usize) -> Vec<usize> {
        (0..self.g.len()).filter(|&y| self.adj(x, y)).collect()
    }
    fn remove(&mut self, x: usize, y: usize) {
        self.g[x][y] = 0;
        self.g[y][x] = 0;
    }
    fn copies(&self, x: usize, y: usize) -> Vec<(usize, usize)> {
        (0..self.lags.len() - (x / self.d).max(y / self.d))
            .map(|k| (x + k * self.d, y + k * self.d))
            .collect()
    }
    fn remove_copies(&mut self, x: usize, y: usize) {
        for (a, b) in self.copies(x, y) {
            self.remove(a, b);
        }
    }
    fn orient(&mut self, x: usize, y: usize) {
        self.g[x][y] = -1;
        self.g[y][x] = 1;
    }
    fn allowed(&self, x: usize, y: usize) -> bool {
        !self.forbidden.contains(&(x, y)) && !self.required.contains(&(y, x))
    }
    fn required_pair(&self, x: usize, y: usize) -> bool {
        self.required.contains(&(x, y)) || self.required.contains(&(y, x))
    }
    fn separate(&mut self, x: usize, y: usize, z: &[usize]) {
        for (a, b) in [(x, y), (y, x)] {
            self.sep_known[a][b] = true;
            self.sep[a][b].extend(z.iter().copied());
        }
    }
    fn reaches(&self, start: usize, end: usize) -> bool {
        let mut stack = vec![start];
        let mut seen = vec![false; self.g.len()];
        seen[start] = true;
        while let Some(x) = stack.pop() {
            for (y, visited) in seen.iter_mut().enumerate() {
                if self.directed(x, y) {
                    if y == end {
                        return true;
                    }
                    if !*visited {
                        *visited = true;
                        stack.push(y);
                    }
                }
            }
        }
        false
    }
    fn orient_copies(&mut self, x: usize, y: usize) {
        for (a, b) in self.copies(x, y) {
            if self.adj(a, b) && self.allowed(a, b) && !self.reaches(b, a) {
                self.orient(a, b);
            }
        }
    }
}

struct Tester<'a, T: ConditionalIndependence + ?Sized> {
    data: &'a [Vec<f64>],
    ci: &'a T,
    prepared: Option<Box<dyn PreparedConditionalIndependence + 'a>>,
    cache: BTreeMap<(usize, usize, Vec<usize>), CiResult>,
    records: Vec<CiRecord>,
    calls: usize,
}
impl<'a, T: ConditionalIndependence + ?Sized> Tester<'a, T> {
    fn new(data: &'a [Vec<f64>], ci: &'a T) -> Self {
        Self {
            data,
            ci,
            prepared: ci.prepare(data),
            cache: BTreeMap::new(),
            records: vec![],
            calls: 0,
        }
    }

    fn test(&mut self, x: usize, y: usize, z: &[usize], record: bool) -> Result<CiResult> {
        let mut z = z.to_vec();
        z.sort_unstable();
        z.dedup();
        z.retain(|&p| p != x && p != y);
        let key = (x.min(y), x.max(y), z.clone());
        self.calls += 1;
        let result = if let Some(r) = self.cache.get(&key) {
            r.clone()
        } else {
            let r = if let Some(prepared) = &self.prepared {
                prepared.test_indices(key.0, key.1, &z)?
            } else {
                let mut cols = vec![self.data[key.0].as_slice(), self.data[key.1].as_slice()];
                cols.extend(z.iter().map(|&p| self.data[p].as_slice()));
                self.ci.test(&cols)?
            };
            self.cache.insert(key, r.clone());
            r
        };
        if record {
            self.records.push(CiRecord {
                effect_node: x,
                cause_node: y,
                conditioning_set: z,
                p_value: result.p_value,
                statistic: result.statistic,
            });
        }
        Ok(result)
    }
}

fn combinations(items: &[usize], k: usize, cap: Option<usize>) -> Vec<Vec<usize>> {
    if k > items.len() {
        return vec![];
    }
    let cap = cap.unwrap_or(usize::MAX);
    if cap == 0 {
        return vec![];
    }
    if k == 0 {
        return vec![vec![]];
    }
    let mut pos: Vec<usize> = (0..k).collect();
    let mut out = vec![];
    loop {
        out.push(pos.iter().map(|&i| items[i]).collect());
        if out.len() >= cap {
            break;
        }
        let Some(i) = (0..k).rev().find(|&i| pos[i] < items.len() - k + i) else {
            break;
        };
        pos[i] += 1;
        for j in i + 1..k {
            pos[j] = pos[j - 1] + 1;
        }
    }
    out
}

fn skeleton<T: ConditionalIndependence + ?Sized>(
    g: &mut WindowGraph,
    t: &mut Tester<T>,
    cfg: &CdnotsConfig,
) -> Result<()> {
    let mut strength: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for depth in 0..g.g.len() {
        if cfg.max_degree.is_some_and(|n| depth > n)
            || g.g
                .iter()
                .map(|r| r.iter().filter(|&&v| v != 0).count())
                .max()
                .unwrap_or(0)
                <= depth
        {
            break;
        }
        let mut removals = BTreeSet::new();
        let strength_snapshot = strength.clone();
        for x in 0..g.d {
            for y in g.neighbors(x) {
                if !g.adj(x, y) {
                    continue;
                }
                let mut sep = BTreeSet::new();
                let mut removed = false;
                for (node, other) in [(x, y), (y, x)] {
                    let mut nbr = g.neighbors(node);
                    nbr.retain(|&p| p != other);
                    if cfg.max_combinations.is_some() {
                        let strengths = if cfg.stable {
                            &strength_snapshot
                        } else {
                            &strength
                        };
                        nbr.sort_by(|&a, &b| {
                            let key = |p| (node.min(p), node.max(p));
                            strengths
                                .get(&key(b))
                                .unwrap_or(&0.0)
                                .total_cmp(strengths.get(&key(a)).unwrap_or(&0.0))
                        });
                    }
                    for z in combinations(&nbr, depth, cfg.max_combinations) {
                        let r = t.test(x, y, &z, true)?;
                        let key = (x.min(y), x.max(y));
                        let val = r.statistic.abs();
                        strength
                            .entry(key)
                            .and_modify(|v| *v = v.min(val))
                            .or_insert(val);
                        if r.p_value.is_some_and(|p| p > cfg.alpha) {
                            sep.extend(z.iter().copied());
                            if !g.required_pair(x, y) {
                                removals.insert((x, y));
                                if !cfg.stable {
                                    g.remove_copies(x, y);
                                    removed = true;
                                    break;
                                }
                            }
                        }
                    }
                    if removed {
                        break;
                    }
                }
                g.separate(x, y, &sep.into_iter().collect::<Vec<_>>());
            }
        }
        for (x, y) in removals {
            g.remove_copies(x, y);
        }
    }
    Ok(())
}

fn lag_verify<T: ConditionalIndependence + ?Sized>(
    g: &mut WindowGraph,
    t: &mut Tester<T>,
    cfg: &CdnotsConfig,
) -> Result<()> {
    let mut remove = vec![];
    for x in 0..g.observed {
        for c in 0..g.observed {
            let candidates: Vec<usize> = g
                .neighbors(x)
                .into_iter()
                .filter(|&y| y >= g.d && y % g.d == c)
                .collect();
            if candidates.len() < 2 {
                continue;
            }
            let parents: Vec<usize> = g
                .neighbors(x)
                .into_iter()
                .filter(|&p| (p >= g.d && p % g.d < g.observed) || (p < g.d && p >= g.observed))
                .collect();
            for y in candidates {
                let z: Vec<usize> = parents.iter().copied().filter(|&p| p != y).collect();
                if t.test(x, y, &z, false)?
                    .p_value
                    .is_some_and(|p| p > cfg.alpha_lv.unwrap_or(cfg.alpha / 5.0))
                    && !g.required_pair(x, y)
                {
                    remove.push((x, y));
                }
            }
        }
    }
    for (x, y) in remove {
        g.remove_copies(x, y);
    }
    Ok(())
}

fn prepare(
    data: &TimeSeries,
    covariates: Option<&TimeSeries>,
    cfg: &CdnotsConfig,
) -> Result<(TimeSeries, Vec<usize>, Vec<Vec<f64>>)> {
    if !(cfg.alpha > 0.0 && cfg.alpha < 1.0)
        || cfg.alpha_lv.is_some_and(|a| !(a > 0.0 && a < 1.0))
        || !(0.0..1.0).contains(&cfg.orient_margin)
        || cfg.priority > 2
        || cfg.max_combinations == Some(0)
        || cfg.collider_max_combinations == 0
    {
        return Err(Error::InvalidConfig(
            "invalid CDNOTS significance, orientation or search budget".into(),
        ));
    }
    if cfg.lag_list.is_none() && cfg.max_lag >= data.nrows() {
        return Err(Error::InvalidConfig(
            "maximum lag exceeds the number of observations".into(),
        ));
    }
    let mut lags = vec![0];
    lags.extend(
        cfg.lag_list
            .clone()
            .unwrap_or_else(|| (1..=cfg.max_lag).collect()),
    );
    if lags.windows(2).any(|w| w[0] >= w[1]) {
        return Err(Error::InvalidConfig(
            "lag_list must contain strictly increasing positive lags".into(),
        ));
    }
    let maxlag = *lags.last().unwrap();
    let start = maxlag
        .checked_mul(2)
        .ok_or_else(|| Error::InvalidConfig("maximum lag is too large".into()))?;
    if data.nrows() < start.saturating_add(10) {
        return Err(Error::InvalidData(
            "CDNOTS requires at least 10 observations after dropping twice the maximum lag".into(),
        ));
    }
    if covariates.is_some() && !cfg.include_c {
        return Err(Error::InvalidConfig("covariates require include_c".into()));
    }
    let frame = if cfg.include_c {
        if let Some(c) = covariates {
            data.with_covariates(c)?
        } else {
            data.with_time_basis(cfg.c_preset)?
        }
    } else {
        data.clone()
    };
    let nrows = frame.nrows();
    let cols = lags
        .iter()
        .flat_map(|&l| {
            frame
                .columns()
                .iter()
                .map(move |c| c[start - l..nrows - l].to_vec())
        })
        .collect();
    Ok((frame, lags, cols))
}

fn finish<T: ConditionalIndependence + ?Sized>(
    frame: &TimeSeries,
    g: WindowGraph,
    t: Tester<T>,
    cfg: &CdnotsConfig,
    plus: bool,
) -> CdnotsResult {
    let maxlag = *g.lags.last().unwrap();
    let mut marks = Cube::new(g.d, maxlag + 1, String::new());
    let mut binary = Cube::new(g.d, maxlag + 1, 0u8);
    let mut pvals = Cube::new(g.d, maxlag + 1, None::<f64>);
    for r in &t.records {
        if r.effect_node < g.d {
            let c = r.cause_node % g.d;
            let e = r.effect_node;
            let lag = g.lags[r.cause_node / g.d];
            if let Some(p) = r.p_value {
                pvals[(c, e, lag)] = Some(pvals[(c, e, lag)].map_or(p, |v| v.min(p)));
            }
        }
    }
    let mut edges = vec![];
    for (s, &lag) in g.lags.iter().enumerate() {
        for c in 0..g.d {
            for e in 0..g.d {
                let x = c + s * g.d;
                let m = match (g.g[x][e], g.g[e][x]) {
                    (-1, 1) => "-->",
                    (1, -1) => "<--",
                    (-1, -1) => "o-o",
                    (1, 1) => "x-x",
                    _ => "",
                };
                marks[(c, e, lag)] = m.into();
                binary[(c, e, lag)] = u8::from(m == "-->" || (!plus && m == "o-o"));
                if m == "-->" || ((m == "o-o" || m == "x-x") && (lag > 0 || c < e)) {
                    edges.push(MarkedEdge {
                        cause: frame.names()[c].clone(),
                        effect: frame.names()[e].clone(),
                        lag,
                        mark: m.into(),
                        p_value: pvals[(c, e, lag)],
                    });
                }
            }
        }
    }
    let mut warnings = vec![];
    if t.cache.values().any(|r| r.p_value.is_none()) {
        warnings.push(
            "Some CI tests had insufficient complete observations; untestable edges were retained."
                .into(),
        );
    }
    CdnotsResult {
        schema_version: 1,
        algorithm: if plus { "cdnots+" } else { "cdnots" }.into(),
        ci_test: t.ci.name().into(),
        upstream_revision: UPSTREAM_REVISION.into(),
        n_observations: frame.nrows(),
        effective_observations: t.data[0].len(),
        causal_graph: Graph {
            var_names: frame.names().to_vec(),
            max_lag: maxlag,
            graph: binary.nested(),
        },
        marks: marks.nested(),
        endpoint_matrix: g.g,
        lags: g.lags,
        edges,
        pvalue_matrix: pvals.nested(),
        ci_tests: t.records,
        n_tests: t.calls,
        ambiguous_triples: g.ambiguous.into_iter().collect(),
        conflict_edges: g.conflicts.into_iter().collect(),
        c_node_names: frame.names()[g.observed..].to_vec(),
        warnings,
        config: cfg.clone(),
        plus_config: None,
    }
}

pub fn run_cdnots<T: ConditionalIndependence + ?Sized>(
    data: &TimeSeries,
    ci: &T,
    config: &CdnotsConfig,
) -> Result<CdnotsResult> {
    run_cdnots_with_covariates(data, None, ci, config)
}

pub fn run_cdnots_with_covariates<T: ConditionalIndependence + ?Sized>(
    data: &TimeSeries,
    covariates: Option<&TimeSeries>,
    ci: &T,
    config: &CdnotsConfig,
) -> Result<CdnotsResult> {
    let (frame, lags, columns) = prepare(data, covariates, config)?;
    let mut g = WindowGraph::new(&frame, data.ncols(), lags, config)?;
    let mut t = Tester::new(&columns, ci);
    skeleton(&mut g, &mut t, config)?;
    if config.lag_verify {
        lag_verify(&mut g, &mut t, config)?;
    }
    orientation::temporal(&mut g);
    orientation::colliders(&mut g, &mut t, config, None)?;
    orientation::meek(&mut g);
    kernel::phase_three(&mut g, t.data, config)?;
    Ok(finish(&frame, g, t, config, false))
}

#[cfg(feature = "neural")]
pub(crate) fn ci_skeleton<T: ConditionalIndependence + ?Sized>(
    data: &TimeSeries,
    ci: &T,
    cfg: &CdnotsConfig,
) -> Result<(Graph, Cube<f64>)> {
    let (frame, lags, _) = prepare(data, None, cfg)?;
    let mut g = WindowGraph::new(&frame, data.ncols(), lags.clone(), cfg)?;
    let mut columns = vec![];
    for &l in &lags {
        for col in frame.columns() {
            columns.push(col[cfg.max_lag - l..frame.nrows() - l].to_vec());
        }
    }
    let mut t = Tester::new(&columns, ci);
    skeleton(&mut g, &mut t, cfg)?;
    let d = data.ncols();
    let mut pvalues = Cube::new(d, cfg.max_lag + 1, None::<f64>);
    for r in t.records {
        let c = r.cause_node % g.d;
        let e = r.effect_node;
        if e < d && c < d {
            let l = r.cause_node / g.d;
            if let Some(p) = r.p_value {
                pvalues[(c, e, l)] = Some(pvalues[(c, e, l)].map_or(p, |v| v.max(p)));
            }
        }
    }
    let mut p = Cube::new(d, cfg.max_lag + 1, 1.0);
    let mut graph = Cube::new(d, cfg.max_lag + 1, 0u8);
    for i in 0..p.values.len() {
        p.values[i] = pvalues.values[i].unwrap_or(1.0);
        graph.values[i] = u8::from(p.values[i] <= cfg.alpha);
    }
    Ok((
        Graph {
            var_names: data.names().to_vec(),
            max_lag: cfg.max_lag,
            graph: graph.nested(),
        },
        p,
    ))
}
