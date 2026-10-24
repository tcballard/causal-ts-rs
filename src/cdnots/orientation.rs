use super::*;

pub(super) struct MajorityParents {
    pub parents: Vec<Vec<usize>>,
    pub lags: Vec<usize>,
    pub max_py: Option<usize>,
    pub max_px: Option<usize>,
}

pub(super) fn temporal(g: &mut WindowGraph) {
    for x in 0..g.g.len() {
        for y in 0..g.g.len() {
            if g.adj(x, y)
                && (x / g.d > y / g.d
                    || (x / g.d == y / g.d && x % g.d >= g.observed && y % g.d < g.observed))
                && g.allowed(x, y)
            {
                g.orient(x, y);
            }
        }
    }
    for (x, y) in g.required.clone() {
        if g.adj(x, y) {
            g.orient(x, y);
        }
    }
    for (x, y) in g.forbidden.clone() {
        if g.undirected(x, y) && g.allowed(y, x) {
            g.orient(y, x);
        }
    }
}

fn triples(g: &WindowGraph) -> Vec<(usize, usize, usize)> {
    let mut out = vec![];
    for y in 0..g.g.len() {
        for x in g.neighbors(y) {
            for z in g.neighbors(y) {
                if x != z && !g.adj(x, z) {
                    out.push((x, y, z));
                }
            }
        }
    }
    out
}

fn vote<T: ConditionalIndependence + ?Sized>(
    g: &WindowGraph,
    t: &mut Tester<T>,
    cfg: &CdnotsConfig,
    triple: (usize, usize, usize),
    parents: Option<&MajorityParents>,
) -> Result<(Option<bool>, (f64, f64))> {
    let (x, y, z) = triple;
    if matches!(cfg.collider_search, ColliderSearch::Classic) {
        return Ok((Some(!g.sep[x][z].contains(&y)), (0.5, 0.0)));
    }
    let mut subsets = vec![];
    let mut seen = BTreeSet::new();
    let cap = cfg.collider_max_combinations;
    let mut base = vec![];
    if let Some(p) = parents {
        let j = if z < g.d { z } else { x };
        let i = if z < g.d { x } else { z };
        for node in [j, i] {
            if node == i && i >= g.d {
                continue;
            }
            let adj: Vec<usize> = g
                .neighbors(node)
                .into_iter()
                .filter(|&v| v < g.d && v != if node == j { i } else { j })
                .collect();
            for k in 0..=adj.len() {
                for s in combinations(&adj, k, Some(cap.saturating_sub(subsets.len()))) {
                    if seen.insert(s.clone()) {
                        subsets.push(s);
                    }
                }
                if subsets.len() >= cap {
                    break;
                }
            }
        }
        for (node, limit) in [(j, p.max_py), (i, p.max_px)] {
            let shift = g.lags[node / g.d];
            let mut list = vec![];
            for &parent in &p.parents[node % g.d] {
                let lag = g.lags[parent / g.d] + shift;
                if let Some(s) = p.lags.iter().position(|&l| l == lag) {
                    list.push(parent % g.d + s * g.d);
                }
            }
            list.truncate(limit.unwrap_or(usize::MAX));
            base.extend(list);
        }
    } else if matches!(cfg.collider_search, ColliderSearch::Sepset) && g.sep_known[x][z] {
        let mut adj = g.sep[x][z].clone();
        adj.remove(&x);
        adj.remove(&z);
        adj.insert(y);
        let adj: Vec<usize> = adj.into_iter().collect();
        for k in 0..=adj.len() {
            subsets.extend(combinations(&adj, k, Some(cap - subsets.len())));
            if subsets.len() == cap {
                break;
            }
        }
    } else {
        let sides: Vec<Vec<usize>> = [(x, z), (z, x)]
            .iter()
            .map(|&(a, b)| {
                let mut n = g.neighbors(a);
                n.retain(|&v| v != b);
                n.sort_by_key(|v| (!g.sep[x][z].contains(v), *v));
                n
            })
            .collect();
        let top = sides[0]
            .len()
            .max(sides[1].len())
            .min(cfg.max_degree.unwrap_or(usize::MAX));
        'outer: for k in 0..=top {
            let a = combinations(&sides[0], k, cfg.max_combinations);
            let b = combinations(&sides[1], k, cfg.max_combinations);
            for i in 0..a.len().max(b.len()) {
                for side in [&a, &b] {
                    if let Some(s) = side.get(i) {
                        if seen.insert(s.clone()) {
                            subsets.push(s.clone());
                            if subsets.len() >= cap {
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }
    }
    let (mut yes, mut no, mut max_p) = (0usize, 0usize, 0.0f64);
    for s in subsets.iter().take(cap) {
        let mut conditions = s.clone();
        conditions.extend(&base);
        let r = t.test(x, z, &conditions, false)?;
        if let Some(p) = r.p_value {
            if s.contains(&y) {
                max_p = max_p.max(p);
            }
            if p > cfg.alpha {
                if s.contains(&y) {
                    yes += 1;
                } else {
                    no += 1;
                }
            }
        }
    }
    let decision = if yes == no { None } else { Some(yes < no) };
    let weight = if yes + no == 0 {
        -0.5
    } else {
        0.5 - yes as f64 / (yes + no) as f64
    };
    Ok((decision, (weight, -max_p.min(1.0))))
}

fn apply_collider(g: &mut WindowGraph, triple: (usize, usize, usize), priority: u8) {
    let (x, y, z) = triple;
    if priority == 2 && (g.directed(y, x) || g.directed(y, z)) {
        return;
    }
    for a in [x, z] {
        for (c, e) in g.copies(a, y) {
            if !g.adj(c, e) || !g.allowed(c, e) {
                continue;
            }
            if priority == 1 {
                if g.undirected(c, e) {
                    g.orient(c, e);
                } else if g.directed(e, c) {
                    g.g[c][e] = 1;
                    g.g[e][c] = 1;
                }
            } else {
                g.orient(c, e);
            }
        }
    }
}

fn arrows(g: &WindowGraph) -> BTreeSet<(usize, usize)> {
    (0..g.g.len())
        .flat_map(|x| {
            (0..g.g.len())
                .filter(move |&y| g.directed(x, y))
                .map(move |y| (x, y))
        })
        .collect()
}

fn cycle_edges(edges: &BTreeSet<(usize, usize)>, n: usize) -> BTreeSet<(usize, usize)> {
    let mut reach = vec![vec![false; n]; n];
    for &(x, y) in edges {
        reach[x][y] = true;
    }
    for k in 0..n {
        let through = reach[k].clone();
        for row in &mut reach {
            if row[k] {
                for (v, &r) in row.iter_mut().zip(&through) {
                    *v |= r;
                }
            }
        }
    }
    edges
        .iter()
        .copied()
        .filter(|&(x, y)| reach[y][x])
        .collect()
}

fn select_colliders(
    candidates: &[(usize, usize, usize)],
    weights: &BTreeMap<(usize, usize, usize), (f64, f64)>,
    others: &BTreeSet<(usize, usize)>,
    ignore: &BTreeSet<(usize, usize)>,
    d: usize,
) -> BTreeSet<(usize, usize, usize)> {
    let acyclic = |subset: &BTreeSet<(usize, usize, usize)>| {
        let mut a = others.clone();
        for &(x, y, z) in subset {
            for c in [x, z] {
                if c < d {
                    a.insert((c, y));
                }
            }
        }
        let e = a
            .iter()
            .copied()
            .filter(|&(x, y)| !a.contains(&(y, x)) && !ignore.contains(&(x, y)))
            .collect();
        cycle_edges(&e, d).is_empty()
    };
    let mut candidates = candidates.to_vec();
    candidates.sort_unstable();
    if candidates.len() > 16 {
        candidates.sort_by(|a, b| weights[b].partial_cmp(&weights[a]).unwrap());
        let mut kept = BTreeSet::new();
        for c in candidates {
            kept.insert(c);
            if !acyclic(&kept) {
                kept.remove(&c);
            }
        }
        return kept;
    }
    let mut best = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    let mut keep = BTreeSet::new();
    for mask in 0..1usize << candidates.len() {
        let sub: BTreeSet<_> = candidates
            .iter()
            .enumerate()
            .filter(|(i, _)| mask & (1 << i) != 0)
            .map(|(_, t)| *t)
            .collect();
        let (a, b) = sub
            .iter()
            .fold((0.0, 0.0), |(a, b), t| (a + weights[t].0, b + weights[t].1));
        let weight = ((a * 1e9).round() / 1e9, (b * 1e9).round() / 1e9);
        if weight < best || !acyclic(&sub) {
            continue;
        }
        if weight > best {
            best = weight;
            keep = sub;
        } else {
            keep = keep.intersection(&sub).copied().collect();
        }
    }
    keep
}

pub(super) fn colliders<T: ConditionalIndependence + ?Sized>(
    g: &mut WindowGraph,
    t: &mut Tester<T>,
    cfg: &CdnotsConfig,
    parents: Option<&MajorityParents>,
) -> Result<()> {
    let initial = g.clone();
    let mut candidates = vec![];
    let mut skipped = vec![];
    let mut weights = BTreeMap::new();
    let triples: Vec<_> = triples(g)
        .into_iter()
        .filter(|&(x, y, z)| x < z && y < g.d)
        .collect();
    for &(x, y, z) in &triples {
        if !g.allowed(x, y) || !g.allowed(z, y) {
            continue;
        }
        if g.directed(x, y) && g.directed(z, y) {
            skipped.push((x, y, z));
            continue;
        }
        let (decision, weight) = vote(g, t, cfg, (x, y, z), parents)?;
        match decision {
            Some(true) => {
                candidates.push((x, y, z));
                weights.insert((x, y, z), weight);
                apply_collider(g, (x, y, z), cfg.priority);
            }
            None => {
                g.ambiguous.extend([(x, y, z), (z, y, x)]);
            }
            _ => {}
        }
    }
    if !cycle_edges(&arrows(g), g.g.len()).is_empty() {
        for tr in skipped {
            let (decision, weight) = vote(g, t, cfg, tr, parents)?;
            if decision == Some(true) {
                candidates.push(tr);
                weights.insert(tr, weight);
            } else if decision.is_none() {
                g.ambiguous.extend([tr, (tr.2, tr.1, tr.0)]);
            }
        }
        let fixed_all = arrows(&initial);
        let fixed_cycle = cycle_edges(&fixed_all, g.g.len());
        let fixed: BTreeSet<_> = fixed_all
            .iter()
            .copied()
            .filter(|&(a, b)| a < g.d && b < g.d)
            .collect();
        let ignore = fixed_cycle
            .iter()
            .filter(|&&(a, b)| a / g.d == b / g.d)
            .map(|&(a, b)| (a % g.d, b % g.d))
            .collect();
        let mut rejected = BTreeSet::new();
        for _ in 0..candidates.len() {
            let cycles = cycle_edges(&arrows(g), g.g.len());
            if cycles.is_empty() {
                break;
            }
            let cyc: BTreeSet<_> = cycles
                .difference(&fixed_cycle)
                .filter_map(|&(a, b)| a.checked_sub(b / g.d * g.d).map(|a| (a, b % g.d)))
                .collect();
            let live: Vec<_> = candidates
                .iter()
                .copied()
                .filter(|tr| !rejected.contains(tr))
                .collect();
            let implicated: Vec<_> = live
                .iter()
                .copied()
                .filter(|&(x, y, z)| cyc.contains(&(x, y)) || cyc.contains(&(z, y)))
                .collect();
            let mut others = fixed.clone();
            for &(x, y, z) in &live {
                if !implicated.contains(&(x, y, z)) {
                    for a in [x, z] {
                        if a < g.d {
                            others.insert((a, y));
                        }
                    }
                }
            }
            let keep = select_colliders(&implicated, &weights, &others, &ignore, g.d);
            let dropped: Vec<_> = implicated
                .into_iter()
                .filter(|tr| !keep.contains(tr))
                .collect();
            if dropped.is_empty() {
                break;
            }
            for (x, y, z) in dropped {
                rejected.insert((x, y, z));
                g.ambiguous.extend([(x, y, z), (z, y, x)]);
            }
            let amb = g.ambiguous.clone();
            *g = initial.clone();
            g.ambiguous = amb;
            for tr in &triples {
                if candidates.contains(tr) && !rejected.contains(tr) {
                    apply_collider(g, *tr, cfg.priority);
                }
            }
        }
        for &(x, y) in cycle_edges(&arrows(g), g.g.len()).difference(&fixed_all) {
            g.g[x][y] = 1;
            g.g[y][x] = 1;
        }
    }
    Ok(())
}

pub(super) fn meek(g: &mut WindowGraph) {
    let triples = triples(g);
    loop {
        let before = g.g.clone();
        let mut r1 = BTreeSet::new();
        for &(i, j, k) in &triples {
            if j < g.d
                && !g.ambiguous.contains(&(i, j, k))
                && !g.conflicts.contains(&(j, k))
                && g.directed(i, j)
                && g.undirected(j, k)
                && g.allowed(j, k)
            {
                r1.insert((j, k));
            }
        }
        for &(j, k) in &r1 {
            if r1.contains(&(k, j)) {
                g.conflicts.extend([(j, k), (k, j)]);
            } else if g.undirected(j, k) {
                g.orient_copies(j, k);
            }
        }
        for i in 0..g.d {
            for j in 0..g.g.len() {
                if !g.directed(i, j) {
                    continue;
                }
                for k in 0..g.d {
                    if !g.conflicts.contains(&(i, k))
                        && g.directed(j, k)
                        && g.undirected(i, k)
                        && g.allowed(i, k)
                    {
                        g.orient_copies(i, k);
                    }
                }
            }
        }
        for i in 0..g.d {
            for l in 0..g.d {
                if !g.undirected(i, l) || g.conflicts.contains(&(i, l)) || !g.allowed(i, l) {
                    continue;
                }
                let neighbors: Vec<_> = (0..g.d)
                    .filter(|&j| g.undirected(i, j) && g.directed(j, l))
                    .collect();
                if neighbors.iter().any(|&j| {
                    neighbors
                        .iter()
                        .any(|&k| j != k && !g.adj(j, k) && !g.ambiguous.contains(&(j, i, k)))
                }) {
                    g.orient_copies(i, l);
                }
            }
        }
        if before == g.g {
            break;
        }
    }
}
