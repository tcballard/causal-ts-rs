use causal_ts::{run_cdnots, CdnotsConfig, CiEngine, CiKind, TimeSeries};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    name: String,
    dataset: String,
    config: CdnotsConfig,
    ci: CiKind,
    graph: Vec<Vec<Vec<u8>>>,
    marks: Vec<Vec<Vec<String>>>,
    endpoints: Vec<Vec<i8>>,
    pvalues: Vec<Vec<Vec<Option<f64>>>>,
}
fn check(name: &str) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let fixtures: Vec<Fixture> =
        serde_json::from_reader(std::fs::File::open(path.join("cdnots_reference.json")).unwrap())
            .unwrap();
    let f = fixtures.into_iter().find(|f| f.name == name).unwrap();
    let data = TimeSeries::from_csv(std::fs::File::open(path.join(f.dataset)).unwrap()).unwrap();
    let r = run_cdnots(&data, &CiEngine::new(f.ci), &f.config).unwrap();
    assert_eq!(r.endpoint_matrix, f.endpoints, "{name} endpoints");
    for c in 0..r.marks.len() {
        for e in 0..r.marks.len() {
            for (s, &lag) in r.lags.iter().enumerate() {
                assert_eq!(
                    r.marks[c][e][lag], f.marks[c][e][s],
                    "{name} marks {c} {e} {lag}"
                );
                assert_eq!(
                    r.causal_graph.graph[c][e][lag], f.graph[c][e][s],
                    "{name} binary {c} {e} {lag}"
                );
                if let (Some(a), Some(b)) = (r.pvalue_matrix[c][e][lag], f.pvalues[c][e][s]) {
                    assert!(
                        (a - b).abs() < 2e-6,
                        "{name} pvalues {c} {e} {lag}: {a} {b}"
                    );
                }
            }
        }
    }
}
macro_rules! cases {($($name:ident),*)=>{$(#[test]fn $name(){check(stringify!($name));})*}}
cases!(
    default,
    lag2,
    no_c,
    no_c_lag2,
    lag0,
    depth1,
    all_combinations,
    sepset,
    priority1,
    priority0,
    lag_verify,
    sparse,
    trend,
    multi_c,
    missing,
    gcmi,
    nonstable
);

#[test]
fn invalid_config_and_marked_exports() {
    let (data, _) = causal_ts::synthetic::chain(60, 2, 42).unwrap();
    let ci = CiEngine::default();
    assert!(run_cdnots(
        &data,
        &ci,
        &CdnotsConfig {
            priority: 3,
            ..Default::default()
        }
    )
    .is_err());
    assert!(run_cdnots(
        &data,
        &ci,
        &CdnotsConfig {
            lag_list: Some(vec![1, 1]),
            ..Default::default()
        }
    )
    .is_err());
    assert!(run_cdnots(
        &data,
        &ci,
        &CdnotsConfig {
            max_lag: 40,
            ..Default::default()
        }
    )
    .is_err());
    let r = run_cdnots(
        &data,
        &ci,
        &CdnotsConfig {
            max_lag: 0,
            include_c: false,
            ..Default::default()
        },
    )
    .unwrap();
    let s = serde_json::to_string(&r).unwrap();
    let decoded: causal_ts::CdnotsResult = serde_json::from_str(&s).unwrap();
    assert_eq!(r.marks, decoded.marks);
    assert_eq!(r.to_binary(true, false).graph, r.causal_graph.graph);
    assert!(r.to_dot().unwrap().contains("digraph"));
}

#[test]
fn nonstationarity_kernel_matches_upstream() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let data = TimeSeries::from_csv(std::fs::File::open(path.join("trend.csv")).unwrap()).unwrap();
    let reference: Vec<f64> =
        serde_json::from_reader(std::fs::File::open(path.join("kernel_reference.json")).unwrap())
            .unwrap();
    for (i, indices) in [vec![0], vec![0, 2]].iter().enumerate() {
        let x: Vec<_> = indices
            .iter()
            .map(|&c| data.column(c)[..80].to_vec())
            .collect();
        let score = causal_ts::cdnots::nonstationarity_score(
            &x,
            &data.column(1)[..80],
            &[(0..80).map(|t| t as f64).collect()],
        )
        .unwrap()
        .unwrap();
        assert!(
            (score - reference[i]).abs() < 1e-6,
            "{score} {}",
            reference[i]
        );
    }
}

#[test]
fn nonstationary() {
    check("nonstationary");
}

#[test]
fn cdnots_constraints_and_sparse_plus_are_enforced() {
    use causal_ts::{run_cdnots_plus, CdnotsPlusConfig, EdgeConstraint};
    let (data, _) = causal_ts::synthetic::chain(120, 3, 31).unwrap();
    let ci = CiEngine::default();
    let config = CdnotsConfig {
        include_c: false,
        max_lag: 2,
        forbidden: vec![EdgeConstraint {
            cause: "X0".into(),
            effect: "X1".into(),
            lag: None,
        }],
        required: vec![EdgeConstraint {
            cause: "X2".into(),
            effect: "X0".into(),
            lag: Some(2),
        }],
        ..Default::default()
    };
    let r = run_cdnots(&data, &ci, &config).unwrap();
    assert_eq!(r.marks[2][0][2], "-->");
    assert!(r.causal_graph.graph[0][1].iter().all(|&v| v == 0));
    let mut config = config;
    config.forbidden.extend(config.required.clone());
    assert!(run_cdnots(&data, &ci, &config).is_err());
    let mut config = CdnotsPlusConfig::default();
    config.discovery.include_c = false;
    config.discovery.lag_list = Some(vec![1, 3]);
    let r = run_cdnots_plus(&data, &ci, &config).unwrap();
    assert_eq!(r.lags, vec![0, 1, 3]);
    assert_eq!(r.causal_graph.max_lag, 3);
    for row in r.marks {
        for edge in row {
            assert_eq!(edge[2], "");
        }
    }
}
