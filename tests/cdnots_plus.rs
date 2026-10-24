use causal_ts::{run_cdnots_plus, CdnotsPlusConfig, CiEngine, CiKind, TimeSeries};
use serde::Deserialize;
#[derive(Deserialize)]
struct Fixture {
    name: String,
    dataset: String,
    config: serde_json::Value,
    ci: CiKind,
    graph: Vec<Vec<Vec<u8>>>,
    marks: Vec<Vec<Vec<String>>>,
    endpoints: Vec<Vec<i8>>,
    pvalues: Vec<Vec<Vec<Option<f64>>>>,
}
fn check(name: &str) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let fixtures: Vec<Fixture> = serde_json::from_reader(
        std::fs::File::open(path.join("cdnots_plus_reference.json")).unwrap(),
    )
    .unwrap();
    let f = fixtures.into_iter().find(|f| f.name == name).unwrap();
    let data = TimeSeries::from_csv(std::fs::File::open(path.join(f.dataset)).unwrap()).unwrap();
    let mut config = serde_json::to_value(CdnotsPlusConfig::default()).unwrap();
    for (key, value) in f.config.as_object().unwrap() {
        if config.get(key).is_some() {
            config[key] = value.clone();
        } else {
            config["discovery"][key] = value.clone();
        }
    }
    let config: CdnotsPlusConfig = serde_json::from_value(config).unwrap();
    let r = run_cdnots_plus(&data, &CiEngine::new(f.ci), &config).unwrap();
    assert_eq!(r.endpoint_matrix, f.endpoints, "{name} endpoints");
    assert_eq!(r.causal_graph.graph, f.graph, "{name} binary");
    assert_eq!(r.marks, f.marks, "{name} marks");
    for (a, b) in r
        .pvalue_matrix
        .iter()
        .flatten()
        .flatten()
        .zip(f.pvalues.iter().flatten().flatten())
    {
        match (a, b) {
            (Some(a), Some(b)) => assert!((a - b).abs() < 1e-5, "{name} pvalues {a} {b}"),
            (None, None) => {}
            _ => panic!("{name} missing pvalue"),
        }
    }
}
macro_rules! cases {($($name:ident),*)=>{$(#[test]fn $name(){check(stringify!($name));})*}}
cases!(
    default, lag2, lag3, no_c, no_c_lag2, lag0, depth1, priority2, priority0, lag_verify, trend,
    multi_c, missing, gcmi, caps, legacy, alpha
);
