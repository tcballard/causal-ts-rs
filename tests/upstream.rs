use causal_ts::{run_cedar, CedarConfig, CiEngine, CiKind, TimeSeries};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    name: String,
    dataset: String,
    config: CedarConfig,
    ci: CiKind,
    graph: Vec<Vec<Vec<u8>>>,
    lag_importance: Vec<Vec<Vec<f64>>>,
    lag_pvalues: Vec<Vec<Vec<f64>>>,
    selected_lags: Vec<Vec<Option<usize>>>,
    val_matrix: Vec<Vec<Vec<f64>>>,
    pvalue_matrix: Vec<Vec<Vec<f64>>>,
    n_tests: usize,
    ar_order: Vec<usize>,
}

fn fixture_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn close(
    name: &str,
    field: &str,
    a: &[Vec<Vec<f64>>],
    b: &[Vec<Vec<f64>>],
    tol: f64,
    effect_columns: usize,
) {
    assert_eq!(a.len(), b.len());
    let mut max_error = 0.0_f64;
    for (i, (a, b)) in a
        .iter()
        .flat_map(|row| row.iter().take(effect_columns).flatten())
        .zip(
            b.iter()
                .flat_map(|row| row.iter().take(effect_columns).flatten()),
        )
        .enumerate()
    {
        max_error = max_error.max((a - b).abs());
        assert!(
            (a - b).abs() <= tol,
            "{name} {field}[{i}]: {a} != {b}; max error {max_error}"
        );
    }
}

fn check(name: &str) {
    let fixtures: Vec<Fixture> =
        serde_json::from_str(include_str!("fixtures/cedar_reference.json")).unwrap();
    let fixture = fixtures.into_iter().find(|f| f.name == name).unwrap();
    let data =
        TimeSeries::from_csv(std::fs::File::open(fixture_path(&fixture.dataset)).unwrap()).unwrap();
    let result = run_cedar(&data, &CiEngine::new(fixture.ci), &fixture.config).unwrap();
    assert_eq!(result.causal_graph.graph, fixture.graph, "{name}: graph");
    assert_eq!(
        result.selected_lags, fixture.selected_lags,
        "{name}: selected lags"
    );
    assert_eq!(result.n_tests, fixture.n_tests, "{name}: CI test count");
    assert_eq!(result.ar_order, fixture.ar_order, "{name}: AR order");
    close(
        name,
        "lag importance",
        &result.lag_importance,
        &fixture.lag_importance,
        1e-10,
        data.ncols(),
    );
    close(
        name,
        "lag p-values",
        &result.lag_pvalues,
        &fixture.lag_pvalues,
        1e-9,
        data.ncols(),
    );
    let tolerance = if matches!(fixture.ci, CiKind::Gcmi) {
        1e-5
    } else {
        1e-8
    };
    close(
        name,
        "statistic",
        &result.val_matrix,
        &fixture.val_matrix,
        tolerance,
        result.causal_graph.var_names.len(),
    );
    close(
        name,
        "p-values",
        &result.pvalue_matrix,
        &fixture.pvalue_matrix,
        tolerance,
        result.causal_graph.var_names.len(),
    );
}

macro_rules! reference_tests {
    ($($name:ident),*) => { $(#[test] fn $name() { check(stringify!($name)); })* };
}

reference_tests!(
    default,
    lag2,
    lag3,
    unpruned,
    single_lag,
    multi_keep_all,
    sweep,
    dcor,
    dcor_biased,
    pearson,
    zscore,
    raw,
    lag_zero,
    bic_ar,
    no_autoreg,
    target,
    children,
    cross_lag_children,
    relative_filter,
    required,
    forbidden,
    missing,
    c_linear,
    c_multi,
    gcmi
);

#[test]
fn partial_correlation_reference_statistics() {
    #[derive(Deserialize)]
    struct Test {
        x: usize,
        y: usize,
        z: Vec<usize>,
        p: f64,
        statistic: f64,
    }
    let tests: Vec<Test> =
        serde_json::from_str(include_str!("fixtures/ci_reference.json")).unwrap();
    let data = TimeSeries::from_csv(std::fs::File::open(fixture_path("var.csv")).unwrap()).unwrap();
    for test in tests {
        let result = CiEngine::default()
            .test_indices(&data, test.x, test.y, &test.z)
            .unwrap();
        assert!((result.statistic - test.statistic).abs() < 1e-12);
        assert!((result.p_value.unwrap() - test.p).abs() < 1e-10);
    }
}
