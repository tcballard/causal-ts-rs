use causal_ts::{
    cedar::{CedarConfig, MultiLagKeep},
    ci::{CiEngine, CiKind, CiStatus, ConditionalIndependence},
    data::TimeBasis,
    graph::EdgeConstraint,
    lag::{select_lags, LagConfig, LagPvalueMethod},
    run_cedar, stats, synthetic, TimeSeries,
};

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "{a} != {b} (tolerance {tol})");
}

#[test]
fn analytic_partial_correlation() {
    let n = 256;
    let z: Vec<_> = (0..n)
        .map(|i| (std::f64::consts::TAU * i as f64 / n as f64).sin())
        .collect();
    let a: Vec<_> = (0..n)
        .map(|i| (2.0 * std::f64::consts::TAU * i as f64 / n as f64).sin())
        .collect();
    let b: Vec<_> = (0..n)
        .map(|i| (3.0 * std::f64::consts::TAU * i as f64 / n as f64).sin())
        .collect();
    let x: Vec<_> = (0..n).map(|i| 2.0 * z[i] + a[i]).collect();
    let y: Vec<_> = (0..n).map(|i| 3.0 * z[i] + b[i]).collect();
    let ci = CiEngine::default();
    let unconditional = ci.test(&[&x, &y]).unwrap();
    assert!(unconditional.p_value.unwrap() < 1e-30);
    let conditional = ci.test(&[&x, &y, &z]).unwrap();
    close(conditional.statistic, 0.0, 1e-12);
    close(conditional.p_value.unwrap(), 1.0, 1e-10);
}

#[test]
fn student_t_and_bh_reference_values() {
    close(
        stats::correlation_pvalue(0.5, 10, 0).unwrap(),
        0.14111328125,
        1e-12,
    );
    close(stats::student_survival(1.0, 1.0), 0.25, 1e-13);
    assert_eq!(stats::correlation_pvalue(0.0, 3, 1), None);
    let actual = stats::bh_adjust(&[0.01, 0.04, 0.03, 0.2]);
    for (a, b) in actual
        .iter()
        .zip([0.04, 0.05333333333333333, 0.05333333333333333, 0.2])
    {
        close(*a, b, 1e-14);
    }
}

#[test]
fn pairwise_missing_and_unrelated_nan_columns() {
    let (data, _) = synthetic::chain(100, 3, 2).unwrap();
    let mut cols = data.columns().to_vec();
    cols[0][7] = f64::NAN;
    cols[2].fill(f64::NAN);
    let incomplete = TimeSeries::new(data.names().to_vec(), cols).unwrap();
    let result = CiEngine::default()
        .test_indices(&incomplete, 0, 1, &[])
        .unwrap();
    assert_eq!(result.n_samples, 99);
    assert!(result.p_value.unwrap().is_finite());
    let missing = CiEngine::default()
        .test_indices(&incomplete, 0, 1, &[2])
        .unwrap();
    assert_eq!(missing.status, CiStatus::InsufficientSamples);
    assert_eq!(missing.p_value, None);
}

#[test]
fn degenerate_and_rank_deficient_inputs_are_finite() {
    let (data, _) = synthetic::chain(100, 3, 22).unwrap();
    let constant = vec![1.0; 100];
    let ci = CiEngine::default();
    let result = ci.test(&[&constant, data.column(0)]).unwrap();
    assert_eq!(result.status, CiStatus::Degenerate);
    assert_eq!(result.p_value, Some(1.0));
    let one = ci
        .test(&[data.column(0), data.column(1), data.column(2)])
        .unwrap();
    let twice = ci
        .test(&[
            data.column(0),
            data.column(1),
            data.column(2),
            data.column(2),
        ])
        .unwrap();
    close(one.statistic, twice.statistic, 1e-10);
}

#[test]
fn gaussian_copula_is_monotone_invariant() {
    let (data, _) = synthetic::chain(120, 3, 7).unwrap();
    let transformed: Vec<Vec<f64>> = data
        .columns()
        .iter()
        .map(|c| c.iter().map(|v| v.exp()).collect())
        .collect();
    let transformed = TimeSeries::new(data.names().to_vec(), transformed).unwrap();
    let ci = CiEngine::new(CiKind::Gcmi);
    let a = ci.test_indices(&data, 0, 1, &[2]).unwrap();
    let b = ci.test_indices(&transformed, 0, 1, &[2]).unwrap();
    close(a.statistic, b.statistic, 1e-14);
    close(a.p_value.unwrap(), b.p_value.unwrap(), 1e-14);
}

#[test]
fn batch_matches_scalar_and_condition_order() {
    let (data, _) = synthetic::chain(100, 4, 42).unwrap();
    let ci = CiEngine::default();
    let tests = vec![(0, 1, vec![2, 3]), (1, 0, vec![3, 2]), (0, 2, vec![])];
    let result = ci.batch_test(&data, &tests).unwrap();
    for (r, (x, y, z)) in result.iter().zip(&tests) {
        let scalar = ci.test_indices(&data, *x, *y, z).unwrap();
        // Cached matrix products can round differently from a scalar dot product.
        close(r.statistic, scalar.statistic, 1e-14);
        close(r.p_value.unwrap(), scalar.p_value.unwrap(), 1e-12);
        assert_eq!(r.n_samples, scalar.n_samples);
        assert_eq!(r.conditioning_size, scalar.conditioning_size);
        assert_eq!(r.status, scalar.status);
    }
    assert_eq!(result[0].statistic, result[1].statistic);
}

#[test]
fn csv_roundtrip_and_validation() {
    let data = TimeSeries::from_csv("\"a,b\",c\n1,2\n3,NA\n".as_bytes()).unwrap();
    assert_eq!(data.names()[0], "a,b");
    assert!(data.get(1, 1).is_nan());
    let mut csv = Vec::new();
    data.write_csv(&mut csv).unwrap();
    let copy = TimeSeries::from_csv(csv.as_slice()).unwrap();
    assert_eq!(data.names(), copy.names());
    assert!(TimeSeries::from_csv("a,a\n1,2\n".as_bytes()).is_err());
    assert!(TimeSeries::from_csv("a,b\n1,infinity\n".as_bytes()).is_err());
    assert!(TimeSeries::from_csv("a,b\n1,2,3\n".as_bytes()).is_err());
}

#[test]
fn causal_chain_matches_upstream_decisions() {
    let (data, truth) = synthetic::chain(500, 3, 42).unwrap();
    let result = run_cedar(&data, &CiEngine::default(), &CedarConfig::default()).unwrap();
    let metrics = result.causal_graph.evaluate(&truth).unwrap();
    assert_eq!(metrics.false_negative, 1);
    assert_eq!(metrics.false_positive, 0);
    assert!(result.n_tests > 0);
    assert_eq!(result.causal_graph.graph[0][2][1], 0);
    assert_eq!(result.causal_graph.graph[0][1][1], 1);
    assert_eq!(result.causal_graph.graph[1][2][1], 0);
}

#[test]
fn required_and_forbidden_edges_are_honored() {
    let (data, _) = synthetic::chain(100, 3, 8).unwrap();
    let mut config = CedarConfig::default();
    config.required.push(EdgeConstraint {
        cause: "X2".into(),
        effect: "X0".into(),
        lag: Some(1),
    });
    config.forbidden.push(EdgeConstraint {
        cause: "X0".into(),
        effect: "X1".into(),
        lag: None,
    });
    config.forbidden.push(EdgeConstraint {
        cause: "X1".into(),
        effect: "X1".into(),
        lag: Some(1),
    });
    let result = run_cedar(&data, &CiEngine::default(), &config).unwrap();
    assert_eq!(result.causal_graph.graph[2][0][1], 1);
    assert_eq!(result.causal_graph.graph[0][1][1], 0);
    assert_eq!(result.causal_graph.graph[1][1][1], 0);
    let forced = result
        .edges
        .iter()
        .find(|e| e.cause == "X2" && e.effect == "X0")
        .unwrap();
    assert_eq!(forced.origin, "required");
    assert_eq!(forced.p_value, None);
    config.forbidden.push(config.required[0].clone());
    assert!(run_cedar(&data, &CiEngine::default(), &config).is_err());
}

#[test]
fn temporal_covariates_are_exogenous() {
    let (data, _) = synthetic::chain(100, 2, 42).unwrap();
    let config = CedarConfig {
        include_c: true,
        c_preset: TimeBasis::LinearQuad,
        ..Default::default()
    };
    let result = run_cedar(&data, &CiEngine::default(), &config).unwrap();
    assert_eq!(result.c_node_names, ["C_lin", "C_quad"]);
    assert!(result.edges.iter().all(|e| !e.effect.starts_with("C_")));
    assert!(result
        .edges
        .iter()
        .filter(|e| e.cause.starts_with("C_"))
        .all(|e| e.lag == 0));
}

#[test]
fn multi_lag_small_sample_windows_do_not_panic() {
    for seed in 0..6 {
        let (data, _) = synthetic::chain(40, 3, seed).unwrap();
        let config = CedarConfig {
            max_lag: 8,
            multi_lag_keep: MultiLagKeep::All,
            sweep_all_lags: true,
            ..Default::default()
        };
        run_cedar(&data, &CiEngine::default(), &config).unwrap();
    }
}

#[test]
fn seeded_permutation_tests_are_repeatable() {
    let (data, _) = synthetic::chain(40, 2, 4).unwrap();
    for method in [LagPvalueMethod::CircularShift, LagPvalueMethod::Permutation] {
        let config = LagConfig {
            n_permutations: 7,
            pvalue_method: method,
            ..Default::default()
        };
        let a = select_lags(&data, &config).unwrap();
        let b = select_lags(&data, &config).unwrap();
        assert_eq!(a.pvalues.values, b.pvalues.values);
        assert!(a.pvalues.values.iter().all(|p| *p > 0.0 && *p <= 1.0));
    }
}

#[test]
fn invalid_configs_fail_explicitly() {
    let (data, _) = synthetic::chain(30, 2, 1).unwrap();
    for max_lag in [0, 10, usize::MAX] {
        assert!(run_cedar(
            &data,
            &CiEngine::default(),
            &CedarConfig {
                max_lag,
                ..Default::default()
            }
        )
        .is_err());
    }
    assert!(serde_json::from_str::<CedarConfig>("{\"typo\":true}").is_err());
}

fn slow_dcor(x: &[f64], y: &[f64], unbiased: bool) -> f64 {
    let n = x.len();
    let f = n as f64;
    let centered = |v: &[f64]| {
        let mut a = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                a[i][j] = (v[i] - v[j]).abs();
            }
        }
        let sums: Vec<_> = a.iter().map(|r| r.iter().sum::<f64>()).collect();
        let total = sums.iter().sum::<f64>();
        let rd = if unbiased { f - 2.0 } else { f };
        let td = if unbiased {
            (f - 1.0) * (f - 2.0)
        } else {
            f * f
        };
        for i in 0..n {
            for j in 0..n {
                a[i][j] = if unbiased && i == j {
                    0.0
                } else {
                    a[i][j] - (sums[i] + sums[j]) / rd + total / td
                };
            }
        }
        a.into_iter().flatten().collect::<Vec<_>>()
    };
    let a = centered(x);
    let b = centered(y);
    let xy = a.iter().zip(&b).map(|(a, b)| a * b).sum::<f64>();
    let denom =
        (a.iter().map(|v| v * v).sum::<f64>() * b.iter().map(|v| v * v).sum::<f64>()).sqrt();
    if denom == 0.0 {
        0.0
    } else if unbiased {
        xy / denom
    } else {
        (xy / denom).max(0.0).sqrt()
    }
}

#[test]
fn fast_distance_correlation_matches_matrix_definition() {
    for n in [4, 10, 40, 100] {
        for seed in 0..8 {
            let (data, _) = synthetic::chain(n, 2, seed).unwrap();
            let x = data.column(0);
            let tied: Vec<_> = data.column(1).iter().map(|v| (v * 2.0).round()).collect();
            for y in [data.column(1), tied.as_slice(), x] {
                for unbiased in [false, true] {
                    close(
                        stats::distance_correlation(x, y, unbiased),
                        slow_dcor(x, y, unbiased),
                        1e-10,
                    );
                }
            }
        }
    }
    assert_eq!(
        stats::distance_correlation(&[1.0; 10], &[2.0; 10], true),
        0.0
    );
}

#[test]
fn one_and_four_threads_match() {
    let (data, _) = synthetic::chain(80, 3, 31).unwrap();
    let run = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| run_cedar(&data, &CiEngine::default(), &CedarConfig::default()).unwrap())
    };
    let a = run(1);
    let b = run(4);
    assert_eq!(
        serde_json::to_value(a).unwrap(),
        serde_json::to_value(b).unwrap()
    );
}
