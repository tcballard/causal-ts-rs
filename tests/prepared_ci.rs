use causal_ts::{
    ci::{CiResult, CiStatus},
    run_cdnots, run_cdnots_plus, CdnotsConfig, CdnotsPlusConfig, CiEngine, CiKind,
    ConditionalIndependence, Result, TimeSeries,
};

fn same(a: CiResult, b: CiResult) {
    assert_eq!(a.n_samples, b.n_samples);
    assert_eq!(a.conditioning_size, b.conditioning_size);
    assert_eq!(a.status, b.status);
    assert!(
        (a.statistic - b.statistic).abs() < 1e-11,
        "{} {}",
        a.statistic,
        b.statistic
    );
    match (a.p_value, b.p_value) {
        (Some(x), Some(y)) => assert!((x - y).abs() < 1e-9, "{x} {y}"),
        (None, None) => {}
        _ => panic!("p-value availability differs"),
    }
}

#[test]
fn prepared_complete_data_matches_scalar_and_parallel_batch() -> Result<()> {
    let (data, _) = causal_ts::synthetic::chain(300, 6, 123)?;
    let ci = CiEngine::default();
    let prepared = ci.prepare(data.columns()).unwrap();
    let mut tests = vec![];
    for x in 0..6 {
        for y in 0..6 {
            if x == y {
                continue;
            }
            let rest: Vec<_> = (0..6).filter(|&z| z != x && z != y).collect();
            for mask in 0..16 {
                let z: Vec<_> = rest
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &z)| ((mask >> i) & 1 == 1).then_some(z))
                    .collect();
                same(
                    prepared.test_indices(x, y, &z)?,
                    ci.test_indices(&data, x, y, &z)?,
                );
                tests.push((x, y, z));
            }
        }
    }
    for (result, (x, y, z)) in ci.batch_test(&data, &tests)?.into_iter().zip(tests) {
        same(result, ci.test_indices(&data, x, y, &z)?);
    }
    Ok(())
}

#[test]
fn missing_rows_are_selected_for_each_test() -> Result<()> {
    let (data, _) = causal_ts::synthetic::chain(100, 5, 11)?;
    let mut columns = data.columns().to_vec();
    columns[2][..20].fill(f64::NAN);
    columns[3][10..30].fill(f64::NAN);
    columns[4][50] = f64::INFINITY;
    let ci = CiEngine::default();
    let prepared = ci.prepare(&columns).unwrap();
    for (z, expected_rows) in [
        (vec![], 100),
        (vec![2], 80),
        (vec![3], 80),
        (vec![2, 3], 70),
    ] {
        let mut selected = vec![columns[0].as_slice(), columns[1].as_slice()];
        selected.extend(z.iter().map(|&i| columns[i].as_slice()));
        let result = prepared.test_indices(0, 1, &z)?;
        assert_eq!(result.n_samples, expected_rows);
        same(result, ci.test(&selected)?);
    }
    assert!(prepared.test_indices(0, 1, &[4]).is_err());
    let all_missing = vec![vec![f64::NAN; 30]; 3];
    assert!(ci.prepare(&all_missing).is_none());
    Ok(())
}

#[test]
fn singular_and_constant_columns_keep_the_existing_fallback() -> Result<()> {
    let (data, _) = causal_ts::synthetic::chain(160, 3, 23)?;
    let mut columns = data.columns().to_vec();
    columns.push(columns[2].clone());
    columns.push(
        columns[2]
            .iter()
            .enumerate()
            .map(|(i, x)| x + (i as f64).sin() * 1e-8)
            .collect(),
    );
    columns.push(vec![2.0; 160]);
    let ci = CiEngine::default();
    let prepared = ci.prepare(&columns).unwrap();
    for indices in [
        vec![0, 1, 2, 3],
        vec![0, 1, 2, 4],
        vec![0, 1, 5],
        vec![0, 5],
        vec![2, 3, 0],
    ] {
        let selected: Vec<_> = indices.iter().map(|&i| columns[i].as_slice()).collect();
        same(
            prepared.test_indices(indices[0], indices[1], &indices[2..])?,
            ci.test(&selected)?,
        );
    }
    assert_eq!(
        prepared.test_indices(0, 5, &[])?.status,
        CiStatus::Degenerate
    );
    Ok(())
}

#[test]
fn prepared_validation_and_insufficient_sample_semantics() -> Result<()> {
    let ci = CiEngine::default();
    assert!(ci.prepare(&[vec![0.0; 9], vec![1.0; 9]]).is_none());
    assert!(ci.prepare(&[vec![0.0; 12], vec![1.0; 13]]).is_none());
    let columns: Vec<Vec<f64>> = (0..13)
        .map(|j| (0..12).map(|i| ((i * (j + 1)) as f64).sin()).collect())
        .collect();
    let prepared = ci.prepare(&columns).unwrap();
    let z: Vec<_> = (2..13).collect();
    let result = prepared.test_indices(0, 1, &z)?;
    assert_eq!(result.status, CiStatus::InsufficientSamples);
    assert_eq!(result.n_samples, 12);
    assert_eq!(result.p_value, None);
    for (x, y, z) in [
        (0, 0, vec![]),
        (0, 13, vec![]),
        (0, 1, vec![13]),
        (0, 1, vec![0]),
        (0, 1, vec![1]),
    ] {
        assert!(prepared.test_indices(x, y, &z).is_err());
    }
    same(
        prepared.test_indices(1, 0, &[3, 2, 3])?,
        prepared.test_indices(0, 1, &[2, 3])?,
    );
    assert!(CiEngine::new(CiKind::Gcmi).prepare(&columns).is_none());
    Ok(())
}

struct Scalar(CiEngine);
impl ConditionalIndependence for Scalar {
    fn name(&self) -> &str {
        self.0.name()
    }
    fn test(&self, columns: &[&[f64]]) -> Result<CiResult> {
        self.0.test(columns)
    }
}

#[test]
fn discovery_preparation_is_scoped_to_each_data_window() -> Result<()> {
    let ci = CiEngine::default();
    let scalar = Scalar(CiEngine::default());
    for seed in [42, 13, 42] {
        let (data, _) = causal_ts::synthetic::chain(150, 4, seed)?;
        let mut columns = data.columns().to_vec();
        if seed == 13 {
            for i in (0..150).step_by(7) {
                columns[1][i] = f64::NAN;
            }
        }
        let data = TimeSeries::new(data.names().to_vec(), columns)?;
        assert!(scalar.prepare(data.columns()).is_none());
        let config = CdnotsConfig {
            max_lag: 2,
            include_c: false,
            ..Default::default()
        };
        let a = run_cdnots(&data, &ci, &config)?;
        let b = run_cdnots(&data, &scalar, &config)?;
        assert_eq!(a.endpoint_matrix, b.endpoint_matrix);
        assert_eq!(a.marks, b.marks);
        assert_eq!(a.n_tests, b.n_tests);
        let mut config = CdnotsPlusConfig::default();
        config.discovery.max_lag = 2;
        config.discovery.include_c = false;
        let a = run_cdnots_plus(&data, &ci, &config)?;
        let b = run_cdnots_plus(&data, &scalar, &config)?;
        assert_eq!(a.endpoint_matrix, b.endpoint_matrix);
        assert_eq!(a.marks, b.marks);
        assert_eq!(a.n_tests, b.n_tests);
    }
    Ok(())
}
