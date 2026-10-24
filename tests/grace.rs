#![cfg(feature = "neural")]
use candle_core::{Device, Tensor};
use causal_ts::{
    grace::{
        model::{Adam, ModelConfig, TensorData},
        GatedModel,
    },
    Result,
};
use serde::Deserialize;
use std::collections::BTreeMap;
#[derive(Deserialize)]
struct Fixture {
    name: String,
    config: ModelConfig,
    state: BTreeMap<String, TensorData>,
    input: TensorData,
    uniforms: TensorData,
    skeleton: Vec<Vec<Vec<u8>>>,
    eval_mu: TensorData,
    eval_sigma: TensorData,
    eval_loss: f64,
    gates: Vec<Vec<Vec<f64>>>,
    train_mu: TensorData,
    train_sigma: TensorData,
    loss: f64,
    gradients: BTreeMap<String, TensorData>,
    after_adam: BTreeMap<String, TensorData>,
}
fn tensor(v: &TensorData) -> Tensor {
    Tensor::from_vec(v.values.clone(), v.shape.clone(), &Device::Cpu).unwrap()
}
fn close(actual: &Tensor, expected: &TensorData, tol: f32, name: &str) {
    assert_eq!(actual.dims(), expected.shape);
    for (i, (a, b)) in actual
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .zip(&expected.values)
        .enumerate()
    {
        assert!((a - b).abs() < tol, "{name}[{i}]: {a} != {b}");
    }
}
fn check(name: &str) -> Result<()> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/grace_reference.json");
    let fixtures: Vec<Fixture> = serde_json::from_reader(std::fs::File::open(path)?)?;
    let f = fixtures.into_iter().find(|f| f.name == name).unwrap();
    let mut model = GatedModel::new(3, 2, &f.config, &Device::Cpu, 42)?;
    model.initialize_skeleton(&f.skeleton, true)?;
    model.load_state(&f.state)?;
    let x = tensor(&f.input);
    let u = tensor(&f.uniforms);
    let (mu, sigma) = model.forward(&x, None)?;
    close(&mu, &f.eval_mu, 2e-5, "eval_mu");
    close(&sigma, &f.eval_sigma, 2e-5, "eval_sigma");
    let loss = model
        .loss(&x, &mu, &sigma, 0.03, 3, false)?
        .to_scalar::<f32>()? as f64;
    assert!((loss - f.eval_loss).abs() < 2e-5);
    let gates = model.gate_values()?.nested();
    for (a, b) in gates
        .iter()
        .flatten()
        .flatten()
        .zip(f.gates.iter().flatten().flatten())
    {
        assert!((a - b).abs() < 1e-6);
    }
    let (mu, sigma) = model.forward(&x, Some(&u))?;
    close(&mu, &f.train_mu, 2e-5, "train_mu");
    close(&sigma, &f.train_sigma, 2e-5, "train_sigma");
    let loss = model.loss(&x, &mu, &sigma, 0.03, 3, true)?;
    assert!((loss.to_scalar::<f32>()? as f64 - f.loss).abs() < 2e-5);
    let gradients = model.gradients(&loss)?;
    for (name, g) in gradients {
        close(
            &tensor(&g),
            &f.gradients[&name],
            5e-5,
            &format!("gradient {name}"),
        );
    }
    let mut adam = Adam::new();
    adam.step(&model, &loss)?;
    for (name, p) in model.state()? {
        close(
            &tensor(&p),
            &f.after_adam[&name],
            2e-5,
            &format!("adam {name}"),
        );
    }
    Ok(())
}
#[test]
fn basis() -> Result<()> {
    check("basis")
}
#[test]
fn linear() -> Result<()> {
    check("linear")
}
#[test]
fn nonlinear() -> Result<()> {
    check("nonlinear")
}

#[test]
fn helper_formulas_and_windows_match_upstream() -> Result<()> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/grace_helpers_reference.json");
    let reference: serde_json::Value = serde_json::from_reader(std::fs::File::open(path)?)?;
    for (i, (d, t, rho)) in [(3, 300, 0.2), (10, 1000, 0.5), (100, 100, 0.01)]
        .into_iter()
        .enumerate()
    {
        assert!(
            (causal_ts::grace::compute_lambda(d, t, rho)?
                - reference["lambdas"][i].as_f64().unwrap())
            .abs()
                < 1e-12
        );
    }
    for n in [1, 2, 3, 5, 10, 20] {
        let grid = causal_ts::grace::smart_lambda_grid(n)?;
        let expected = reference["grids"][n.to_string()].as_array().unwrap();
        assert_eq!(grid.len(), expected.len());
        for (a, b) in grid.iter().zip(expected) {
            assert!((a - b.as_f64().unwrap()).abs() < 1e-12);
        }
    }
    let data = causal_ts::TimeSeries::new(
        vec!["x".into(), "y".into()],
        vec![vec![1., 2., 3., 4.], vec![5., 6., 7., 8.]],
    )?;
    let windows =
        causal_ts::grace::prepare_data(&data, 2, false, &Device::Cpu)?.to_vec3::<f32>()?;
    assert_eq!(
        windows,
        vec![
            vec![vec![1., 2., 3.], vec![5., 6., 7.]],
            vec![vec![2., 3., 4.], vec![6., 7., 8.]]
        ]
    );
    Ok(())
}

fn small_config() -> causal_ts::grace::GraceConfig {
    causal_ts::grace::GraceConfig {
        max_lag: 1,
        include_c: false,
        max_epochs: 8,
        batch_size: Some(32),
        patience: 8,
        model: ModelConfig {
            hidden_size: 4,
            decoder_hidden: 5,
            decoder_layers: 1,
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn gated_training_is_reproducible_and_respects_skeleton() -> Result<()> {
    use causal_ts::grace::run_grace_with_skeleton;
    let (data, skeleton) = causal_ts::synthetic::chain(96, 2, 13)?;
    let ci = causal_ts::CiEngine::default();
    let config = small_config();
    let a = run_grace_with_skeleton(&data, Some(&skeleton), &ci, &config)?;
    let b = run_grace_with_skeleton(&data, Some(&skeleton), &ci, &config)?;
    assert_eq!(serde_json::to_string(&a)?, serde_json::to_string(&b)?);
    assert!(
        a.training_runs[0].loss_history.last().unwrap()
            < a.training_runs[0].loss_history.first().unwrap()
    );
    for c in 0..2 {
        for e in 0..2 {
            for l in 0..2 {
                if skeleton.graph[c][e][l] == 0 {
                    assert_eq!(a.gate_values[c][e][l], 0.0);
                }
            }
        }
    }
    assert_eq!(a.causal_graph.max_lag, 1);
    assert_eq!(a.backend, "cpu");
    Ok(())
}

#[test]
fn stability_and_cross_validation_execute() -> Result<()> {
    use causal_ts::grace::{run_grace_with_skeleton, GraceMode};
    let (data, skeleton) = causal_ts::synthetic::chain(64, 2, 17)?;
    let ci = causal_ts::CiEngine::default();
    let mut cfg = small_config();
    cfg.mode = GraceMode::Stability;
    cfg.max_epochs = 2;
    cfg.warm_start_epochs = 1;
    cfg.n_subsamples = 2;
    cfg.lambda_grid = Some(vec![0.01, 0.03]);
    cfg.use_skeleton_mask = true;
    let r = run_grace_with_skeleton(&data, Some(&skeleton), &ci, &cfg)?;
    assert_eq!(r.training_runs.len(), 4);
    assert!(r
        .stability_scores
        .as_ref()
        .unwrap()
        .iter()
        .flatten()
        .flatten()
        .all(|s| (0.0..=1.0).contains(s)));
    for c in 0..2 {
        for e in 0..2 {
            assert!(
                r.causal_graph.graph[c][e]
                    .iter()
                    .map(|&v| v as usize)
                    .sum::<usize>()
                    <= 1
            );
        }
    }
    cfg = small_config();
    cfg.max_epochs = 2;
    cfg.lambda_cv = true;
    cfg.cv_lambdas = 2;
    cfg.cv_epochs = 2;
    cfg.cv_warm_start_epochs = 1;
    let r = run_grace_with_skeleton(&data, Some(&skeleton), &ci, &cfg)?;
    assert_eq!(r.cross_validation.len(), 2);
    assert!(r
        .cross_validation
        .iter()
        .any(|v| Some(v.lambda) == r.lambda_l0));
    Ok(())
}

#[test]
fn training_rejects_invalid_data_and_configuration() -> Result<()> {
    use causal_ts::grace::run_grace;
    let (data, _) = causal_ts::synthetic::chain(32, 2, 17)?;
    let ci = causal_ts::CiEngine::default();
    let mut cfg = small_config();
    cfg.max_epochs = 0;
    assert!(run_grace(&data, &ci, &cfg).is_err());
    cfg = small_config();
    cfg.model.temperature = 0.0;
    assert!(run_grace(&data, &ci, &cfg).is_err());
    cfg = small_config();
    cfg.include_c_in_model = true;
    assert!(run_grace(&data, &ci, &cfg).is_err());
    let bad = causal_ts::TimeSeries::new(vec!["x".into()], vec![vec![f64::NAN; 32]])?;
    assert!(run_grace(&bad, &ci, &small_config()).is_err());
    let a: GatedModel = GatedModel::new(
        2,
        1,
        &ModelConfig {
            hidden_size: 2,
            ..Default::default()
        },
        &Device::Cpu,
        42,
    )?;
    assert!(a.load_state(&BTreeMap::new()).is_err());
    let partial: causal_ts::CdnotsPlusConfig =
        serde_json::from_str("{\"discovery\":{\"max_lag\":2}}")?;
    assert_eq!(partial.discovery.alpha, 0.01);
    assert_eq!(partial.discovery.priority, 1);
    Ok(())
}

#[test]
fn gpu_backends_fail_explicitly_when_not_compiled() {
    use causal_ts::backend::Backend;
    #[cfg(not(feature = "cuda"))]
    assert!(Backend::Cuda.device().is_err());
    #[cfg(not(feature = "metal"))]
    assert!(Backend::Metal.device().is_err());
    assert_eq!(Backend::name(&Backend::Cpu.device().unwrap()), "cpu");
}

#[test]
fn tensor_ci_matches_native_engine() -> Result<()> {
    use causal_ts::{
        accelerated::TensorCiEngine, backend::Backend, CiEngine, CiKind, ConditionalIndependence,
    };
    let (data, _) = causal_ts::synthetic::chain(200, 4, 19)?;
    for kind in [CiKind::ParCorr, CiKind::Gcmi] {
        let scalar = CiEngine::new(kind);
        let tensor = TensorCiEngine::new(kind, Backend::Cpu)?;
        for indices in [vec![0, 1], vec![0, 1, 2], vec![1, 3, 2, 0]] {
            let columns: Vec<_> = indices.iter().map(|&i| data.column(i)).collect();
            let a = scalar.test(&columns)?;
            let b = tensor.test(&columns)?;
            assert!((a.statistic - b.statistic).abs() < 1e-10);
            assert!((a.p_value.unwrap() - b.p_value.unwrap()).abs() < 1e-10);
        }
        let constant = vec![2.0; 200];
        let columns = vec![data.column(0), constant.as_slice(), data.column(1)];
        assert_eq!(
            serde_json::to_string(&scalar.test(&columns)?)?,
            serde_json::to_string(&tensor.test(&columns)?)?
        );
    }
    Ok(())
}

#[test]
fn ci_skeleton_matches_upstream_and_c_model_is_trimmed() -> Result<()> {
    use causal_ts::grace::{run_grace, GraceMode};
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let data = causal_ts::TimeSeries::from_csv(std::fs::File::open(root.join("var.csv"))?)?;
    let expected: Vec<Vec<Vec<u8>>> = serde_json::from_reader(std::fs::File::open(
        root.join("grace_skeleton_reference.json"),
    )?)?;
    let mut cfg = small_config();
    cfg.mode = GraceMode::Stability;
    cfg.include_c = true;
    cfg.max_lag = 2;
    cfg.max_epochs = 1;
    cfg.lambda_grid = Some(vec![0.01]);
    cfg.stability_use_ci_skeleton = true;
    cfg.use_skeleton_mask = true;
    let r = run_grace(&data, &causal_ts::CiEngine::default(), &cfg)?;
    assert_eq!(r.skeleton.unwrap().graph, expected);
    let (data, _) = causal_ts::synthetic::chain(64, 2, 11)?;
    let mut cfg = small_config();
    cfg.include_c = true;
    cfg.include_c_in_model = true;
    cfg.c_preset = causal_ts::data::TimeBasis::LinearQuad;
    cfg.model.include_instantaneous = true;
    cfg.max_epochs = 1;
    let r = run_grace(&data, &causal_ts::CiEngine::default(), &cfg)?;
    assert_eq!(r.causal_graph.var_names, data.names());
    assert_eq!(r.gate_values.len(), 2);
    assert_eq!(r.skeleton.unwrap().var_names.len(), 4);
    Ok(())
}
