#![cfg(feature = "neural")]
use candle_core::{Device, Tensor};
use causal_ts::{
    accelerated::TensorCiEngine,
    backend::Backend,
    grace::{model::Adam, GatedModel, ModelConfig},
    CiEngine, CiKind, ConditionalIndependence, Result,
};

fn check(backend: Backend) -> Result<()> {
    let device = backend.device()?;
    assert_ne!(Backend::name(&device), "cpu");
    let config = ModelConfig {
        hidden_size: 4,
        decoder_hidden: 5,
        decoder_layers: 1,
        include_instantaneous: true,
        ..Default::default()
    };
    let cpu = GatedModel::new(2, 1, &config, &Device::Cpu, 101)?;
    let gpu = GatedModel::new(2, 1, &config, &device, 101)?;
    let values: Vec<f32> = (0..40).map(|i| (i as f32 * 0.3).sin()).collect();
    let x = Tensor::from_vec(values, (10, 2, 2), &Device::Cpu)?;
    let gx = x.to_device(&device)?;
    let u = Tensor::from_vec(
        vec![0.2f32, 0.8, 0.3, 0.7, 0.4, 0.6, 0.9, 0.1],
        (2, 2, 2),
        &Device::Cpu,
    )?;
    let (mu, sig) = cpu.forward(&x, Some(&u))?;
    let (gmu, gsig) = gpu.forward(&gx, Some(&u.to_device(&device)?))?;
    let loss = cpu.loss(&x, &mu, &sig, 0.02, 2, true)?;
    let gloss = gpu.loss(&gx, &gmu, &gsig, 0.02, 2, true)?;
    assert!((loss.to_scalar::<f32>()? - gloss.to_scalar::<f32>()?).abs() < 1e-4);
    Adam::new().step(&cpu, &loss)?;
    Adam::new().step(&gpu, &gloss)?;
    let gs = gpu.state()?;
    for (name, s) in cpu.state()? {
        for (a, b) in s.values.iter().zip(&gs[&name].values) {
            assert!((a - b).abs() < 1e-4, "{name}: {a} {b}");
        }
    }
    let (data, _) = causal_ts::synthetic::chain(200, 3, 33)?;
    let columns = vec![data.column(0), data.column(1), data.column(2)];
    for kind in [CiKind::ParCorr, CiKind::Gcmi] {
        let a = CiEngine::new(kind).test(&columns)?;
        let b = TensorCiEngine::new(kind, backend)?.test(&columns)?;
        assert!((a.statistic - b.statistic).abs() < 5e-4);
        assert!((a.p_value.unwrap() - b.p_value.unwrap()).abs() < 5e-4);
    }
    Ok(())
}

#[test]
#[ignore = "requires a CUDA build and GPU"]
fn cuda_forward_backward_adam_and_ci() -> Result<()> {
    check(Backend::Cuda)
}

#[test]
#[ignore = "requires a Metal build and Apple GPU"]
fn metal_forward_backward_adam_and_ci() -> Result<()> {
    check(Backend::Metal)
}
