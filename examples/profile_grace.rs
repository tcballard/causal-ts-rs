//! CPU phase profiling without modifying the production GRACE implementation.
use candle_core::Device;
pub use causal_ts::{graph, Error, Result, TimeSeries};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Deserialize;
use serde_json::{json, Value};
use std::{fs, fs::File, hint::black_box, time::Instant};

#[allow(dead_code)]
mod measured {
    include!("../src/grace/model.rs");
    include!("profile_grace/adam_profile.rs");
}

#[derive(Deserialize)]
struct Request {
    input: String,
    baseline_result: String,
    shared_output: String,
    blocks: usize,
    steps: usize,
    warmups: usize,
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()?;
    let request: Request = serde_json::from_reader(File::open(
        std::env::args().nth(1).ok_or("request path required")?,
    )?)?;
    let prior: Value = serde_json::from_reader(File::open(&request.baseline_result)?)?;
    let prior = &prior["result"];
    let skeleton: Vec<Vec<Vec<u8>>> = serde_json::from_value(prior["skeleton"]["graph"].clone())?;
    let lambda = prior["lambda_l0"].as_f64().ok_or("lambda missing")?;
    let lag = prior["config"]["max_lag"].as_u64().ok_or("lag missing")? as usize;
    let data = TimeSeries::from_csv(File::open(&request.input)?)?;
    let d = data.ncols();
    let device = Device::Cpu;
    let config = measured::ModelConfig::default();
    let windows = measured::prepare_data(&data, lag, true, &device)?;
    let batch = 128.min(windows.dim(0)?);
    let steps_per_epoch = windows.dim(0)?.div_ceil(batch);
    let x = windows.narrow(0, 0, batch)?.contiguous()?;
    let make_model = || -> Result<measured::GatedModel> {
        let mut model = measured::GatedModel::new(d, lag, &config, &device, 42)?;
        model.initialize_skeleton(&skeleton, true)?;
        Ok(model)
    };
    let initial = make_model()?;
    fs::write(
        &request.shared_output,
        serde_json::to_vec(&json!({
            "state": initial.state()?, "input": {"shape": x.dims(), "values": x.flatten_all()?.to_vec1::<f32>()?},
            "config": config, "skeleton": skeleton, "lambda": lambda, "steps_per_epoch": steps_per_epoch,
            "num_vars": d, "max_lag": lag, "blocks": request.blocks, "steps": request.steps, "warmups": request.warmups,
            "model_sha256": measured::MODEL_SHA256,
        }))?,
    )?;

    // Verify timed Adam against the public, uninstrumented library implementation.
    let checked = make_model()?;
    let reference_config = serde_json::from_value(serde_json::to_value(&config)?)?;
    let mut reference =
        causal_ts::grace::model::GatedModel::new(d, lag, &reference_config, &device, 42)?;
    reference.initialize_skeleton(&skeleton, true)?;
    let mut checked_adam = measured::Adam::new();
    let mut reference_adam = causal_ts::grace::model::Adam::new();
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    for _ in 0..5 {
        let u = measured::uniform_noise(d, lag + 1, &mut rng, &device)?;
        let (mu, sigma) = checked.forward(&x, Some(&u))?;
        let loss = checked.loss(&x, &mu, &sigma, lambda, steps_per_epoch, true)?;
        checked_adam.step_profiled(&checked, &loss)?;
        let (mu, sigma) = reference.forward(&x, Some(&u))?;
        let loss = reference.loss(&x, &mu, &sigma, lambda, steps_per_epoch, true)?;
        reference_adam.step(&reference, &loss)?;
        let expected = reference.state()?;
        for (name, value) in checked.state()? {
            assert_eq!(
                value.values, expected[&name].values,
                "instrumented Adam differs: {name}"
            );
        }
    }
    let mut blocks = vec![];
    for _ in 0..request.blocks {
        let model = make_model()?;
        let mut optimizer = measured::Adam::new();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut times = vec![];
        for step in 0..request.warmups + request.steps {
            let total_start = Instant::now();
            let (forward_ms, loss_ms, backward_ms, adam_ms);
            {
                let start = Instant::now();
                let u = measured::uniform_noise(d, lag + 1, &mut rng, &device)?;
                let (mu, sigma) = model.forward(&x, Some(&u))?;
                forward_ms = start.elapsed().as_secs_f64() * 1000.0;
                let start = Instant::now();
                let loss = model.loss(&x, &mu, &sigma, lambda, steps_per_epoch, true)?;
                assert!(black_box(loss.to_scalar::<f32>()?).is_finite());
                loss_ms = start.elapsed().as_secs_f64() * 1000.0;
                (backward_ms, adam_ms) = optimizer.step_profiled(&model, &loss)?;
            }
            let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
            if step >= request.warmups {
                times.push(
                    json!({"forward_ms": forward_ms, "loss_ms": loss_ms, "backward_ms": backward_ms,
                    "adam_ms": adam_ms, "total_ms": total_ms,
                    "other_ms": total_ms - forward_ms - loss_ms - backward_ms - adam_ms}),
                );
            }
        }
        blocks.push(times);
    }
    println!(
        "{}",
        json!({"implementation": "rust", "batch": batch, "num_vars": d, "max_lag": lag,
        "verified_update_steps": 5, "model_sha256": measured::MODEL_SHA256, "blocks": blocks})
    );
    Ok(())
}
