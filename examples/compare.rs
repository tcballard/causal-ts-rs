use causal_ts::{
    run_cdnots, run_cdnots_plus, run_cedar, CdnotsConfig, CdnotsPlusConfig, CedarConfig, CiEngine,
    TimeSeries,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    fs::File,
    hint::black_box,
    io::{self, BufRead},
    time::Instant,
};

#[derive(Deserialize)]
struct Request {
    algorithm: String,
    input: String,
    config: Value,
    runs: usize,
    warmups: usize,
}

fn measure<T: Serialize>(
    runs: usize,
    warmups: usize,
    mut run: impl FnMut() -> causal_ts::Result<T>,
) -> Result<Value, Box<dyn std::error::Error>> {
    for _ in 0..warmups {
        black_box(run()?);
    }
    let mut times = Vec::with_capacity(runs);
    let mut output: Option<T> = None;
    for _ in 0..runs {
        drop(output.take());
        let start = Instant::now();
        let result = black_box(run()?);
        times.push(start.elapsed().as_secs_f64() * 1000.0);
        output = Some(result);
    }
    Ok(json!({"times_ms": times, "result": serde_json::to_value(output)?}))
}

fn execute(r: Request) -> Result<Value, Box<dyn std::error::Error>> {
    let data = TimeSeries::from_csv(File::open(&r.input)?)?;
    match r.algorithm.as_str() {
        "cedar" => {
            let c: CedarConfig = serde_json::from_value(r.config)?;
            measure(r.runs, r.warmups, || {
                run_cedar(&data, &CiEngine::default(), &c)
            })
        }
        "cdnots" => {
            let c: CdnotsConfig = serde_json::from_value(r.config)?;
            measure(r.runs, r.warmups, || {
                run_cdnots(&data, &CiEngine::default(), &c)
            })
        }
        "cdnots+" => {
            let c: CdnotsPlusConfig = serde_json::from_value(r.config)?;
            measure(r.runs, r.warmups, || {
                run_cdnots_plus(&data, &CiEngine::default(), &c)
            })
        }
        "grace" => {
            let c: causal_ts::grace::GraceConfig = serde_json::from_value(r.config)?;
            measure(r.runs, r.warmups, || {
                causal_ts::grace::run_grace(&data, &CiEngine::default(), &c)
            })
        }
        _ => Err("unknown algorithm".into()),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let threads = std::env::var("BENCH_THREADS")
        .unwrap_or_else(|_| "1".into())
        .parse()?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    for line in io::stdin().lock().lines() {
        let result = execute(serde_json::from_str(&line?)?);
        match result {
            Ok(v) => println!("{v}"),
            Err(e) => println!("{}", json!({"error": e.to_string()})),
        }
    }
    Ok(())
}
