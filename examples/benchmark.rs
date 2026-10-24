use causal_ts::{run_cedar, CedarConfig, CiEngine, TimeSeries};
use std::{fs::File, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let path = args
        .get(1)
        .ok_or("usage: benchmark DATA.csv [RUNS] [THREADS]")?;
    let runs: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(5);
    let threads: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(1);
    if runs == 0 || threads == 0 {
        return Err("runs and threads must be positive".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    let data = TimeSeries::from_csv(File::open(path)?)?;
    let config = CedarConfig {
        max_lag: 2,
        ..Default::default()
    };
    let ci = CiEngine::default();
    std::hint::black_box(run_cedar(&data, &ci, &config)?);
    let mut times = Vec::new();
    let mut edges = 0;
    for _ in 0..runs {
        let start = Instant::now();
        let result = std::hint::black_box(run_cedar(&data, &ci, &config)?);
        times.push(start.elapsed().as_secs_f64() * 1000.0);
        edges = result.edges.len();
    }
    times.sort_by(f64::total_cmp);
    println!(
        "{}",
        serde_json::json!({"rows":data.nrows(),"columns":data.ncols(),"runs":runs,"threads":threads,"max_lag":2,"median_ms":times[times.len()/2],"min_ms":times[0],"edge_count":edges,"times_ms":times})
    );
    Ok(())
}
