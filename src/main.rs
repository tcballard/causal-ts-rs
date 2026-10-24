use causal_ts::{
    cdnots::{
        run_cdnots_plus_with_covariates, run_cdnots_with_covariates, CdnotsConfig, CdnotsPlusConfig,
    },
    cedar::{run_cedar_with_covariates, CedarConfig},
    ci::{CiEngine, CiKind},
    data::{Normalization, TimeBasis},
    graph::Graph,
    lag::LagMethod,
    synthetic, Error, Result, TimeSeries,
};
use clap::{Args, Parser, Subcommand, ValueEnum};
use std::{
    fs::{self, File},
    io::{self, BufReader, Write},
    path::{Path, PathBuf},
};

#[derive(Parser)]
#[command(version, about = "Native Rust causal discovery for time series")]
struct Cli {
    #[command(subcommand)]
    command: Command,
    #[arg(long, global = true)]
    threads: Option<usize>,
    #[arg(long, global=true, value_parser=["cpu","auto","cuda","metal"])]
    ci_backend: Option<String>,
}

#[derive(Subcommand)]
enum Command {
    #[command(about = "Discover a causal graph")]
    Discover(Discover),
    #[command(about = "Test conditional independence of two columns")]
    CiTest {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        x: String,
        #[arg(long)]
        y: String,
        #[arg(long, value_delimiter = ',')]
        given: Vec<String>,
        #[arg(long, value_enum, default_value = "parcorr")]
        test: CiKind,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    #[command(about = "Generate a seeded autoregressive chain and its ground-truth graph")]
    Generate {
        #[arg(long, default_value_t = 500)]
        samples: usize,
        #[arg(long, default_value_t = 3)]
        variables: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        truth: Option<PathBuf>,
    },
    #[command(about = "Evaluate directed lag-specific edges against a ground-truth graph")]
    Evaluate {
        #[arg(long)]
        graph: PathBuf,
        #[arg(long)]
        truth: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    #[command(about = "Print the complete default algorithm configuration")]
    Config {
        #[arg(long, value_enum, default_value = "cedar")]
        algorithm: Algorithm,
    },
}

#[derive(Clone, Copy, PartialEq, ValueEnum)]
enum Algorithm {
    Cedar,
    Cdnots,
    #[value(name = "cdnots+", alias = "cdnots-plus")]
    CdnotsPlus,
    #[cfg(feature = "neural")]
    Grace,
}

#[derive(Args)]
struct Discover {
    #[arg(long, value_enum, default_value = "cedar")]
    algorithm: Algorithm,
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    max_lag: Option<usize>,
    #[arg(long)]
    alpha: Option<f64>,
    #[arg(long, value_enum, default_value = "parcorr")]
    test: CiKind,
    #[arg(long, value_enum)]
    lag_method: Option<LagMethod>,
    #[arg(long, value_enum)]
    normalize: Option<Normalization>,
    #[arg(long)]
    no_prune: bool,
    #[arg(long)]
    include_lag0: bool,
    #[arg(long)]
    include_c: bool,
    #[arg(long, conflicts_with = "include_c")]
    no_c: bool,
    #[arg(long)]
    max_degree: Option<usize>,
    #[arg(long, value_enum)]
    c_preset: Option<TimeBasis>,
    #[arg(long)]
    covariates: Option<PathBuf>,
    #[arg(long)]
    estimate_ar_order: bool,
    #[arg(long)]
    no_autoreg: bool,
    #[arg(long)]
    target: Option<String>,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long, value_enum, default_value = "json")]
    format: OutputFormat,
    #[cfg(feature = "neural")]
    #[arg(long, value_enum)]
    mode: Option<causal_ts::grace::GraceMode>,
    #[cfg(feature = "neural")]
    #[arg(long, value_enum)]
    backend: Option<causal_ts::backend::Backend>,
    #[cfg(feature = "neural")]
    #[arg(long)]
    epochs: Option<usize>,
    #[cfg(feature = "neural")]
    #[arg(long)]
    batch_size: Option<usize>,
    #[cfg(feature = "neural")]
    #[arg(long)]
    skeleton: Option<PathBuf>,
    #[cfg(feature = "neural")]
    #[arg(long)]
    lambda: Option<f64>,
    #[cfg(feature = "neural")]
    #[arg(long)]
    stability_threshold: Option<f64>,
}

#[derive(Clone, Copy, ValueEnum)]
enum OutputFormat {
    Json,
    Dot,
    Edges,
}

fn read_data(path: &Path) -> Result<TimeSeries> {
    TimeSeries::from_csv(BufReader::new(File::open(path)?))
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    Ok(serde_json::from_reader(BufReader::new(File::open(path)?))?)
}

fn save(path: Option<&Path>, bytes: &[u8]) -> Result<()> {
    if let Some(path) = path {
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let temp = parent.join(format!(
            ".causal-ts-{}-{}.tmp",
            std::process::id(),
            rand::random::<u64>()
        ));
        let result = (|| -> Result<()> {
            let mut file = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp)?;
            file.write_all(bytes)?;
            file.sync_all()?;
            drop(file);
            fs::rename(&temp, path)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(temp);
        }
        result
    } else {
        let mut stdout = io::stdout().lock();
        stdout.write_all(bytes)?;
        Ok(())
    }
}

fn json<T: serde::Serialize>(value: &T) -> Result<Vec<u8>> {
    let mut b = serde_json::to_vec_pretty(value)?;
    b.push(b'\n');
    Ok(b)
}

fn main() {
    if let Err(error) = run(Cli::parse()) {
        if matches!(&error, Error::Io(e) if e.kind() == io::ErrorKind::BrokenPipe) {
            return;
        }
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<()> {
    let ci_backend = cli.ci_backend.as_deref();
    if let Some(n) = cli.threads {
        if n == 0 {
            return Err(Error::InvalidConfig("threads must be positive".into()));
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .map_err(|e| Error::InvalidConfig(e.to_string()))?;
    }
    match cli.command {
        Command::Config { algorithm } => save(
            None,
            &match algorithm {
                Algorithm::Cedar => json(&CedarConfig::default())?,
                Algorithm::Cdnots => json(&CdnotsConfig::default())?,
                Algorithm::CdnotsPlus => json(&CdnotsPlusConfig::default())?,
                #[cfg(feature = "neural")]
                Algorithm::Grace => json(&causal_ts::grace::GraceConfig::default())?,
            },
        ),
        Command::Discover(args) => {
            #[cfg(feature = "neural")]
            {
                if args.algorithm == Algorithm::Grace {
                    return discover_grace(args, ci_backend);
                }
                if args.mode.is_some()
                    || args.backend.is_some()
                    || args.epochs.is_some()
                    || args.batch_size.is_some()
                    || args.skeleton.is_some()
                    || args.lambda.is_some()
                    || args.stability_threshold.is_some()
                {
                    return Err(Error::InvalidConfig(
                        "neural training options require --algorithm grace".into(),
                    ));
                }
            }
            if matches!(args.algorithm, Algorithm::Cdnots | Algorithm::CdnotsPlus) {
                return discover_cdnots(args, ci_backend);
            }
            if args.max_degree.is_some() {
                return Err(Error::InvalidConfig("max-degree applies to CDNOTS".into()));
            }
            let data = read_data(&args.input)?;
            let mut config = args
                .config
                .as_ref()
                .map(|p| read_json::<CedarConfig>(p))
                .transpose()?
                .unwrap_or_default();
            if let Some(l) = args.max_lag {
                config.max_lag = l;
            }
            if let Some(a) = args.alpha {
                config.alpha_cond1 = a;
                config.alpha_cond2 = a;
            }
            if let Some(m) = args.lag_method {
                config.lag_method = m;
            }
            if let Some(m) = args.normalize {
                config.normalize = m;
            }
            if args.no_prune {
                config.prune = false;
            }
            if args.include_lag0 {
                config.include_lag0 = true;
            }
            if args.include_c {
                config.include_c = true;
            }
            if args.no_c {
                config.include_c = false;
            }
            if let Some(p) = args.c_preset {
                config.c_preset = p;
                config.include_c = true;
            }
            if args.estimate_ar_order {
                config.assume_ar1 = false;
            }
            if args.no_autoreg {
                config.include_autoreg = false;
            }
            if let Some(t) = args.target {
                config.target_var = Some(t);
            }
            if let Some(s) = args.seed {
                config.seed = s;
            }
            let covariates = args.covariates.as_ref().map(|p| read_data(p)).transpose()?;
            let result = run_cedar_with_covariates(
                &data,
                covariates.as_ref(),
                &selected_ci(args.test, ci_backend)?,
                &config,
            )?;
            let bytes = match args.format {
                OutputFormat::Json => json(&result)?,
                OutputFormat::Dot => result.causal_graph.to_dot()?.into_bytes(),
                OutputFormat::Edges => {
                    let mut writer = csv::Writer::from_writer(Vec::new());
                    writer.write_record([
                        "cause",
                        "effect",
                        "lag",
                        "statistic",
                        "p_value",
                        "origin",
                    ])?;
                    for e in &result.edges {
                        writer.write_record([
                            e.cause.clone(),
                            e.effect.clone(),
                            e.lag.to_string(),
                            e.statistic.map(|v| v.to_string()).unwrap_or_default(),
                            e.p_value.map(|v| v.to_string()).unwrap_or_default(),
                            e.origin.clone(),
                        ])?;
                    }
                    writer.into_inner().map_err(|e| Error::Io(e.into_error()))?
                }
            };
            save(args.output.as_deref(), &bytes)
        }
        Command::CiTest {
            input,
            x,
            y,
            given,
            test,
            output,
        } => {
            let data = read_data(&input)?;
            let z = given
                .iter()
                .map(|s| data.index(s))
                .collect::<Result<Vec<_>>>()?;
            let result = selected_ci(test, ci_backend)?.test_indices(
                &data,
                data.index(&x)?,
                data.index(&y)?,
                &z,
            )?;
            save(output.as_deref(), &json(&result)?)
        }
        Command::Generate {
            samples,
            variables,
            seed,
            output,
            truth,
        } => {
            if truth.as_ref() == Some(&output) {
                return Err(Error::InvalidConfig(
                    "data and truth output paths must differ".into(),
                ));
            }
            let (data, graph) = synthetic::chain(samples, variables, seed)?;
            let mut csv = Vec::new();
            data.write_csv(&mut csv)?;
            save(Some(&output), &csv)?;
            if let Some(p) = truth {
                save(Some(&p), &json(&graph)?)?;
            }
            Ok(())
        }
        Command::Evaluate {
            graph,
            truth,
            output,
        } => {
            let prediction: Graph = read_json(&graph)?;
            let truth: Graph = read_json(&truth)?;
            save(output.as_deref(), &json(&prediction.evaluate(&truth)?)?)
        }
    }
}

fn discover_cdnots(args: Discover, ci_backend: Option<&str>) -> Result<()> {
    if args.lag_method.is_some()
        || args.normalize.is_some()
        || args.no_prune
        || args.include_lag0
        || args.estimate_ar_order
        || args.no_autoreg
        || args.target.is_some()
        || args.seed.is_some()
    {
        return Err(Error::InvalidConfig(
            "CEDAR-specific options are not valid for CDNOTS".into(),
        ));
    }
    let data = read_data(&args.input)?;
    let mut plus = if args.algorithm == Algorithm::CdnotsPlus {
        Some(
            args.config
                .as_ref()
                .map(|p| read_json::<CdnotsPlusConfig>(p))
                .transpose()?
                .unwrap_or_default(),
        )
    } else {
        None
    };
    let mut plain = if plus.is_none() {
        args.config
            .as_ref()
            .map(|p| read_json::<CdnotsConfig>(p))
            .transpose()?
            .unwrap_or_default()
    } else {
        CdnotsConfig::default()
    };
    let config = if let Some(p) = &mut plus {
        &mut p.discovery
    } else {
        &mut plain
    };
    if let Some(l) = args.max_lag {
        config.max_lag = l;
        config.lag_list = None;
    }
    if let Some(a) = args.alpha {
        config.alpha = a;
    }
    if args.include_c {
        config.include_c = true;
    }
    if args.no_c {
        config.include_c = false;
    }
    if let Some(p) = args.c_preset {
        config.c_preset = p;
        config.include_c = true;
    }
    if let Some(d) = args.max_degree {
        config.max_degree = Some(d);
    }
    let covariates = args.covariates.as_ref().map(|p| read_data(p)).transpose()?;
    let ci = selected_ci(args.test, ci_backend)?;
    let result = if let Some(config) = &plus {
        run_cdnots_plus_with_covariates(&data, covariates.as_ref(), &ci, config)?
    } else {
        run_cdnots_with_covariates(&data, covariates.as_ref(), &ci, &plain)?
    };
    let bytes = match args.format {
        OutputFormat::Json => json(&result)?,
        OutputFormat::Dot => result.to_dot()?.into_bytes(),
        OutputFormat::Edges => {
            let mut w = csv::Writer::from_writer(Vec::new());
            w.write_record(["cause", "effect", "lag", "mark", "p_value"])?;
            for e in &result.edges {
                w.write_record([
                    e.cause.clone(),
                    e.effect.clone(),
                    e.lag.to_string(),
                    e.mark.clone(),
                    e.p_value.map(|p| p.to_string()).unwrap_or_default(),
                ])?;
            }
            w.into_inner().map_err(|e| Error::Io(e.into_error()))?
        }
    };
    save(args.output.as_deref(), &bytes)
}

#[cfg(feature = "neural")]
fn discover_grace(args: Discover, ci_backend: Option<&str>) -> Result<()> {
    use causal_ts::grace::{run_grace_with_skeleton, GraceConfig};
    if args.lag_method.is_some()
        || args.no_prune
        || args.estimate_ar_order
        || args.no_autoreg
        || args.target.is_some()
        || args.covariates.is_some()
    {
        return Err(Error::InvalidConfig(
            "unsupported option for GRACE; use its algorithm configuration".into(),
        ));
    }
    let data = read_data(&args.input)?;
    let mut config = args
        .config
        .as_ref()
        .map(|p| read_json::<GraceConfig>(p))
        .transpose()?
        .unwrap_or_default();
    if let Some(l) = args.max_lag {
        config.max_lag = l;
    }
    if let Some(a) = args.alpha {
        config.ci_alpha = a;
    }
    if let Some(d) = args.max_degree {
        config.ci_max_degree = Some(d);
    }
    if args.include_c {
        config.include_c = true;
    }
    if args.no_c {
        config.include_c = false;
    }
    if let Some(p) = args.c_preset {
        config.c_preset = p;
        config.include_c = true;
    }
    if let Some(n) = args.normalize {
        config.normalize = match n {
            Normalization::None => false,
            Normalization::Zscore => true,
            Normalization::Minmax => {
                return Err(Error::InvalidConfig(
                    "GRACE supports zscore or none normalization".into(),
                ))
            }
        };
    }
    if args.include_lag0 {
        config.model.include_instantaneous = true;
    }
    if let Some(s) = args.seed {
        config.seed = s;
    }
    if let Some(m) = args.mode {
        config.mode = m;
    }
    if let Some(b) = args.backend {
        config.backend = b;
    }
    if let Some(e) = args.epochs {
        config.max_epochs = e;
    }
    if let Some(b) = args.batch_size {
        config.batch_size = Some(b);
    }
    if let Some(l) = args.lambda {
        config.lambda_l0 = Some(l);
    }
    if let Some(t) = args.stability_threshold {
        config.stability_threshold = t;
    }
    let skeleton = args
        .skeleton
        .as_ref()
        .map(|p| read_json::<Graph>(p))
        .transpose()?;
    let result = run_grace_with_skeleton(
        &data,
        skeleton.as_ref(),
        &selected_ci(args.test, ci_backend)?,
        &config,
    )?;
    let bytes = match args.format {
        OutputFormat::Json => json(&result)?,
        OutputFormat::Dot => result.causal_graph.to_dot()?.into_bytes(),
        OutputFormat::Edges => {
            let mut w = csv::Writer::from_writer(Vec::new());
            w.write_record(["cause", "effect", "lag", "score", "origin"])?;
            for e in &result.edges {
                w.write_record([
                    e.cause.clone(),
                    e.effect.clone(),
                    e.lag.to_string(),
                    e.statistic.map(|v| v.to_string()).unwrap_or_default(),
                    e.origin.clone(),
                ])?;
            }
            w.into_inner().map_err(|e| Error::Io(e.into_error()))?
        }
    };
    save(args.output.as_deref(), &bytes)
}

struct SelectedCi(Box<dyn causal_ts::ConditionalIndependence>);
impl causal_ts::ConditionalIndependence for SelectedCi {
    fn name(&self) -> &str {
        self.0.name()
    }
    fn test(&self, columns: &[&[f64]]) -> Result<causal_ts::CiResult> {
        self.0.test(columns)
    }
    fn prepare<'a>(
        &self,
        columns: &'a [Vec<f64>],
    ) -> Option<Box<dyn causal_ts::ci::PreparedConditionalIndependence + 'a>> {
        self.0.prepare(columns)
    }
}
impl SelectedCi {
    fn test_indices(
        &self,
        data: &TimeSeries,
        x: usize,
        y: usize,
        z: &[usize],
    ) -> Result<causal_ts::CiResult> {
        if x == y || z.contains(&x) || z.contains(&y) {
            return Err(Error::InvalidConfig(
                "X, Y and conditioning variables must be distinct".into(),
            ));
        }
        let mut z = z.to_vec();
        z.sort_unstable();
        z.dedup();
        let mut columns = vec![data.column(x.min(y)), data.column(x.max(y))];
        columns.extend(z.iter().map(|&i| data.column(i)));
        self.0.test(&columns)
    }
}
fn selected_ci(kind: CiKind, backend: Option<&str>) -> Result<SelectedCi> {
    if let Some(backend) = backend {
        #[cfg(feature = "neural")]
        {
            use causal_ts::backend::Backend;
            let backend = match backend {
                "cpu" => Backend::Cpu,
                "auto" => Backend::Auto,
                "cuda" => Backend::Cuda,
                "metal" => Backend::Metal,
                _ => return Err(Error::InvalidConfig("unknown CI backend".into())),
            };
            return Ok(SelectedCi(Box::new(
                causal_ts::accelerated::TensorCiEngine::new(kind, backend)?,
            )));
        }
        #[cfg(not(feature = "neural"))]
        {
            let _ = backend;
            return Err(Error::InvalidConfig(
                "tensor CI backends require the neural feature".into(),
            ));
        }
    }
    Ok(SelectedCi(Box::new(CiEngine::new(kind))))
}
