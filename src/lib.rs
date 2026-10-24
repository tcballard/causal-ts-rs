#[cfg(feature = "neural")]
pub mod accelerated;
#[cfg(feature = "neural")]
pub mod backend;
pub mod cdnots;
pub mod cedar;
pub mod ci;
pub mod data;
#[cfg(feature = "neural")]
pub mod grace;
pub mod graph;
pub mod lag;
pub mod stats;
pub mod synthetic;

pub use cdnots::{run_cdnots, CdnotsConfig, CdnotsResult};
pub use cdnots::{run_cdnots_plus, CdnotsPlusConfig};
pub use cedar::{run_cedar, CedarConfig, CedarResult};
pub use ci::{CiEngine, CiKind, CiResult, ConditionalIndependence};
pub use data::TimeSeries;
pub use graph::{Edge, EdgeConstraint, Graph};

pub const UPSTREAM_REVISION: &str = "0d1fae061b2445a51d0c8b75fc565d897dd54fb7";

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[cfg(feature = "neural")]
    #[error(transparent)]
    Tensor(#[from] candle_core::Error),
    #[error("{0}")]
    InvalidData(String),
    #[error("{0}")]
    InvalidConfig(String),
    #[error("numerical calculation failed: {0}")]
    Numerical(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Csv(#[from] csv::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
