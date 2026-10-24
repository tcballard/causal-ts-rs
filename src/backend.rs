use crate::{Error, Result};
use candle_core::Device;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    #[default]
    Cpu,
    Auto,
    Cuda,
    Metal,
}

impl Backend {
    pub fn device(self) -> Result<Device> {
        match self {
            Self::Cpu=>Ok(Device::Cpu),
            Self::Auto=>Ok(Device::new_cuda(0).or_else(|_|Device::new_metal(0)).unwrap_or(Device::Cpu)),
            Self::Cuda=>Device::new_cuda(0).map_err(|e|Error::InvalidConfig(format!("CUDA is unavailable: {e}. Build with --features cuda and install the CUDA toolkit and driver."))),
            Self::Metal=>Device::new_metal(0).map_err(|e|Error::InvalidConfig(format!("Metal is unavailable: {e}. Build with --features metal on macOS."))),
        }
    }
    pub fn name(device: &Device) -> &'static str {
        match device {
            Device::Cpu => "cpu",
            Device::Cuda(_) => "cuda",
            Device::Metal(_) => "metal",
        }
    }
}
