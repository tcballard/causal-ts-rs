use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet,
    io::{Read, Write},
};

#[derive(Clone, Debug)]
pub struct TimeSeries {
    names: Vec<String>,
    columns: Vec<Vec<f64>>,
    rows: usize,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Normalization {
    #[default]
    Minmax,
    Zscore,
    None,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum TimeBasis {
    #[default]
    Linear,
    LinearQuad,
    LinearSin,
    LinearExp,
    Step,
    StepLinear,
}

impl TimeSeries {
    pub fn new(names: Vec<String>, columns: Vec<Vec<f64>>) -> Result<Self> {
        if names.is_empty() || names.len() != columns.len() {
            return Err(Error::InvalidData(
                "column names and data must have equal nonzero width".into(),
            ));
        }
        let mut seen = HashSet::new();
        if names
            .iter()
            .any(|name| name.trim().is_empty() || !seen.insert(name))
        {
            return Err(Error::InvalidData(
                "column names must be nonempty and unique".into(),
            ));
        }
        let rows = columns[0].len();
        if rows == 0 || columns.iter().any(|column| column.len() != rows) {
            return Err(Error::InvalidData(
                "columns must have equal nonzero length".into(),
            ));
        }
        if columns.iter().flatten().any(|v| v.is_infinite()) {
            return Err(Error::InvalidData(
                "infinite observations are not supported; use NaN for missing values".into(),
            ));
        }
        Ok(Self {
            names,
            columns,
            rows,
        })
    }

    pub fn from_rows(names: Vec<String>, rows: Vec<Vec<f64>>) -> Result<Self> {
        if rows.iter().any(|r| r.len() != names.len()) {
            return Err(Error::InvalidData(
                "row width differs from header width".into(),
            ));
        }
        let columns = (0..names.len())
            .map(|j| rows.iter().map(|r| r[j]).collect())
            .collect();
        Self::new(names, columns)
    }

    pub fn from_csv(reader: impl Read) -> Result<Self> {
        let mut reader = csv::ReaderBuilder::new()
            .trim(csv::Trim::All)
            .from_reader(reader);
        let names = reader
            .headers()?
            .iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let mut columns = vec![Vec::new(); names.len()];
        for (row, record) in reader.records().enumerate() {
            for (j, value) in record?.iter().enumerate() {
                let v = if value.is_empty()
                    || ["na", "nan", "null"].contains(&value.to_ascii_lowercase().as_str())
                {
                    f64::NAN
                } else {
                    value.parse::<f64>().map_err(|_| {
                        Error::InvalidData(format!(
                            "non-numeric value at row {}, column {}: {value:?}",
                            row + 2,
                            j + 1
                        ))
                    })?
                };
                columns[j].push(v);
            }
        }
        Self::new(names, columns)
    }

    pub fn write_csv(&self, writer: impl Write) -> Result<()> {
        let mut writer = csv::Writer::from_writer(writer);
        writer.write_record(&self.names)?;
        for row in 0..self.rows {
            writer.write_record(self.columns.iter().map(|c| {
                if c[row].is_nan() {
                    String::new()
                } else {
                    c[row].to_string()
                }
            }))?;
        }
        writer.flush()?;
        Ok(())
    }

    pub fn nrows(&self) -> usize {
        self.rows
    }
    pub fn ncols(&self) -> usize {
        self.columns.len()
    }
    pub fn names(&self) -> &[String] {
        &self.names
    }
    pub fn columns(&self) -> &[Vec<f64>] {
        &self.columns
    }
    pub fn column(&self, index: usize) -> &[f64] {
        &self.columns[index]
    }
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.columns[col][row]
    }

    pub fn index(&self, name: &str) -> Result<usize> {
        self.names
            .iter()
            .position(|s| s == name)
            .ok_or_else(|| Error::InvalidConfig(format!("unknown variable {name:?}")))
    }

    pub fn normalized(&self, method: Normalization) -> Self {
        let mut result = self.clone();
        for col in &mut result.columns {
            let values: Vec<_> = col.iter().copied().filter(|x| x.is_finite()).collect();
            if values.is_empty() {
                continue;
            }
            let (offset, scale) = match method {
                Normalization::None => continue,
                Normalization::Minmax => {
                    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
                    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    (min, max - min)
                }
                Normalization::Zscore => {
                    let mean = values.iter().sum::<f64>() / values.len() as f64;
                    (
                        mean,
                        (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                            / values.len() as f64)
                            .sqrt(),
                    )
                }
            };
            let scale = if scale == 0.0 { 1.0 } else { scale };
            for v in col {
                *v = (*v - offset) / scale;
            }
        }
        result
    }

    pub fn with_time_basis(&self, basis: TimeBasis) -> Result<Self> {
        let t: Vec<_> = (0..self.rows).map(|i| i as f64).collect();
        let step: Vec<_> = (0..self.rows)
            .map(|i| if i >= self.rows / 2 { 1.0 } else { 0.0 })
            .collect();
        let n = self.rows as f64;
        let extra = match basis {
            TimeBasis::Linear => vec![("C_lin", t)],
            TimeBasis::LinearQuad => vec![
                ("C_lin", t.clone()),
                ("C_quad", t.iter().map(|v| v * v / n).collect()),
            ],
            TimeBasis::LinearSin => vec![
                ("C_lin", t.clone()),
                (
                    "C_sin",
                    t.iter()
                        .map(|v| (std::f64::consts::TAU * v / n).sin())
                        .collect(),
                ),
            ],
            TimeBasis::LinearExp => vec![
                ("C_lin", t.clone()),
                ("C_exp", t.iter().map(|v| (v / n).exp() - 1.0).collect()),
            ],
            TimeBasis::Step => vec![("C_step", step)],
            TimeBasis::StepLinear => vec![("C_step", step), ("C_lin", t)],
        };
        let mut names = self.names.clone();
        let mut columns = self.columns.clone();
        for (name, col) in extra {
            names.push(name.into());
            columns.push(col);
        }
        Self::new(names, columns)
    }

    pub fn with_covariates(&self, covariates: &Self) -> Result<Self> {
        if self.nrows() != covariates.nrows() {
            return Err(Error::InvalidData(
                "covariates must have one row per observation".into(),
            ));
        }
        let mut names = self.names.clone();
        let mut columns = self.columns.clone();
        names.extend(covariates.names.clone());
        columns.extend(covariates.columns.clone());
        Self::new(names, columns)
    }
}
