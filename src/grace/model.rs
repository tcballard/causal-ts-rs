use crate::{graph::Cube, Error, Result};
use candle_core::{DType, Device, Tensor, Var, D};
use candle_nn::ops::sigmoid;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub decoder_hidden: usize,
    pub decoder_layers: usize,
    pub lambda_lag_group: f64,
    pub max_lags_per_pair: usize,
    pub per_pair_lag_budget: Option<Vec<Vec<usize>>>,
    pub temperature: f64,
    pub stretch_gamma: f64,
    pub stretch_zeta: f64,
    pub include_instantaneous: bool,
    pub lr: f64,
    pub weight_decay: f64,
    pub normalize_l0: bool,
    pub nonlinear_encoder: bool,
    pub encoder_hidden: usize,
    pub embed_dim: usize,
    pub basis_expansion: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            hidden_size: 64,
            decoder_hidden: 64,
            decoder_layers: 2,
            lambda_lag_group: 0.0,
            max_lags_per_pair: 1,
            per_pair_lag_budget: None,
            temperature: 2.0 / 3.0,
            stretch_gamma: -0.1,
            stretch_zeta: 1.1,
            include_instantaneous: false,
            lr: 1e-3,
            weight_decay: 1e-4,
            normalize_l0: true,
            nonlinear_encoder: false,
            encoder_hidden: 32,
            embed_dim: 8,
            basis_expansion: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorData {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

pub struct GatedModel {
    pub config: ModelConfig,
    pub num_vars: usize,
    pub max_lag: usize,
    pub device: Device,
    parameters: BTreeMap<String, Var>,
    mask: Tensor,
}

fn silu(x: &Tensor) -> candle_core::Result<Tensor> {
    x * sigmoid(x)?
}
fn softplus(x: &Tensor) -> candle_core::Result<Tensor> {
    x.relu()? + x.abs()?.neg()?.exp()?.affine(1.0, 1.0)?.log()?
}

impl GatedModel {
    pub fn new(
        d: usize,
        max_lag: usize,
        config: &ModelConfig,
        device: &Device,
        seed: u64,
    ) -> Result<Self> {
        if d == 0
            || max_lag == usize::MAX
            || config.hidden_size == 0
            || config.decoder_hidden == 0
            || config.encoder_hidden == 0
            || config.embed_dim == 0
            || config.decoder_layers > 64
            || !(config.temperature > 0.0 && config.temperature.is_finite())
            || !(config.stretch_gamma < 0.0 && config.stretch_gamma.is_finite())
            || !(config.stretch_zeta > 1.0 && config.stretch_zeta.is_finite())
            || !(config.lr > 0.0 && config.lr.is_finite())
            || config.weight_decay < 0.0
            || !config.weight_decay.is_finite()
            || config.lambda_lag_group < 0.0
            || !config.lambda_lag_group.is_finite()
        {
            return Err(Error::InvalidConfig(
                "invalid GRACE model dimensions or training parameters".into(),
            ));
        }
        if let Some(b) = &config.per_pair_lag_budget {
            if b.len() != d || b.iter().any(|r| r.len() != d) {
                return Err(Error::InvalidConfig(
                    "per-pair lag budgets must have shape [variables, variables]".into(),
                ));
            }
        }
        let l = max_lag + 1;
        let h = config.hidden_size;
        let mut parameters = BTreeMap::new();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut add = |name: String, shape: Vec<usize>, std: f64, uniform: bool| -> Result<()> {
            let n = shape
                .iter()
                .try_fold(1usize, |a, b| a.checked_mul(*b))
                .ok_or_else(|| Error::InvalidConfig("model dimensions overflow".into()))?;
            let values: Vec<f32> = (0..n)
                .map(|_| {
                    if std == 0.0 {
                        0.0
                    } else if uniform {
                        rng.gen_range(-std..std) as f32
                    } else {
                        let v: f64 = StandardNormal.sample(&mut rng);
                        (v * std) as f32
                    }
                })
                .collect();
            parameters.insert(name, Var::from_vec(values, shape, device)?);
            Ok(())
        };
        add("log_alpha".into(), vec![d, l, d], 0.0, false)?;
        if config.nonlinear_encoder {
            add(
                "encoder.embed.weight".into(),
                vec![d * l, config.embed_dim],
                1.0,
                false,
            )?;
            add(
                "encoder.net.0.weight".into(),
                vec![config.encoder_hidden, 1 + config.embed_dim],
                1.0 / ((1 + config.embed_dim) as f64).sqrt(),
                false,
            )?;
            add(
                "encoder.net.0.bias".into(),
                vec![config.encoder_hidden],
                0.0,
                false,
            )?;
            add(
                "encoder.net.2.weight".into(),
                vec![h, config.encoder_hidden],
                (6.0 / (h + config.encoder_hidden) as f64).sqrt(),
                true,
            )?;
            add("encoder.net.2.bias".into(), vec![h], 0.0, false)?;
        } else {
            let basis = if config.basis_expansion { 3 } else { 1 };
            add(
                "W".into(),
                vec![d, l, basis * h],
                (2.0 / basis as f64).sqrt(),
                false,
            )?;
            add("bias".into(), vec![d, l, h], 0.0, false)?;
        }
        for e in 0..d {
            let mut input = h;
            for layer in 0..=config.decoder_layers {
                let output = if layer == config.decoder_layers {
                    2
                } else {
                    config.decoder_hidden
                };
                let final_layer = layer == config.decoder_layers;
                let scale = if final_layer {
                    0.1 * (6.0 / (input + output) as f64).sqrt()
                } else {
                    1.0 / (input as f64).sqrt()
                };
                add(
                    format!("decoders.{e}.net.{}.weight", layer * 2),
                    vec![output, input],
                    scale,
                    final_layer,
                )?;
                add(
                    format!("decoders.{e}.net.{}.bias", layer * 2),
                    vec![output],
                    0.0,
                    false,
                )?;
                input = output;
            }
        }
        let mut mask = vec![1f32; d * l * d];
        for c in 0..d {
            for e in 0..d {
                if c == e || !config.include_instantaneous {
                    mask[(c * l + max_lag) * d + e] = 0.0;
                }
            }
        }
        parameters["log_alpha"].set(&Tensor::full(-0.5f32, (d, l, d), device)?)?;
        Ok(Self {
            config: config.clone(),
            num_vars: d,
            max_lag,
            device: device.clone(),
            parameters,
            mask: Tensor::from_vec(mask, (d, l, d), device)?,
        })
    }

    fn p(&self, name: &str) -> &Tensor {
        self.parameters[name].as_tensor()
    }
    fn linear(&self, x: &Tensor, prefix: &str) -> candle_core::Result<Tensor> {
        let w = self.p(&format!("{prefix}.weight"));
        let b = self.p(&format!("{prefix}.bias"));
        let n = x.elem_count() / x.dim(D::Minus1)?;
        let out = x.reshape((n, ()))?.matmul(&w.t()?)?.broadcast_add(b)?;
        let mut shape = x.dims().to_vec();
        *shape.last_mut().unwrap() = w.dim(0)?;
        out.reshape(shape)
    }

    pub fn initialize_skeleton(
        &mut self,
        skeleton: &[Vec<Vec<u8>>],
        hard_mask: bool,
    ) -> Result<()> {
        let d = self.num_vars;
        let l = self.max_lag + 1;
        if skeleton.len() != d
            || skeleton
                .iter()
                .any(|r| r.len() != d || r.iter().any(|v| v.len() != l || v.iter().any(|&b| b > 1)))
        {
            return Err(Error::InvalidConfig(
                "GRACE skeleton has invalid dimensions or values".into(),
            ));
        }
        let mut alpha = vec![-2f32; d * l * d];
        let mut mask = vec![0f32; d * l * d];
        for (c, row) in skeleton.iter().enumerate() {
            for (e, lags) in row.iter().enumerate() {
                for (lag, &edge) in lags.iter().enumerate() {
                    let i = (c * l + self.max_lag - lag) * d + e;
                    let on = edge != 0;
                    alpha[i] = if on { 0.5 } else { -2.0 };
                    mask[i] = f32::from(
                        (on || !hard_mask)
                            && (lag != 0 || (c != e && self.config.include_instantaneous)),
                    );
                }
            }
        }
        self.parameters["log_alpha"].set(&Tensor::from_vec(alpha, (d, l, d), &self.device)?)?;
        self.mask = Tensor::from_vec(mask, (d, l, d), &self.device)?;
        Ok(())
    }

    pub fn gates(&self, uniforms: Option<&Tensor>) -> Result<Tensor> {
        let a = self.p("log_alpha");
        let s = if let Some(u) = uniforms {
            if u.dims() != a.dims() {
                return Err(Error::InvalidData("gate noise has the wrong shape".into()));
            }
            let logistic = (&u.log()? - u.affine(-1.0, 1.0)?.log()?)?;
            sigmoid(&(&logistic + a)?.affine(1.0 / self.config.temperature, 0.0)?)?
        } else {
            sigmoid(a)?
        };
        Ok(s.affine(
            self.config.stretch_zeta - self.config.stretch_gamma,
            self.config.stretch_gamma,
        )?
        .clamp(0f32, 1f32)?
        .mul(&self.mask)?)
    }

    pub fn forward(&self, x: &Tensor, uniforms: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        let (batch, d, l) = x.dims3()?;
        if d != self.num_vars || l != self.max_lag + 1 || batch == 0 {
            return Err(Error::InvalidData(
                "GRACE input must have shape [batch, variables, max_lag+1]".into(),
            ));
        }
        let h = self.config.hidden_size;
        let gates = self.gates(uniforms)?;
        let encoded = if self.config.nonlinear_encoder {
            let emb = self
                .p("encoder.embed.weight")
                .reshape((d, l, self.config.embed_dim))?
                .unsqueeze(0)?
                .broadcast_as((batch, d, l, self.config.embed_dim))?;
            let input = Tensor::cat(&[x.unsqueeze(3)?, emb], 3)?;
            self.linear(
                &silu(&self.linear(&input, "encoder.net.0")?)?,
                "encoder.net.2",
            )?
        } else if self.config.basis_expansion {
            let basis = Tensor::stack(&[x.clone(), x.sqr()?, x.abs()?], 3)?.unsqueeze(4)?;
            let w = self.p("W").reshape((d, l, 3, h))?.unsqueeze(0)?;
            basis
                .broadcast_mul(&w)?
                .sum(3)?
                .broadcast_add(&self.p("bias").unsqueeze(0)?)?
        } else {
            x.unsqueeze(3)?
                .broadcast_mul(&self.p("W").unsqueeze(0)?)?
                .broadcast_add(&self.p("bias").unsqueeze(0)?)?
        };
        let aggregate = encoded
            .reshape((batch, d * l, h))?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch * h, d * l))?
            .matmul(&gates.reshape((d * l, d))?)?
            .reshape((batch, h, d))?
            .transpose(1, 2)?;
        let mut outputs = vec![];
        for e in 0..d {
            let mut v = aggregate.narrow(1, e, 1)?.squeeze(1)?;
            for layer in 0..=self.config.decoder_layers {
                v = self.linear(&v, &format!("decoders.{e}.net.{}", layer * 2))?;
                if layer < self.config.decoder_layers {
                    v = silu(&v)?;
                }
            }
            outputs.push(v);
        }
        let out = Tensor::stack(&outputs, 1)?;
        let mu = out.narrow(2, 0, 1)?.squeeze(2)?;
        let sigma = softplus(&out.narrow(2, 1, 1)?.squeeze(2)?)?.affine(1.0, 1e-6)?;
        Ok((mu, sigma))
    }

    pub fn loss(
        &self,
        x: &Tensor,
        mu: &Tensor,
        sigma: &Tensor,
        lambda: f64,
        steps_per_epoch: usize,
        training: bool,
    ) -> Result<Tensor> {
        if steps_per_epoch == 0 || lambda < 0.0 || !lambda.is_finite() {
            return Err(Error::InvalidConfig(
                "invalid loss regularization or step count".into(),
            ));
        }
        let target = x.narrow(2, self.max_lag, 1)?.squeeze(2)?;
        let sq = (&target - mu)?.sqr()?;
        let likelihood = (sq.div(&sigma.sqr()?.affine(2.0, 0.0)?)? + sigma.log()?)?
            .affine(1.0, 0.5 * (2.0 * std::f64::consts::PI).ln())?;
        let mut loss = likelihood.mean_all()?;
        if training {
            let offset = -self.config.temperature
                * (-self.config.stretch_gamma / self.config.stretch_zeta).ln();
            let probabilities =
                sigmoid(&self.p("log_alpha").affine(1.0, offset)?)?.mul(&self.mask)?;
            let scale = if self.config.normalize_l0 {
                1.0 / steps_per_epoch as f64
            } else {
                1.0
            };
            loss = (loss + probabilities.sum_all()?.affine(lambda * scale, 0.0)?)?;
            if self.config.lambda_lag_group > 0.0 {
                let counts = probabilities.sum(1)?;
                let budget = if let Some(b) = &self.config.per_pair_lag_budget {
                    Tensor::from_vec(
                        b.iter().flatten().map(|&n| n as f32).collect(),
                        (self.num_vars, self.num_vars),
                        &self.device,
                    )?
                } else {
                    Tensor::full(
                        self.config.max_lags_per_pair as f32,
                        (self.num_vars, self.num_vars),
                        &self.device,
                    )?
                };
                loss = (loss
                    + (counts - budget)?
                        .relu()?
                        .sum_all()?
                        .affine(lambda * self.config.lambda_lag_group, 0.0)?)?;
            }
        }
        Ok(loss)
    }

    pub fn gate_values(&self) -> Result<Cube<f64>> {
        let flat = self.gates(None)?.flatten_all()?.to_vec1::<f32>()?;
        let d = self.num_vars;
        let l = self.max_lag + 1;
        let mut out = Cube::new(d, l, 0.0);
        for c in 0..d {
            for e in 0..d {
                for lag in 0..l {
                    out[(c, e, lag)] = flat[(c * l + self.max_lag - lag) * d + e] as f64;
                }
            }
        }
        Ok(out)
    }

    pub fn state(&self) -> Result<BTreeMap<String, TensorData>> {
        self.parameters
            .iter()
            .map(|(name, p)| {
                Ok((
                    name.clone(),
                    TensorData {
                        shape: p.dims().to_vec(),
                        values: p.flatten_all()?.to_vec1::<f32>()?,
                    },
                ))
            })
            .collect()
    }
    pub fn load_state(&self, state: &BTreeMap<String, TensorData>) -> Result<()> {
        if state.len() != self.parameters.len()
            || state.iter().any(|(name, s)| {
                self.parameters
                    .get(name)
                    .is_none_or(|p| p.dims() != s.shape || p.elem_count() != s.values.len())
                    || s.values.iter().any(|v| !v.is_finite())
            })
        {
            return Err(Error::InvalidData(
                "model state names, shapes or values are invalid".into(),
            ));
        }
        for (name, s) in state {
            self.parameters[name].set(&Tensor::from_vec(
                s.values.clone(),
                s.shape.clone(),
                &self.device,
            )?)?;
        }
        Ok(())
    }
    pub fn gradients(&self, loss: &Tensor) -> Result<BTreeMap<String, TensorData>> {
        let grads = loss.backward()?;
        self.parameters
            .iter()
            .map(|(name, p)| {
                let g = grads.get(p).cloned().unwrap_or(p.zeros_like()?);
                Ok((
                    name.clone(),
                    TensorData {
                        shape: p.dims().to_vec(),
                        values: g.flatten_all()?.to_vec1::<f32>()?,
                    },
                ))
            })
            .collect()
    }
}

pub struct Adam {
    step: usize,
    states: BTreeMap<String, (Tensor, Tensor)>,
}
impl Default for Adam {
    fn default() -> Self {
        Self::new()
    }
}
impl Adam {
    pub fn new() -> Self {
        Self {
            step: 0,
            states: BTreeMap::new(),
        }
    }
    pub fn step(&mut self, model: &GatedModel, loss: &Tensor) -> Result<()> {
        let gradients = loss.backward()?;
        self.step += 1;
        let t = self.step as f64;
        for (name, p) in &model.parameters {
            let Some(gradient) = gradients.get(p) else {
                continue;
            };
            let gradient = (gradient + p.affine(model.config.weight_decay, 0.0)?)?.detach();
            let (m, v) = self
                .states
                .entry(name.clone())
                .or_insert((p.zeros_like()?, p.zeros_like()?));
            *m = (m.affine(0.9, 0.0)? + gradient.affine(0.1, 0.0)?)?.detach();
            *v = (v.affine(0.999, 0.0)? + gradient.sqr()?.affine(0.001, 0.0)?)?.detach();
            let denom = v
                .affine(1.0 / (1.0 - 0.999f64.powf(t)), 0.0)?
                .sqrt()?
                .affine(1.0, 1e-8)?;
            let delta = m
                .affine(model.config.lr / (1.0 - 0.9f64.powf(t)), 0.0)?
                .div(&denom)?;
            p.set(&(p.as_tensor() - delta)?.detach())?;
        }
        Ok(())
    }
}

pub fn uniform_noise(d: usize, l: usize, rng: &mut ChaCha8Rng, device: &Device) -> Result<Tensor> {
    Ok(Tensor::from_vec(
        (0..d * l * d)
            .map(|_| rng.gen_range(1e-6f32..1.0 - 1e-6f32))
            .collect(),
        (d, l, d),
        device,
    )?)
}

pub fn prepare_data(
    data: &crate::TimeSeries,
    max_lag: usize,
    normalize: bool,
    device: &Device,
) -> Result<Tensor> {
    if max_lag >= data.nrows() || data.columns().iter().flatten().any(|v| !v.is_finite()) {
        return Err(Error::InvalidData(
            "GRACE requires finite data and maximum lag below the number of observations".into(),
        ));
    }
    let d = data.ncols();
    let n = data.nrows();
    let columns: Vec<Vec<f32>> = data
        .columns()
        .iter()
        .map(|c| {
            let c: Vec<f32> = c.iter().map(|&v| v as f32).collect();
            if !normalize {
                return c;
            }
            let mean = c.iter().sum::<f32>() / n as f32;
            let sd = (c.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32).sqrt() + 1e-8;
            c.iter().map(|v| (v - mean) / sd).collect()
        })
        .collect();
    if columns.iter().flatten().any(|v| !v.is_finite()) {
        return Err(Error::InvalidData(
            "GRACE data cannot be represented as finite float32 values".into(),
        ));
    }
    let mut windows = Vec::with_capacity((n - max_lag) * d * (max_lag + 1));
    for row in max_lag..n {
        for col in &columns {
            windows.extend_from_slice(&col[row - max_lag..=row]);
        }
    }
    Ok(Tensor::from_vec(windows, (n - max_lag, d, max_lag + 1), device)?.to_dtype(DType::F32)?)
}
