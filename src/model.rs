//! Candle policy/value models.

use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor, Var};
use candle_nn::{
    Conv2d, Conv2dConfig, GroupNorm, Linear, Module, VarBuilder, VarMap, conv2d, group_norm, linear,
};

const TIC_TAC_TOE_INPUTS: usize = 18;
const TIC_TAC_TOE_HIDDEN: usize = 64;
const TIC_TAC_TOE_ACTIONS: usize = 9;
const OTHELLO_CHANNELS: usize = 128;
const OTHELLO_RESIDUAL_BLOCKS: usize = 10;
const OTHELLO_NORM_GROUPS: usize = 8;
const OTHELLO_ACTIONS: usize = 65;

/// A small policy/value MLP for tic-tac-toe inference tests.
pub struct TicTacToeNetwork {
    trunk: Linear,
    policy: Linear,
    value: Linear,
    _variables: VarMap,
}

impl TicTacToeNetwork {
    /// Creates a reproducibly initialized random network.
    pub fn new(device: &Device, seed: u64) -> Result<Self> {
        let mut variables = VarMap::new();
        let vb = VarBuilder::from_varmap(&variables, DType::F32, device);
        let trunk = linear(TIC_TAC_TOE_INPUTS, TIC_TAC_TOE_HIDDEN, vb.pp("trunk"))?;
        let policy = linear(TIC_TAC_TOE_HIDDEN, TIC_TAC_TOE_ACTIONS, vb.pp("policy"))?;
        let value = linear(TIC_TAC_TOE_HIDDEN, 1, vb.pp("value"))?;

        let mut random = SplitMix64(seed);
        initialize_linear(
            &mut variables,
            "trunk",
            TIC_TAC_TOE_INPUTS,
            TIC_TAC_TOE_HIDDEN,
            &mut random,
            device,
        )?;
        initialize_linear(
            &mut variables,
            "policy",
            TIC_TAC_TOE_HIDDEN,
            TIC_TAC_TOE_ACTIONS,
            &mut random,
            device,
        )?;
        initialize_linear(
            &mut variables,
            "value",
            TIC_TAC_TOE_HIDDEN,
            1,
            &mut random,
            device,
        )?;

        Ok(Self {
            trunk,
            policy,
            value,
            _variables: variables,
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<(Tensor, Tensor)> {
        let batch = input.dim(0)?;
        let input = input.reshape((batch, TIC_TAC_TOE_INPUTS))?;
        let hidden = self.trunk.forward(&input)?.relu()?;
        let policy_logits = self.policy.forward(&hidden)?;
        let value = self.value.forward(&hidden)?.tanh()?;
        Ok((policy_logits, value))
    }
}

struct ResidualBlock {
    conv1: Conv2d,
    norm1: GroupNorm,
    conv2: Conv2d,
    norm2: GroupNorm,
}

struct OthelloModel {
    stem: Conv2d,
    stem_norm: GroupNorm,
    blocks: Vec<ResidualBlock>,
    policy_conv: Conv2d,
    policy: Linear,
    value_conv: Conv2d,
    value_hidden: Linear,
    value: Linear,
}

/// A residual policy/value network for the 8x8 Othello board.
pub struct OthelloNetwork {
    model: OthelloModel,
    variables: VarMap,
    device: Device,
}

fn build_othello_model(vb: VarBuilder<'_>) -> Result<OthelloModel> {
    let padded = Conv2dConfig {
        padding: 1,
        ..Conv2dConfig::default()
    };
    let stem = conv2d(2, OTHELLO_CHANNELS, 3, padded, vb.pp("stem.conv"))?;
    let stem_norm = group_norm(
        OTHELLO_NORM_GROUPS,
        OTHELLO_CHANNELS,
        1e-5,
        vb.pp("stem.norm"),
    )?;
    let mut blocks = Vec::with_capacity(OTHELLO_RESIDUAL_BLOCKS);
    for block in 0..OTHELLO_RESIDUAL_BLOCKS {
        let block_vb = vb.pp(format!("blocks.{block}"));
        blocks.push(ResidualBlock {
            conv1: conv2d(
                OTHELLO_CHANNELS,
                OTHELLO_CHANNELS,
                3,
                padded,
                block_vb.pp("conv1"),
            )?,
            norm1: group_norm(
                OTHELLO_NORM_GROUPS,
                OTHELLO_CHANNELS,
                1e-5,
                block_vb.pp("norm1"),
            )?,
            conv2: conv2d(
                OTHELLO_CHANNELS,
                OTHELLO_CHANNELS,
                3,
                padded,
                block_vb.pp("conv2"),
            )?,
            norm2: group_norm(
                OTHELLO_NORM_GROUPS,
                OTHELLO_CHANNELS,
                1e-5,
                block_vb.pp("norm2"),
            )?,
        });
    }
    let policy_conv = conv2d(
        OTHELLO_CHANNELS,
        2,
        1,
        Conv2dConfig::default(),
        vb.pp("policy.conv"),
    )?;
    let policy = linear(2 * 8 * 8, OTHELLO_ACTIONS, vb.pp("policy.linear"))?;
    let value_conv = conv2d(
        OTHELLO_CHANNELS,
        1,
        1,
        Conv2dConfig::default(),
        vb.pp("value.conv"),
    )?;
    let value_hidden = linear(8 * 8, OTHELLO_CHANNELS, vb.pp("value.hidden"))?;
    let value = linear(OTHELLO_CHANNELS, 1, vb.pp("value.output"))?;

    Ok(OthelloModel {
        stem,
        stem_norm,
        blocks,
        policy_conv,
        policy,
        value_conv,
        value_hidden,
        value,
    })
}

impl OthelloModel {
    fn forward(&self, input: &Tensor) -> Result<(Tensor, Tensor)> {
        let batch = input.dim(0)?;
        let mut hidden = self.stem_norm.forward(&self.stem.forward(input)?)?.relu()?;
        for block in &self.blocks {
            let residual = hidden.clone();
            hidden = block
                .norm1
                .forward(&block.conv1.forward(&hidden)?)?
                .relu()?;
            hidden = block.norm2.forward(&block.conv2.forward(&hidden)?)?;
            hidden = (&hidden + &residual)?.relu()?;
        }
        let policy = self
            .policy_conv
            .forward(&hidden)?
            .relu()?
            .reshape((batch, 2 * 8 * 8))?;
        let policy = self.policy.forward(&policy)?;
        let value = self
            .value_conv
            .forward(&hidden)?
            .relu()?
            .reshape((batch, 8 * 8))?;
        let value = self.value_hidden.forward(&value)?.relu()?;
        let value = self.value.forward(&value)?.tanh()?;
        Ok((policy, value))
    }
}

impl OthelloNetwork {
    pub fn new(device: &Device, seed: u64) -> Result<Self> {
        let mut variables = VarMap::new();
        let model = build_othello_model(VarBuilder::from_varmap(&variables, DType::F32, device))?;

        let mut random = SplitMix64(seed);
        initialize_conv2d(
            &mut variables,
            "stem.conv",
            2,
            OTHELLO_CHANNELS,
            3,
            &mut random,
            device,
        )?;
        for block in 0..OTHELLO_RESIDUAL_BLOCKS {
            for conv in ["conv1", "conv2"] {
                initialize_conv2d(
                    &mut variables,
                    &format!("blocks.{block}.{conv}"),
                    OTHELLO_CHANNELS,
                    OTHELLO_CHANNELS,
                    3,
                    &mut random,
                    device,
                )?;
            }
        }
        initialize_conv2d(
            &mut variables,
            "policy.conv",
            OTHELLO_CHANNELS,
            2,
            1,
            &mut random,
            device,
        )?;
        initialize_linear(
            &mut variables,
            "policy.linear",
            2 * 8 * 8,
            OTHELLO_ACTIONS,
            &mut random,
            device,
        )?;
        initialize_conv2d(
            &mut variables,
            "value.conv",
            OTHELLO_CHANNELS,
            1,
            1,
            &mut random,
            device,
        )?;
        initialize_linear(
            &mut variables,
            "value.hidden",
            8 * 8,
            OTHELLO_CHANNELS,
            &mut random,
            device,
        )?;
        initialize_linear(
            &mut variables,
            "value.output",
            OTHELLO_CHANNELS,
            1,
            &mut random,
            device,
        )?;

        Ok(Self {
            model,
            variables,
            device: device.clone(),
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<(Tensor, Tensor)> {
        self.model.forward(input)
    }

    pub const fn variables(&self) -> &VarMap {
        &self.variables
    }

    pub fn named_variables(&self) -> Vec<(String, Var)> {
        let variables = self.variables.data().lock().expect("variable map poisoned");
        let mut variables = variables
            .iter()
            .map(|(name, variable)| (name.clone(), variable.clone()))
            .collect::<Vec<_>>();
        variables.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        variables
    }

    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        self.variables.save(path)
    }

    pub fn load(path: impl AsRef<std::path::Path>, device: &Device) -> Result<Self> {
        let mut network = Self::new(device, 0)?;
        network.variables.load(path)?;
        Ok(network)
    }

    pub fn load_weights(&mut self, path: impl AsRef<std::path::Path>) -> Result<()> {
        self.variables.load(path)
    }

    pub fn detached(&self) -> Self {
        let tensors = self
            .named_variables()
            .into_iter()
            .map(|(name, variable)| (name, variable.as_tensor().detach()))
            .collect::<HashMap<_, _>>();
        let model =
            build_othello_model(VarBuilder::from_tensors(tensors, DType::F32, &self.device))
                .expect("detached model uses tensors from a valid Othello model");
        Self {
            model,
            variables: VarMap::new(),
            device: self.device.clone(),
        }
    }
}

fn initialize_conv2d(
    variables: &mut VarMap,
    name: &str,
    inputs: usize,
    outputs: usize,
    kernel: usize,
    random: &mut SplitMix64,
    device: &Device,
) -> Result<()> {
    let fan_in = inputs * kernel * kernel;
    let bound = (6.0 / fan_in as f32).sqrt();
    let weights = (0..outputs * fan_in)
        .map(|_| random.next_f32(-bound, bound))
        .collect::<Vec<_>>();
    let weights = Tensor::from_vec(weights, (outputs, inputs, kernel, kernel), device)?;
    let bias_bound = 1.0 / (fan_in as f32).sqrt();
    let bias = (0..outputs)
        .map(|_| random.next_f32(-bias_bound, bias_bound))
        .collect::<Vec<_>>();
    let bias = Tensor::from_vec(bias, outputs, device)?;
    variables.set_one(format!("{name}.weight"), &weights)?;
    variables.set_one(format!("{name}.bias"), &bias)?;
    Ok(())
}

fn initialize_linear(
    variables: &mut VarMap,
    name: &str,
    inputs: usize,
    outputs: usize,
    random: &mut SplitMix64,
    device: &Device,
) -> Result<()> {
    let bound = (6.0 / inputs as f32).sqrt();
    let weights = (0..inputs * outputs)
        .map(|_| random.next_f32(-bound, bound))
        .collect::<Vec<_>>();
    let weights = Tensor::from_vec(weights, (outputs, inputs), device)?;
    let bias_bound = 1.0 / (inputs as f32).sqrt();
    let bias = (0..outputs)
        .map(|_| random.next_f32(-bias_bound, bias_bound))
        .collect::<Vec<_>>();
    let bias = Tensor::from_vec(bias, outputs, device)?;
    variables.set_one(format!("{name}.weight"), &weights)?;
    variables.set_one(format!("{name}.bias"), &bias)?;
    Ok(())
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn next_f32(&mut self, min: f32, max: f32) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1_u32 << 24) as f32;
        min + unit * (max - min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_initialization_is_reproducible() {
        let input = Tensor::zeros((2, 2, 3, 3), DType::F32, &Device::Cpu).unwrap();
        let first = TicTacToeNetwork::new(&Device::Cpu, 7).unwrap();
        let second = TicTacToeNetwork::new(&Device::Cpu, 7).unwrap();
        let (first_policy, first_value) = first.forward(&input).unwrap();
        let (second_policy, second_value) = second.forward(&input).unwrap();

        assert_eq!(first_policy.dims(), [2, 9]);
        assert_eq!(first_value.dims(), [2, 1]);
        let first_policy = first_policy
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(first_policy.iter().any(|&logit| logit != 0.0));
        assert_eq!(
            first_policy,
            second_policy
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        );
        assert_eq!(
            first_value.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            second_value
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        );
        assert!(
            first_value
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .all(|value| (-1.0..=1.0).contains(&value))
        );
    }

    #[test]
    fn trainable_and_detached_othello_outputs_are_equal() {
        let input = Tensor::zeros((2, 2, 8, 8), DType::F32, &Device::Cpu).unwrap();
        let network = OthelloNetwork::new(&Device::Cpu, 7).unwrap();
        let (policy, value) = network.forward(&input).unwrap();
        let (detached_policy, detached_value) = network.detached().forward(&input).unwrap();

        assert_eq!(policy.dims(), [2, OTHELLO_ACTIONS]);
        assert_eq!(value.dims(), [2, 1]);
        assert_eq!(
            policy.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            detached_policy
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        );
        assert_eq!(
            value.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            detached_value
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        );
        assert!(
            value
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .all(|value| value.is_finite() && (-1.0..=1.0).contains(&value))
        );
    }

    #[test]
    fn othello_parameter_names_match_checkpoint_schema() {
        let network = OthelloNetwork::new(&Device::Cpu, 7).unwrap();
        let mut expected = vec![
            "stem.conv.bias".to_string(),
            "stem.conv.weight".to_string(),
            "stem.norm.bias".to_string(),
            "stem.norm.weight".to_string(),
        ];
        for block in 0..OTHELLO_RESIDUAL_BLOCKS {
            for layer in ["conv1", "conv2", "norm1", "norm2"] {
                expected.push(format!("blocks.{block}.{layer}.bias"));
                expected.push(format!("blocks.{block}.{layer}.weight"));
            }
        }
        for layer in [
            "policy.conv",
            "policy.linear",
            "value.conv",
            "value.hidden",
            "value.output",
        ] {
            expected.push(format!("{layer}.bias"));
            expected.push(format!("{layer}.weight"));
        }
        expected.sort_unstable();

        let actual = network
            .named_variables()
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>();
        assert_eq!(actual.len(), 94);
        assert_eq!(actual, expected);
    }

    #[test]
    fn detached_othello_model_observes_original_parameter_updates() {
        let input = Tensor::zeros((1, 2, 8, 8), DType::F32, &Device::Cpu).unwrap();
        let network = OthelloNetwork::new(&Device::Cpu, 7).unwrap();
        let detached = network.detached();
        let before = detached.forward(&input).unwrap().0;
        let policy_bias = network
            .named_variables()
            .into_iter()
            .find(|(name, _)| name == "policy.linear.bias")
            .unwrap()
            .1;
        policy_bias
            .set(&(policy_bias.as_tensor() + 1.0).unwrap())
            .unwrap();
        let after = detached.forward(&input).unwrap().0;

        let difference = (after - before)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            difference
                .into_iter()
                .all(|difference| (difference - 1.0).abs() < 1e-5)
        );
    }

    #[test]
    fn detached_othello_inference_does_not_gradient_model_parameters() {
        let input = Tensor::zeros((1, 2, 8, 8), DType::F32, &Device::Cpu).unwrap();
        let network = OthelloNetwork::new(&Device::Cpu, 7).unwrap();
        let detached = network.detached();
        let (policy, value) = detached.forward(&input).unwrap();
        let loss = (policy.sum_all().unwrap() + value.sum_all().unwrap()).unwrap();
        let gradients = loss.backward().unwrap();

        for (name, variable) in network.named_variables() {
            assert!(
                gradients.get(&variable).is_none(),
                "detached inference produced a gradient for {name}"
            );
        }
    }
}
