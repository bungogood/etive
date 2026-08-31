//! Burn policy/value models.

use std::path::Path;
use std::sync::Mutex;

use burn::module::{AutodiffModule, Module, ModuleMapper, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{GroupNorm, GroupNormConfig, Linear, LinearConfig, PaddingConfig2d};
use burn::store::{ModuleRecord, RecordError};
use burn::tensor::{Device, FloatDType, Tensor, activation};
use serde::{Deserialize, Serialize};

use super::Board as OthelloBoard;
use crate::game::Game;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OthelloModelConfig {
    pub channels: usize,
    pub residual_blocks: usize,
    pub norm_groups: usize,
}

impl OthelloModelConfig {
    pub const LEGACY: Self = Self {
        channels: 128,
        residual_blocks: 10,
        norm_groups: 8,
    };

    pub const WEEKEND: Self = Self {
        channels: 64,
        residual_blocks: 4,
        norm_groups: 8,
    };

    pub fn validate(self) -> Result<(), &'static str> {
        if self.channels == 0
            || self.residual_blocks == 0
            || self.norm_groups == 0
            || !self.channels.is_multiple_of(self.norm_groups)
        {
            return Err(
                "model channels and blocks must be positive and channels divisible by norm groups",
            );
        }
        Ok(())
    }
}

impl Default for OthelloModelConfig {
    fn default() -> Self {
        Self::LEGACY
    }
}

#[derive(Module, Debug)]
struct ResidualBlock {
    conv1: Conv2d,
    norm1: GroupNorm,
    conv2: Conv2d,
    norm2: GroupNorm,
}

impl ResidualBlock {
    fn new(device: &Device, config: OthelloModelConfig) -> Self {
        let convolution = || {
            Conv2dConfig::new([config.channels, config.channels], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device)
        };
        let normalization =
            || GroupNormConfig::new(config.norm_groups, config.channels).init(device);
        Self {
            conv1: convolution(),
            norm1: normalization(),
            conv2: convolution(),
            norm2: normalization(),
        }
    }

    fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let residual = input.clone();
        let hidden = activation::relu(self.norm1.forward(self.conv1.forward(input)));
        activation::relu(self.norm2.forward(self.conv2.forward(hidden)) + residual)
    }
}

/// A residual policy/value network for the 8x8 Othello board.
#[derive(Module, Debug)]
pub struct OthelloNetwork {
    stem: Conv2d,
    stem_norm: GroupNorm,
    blocks: Vec<ResidualBlock>,
    policy_conv: Conv2d,
    policy: Linear,
    value_conv: Conv2d,
    value_hidden: Linear,
    value: Linear,
}

static MODEL_INITIALIZATION_LOCK: Mutex<()> = Mutex::new(());

impl OthelloNetwork {
    /// Creates a reproducibly initialized random network.
    pub fn new(device: &Device, seed: u64) -> Self {
        Self::new_with_config(device, seed, OthelloModelConfig::LEGACY)
    }

    pub fn new_with_config(device: &Device, seed: u64, config: OthelloModelConfig) -> Self {
        config
            .validate()
            .expect("valid Othello model configuration");
        let _initialization = MODEL_INITIALIZATION_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        device.seed(seed);
        let network = Self {
            stem: Conv2dConfig::new([2, config.channels], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            stem_norm: GroupNormConfig::new(config.norm_groups, config.channels).init(device),
            blocks: (0..config.residual_blocks)
                .map(|_| ResidualBlock::new(device, config))
                .collect(),
            policy_conv: Conv2dConfig::new([config.channels, 2], [1, 1]).init(device),
            policy: LinearConfig::new(2 * 8 * 8, OthelloBoard::ACTION_COUNT).init(device),
            value_conv: Conv2dConfig::new([config.channels, 1], [1, 1]).init(device),
            value_hidden: LinearConfig::new(8 * 8, config.channels).init(device),
            value: LinearConfig::new(config.channels, 1).init(device),
        };
        let _ = network.clone().into_record();
        device
            .sync()
            .expect("model parameter initialization must complete");
        network
    }

    pub fn forward(&self, input: Tensor<4>) -> (Tensor<2>, Tensor<2>) {
        let [batch, _, _, _] = input.dims();
        let mut hidden = activation::relu(self.stem_norm.forward(self.stem.forward(input)));
        for block in &self.blocks {
            hidden = block.forward(hidden);
        }

        let policy = activation::relu(self.policy_conv.forward(hidden.clone()));
        let policy = self.policy.forward(policy.reshape([batch, 2 * 8 * 8]));
        let value = activation::relu(self.value_conv.forward(hidden));
        let value = activation::relu(self.value_hidden.forward(value.reshape([batch, 8 * 8])));
        let value = self.value.forward(value).tanh();
        (policy, value)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), RecordError> {
        self.clone().into_record().save(path)
    }

    pub fn load(path: impl AsRef<Path>, device: &Device) -> Result<Self, RecordError> {
        let path = path.as_ref();
        let config = Self::checkpoint_config(path).unwrap_or(OthelloModelConfig::LEGACY);
        Self::load_with_config(path, device, config)
    }

    pub fn load_with_config(
        path: impl AsRef<Path>,
        device: &Device,
        config: OthelloModelConfig,
    ) -> Result<Self, RecordError> {
        let record = ModuleRecord::load(path)?;
        Ok(Self::new_with_config(device, 0, config).load_record(record))
    }

    pub fn checkpoint_config(path: impl AsRef<Path>) -> Option<OthelloModelConfig> {
        let path = path.as_ref().parent()?.join("model.toml");
        let source = std::fs::read_to_string(path).ok()?;
        toml::from_str(&source).ok()
    }

    /// Returns an inference-only snapshot of this network.
    pub fn detached(&self) -> Self {
        self.valid()
    }

    /// Casts every floating-point parameter to the requested dtype.
    pub fn cast_float(self, dtype: FloatDType) -> Self {
        struct CastMapper(FloatDType);

        impl ModuleMapper for CastMapper {
            fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
                let (id, tensor, mapper) = param.consume();
                Param::from_mapped_value(id, tensor.cast(self.0), mapper)
            }
        }

        self.map(&mut CastMapper(dtype))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "cuda")]
    fn assert_finite<const D: usize>(name: &str, tensor: &Tensor<D>) {
        let values = tensor.clone().into_data().try_to_vec_as::<f32>().unwrap();
        let (minimum, maximum) = values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .fold(
                (f32::INFINITY, f32::NEG_INFINITY),
                |(minimum, maximum), value| (minimum.min(value), maximum.max(value)),
            );
        let non_finite = values.iter().filter(|value| !value.is_finite()).count();
        eprintln!("{name}: min={minimum:.6} max={maximum:.6} non_finite={non_finite}");
        assert!(non_finite == 0, "{name} contained a non-finite value",);
    }

    fn values(tensor: Tensor<2>) -> Vec<f32> {
        tensor.into_data().try_to_vec::<f32>().unwrap()
    }

    #[test]
    fn seeded_initialization_is_reproducible() {
        let device = Device::flex();
        let input = Tensor::zeros([2, 2, 8, 8], &device);
        let first = OthelloNetwork::new_with_config(&device, 7, OthelloModelConfig::WEEKEND);
        let second = OthelloNetwork::new_with_config(&device, 7, OthelloModelConfig::WEEKEND);
        let (first_policy, first_value) = first.forward(input.clone());
        let (second_policy, second_value) = second.forward(input);

        assert_eq!(first_policy.dims(), [2, OthelloBoard::ACTION_COUNT]);
        assert_eq!(first_value.dims(), [2, 1]);
        let first_policy = values(first_policy);
        assert!(first_policy.iter().any(|&logit| logit != 0.0));
        assert_eq!(first_policy, values(second_policy));
        assert_eq!(values(first_value), values(second_value));
    }

    #[test]
    fn trainable_and_detached_othello_outputs_are_equal() {
        let device = Device::flex();
        let network = OthelloNetwork::new(&device.clone().autodiff(), 7);
        let (policy, value) =
            network.forward(Tensor::zeros([2, 2, 8, 8], &device.clone().autodiff()));
        let (detached_policy, detached_value) = network
            .detached()
            .forward(Tensor::zeros([2, 2, 8, 8], &device));

        assert_eq!(policy.dims(), [2, OthelloBoard::ACTION_COUNT]);
        assert_eq!(value.dims(), [2, 1]);
        assert_eq!(values(policy), values(detached_policy));
        assert!(
            values(value)
                .into_iter()
                .all(|value| value.is_finite() && (-1.0..=1.0).contains(&value))
        );
        assert!(
            values(detached_value)
                .into_iter()
                .all(|value| value.is_finite() && (-1.0..=1.0).contains(&value))
        );
    }

    #[test]
    fn saved_othello_network_round_trips() {
        let path =
            std::env::temp_dir().join(format!("etive-model-{}.burnpack", std::process::id()));
        let device = Device::flex();
        let input = Tensor::zeros([1, 2, 8, 8], &device);
        let network = OthelloNetwork::new(&device, 7);
        let expected = network.forward(input.clone());
        network.save(&path).unwrap();

        let loaded = OthelloNetwork::load(&path, &device).unwrap();
        let actual = loaded.forward(input);
        std::fs::remove_file(path).unwrap();

        assert_eq!(values(expected.0), values(actual.0));
        assert_eq!(values(expected.1), values(actual.1));
    }

    #[test]
    fn othello_network_has_expected_parameter_count() {
        let network = OthelloNetwork::new(&Device::flex(), 7);
        assert_eq!(network.num_params(), 2_976_709);
    }

    #[test]
    fn weekend_network_is_small() {
        let network =
            OthelloNetwork::new_with_config(&Device::flex(), 7, OthelloModelConfig::WEEKEND);
        assert!(network.num_params() < 500_000);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn weekend_cuda_forward_is_finite_at_every_operation() {
        let device = Device::cuda(0);
        let network = OthelloNetwork::new_with_config(&device, 7, OthelloModelConfig::WEEKEND)
            .cast_float(FloatDType::F16);
        let input = Tensor::<4>::zeros([1024, 2, 8, 8], &device).cast(FloatDType::F16);

        let mut hidden = network.stem.forward(input);
        assert_finite("stem.conv", &hidden);
        hidden = network.stem_norm.forward(hidden);
        assert_finite("stem.norm", &hidden);
        hidden = activation::relu(hidden);
        assert_finite("stem.relu", &hidden);

        for (index, block) in network.blocks.iter().enumerate() {
            let residual = hidden.clone();
            let mut next = block.conv1.forward(hidden);
            assert_finite(&format!("block.{index}.conv1"), &next);
            next = block.norm1.forward(next);
            assert_finite(&format!("block.{index}.norm1"), &next);
            next = activation::relu(next);
            assert_finite(&format!("block.{index}.relu1"), &next);
            next = block.conv2.forward(next);
            assert_finite(&format!("block.{index}.conv2"), &next);
            next = block.norm2.forward(next);
            assert_finite(&format!("block.{index}.norm2"), &next);
            next = next + residual;
            assert_finite(&format!("block.{index}.add"), &next);
            hidden = activation::relu(next);
            assert_finite(&format!("block.{index}.relu2"), &hidden);
        }

        let mut policy = network.policy_conv.forward(hidden.clone());
        assert_finite("policy.conv", &policy);
        policy = activation::relu(policy);
        assert_finite("policy.relu", &policy);
        let policy = network.policy.forward(policy.reshape([1024, 2 * 8 * 8]));
        assert_finite("policy.linear", &policy);

        let mut value = network.value_conv.forward(hidden);
        assert_finite("value.conv", &value);
        value = activation::relu(value);
        assert_finite("value.relu1", &value);
        let mut value = network.value_hidden.forward(value.reshape([1024, 8 * 8]));
        assert_finite("value.hidden_linear", &value);
        value = activation::relu(value);
        assert_finite("value.relu2", &value);
        value = network.value.forward(value);
        assert_finite("value.linear", &value);
        value = value.tanh();
        assert_finite("value.tanh", &value);
    }
}
