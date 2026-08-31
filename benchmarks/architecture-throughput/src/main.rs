use std::hint::black_box;
use std::time::Instant;

use burn::module::{Module, ModuleMapper, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::norm::Normalization;
use burn::nn::{BatchNormConfig, GroupNormConfig, Identity, Linear, LinearConfig, PaddingConfig2d};
use burn::tensor::{Device, FloatDType, Tensor, TensorData, activation};

const CHANNELS: usize = 64;
const RESIDUAL_BLOCKS: usize = 4;
const NORM_GROUPS: usize = 8;
const ACTIONS: usize = 65;

struct CastFloatMapper(FloatDType);

impl ModuleMapper for CastFloatMapper {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let (id, tensor, mapper) = param.consume();
        Param::from_mapped_value(id, tensor.cast(self.0), mapper)
    }
}

#[derive(Module, Debug)]
struct ResidualBlock {
    conv1: Conv2d,
    norm1: Normalization,
    conv2: Conv2d,
    norm2: Normalization,
}

impl ResidualBlock {
    fn new(device: &Device, norm: &str) -> Self {
        let convolution = || {
            Conv2dConfig::new([CHANNELS, CHANNELS], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device)
        };
        Self {
            conv1: convolution(),
            norm1: normalization(device, norm),
            conv2: convolution(),
            norm2: normalization(device, norm),
        }
    }

    fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let residual = input.clone();
        let hidden = activation::relu(self.norm1.forward(self.conv1.forward(input)));
        activation::relu(self.norm2.forward(self.conv2.forward(hidden)) + residual)
    }

    fn channels_last_weights(mut self) -> Self {
        self.conv1 = channels_last_weight(self.conv1);
        self.conv2 = channels_last_weight(self.conv2);
        self
    }
}

#[derive(Module, Debug)]
struct Network {
    stem: Conv2d,
    stem_norm: Normalization,
    blocks: Vec<ResidualBlock>,
    policy_conv: Conv2d,
    policy: Linear,
    value_conv: Conv2d,
    value_hidden: Linear,
    value: Linear,
}

impl Network {
    fn new(device: &Device, norm: &str) -> Self {
        device.seed(7);
        Self {
            stem: Conv2dConfig::new([2, CHANNELS], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            stem_norm: normalization(device, norm),
            blocks: (0..RESIDUAL_BLOCKS)
                .map(|_| ResidualBlock::new(device, norm))
                .collect(),
            policy_conv: Conv2dConfig::new([CHANNELS, 2], [1, 1]).init(device),
            policy: LinearConfig::new(2 * 8 * 8, ACTIONS).init(device),
            value_conv: Conv2dConfig::new([CHANNELS, 1], [1, 1]).init(device),
            value_hidden: LinearConfig::new(8 * 8, CHANNELS).init(device),
            value: LinearConfig::new(CHANNELS, 1).init(device),
        }
    }

    fn cast_float(self, dtype: FloatDType) -> Self {
        self.map(&mut CastFloatMapper(dtype))
    }

    fn channels_last_weights(mut self) -> Self {
        self.stem = channels_last_weight(self.stem);
        self.blocks = self
            .blocks
            .into_iter()
            .map(ResidualBlock::channels_last_weights)
            .collect();
        self.policy_conv = channels_last_weight(self.policy_conv);
        self.value_conv = channels_last_weight(self.value_conv);
        self
    }

    fn forward(&self, input: Tensor<4>) -> (Tensor<2>, Tensor<2>) {
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
}

fn channels_last_weight(mut convolution: Conv2d) -> Conv2d {
    let (id, weight, mapper) = convolution.weight.consume();
    let [out_channels, in_channels, kernel_height, kernel_width] = weight.dims();
    let weight = weight
        .permute([0, 2, 3, 1])
        .reshape([out_channels * kernel_height * kernel_width * in_channels])
        .reshape([out_channels, kernel_height, kernel_width, in_channels])
        .permute([0, 3, 1, 2]);
    convolution.weight = Param::from_mapped_value(id, weight, mapper);
    convolution
}

fn normalization(device: &Device, norm: &str) -> Normalization {
    match norm {
        "group" => GroupNormConfig::new(NORM_GROUPS, CHANNELS)
            .init(device)
            .into(),
        "batch" => BatchNormConfig::new(CHANNELS).init(device).into(),
        "none" => Identity::new().into(),
        _ => panic!("--norm must be group, batch, or none"),
    }
}

fn validate_group_norm(device: &Device, channels_last: bool) {
    let batch = 4;
    let elements = batch * CHANNELS * 8 * 8;
    let values = (0..elements)
        .map(|index| ((index as f32 * 0.013).sin() * 3.0) + (index % 17) as f32 * 0.01)
        .collect::<Vec<_>>();
    let input = Tensor::<4>::from_data(TensorData::new(values, [batch, CHANNELS, 8, 8]), device)
        .cast(FloatDType::F16);
    let input = if channels_last {
        input
            .permute([0, 2, 3, 1])
            .reshape([elements])
            .reshape([batch, 8, 8, CHANNELS])
            .permute([0, 3, 1, 2])
    } else {
        input
    };
    let quantized_input = input
        .clone()
        .cast(FloatDType::F32)
        .into_data()
        .try_to_vec::<f32>()
        .expect("failed to read validation input");
    let output = normalization(device, "group")
        .map(&mut CastFloatMapper(FloatDType::F16))
        .forward(input)
        .cast(FloatDType::F32)
        .into_data()
        .try_to_vec::<f32>()
        .expect("failed to read GroupNorm output");

    let channels_per_group = CHANNELS / NORM_GROUPS;
    let group_size = channels_per_group * 8 * 8;
    let mut max_absolute_error = 0.0_f32;
    for batch_index in 0..batch {
        for group in 0..NORM_GROUPS {
            let start = (batch_index * CHANNELS + group * channels_per_group) * 8 * 8;
            let values = &quantized_input[start..start + group_size];
            let mean = values.iter().map(|&value| value as f64).sum::<f64>() / group_size as f64;
            let variance = values
                .iter()
                .map(|&value| {
                    let deviation = value as f64 - mean;
                    deviation * deviation
                })
                .sum::<f64>()
                / group_size as f64;
            let denominator = (variance + 1e-5).sqrt();
            for (offset, &value) in values.iter().enumerate() {
                let expected = ((value as f64 - mean) / denominator) as f32;
                max_absolute_error =
                    max_absolute_error.max((output[start + offset] - expected).abs());
            }
        }
    }

    println!(
        "group_norm_validation layout={} max_absolute_error={max_absolute_error:.8}",
        if channels_last {
            "channels-last"
        } else {
            "nchw"
        }
    );
    assert!(max_absolute_error <= 0.005);
}

fn argument(name: &str, default: usize) -> usize {
    let mut args = std::env::args();
    while let Some(argument) = args.next() {
        if argument == name {
            return args
                .next()
                .unwrap_or_else(|| panic!("missing value for {name}"))
                .parse()
                .unwrap_or_else(|_| panic!("invalid value for {name}"));
        }
    }
    default
}

fn string_argument(name: &str, default: &str) -> String {
    let mut args = std::env::args();
    while let Some(argument) = args.next() {
        if argument == name {
            return args
                .next()
                .unwrap_or_else(|| panic!("missing value for {name}"));
        }
    }
    default.to_string()
}

fn run_forwards(network: &Network, input: &Tensor<4>, device: &Device, iterations: usize) {
    for _ in 0..iterations {
        let output = network.forward(input.clone());
        black_box(&output);
        device.flush();
    }
    device.sync().expect("CUDA synchronization failed");
}

fn main() {
    let batch = argument("--batch", 1024);
    let warmup = argument("--warmup", 50);
    let iterations = argument("--iterations", 500);
    let norm = string_argument("--norm", "group");
    let weight_layout = string_argument("--weight-layout", "contiguous");
    assert!(batch > 0 && iterations > 0);

    let device = Device::cuda(0);
    if std::env::args().any(|argument| argument == "--validate-group-norm") {
        validate_group_norm(&device, false);
        validate_group_norm(&device, true);
        return;
    }
    let network = Network::new(&device, &norm).cast_float(FloatDType::F16);
    let network = match weight_layout.as_str() {
        "contiguous" => network,
        "channels-last" => network.channels_last_weights(),
        _ => panic!("--weight-layout must be contiguous or channels-last"),
    };
    let input = Tensor::<4>::zeros([batch, 2, 8, 8], &device).cast(FloatDType::F16);

    println!(
        "backend=burn dtype=float16 layout=nchw weight_layout={weight_layout} norm={norm} batch={batch} channels={CHANNELS} blocks={RESIDUAL_BLOCKS} groups={NORM_GROUPS}"
    );
    run_forwards(&network, &input, &device, warmup);

    device.sync().expect("CUDA synchronization failed");
    let started = Instant::now();
    run_forwards(&network, &input, &device, iterations);
    let elapsed = started.elapsed();
    let seconds = elapsed.as_secs_f64();
    let batches_per_second = iterations as f64 / seconds;
    let positions_per_second = batches_per_second * batch as f64;

    println!("iterations={iterations} elapsed_seconds={seconds:.6}");
    println!("milliseconds_per_batch={:.3}", 1000.0 / batches_per_second);
    println!("positions_per_second={positions_per_second:.0}");
}
