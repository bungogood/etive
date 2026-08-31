//! Othello policy/value loss calculation, optimization, and symmetry augmentation.

use std::path::Path;
use std::time::{Duration, Instant};

use burn::module::AutodiffModule;
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::{AdamWConfig, GradientsParams, ModuleOptimizer};
use burn::store::RecordError;
use burn::tensor::{Device, Tensor, TensorData, activation};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tracing::info;

use super::replay::SelfPlaySample;
use super::{Board, OthelloEncoding, OthelloNetwork};
use crate::game::Game;
use crate::metrics::PolicyValueMetrics;

#[derive(Debug, thiserror::Error)]
pub enum TrainingError {
    #[error("{0}")]
    InvalidInput(&'static str),
    #[error(transparent)]
    Record(#[from] RecordError),
}

#[derive(Clone, Copy, Debug)]
pub struct TrainingReport {
    pub metrics: PolicyValueMetrics<f32>,
    pub elapsed: Duration,
}

pub struct TrainingSession {
    optimizer: ModuleOptimizer,
    device: Device,
    learning_rate: f64,
    random: StdRng,
    batch_size: usize,
    inputs: Vec<f32>,
    policies: Vec<f32>,
    outcomes: Vec<f32>,
}

pub fn evaluate_loss(
    network: &OthelloNetwork,
    device: &Device,
    samples: &[SelfPlaySample],
    batch_size: usize,
) -> Result<PolicyValueMetrics<f32>, TrainingError> {
    if samples.is_empty() || batch_size == 0 {
        return Err(TrainingError::InvalidInput(
            "validation data and batch size must be non-empty",
        ));
    }
    let network = network.valid();
    let mut policy_total = 0.0;
    let mut policy_entropy_total = 0.0;
    let mut value_total = 0.0;
    for batch in samples.chunks(batch_size) {
        let mut inputs = vec![0.0; batch.len() * OthelloEncoding::LEN];
        let mut policies = vec![0.0; batch.len() * Board::ACTION_COUNT];
        let mut outcomes = vec![0.0; batch.len()];
        for (index, sample) in batch.iter().enumerate() {
            let input_start = index * OthelloEncoding::LEN;
            let policy_start = index * Board::ACTION_COUNT;
            outcomes[index] = write_sample(
                sample,
                &mut inputs[input_start..input_start + OthelloEncoding::LEN],
                &mut policies[policy_start..policy_start + Board::ACTION_COUNT],
            );
        }
        let input = tensor(&inputs, [batch.len(), 2, 8, 8], device);
        let target_policy = tensor(&policies, [batch.len(), Board::ACTION_COUNT], device);
        let target_value = tensor(&outcomes, [batch.len(), 1], device);
        let (policy_logits, values) = network.forward(input);
        let policy_loss = policy_loss(policy_logits, target_policy, batch.len());
        let value_loss = MseLoss::new().forward(values, target_value, Reduction::Mean);
        policy_total += policy_loss.into_scalar::<f32>() * batch.len() as f32;
        policy_entropy_total += target_entropy(&policies, batch.len()) * batch.len() as f32;
        value_total += value_loss.into_scalar::<f32>() * batch.len() as f32;
    }
    Ok(PolicyValueMetrics::new(
        policy_total / samples.len() as f32,
        policy_entropy_total / samples.len() as f32,
        value_total / samples.len() as f32,
    ))
}

impl TrainingSession {
    pub fn new(
        device: Device,
        batch_size: usize,
        learning_rate: f64,
        weight_decay: f32,
        seed: u64,
    ) -> Result<Self, TrainingError> {
        if batch_size == 0
            || !learning_rate.is_finite()
            || learning_rate <= 0.0
            || !weight_decay.is_finite()
            || weight_decay < 0.0
        {
            return Err(TrainingError::InvalidInput(
                "training batch size and learning rate must be positive",
            ));
        }
        let optimizer = AdamWConfig::new()
            .with_epsilon(1e-8)
            .with_weight_decay(weight_decay)
            .init();
        Ok(Self {
            optimizer,
            device,
            learning_rate,
            random: StdRng::seed_from_u64(seed),
            batch_size,
            inputs: vec![0.0; batch_size * OthelloEncoding::LEN],
            policies: vec![0.0; batch_size * Board::ACTION_COUNT],
            outcomes: vec![0.0; batch_size],
        })
    }

    pub fn train_steps(
        &mut self,
        network: &mut OthelloNetwork,
        replay: &[&[SelfPlaySample]],
        steps: usize,
    ) -> Result<TrainingReport, TrainingError> {
        let sample_count = replay.iter().map(|samples| samples.len()).sum::<usize>();
        if sample_count == 0 || steps == 0 {
            return Err(TrainingError::InvalidInput(
                "replay data and training steps must be non-empty",
            ));
        }
        let start = Instant::now();
        let mut last_progress = start;
        let mut last_step = 0;
        let mut policy_cross_entropy_total = 0.0;
        let mut policy_entropy_total = 0.0;
        let mut value_mse_total = 0.0;
        for step in 1..=steps {
            let metrics = self.step(network, replay);
            policy_cross_entropy_total += metrics.policy_cross_entropy;
            policy_entropy_total += metrics.policy_target_entropy;
            value_mse_total += metrics.value_mse;
            let interval = last_progress.elapsed();
            if interval >= Duration::from_secs(5) || step == steps {
                info!(
                    step,
                    total_steps = steps,
                    steps_per_second = %format_args!(
                        "{:.1}",
                        (step - last_step) as f64 / interval.as_secs_f64()
                    ),
                    policy_cross_entropy = %format_args!("{:.4}", metrics.policy_cross_entropy),
                    policy_kl = %format_args!("{:.4}", metrics.policy_kl()),
                    value_mse = %format_args!("{:.4}", metrics.value_mse),
                    elapsed = %format_args!("{:.1}s", start.elapsed().as_secs_f64()),
                    "training progress"
                );
                last_progress = Instant::now();
                last_step = step;
            }
        }
        Ok(TrainingReport {
            metrics: PolicyValueMetrics::new(
                policy_cross_entropy_total / steps as f32,
                policy_entropy_total / steps as f32,
                value_mse_total / steps as f32,
            ),
            elapsed: start.elapsed(),
        })
    }

    pub fn set_learning_rate(&mut self, learning_rate: f64) {
        self.learning_rate = learning_rate;
    }

    pub fn reseed(&mut self, seed: u64) {
        self.random = StdRng::seed_from_u64(seed);
    }

    pub fn save_optimizer(&self, path: impl AsRef<Path>) -> Result<(), RecordError> {
        self.optimizer.save(path)
    }

    pub fn load_optimizer(&mut self, path: impl AsRef<Path>) -> Result<(), RecordError> {
        self.optimizer = self.optimizer.clone().load(path)?;
        Ok(())
    }

    fn step(
        &mut self,
        network: &mut OthelloNetwork,
        replay: &[&[SelfPlaySample]],
    ) -> PolicyValueMetrics<f32> {
        let sample_count = replay.iter().map(|samples| samples.len()).sum::<usize>();
        for (batch_index, outcome) in self.outcomes.iter_mut().enumerate() {
            let mut sample_index = self.random.random_range(0..sample_count);
            let mut sample = None;
            for samples in replay {
                if sample_index < samples.len() {
                    sample = Some(&samples[sample_index]);
                    break;
                }
                sample_index -= samples.len();
            }
            let sample = sample.expect("sample index must fall within replay data");
            let input_start = batch_index * OthelloEncoding::LEN;
            let policy_start = batch_index * Board::ACTION_COUNT;
            *outcome = write_sample(
                sample,
                &mut self.inputs[input_start..input_start + OthelloEncoding::LEN],
                &mut self.policies[policy_start..policy_start + Board::ACTION_COUNT],
            );
            apply_symmetry(
                &mut self.inputs[input_start..input_start + OthelloEncoding::LEN],
                &mut self.policies[policy_start..policy_start + Board::ACTION_COUNT],
                self.random.random_range(0..8),
            );
        }

        let input = tensor(&self.inputs, [self.batch_size, 2, 8, 8], &self.device);
        let target_policy = tensor(
            &self.policies,
            [self.batch_size, Board::ACTION_COUNT],
            &self.device,
        );
        let target_value = tensor(&self.outcomes, [self.batch_size, 1], &self.device);
        let (policy_cross_entropy, value_mse) =
            self.train_tensor_step(network, input, target_policy, target_value);
        PolicyValueMetrics::new(
            policy_cross_entropy,
            target_entropy(&self.policies, self.batch_size),
            value_mse,
        )
    }

    pub fn train_tensor_step(
        &mut self,
        network: &mut OthelloNetwork,
        input: Tensor<4>,
        target_policy: Tensor<2>,
        target_value: Tensor<2>,
    ) -> (f32, f32) {
        let (policy_logits, values) = network.forward(input);
        let policy_loss = policy_loss(policy_logits, target_policy, self.batch_size);
        let value_loss = MseLoss::new().forward(values, target_value, Reduction::Mean);
        let loss_values = Tensor::cat(vec![policy_loss.clone(), value_loss.clone()], 0);
        let gradients = (policy_loss + value_loss).backward();
        let gradients = GradientsParams::from_grads(gradients, network);
        *network = self
            .optimizer
            .step(self.learning_rate, network.clone(), gradients);
        let losses = loss_values.into_data().try_to_vec::<f32>().unwrap();
        (losses[0], losses[1])
    }
}

fn write_sample(sample: &SelfPlaySample, input: &mut [f32], policy: &mut [f32]) -> f32 {
    OthelloEncoding::encode(&sample.position, input);
    policy.copy_from_slice(&sample.policy);
    sample.outcome.value()
}

fn tensor<const D: usize>(data: &[f32], shape: [usize; D], device: &Device) -> Tensor<D> {
    Tensor::<1>::from_data(TensorData::from(data), device).reshape(shape)
}

fn policy_loss(logits: Tensor<2>, target: Tensor<2>, batch_size: usize) -> Tensor<1> {
    (target * activation::log_softmax(logits, 1))
        .sum()
        .mul_scalar(-1.0 / batch_size as f32)
}

fn target_entropy(policies: &[f32], batch_size: usize) -> f32 {
    -policies
        .iter()
        .filter(|&&probability| probability > 0.0)
        .map(|&probability| probability * probability.ln())
        .sum::<f32>()
        / batch_size as f32
}

fn apply_symmetry(input: &mut [f32], policy: &mut [f32], symmetry: usize) {
    debug_assert_eq!(input.len(), 128);
    debug_assert_eq!(policy.len(), Board::ACTION_COUNT);
    let original_input: [f32; 128] = input.try_into().expect("fixed Othello input size");
    let original_policy: [f32; Board::ACTION_COUNT] =
        policy.try_into().expect("fixed Othello policy size");
    for plane in 0..2 {
        for source in 0..64 {
            input[plane * 64 + transform_square(source, symmetry)] =
                original_input[plane * 64 + source];
        }
    }
    for source in 0..64 {
        policy[transform_square(source, symmetry)] = original_policy[source];
    }
    policy[64] = original_policy[64];
}

fn transform_square(index: usize, symmetry: usize) -> usize {
    let mut row = index / 8;
    let mut column = index % 8;
    for _ in 0..symmetry % 4 {
        (row, column) = (column, 7 - row);
    }
    if symmetry >= 4 {
        column = 7 - column;
    }
    row * 8 + column
}

#[cfg(test)]
mod tests {
    use crate::game::Outcome;
    use crate::othello::OthelloModelConfig;

    use super::*;

    #[test]
    fn optimizer_consumes_soft_policy_and_value_targets() {
        let inference_device = Device::flex();
        let device = inference_device.clone().autodiff();
        let mut network = OthelloNetwork::new(&device, 7);
        let before = network
            .valid()
            .forward(Tensor::zeros([1, 2, 8, 8], &inference_device))
            .0
            .into_data();
        let mut policy = [0.0; Board::ACTION_COUNT];
        for action in Board::default().legal_actions() {
            policy[Board::action_index(action)] = 0.25;
        }
        let sample = SelfPlaySample {
            position: Board::default(),
            policy,
            outcome: Outcome::Draw,
            game: 1,
        };
        let samples = [sample];

        let mut session = TrainingSession::new(device.clone(), 2, 0.001, 0.0001, 11).unwrap();
        let report = session.train_steps(&mut network, &[&samples], 1).unwrap();

        assert!(report.metrics.policy_cross_entropy.is_finite());
        assert!(report.metrics.value_mse.is_finite());
        let after = network
            .valid()
            .forward(Tensor::zeros([1, 2, 8, 8], &inference_device))
            .0
            .into_data();
        assert_ne!(before.as_bytes(), after.as_bytes());

        let directory = tempfile::tempdir().unwrap();
        let model_path = directory.path().join("model.burnpack");
        let optimizer_path = directory.path().join("optimizer.burnpack");
        network.save(&model_path).unwrap();
        session.save_optimizer(&optimizer_path).unwrap();
        let mut network = OthelloNetwork::load(&model_path, &device).unwrap();
        let mut session = TrainingSession::new(device, 2, 0.001, 0.0001, 11).unwrap();
        session.load_optimizer(&optimizer_path).unwrap();
        let resumed = session.train_steps(&mut network, &[&samples], 1).unwrap();
        assert!(resumed.metrics.policy_cross_entropy.is_finite());
        assert!(resumed.metrics.value_mse.is_finite());
    }

    #[ignore = "explicit preflight gate for production training"]
    #[test]
    fn fixed_batch_overfit_gate() {
        #[cfg(feature = "cuda")]
        let inference_device = Device::cuda(0);
        #[cfg(not(feature = "cuda"))]
        let inference_device = Device::flex();
        let device = inference_device.clone().autodiff();
        let mut network = OthelloNetwork::new_with_config(&device, 7, OthelloModelConfig::WEEKEND);
        let mut policy = [0.0; Board::ACTION_COUNT];
        for action in Board::default().legal_actions() {
            policy[Board::action_index(action)] = 0.25;
        }
        let samples = [SelfPlaySample {
            position: Board::default(),
            policy,
            outcome: Outcome::Win,
            game: 1,
        }];
        let initial = evaluate_loss(&network, &inference_device, &samples, 1).unwrap();
        let mut session = TrainingSession::new(device, 16, 0.001, 0.0001, 11).unwrap();

        session.train_steps(&mut network, &[&samples], 500).unwrap();
        let final_loss = evaluate_loss(&network, &inference_device, &samples, 1).unwrap();

        assert!(
            final_loss.policy_kl() < 0.1,
            "policy did not approach target distribution: {initial:?} -> {final_loss:?}"
        );
        assert!(
            final_loss.value_mse < 0.01,
            "value did not overfit: {initial:?} -> {final_loss:?}"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_forward_backward_and_optimizer_step() {
        let inference_device = Device::cuda(0);
        let device = inference_device.clone().autodiff();
        let mut network = OthelloNetwork::new(&device, 7);
        let mut policy = [0.0; Board::ACTION_COUNT];
        for action in Board::default().legal_actions() {
            policy[Board::action_index(action)] = 0.25;
        }
        let sample = SelfPlaySample {
            position: Board::default(),
            policy,
            outcome: Outcome::Draw,
            game: 1,
        };
        let mut session = TrainingSession::new(device, 2, 0.001, 0.0001, 11).unwrap();

        let report = session.train_steps(&mut network, &[&[sample]], 1).unwrap();
        let (policy, value) = network
            .valid()
            .forward(Tensor::zeros([1, 2, 8, 8], &inference_device));

        assert!(report.metrics.policy_cross_entropy.is_finite());
        assert!(report.metrics.value_mse.is_finite());
        assert!(
            policy
                .into_data()
                .try_to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|value| value.is_finite())
        );
        assert!(value.into_scalar::<f32>().is_finite());
    }

    #[test]
    fn symmetry_transforms_state_and_policy_together() {
        let mut input = [0.0; 128];
        let mut policy = [0.0; Board::ACTION_COUNT];
        input[8] = 1.0;
        input[64 + 8] = 2.0;
        policy[8] = 0.75;
        policy[64] = 0.25;

        apply_symmetry(&mut input, &mut policy, 1);

        assert_eq!(input[6], 1.0);
        assert_eq!(input[64 + 6], 2.0);
        assert_eq!(policy[6], 0.75);
        assert_eq!(policy[64], 0.25);
        assert_eq!(policy.iter().sum::<f32>(), 1.0);
    }
}
