use std::error::Error;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use burn::tensor::{Device, FloatDType};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::index;

use super::Board;
use super::actors::SelfPlaySample;
use super::experiment::replay::read_replay;
use crate::evaluator::{BatchEvaluator, OthelloBurnEvaluator};
use crate::game::Game;
use crate::model::{OthelloModelConfig, OthelloNetwork};

use super::training::TrainingSession;

const ACTION_COUNT: usize = Board::ACTION_COUNT;

/// Configuration for a fixed number of training steps over immutable replay shards.
#[derive(Clone, Debug)]
pub struct FrozenTrainingConfig {
    pub steps: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub weight_decay: f32,
    pub seed: u64,
    pub output: PathBuf,
}

/// Metrics emitted by a frozen-replay training run.
#[derive(Clone, Copy, Debug)]
pub struct FrozenTrainingReport {
    pub steps: usize,
    pub replay_rows: usize,
    pub batch_size: usize,
    pub seed: u64,
    pub learning_rate: f64,
    pub weight_decay: f32,
    pub training_seconds: f64,
    pub policy_cross_entropy: f32,
    pub policy_target_entropy: f32,
    pub policy_kl: f32,
    pub value_mse: f32,
}

impl FrozenTrainingReport {
    pub const CSV_HEADER: &'static str = "steps,replay_rows,batch_size,seed,learning_rate,weight_decay,training_seconds,policy_cross_entropy,policy_target_entropy,policy_kl,value_mse";

    fn csv(&self) -> String {
        format!("{}\n{}\n", Self::CSV_HEADER, self)
    }
}

impl fmt::Display for FrozenTrainingReport {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            output,
            "{},{},{},{},{},{},{},{},{},{},{}",
            self.steps,
            self.replay_rows,
            self.batch_size,
            self.seed,
            self.learning_rate,
            self.weight_decay,
            self.training_seconds,
            self.policy_cross_entropy,
            self.policy_target_entropy,
            self.policy_kl,
            self.value_mse,
        )
    }
}

/// Restores a model and optimizer, then trains over fixed replay data for an exact step count.
pub fn train_frozen(
    checkpoint: impl AsRef<Path>,
    optimizer: impl AsRef<Path>,
    replay_paths: &[PathBuf],
    config: FrozenTrainingConfig,
    device: Device,
) -> Result<FrozenTrainingReport, Box<dyn Error>> {
    validate_frozen_training(replay_paths, &config)?;

    let mut replay = Vec::with_capacity(replay_paths.len());
    for path in replay_paths {
        replay.push(read_replay(path)?);
    }
    let replay_rows = replay.iter().map(Vec::len).sum::<usize>();
    if replay_rows == 0 {
        return Err(invalid_input("replay data contains no rows"));
    }

    let staging = prepare_frozen_output(&config.output)?;
    let result = (|| {
        let model_config = resolve_checkpoint_config(checkpoint.as_ref());
        let training_device = device.autodiff();
        let mut network =
            OthelloNetwork::load_with_config(checkpoint.as_ref(), &training_device, model_config)?;
        let mut trainer = TrainingSession::new(
            training_device,
            config.batch_size,
            config.learning_rate,
            config.weight_decay,
            config.seed,
        )?;
        trainer.load_optimizer(optimizer)?;
        let replay_slices = replay.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let training = trainer.train_steps(&mut network, &replay_slices, config.steps)?;
        let report = FrozenTrainingReport {
            steps: config.steps,
            replay_rows,
            batch_size: config.batch_size,
            seed: config.seed,
            learning_rate: config.learning_rate,
            weight_decay: config.weight_decay,
            training_seconds: training.elapsed.as_secs_f64(),
            policy_cross_entropy: training.policy_loss,
            policy_target_entropy: training.policy_target_entropy,
            policy_kl: training.policy_kl(),
            value_mse: training.value_loss,
        };

        network.save(staging.join("model.burnpack"))?;
        trainer.save_optimizer(staging.join("optimizer.burnpack"))?;
        fs::write(staging.join("model.toml"), toml::to_string(&model_config)?)?;
        fs::write(staging.join("metrics.csv"), report.csv())?;
        if config.output.exists() {
            return Err(invalid_input("output already exists"));
        }
        fs::rename(&staging, &config.output)?;
        Ok(report)
    })();
    if result.is_err() && staging.exists() {
        fs::remove_dir_all(staging)?;
    }
    result
}

fn validate_frozen_training(
    replay_paths: &[PathBuf],
    config: &FrozenTrainingConfig,
) -> Result<(), Box<dyn Error>> {
    if replay_paths.is_empty() {
        return Err(invalid_input("at least one replay path is required"));
    }
    if config.steps == 0 {
        return Err(invalid_input("steps must be positive"));
    }
    if config.batch_size == 0 {
        return Err(invalid_input("batch size must be positive"));
    }
    if !config.learning_rate.is_finite() || config.learning_rate <= 0.0 {
        return Err(invalid_input("learning rate must be finite and positive"));
    }
    if !config.weight_decay.is_finite() || config.weight_decay < 0.0 {
        return Err(invalid_input("weight decay must be finite and nonnegative"));
    }
    Ok(())
}

fn resolve_checkpoint_config(checkpoint: &Path) -> OthelloModelConfig {
    OthelloNetwork::checkpoint_config(checkpoint).unwrap_or(OthelloModelConfig::LEGACY)
}

fn prepare_frozen_output(output: &Path) -> Result<PathBuf, Box<dyn Error>> {
    if output.exists() {
        return Err(invalid_input("output already exists"));
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    for attempt in 0..1000 {
        let staging = suffixed_path(
            output,
            &format!(".train-frozen-{}-{attempt}.tmp", std::process::id()),
        );
        match fs::create_dir(&staging) {
            Ok(()) => return Ok(staging),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "unable to create a unique output staging directory",
    )
    .into())
}

fn suffixed_path(path: &Path, suffix: &str) -> PathBuf {
    let mut value = path.as_os_str().to_owned();
    value.push(suffix);
    value.into()
}

/// Aggregate checkpoint diagnostics over replay positions.
#[derive(Clone, Copy, Debug)]
pub struct DiagnosticsReport {
    pub available_rows: usize,
    pub sampled_rows: usize,
    pub seed: u64,
    pub policy_cross_entropy: f64,
    pub policy_target_entropy: f64,
    pub policy_kl: f64,
    pub policy_predicted_entropy: f64,
    pub policy_legal_mass: f64,
    pub policy_top1_target_agreement: f64,
    pub value_mse: f64,
    pub value_mae: f64,
    pub value_sign_accuracy: f64,
    pub value_correlation: f64,
    pub value_predicted_mean: f64,
    pub value_predicted_std: f64,
}

impl DiagnosticsReport {
    pub const CSV_HEADER: &'static str = "available_rows,sampled_rows,seed,policy_cross_entropy,policy_target_entropy,policy_kl,policy_predicted_entropy,policy_legal_mass,policy_top1_target_agreement,value_mse,value_mae,value_sign_accuracy,value_correlation,value_predicted_mean,value_predicted_std";
}

impl fmt::Display for DiagnosticsReport {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            output,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            self.available_rows,
            self.sampled_rows,
            self.seed,
            self.policy_cross_entropy,
            self.policy_target_entropy,
            self.policy_kl,
            self.policy_predicted_entropy,
            self.policy_legal_mass,
            self.policy_top1_target_agreement,
            self.value_mse,
            self.value_mae,
            self.value_sign_accuracy,
            self.value_correlation,
            self.value_predicted_mean,
            self.value_predicted_std,
        )
    }
}

/// Evaluates a checkpoint against a deterministic sample of validated replay rows.
pub fn diagnose_replay(
    checkpoint: impl AsRef<Path>,
    replay_paths: &[PathBuf],
    rows: Option<usize>,
    seed: u64,
    batch_size: usize,
    float32: bool,
    device: Device,
) -> Result<DiagnosticsReport, Box<dyn Error>> {
    if rows == Some(0) {
        return Err(invalid_input("rows must be positive"));
    }
    if batch_size == 0 {
        return Err(invalid_input("batch size must be positive"));
    }
    if replay_paths.is_empty() {
        return Err(invalid_input("at least one replay path is required"));
    }

    let mut samples = Vec::new();
    for path in replay_paths {
        samples.extend(read_replay(path)?);
    }
    let available_rows = samples.len();
    if available_rows == 0 {
        return Err(invalid_input("replay data contains no rows"));
    }

    let selected = sample_indices(available_rows, rows.unwrap_or(available_rows), seed);
    let network = OthelloNetwork::load(checkpoint, &device)?;
    let dtype = if float32 {
        FloatDType::F32
    } else {
        FloatDType::F16
    };
    let mut evaluator = OthelloBurnEvaluator::from_network_with_dtype(device, &network, dtype);
    let mut metrics = Metrics::default();

    for indices in selected.chunks(batch_size) {
        let positions = indices
            .iter()
            .map(|&sample| samples[sample].position)
            .collect::<Vec<_>>();
        let mut policy_logits = vec![0.0; positions.len() * ACTION_COUNT];
        let mut values = vec![0.0; positions.len()];
        evaluator.evaluate_batch(&positions, &mut policy_logits, &mut values)?;
        let (policy_rows, remainder) = policy_logits.as_chunks::<ACTION_COUNT>();
        debug_assert!(remainder.is_empty());
        for ((&sample, logits), &value) in indices.iter().zip(policy_rows).zip(&values) {
            metrics.add(&samples[sample], logits, value)?;
        }
    }

    Ok(metrics.report(available_rows, selected.len(), seed))
}

fn sample_indices(available: usize, requested: usize, seed: u64) -> Vec<usize> {
    let count = requested.min(available);
    if count == available {
        return (0..available).collect();
    }
    let mut random = StdRng::seed_from_u64(seed);
    let mut selected = index::sample(&mut random, available, count).into_vec();
    selected.sort_unstable();
    selected
}

#[derive(Default)]
struct Metrics {
    policy_cross_entropy: f64,
    policy_target_entropy: f64,
    policy_predicted_entropy: f64,
    policy_legal_mass: f64,
    policy_top1_target_agreement: f64,
    value_squared_error: f64,
    value_absolute_error: f64,
    value_sign_agreement: f64,
    value_sum: f64,
    value_squared_sum: f64,
    target_sum: f64,
    target_squared_sum: f64,
    value_target_product_sum: f64,
    rows: usize,
}

impl Metrics {
    fn add(
        &mut self,
        sample: &SelfPlaySample,
        logits: &[f32],
        predicted_value: f32,
    ) -> Result<(), Box<dyn Error>> {
        if logits.len() != ACTION_COUNT
            || logits.iter().any(|value| !value.is_finite())
            || !predicted_value.is_finite()
        {
            return Err(invalid_input(
                "model produced non-finite or malformed output",
            ));
        }

        let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let log_sum_exp = maximum
            + logits
                .iter()
                .map(|&logit| (f64::from(logit) - maximum).exp())
                .sum::<f64>()
                .ln();
        let target_maximum = sample.policy.iter().copied().fold(0.0, f32::max);
        let predicted_maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let predicted_index = logits
            .iter()
            .position(|&logit| logit == predicted_maximum)
            .expect("policy logits must be non-empty");
        let top1_agrees = sample.policy[predicted_index] == target_maximum;
        let mut legal = [false; ACTION_COUNT];
        for action in sample.position.legal_actions() {
            legal[Board::action_index(action)] = true;
        }

        for (index, (&logit, &target)) in logits.iter().zip(&sample.policy).enumerate() {
            let log_probability = f64::from(logit) - log_sum_exp;
            let probability = log_probability.exp();
            let target = f64::from(target);
            if target > 0.0 {
                self.policy_cross_entropy -= target * log_probability;
                self.policy_target_entropy -= target * target.ln();
            }
            self.policy_predicted_entropy -= probability * log_probability;
            if legal[index] {
                self.policy_legal_mass += probability;
            }
        }
        self.policy_top1_target_agreement += f64::from(top1_agrees);

        let predicted = f64::from(predicted_value);
        let target = f64::from(sample.outcome.value());
        let error = predicted - target;
        self.value_squared_error += error * error;
        self.value_absolute_error += error.abs();
        self.value_sign_agreement += f64::from(predicted.signum() == target.signum());
        self.value_sum += predicted;
        self.value_squared_sum += predicted * predicted;
        self.target_sum += target;
        self.target_squared_sum += target * target;
        self.value_target_product_sum += predicted * target;
        self.rows += 1;
        Ok(())
    }

    fn report(&self, available_rows: usize, sampled_rows: usize, seed: u64) -> DiagnosticsReport {
        let count = self.rows as f64;
        let predicted_mean = self.value_sum / count;
        let predicted_variance = (self.value_squared_sum / count - predicted_mean.powi(2)).max(0.0);
        let target_mean = self.target_sum / count;
        let target_variance = (self.target_squared_sum / count - target_mean.powi(2)).max(0.0);
        let covariance = self.value_target_product_sum / count - predicted_mean * target_mean;
        let correlation = if predicted_variance == 0.0 || target_variance == 0.0 {
            f64::NAN
        } else {
            covariance / (predicted_variance * target_variance).sqrt()
        };
        let policy_cross_entropy = self.policy_cross_entropy / count;
        let policy_target_entropy = self.policy_target_entropy / count;

        DiagnosticsReport {
            available_rows,
            sampled_rows,
            seed,
            policy_cross_entropy,
            policy_target_entropy,
            policy_kl: policy_cross_entropy - policy_target_entropy,
            policy_predicted_entropy: self.policy_predicted_entropy / count,
            policy_legal_mass: self.policy_legal_mass / count,
            policy_top1_target_agreement: self.policy_top1_target_agreement / count,
            value_mse: self.value_squared_error / count,
            value_mae: self.value_absolute_error / count,
            value_sign_accuracy: self.value_sign_agreement / count,
            value_correlation: correlation,
            value_predicted_mean: predicted_mean,
            value_predicted_std: predicted_variance.sqrt(),
        }
    }
}

fn invalid_input(message: &str) -> Box<dyn Error> {
    io::Error::new(io::ErrorKind::InvalidInput, message).into()
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::*;
    use crate::game::Outcome;
    use crate::othello::{Color, Square};

    fn sample(position: Board, policy: [f32; ACTION_COUNT], outcome: Outcome) -> SelfPlaySample {
        SelfPlaySample {
            position,
            policy,
            outcome,
            game: 0,
        }
    }

    fn one_hot_sample(outcome: Outcome) -> SelfPlaySample {
        let mut policy = [0.0; ACTION_COUNT];
        policy[19] = 1.0;
        sample(Board::default(), policy, outcome)
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
    }

    fn frozen_config(output: PathBuf) -> FrozenTrainingConfig {
        FrozenTrainingConfig {
            steps: 1,
            batch_size: 128,
            learning_rate: 0.001,
            weight_decay: 0.0001,
            seed: 7,
            output,
        }
    }

    #[test]
    fn frozen_training_validates_arguments_and_empty_replay() {
        let base =
            std::env::temp_dir().join(format!("etive-frozen-validation-{}", std::process::id()));
        let config = frozen_config(base.join("output"));
        assert!(validate_frozen_training(&[], &config).is_err());

        let replay = base.join("empty.bin");
        fs::create_dir_all(&base).unwrap();
        let bytes = bincode::encode_to_vec(
            (2u8, Vec::<SelfPlaySample>::new()),
            bincode::config::standard(),
        )
        .unwrap();
        fs::write(&replay, bytes).unwrap();
        let error = train_frozen(
            base.join("missing-model.burnpack"),
            base.join("missing-optimizer.burnpack"),
            &[replay],
            config,
            Device::flex(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("contains no rows"));
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn frozen_output_refuses_collisions_and_resolves_architecture_metadata() {
        let base = std::env::temp_dir().join(format!("etive-frozen-output-{}", std::process::id()));
        if base.exists() {
            fs::remove_dir_all(&base).unwrap();
        }
        fs::create_dir_all(&base).unwrap();
        fs::write(
            base.join("model.toml"),
            toml::to_string(&OthelloModelConfig::WEEKEND).unwrap(),
        )
        .unwrap();
        assert_eq!(
            resolve_checkpoint_config(&base.join("model.burnpack")),
            OthelloModelConfig::WEEKEND
        );
        assert_eq!(
            resolve_checkpoint_config(&base.join("nested/missing.burnpack")),
            OthelloModelConfig::LEGACY
        );
        assert!(prepare_frozen_output(&base).is_err());
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn frozen_metrics_csv_has_header_and_one_row() {
        let report = FrozenTrainingReport {
            steps: 3,
            replay_rows: 20,
            batch_size: 4,
            seed: 7,
            learning_rate: 0.001,
            weight_decay: 0.0001,
            training_seconds: 1.25,
            policy_cross_entropy: 2.0,
            policy_target_entropy: 1.5,
            policy_kl: 0.5,
            value_mse: 0.25,
        };
        let csv = report.csv();
        let lines = csv.lines().collect::<Vec<_>>();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], FrozenTrainingReport::CSV_HEADER);
        assert_eq!(lines[1], report.to_string());
        assert_eq!(lines[0].split(',').count(), 11);
        assert_eq!(lines[1].split(',').count(), 11);
    }

    #[test]
    fn uniform_policy_metrics_are_exact() {
        let mut policy = [0.0; ACTION_COUNT];
        for action in Board::default().legal_actions() {
            policy[Board::action_index(action)] = 0.25;
        }
        let mut metrics = Metrics::default();
        metrics
            .add(
                &sample(Board::default(), policy, Outcome::Draw),
                &[0.0; ACTION_COUNT],
                0.0,
            )
            .unwrap();
        let report = metrics.report(1, 1, 7);

        assert_close(report.policy_cross_entropy, 65.0f64.ln());
        assert_close(report.policy_target_entropy, 4.0f64.ln());
        assert_close(report.policy_kl, (65.0f64 / 4.0).ln());
        assert_close(report.policy_predicted_entropy, 65.0f64.ln());
        assert_close(report.policy_legal_mass, 4.0 / 65.0);
    }

    #[test]
    fn forced_pass_is_included_in_legal_mass() {
        let a1 = Square::from_str("a1").unwrap().bitboard();
        let b1 = Square::from_str("b1").unwrap().bitboard();
        let board = Board::from_discs(a1, b1, Color::White).unwrap();
        let mut policy = [0.0; ACTION_COUNT];
        policy[ACTION_COUNT - 1] = 1.0;
        let mut metrics = Metrics::default();
        metrics
            .add(
                &sample(board, policy, Outcome::Draw),
                &[0.0; ACTION_COUNT],
                0.0,
            )
            .unwrap();

        assert_close(metrics.report(1, 1, 7).policy_legal_mass, 1.0 / 65.0);
    }

    #[test]
    fn top1_agreement_handles_ties_and_illegal_predictions() {
        let mut policy = [0.0; ACTION_COUNT];
        policy[19] = 0.5;
        policy[26] = 0.5;
        let target = sample(Board::default(), policy, Outcome::Draw);
        let mut tied_target = [-1.0; ACTION_COUNT];
        tied_target[26] = 2.0;
        let mut agreeing = Metrics::default();
        agreeing.add(&target, &tied_target, 0.0).unwrap();
        assert_eq!(agreeing.report(1, 1, 7).policy_top1_target_agreement, 1.0);

        let mut illegal = [-1.0; ACTION_COUNT];
        illegal[0] = 2.0;
        let mut disagreeing = Metrics::default();
        disagreeing.add(&target, &illegal, 0.0).unwrap();
        assert_eq!(
            disagreeing.report(1, 1, 7).policy_top1_target_agreement,
            0.0
        );

        let mut predicted_tie = [-1.0; ACTION_COUNT];
        predicted_tie[0] = 2.0;
        predicted_tie[19] = 2.0;
        let mut deterministic = Metrics::default();
        deterministic.add(&target, &predicted_tie, 0.0).unwrap();
        assert_eq!(
            deterministic.report(1, 1, 7).policy_top1_target_agreement,
            0.0
        );
    }

    #[test]
    fn value_metrics_and_degenerate_correlation() {
        let mut metrics = Metrics::default();
        for (outcome, prediction) in [
            (Outcome::Loss, -2.0),
            (Outcome::Draw, 0.0),
            (Outcome::Win, 2.0),
        ] {
            metrics
                .add(&one_hot_sample(outcome), &[0.0; ACTION_COUNT], prediction)
                .unwrap();
        }
        let report = metrics.report(3, 3, 7);
        assert_close(report.value_mse, 2.0 / 3.0);
        assert_close(report.value_mae, 2.0 / 3.0);
        assert_eq!(report.value_sign_accuracy, 1.0);
        assert_close(report.value_correlation, 1.0);
        assert_close(report.value_predicted_mean, 0.0);
        assert_close(report.value_predicted_std, (8.0f64 / 3.0).sqrt());

        let mut degenerate = Metrics::default();
        for outcome in [Outcome::Loss, Outcome::Win] {
            degenerate
                .add(&one_hot_sample(outcome), &[0.0; ACTION_COUNT], 0.0)
                .unwrap();
        }
        assert!(degenerate.report(2, 2, 7).value_correlation.is_nan());

        let mut constant_target = Metrics::default();
        for prediction in [-1.0, 1.0] {
            constant_target
                .add(
                    &one_hot_sample(Outcome::Draw),
                    &[0.0; ACTION_COUNT],
                    prediction,
                )
                .unwrap();
        }
        assert!(constant_target.report(2, 2, 7).value_correlation.is_nan());
    }

    #[test]
    fn sampling_is_deterministic_sorted_and_clamped() {
        let first = sample_indices(100, 12, 7);
        assert_eq!(first, sample_indices(100, 12, 7));
        assert_ne!(first, sample_indices(100, 12, 8));
        assert!(first.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(sample_indices(5, 20, 7), vec![0, 1, 2, 3, 4]);
    }
}
