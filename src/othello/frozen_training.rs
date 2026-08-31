use std::error::Error;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use burn::tensor::Device;

use super::replay::read_replay;
use super::training::TrainingSession;
use super::{OthelloModelConfig, OthelloNetwork};

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

fn invalid_input(message: &str) -> Box<dyn Error> {
    io::Error::new(io::ErrorKind::InvalidInput, message).into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::othello::replay::SelfPlaySample;

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
    fn validates_arguments_and_empty_replay() {
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
    fn output_refuses_collisions_and_resolves_architecture_metadata() {
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
    fn metrics_csv_has_header_and_one_row() {
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
}
