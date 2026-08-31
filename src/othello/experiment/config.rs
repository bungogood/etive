use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::self_play;

use super::super::OthelloModelConfig;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Config {
    pub(super) output: PathBuf,
    pub(super) hours: f64,
    pub(super) seed: u64,
    pub(super) checkpoint: Option<PathBuf>,
    pub(super) model: OthelloModelConfig,
    pub(super) self_play: self_play::Config,
    pub(super) train: Train,
    pub(super) eval: Eval,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Train {
    pub(super) batch_size: usize,
    pub(super) replay_positions: usize,
    pub(super) replay_reuse: usize,
    pub(super) learning_rate: f64,
    pub(super) final_learning_rate: f64,
    pub(super) weight_decay: f32,
    pub(super) validation_game_modulus: u64,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Eval {
    pub(super) interval: usize,
    pub(super) games: usize,
    pub(super) simulations: u32,
    pub(super) opening_plies: usize,
    pub(super) seed: u64,
    pub(super) promotion_los: f64,
}

pub struct SelfPlayBenchmarkConfig {
    pub model: OthelloModelConfig,
    pub checkpoint: Option<PathBuf>,
    pub seed: u64,
    pub self_play: self_play::Config,
}

pub fn load_self_play_benchmark_config(
    config_path: impl AsRef<Path>,
) -> Result<SelfPlayBenchmarkConfig, Box<dyn Error>> {
    let config_path = config_path.as_ref();
    let mut config: Config = toml::from_str(&fs::read_to_string(config_path)?)?;
    resolve_paths(&mut config, config_path.parent().unwrap_or(Path::new(".")));
    validate(&config)?;
    Ok(SelfPlayBenchmarkConfig {
        model: config.model,
        checkpoint: config.checkpoint,
        seed: config.seed,
        self_play: config.self_play,
    })
}

pub(super) fn validate(config: &Config) -> Result<(), Box<dyn Error>> {
    if !config.hours.is_finite()
        || config.hours <= 0.0
        || !config.self_play.is_valid()
        || config.train.batch_size == 0
        || config.train.replay_positions == 0
        || config.train.replay_reuse == 0
        || !config.train.learning_rate.is_finite()
        || config.train.learning_rate <= 0.0
        || !config.train.final_learning_rate.is_finite()
        || config.train.final_learning_rate <= 0.0
        || !config.train.weight_decay.is_finite()
        || config.train.weight_decay < 0.0
        || config.train.validation_game_modulus < 2
        || config.eval.interval == 0
        || config.eval.games == 0
        || !config.eval.games.is_multiple_of(2)
        || config.eval.simulations < 2
        || !(0.5..1.0).contains(&config.eval.promotion_los)
        || config.model.validate().is_err()
    {
        return Err("invalid experiment configuration".into());
    }
    Ok(())
}

pub(super) fn resolve_paths(config: &mut Config, base: &Path) {
    if config.output.is_relative() {
        config.output = base.join(&config.output);
    }
    if let Some(checkpoint) = &mut config.checkpoint
        && checkpoint.is_relative()
    {
        *checkpoint = base.join(&*checkpoint);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = r#"
output = "checkpoints/run"
hours = 24.0
seed = 7

[model]
channels = 128
residual_blocks = 10
norm_groups = 8

[self_play]
games = 4096
simulations = 256
workers = 8
inference_batch_size = 1024
dirichlet_alpha = 0.3
dirichlet_fraction = 0.25
temperature_moves = 20

[train]
batch_size = 256
replay_positions = 4000000
replay_reuse = 4
learning_rate = 0.001
final_learning_rate = 0.0003
weight_decay = 0.01
validation_game_modulus = 20

[eval]
interval = 2
games = 500
simulations = 256
opening_plies = 8
seed = 4242
promotion_los = 0.95
"#;

    #[test]
    fn config_has_three_strict_sections() {
        let config: Config = toml::from_str(CONFIG).unwrap();

        assert_eq!(config.self_play.games, 4096);
        assert_eq!(config.train.batch_size, 256);
        assert_eq!(config.eval.simulations, 256);
        assert!(validate(&config).is_ok());
        assert!(toml::from_str::<Config>(&format!("{CONFIG}\nextra = true")).is_err());
    }

    #[test]
    fn tracked_experiment_configs_use_the_current_schema() {
        let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("experiments");
        for entry in fs::read_dir(directory).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_none_or(|extension| extension != "toml") {
                continue;
            }
            let config: Config = toml::from_str(&fs::read_to_string(&path).unwrap())
                .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
            assert!(validate(&config).is_ok(), "{}", path.display());
        }
    }
}
