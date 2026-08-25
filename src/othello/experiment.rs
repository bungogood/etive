//! TOML-configured, resumable Othello self-play experiments.

mod replay;

use std::collections::VecDeque;
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use candle_core::Device;
use serde::{Deserialize, Serialize};

use super::actors::{ActorConfig, SelfPlaySample, run as run_actors};
use super::evaluation::{EvalConfig, evaluate};
use super::training::{TrainingSession, evaluate_loss};
use crate::evaluator::OthelloCandleEvaluator;
use crate::model::OthelloNetwork;

use replay::{
    atomic_replay_save, load_replay, read_replay, replay_path, trim_replay, validation_replay_path,
};

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    output: PathBuf,
    hours: f64,
    seed: u64,
    checkpoint: Option<PathBuf>,
    self_play: SelfPlay,
    train: Train,
    eval: Eval,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SelfPlay {
    games: usize,
    simulations: u32,
    workers: usize,
    inference_batch_size: usize,
    dirichlet_alpha: f64,
    dirichlet_fraction: f32,
    temperature_moves: usize,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Train {
    batch_size: usize,
    replay_positions: usize,
    replay_reuse: usize,
    learning_rate: f64,
    final_learning_rate: f64,
    validation_game_modulus: u64,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Eval {
    interval: usize,
    games: usize,
    simulations: u32,
    opening_plies: usize,
    seed: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
struct RunState {
    generation: usize,
    elapsed_seconds: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
struct PendingSelfPlay {
    generation: usize,
    training_samples: usize,
    validation_samples: usize,
    elapsed_seconds: f64,
    evaluations: u64,
    unique_games: usize,
}

pub fn run(
    config_path: impl AsRef<Path>,
    device: Device,
    clean: bool,
) -> Result<(), Box<dyn Error>> {
    let config_path = config_path.as_ref();
    let source = fs::read_to_string(config_path)?;
    let mut config: Config = toml::from_str(&source)?;
    resolve_paths(&mut config, config_path.parent().unwrap_or(Path::new(".")));
    validate(&config)?;

    if clean {
        clean_output(&config.output)?;
    }
    if config.output.exists() {
        resume_run(config, source, device)
    } else {
        start_run(config, source, device)
    }
}

fn start_run(config: Config, source: String, device: Device) -> Result<(), Box<dyn Error>> {
    if config.output.exists() {
        return Err(format!("output already exists: {}", config.output.display()).into());
    }
    fs::create_dir_all(config.output.join("replay"))?;
    fs::write(config.output.join("config.toml"), source)?;
    write_metrics_header(&config.output.join("metrics.csv"))?;

    let network = match &config.checkpoint {
        Some(path) => OthelloNetwork::load(path, &device)?,
        None => OthelloNetwork::new(&device, config.seed)?,
    };
    let trainer = TrainingSession::new(
        &network,
        device.clone(),
        config.train.batch_size,
        config.train.learning_rate,
        config.seed,
    )?;
    let checkpoint = checkpoint_path(&config.output, 0);
    let optimizer = optimizer_path(&config.output, 0);
    atomic_network_save(&network, &checkpoint)?;
    atomic_optimizer_save(&trainer, &optimizer)?;
    atomic_toml_save(
        &config.output.join("state.toml"),
        &RunState {
            generation: 0,
            elapsed_seconds: 0.0,
        },
    )?;

    run_loop(config, device, network, trainer, VecDeque::new(), 0, 0.0)
}

fn resume_run(config: Config, source: String, device: Device) -> Result<(), Box<dyn Error>> {
    let stored_config = fs::read_to_string(config.output.join("config.toml"))?;
    let state: RunState = toml::from_str(&fs::read_to_string(config.output.join("state.toml"))?)?;
    if stored_config != source {
        fs::write(
            config.output.join(format!(
                "config-before-resume-generation-{:04}.toml",
                state.generation
            )),
            stored_config,
        )?;
        fs::write(config.output.join("config.toml"), &source)?;
        println!(
            "configuration changed; archived the previous config at generation {}",
            state.generation
        );
    }
    repair_metrics(&config.output.join("metrics.csv"), state.generation)?;
    let network = OthelloNetwork::load(checkpoint_path(&config.output, state.generation), &device)?;
    let mut trainer = TrainingSession::new(
        &network,
        device.clone(),
        config.train.batch_size,
        config.train.learning_rate,
        config.seed,
    )?;
    trainer.load_optimizer(optimizer_path(&config.output, state.generation))?;
    let replay = load_replay(
        &config.output,
        state.generation,
        config.train.replay_positions,
    )?;
    println!(
        "resuming generation {} with {} replay positions after {:.1}s",
        state.generation,
        replay.iter().map(Vec::len).sum::<usize>(),
        state.elapsed_seconds
    );
    run_loop(
        config,
        device,
        network,
        trainer,
        replay,
        state.generation,
        state.elapsed_seconds,
    )
}

fn run_loop(
    config: Config,
    device: Device,
    network: OthelloNetwork,
    mut trainer: TrainingSession,
    mut replay: VecDeque<Vec<SelfPlaySample>>,
    mut generation: usize,
    prior_elapsed: f64,
) -> Result<(), Box<dyn Error>> {
    let run_start = Instant::now();
    let run_seconds = config.hours * 60.0 * 60.0;
    let state_path = config.output.join("state.toml");
    let metrics_path = config.output.join("metrics.csv");
    discard_committed_self_play(&config.output, generation)?;
    let mut recovered_elapsed = 0.0;

    while prior_elapsed + recovered_elapsed + run_start.elapsed().as_secs_f64() < run_seconds {
        generation += 1;
        let pending_path = config.output.join("pending-self-play.toml");
        let training_path = replay_path(&config.output, generation);
        let validation_path = validation_replay_path(&config.output, generation);
        let (training, validation, pending) = if let Some(recovered) =
            recover_self_play(&pending_path, &training_path, &validation_path, generation)?
        {
            let (training, validation, pending) = recovered;
            recovered_elapsed += pending.elapsed_seconds;
            println!(
                "generation {generation}: recovered {} persisted self-play positions",
                training.len() + validation.len()
            );
            (training, validation, pending)
        } else {
            println!("generation {generation}: starting self-play");
            io::stdout().flush()?;
            let self_play_start = Instant::now();
            let self_play = run_actors(
                OthelloCandleEvaluator::from_network(device.clone(), &network),
                ActorConfig {
                    games: config.self_play.games,
                    simulations: config.self_play.simulations,
                    workers: config.self_play.workers,
                    inference_batch_size: config.self_play.inference_batch_size,
                    seed: config.seed.wrapping_add(generation as u64),
                    dirichlet_alpha: config.self_play.dirichlet_alpha,
                    dirichlet_fraction: config.self_play.dirichlet_fraction,
                    temperature_moves: config.self_play.temperature_moves,
                },
            )?;
            let self_play_elapsed = self_play_start.elapsed();
            let (validation, training): (Vec<_>, Vec<_>) = self_play
                .samples
                .into_iter()
                .partition(|sample| sample.game % config.train.validation_game_modulus == 0);
            let pending = PendingSelfPlay {
                generation,
                training_samples: training.len(),
                validation_samples: validation.len(),
                elapsed_seconds: self_play_elapsed.as_secs_f64(),
                evaluations: self_play.evaluations,
                unique_games: self_play.unique_games,
            };
            atomic_replay_save(&training, &training_path)?;
            atomic_replay_save(&validation, &validation_path)?;
            atomic_toml_save(&pending_path, &pending)?;
            (training, validation, pending)
        };
        let sample_count = pending.training_samples + pending.validation_samples;
        let training_sample_count = pending.training_samples;
        let validation_sample_count = pending.validation_samples;
        let evaluations = pending.evaluations;
        let unique_games = pending.unique_games;
        replay.push_back(training);
        trim_replay(&mut replay, config.train.replay_positions);
        let replay_samples = replay.iter().map(Vec::len).sum::<usize>();
        println!(
            "generation {generation}: generated {sample_count} positions from {unique_games} unique games in {:.3?}",
            Duration::from_secs_f64(pending.elapsed_seconds)
        );

        let total_elapsed = prior_elapsed + recovered_elapsed + run_start.elapsed().as_secs_f64();
        let progress = (total_elapsed / run_seconds).min(1.0);
        let learning_rate = config.train.learning_rate
            * (config.train.final_learning_rate / config.train.learning_rate).powf(progress);
        trainer.set_learning_rate(learning_rate);
        trainer.reseed(config.seed.wrapping_add(generation as u64));
        let training_steps = training_sample_count
            .saturating_mul(config.train.replay_reuse)
            .div_ceil(config.train.batch_size);
        let replay_slices = replay.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let training_report = trainer.train_steps(&network, &replay_slices, training_steps)?;
        let validation_loss = if validation.is_empty() {
            None
        } else {
            Some(evaluate_loss(
                &network,
                &device,
                &validation,
                config.train.batch_size,
            )?)
        };
        println!(
            "generation {generation}: trained {training_steps} steps over {replay_samples} replay positions in {:.3?} at lr {learning_rate:.6}",
            training_report.elapsed
        );

        let checkpoint = checkpoint_path(&config.output, generation);
        let optimizer = optimizer_path(&config.output, generation);
        atomic_network_save(&network, &checkpoint)?;
        atomic_optimizer_save(&trainer, &optimizer)?;
        let evaluation = if generation.is_multiple_of(config.eval.interval) {
            let previous_network =
                OthelloNetwork::load(checkpoint_path(&config.output, generation - 1), &device)?;
            let mut previous =
                OthelloCandleEvaluator::from_network(device.clone(), &previous_network);
            let mut current = OthelloCandleEvaluator::from_network(device.clone(), &network);
            let result = evaluate(
                &mut current,
                &mut previous,
                EvalConfig {
                    games: config.eval.games,
                    simulations: config.eval.simulations,
                    batch_size: config.self_play.inference_batch_size,
                    opening_plies: config.eval.opening_plies,
                    seed: config.eval.seed.wrapping_add(generation as u64),
                },
                |_| {},
            )?;
            Some(result)
        } else {
            None
        };

        let (current_wins, previous_wins, draws, score) = match evaluation {
            Some(result) => {
                let score = (result.candidate_wins as f32 + 0.5 * result.draws as f32)
                    / config.eval.games as f32;
                println!(
                    "generation {generation}: evaluation {}-{}-{}, score {:.1}%",
                    result.candidate_wins,
                    result.baseline_wins,
                    result.draws,
                    score * 100.0
                );
                (
                    result.candidate_wins,
                    result.baseline_wins,
                    result.draws,
                    score,
                )
            }
            None => (0, 0, 0, f32::NAN),
        };
        let (validation_policy, validation_value) = validation_loss
            .map(|loss| (loss.policy_loss, loss.value_loss))
            .unwrap_or((f32::NAN, f32::NAN));
        append_metrics(
            &metrics_path,
            format!(
                "{generation},{sample_count},{training_sample_count},{validation_sample_count},{replay_samples},{:.6},{evaluations},{:.3},{learning_rate:.8},{},{:.6},{:.6},{:.6},{validation_policy:.6},{validation_value:.6},{},{current_wins},{previous_wins},{draws},{score:.6},{}",
                pending.elapsed_seconds,
                evaluations as f64 / pending.elapsed_seconds.max(f64::MIN_POSITIVE),
                training_report.steps,
                training_report.elapsed.as_secs_f64(),
                training_report.policy_loss,
                training_report.value_loss,
                evaluation.is_some(),
                checkpoint.display()
            ),
        )?;
        atomic_toml_save(
            &state_path,
            &RunState {
                generation,
                elapsed_seconds: prior_elapsed
                    + recovered_elapsed
                    + run_start.elapsed().as_secs_f64(),
            },
        )?;
        if pending_path.exists() {
            fs::remove_file(&pending_path)?;
        }
        if validation_path.exists() {
            fs::remove_file(&validation_path)?;
        }
    }
    println!(
        "experiment completed at generation {generation} after {:.3?}",
        Duration::from_secs_f64(
            prior_elapsed + recovered_elapsed + run_start.elapsed().as_secs_f64(),
        )
    );
    Ok(())
}

type RecoveredSelfPlay = (Vec<SelfPlaySample>, Vec<SelfPlaySample>, PendingSelfPlay);

fn discard_committed_self_play(output: &Path, generation: usize) -> Result<(), Box<dyn Error>> {
    let manifest_path = output.join("pending-self-play.toml");
    if !manifest_path.exists() {
        return Ok(());
    }
    let pending = toml::from_str::<PendingSelfPlay>(&fs::read_to_string(&manifest_path)?)?;
    if pending.generation <= generation {
        fs::remove_file(manifest_path)?;
        let validation_path = validation_replay_path(output, pending.generation);
        if validation_path.exists() {
            fs::remove_file(validation_path)?;
        }
    }
    Ok(())
}

fn recover_self_play(
    manifest_path: &Path,
    training_path: &Path,
    validation_path: &Path,
    generation: usize,
) -> Result<Option<RecoveredSelfPlay>, Box<dyn Error>> {
    if !manifest_path.exists() {
        return Ok(None);
    }
    let pending = toml::from_str::<PendingSelfPlay>(&fs::read_to_string(manifest_path)?)?;
    if pending.generation != generation {
        return Err("pending self-play generation does not match run state".into());
    }
    if !training_path.exists() || !validation_path.exists() {
        return Err("pending self-play manifest references missing replay data".into());
    }
    let training = read_replay(training_path)?;
    let validation = read_replay(validation_path)?;
    if pending.training_samples != training.len() || pending.validation_samples != validation.len()
    {
        return Err("pending self-play manifest does not match replay data".into());
    }
    Ok(Some((training, validation, pending)))
}

fn validate(config: &Config) -> Result<(), Box<dyn Error>> {
    if !config.hours.is_finite()
        || config.hours <= 0.0
        || config.self_play.games == 0
        || config.self_play.simulations < 2
        || config.self_play.workers == 0
        || config.self_play.inference_batch_size == 0
        || !config.self_play.dirichlet_alpha.is_finite()
        || config.self_play.dirichlet_alpha <= 0.0
        || !config.self_play.dirichlet_fraction.is_finite()
        || !(0.0..=1.0).contains(&config.self_play.dirichlet_fraction)
        || config.train.batch_size == 0
        || config.train.replay_positions == 0
        || config.train.replay_reuse == 0
        || !config.train.learning_rate.is_finite()
        || config.train.learning_rate <= 0.0
        || !config.train.final_learning_rate.is_finite()
        || config.train.final_learning_rate <= 0.0
        || config.train.validation_game_modulus < 2
        || config.eval.interval == 0
        || config.eval.games == 0
        || !config.eval.games.is_multiple_of(2)
        || config.eval.simulations < 2
    {
        return Err("invalid experiment configuration".into());
    }
    Ok(())
}

fn resolve_paths(config: &mut Config, base: &Path) {
    if config.output.is_relative() {
        config.output = base.join(&config.output);
    }
    if let Some(checkpoint) = &mut config.checkpoint
        && checkpoint.is_relative()
    {
        *checkpoint = base.join(&*checkpoint);
    }
}

fn clean_output(output: &Path) -> Result<(), Box<dyn Error>> {
    if !output.exists() {
        return Ok(());
    }
    if !output.join("config.toml").is_file() || !output.join("state.toml").is_file() {
        return Err(format!(
            "refusing to clean unrecognized output directory: {}",
            output.display()
        )
        .into());
    }
    fs::remove_dir_all(output)?;
    Ok(())
}

fn write_metrics_header(path: &Path) -> io::Result<()> {
    fs::write(
        path,
        "generation,samples,training_samples,validation_samples,replay_samples,self_play_seconds,self_play_evaluations,self_play_evaluations_per_second,learning_rate,training_steps,training_seconds,policy_loss,value_loss,validation_policy_loss,validation_value_loss,evaluated,current_wins,previous_wins,draws,score,checkpoint\n",
    )
}

fn append_metrics(path: &Path, row: String) -> io::Result<()> {
    let mut file = OpenOptions::new().append(true).open(path)?;
    writeln!(file, "{row}")?;
    file.flush()
}

fn repair_metrics(path: &Path, generation: usize) -> io::Result<()> {
    let contents = fs::read_to_string(path)?;
    let mut lines = contents.lines();
    let Some(header) = lines.next() else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "metrics header missing",
        ));
    };
    let mut repaired = String::with_capacity(contents.len());
    repaired.push_str(header);
    repaired.push('\n');
    for line in lines {
        let row_generation = line
            .split(',')
            .next()
            .and_then(|value| value.parse::<usize>().ok());
        if row_generation.is_some_and(|row| row <= generation) {
            repaired.push_str(line);
            repaired.push('\n');
        }
    }
    fs::write(path, repaired)
}

fn atomic_network_save(network: &OthelloNetwork, path: &Path) -> candle_core::Result<()> {
    let temporary = temporary_path(path);
    network.save(&temporary)?;
    fs::rename(&temporary, path).map_err(candle_core::Error::from)?;
    Ok(())
}

fn atomic_optimizer_save(trainer: &TrainingSession, path: &Path) -> candle_core::Result<()> {
    let temporary = temporary_path(path);
    trainer.save_optimizer(&temporary)?;
    fs::rename(&temporary, path).map_err(candle_core::Error::from)?;
    Ok(())
}

fn atomic_toml_save(path: &Path, value: &impl Serialize) -> Result<(), Box<dyn Error>> {
    let temporary = temporary_path(path);
    fs::write(&temporary, toml::to_string(value)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn temporary_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".tmp");
    PathBuf::from(name)
}

fn checkpoint_path(output: &Path, generation: usize) -> PathBuf {
    output.join(format!("generation-{generation:04}.safetensors"))
}

fn optimizer_path(output: &Path, generation: usize) -> PathBuf {
    output.join(format!("generation-{generation:04}-optimizer.safetensors"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Outcome;
    use crate::othello::Board;

    const CONFIG: &str = r#"
output = "checkpoints/run"
hours = 24.0
seed = 7

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
validation_game_modulus = 20

[eval]
interval = 2
games = 500
simulations = 256
opening_plies = 8
seed = 4242
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
    fn clean_only_removes_recognized_runs() {
        let output = std::env::temp_dir().join(format!("etive-clean-{}", std::process::id()));
        if output.exists() {
            fs::remove_dir_all(&output).unwrap();
        }
        fs::create_dir(&output).unwrap();

        assert!(clean_output(&output).is_err());
        assert!(output.exists());

        fs::write(output.join("config.toml"), CONFIG).unwrap();
        fs::write(
            output.join("state.toml"),
            "generation = 0\nelapsed_seconds = 0.0\n",
        )
        .unwrap();
        clean_output(&output).unwrap();
        assert!(!output.exists());
    }

    #[test]
    fn recovery_requires_a_matching_committed_manifest() {
        let output = std::env::temp_dir().join(format!("etive-recovery-{}", std::process::id()));
        if output.exists() {
            fs::remove_dir_all(&output).unwrap();
        }
        let replay = output.join("replay");
        fs::create_dir_all(&replay).unwrap();
        let manifest_path = output.join("pending-self-play.toml");
        let training_path = replay_path(&output, 1);
        let validation_path = validation_replay_path(&output, 1);
        let mut policy = [0.0; 65];
        policy[19] = 1.0;
        let sample = SelfPlaySample {
            position: Board::default(),
            policy,
            outcome: Outcome::Draw,
            game: 1,
        };
        atomic_replay_save(std::slice::from_ref(&sample), &training_path).unwrap();

        assert!(
            recover_self_play(&manifest_path, &training_path, &validation_path, 1)
                .unwrap()
                .is_none()
        );

        atomic_replay_save(&[], &validation_path).unwrap();
        atomic_toml_save(
            &manifest_path,
            &PendingSelfPlay {
                generation: 1,
                training_samples: 1,
                validation_samples: 0,
                elapsed_seconds: 1.0,
                evaluations: 2,
                unique_games: 1,
            },
        )
        .unwrap();
        let recovered =
            recover_self_play(&manifest_path, &training_path, &validation_path, 1).unwrap();
        assert_eq!(recovered.unwrap().0.len(), 1);

        discard_committed_self_play(&output, 1).unwrap();
        assert!(!manifest_path.exists());
        assert!(!validation_path.exists());
        assert!(training_path.exists());

        fs::remove_dir_all(output).unwrap();
    }
}
