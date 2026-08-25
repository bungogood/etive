//! TOML-configured, resumable Othello self-play experiments.

use std::collections::VecDeque;
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use candle_core::Device;
use serde::{Deserialize, Serialize};

use super::actors::{ActorConfig, SelfPlaySample, run as run_actors};
use super::training::{ArenaConfig, TrainingSession, arena_with_progress, evaluate_loss};
use super::{BitBoard, Board, Color};
use crate::evaluator::OthelloCandleEvaluator;
use crate::game::Outcome;
use crate::model::OthelloNetwork;

const REPLAY_MAGIC: &[u8; 8] = b"ETRP0001";

#[derive(Clone, Debug, Deserialize)]
pub struct ExperimentConfig {
    pub model: ModelConfig,
    pub run: RunConfig,
    pub self_play: SelfPlayConfig,
    pub training: LearnerConfig,
    pub evaluation: EvaluationConfig,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ModelConfig {
    pub architecture: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RunConfig {
    pub output: PathBuf,
    pub hours: f64,
    pub seed: u64,
    pub start_generation: usize,
    pub checkpoint: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct SelfPlayConfig {
    pub games: usize,
    pub simulations: u32,
    pub workers: usize,
    pub inference_batch_size: usize,
    pub dirichlet_alpha: f64,
    pub dirichlet_fraction: f32,
    pub temperature_moves: usize,
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct LearnerConfig {
    pub batch_size: usize,
    pub replay_positions: usize,
    pub replay_reuse: usize,
    pub learning_rate: f64,
    pub final_learning_rate: f64,
    pub validation_game_modulus: u64,
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct EvaluationConfig {
    pub interval: usize,
    pub games: usize,
    pub simulations: u32,
    pub opening_plies: usize,
    pub seed: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
struct RunState {
    generation: usize,
    elapsed_seconds: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
struct PendingSelfPlay {
    generation: usize,
    samples: usize,
    training_samples: usize,
    validation_samples: usize,
    elapsed_seconds: f64,
    evaluations: u64,
    unique_games: usize,
}

pub fn run(
    config_path: impl AsRef<Path>,
    device: Device,
    resume: bool,
) -> Result<(), Box<dyn Error>> {
    let config_path = config_path.as_ref();
    let source = fs::read_to_string(config_path)?;
    let mut config: ExperimentConfig = toml::from_str(&source)?;
    resolve_paths(&mut config, config_path.parent().unwrap_or(Path::new(".")));
    validate(&config)?;

    if resume {
        resume_run(config, source, device)
    } else {
        start_run(config, source, device)
    }
}

fn start_run(
    config: ExperimentConfig,
    source: String,
    device: Device,
) -> Result<(), Box<dyn Error>> {
    if config.run.output.exists() {
        return Err(format!("output already exists: {}", config.run.output.display()).into());
    }
    fs::create_dir_all(config.run.output.join("replay"))?;
    fs::write(config.run.output.join("config.toml"), source)?;
    write_metrics_header(&config.run.output.join("metrics.csv"))?;

    let network = match &config.run.checkpoint {
        Some(path) => OthelloNetwork::load(path, &device)?,
        None => OthelloNetwork::new(&device, config.run.seed)?,
    };
    let trainer = TrainingSession::new(
        &network,
        device.clone(),
        config.training.batch_size,
        config.training.learning_rate,
        config.run.seed,
    )?;
    let checkpoint = checkpoint_path(&config.run.output, config.run.start_generation);
    let optimizer = optimizer_path(&config.run.output, config.run.start_generation);
    atomic_network_save(&network, &checkpoint)?;
    atomic_optimizer_save(&trainer, &optimizer)?;
    atomic_state_save(
        &config.run.output.join("state.toml"),
        RunState {
            generation: config.run.start_generation,
            elapsed_seconds: 0.0,
        },
    )?;

    run_loop(config, device, network, trainer, VecDeque::new(), 0.0)
}

fn resume_run(
    config: ExperimentConfig,
    source: String,
    device: Device,
) -> Result<(), Box<dyn Error>> {
    let stored_config = fs::read_to_string(config.run.output.join("config.toml"))?;
    let state: RunState =
        toml::from_str(&fs::read_to_string(config.run.output.join("state.toml"))?)?;
    if stored_config != source {
        fs::write(
            config.run.output.join(format!(
                "config-before-resume-generation-{:04}.toml",
                state.generation
            )),
            stored_config,
        )?;
        fs::write(config.run.output.join("config.toml"), &source)?;
        println!(
            "configuration changed; archived the previous config at generation {}",
            state.generation
        );
    }
    repair_metrics(&config.run.output.join("metrics.csv"), state.generation)?;
    let network = OthelloNetwork::load(
        checkpoint_path(&config.run.output, state.generation),
        &device,
    )?;
    let mut trainer = TrainingSession::new(
        &network,
        device.clone(),
        config.training.batch_size,
        config.training.learning_rate,
        config.run.seed,
    )?;
    trainer.load_optimizer(optimizer_path(&config.run.output, state.generation))?;
    let replay = load_replay(
        &config.run.output,
        state.generation,
        config.training.replay_positions,
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
        state.elapsed_seconds,
    )
}

fn run_loop(
    config: ExperimentConfig,
    device: Device,
    mut network: OthelloNetwork,
    mut trainer: TrainingSession,
    mut replay: VecDeque<Vec<SelfPlaySample>>,
    prior_elapsed: f64,
) -> Result<(), Box<dyn Error>> {
    let run_start = Instant::now();
    let run_seconds = config.run.hours * 60.0 * 60.0;
    let state_path = config.run.output.join("state.toml");
    let metrics_path = config.run.output.join("metrics.csv");
    let mut generation = current_generation(&state_path)?;
    let mut recovered_elapsed = 0.0;

    while prior_elapsed + recovered_elapsed + run_start.elapsed().as_secs_f64() < run_seconds {
        generation += 1;
        let pending_path = config.run.output.join("pending-self-play.toml");
        let training_path = replay_path(&config.run.output, generation);
        let validation_path = validation_replay_path(&config.run.output, generation);
        let (training, validation, pending) = if training_path.exists() {
            let training = read_replay(&training_path)?;
            let validation = if validation_path.exists() {
                read_replay(&validation_path)?
            } else {
                Vec::new()
            };
            let pending = if pending_path.exists() {
                toml::from_str::<PendingSelfPlay>(&fs::read_to_string(&pending_path)?)?
            } else {
                PendingSelfPlay {
                    generation,
                    samples: training.len() + validation.len(),
                    training_samples: training.len(),
                    validation_samples: validation.len(),
                    elapsed_seconds: 0.0,
                    evaluations: 0,
                    unique_games: 0,
                }
            };
            if pending.generation != generation {
                return Err("pending self-play generation does not match run state".into());
            }
            recovered_elapsed += pending.elapsed_seconds;
            println!(
                "generation {generation}: recovered {} persisted self-play positions",
                pending.samples
            );
            (training, validation, pending)
        } else {
            println!("generation {generation}: starting self-play");
            io::stdout().flush()?;
            let self_play_start = Instant::now();
            let (self_play, evaluator) = run_actors(
                OthelloCandleEvaluator::from_network(device.clone(), network),
                ActorConfig {
                    games: config.self_play.games,
                    simulations: config.self_play.simulations,
                    workers: config.self_play.workers,
                    inference_batch_size: config.self_play.inference_batch_size,
                    seed: config.run.seed.wrapping_add(generation as u64),
                    dirichlet_alpha: config.self_play.dirichlet_alpha,
                    dirichlet_fraction: config.self_play.dirichlet_fraction,
                    temperature_moves: config.self_play.temperature_moves,
                },
            )?;
            let self_play_elapsed = self_play_start.elapsed();
            network = evaluator.into_network();
            let (validation, training): (Vec<_>, Vec<_>) = self_play
                .samples
                .into_iter()
                .partition(|sample| sample.game % config.training.validation_game_modulus == 0);
            let pending = PendingSelfPlay {
                generation,
                samples: training.len() + validation.len(),
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
        let sample_count = pending.samples;
        let training_sample_count = pending.training_samples;
        let validation_sample_count = pending.validation_samples;
        let evaluations = pending.evaluations;
        let unique_games = pending.unique_games;
        replay.push_back(training);
        trim_replay(&mut replay, config.training.replay_positions);
        let replay_samples = replay.iter().map(Vec::len).sum::<usize>();
        println!(
            "generation {generation}: generated {sample_count} positions from {unique_games} unique games in {:.3?}",
            Duration::from_secs_f64(pending.elapsed_seconds)
        );

        let total_elapsed = prior_elapsed + recovered_elapsed + run_start.elapsed().as_secs_f64();
        let progress = (total_elapsed / run_seconds).min(1.0);
        let learning_rate = config.training.learning_rate
            * (config.training.final_learning_rate / config.training.learning_rate).powf(progress);
        trainer.set_learning_rate(learning_rate);
        trainer.reseed(config.run.seed.wrapping_add(generation as u64));
        let training_steps = training_sample_count
            .saturating_mul(config.training.replay_reuse)
            .div_ceil(config.training.batch_size);
        let replay_slices = replay.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let training_report = trainer.train_steps(&network, &replay_slices, training_steps)?;
        let validation_loss = if validation.is_empty() {
            None
        } else {
            Some(evaluate_loss(
                &network,
                &device,
                &validation,
                config.training.batch_size,
            )?)
        };
        println!(
            "generation {generation}: trained {training_steps} steps over {replay_samples} replay positions in {:.3?} at lr {learning_rate:.6}",
            training_report.elapsed
        );

        let checkpoint = checkpoint_path(&config.run.output, generation);
        let optimizer = optimizer_path(&config.run.output, generation);
        atomic_network_save(&network, &checkpoint)?;
        atomic_optimizer_save(&trainer, &optimizer)?;
        let evaluation = if generation.is_multiple_of(config.evaluation.interval) {
            let previous_network =
                OthelloNetwork::load(checkpoint_path(&config.run.output, generation - 1), &device)?;
            let mut previous =
                OthelloCandleEvaluator::from_network(device.clone(), previous_network);
            let mut current = OthelloCandleEvaluator::from_network(device.clone(), network);
            let result = arena_with_progress(
                &mut current,
                &mut previous,
                ArenaConfig {
                    games: config.evaluation.games,
                    simulations: config.evaluation.simulations,
                    batch_size: config.self_play.inference_batch_size,
                    opening_plies: config.evaluation.opening_plies,
                    seed: config.evaluation.seed.wrapping_add(generation as u64),
                },
                |_| {},
            )?;
            network = current.into_network();
            Some(result)
        } else {
            None
        };

        let (current_wins, previous_wins, draws, score) = match evaluation {
            Some(result) => {
                let score = (result.trained_wins as f32 + 0.5 * result.draws as f32)
                    / config.evaluation.games as f32;
                println!(
                    "generation {generation}: evaluation {}-{}-{}, score {:.1}%",
                    result.trained_wins,
                    result.initial_wins,
                    result.draws,
                    score * 100.0
                );
                (
                    result.trained_wins,
                    result.initial_wins,
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
        atomic_state_save(
            &state_path,
            RunState {
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

fn validate(config: &ExperimentConfig) -> Result<(), Box<dyn Error>> {
    if config.model.architecture != "residual-10x128-groupnorm" {
        return Err(format!(
            "unsupported model architecture: {}",
            config.model.architecture
        )
        .into());
    }
    if !config.run.hours.is_finite()
        || config.run.hours <= 0.0
        || config.self_play.games == 0
        || config.self_play.simulations < 2
        || config.self_play.workers == 0
        || config.self_play.inference_batch_size == 0
        || config.training.batch_size == 0
        || config.training.replay_positions == 0
        || config.training.replay_reuse == 0
        || config.training.learning_rate <= 0.0
        || config.training.final_learning_rate <= 0.0
        || config.training.validation_game_modulus < 2
        || config.evaluation.interval == 0
        || config.evaluation.games == 0
        || !config.evaluation.games.is_multiple_of(2)
        || config.evaluation.simulations < 2
    {
        return Err("invalid experiment configuration".into());
    }
    Ok(())
}

fn resolve_paths(config: &mut ExperimentConfig, base: &Path) {
    if config.run.output.is_relative() {
        config.run.output = base.join(&config.run.output);
    }
    if let Some(checkpoint) = &mut config.run.checkpoint
        && checkpoint.is_relative()
    {
        *checkpoint = base.join(&*checkpoint);
    }
}

fn trim_replay(replay: &mut VecDeque<Vec<SelfPlaySample>>, maximum: usize) {
    let mut samples = replay.iter().map(Vec::len).sum::<usize>();
    while samples > maximum && replay.len() > 1 {
        samples -= replay.pop_front().expect("non-empty replay").len();
    }
}

fn load_replay(
    output: &Path,
    generation: usize,
    maximum: usize,
) -> Result<VecDeque<Vec<SelfPlaySample>>, Box<dyn Error>> {
    let mut replay = VecDeque::new();
    let mut samples = 0;
    for generation in (1..=generation).rev() {
        let path = replay_path(output, generation);
        if !path.exists() {
            continue;
        }
        let shard = read_replay(&path)?;
        samples += shard.len();
        replay.push_front(shard);
        if samples >= maximum {
            break;
        }
    }
    trim_replay(&mut replay, maximum);
    Ok(replay)
}

fn atomic_replay_save(samples: &[SelfPlaySample], path: &Path) -> Result<(), Box<dyn Error>> {
    let temporary = temporary_path(path);
    let mut writer = BufWriter::new(File::create(&temporary)?);
    writer.write_all(REPLAY_MAGIC)?;
    writer.write_all(&(samples.len() as u64).to_le_bytes())?;
    for sample in samples {
        writer.write_all(&sample.position.discs(Color::Black).0.to_le_bytes())?;
        writer.write_all(&sample.position.discs(Color::White).0.to_le_bytes())?;
        writer.write_all(&[match sample.position.side_to_move() {
            Color::Black => 0,
            Color::White => 1,
        }])?;
        for probability in sample.policy {
            writer.write_all(&probability.to_le_bytes())?;
        }
        writer.write_all(&[match sample.outcome {
            Outcome::Loss => 0,
            Outcome::Draw => 1,
            Outcome::Win => 2,
        }])?;
        writer.write_all(&sample.game.to_le_bytes())?;
    }
    writer.flush()?;
    drop(writer);
    fs::rename(temporary, path)?;
    Ok(())
}

fn read_replay(path: &Path) -> Result<Vec<SelfPlaySample>, Box<dyn Error>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut magic = [0; 8];
    reader.read_exact(&mut magic)?;
    if &magic != REPLAY_MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid replay header").into());
    }
    let count = read_u64(&mut reader)? as usize;
    let mut samples = Vec::with_capacity(count);
    for _ in 0..count {
        let black = read_u64(&mut reader)?;
        let white = read_u64(&mut reader)?;
        let side = match read_u8(&mut reader)? {
            0 => Color::Black,
            1 => Color::White,
            _ => return Err(invalid_replay("invalid side to move")),
        };
        let position = Board::from_discs(BitBoard(black), BitBoard(white), side)?;
        let mut policy = [0.0; 65];
        for probability in &mut policy {
            *probability = read_f32(&mut reader)?;
        }
        let outcome = match read_u8(&mut reader)? {
            0 => Outcome::Loss,
            1 => Outcome::Draw,
            2 => Outcome::Win,
            _ => return Err(invalid_replay("invalid outcome")),
        };
        samples.push(SelfPlaySample {
            position,
            policy,
            outcome,
            game: read_u64(&mut reader)?,
        });
    }
    Ok(samples)
}

fn read_u8(reader: &mut impl Read) -> io::Result<u8> {
    let mut bytes = [0];
    reader.read_exact(&mut bytes)?;
    Ok(bytes[0])
}

fn read_u64(reader: &mut impl Read) -> io::Result<u64> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_f32(reader: &mut impl Read) -> io::Result<f32> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

fn invalid_replay(message: &str) -> Box<dyn Error> {
    io::Error::new(io::ErrorKind::InvalidData, message).into()
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

fn current_generation(path: &Path) -> Result<usize, Box<dyn Error>> {
    let state: RunState = toml::from_str(&fs::read_to_string(path)?)?;
    Ok(state.generation)
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

fn atomic_state_save(path: &Path, state: RunState) -> Result<(), Box<dyn Error>> {
    atomic_toml_save(path, &state)
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

fn replay_path(output: &Path, generation: usize) -> PathBuf {
    output
        .join("replay")
        .join(format!("generation-{generation:04}.bin"))
}

fn validation_replay_path(output: &Path, generation: usize) -> PathBuf {
    output
        .join("replay")
        .join(format!("generation-{generation:04}-validation.bin"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_round_trips() {
        let path = std::env::temp_dir().join(format!(
            "etive-replay-{}-{}.bin",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let mut policy = [0.0; 65];
        policy[19] = 1.0;
        let samples = [SelfPlaySample {
            position: Board::default(),
            policy,
            outcome: Outcome::Win,
            game: 42,
        }];

        atomic_replay_save(&samples, &path).unwrap();
        let loaded = read_replay(&path).unwrap();
        fs::remove_file(path).unwrap();

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].position, samples[0].position);
        assert_eq!(loaded[0].policy, samples[0].policy);
        assert_eq!(loaded[0].outcome, Outcome::Win);
        assert_eq!(loaded[0].game, 42);
    }
}
