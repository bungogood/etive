//! TOML-configured, resumable Othello self-play experiments.

pub(crate) mod replay;

use std::collections::VecDeque;
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use burn::tensor::Device;
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use tracing::{info, info_span, warn};

use super::actors::{ActorConfig, SelfPlaySample, run as run_actors};
use super::evaluation::{EvalConfig, EvalResult, evaluate};
use super::training::{TrainingSession, evaluate_loss};
use crate::evaluator::OthelloBurnEvaluator;
use crate::model::{OthelloModelConfig, OthelloNetwork};

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
    #[serde(default)]
    model: OthelloModelConfig,
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

impl SelfPlay {
    fn actor_config(self, seed: u64) -> ActorConfig {
        ActorConfig {
            games: self.games,
            simulations: self.simulations,
            workers: self.workers,
            inference_batch_size: self.inference_batch_size,
            seed,
            dirichlet_alpha: self.dirichlet_alpha,
            dirichlet_fraction: self.dirichlet_fraction,
            temperature_moves: self.temperature_moves,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Train {
    batch_size: usize,
    replay_positions: usize,
    replay_reuse: usize,
    learning_rate: f64,
    final_learning_rate: f64,
    #[serde(default = "default_weight_decay")]
    weight_decay: f32,
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
    #[serde(default)]
    promotion_threshold: Option<f32>,
    #[serde(default)]
    promotion_los: Option<f64>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
struct RunState {
    generation: usize,
    elapsed_seconds: f64,
    #[serde(default)]
    champion_generation: Option<usize>,
    #[serde(default)]
    network_generation: Option<usize>,
}

const fn default_weight_decay() -> f32 {
    0.01
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

struct GenerationSelfPlay {
    training: Vec<SelfPlaySample>,
    validation: Vec<SelfPlaySample>,
    pending: PendingSelfPlay,
    recovered: bool,
}

pub struct SelfPlayBenchmarkConfig {
    pub model: OthelloModelConfig,
    pub checkpoint: Option<PathBuf>,
    pub actor: ActorConfig,
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
        actor: config.self_play.actor_config(config.seed),
    })
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

    let _lock = acquire_run_lock(&config.output)?;
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
    let staging = prepare_staging(&config.output)?;

    let training_device = device.clone().autodiff();
    let network = match &config.checkpoint {
        Some(path) => {
            let checkpoint_config =
                OthelloNetwork::checkpoint_config(path).unwrap_or(OthelloModelConfig::LEGACY);
            if checkpoint_config != config.model {
                return Err("checkpoint model configuration does not match experiment".into());
            }
            OthelloNetwork::load_with_config(path, &training_device, config.model)?
        }
        None => OthelloNetwork::new_with_config(&training_device, config.seed, config.model),
    };
    let trainer = TrainingSession::new(
        training_device,
        config.train.batch_size,
        config.train.learning_rate,
        config.train.weight_decay,
        config.seed,
    )?;
    fs::write(staging.join("config.toml"), source)?;
    fs::write(staging.join("model.toml"), toml::to_string(&config.model)?)?;
    write_metrics_header(&staging.join("metrics.csv"))?;
    let checkpoint = checkpoint_path(&staging, 0);
    let optimizer = optimizer_path(&staging, 0);
    atomic_network_save(&network, &checkpoint)?;
    atomic_optimizer_save(&trainer, &optimizer)?;
    let state = RunState {
        generation: 0,
        elapsed_seconds: 0.0,
        champion_generation: Some(0),
        network_generation: Some(0),
    };
    atomic_toml_save(&staging.join("state.toml"), &state)?;
    fs::rename(staging, &config.output)?;

    run_loop(config, device, network, trainer, VecDeque::new(), state)
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
        warn!(
            generation = state.generation,
            "configuration changed; archived previous configuration"
        );
    }
    repair_metrics(&config.output.join("metrics.csv"), state.generation)?;
    let (champion_generation, network_generation) = resume_generations(state)?;
    let network_path = checkpoint_path(&config.output, network_generation);
    let stored_model =
        OthelloNetwork::checkpoint_config(&network_path).unwrap_or(OthelloModelConfig::LEGACY);
    if stored_model != config.model {
        return Err("stored model configuration does not match experiment".into());
    }
    let training_device = device.clone().autodiff();
    let network = OthelloNetwork::load(network_path, &training_device)?;
    let mut trainer = TrainingSession::new(
        training_device,
        config.train.batch_size,
        config.train.learning_rate,
        config.train.weight_decay,
        config.seed,
    )?;
    trainer.load_optimizer(optimizer_path(&config.output, network_generation))?;
    let replay = load_replay(
        &config.output,
        state.generation,
        config.train.replay_positions,
    )?;
    info!(
        generation = state.generation,
        champion_generation,
        network_generation,
        replay_positions = replay.iter().map(Vec::len).sum::<usize>(),
        elapsed_seconds = state.elapsed_seconds,
        "resuming experiment"
    );
    run_loop(config, device, network, trainer, replay, state)
}

fn run_loop(
    config: Config,
    device: Device,
    mut network: OthelloNetwork,
    mut trainer: TrainingSession,
    mut replay: VecDeque<Vec<SelfPlaySample>>,
    state: RunState,
) -> Result<(), Box<dyn Error>> {
    let mut generation = state.generation;
    let mut champion_generation = state.champion_generation.unwrap_or(generation);
    let prior_elapsed = state.elapsed_seconds;
    let run_start = Instant::now();
    let run_seconds = config.hours * 60.0 * 60.0;
    let state_path = config.output.join("state.toml");
    let metrics_path = config.output.join("metrics.csv");
    discard_committed_self_play(&config.output, generation)?;
    let mut recovered_elapsed = 0.0;

    while prior_elapsed + recovered_elapsed + run_start.elapsed().as_secs_f64() < run_seconds {
        generation += 1;
        let generation_span = info_span!("generation", number = generation);
        let _generation_guard = generation_span.enter();
        let self_play = {
            let self_play_network = OthelloNetwork::load(
                checkpoint_path(&config.output, champion_generation),
                &device,
            )?;
            generate_or_recover_self_play(&config, &device, &self_play_network, generation)?
        };
        cleanup_device_memory(&device, "self-play")?;
        if self_play.recovered {
            recovered_elapsed += self_play.pending.elapsed_seconds;
        }
        let GenerationSelfPlay {
            training,
            validation,
            pending,
            ..
        } = self_play;
        let sample_count = pending.training_samples + pending.validation_samples;
        let training_sample_count = pending.training_samples;
        let validation_sample_count = pending.validation_samples;
        let evaluations = pending.evaluations;
        let unique_games = pending.unique_games;
        replay.push_back(training);
        trim_replay(&mut replay, config.train.replay_positions);
        let replay_samples = replay.iter().map(Vec::len).sum::<usize>();
        info!(
            positions = sample_count,
            unique_games,
            evaluations,
            elapsed = ?Duration::from_secs_f64(pending.elapsed_seconds),
            "self-play complete"
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
        let training_report = trainer.train_steps(&mut network, &replay_slices, training_steps)?;
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
        info!(
            steps = training_steps,
            replay_positions = replay_samples,
            elapsed = ?training_report.elapsed,
            learning_rate,
            policy_loss = training_report.policy_loss,
            policy_target_entropy = training_report.policy_target_entropy,
            policy_kl = training_report.policy_kl(),
            value_loss = training_report.value_loss,
            "training complete"
        );

        let checkpoint = checkpoint_path(&config.output, generation);
        let optimizer = optimizer_path(&config.output, generation);
        atomic_network_save(&network, &checkpoint)?;
        atomic_optimizer_save(&trainer, &optimizer)?;
        let evaluation =
            evaluate_generation(&config, &device, &network, generation, champion_generation)?;

        let evaluated = evaluation.is_some();
        let los = evaluation.map_or(f64::NAN, EvalResult::paired_los);
        let (current_wins, previous_wins, draws, pair_scores, score) = match evaluation {
            Some(result) => {
                let score = result.score();
                info!(
                    current_wins = result.candidate_wins,
                    previous_wins = result.baseline_wins,
                    draws = result.draws,
                    pair_scores = ?result.pair_scores,
                    score,
                    los,
                    "evaluation complete"
                );
                (
                    result.candidate_wins,
                    result.baseline_wins,
                    result.draws,
                    result.pair_scores,
                    score,
                )
            }
            None => (0, 0, 0, [0; 5], f64::NAN),
        };
        let promoted = evaluated
            && if let Some(threshold) = config.eval.promotion_los {
                los >= threshold
            } else {
                config
                    .eval
                    .promotion_threshold
                    .is_none_or(|threshold| score >= f64::from(threshold))
            };
        if !evaluated {
            info!(
                champion_generation,
                "evaluation skipped; continuing candidate training"
            );
        } else if promoted {
            champion_generation = generation;
            info!(champion_generation, "candidate promoted");
        } else {
            warn!(
                generation,
                champion_generation, score, los, "candidate rejected; learner continues training"
            );
        }
        let (validation_policy, validation_entropy, validation_kl, validation_value) =
            validation_loss
                .map(|loss| {
                    (
                        loss.policy_loss,
                        loss.policy_target_entropy,
                        loss.policy_kl(),
                        loss.value_loss,
                    )
                })
                .unwrap_or((f32::NAN, f32::NAN, f32::NAN, f32::NAN));
        append_metrics(
            &metrics_path,
            format!(
                "{generation},{sample_count},{training_sample_count},{validation_sample_count},{replay_samples},{:.6},{evaluations},{:.3},{learning_rate:.8},{},{:.6},{:.6},{:.6},{:.6},{:.6},{validation_policy:.6},{validation_entropy:.6},{validation_kl:.6},{validation_value:.6},{},{current_wins},{previous_wins},{draws},{},{},{},{},{},{score:.6},{los:.6},{promoted},{champion_generation},{}",
                pending.elapsed_seconds,
                evaluations as f64 / pending.elapsed_seconds.max(f64::MIN_POSITIVE),
                training_steps,
                training_report.elapsed.as_secs_f64(),
                training_report.policy_loss,
                training_report.policy_target_entropy,
                training_report.policy_kl(),
                training_report.value_loss,
                evaluated,
                pair_scores[0],
                pair_scores[1],
                pair_scores[2],
                pair_scores[3],
                pair_scores[4],
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
                champion_generation: Some(champion_generation),
                network_generation: Some(generation),
            },
        )?;
        let pending_path = config.output.join("pending-self-play.toml");
        if pending_path.exists() {
            fs::remove_file(pending_path)?;
        }
        let validation_path = validation_replay_path(&config.output, generation);
        if validation_path.exists() {
            fs::remove_file(validation_path)?;
        }
        cleanup_device_memory(&device, "generation")?;
    }
    info!(
        generation,
        elapsed = ?Duration::from_secs_f64(
            prior_elapsed + recovered_elapsed + run_start.elapsed().as_secs_f64()
        ),
        "experiment complete"
    );
    Ok(())
}

fn generate_or_recover_self_play(
    config: &Config,
    device: &Device,
    network: &OthelloNetwork,
    generation: usize,
) -> Result<GenerationSelfPlay, Box<dyn Error>> {
    let pending_path = config.output.join("pending-self-play.toml");
    let training_path = replay_path(&config.output, generation);
    let validation_path = validation_replay_path(&config.output, generation);
    if let Some((training, validation, pending)) =
        recover_self_play(&pending_path, &training_path, &validation_path, generation)?
    {
        info!(
            positions = training.len() + validation.len(),
            "recovered persisted self-play"
        );
        return Ok(GenerationSelfPlay {
            training,
            validation,
            pending,
            recovered: true,
        });
    }

    info!("starting self-play");
    let start = Instant::now();
    let self_play = run_actors(
        OthelloBurnEvaluator::from_network(device.clone(), network),
        config
            .self_play
            .actor_config(config.seed.wrapping_add(generation as u64)),
    )?;
    let (validation, training): (Vec<_>, Vec<_>) = self_play
        .samples
        .into_iter()
        .partition(|sample| is_validation_game(sample.game, config.train.validation_game_modulus));
    let pending = PendingSelfPlay {
        generation,
        training_samples: training.len(),
        validation_samples: validation.len(),
        elapsed_seconds: start.elapsed().as_secs_f64(),
        evaluations: self_play.evaluations,
        unique_games: self_play.unique_games,
    };
    atomic_replay_save(&training, &training_path)?;
    atomic_replay_save(&validation, &validation_path)?;
    atomic_toml_save(&pending_path, &pending)?;
    Ok(GenerationSelfPlay {
        training,
        validation,
        pending,
        recovered: false,
    })
}

fn is_validation_game(game: u64, modulus: u64) -> bool {
    // The trajectory hash has biased low bits, so mix it before taking a bucket.
    let mut hash = game;
    hash = (hash ^ (hash >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    hash = (hash ^ (hash >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    (hash ^ (hash >> 31)).is_multiple_of(modulus)
}

fn cleanup_device_memory(device: &Device, phase: &'static str) -> Result<(), Box<dyn Error>> {
    device.sync()?;
    device.memory_cleanup();
    if let Some(usage) = device.memory_pool_usage() {
        info!(
            phase,
            allocations = usage.number_allocs,
            bytes_in_use = usage.bytes_in_use,
            bytes_reserved = usage.bytes_reserved,
            "device memory cleanup complete"
        );
    }
    Ok(())
}

fn evaluate_generation(
    config: &Config,
    device: &Device,
    network: &OthelloNetwork,
    generation: usize,
    champion_generation: usize,
) -> Result<Option<EvalResult>, Box<dyn Error>> {
    if !generation.is_multiple_of(config.eval.interval) {
        return Ok(None);
    }
    let previous_network =
        OthelloNetwork::load(checkpoint_path(&config.output, champion_generation), device)?;
    let mut previous = OthelloBurnEvaluator::from_network(device.clone(), &previous_network);
    drop(previous_network);
    let mut current = OthelloBurnEvaluator::from_network(device.clone(), network);
    let start = Instant::now();
    let mut last_progress = start;
    let mut last_evaluations = 0;
    evaluate(
        &mut current,
        &mut previous,
        EvalConfig {
            games: config.eval.games,
            simulations: config.eval.simulations,
            batch_size: config.self_play.inference_batch_size,
            opening_plies: config.eval.opening_plies,
            seed: config.eval.seed,
        },
        |progress| {
            let interval = last_progress.elapsed();
            if interval >= Duration::from_secs(5) || progress.completed == progress.total {
                info!(
                    completed = progress.completed,
                    total = progress.total,
                    moves = progress.moves,
                    evaluations = progress.evaluations,
                    nps = (progress.evaluations - last_evaluations) as f64
                        / interval.as_secs_f64(),
                    games_per_second = progress.completed as f64 / start.elapsed().as_secs_f64(),
                    elapsed = ?start.elapsed(),
                    "generation evaluation progress"
                );
                last_progress = Instant::now();
                last_evaluations = progress.evaluations;
            }
        },
    )
    .map(Some)
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
        || !config.self_play.actor_config(config.seed).is_valid()
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
        || config.eval.promotion_threshold.is_some() && config.eval.promotion_los.is_some()
        || config
            .eval
            .promotion_threshold
            .is_some_and(|threshold| !(0.5..=1.0).contains(&threshold))
        || config
            .eval
            .promotion_los
            .is_some_and(|threshold| !(0.5..1.0).contains(&threshold))
        || config.model.validate().is_err()
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

fn acquire_run_lock(output: &Path) -> io::Result<File> {
    let path = suffixed_path(output, ".lock");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&path)?;
    FileExt::try_lock_exclusive(&file).map_err(|error| {
        if error.kind() == io::ErrorKind::WouldBlock {
            io::Error::new(
                error.kind(),
                format!("experiment output is already locked: {}", output.display()),
            )
        } else {
            error
        }
    })?;
    Ok(file)
}

fn prepare_staging(output: &Path) -> io::Result<PathBuf> {
    let staging = suffixed_path(output, ".initializing");
    if staging.exists() {
        fs::remove_dir_all(&staging)?;
    }
    fs::create_dir_all(staging.join("replay"))?;
    Ok(staging)
}

fn suffixed_path(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
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
        "generation,samples,training_samples,validation_samples,replay_samples,self_play_seconds,self_play_evaluations,self_play_evaluations_per_second,learning_rate,training_steps,training_seconds,policy_loss,policy_target_entropy,policy_kl,value_loss,validation_policy_loss,validation_policy_target_entropy,validation_policy_kl,validation_value_loss,evaluated,current_wins,previous_wins,draws,pair_0,pair_0_5,pair_1,pair_1_5,pair_2,score,los,promoted,champion_generation,checkpoint\n",
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

fn atomic_network_save(network: &OthelloNetwork, path: &Path) -> Result<(), Box<dyn Error>> {
    let temporary = temporary_path(path);
    network.save(&temporary)?;
    fs::rename(&temporary, path)?;
    Ok(())
}

fn atomic_optimizer_save(trainer: &TrainingSession, path: &Path) -> Result<(), Box<dyn Error>> {
    let temporary = temporary_path(path);
    trainer.save_optimizer(&temporary)?;
    fs::rename(&temporary, path)?;
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

fn resume_generations(state: RunState) -> io::Result<(usize, usize)> {
    let champion = state.champion_generation.unwrap_or(state.generation);
    let network = state.network_generation.unwrap_or(state.generation);
    if champion > state.generation || network > state.generation {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "run state references a future generation",
        ));
    }
    Ok((champion, network))
}

fn checkpoint_path(output: &Path, generation: usize) -> PathBuf {
    output.join(format!("generation-{generation:04}.burnpack"))
}

fn optimizer_path(output: &Path, generation: usize) -> PathBuf {
    output.join(format!("generation-{generation:04}-optimizer.burnpack"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{Game, Outcome};
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
promotion_threshold = 0.55
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
    fn run_state_resolves_legacy_and_explicit_generations() {
        let legacy: RunState = toml::from_str("generation = 4\nelapsed_seconds = 300.0\n").unwrap();
        assert_eq!(resume_generations(legacy).unwrap(), (4, 4));

        let champion_only: RunState =
            toml::from_str("generation = 4\nelapsed_seconds = 300.0\nchampion_generation = 2\n")
                .unwrap();
        assert_eq!(resume_generations(champion_only).unwrap(), (2, 4));

        let explicit: RunState = toml::from_str(
            "generation = 4\nelapsed_seconds = 300.0\nchampion_generation = 2\nnetwork_generation = 3\n",
        )
        .unwrap();
        assert_eq!(resume_generations(explicit).unwrap(), (2, 3));

        let future: RunState = toml::from_str(
            "generation = 4\nelapsed_seconds = 300.0\nchampion_generation = 5\nnetwork_generation = 4\n",
        )
        .unwrap();
        assert!(resume_generations(future).is_err());
    }

    #[test]
    fn validation_split_mixes_trajectory_hashes() {
        let validation_games = (0..10_000)
            .filter(|game| is_validation_game(*game, 10))
            .count();

        assert!((900..=1_100).contains(&validation_games));
        assert_eq!(is_validation_game(42, 10), is_validation_game(42, 10));
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
    fn run_lock_rejects_a_second_owner() {
        let output = std::env::temp_dir().join(format!("etive-lock-{}", std::process::id()));
        let lock_path = suffixed_path(&output, ".lock");
        if lock_path.exists() {
            fs::remove_file(&lock_path).unwrap();
        }

        let first = acquire_run_lock(&output).unwrap();
        let error = match acquire_run_lock(&output) {
            Ok(_) => panic!("second run unexpectedly acquired the output lock"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("already locked"));

        drop(first);
        drop(acquire_run_lock(&output).unwrap());
        fs::remove_file(lock_path).unwrap();
    }

    #[test]
    fn staging_replaces_interrupted_initialization() {
        let output = std::env::temp_dir().join(format!("etive-staging-{}", std::process::id()));
        let staging = suffixed_path(&output, ".initializing");
        if output.exists() {
            fs::remove_dir_all(&output).unwrap();
        }
        if staging.exists() {
            fs::remove_dir_all(&staging).unwrap();
        }
        fs::create_dir(&staging).unwrap();
        fs::write(staging.join("partial"), []).unwrap();

        let staging = prepare_staging(&output).unwrap();
        assert!(!staging.join("partial").exists());
        assert!(!output.exists());
        assert!(staging.join("replay").is_dir());

        fs::remove_dir_all(staging).unwrap();
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
        let mut policy = [0.0; Board::ACTION_COUNT];
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
