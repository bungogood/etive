//! TOML-configured, resumable Othello self-play experiments.

use std::collections::VecDeque;
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use burn::tensor::Device;
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use tracing::{info, info_span, warn};

use super::OthelloBurnEvaluator;
use super::OthelloNetwork;
use super::actors::run as run_actors;
use super::evaluation::{EvalConfig, EvalResult, evaluate};
use super::replay::{
    SelfPlaySample, atomic_replay_save, load_replay, read_replay, replay_path, trim_replay,
    validation_replay_path,
};
use super::training::{TrainingSession, evaluate_loss};

mod config;

pub use config::{SelfPlayBenchmarkConfig, load_self_play_benchmark_config};

use config::{Config, resolve_paths, validate};

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RunState {
    generation: usize,
    champion_generation: usize,
    elapsed_seconds: f64,
}

const RUN_MARKER: &str = "etive-run-v1\n";

#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct GenerationMetrics {
    generation: usize,
    samples: usize,
    training_samples: usize,
    validation_samples: usize,
    replay_samples: usize,
    self_play_seconds: f64,
    self_play_evaluations: u64,
    self_play_evaluations_per_second: f64,
    learning_rate: f64,
    training_steps: usize,
    training_seconds: f64,
    policy_loss: f64,
    policy_target_entropy: f64,
    policy_kl: f64,
    value_loss: f64,
    validation_policy_loss: f64,
    validation_policy_target_entropy: f64,
    validation_policy_kl: f64,
    validation_value_loss: f64,
    evaluated: bool,
    candidate_wins: usize,
    baseline_wins: usize,
    draws: usize,
    pair_0: usize,
    pair_0_5: usize,
    pair_1: usize,
    pair_1_5: usize,
    pair_2: usize,
    score: f64,
    los: f64,
    promoted: bool,
    baseline_generation: usize,
    champion_generation: usize,
    checkpoint: PathBuf,
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
            let checkpoint_config = OthelloNetwork::checkpoint_config(path)
                .ok_or("checkpoint is missing valid model metadata")?;
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
    fs::write(staging.join(".etive-run"), RUN_MARKER)?;
    fs::write(staging.join("model.toml"), toml::to_string(&config.model)?)?;
    File::create(staging.join("metrics.csv"))?;
    let checkpoint = checkpoint_path(&staging, 0);
    let optimizer = optimizer_path(&staging, 0);
    atomic_network_save(&network, &checkpoint)?;
    atomic_optimizer_save(&trainer, &optimizer)?;
    let state = RunState {
        generation: 0,
        champion_generation: 0,
        elapsed_seconds: 0.0,
    };
    atomic_toml_save(&staging.join("state.toml"), &state)?;
    fs::rename(staging, &config.output)?;

    run_loop(config, device, network, trainer, VecDeque::new(), state)
}

fn resume_run(config: Config, source: String, device: Device) -> Result<(), Box<dyn Error>> {
    verify_run_marker(&config.output)?;
    let stored_config = fs::read_to_string(config.output.join("config.toml"))?;
    let state: RunState = toml::from_str(&fs::read_to_string(config.output.join("state.toml"))?)?;
    if stored_config != source {
        return Err("experiment configuration changed; use --clean to start a new run".into());
    }
    validate_run_state(state)?;
    validate_metrics(&config.output.join("metrics.csv"), state.generation)?;
    let network_path = checkpoint_path(&config.output, state.generation);
    let stored_model = OthelloNetwork::checkpoint_config(&network_path)
        .ok_or("stored checkpoint is missing valid model metadata")?;
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
    trainer.load_optimizer(optimizer_path(&config.output, state.generation))?;
    let replay = load_replay(
        &config.output,
        state.generation,
        config.train.replay_positions,
    )?;
    info!(
        generation = state.generation,
        champion_generation = state.champion_generation,
        replay_positions = replay.iter().map(Vec::len).sum::<usize>(),
        elapsed = %format_args!("{:.1}s", state.elapsed_seconds),
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
    let mut champion_generation = state.champion_generation;
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
            elapsed = %format_args!("{:.1}s", pending.elapsed_seconds),
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
            elapsed = %format_args!("{:.1}s", training_report.elapsed.as_secs_f64()),
            learning_rate = %format_args!("{learning_rate:.2e}"),
            policy_loss = %format_args!("{:.4}", training_report.policy_loss),
            policy_target_entropy = %format_args!("{:.4}", training_report.policy_target_entropy),
            policy_kl = %format_args!("{:.4}", training_report.policy_kl()),
            value_loss = %format_args!("{:.4}", training_report.value_loss),
            "training complete"
        );

        let checkpoint = checkpoint_path(&config.output, generation);
        let optimizer = optimizer_path(&config.output, generation);
        atomic_network_save(&network, &checkpoint)?;
        atomic_optimizer_save(&trainer, &optimizer)?;
        let baseline_generation = champion_generation;
        let evaluation =
            evaluate_generation(&config, &device, &network, generation, baseline_generation)?;

        let evaluated = evaluation.is_some();
        let los = evaluation.map_or(f64::NAN, EvalResult::paired_los);
        let (candidate_wins, baseline_wins, draws, pair_scores, score) = match evaluation {
            Some(result) => {
                let score = result.score();
                info!(
                    candidate_wins = result.candidate_wins,
                    baseline_wins = result.baseline_wins,
                    draws = result.draws,
                    pair_scores = ?result.pair_scores,
                    score = %format_args!("{:.1}%", score * 100.0),
                    los = %format_args!("{:.1}%", los * 100.0),
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
        let promoted = evaluated && los >= config.eval.promotion_los;
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
                champion_generation,
                score = %format_args!("{:.1}%", score * 100.0),
                los = %format_args!("{:.1}%", los * 100.0),
                "candidate rejected; learner continues training"
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
            &GenerationMetrics {
                generation,
                samples: sample_count,
                training_samples: training_sample_count,
                validation_samples: validation_sample_count,
                replay_samples,
                self_play_seconds: rounded(pending.elapsed_seconds, 3),
                self_play_evaluations: evaluations,
                self_play_evaluations_per_second: rounded(
                    evaluations as f64 / pending.elapsed_seconds.max(f64::MIN_POSITIVE),
                    1,
                ),
                learning_rate: rounded(learning_rate, 8),
                training_steps,
                training_seconds: rounded(training_report.elapsed.as_secs_f64(), 3),
                policy_loss: rounded(f64::from(training_report.policy_loss), 5),
                policy_target_entropy: rounded(f64::from(training_report.policy_target_entropy), 5),
                policy_kl: rounded(f64::from(training_report.policy_kl()), 5),
                value_loss: rounded(f64::from(training_report.value_loss), 5),
                validation_policy_loss: rounded(f64::from(validation_policy), 5),
                validation_policy_target_entropy: rounded(f64::from(validation_entropy), 5),
                validation_policy_kl: rounded(f64::from(validation_kl), 5),
                validation_value_loss: rounded(f64::from(validation_value), 5),
                evaluated,
                candidate_wins,
                baseline_wins,
                draws,
                pair_0: pair_scores[0],
                pair_0_5: pair_scores[1],
                pair_1: pair_scores[2],
                pair_1_5: pair_scores[3],
                pair_2: pair_scores[4],
                score: rounded(score, 5),
                los: rounded(los, 5),
                promoted,
                baseline_generation,
                champion_generation,
                checkpoint,
            },
        )?;
        atomic_toml_save(
            &state_path,
            &RunState {
                generation,
                champion_generation,
                elapsed_seconds: prior_elapsed
                    + recovered_elapsed
                    + run_start.elapsed().as_secs_f64(),
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
        elapsed = %format_args!(
            "{:.1}s",
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
    let (training, validation) =
        split_training_validation(self_play.samples, config.train.validation_game_modulus);
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

fn split_training_validation(
    samples: Vec<SelfPlaySample>,
    modulus: u64,
) -> (Vec<SelfPlaySample>, Vec<SelfPlaySample>) {
    let (mut validation, mut training): (Vec<_>, Vec<_>) = samples
        .into_iter()
        .partition(|sample| is_validation_game(sample.game, modulus));
    if training.is_empty() && !validation.is_empty() {
        let training_game = validation[0].game;
        let (remaining, fallback): (Vec<_>, Vec<_>) = validation
            .into_iter()
            .partition(|sample| sample.game != training_game);
        validation = remaining;
        training = fallback;
    }
    (training, validation)
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
                    evaluations_per_second = %format_args!(
                        "{:.0}",
                        (progress.evaluations - last_evaluations) as f64 / interval.as_secs_f64()
                    ),
                    games_per_second = %format_args!(
                        "{:.1}",
                        progress.completed as f64 / start.elapsed().as_secs_f64()
                    ),
                    elapsed = %format_args!("{:.1}s", start.elapsed().as_secs_f64()),
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
    if verify_run_marker(output).is_err() {
        return Err(format!(
            "refusing to clean unrecognized output directory: {}",
            output.display()
        )
        .into());
    }
    fs::remove_dir_all(output)?;
    Ok(())
}

fn rounded(value: f64, decimal_places: i32) -> f64 {
    let scale = 10_f64.powi(decimal_places);
    (value * scale).round() / scale
}

fn append_metrics(path: &Path, metrics: &GenerationMetrics) -> Result<(), Box<dyn Error>> {
    let file = OpenOptions::new().append(true).open(path)?;
    let write_header = file.metadata()?.len() == 0;
    let mut writer = csv::WriterBuilder::new()
        .has_headers(write_header)
        .from_writer(file);
    writer.serialize(metrics)?;
    writer.flush()?;
    Ok(())
}

fn validate_metrics(path: &Path, generation: usize) -> Result<(), Box<dyn Error>> {
    if fs::metadata(path)?.len() == 0 {
        return if generation == 0 {
            Ok(())
        } else {
            Err("metrics are empty for a committed run".into())
        };
    }

    let mut reader = csv::Reader::from_path(path)?;
    if reader.headers()?.clone() != metrics_headers()? {
        return Err("metrics schema does not match the current Etive format".into());
    }
    let mut rows = 0;
    for (index, row) in reader.deserialize::<GenerationMetrics>().enumerate() {
        let row = row?;
        if row.generation != index + 1 {
            return Err("metrics generations are not contiguous".into());
        }
        rows += 1;
    }
    if rows != generation {
        return Err("metrics do not contain exactly one row per committed generation".into());
    }
    Ok(())
}

fn metrics_headers() -> Result<csv::StringRecord, Box<dyn Error>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.serialize(GenerationMetrics::default())?;
    let bytes = writer.into_inner()?;
    let mut reader = csv::Reader::from_reader(bytes.as_slice());
    Ok(reader.headers()?.clone())
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

fn verify_run_marker(output: &Path) -> io::Result<()> {
    if fs::read_to_string(output.join(".etive-run"))? != RUN_MARKER {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid Etive run marker",
        ));
    }
    Ok(())
}

fn validate_run_state(state: RunState) -> io::Result<()> {
    if state.champion_generation > state.generation
        || !state.elapsed_seconds.is_finite()
        || state.elapsed_seconds < 0.0
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid run state",
        ));
    }
    Ok(())
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

    #[test]
    fn run_state_requires_current_explicit_fields() {
        let state: RunState =
            toml::from_str("generation = 4\nchampion_generation = 2\nelapsed_seconds = 300.0\n")
                .unwrap();
        assert!(validate_run_state(state).is_ok());

        assert!(toml::from_str::<RunState>("generation = 4\nelapsed_seconds = 300.0\n").is_err());
        assert!(
            toml::from_str::<RunState>(
                "generation = 4\nchampion_generation = 2\nnetwork_generation = 4\nelapsed_seconds = 300.0\n"
            )
            .is_err()
        );

        let future: RunState =
            toml::from_str("generation = 4\nchampion_generation = 5\nelapsed_seconds = 300.0\n")
                .unwrap();
        assert!(validate_run_state(future).is_err());
        assert!(
            validate_run_state(RunState {
                elapsed_seconds: f64::NAN,
                ..state
            })
            .is_err()
        );
        assert!(
            validate_run_state(RunState {
                elapsed_seconds: -1.0,
                ..state
            })
            .is_err()
        );
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
    fn validation_split_always_keeps_training_data() {
        let game = (0..).find(|game| is_validation_game(*game, 2)).unwrap();
        let mut policy = [0.0; Board::ACTION_COUNT];
        policy[19] = 1.0;
        let sample = SelfPlaySample {
            position: Board::default(),
            policy,
            outcome: Outcome::Draw,
            game,
        };

        let (training, validation) = split_training_validation(vec![sample], 2);

        assert_eq!(training.len(), 1);
        assert!(validation.is_empty());
    }

    #[test]
    fn metrics_are_typed_and_match_committed_generations() {
        let path = std::env::temp_dir().join(format!("etive-metrics-{}.csv", std::process::id()));
        if path.exists() {
            fs::remove_file(&path).unwrap();
        }
        File::create(&path).unwrap();
        append_metrics(
            &path,
            &GenerationMetrics {
                generation: 1,
                checkpoint: "generation-0001.burnpack".into(),
                ..GenerationMetrics::default()
            },
        )
        .unwrap();

        validate_metrics(&path, 1).unwrap();
        assert!(validate_metrics(&path, 0).is_err());
        let header = fs::read_to_string(&path).unwrap();
        assert!(header.starts_with("generation,samples,training_samples"));
        assert!(header.contains("candidate_wins,baseline_wins"));

        fs::remove_file(path).unwrap();
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

        fs::write(output.join(".etive-run"), RUN_MARKER).unwrap();
        fs::write(
            output.join("state.toml"),
            "generation = 0\nchampion_generation = 0\nelapsed_seconds = 0.0\n",
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
