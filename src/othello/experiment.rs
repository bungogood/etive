//! TOML-configured, resumable Othello self-play experiments.

use std::collections::VecDeque;
use std::error::Error;
use std::fs::{self, File};
use std::path::Path;
use std::time::Instant;

use burn::tensor::Device;
use tracing::{info, info_span, warn};

use crate::metrics::PolicyValueMetrics;
use crate::self_play;

use super::evaluation::{EvalConfig, EvalResult, evaluate};
use super::replay::{
    SelfPlaySample, atomic_replay_save, load_replay, replay_path, trim_replay,
    validation_replay_path,
};
use super::training::{TrainingSession, evaluate_loss};
use super::{Board, OthelloBurnEvaluator, OthelloNetwork};

mod config;
mod storage;

pub use config::{SelfPlayBenchmarkConfig, load_self_play_benchmark_config};

use config::{Config, resolve_paths, validate};
use storage::{
    GenerationMetrics, PendingSelfPlay, RUN_MARKER, RecoveredSelfPlay, RunState, acquire_run_lock,
    append_metrics, atomic_network_save, atomic_optimizer_save, atomic_toml_save, checkpoint_path,
    clean_output, discard_committed_self_play, optimizer_path, prepare_staging, recover_self_play,
    validate_metrics, validate_run_state, verify_run_marker,
};

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
            policy_cross_entropy = %format_args!("{:.4}", training_report.metrics.policy_cross_entropy),
            policy_target_entropy = %format_args!("{:.4}", training_report.metrics.policy_target_entropy),
            policy_kl = %format_args!("{:.4}", training_report.metrics.policy_kl()),
            value_mse = %format_args!("{:.4}", training_report.metrics.value_mse),
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
        let validation_metrics = validation_loss
            .unwrap_or_else(|| PolicyValueMetrics::new(f32::NAN, f32::NAN, f32::NAN));
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
                policy_loss: rounded(f64::from(training_report.metrics.policy_cross_entropy), 5),
                policy_target_entropy: rounded(
                    f64::from(training_report.metrics.policy_target_entropy),
                    5,
                ),
                policy_kl: rounded(f64::from(training_report.metrics.policy_kl()), 5),
                value_loss: rounded(f64::from(training_report.metrics.value_mse), 5),
                validation_policy_loss: rounded(
                    f64::from(validation_metrics.policy_cross_entropy),
                    5,
                ),
                validation_policy_target_entropy: rounded(
                    f64::from(validation_metrics.policy_target_entropy),
                    5,
                ),
                validation_policy_kl: rounded(f64::from(validation_metrics.policy_kl()), 5),
                validation_value_loss: rounded(f64::from(validation_metrics.value_mse), 5),
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
        discard_committed_self_play(&config.output, generation)?;
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
    if let Some(recovered) =
        recover_self_play(&pending_path, &training_path, &validation_path, generation)?
    {
        let RecoveredSelfPlay {
            training,
            validation,
            pending,
        } = recovered;
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
    let self_play = self_play::run::<Board, _>(
        OthelloBurnEvaluator::from_network(device.clone(), network),
        config
            .self_play
            .config(config.seed.wrapping_add(generation as u64)),
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
    )
    .map(Some)
}

fn rounded(value: f64, decimal_places: i32) -> f64 {
    let scale = 10_f64.powi(decimal_places);
    (value * scale).round() / scale
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{Game, Outcome};
    use crate::othello::Board;

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
}
