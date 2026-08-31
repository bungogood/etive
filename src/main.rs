use std::path::PathBuf;
use std::time::Instant;

use burn::tensor::Device;
use clap::{Parser, Subcommand};
use etive::metrics::write_csv;
use etive::othello::evaluation::{EvalConfig, evaluate};
use etive::othello::experiment;
use etive::othello::{
    Board, FrozenTrainingConfig, OthelloBurnEvaluator, OthelloNetwork, diagnose_replay, perft,
    train_frozen,
};
use etive::self_play;
use tracing_subscriber::EnvFilter;

mod gtp;

#[derive(Parser)]
#[command(version, about, arg_required_else_help = true)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compare two Othello checkpoints with fixed-search games.
    Eval {
        /// Checkpoint for the previous network.
        previous: PathBuf,
        /// Checkpoint for the contender network.
        contender: PathBuf,
        /// Number of color-balanced games.
        #[arg(long, default_value_t = 500)]
        games: usize,
        /// Simulations performed before each move.
        #[arg(long, default_value_t = 128)]
        simulations: u32,
        /// Maximum positions in one network invocation.
        #[arg(long, default_value_t = 4096)]
        batch_size: usize,
        /// Seeded random plies applied before measured play begins.
        #[arg(long, default_value_t = 8)]
        opening_plies: usize,
        /// Reproducible opening-suite seed.
        #[arg(long, default_value_t = 7)]
        seed: u64,
    },
    /// Run as a Go Text Protocol v2 Othello engine.
    Gtp {
        /// Trained checkpoint used for MCTS moves; omitted for rules-only mode.
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        /// Simulations performed before each generated move.
        #[arg(long, default_value_t = 128)]
        simulations: u32,
        /// Maximum leaves evaluated together during parallel search.
        #[arg(long, default_value_t = 128)]
        batch_size: usize,
    },
    /// Start or resume a TOML-configured training run.
    Train {
        /// Training TOML file.
        config: PathBuf,
        /// Delete an existing Etive run before starting again.
        #[arg(long)]
        clean: bool,
    },
    /// Measure the production self-play pipeline.
    Bench {
        /// Experiment TOML supplying model and self-play settings.
        config: PathBuf,
    },
    /// Compare a checkpoint's predictions with validated replay targets.
    DiagnoseReplay {
        /// Checkpoint to evaluate.
        checkpoint: PathBuf,
        /// Replay shards, flattened in command-line order.
        #[arg(required = true)]
        replay: Vec<PathBuf>,
        /// Number of replay rows to sample; values above availability are clamped.
        #[arg(long)]
        rows: Option<usize>,
        /// Reproducible sampling seed.
        #[arg(long, default_value_t = 7)]
        seed: u64,
        /// Maximum positions in one network invocation.
        #[arg(long, default_value_t = 1024)]
        batch_size: usize,
        /// Evaluate with FP32 instead of production FP16 inference.
        #[arg(long)]
        float32: bool,
    },
    /// Train a checkpoint for a fixed number of steps over frozen replay shards.
    TrainFrozen {
        /// Model checkpoint to restore.
        checkpoint: PathBuf,
        /// Optimizer checkpoint to restore.
        optimizer: PathBuf,
        /// Validated replay shard; repeat to supply multiple shards in order.
        #[arg(long, required = true)]
        replay: Vec<PathBuf>,
        /// Exact number of optimizer steps to perform.
        #[arg(long, required = true, value_parser = parse_positive_usize)]
        steps: usize,
        /// Samples drawn per optimizer step.
        #[arg(long, default_value_t = 128)]
        batch_size: usize,
        /// AdamW learning rate.
        #[arg(long, default_value_t = 0.001)]
        learning_rate: f64,
        /// AdamW weight decay.
        #[arg(long, default_value_t = 0.0001)]
        weight_decay: f32,
        /// Reproducible sampling and augmentation seed.
        #[arg(long, default_value_t = 7)]
        seed: u64,
        /// New directory for the trained artifacts and metrics.
        #[arg(long, required = true)]
        output: PathBuf,
    },
    /// Count opening-position leaves.
    Perft {
        /// Search depth in plies.
        #[arg(default_value_t = 10)]
        depth: u8,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_target(false)
        .compact()
        .init();

    match Cli::parse().command {
        Command::Eval {
            previous,
            contender,
            games,
            simulations,
            batch_size,
            opening_plies,
            seed,
        } => {
            let device = burn_device();
            let previous = OthelloNetwork::load(previous, &device)?;
            let contender = OthelloNetwork::load(contender, &device)?;
            let mut previous = OthelloBurnEvaluator::from_network(device.clone(), &previous);
            let mut contender = OthelloBurnEvaluator::from_network(device, &contender);
            let start = Instant::now();
            let result = evaluate(
                &mut contender,
                &mut previous,
                EvalConfig {
                    games,
                    simulations,
                    batch_size,
                    opening_plies,
                    seed,
                },
            )?;
            println!(
                "eval: contender {} wins, baseline {} wins, {} draws, score {:.1}%, paired LOS {:.1}% in {:.1?}",
                result.candidate_wins,
                result.baseline_wins,
                result.draws,
                result.score() * 100.0,
                result.paired_los() * 100.0,
                start.elapsed()
            );
        }
        Command::Perft { depth } => {
            let board = Board::default();
            let start = Instant::now();
            let nodes = perft(&board, depth);
            let elapsed = start.elapsed();
            let nps = nodes as f64 / elapsed.as_secs_f64();
            println!("{nodes} nodes in {elapsed:.3?} ({nps:.0} nps)");
        }
        Command::Gtp {
            checkpoint,
            simulations,
            batch_size,
        } => {
            let stdin = std::io::stdin();
            let stdout = std::io::stdout();
            if let Some(path) = checkpoint {
                if simulations < 2 || batch_size == 0 {
                    return Err("simulations must be at least two and batch size positive".into());
                }
                let device = burn_device();
                let network = OthelloNetwork::load(path, &device)?;
                let evaluator = OthelloBurnEvaluator::from_network(device, &network);
                gtp::run_with_evaluator(
                    stdin.lock(),
                    stdout.lock(),
                    evaluator,
                    simulations,
                    batch_size,
                )?;
            } else {
                gtp::run(stdin.lock(), stdout.lock())?;
            }
        }
        Command::Train { config, clean } => {
            experiment::run(config, burn_device(), clean)?;
        }
        Command::Bench { config } => benchmark_self_play(&burn_device(), &config)?,
        Command::DiagnoseReplay {
            checkpoint,
            replay,
            rows,
            seed,
            batch_size,
            float32,
        } => {
            let report = diagnose_replay(
                checkpoint,
                &replay,
                rows,
                seed,
                batch_size,
                float32,
                burn_device(),
            )?;
            write_csv(std::io::stdout().lock(), &report)?;
        }
        Command::TrainFrozen {
            checkpoint,
            optimizer,
            replay,
            steps,
            batch_size,
            learning_rate,
            weight_decay,
            seed,
            output,
        } => {
            let report = train_frozen(
                checkpoint,
                optimizer,
                &replay,
                FrozenTrainingConfig {
                    steps,
                    batch_size,
                    learning_rate,
                    weight_decay,
                    seed,
                    output,
                },
                burn_device(),
            )?;
            write_csv(std::io::stdout().lock(), &report)?;
        }
    }
    Ok(())
}

fn parse_positive_usize(value: &str) -> Result<usize, String> {
    match value.parse::<usize>() {
        Ok(value) if value > 0 => Ok(value),
        _ => Err("value must be a positive integer".to_owned()),
    }
}

fn benchmark_self_play(
    device: &Device,
    config_path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = experiment::load_self_play_benchmark_config(config_path)?;
    let network = match config.checkpoint {
        Some(path) => OthelloNetwork::load_with_config(&path, device, config.model)?,
        None => OthelloNetwork::new_with_config(device, config.self_play.seed, config.model),
    };
    println!(
        "model: channels={} residual_blocks={} norm_groups={}",
        config.model.channels, config.model.residual_blocks, config.model.norm_groups
    );
    println!(
        "self-play: games={} workers={} max_batch={} simulations={}",
        config.self_play.games,
        config.self_play.workers,
        config.self_play.inference_batch_size,
        config.self_play.simulations,
    );
    let start = Instant::now();
    let result = self_play::run::<Board, _>(
        OthelloBurnEvaluator::from_network(device.clone(), &network),
        config.self_play,
    )?;
    let elapsed = start.elapsed();
    let seconds = elapsed.as_secs_f64();
    println!(
        "self-play: elapsed={elapsed:.3?} games/s={:.2} evaluations/s={:.0} evaluations={} inference_batches={} average_batch={:.2} positions={} unique_games={}",
        config.self_play.games as f64 / seconds,
        result.evaluations as f64 / seconds,
        result.evaluations,
        result.inference_batches,
        result.evaluations as f64 / result.inference_batches as f64,
        result.samples.len(),
        result.unique_games,
    );
    Ok(())
}

fn burn_device() -> Device {
    #[cfg(feature = "cuda")]
    return Device::cuda(0);

    #[cfg(all(not(feature = "cuda"), feature = "metal"))]
    return Device::metal(burn::tensor::DeviceKind::default());

    #[cfg(all(not(feature = "cuda"), not(feature = "metal"), feature = "flex"))]
    return Device::flex();

    #[cfg(all(
        not(feature = "cuda"),
        not(feature = "metal"),
        not(feature = "flex"),
        feature = "cpu"
    ))]
    Device::cpu()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_frozen_cli_preserves_replay_order_and_requires_positive_steps() {
        let cli = Cli::try_parse_from([
            "etive",
            "train-frozen",
            "model.burnpack",
            "optimizer.burnpack",
            "--replay",
            "first.bin",
            "--replay",
            "second.bin",
            "--steps",
            "2",
            "--output",
            "trained",
        ])
        .unwrap();
        let Command::TrainFrozen { replay, steps, .. } = cli.command else {
            panic!("expected train-frozen command");
        };
        assert_eq!(
            replay,
            [PathBuf::from("first.bin"), PathBuf::from("second.bin")]
        );
        assert_eq!(steps, 2);

        assert!(
            Cli::try_parse_from([
                "etive",
                "train-frozen",
                "model.burnpack",
                "optimizer.burnpack",
                "--replay",
                "replay.bin",
                "--steps",
                "0",
                "--output",
                "trained",
            ])
            .is_err()
        );
    }
}
