use std::path::PathBuf;
use std::time::Instant;

use burn::tensor::Device;
use clap::{Parser, Subcommand};
use etive::othello::evaluation::{EvalConfig, evaluate};
use etive::othello::experiment;
use etive::othello::{Board, OthelloBurnEvaluator, OthelloNetwork, perft};
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
    /// Count opening-position leaves.
    Perft {
        /// Search depth in plies.
        #[arg(default_value_t = 10)]
        depth: u8,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
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
                gtp::run_with_evaluator(stdin.lock(), stdout, evaluator, simulations, batch_size)?;
            } else {
                gtp::run(stdin.lock(), stdout)?;
            }
        }
        Command::Train { config, clean } => {
            experiment::run(config, burn_device(), clean)?;
        }
        Command::Bench { config } => benchmark_self_play(&burn_device(), &config)?,
    }
    Ok(())
}

fn benchmark_self_play(
    device: &Device,
    config_path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = experiment::load_self_play_benchmark_config(config_path)?;
    let network = match config.checkpoint {
        Some(path) => OthelloNetwork::load_with_config(&path, device, config.model)?,
        None => OthelloNetwork::new_with_config(device, config.seed, config.model),
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
        config.seed,
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
