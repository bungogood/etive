use std::path::PathBuf;
use std::time::{Duration, Instant};

use candle_core::Device;
use clap::{Parser, Subcommand};
use etive::evaluator::OthelloCandleEvaluator;
use etive::model::OthelloNetwork;
use etive::othello::experiment;
use etive::othello::training::{ArenaConfig, arena_with_progress};
use etive::othello::{Board, perft};

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
    /// Count opening-position leaves.
    Perft {
        /// Search depth in plies.
        #[arg(default_value_t = 10)]
        depth: u8,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
            let device = candle_device()?;
            let previous = OthelloNetwork::load(previous, &device)?;
            let contender = OthelloNetwork::load(contender, &device)?;
            let mut previous = OthelloCandleEvaluator::from_network(device.clone(), previous);
            let mut contender = OthelloCandleEvaluator::from_network(device, contender);
            let start = Instant::now();
            let mut last_progress = Instant::now();
            let result = arena_with_progress(
                &mut contender,
                &mut previous,
                ArenaConfig {
                    games,
                    simulations,
                    batch_size,
                    opening_plies,
                    seed,
                },
                |progress| {
                    let elapsed = start.elapsed();
                    if last_progress.elapsed() >= Duration::from_secs(5)
                        || progress.completed == progress.total
                    {
                        eprintln!(
                            "progress: {}/{} games, {} moves, {:.2} games/s, {:.0} evaluations/s",
                            progress.completed,
                            progress.total,
                            progress.moves,
                            progress.completed as f64 / elapsed.as_secs_f64(),
                            progress.evaluations as f64 / elapsed.as_secs_f64()
                        );
                        last_progress = Instant::now();
                    }
                },
            )?;
            println!(
                "eval: contender {} wins, previous {} wins, {} draws in {:.3?}",
                result.trained_wins,
                result.initial_wins,
                result.draws,
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
                let device = candle_device()?;
                let network = OthelloNetwork::load(path, &device)?;
                let evaluator = OthelloCandleEvaluator::from_network(device, network);
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
            experiment::run(config, candle_device()?, clean)?;
        }
    }
    Ok(())
}

fn candle_device() -> candle_core::Result<Device> {
    #[cfg(feature = "cuda")]
    return Device::new_cuda(0);

    #[cfg(all(not(feature = "cuda"), feature = "cudnn"))]
    return Device::new_cuda(0);

    #[cfg(all(not(feature = "cuda"), not(feature = "cudnn"), feature = "metal"))]
    return Device::new_metal(0);

    #[cfg(not(any(feature = "cuda", feature = "cudnn", feature = "metal")))]
    Ok(Device::Cpu)
}
