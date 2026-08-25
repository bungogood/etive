use std::path::PathBuf;
use std::time::{Duration, Instant};

use candle_core::{Device, Tensor};
use clap::{Parser, Subcommand};
use etive::evaluator::OthelloCandleEvaluator;
use etive::model::OthelloNetwork;
use etive::othello::actors::{ActorConfig, run as run_actors};
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
    Arena {
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
    /// Verify Candle on the selected device.
    Candle,
    /// Run as a Go Text Protocol v2 Othello engine.
    Gtp {
        /// Trained checkpoint used for MCTS moves; omitted for rules-only mode.
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        /// Simulations performed before each generated move.
        #[arg(long, default_value_t = 128)]
        simulations: u32,
    },
    /// Play Othello games with random Candle evaluation and MCTS.
    Mcts {
        /// Simulations performed before each move.
        #[arg(long, default_value_t = 128)]
        simulations: u32,
        /// Reproducible random network seed.
        #[arg(long, default_value_t = 7)]
        seed: u64,
        /// Number of independent games searched together.
        #[arg(long, default_value_t = 1)]
        games: usize,
        /// Maximum positions in one Candle invocation.
        #[arg(long, default_value_t = 64)]
        batch_size: usize,
        /// Persistent game-owning actor threads.
        #[arg(long)]
        workers: Option<usize>,
        /// Dirichlet concentration mixed into each self-play root.
        #[arg(long, default_value_t = 0.3)]
        dirichlet_alpha: f64,
        /// Fraction of each root prior replaced by Dirichlet noise.
        #[arg(long, default_value_t = 0.25)]
        dirichlet_fraction: f32,
        /// Opening plies sampled from visit counts before greedy play.
        #[arg(long, default_value_t = 20)]
        temperature_moves: usize,
    },
    /// Run or resume a TOML-configured training experiment.
    Learn {
        /// Experiment TOML file.
        config: PathBuf,
        /// Resume the run recorded in the configured output directory.
        #[arg(long)]
        resume: bool,
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
        Command::Arena {
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
                "arena: contender {} wins, previous {} wins, {} draws in {:.3?}",
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
        Command::Candle => {
            let device = candle_device()?;
            let tensor = Tensor::new(&[1_f32, 2.0, 3.0, 4.0], &device)?;
            let result = tensor.sqr()?.sum_all()?.to_scalar::<f32>()?;
            println!("Candle smoke test passed on {device:?}: {result}");
        }
        Command::Gtp {
            checkpoint,
            simulations,
        } => {
            let stdin = std::io::stdin();
            let stdout = std::io::stdout();
            if let Some(path) = checkpoint {
                if simulations < 2 {
                    return Err("simulations must be at least two".into());
                }
                let device = candle_device()?;
                let network = OthelloNetwork::load(path, &device)?;
                let evaluator = OthelloCandleEvaluator::from_network(device, network);
                gtp::run_with_evaluator(stdin.lock(), stdout.lock(), evaluator, simulations)?;
            } else {
                gtp::run(stdin.lock(), stdout.lock())?;
            }
        }
        Command::Mcts {
            simulations,
            seed,
            games,
            batch_size,
            workers,
            dirichlet_alpha,
            dirichlet_fraction,
            temperature_moves,
        } => {
            if simulations < 2 || games == 0 || batch_size == 0 {
                return Err(
                    "games and batch size must be positive; simulations must be at least two"
                        .into(),
                );
            }
            let device = candle_device()?;
            let evaluator = OthelloCandleEvaluator::new(device, seed)?;
            let workers = workers.unwrap_or_else(default_actor_workers);
            let start = Instant::now();
            let (result, _) = run_actors(
                evaluator,
                ActorConfig {
                    games,
                    simulations,
                    workers,
                    inference_batch_size: batch_size,
                    seed,
                    dirichlet_alpha,
                    dirichlet_fraction,
                    temperature_moves,
                },
            )?;
            let elapsed = start.elapsed();
            println!("first game actions: {:?}", result.first_game_actions);
            println!("draws: {}/{games}", result.draws);
            println!(
                "self-play data: {} positions from {} unique games",
                result.samples.len(),
                result.unique_games
            );
            println!(
                "network evaluations: {} in {} batches ({:.1} average)",
                result.evaluations,
                result.batches,
                result.evaluations as f64 / result.batches as f64
            );
            println!(
                "search time: {elapsed:.3?} ({:.0} evaluations/s)",
                result.evaluations as f64 / elapsed.as_secs_f64()
            );
        }
        Command::Learn { config, resume } => {
            experiment::run(config, candle_device()?, resume)?;
        }
    }
    Ok(())
}

fn default_actor_workers() -> usize {
    std::thread::available_parallelism()
        .map(|threads| threads.get().saturating_sub(1).clamp(1, 7))
        .unwrap_or(1)
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
