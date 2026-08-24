use std::time::Instant;

use candle_core::{Device, Tensor};
use clap::{Parser, Subcommand};
use etive::evaluator::OthelloCandleEvaluator;
use etive::othello::actors::{ActorConfig, run as run_actors};
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
    /// Verify Candle on the selected device.
    Candle,
    /// Run as a Go Text Protocol v2 Othello engine.
    Gtp,
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
        Command::Gtp => {
            let stdin = std::io::stdin();
            let stdout = std::io::stdout();
            gtp::run(stdin.lock(), stdout.lock())?;
        }
        Command::Mcts {
            simulations,
            seed,
            games,
            batch_size,
            workers,
        } => {
            if simulations == 0 || games == 0 || batch_size == 0 {
                return Err("simulations, games, and batch size must be greater than zero".into());
            }
            let device = candle_device()?;
            let evaluator = OthelloCandleEvaluator::new(device, seed)?;
            let workers = workers.unwrap_or_else(default_actor_workers);
            let start = Instant::now();
            let result = run_actors(
                evaluator,
                ActorConfig {
                    games,
                    simulations,
                    workers,
                    inference_batch_size: batch_size,
                },
            )?;
            let elapsed = start.elapsed();
            println!("first game actions: {:?}", result.first_game_actions);
            println!("draws: {}/{games}", result.draws);
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
