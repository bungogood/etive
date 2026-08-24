use std::time::Instant;

use candle_core::{Device, Tensor};
use clap::{Parser, Subcommand};
use etive::evaluator::OthelloCandleEvaluator;
use etive::game::Game;
use etive::mcts::{Mcts, MctsConfig, run_batched};
use etive::othello::{Board, perft};
use rayon::prelude::*;

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
        } => {
            if simulations == 0 || games == 0 || batch_size == 0 {
                return Err("simulations, games, and batch size must be greater than zero".into());
            }
            let device = candle_device()?;
            let mut evaluator = OthelloCandleEvaluator::new(device, seed)?;
            let mut searches = (0..games)
                .map(|_| Mcts::new(Board::default(), MctsConfig::default()))
                .collect::<Vec<_>>();
            let mut actions = vec![None; games];
            let mut first_game_actions = Vec::new();
            let start = Instant::now();
            while searches
                .par_iter()
                .any(|search| search.root_position().outcome().is_none())
            {
                run_batched(&mut searches, &mut evaluator, simulations, batch_size)?;
                searches
                    .par_iter()
                    .zip(actions.par_iter_mut())
                    .for_each(|(search, action)| {
                        *action = if search.root_position().outcome().is_none() {
                            search.best_action()
                        } else {
                            None
                        };
                    });
                if searches.iter().zip(&actions).any(|(search, action)| {
                    search.root_position().outcome().is_none() && action.is_none()
                }) {
                    return Err("search produced no legal action".into());
                }
                if let Some(action) = actions[0] {
                    first_game_actions.push(Board::action_index(action));
                }
                searches
                    .par_iter_mut()
                    .zip(actions.par_iter())
                    .for_each(|(search, &action)| {
                        if let Some(action) = action {
                            assert!(search.advance(action));
                        }
                    });
            }
            let draws = searches
                .par_iter()
                .filter(|search| {
                    search.root_position().outcome() == Some(etive::game::Outcome::Draw)
                })
                .count();
            let elapsed = start.elapsed();
            let evaluations = evaluator.evaluations();
            let batches = evaluator.batches();
            println!("first game actions: {first_game_actions:?}");
            println!("draws: {draws}/{games}");
            println!(
                "network evaluations: {evaluations} in {batches} batches ({:.1} average)",
                evaluations as f64 / batches as f64
            );
            println!(
                "search time: {elapsed:.3?} ({:.0} evaluations/s)",
                evaluations as f64 / elapsed.as_secs_f64()
            );
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
