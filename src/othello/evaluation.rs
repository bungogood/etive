//! Checkpoint comparison through color-balanced, fixed-search Othello games.

use std::error::Error;
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tracing::info;

use super::{Board, Color, OthelloBurnEvaluator};
use crate::arena::{self, ArenaConfig};
use crate::game::Game;

pub use crate::arena::{EvalProgress, EvalResult};

#[derive(Clone, Copy, Debug)]
pub struct EvalConfig {
    pub games: usize,
    pub simulations: u32,
    pub batch_size: usize,
    pub opening_plies: usize,
    pub seed: u64,
}

pub fn evaluate(
    candidate: &mut OthelloBurnEvaluator,
    baseline: &mut OthelloBurnEvaluator,
    config: EvalConfig,
) -> Result<EvalResult, Box<dyn Error>> {
    let EvalConfig {
        games,
        simulations,
        batch_size,
        opening_plies,
        seed,
    } = config;
    if games == 0 || !games.is_multiple_of(2) || simulations < 2 || batch_size == 0 {
        return Err(
            "evaluation games must be positive and even, simulations at least two, and batch size positive"
                .into(),
        );
    }

    let start = Instant::now();
    let mut last_progress = start;
    let mut last_evaluations = 0;
    arena::evaluate(
        candidate,
        baseline,
        ArenaConfig {
            simulations,
            batch_size,
        },
        openings(games, opening_plies, seed),
        |progress| {
            let interval = last_progress.elapsed();
            if interval >= Duration::from_secs(5) || progress.completed == progress.total {
                info!(
                    completed = progress.completed,
                    total = progress.total,
                    moves = progress.moves,
                    evaluations = progress.evaluations,
                    games_per_second = %format_args!(
                        "{:.1}",
                        progress.completed as f64 / start.elapsed().as_secs_f64()
                    ),
                    evaluations_per_second = %format_args!(
                        "{:.0}",
                        (progress.evaluations - last_evaluations) as f64 / interval.as_secs_f64()
                    ),
                    elapsed = %format_args!("{:.1}s", start.elapsed().as_secs_f64()),
                    "evaluation progress"
                );
                last_progress = Instant::now();
                last_evaluations = progress.evaluations;
            }
        },
    )
}

fn openings(games: usize, opening_plies: usize, seed: u64) -> Vec<(Board, Color)> {
    let mut random = StdRng::seed_from_u64(seed);
    let mut boards = Vec::with_capacity(games);
    for _ in 0..games / 2 {
        let mut board = Board::default();
        for _ in 0..opening_plies {
            if board.outcome().is_some() {
                break;
            }
            let actions = board.legal_actions().collect::<Vec<_>>();
            let action = actions[random.random_range(0..actions.len())];
            board.play_unchecked(action);
        }
        boards.push((board, Color::Black));
        boards.push((board, Color::White));
    }
    boards
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openings_are_diverse_and_color_paired() {
        let openings = openings(10, 8, 7);

        assert_eq!(openings.len(), 10);
        for pair in openings.as_chunks::<2>().0 {
            assert_eq!(pair[0].0, pair[1].0);
            assert_eq!(pair[0].1, Color::Black);
            assert_eq!(pair[1].1, Color::White);
        }
        assert!(
            openings[2..]
                .iter()
                .any(|opening| opening.0 != openings[0].0)
        );
    }

    #[test]
    fn terminal_openings_are_generated_in_pairs() {
        let openings = openings(20, usize::MAX, 11);

        assert!(openings.iter().all(|(board, _)| board.outcome().is_some()));
        for pair in openings.as_chunks::<2>().0 {
            assert_eq!(pair[0].0, pair[1].0);
        }
    }
}
