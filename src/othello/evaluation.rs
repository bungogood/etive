//! Checkpoint comparison through color-balanced, fixed-search Othello games.

use std::error::Error;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::{Board, Color, GameStatus};
use crate::evaluator::OthelloBurnEvaluator;
use crate::game::Game;
use crate::mcts::{Mcts, SearchWorkspace};

#[derive(Clone, Copy, Debug, Default)]
pub struct EvalResult {
    pub candidate_wins: usize,
    pub baseline_wins: usize,
    pub draws: usize,
    /// Opening-pair counts by candidate points: 0, 0.5, 1, 1.5, or 2.
    pub pair_scores: [usize; 5],
}

impl EvalResult {
    pub fn score(self) -> f64 {
        let games = self.candidate_wins + self.baseline_wins + self.draws;
        if games == 0 {
            return f64::NAN;
        }
        (self.candidate_wins as f64 + 0.5 * self.draws as f64) / games as f64
    }

    /// Normal-approximation likelihood that the candidate's paired score exceeds 50%.
    pub fn paired_los(self) -> f64 {
        let pairs = self.pair_scores.iter().sum::<usize>();
        if pairs < 2 {
            return 0.5;
        }
        let mean = self
            .pair_scores
            .iter()
            .enumerate()
            .map(|(half_points, &count)| half_points as f64 * 0.5 * count as f64)
            .sum::<f64>()
            / pairs as f64;
        let variance = self
            .pair_scores
            .iter()
            .enumerate()
            .map(|(half_points, &count)| {
                let difference = half_points as f64 * 0.5 - mean;
                difference * difference * count as f64
            })
            .sum::<f64>()
            / (pairs - 1) as f64;
        let standard_error = (variance / pairs as f64).sqrt();
        if standard_error == 0.0 {
            return match mean.total_cmp(&1.0) {
                std::cmp::Ordering::Less => 0.0,
                std::cmp::Ordering::Equal => 0.5,
                std::cmp::Ordering::Greater => 1.0,
            };
        }
        standard_normal_cdf((mean - 1.0) / standard_error)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EvalProgress {
    pub completed: usize,
    pub total: usize,
    pub moves: usize,
    pub evaluations: u64,
}

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
    mut report_progress: impl FnMut(EvalProgress),
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

    let mut result = EvalResult::default();
    let baseline_evaluations = baseline.evaluations();
    let candidate_evaluations = candidate.evaluations();
    let mut moves = 0;
    let mut boards = openings(games, opening_plies, seed);
    let mut game_points = vec![None; games];
    for (index, &(board, baseline_color)) in boards.iter().enumerate() {
        game_points[index] = score_terminal(&mut result, board, baseline_color);
    }
    let mut workspace = SearchWorkspace::new(batch_size);

    while completed(&result) < games {
        let mut baseline_turn = Vec::with_capacity(games / 2);
        let mut candidate_turn = Vec::with_capacity(games / 2);
        for (index, (board, baseline_color)) in boards.iter().enumerate() {
            if board.outcome().is_some() {
                continue;
            }
            if board.side_to_move() == *baseline_color {
                baseline_turn.push(index);
            } else {
                candidate_turn.push(index);
            }
        }

        search_moves(
            &mut workspace,
            baseline,
            &mut boards,
            &baseline_turn,
            simulations,
        )?;
        search_moves(
            &mut workspace,
            candidate,
            &mut boards,
            &candidate_turn,
            simulations,
        )?;
        moves += baseline_turn.len() + candidate_turn.len();

        for index in baseline_turn.into_iter().chain(candidate_turn) {
            let (board, baseline_color) = boards[index];
            game_points[index] = score_terminal(&mut result, board, baseline_color);
        }
        report_progress(EvalProgress {
            completed: completed(&result),
            total: games,
            moves,
            evaluations: baseline.evaluations().saturating_sub(baseline_evaluations)
                + candidate
                    .evaluations()
                    .saturating_sub(candidate_evaluations),
        });
    }
    for pair in game_points.as_chunks::<2>().0 {
        let half_points = pair[0].expect("evaluated game must be terminal")
            + pair[1].expect("evaluated game must be terminal");
        result.pair_scores[half_points as usize] += 1;
    }
    Ok(result)
}

fn completed(result: &EvalResult) -> usize {
    result.candidate_wins + result.baseline_wins + result.draws
}

fn score_terminal(result: &mut EvalResult, board: Board, baseline_color: Color) -> Option<u8> {
    match board.status() {
        GameStatus::Drawn => {
            result.draws += 1;
            Some(1)
        }
        GameStatus::Won(winner) if winner == baseline_color => {
            result.baseline_wins += 1;
            Some(0)
        }
        GameStatus::Won(_) => {
            result.candidate_wins += 1;
            Some(2)
        }
        GameStatus::Ongoing => None,
    }
}

fn standard_normal_cdf(value: f64) -> f64 {
    let absolute = value.abs();
    let scale = 1.0 / (1.0 + 0.231_641_9 * absolute);
    let density = (-0.5 * absolute * absolute).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let tail = density
        * scale
        * (0.319_381_530
            + scale
                * (-0.356_563_782
                    + scale * (1.781_477_937 + scale * (-1.821_255_978 + scale * 1.330_274_429))));
    if value >= 0.0 { 1.0 - tail } else { tail }
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

fn search_moves(
    workspace: &mut SearchWorkspace<Board>,
    evaluator: &mut OthelloBurnEvaluator,
    boards: &mut [(Board, Color)],
    game_indices: &[usize],
    simulations: u32,
) -> Result<(), Box<dyn Error>> {
    let mut searches = game_indices
        .iter()
        .map(|&index| Mcts::new(boards[index].0))
        .collect::<Vec<_>>();
    workspace.run_batched(&mut searches, evaluator, simulations)?;
    for (&game_index, search) in game_indices.iter().zip(searches) {
        let action = search
            .best_action()
            .ok_or("evaluation search found no action")?;
        boards[game_index].0.play_unchecked(action);
    }
    Ok(())
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
    fn terminal_openings_are_all_scored() {
        let openings = openings(20, usize::MAX, 11);
        assert!(openings.iter().all(|(board, _)| board.outcome().is_some()));
        let mut result = EvalResult::default();

        for (board, baseline_color) in openings {
            let _ = score_terminal(&mut result, board, baseline_color);
        }

        assert_eq!(completed(&result), 20);
    }

    #[test]
    fn paired_los_uses_opening_pairs_as_observations() {
        let equal = EvalResult {
            pair_scores: [0, 0, 100, 0, 0],
            ..EvalResult::default()
        };
        let superior = EvalResult {
            pair_scores: [10, 10, 20, 20, 40],
            ..EvalResult::default()
        };
        let inferior = EvalResult {
            pair_scores: [40, 20, 20, 10, 10],
            ..EvalResult::default()
        };

        assert_eq!(equal.paired_los(), 0.5);
        assert!(superior.paired_los() > 0.99);
        assert!(inferior.paired_los() < 0.01);
    }
}
