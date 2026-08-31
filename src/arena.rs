//! Generic candidate-versus-baseline matches with color-balanced position pairs.

use std::error::Error;

use crate::evaluator::{BatchEvaluator, EvaluationCounter};
use crate::game::{Color, Game, Outcome};
use crate::mcts::{Mcts, SearchWorkspace};

#[derive(Clone, Copy, Debug, Default)]
pub struct EvalResult {
    pub candidate_wins: usize,
    pub baseline_wins: usize,
    pub draws: usize,
    /// Position-pair counts by candidate points: 0, 0.5, 1, 1.5, or 2.
    pub pair_scores: [usize; 5],
}

impl EvalResult {
    pub fn score(self) -> f64 {
        let games = self.completed();
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

    fn completed(self) -> usize {
        self.candidate_wins + self.baseline_wins + self.draws
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
pub struct ArenaConfig {
    pub simulations: u32,
    pub batch_size: usize,
}

/// Evaluates consecutive color-balanced position pairs.
pub fn evaluate<G, C, B>(
    candidate: &mut C,
    baseline: &mut B,
    config: ArenaConfig,
    mut games: Vec<(G, Color)>,
    mut report_progress: impl FnMut(EvalProgress),
) -> Result<EvalResult, Box<dyn Error>>
where
    G: Game,
    C: BatchEvaluator<G> + EvaluationCounter,
    B: BatchEvaluator<G> + EvaluationCounter,
    C::Error: Error + 'static,
    B::Error: Error + 'static,
{
    if games.is_empty()
        || !games.len().is_multiple_of(2)
        || config.simulations < 2
        || config.batch_size == 0
    {
        return Err(
            "arena games must be positive and even, simulations at least two, and batch size positive"
                .into(),
        );
    }

    let total = games.len();
    let mut result = EvalResult::default();
    let baseline_evaluations = baseline.evaluations();
    let candidate_evaluations = candidate.evaluations();
    let mut moves = 0;
    let mut game_points = vec![None; total];
    for (index, &(game, baseline_color)) in games.iter().enumerate() {
        game_points[index] = score_terminal(&mut result, game, baseline_color);
    }
    let mut workspace = SearchWorkspace::new(config.batch_size);

    while result.completed() < total {
        let mut baseline_turn = Vec::with_capacity(total / 2);
        let mut candidate_turn = Vec::with_capacity(total / 2);
        for (index, (game, baseline_color)) in games.iter().enumerate() {
            if game.outcome().is_some() {
                continue;
            }
            if game.side_to_move() == *baseline_color {
                baseline_turn.push(index);
            } else {
                candidate_turn.push(index);
            }
        }

        search_moves(
            &mut workspace,
            baseline,
            &mut games,
            &baseline_turn,
            config.simulations,
        )?;
        search_moves(
            &mut workspace,
            candidate,
            &mut games,
            &candidate_turn,
            config.simulations,
        )?;
        moves += baseline_turn.len() + candidate_turn.len();

        for index in baseline_turn.into_iter().chain(candidate_turn) {
            let (game, baseline_color) = games[index];
            game_points[index] = score_terminal(&mut result, game, baseline_color);
        }
        report_progress(EvalProgress {
            completed: result.completed(),
            total,
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

fn score_terminal<G: Game>(result: &mut EvalResult, game: G, baseline_color: Color) -> Option<u8> {
    let winner = match game.outcome()? {
        Outcome::Win => game.side_to_move(),
        Outcome::Loss => !game.side_to_move(),
        Outcome::Draw => {
            result.draws += 1;
            return Some(1);
        }
    };
    if winner == baseline_color {
        result.baseline_wins += 1;
        Some(0)
    } else {
        result.candidate_wins += 1;
        Some(2)
    }
}

fn search_moves<G, E>(
    workspace: &mut SearchWorkspace<G>,
    evaluator: &mut E,
    games: &mut [(G, Color)],
    game_indices: &[usize],
    simulations: u32,
) -> Result<(), Box<dyn Error>>
where
    G: Game,
    E: BatchEvaluator<G>,
    E::Error: Error + 'static,
{
    let mut searches = game_indices
        .iter()
        .map(|&index| Mcts::new(games[index].0))
        .collect::<Vec<_>>();
    workspace.run_batched(&mut searches, evaluator, simulations)?;
    for (&game_index, search) in game_indices.iter().zip(searches) {
        let action = search.best_action().ok_or("arena search found no action")?;
        games[game_index].0.play_unchecked(action);
    }
    Ok(())
}

fn standard_normal_cdf(value: f64) -> f64 {
    0.5 * (1.0 + libm::erf(value / std::f64::consts::SQRT_2))
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use crate::evaluator::UniformEvaluator;
    use crate::tic_tac_toe::{Board, position};

    use super::*;

    #[test]
    fn paired_los_uses_position_pairs_as_observations() {
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

    struct MeasuredUniformEvaluator {
        evaluator: UniformEvaluator,
        evaluations: u64,
    }

    impl BatchEvaluator<Board> for MeasuredUniformEvaluator {
        type Error = Infallible;

        fn evaluate_batch(
            &mut self,
            games: &[Board],
            policy_logits: &mut [f32],
            values: &mut [f32],
        ) -> Result<(), Self::Error> {
            self.evaluations += games.len() as u64;
            self.evaluator.evaluate_batch(games, policy_logits, values)
        }
    }

    impl EvaluationCounter for MeasuredUniformEvaluator {
        fn evaluations(&self) -> u64 {
            self.evaluations
        }
    }

    #[test]
    fn terminal_position_pairs_are_scored_without_evaluation() {
        let game = position(&[0, 3, 1, 4, 2]);
        let mut candidate = MeasuredUniformEvaluator {
            evaluator: UniformEvaluator,
            evaluations: 0,
        };
        let mut baseline = MeasuredUniformEvaluator {
            evaluator: UniformEvaluator,
            evaluations: 0,
        };

        let result = evaluate(
            &mut candidate,
            &mut baseline,
            ArenaConfig {
                simulations: 2,
                batch_size: 1,
            },
            vec![(game, Color::Black), (game, Color::White)],
            |_| panic!("terminal positions should not report move progress"),
        )
        .unwrap();

        assert_eq!(result.candidate_wins, 1);
        assert_eq!(result.baseline_wins, 1);
        assert_eq!(result.pair_scores, [0, 0, 1, 0, 0]);
        assert_eq!(candidate.evaluations + baseline.evaluations, 0);
    }
}
