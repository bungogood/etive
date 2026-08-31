use crate::evaluator::{BatchEvaluator, InferenceBatch};
use crate::game::Game;

use super::{EvaluationRequest, Mcts, MctsError, SearchError, Selection};

/// Reusable scheduling and inference storage for batched MCTS calls.
pub struct SearchWorkspace<G: Game> {
    completed: Vec<u32>,
    batch: InferenceBatch<G, (usize, EvaluationRequest)>,
}

impl<G: Game> SearchWorkspace<G> {
    pub fn new(maximum: usize) -> Self {
        Self {
            completed: Vec::new(),
            batch: InferenceBatch::new(maximum),
        }
    }

    pub fn run_batched<E: BatchEvaluator<G>>(
        &mut self,
        trees: &mut [Mcts<G>],
        evaluator: &mut E,
        simulations: u32,
    ) -> Result<(), SearchError<E::Error>> {
        self.run(trees, evaluator, simulations, 1)
    }

    pub fn run_parallel<E: BatchEvaluator<G>>(
        &mut self,
        trees: &mut [Mcts<G>],
        evaluator: &mut E,
        simulations: u32,
    ) -> Result<(), SearchError<E::Error>> {
        self.run(trees, evaluator, simulations, self.batch.capacity())
    }

    fn run<E: BatchEvaluator<G>>(
        &mut self,
        trees: &mut [Mcts<G>],
        evaluator: &mut E,
        simulations: u32,
        max_pending_per_tree: usize,
    ) -> Result<(), SearchError<E::Error>> {
        if trees.iter().any(Mcts::is_pending) {
            return Err(SearchError::Mcts(MctsError::EvaluationPending));
        }
        self.completed.resize(trees.len(), 0);
        self.completed.fill(0);
        while self.completed.iter().any(|&count| count < simulations) {
            self.batch.clear();
            for tree_index in 0..trees.len() {
                while self.completed[tree_index] + (trees[tree_index].pending_count() as u32)
                    < simulations
                    && trees[tree_index].pending_count() < max_pending_per_tree
                    && !self.batch.is_full()
                {
                    let selection = match trees[tree_index].select() {
                        Ok(selection) => selection,
                        Err(error) => {
                            for &(selected_tree, request) in self.batch.tags() {
                                trees[selected_tree].cancel(request);
                            }
                            return Err(SearchError::Mcts(error));
                        }
                    };
                    match selection {
                        Selection::Terminal => self.completed[tree_index] += 1,
                        Selection::Evaluate { request, position } => {
                            self.batch.push((tree_index, request), *position);
                        }
                        Selection::Blocked => break,
                    }
                }
                if self.batch.is_full() {
                    break;
                }
            }
            if self.batch.is_empty() {
                debug_assert!(self.completed.iter().all(|&count| count == simulations));
                break;
            }

            if let Err(error) = self.batch.evaluate_batch(evaluator) {
                for &(tree_index, request) in self.batch.tags() {
                    trees[tree_index].cancel(request);
                }
                return Err(SearchError::Evaluator(error));
            }

            for index in (0..self.batch.len()).rev() {
                let (&(tree_index, request), policy, value) = self.batch.result(index);
                if let Err(error) = trees[tree_index].complete(request, policy, value) {
                    for &(waiting_tree, waiting_request) in self.batch.tags().take(index) {
                        trees[waiting_tree].cancel(waiting_request);
                    }
                    return Err(SearchError::Mcts(error));
                }
                self.completed[tree_index] += 1;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::evaluator::{BatchEvaluator, UniformEvaluator};
    use crate::tic_tac_toe::{Board, Square};

    use super::*;

    struct RecordingEvaluator {
        batches: Vec<usize>,
        invalid: bool,
        fail: bool,
    }

    impl BatchEvaluator<Board> for RecordingEvaluator {
        type Error = &'static str;

        fn evaluate_batch(
            &mut self,
            games: &[Board],
            policy_logits: &mut [f32],
            values: &mut [f32],
        ) -> Result<(), Self::Error> {
            self.batches.push(games.len());
            if self.fail {
                return Err("failed");
            }
            policy_logits.fill(0.0);
            values.fill(if self.invalid { 2.0 } else { 0.0 });
            Ok(())
        }
    }

    fn position(actions: &[usize]) -> Board {
        let mut board = Board::default();
        for &index in actions {
            board.play(Square::from_index(index).unwrap());
        }
        board
    }

    #[test]
    fn one_tree_fills_parallel_batches() {
        let mut trees = [Mcts::new(Board::default())];
        trees[0].run(&mut UniformEvaluator, 3).unwrap();
        let mut evaluator = RecordingEvaluator {
            batches: Vec::new(),
            invalid: false,
            fail: false,
        };

        SearchWorkspace::new(4)
            .run_parallel(&mut trees, &mut evaluator, 12)
            .unwrap();

        assert!(evaluator.batches.iter().any(|&size| size > 1));
        assert!(evaluator.batches.iter().all(|&size| size <= 4));
        assert_eq!(trees[0].nodes[trees[0].root].visits, 15);
        assert_eq!(trees[0].pending_count(), 0);
    }

    #[test]
    fn errors_cancel_every_pending_request() {
        for (invalid, fail) in [(false, true), (true, false)] {
            let mut trees = [Mcts::new(Board::default())];
            trees[0].run(&mut UniformEvaluator, 1).unwrap();
            let mut evaluator = RecordingEvaluator {
                batches: Vec::new(),
                invalid,
                fail,
            };

            assert!(
                SearchWorkspace::new(4)
                    .run_parallel(&mut trees, &mut evaluator, 4)
                    .is_err()
            );
            assert_eq!(trees[0].pending_count(), 0);
            assert!(trees[0].nodes.iter().all(|node| node.reservations == 0));
            assert!(trees[0].edges.iter().all(|edge| edge.reservations == 0));
        }
    }

    #[test]
    fn terminal_roots_match_synchronous_search_without_inference() {
        let board = position(&[0, 3, 1, 4, 2]);
        let mut synchronous = Mcts::new(board);
        synchronous.run(&mut UniformEvaluator, 11).unwrap();
        let mut batched = [Mcts::new(board)];
        let mut evaluator = RecordingEvaluator {
            batches: Vec::new(),
            invalid: false,
            fail: false,
        };
        SearchWorkspace::new(4)
            .run_batched(&mut batched, &mut evaluator, 11)
            .unwrap();

        assert!(evaluator.batches.is_empty());
        assert_eq!(batched[0].root_value(), synchronous.root_value());
        assert_eq!(
            batched[0].root_stats().collect::<Vec<_>>(),
            synchronous.root_stats().collect::<Vec<_>>()
        );
    }

    #[test]
    fn batched_search_matches_synchronous_search() {
        let mut synchronous_evaluator = UniformEvaluator;
        let mut synchronous = Mcts::new(Board::default());
        synchronous.run(&mut synchronous_evaluator, 128).unwrap();

        let mut batched_evaluator = RecordingEvaluator {
            batches: Vec::new(),
            invalid: false,
            fail: false,
        };
        let mut batched = (0..32)
            .map(|_| Mcts::new(Board::default()))
            .collect::<Vec<_>>();
        SearchWorkspace::new(16)
            .run_batched(&mut batched, &mut batched_evaluator, 128)
            .unwrap();

        let expected = batched[0].root_stats().collect::<Vec<_>>();
        assert!(
            batched
                .iter()
                .all(|tree| tree.root_stats().collect::<Vec<_>>() == expected)
        );
        assert_eq!(batched[0].best_action(), synchronous.best_action());
        assert_eq!(expected.iter().map(|stats| stats.visits).sum::<u32>(), 127);
        assert!(batched_evaluator.batches.iter().all(|&size| size <= 16));
    }
}
