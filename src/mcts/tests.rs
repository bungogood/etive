use std::convert::Infallible;

use super::*;
use crate::evaluator::{BatchEvaluator, TicTacToeMinimaxEvaluator, UniformEvaluator};
use crate::tic_tac_toe::{Board, position, square};

#[test]
fn expansion_appends_one_contiguous_edge_range() {
    let mut search = Mcts::new(Board::default());
    search.run(&mut UniformEvaluator, 1).unwrap();

    assert_eq!(search.node_count(), 1);
    assert_eq!(search.edge_count(), 9);
    assert_eq!(search.nodes[0].edge_range(), Some(0..9));
    assert!((search.root_stats().map(|stats| stats.prior).sum::<f32>() - 1.0).abs() < 1e-6);
}

#[test]
fn a_pending_root_blocks_colliding_selection_without_changing_stats() {
    let mut search = Mcts::new(Board::default());
    let request = match search.select().unwrap() {
        Selection::Evaluate { request, position } => {
            assert_eq!(position, &Board::default());
            request
        }
        Selection::Terminal => panic!("initial position is not terminal"),
        Selection::Blocked => unreachable!(),
    };

    assert!(search.is_pending());
    assert_eq!(search.pending_count(), 1);
    assert!(matches!(search.select().unwrap(), Selection::Blocked));
    assert_eq!(search.root_value(), 0.0);
    assert_eq!(search.nodes[search.root].visits, 0);
    assert!(!search.advance(square(0)));

    search.complete(request, &[0.0; 9], 0.0).unwrap();
    assert!(!search.is_pending());
    assert_eq!(search.edge_count(), 9);
    assert_eq!(search.nodes[search.root].visits, 1);
}

fn select_request(search: &mut Mcts<Board>) -> EvaluationRequest {
    match search.select().unwrap() {
        Selection::Evaluate { request, .. } => request,
        Selection::Terminal | Selection::Blocked => panic!("expected an evaluation request"),
    }
}

#[test]
fn distinct_leaves_can_be_pending_and_completed_out_of_order() {
    let mut search = Mcts::new(Board::default());
    search.run(&mut UniformEvaluator, 1).unwrap();

    let first = select_request(&mut search);
    let second = select_request(&mut search);

    assert_eq!(search.pending_count(), 2);
    assert_ne!(search.pending[0].node, search.pending[1].node);
    assert!(search.root_stats().all(|stats| stats.visits == 0));

    search.complete(second, &[0.0; 9], 0.25).unwrap();
    assert_eq!(search.pending_count(), 1);
    search.complete(first, &[0.0; 9], -0.5).unwrap();

    assert_eq!(search.pending_count(), 0);
    assert_eq!(search.nodes[search.root].visits, 3);
    assert_eq!(
        search.root_stats().map(|stats| stats.visits).sum::<u32>(),
        2
    );
    assert!(search.nodes.iter().all(|node| node.reservations == 0));
    assert!(search.edges.iter().all(|edge| edge.reservations == 0));
}

#[test]
fn cancelling_one_request_does_not_release_another() {
    let mut search = Mcts::new(Board::default());
    search.run(&mut UniformEvaluator, 1).unwrap();
    let first = select_request(&mut search);
    let second = select_request(&mut search);

    assert!(search.cancel(first));
    assert_eq!(search.pending_count(), 1);
    assert!(matches!(
        search.complete(first, &[0.0; 9], 0.0),
        Err(MctsError::StaleRequest)
    ));
    assert_eq!(search.pending_count(), 1);
    search.complete(second, &[0.0; 9], 0.0).unwrap();

    assert_eq!(search.pending_count(), 0);
    assert!(search.nodes.iter().all(|node| node.reservations == 0));
    assert!(search.edges.iter().all(|edge| edge.reservations == 0));
}

#[test]
fn pending_requests_guard_advance_and_root_prior_mixing() {
    let mut search = Mcts::new(Board::default());
    search.run(&mut UniformEvaluator, 1).unwrap();
    let request = select_request(&mut search);

    assert!(!search.advance(square(0)));
    assert!(!search.mix_root_priors(&[1.0 / 9.0; 9], 0.25));

    assert!(search.cancel(request));
    assert!(search.mix_root_priors(&[1.0 / 9.0; 9], 0.25));
}

#[test]
fn stale_completion_does_not_disturb_the_live_request() {
    let mut search = Mcts::new(Board::default());
    let first = match search.select().unwrap() {
        Selection::Evaluate { request, .. } => request,
        Selection::Terminal => unreachable!(),
        Selection::Blocked => unreachable!(),
    };
    assert!(search.cancel(first));
    let second = match search.select().unwrap() {
        Selection::Evaluate { request, .. } => request,
        Selection::Terminal => unreachable!(),
        Selection::Blocked => unreachable!(),
    };

    assert!(matches!(
        search.complete(first, &[0.0; 9], 0.0),
        Err(MctsError::StaleRequest)
    ));
    assert!(search.is_pending());
    search.complete(second, &[0.0; 9], 0.0).unwrap();
    assert!(!search.is_pending());
}

#[test]
fn requests_are_scoped_to_one_tree() {
    let mut first = Mcts::new(Board::default());
    let mut second = Mcts::new(Board::default());
    let first_request = match first.select().unwrap() {
        Selection::Evaluate { request, .. } => request,
        Selection::Terminal => unreachable!(),
        Selection::Blocked => unreachable!(),
    };
    let second_request = match second.select().unwrap() {
        Selection::Evaluate { request, .. } => request,
        Selection::Terminal => unreachable!(),
        Selection::Blocked => unreachable!(),
    };

    assert_ne!(first_request, second_request);
    assert!(matches!(
        second.complete(first_request, &[0.0; 9], 0.0),
        Err(MctsError::StaleRequest)
    ));
    assert!(second.is_pending());
    second.complete(second_request, &[0.0; 9], 0.0).unwrap();
}

#[test]
fn split_and_synchronous_search_produce_identical_statistics() {
    let mut synchronous = Mcts::new(Board::default());
    synchronous.run(&mut UniformEvaluator, 128).unwrap();

    let mut split = Mcts::new(Board::default());
    for _ in 0..128 {
        match split.select().unwrap() {
            Selection::Terminal => {}
            Selection::Evaluate { request, .. } => {
                split.complete(request, &[0.0; 9], 0.0).unwrap();
            }
            Selection::Blocked => unreachable!(),
        }
    }

    assert_eq!(split.root_value(), synchronous.root_value());
    assert_eq!(
        split.root_stats().collect::<Vec<_>>(),
        synchronous.root_stats().collect::<Vec<_>>()
    );
    assert_eq!(split.node_count(), synchronous.node_count());
    assert_eq!(split.edge_count(), synchronous.edge_count());
}

#[test]
fn evaluator_failure_releases_the_pending_leaf() {
    struct FailingEvaluator;

    impl BatchEvaluator<Board> for FailingEvaluator {
        type Error = &'static str;

        fn evaluate_batch(
            &mut self,
            _games: &[Board],
            _policy_logits: &mut [f32],
            _values: &mut [f32],
        ) -> Result<(), Self::Error> {
            Err("failed")
        }
    }

    let mut search = Mcts::new(Board::default());
    assert!(matches!(
        search.run(&mut FailingEvaluator, 1),
        Err(SearchError::Evaluator("failed"))
    ));
    assert!(!search.is_pending());
    assert!(matches!(search.select(), Ok(Selection::Evaluate { .. })));
}

#[test]
fn search_finds_an_immediate_win_and_backs_it_up_positively() {
    let board = position(&[0, 3, 1, 4]);
    let mut search = Mcts::new(board);
    search.run(&mut UniformEvaluator, 128).unwrap();

    assert_eq!(search.best_action(), Some(square(2)));
    let winning = search
        .root_stats()
        .find(|stats| stats.action == square(2))
        .unwrap();
    assert!(winning.value > 0.9);
    assert!(winning.visits > 0);
}

#[test]
fn minimax_policy_blocks_a_forced_loss() {
    let board = position(&[0, 4, 1]);
    let mut search = Mcts::new(board);
    search.run(&mut TicTacToeMinimaxEvaluator, 64).unwrap();

    assert_eq!(search.best_action(), Some(square(2)));
    assert!(search.root_value().abs() < 0.1);
}

#[test]
fn terminal_nodes_bypass_the_evaluator() {
    struct CountingEvaluator {
        calls: usize,
    }

    impl BatchEvaluator<Board> for CountingEvaluator {
        type Error = Infallible;

        fn evaluate_batch(
            &mut self,
            games: &[Board],
            policy_logits: &mut [f32],
            values: &mut [f32],
        ) -> Result<(), Self::Error> {
            self.calls += games.len();
            policy_logits.fill(0.0);
            values.fill(0.0);
            Ok(())
        }
    }

    let board = position(&[0, 3, 1, 4, 2]);
    let mut evaluator = CountingEvaluator { calls: 0 };
    let mut search = Mcts::new(board);
    search.run(&mut evaluator, 8).unwrap();

    assert_eq!(evaluator.calls, 0);
    assert_eq!(search.node_count(), 1);
    assert_eq!(search.edge_count(), 0);
    assert_eq!(search.root_value(), -1.0);
}

#[test]
fn advance_retains_the_chosen_subtree_and_reclaims_siblings() {
    let mut search = Mcts::new(Board::default());
    search.run(&mut TicTacToeMinimaxEvaluator, 256).unwrap();
    let action = search.best_action().unwrap();
    let mut expected = *search.root_position();
    expected.play(action);
    let nodes_before = search.node_count();
    let edges_before = search.edge_count();
    let root_range = search.nodes[search.root].edge_range().unwrap();
    let selected_edge = root_range
        .into_iter()
        .find(|&index| search.edges[index].action == action)
        .unwrap();
    let selected_child = search.edges[selected_edge].child.unwrap();
    let retained_visits = search.nodes[selected_child].visits;
    let retained_value = search.nodes[selected_child].value_sum;

    assert!(search.advance(action));

    assert_eq!(search.root_position(), &expected);
    assert_eq!(search.root, 0);
    assert_eq!(search.nodes[0].visits, retained_visits);
    assert_eq!(search.nodes[0].value_sum, retained_value);
    assert!(search.node_count() < nodes_before);
    assert!(search.edge_count() < edges_before);
}

#[test]
fn rebase_root_clears_decision_statistics_but_retains_descendants() {
    let mut search = Mcts::new(Board::default());
    search.run(&mut TicTacToeMinimaxEvaluator, 256).unwrap();
    let action = search.best_action().unwrap();
    assert!(search.advance(action));
    let nodes = search.node_count();
    let edges = search.edge_count();

    assert!(search.rebase_root());

    assert_eq!(search.nodes[search.root].visits, 0);
    assert_eq!(search.nodes[search.root].value_sum, 0.0);
    assert!(search.root_stats().all(|stats| stats.visits == 0));
    assert_eq!(search.node_count(), nodes);
    assert_eq!(search.edge_count(), edges);
}
