use etive::evaluator::UniformEvaluator;
use etive::game::{Game, Outcome};
use etive::mcts::Mcts;
use etive::othello;

fn action_indices<G: Game>(game: &G) -> Vec<usize> {
    game.legal_actions().map(G::action_index).collect()
}

#[test]
fn othello_uses_stable_square_and_pass_actions() {
    let board = othello::Board::default();
    assert_eq!(action_indices(&board), [19, 26, 37, 44]);

    for index in 0..othello::Board::ACTION_COUNT {
        let action = othello::Move::from_index(index).unwrap();
        assert_eq!(othello::Board::action_index(action), index);
    }
    assert_eq!(othello::Move::from_index(65), None);

    let a1 = othello::Square::new(0, 0).unwrap().bitboard();
    let b1 = othello::Square::new(1, 0).unwrap().bitboard();
    let pass = othello::Board::from_discs(a1, b1, othello::Color::White).unwrap();
    assert_eq!(action_indices(&pass), [64]);
    assert_eq!(othello::Move::from_index(64), Some(othello::Move::Pass));
}

#[test]
fn othello_terminal_outcome_is_side_relative() {
    let loss = othello::Board::from_discs(
        othello::BitBoard::FULL,
        othello::BitBoard::EMPTY,
        othello::Color::White,
    )
    .unwrap();

    assert_eq!(Game::outcome(&loss), Some(Outcome::Loss));
    assert!(Game::legal_actions(&loss).next().is_none());

    let win = othello::Board::from_discs(
        othello::BitBoard::FULL,
        othello::BitBoard::EMPTY,
        othello::Color::Black,
    )
    .unwrap();
    assert_eq!(Game::outcome(&win), Some(Outcome::Win));

    let draw = othello::Board::from_discs(
        othello::BitBoard(0xaaaa_aaaa_aaaa_aaaa),
        othello::BitBoard(0x5555_5555_5555_5555),
        othello::Color::Black,
    )
    .unwrap();
    assert_eq!(Game::outcome(&draw), Some(Outcome::Draw));
}

#[test]
fn othello_search_expands_and_advances_through_a_forced_pass() {
    let a1 = othello::Square::new(0, 0).unwrap().bitboard();
    let b1 = othello::Square::new(1, 0).unwrap().bitboard();
    let board = othello::Board::from_discs(a1, b1, othello::Color::White).unwrap();
    let mut tree = Mcts::new(board);

    tree.run(&mut UniformEvaluator, 8).unwrap();
    assert_eq!(tree.best_action(), Some(othello::Move::Pass));
    let pass = tree.root_stats().next().unwrap();
    assert!(pass.value < -0.8, "{pass:?}");
    assert!(tree.advance(othello::Move::Pass));
    assert_eq!(tree.root_position().side_to_move(), othello::Color::Black);
    assert_eq!(
        tree.root_position().legal_placements(),
        othello::Square::new(2, 0).unwrap().bitboard()
    );

    assert!(tree.advance(othello::Move::Place(othello::Square::new(2, 0).unwrap())));
    assert_eq!(tree.root_position().outcome(), Some(Outcome::Loss));
}
