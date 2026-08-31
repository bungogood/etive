use super::{Board, GameStatus, Move};
use crate::game::Game;

pub fn perft(board: &Board, depth: u8) -> u64 {
    if depth == 0 {
        return 1;
    }

    let moves = board.legal_placements();
    if moves.is_empty() {
        if board.status() != GameStatus::Ongoing {
            return 1;
        }
        let mut child = *board;
        child.pass_unchecked();
        return perft(&child, depth - 1);
    }
    if depth == 1 {
        return u64::from(moves.len());
    }

    moves
        .into_iter()
        .map(|square| {
            let mut child = *board;
            child.play_unchecked(Move::Place(square));
            perft(&child, depth - 1)
        })
        .sum()
}
