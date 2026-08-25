//! Othello rules, move generation, and correctness tooling.

use std::iter::FusedIterator;

pub mod actors;
mod bitboard;
mod board;
pub mod experiment;
mod movegen;
mod perft;
mod square;
pub mod training;

pub use bitboard::{BitBoard, BitBoardIter};
pub use board::{Board, BoardError, Color, GameStatus, Move, ParseBoardError};
pub use perft::perft;
pub use square::{ParseSquareError, Square};

use crate::game::{Game, Outcome};

struct LegalActions {
    placements: BitBoardIter,
    pass: bool,
}

impl Iterator for LegalActions {
    type Item = Move;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(square) = self.placements.next() {
            return Some(Move::Place(square));
        }
        if self.pass {
            self.pass = false;
            return Some(Move::Pass);
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.placements.len() + usize::from(self.pass);
        (len, Some(len))
    }
}

impl ExactSizeIterator for LegalActions {}
impl FusedIterator for LegalActions {}

impl Game for Board {
    type Action = Move;

    const ACTION_COUNT: usize = 65;

    fn legal_actions(&self) -> impl ExactSizeIterator<Item = Self::Action> + '_ {
        LegalActions {
            placements: self.legal_moves().into_iter(),
            pass: self.is_pass_legal(),
        }
    }

    fn action_index(action: Self::Action) -> usize {
        match action {
            Move::Place(square) => square.index(),
            Move::Pass => 64,
        }
    }

    fn action_from_index(index: usize) -> Option<Self::Action> {
        match index {
            0..64 => Square::from_index(index as u8).map(Move::Place),
            64 => Some(Move::Pass),
            _ => None,
        }
    }

    fn apply(&mut self, action: Self::Action) {
        self.play_unchecked(action);
    }

    fn outcome(&self) -> Option<Outcome> {
        match self.status() {
            GameStatus::Ongoing => None,
            GameStatus::Drawn => Some(Outcome::Draw),
            GameStatus::Won(color) if color == self.side_to_move() => Some(Outcome::Win),
            GameStatus::Won(_) => Some(Outcome::Loss),
        }
    }
}
