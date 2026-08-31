//! Othello rules, move generation, and correctness tooling.

pub mod actors;
mod bitboard;
mod board;
mod diagnostics;
pub mod evaluation;
pub mod experiment;
mod frozen_training;
mod movegen;
mod perft;
mod replay;
mod square;
pub mod training;

pub use crate::game::Color;
pub use bitboard::{BitBoard, BitBoardIter};
pub use board::{
    Board, BoardError, GameStatus, LegalActions, Move, ParseBoardError, ParseMoveError,
};
pub use diagnostics::{DiagnosticsReport, diagnose_replay};
pub use frozen_training::{FrozenTrainingConfig, FrozenTrainingReport, train_frozen};
pub use perft::perft;
pub use square::{ParseSquareError, Square};

use crate::game::{Game, Outcome};

impl Game for Board {
    type Action = Move;

    const ACTION_COUNT: usize = Square::COUNT + 1;

    fn side_to_move(&self) -> Color {
        Board::side_to_move(*self)
    }

    fn legal_actions(&self) -> impl ExactSizeIterator<Item = Self::Action> + '_ {
        (*self).legal_actions()
    }

    fn action_index(action: Self::Action) -> usize {
        match action {
            Move::Place(square) => square.index(),
            Move::Pass => Square::COUNT,
        }
    }

    fn action_from_index(index: usize) -> Option<Self::Action> {
        match index {
            0..Square::COUNT => Square::from_index(index).map(Move::Place),
            Square::COUNT => Some(Move::Pass),
            _ => None,
        }
    }

    fn play_unchecked(&mut self, action: Self::Action) {
        Board::play_unchecked(self, action);
    }

    fn outcome(&self) -> Option<Outcome> {
        (*self).outcome()
    }
}
