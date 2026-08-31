//! Othello rules, move generation, and correctness tooling.

mod bitboard;
mod board;
mod encoding;
pub mod evaluation;
mod evaluator;
pub mod experiment;
mod model;
mod movegen;
mod perft;
mod replay;
mod square;
mod temporary;
pub mod training;

pub use crate::game::Color;
pub use bitboard::{BitBoard, BitBoardIter};
pub use board::{Board, BoardError, GameStatus, LegalActions, Move, ParseMoveError};
pub use evaluator::OthelloBurnEvaluator;
pub use model::{OthelloModelConfig, OthelloNetwork};
pub use perft::perft;
pub use square::{ParseSquareError, Square};

pub(crate) use encoding::OthelloEncoding;
