//! Othello rules, move generation, and correctness tooling.

mod bitboard;
mod board;
mod diagnostics;
mod encoding;
pub mod evaluation;
mod evaluator;
pub mod experiment;
mod frozen_training;
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
pub use diagnostics::{DiagnosticsReport, diagnose_replay};
pub use evaluator::OthelloBurnEvaluator;
pub use frozen_training::{FrozenTrainingConfig, FrozenTrainingReport, train_frozen};
pub use model::{OthelloModelConfig, OthelloNetwork};
pub use perft::perft;
pub use square::{ParseSquareError, Square};

pub(crate) use encoding::OthelloEncoding;
