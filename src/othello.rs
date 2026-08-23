//! Othello rules, move generation, and correctness tooling.

mod bitboard;
mod board;
mod movegen;
mod perft;
mod square;

pub use bitboard::{BitBoard, BitBoardIter};
pub use board::{Board, BoardError, Color, GameStatus, Move, ParseBoardError};
pub use perft::perft;
pub use square::{ParseSquareError, Square};
