//! Tic-tac-toe rules and an exact minimax oracle.

use std::fmt;
use std::iter::FusedIterator;
use std::ops::Not;
use std::str::FromStr;

use crate::game::{Game, Outcome};

const FULL: u16 = 0x01ff;
const WINS: [u16; 8] = [
    0x007, 0x038, 0x1c0, // rows
    0x049, 0x092, 0x124, // columns
    0x111, 0x054, // diagonals
];

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Player {
    X,
    O,
}

impl Not for Player {
    type Output = Self;

    fn not(self) -> Self::Output {
        match self {
            Self::X => Self::O,
            Self::O => Self::X,
        }
    }
}

#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Square(u8);

impl Square {
    pub const COUNT: usize = 9;
    pub const ALL: [Self; Self::COUNT] = all_squares();

    pub const fn new(file: u8, rank: u8) -> Option<Self> {
        if file < 3 && rank < 3 {
            Some(Self(rank * 3 + file))
        } else {
            None
        }
    }

    pub const fn from_index(index: usize) -> Option<Self> {
        if index < Self::COUNT {
            Some(Self(index as u8))
        } else {
            None
        }
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }

    const fn bit(self) -> u16 {
        1 << self.0
    }
}

const fn all_squares() -> [Square; Square::COUNT] {
    let mut squares = [Square(0); Square::COUNT];
    let mut index = 0;
    while index < Square::COUNT {
        squares[index] = Square(index as u8);
        index += 1;
    }
    squares
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParseSquareError;

impl fmt::Display for ParseSquareError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("square must be a coordinate from a1 through c3")
    }
}

impl std::error::Error for ParseSquareError {}

impl FromStr for Square {
    type Err = ParseSquareError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let bytes = value.as_bytes();
        if bytes.len() != 2 {
            return Err(ParseSquareError);
        }
        let file = bytes[0].to_ascii_lowercase().wrapping_sub(b'a');
        let rank = bytes[1].wrapping_sub(b'1');
        Self::new(file, rank).ok_or(ParseSquareError)
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let file = char::from(b'a' + self.0 % 3);
        let rank = char::from(b'1' + self.0 / 3);
        write!(f, "{file}{rank}")
    }
}

impl fmt::Debug for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Board {
    player: u16,
    opponent: u16,
    side_to_move: Player,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BoardError {
    IllegalMove,
}

impl fmt::Display for BoardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IllegalMove => f.write_str("illegal move"),
        }
    }
}

impl std::error::Error for BoardError {}

impl Default for Board {
    fn default() -> Self {
        Self {
            player: 0,
            opponent: 0,
            side_to_move: Player::X,
        }
    }
}

impl Board {
    pub const fn side_to_move(self) -> Player {
        self.side_to_move
    }

    pub const fn marks(self, player: Player) -> u16 {
        if player as u8 == self.side_to_move as u8 {
            self.player
        } else {
            self.opponent
        }
    }

    pub fn legal_actions(self) -> LegalActions {
        if self.outcome().is_some() {
            LegalActions(0)
        } else {
            LegalActions(FULL & !(self.player | self.opponent))
        }
    }

    pub fn is_legal(self, square: Square) -> bool {
        self.legal_actions().any(|legal| legal == square)
    }

    pub fn play(&mut self, square: Square) {
        self.try_play(square)
            .expect("attempted to play an illegal move");
    }

    pub fn try_play(&mut self, square: Square) -> Result<(), BoardError> {
        if !self.is_legal(square) {
            return Err(BoardError::IllegalMove);
        }
        self.play_unchecked(square);
        Ok(())
    }

    pub fn play_unchecked(&mut self, square: Square) {
        debug_assert!(self.is_legal(square), "attempted to play an illegal move");
        let next_player = self.opponent;
        let next_opponent = self.player | square.bit();
        self.player = next_player;
        self.opponent = next_opponent;
        self.side_to_move = !self.side_to_move;
    }

    pub fn outcome(self) -> Option<Outcome> {
        if has_won(self.opponent) {
            Some(Outcome::Loss)
        } else if self.player | self.opponent == FULL {
            Some(Outcome::Draw)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct LegalActions(u16);

impl Iterator for LegalActions {
    type Item = Square;

    fn next(&mut self) -> Option<Self::Item> {
        if self.0 == 0 {
            return None;
        }
        let index = self.0.trailing_zeros() as u8;
        self.0 &= self.0 - 1;
        Some(Square(index))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.0.count_ones() as usize;
        (len, Some(len))
    }
}

impl ExactSizeIterator for LegalActions {}
impl FusedIterator for LegalActions {}

impl Game for Board {
    type Action = Square;

    const ACTION_COUNT: usize = Square::COUNT;

    fn legal_actions(&self) -> impl ExactSizeIterator<Item = Self::Action> + '_ {
        (*self).legal_actions()
    }

    fn action_index(action: Self::Action) -> usize {
        action.index()
    }

    fn action_from_index(index: usize) -> Option<Self::Action> {
        Square::from_index(index)
    }

    fn play_unchecked(&mut self, action: Self::Action) {
        Board::play_unchecked(self, action);
    }

    fn outcome(&self) -> Option<Outcome> {
        (*self).outcome()
    }
}

/// Solves a position exactly and returns its side-to-move-relative result.
pub fn minimax(board: &Board) -> Outcome {
    if let Some(outcome) = board.outcome() {
        return outcome;
    }

    let mut best = Outcome::Loss;
    for action in board.legal_actions() {
        let mut child = *board;
        child.play_unchecked(action);
        let outcome = minimax(&child).reversed();
        if outcome == Outcome::Win {
            return outcome;
        }
        if outcome == Outcome::Draw {
            best = outcome;
        }
    }
    best
}

const fn has_won(marks: u16) -> bool {
    let mut index = 0;
    while index < WINS.len() {
        let win = WINS[index];
        if marks & win == win {
            return true;
        }
        index += 1;
    }
    false
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    fn square(index: usize) -> Square {
        Square::from_index(index).unwrap()
    }

    #[test]
    fn empty_board_is_an_exact_draw() {
        assert_eq!(minimax(&Board::default()), Outcome::Draw);
    }

    #[test]
    fn minimax_finds_an_immediate_win() {
        let mut board = Board::default();
        for action in [0, 3, 1, 4] {
            board.play(square(action));
        }

        assert_eq!(board.side_to_move(), Player::X);
        assert_eq!(minimax(&board), Outcome::Win);
        board.play(square(2));
        assert_eq!(board.outcome(), Some(Outcome::Loss));
    }

    #[test]
    fn all_reachable_positions_have_consistent_actions() {
        fn visit(board: Board, positions: &mut HashSet<Board>) {
            if !positions.insert(board) {
                return;
            }
            let actions = board.legal_actions();
            assert_eq!(actions.len(), actions.clone().count());
            for action in actions {
                assert_eq!(
                    Board::action_from_index(Board::action_index(action)),
                    Some(action)
                );
                let mut child = board;
                child.play_unchecked(action);
                visit(child, positions);
            }
        }

        let mut positions = HashSet::new();
        visit(Board::default(), &mut positions);
        assert_eq!(positions.len(), 5_478);
    }
}
