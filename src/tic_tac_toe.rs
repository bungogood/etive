//! Tic-tac-toe rules and an exact minimax oracle.

use std::iter::FusedIterator;
use std::ops::Not;

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

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Square(u8);

impl Square {
    pub const COUNT: usize = 9;

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

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Board {
    player: u16,
    opponent: u16,
    side_to_move: Player,
}

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
        assert!(self.is_legal(square), "attempted to play an illegal move");
        self.play_unchecked(square);
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

    fn apply(&mut self, action: Self::Action) {
        self.play_unchecked(action);
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
