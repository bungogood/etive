use std::iter::FusedIterator;

use crate::game::{Color, Game, Outcome};

const FULL: u16 = 0x01ff;
const WINS: [u16; 8] = [0x007, 0x038, 0x1c0, 0x049, 0x092, 0x124, 0x111, 0x054];

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
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

    pub(crate) const fn index(self) -> usize {
        self.0 as usize
    }

    const fn bit(self) -> u16 {
        1 << self.0
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Board {
    player: u16,
    opponent: u16,
    side_to_move: Color,
}

impl Board {
    pub fn play(&mut self, square: Square) {
        assert!(self.legal_actions().any(|legal| legal == square));
        self.play_unchecked(square);
    }

    fn legal_actions(self) -> LegalActions {
        if self.outcome().is_some() {
            LegalActions(0)
        } else {
            LegalActions(FULL & !(self.player | self.opponent))
        }
    }

    fn play_unchecked(&mut self, square: Square) {
        let next_player = self.opponent;
        self.opponent = self.player | square.bit();
        self.player = next_player;
        self.side_to_move = !self.side_to_move;
    }

    fn outcome(self) -> Option<Outcome> {
        if has_won(self.opponent) {
            Some(Outcome::Loss)
        } else if self.player | self.opponent == FULL {
            Some(Outcome::Draw)
        } else {
            None
        }
    }
}

fn has_won(marks: u16) -> bool {
    for win in WINS {
        if marks & win == win {
            return true;
        }
    }
    false
}

#[derive(Clone)]
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

    fn side_to_move(&self) -> Color {
        self.side_to_move
    }

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

pub fn minimax(board: &Board) -> Outcome {
    if let Some(outcome) = board.outcome() {
        return outcome;
    }

    board
        .legal_actions()
        .map(|action| {
            let mut child = *board;
            child.play_unchecked(action);
            minimax(&child).reversed()
        })
        .max_by_key(|outcome| match outcome {
            Outcome::Loss => 0,
            Outcome::Draw => 1,
            Outcome::Win => 2,
        })
        .unwrap()
}
