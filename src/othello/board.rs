use std::fmt;
use std::iter::FusedIterator;
use std::str::FromStr;

use crate::game::{Color, Game, Outcome};

use super::movegen;
use super::{BitBoard, BitBoardIter, Square};

const INITIAL_BLACK: u64 = 0x0000_0008_1000_0000;
const INITIAL_WHITE: u64 = 0x0000_0010_0800_0000;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Move {
    Place(Square),
    Pass,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParseMoveError;

impl fmt::Display for ParseMoveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("move must be pass or a square from a1 through h8")
    }
}

impl std::error::Error for ParseMoveError {}

impl FromStr for Move {
    type Err = ParseMoveError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.eq_ignore_ascii_case("pass") {
            Ok(Self::Pass)
        } else {
            value.parse().map(Self::Place).map_err(|_| ParseMoveError)
        }
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Place(square) => square.fmt(f),
            Self::Pass => f.write_str("pass"),
        }
    }
}

#[derive(Clone)]
pub struct LegalActions {
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GameStatus {
    Ongoing,
    Won(Color),
    Drawn,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BoardError {
    OverlappingDiscs,
    IllegalMove,
}

impl fmt::Display for BoardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OverlappingDiscs => f.write_str("black and white discs overlap"),
            Self::IllegalMove => f.write_str("illegal move"),
        }
    }
}

impl std::error::Error for BoardError {}

#[derive(bincode::Decode, bincode::Encode, Clone, Copy, Eq, Hash, PartialEq)]
pub struct Board {
    player: u64,
    opponent: u64,
    side_to_move: Color,
}

impl Default for Board {
    fn default() -> Self {
        Self {
            player: INITIAL_BLACK,
            opponent: INITIAL_WHITE,
            side_to_move: Color::Black,
        }
    }
}

impl Board {
    pub fn from_discs(
        black: BitBoard,
        white: BitBoard,
        side_to_move: Color,
    ) -> Result<Self, BoardError> {
        if black.0 & white.0 != 0 {
            return Err(BoardError::OverlappingDiscs);
        }
        let (player, opponent) = match side_to_move {
            Color::Black => (black.0, white.0),
            Color::White => (white.0, black.0),
        };
        Ok(Self {
            player,
            opponent,
            side_to_move,
        })
    }

    #[inline(always)]
    pub const fn discs(self, color: Color) -> BitBoard {
        if color as u8 == self.side_to_move as u8 {
            BitBoard(self.player)
        } else {
            BitBoard(self.opponent)
        }
    }

    #[inline(always)]
    pub fn legal_placements(self) -> BitBoard {
        BitBoard(movegen::legal_placements(self.player, self.opponent))
    }

    #[inline(always)]
    pub fn has_legal_placement(self) -> bool {
        movegen::legal_placements(self.player, self.opponent) != 0
    }

    /// Returns the discs that placing at `square` would flip.
    ///
    /// Occupied squares and placements that capture no discs return an empty
    /// bitboard.
    #[inline(always)]
    pub fn flips(self, square: Square) -> BitBoard {
        if (self.player | self.opponent) & square.bitboard().0 != 0 {
            return BitBoard::EMPTY;
        }
        BitBoard(movegen::flips(
            square.bitboard().0,
            self.player,
            self.opponent,
        ))
    }

    #[inline(always)]
    pub fn is_pass_legal(self) -> bool {
        !self.has_legal_placement() && movegen::legal_placements(self.opponent, self.player) != 0
    }

    pub fn status(self) -> GameStatus {
        if self.has_legal_placement() || movegen::legal_placements(self.opponent, self.player) != 0
        {
            return GameStatus::Ongoing;
        }
        let black = self.discs(Color::Black).len();
        let white = self.discs(Color::White).len();
        match black.cmp(&white) {
            std::cmp::Ordering::Greater => GameStatus::Won(Color::Black),
            std::cmp::Ordering::Less => GameStatus::Won(Color::White),
            std::cmp::Ordering::Equal => GameStatus::Drawn,
        }
    }

    #[inline(always)]
    pub fn is_legal(self, mv: Move) -> bool {
        match mv {
            Move::Place(square) => self.legal_placements().has(square),
            Move::Pass => self.is_pass_legal(),
        }
    }

    pub fn try_play(&mut self, mv: Move) -> Result<(), BoardError> {
        if !self.is_legal(mv) {
            return Err(BoardError::IllegalMove);
        }
        Game::play_unchecked(self, mv);
        Ok(())
    }

    pub fn play(&mut self, mv: Move) {
        self.try_play(mv)
            .expect("attempted to play an illegal move");
    }

    /// Passes without checking that a pass is currently legal.
    #[inline(always)]
    pub fn pass_unchecked(&mut self) {
        debug_assert!(self.is_pass_legal(), "attempted to play an illegal pass");
        self.swap_players();
    }

    #[inline(always)]
    fn swap_players(&mut self) {
        std::mem::swap(&mut self.player, &mut self.opponent);
        self.side_to_move = !self.side_to_move;
    }
}

impl Game for Board {
    type Action = Move;
    type Policy = [f32; 65];

    const ACTION_COUNT: usize = Square::COUNT + 1;

    fn zero_policy() -> Self::Policy {
        [0.0; 65]
    }

    fn side_to_move(&self) -> Color {
        self.side_to_move
    }

    fn legal_actions(&self) -> impl ExactSizeIterator<Item = Self::Action> + '_ {
        LegalActions {
            placements: self.legal_placements().into_iter(),
            pass: self.is_pass_legal(),
        }
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

    /// Plays a move without checking that it is legal.
    ///
    /// Invalid input is rejected in debug builds but may corrupt the logical
    /// board state in release builds.
    fn play_unchecked(&mut self, mv: Move) {
        debug_assert!(self.is_legal(mv), "attempted to play an illegal move");
        match mv {
            Move::Place(square) => {
                let placed = square.bitboard().0;
                let flips = movegen::flips(placed, self.player, self.opponent);
                let next_player = self.opponent & !flips;
                self.opponent = self.player | placed | flips;
                self.player = next_player;
                self.side_to_move = !self.side_to_move;
            }
            Move::Pass => self.swap_players(),
        }
    }

    fn outcome(&self) -> Option<Outcome> {
        match self.status() {
            GameStatus::Ongoing => None,
            GameStatus::Drawn => Some(Outcome::Draw),
            GameStatus::Won(color) if color == self.side_to_move => Some(Outcome::Win),
            GameStatus::Won(_) => Some(Outcome::Loss),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParseBoardError;

impl fmt::Display for ParseBoardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("expected eight 8-character rows using B, W, or . followed by side b or w")
    }
}

impl std::error::Error for ParseBoardError {}

impl FromStr for Board {
    type Err = ParseBoardError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let mut fields = value.split_ascii_whitespace();
        let placement = fields.next().ok_or(ParseBoardError)?;
        let side = fields.next().ok_or(ParseBoardError)?;
        if fields.next().is_some() {
            return Err(ParseBoardError);
        }

        let rows: Vec<_> = placement.split('/').collect();
        if rows.len() != 8 || rows.iter().any(|row| row.len() != 8) {
            return Err(ParseBoardError);
        }

        let mut black = 0_u64;
        let mut white = 0_u64;
        for (row_index, row) in rows.into_iter().enumerate() {
            let rank = 7 - row_index as u8;
            for (file, disc) in row.bytes().enumerate() {
                let bit = 1_u64 << (rank * 8 + file as u8);
                match disc {
                    b'B' | b'b' => black |= bit,
                    b'W' | b'w' => white |= bit,
                    b'.' => {}
                    _ => return Err(ParseBoardError),
                }
            }
        }

        let side_to_move = match side {
            "b" | "B" => Color::Black,
            "w" | "W" => Color::White,
            _ => return Err(ParseBoardError),
        };
        Board::from_discs(BitBoard(black), BitBoard(white), side_to_move)
            .map_err(|_| ParseBoardError)
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let black = self.discs(Color::Black);
        let white = self.discs(Color::White);
        for rank in (0..8).rev() {
            if rank != 7 {
                f.write_str("/")?;
            }
            for file in 0..8 {
                let square = Square::new(file, rank).unwrap();
                let disc = if black.has(square) {
                    'B'
                } else if white.has(square) {
                    'W'
                } else {
                    '.'
                };
                write!(f, "{disc}")?;
            }
        }
        let side = match self.side_to_move {
            Color::Black => 'b',
            Color::White => 'w',
        };
        write!(f, " {side}")
    }
}

impl fmt::Debug for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
