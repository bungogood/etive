//! Shared contracts for deterministic two-player games.

use std::ops::Not;

/// The two players in an alternating board game.
#[derive(bincode::Decode, bincode::Encode, Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum Color {
    #[default]
    Black,
    White,
}

impl Not for Color {
    type Output = Self;

    fn not(self) -> Self::Output {
        match self {
            Self::Black => Self::White,
            Self::White => Self::Black,
        }
    }
}

/// A terminal result from the perspective of the player to move.
#[derive(bincode::Decode, bincode::Encode, Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Outcome {
    Win,
    Draw,
    Loss,
}

impl Outcome {
    /// Returns the same result from the opposing player's perspective.
    pub const fn reversed(self) -> Self {
        match self {
            Self::Win => Self::Loss,
            Self::Draw => Self::Draw,
            Self::Loss => Self::Win,
        }
    }

    /// Converts the result to a side-to-move-relative scalar value.
    pub const fn value(self) -> f32 {
        match self {
            Self::Win => 1.0,
            Self::Draw => 0.0,
            Self::Loss => -1.0,
        }
    }
}

/// A deterministic, alternating, two-player, zero-sum game.
pub trait Game: Copy + Send + Sync + 'static {
    type Action: Copy + Eq + Send + Sync;
    type Policy: Clone + Send + Sync + AsRef<[f32]> + AsMut<[f32]> + 'static;

    /// Number of stable policy outputs used by the game.
    const ACTION_COUNT: usize;

    /// Creates zero-filled storage for one policy row.
    fn zero_policy() -> Self::Policy;

    /// Returns the player who will take the next action.
    fn side_to_move(&self) -> Color;

    /// Iterates over legal actions without allocating.
    fn legal_actions(&self) -> impl ExactSizeIterator<Item = Self::Action> + '_;

    /// Maps an action to its stable policy output index.
    fn action_index(action: Self::Action) -> usize;

    /// Maps a policy output index back to an action.
    fn action_from_index(index: usize) -> Option<Self::Action>;

    /// Plays an action already obtained from [`Game::legal_actions`].
    fn play_unchecked(&mut self, action: Self::Action);

    /// Returns a terminal result relative to the player to move.
    fn outcome(&self) -> Option<Outcome>;
}
