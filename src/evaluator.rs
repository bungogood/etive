//! Position evaluation independent of any tensor backend.

use std::convert::Infallible;

use crate::game::Game;
#[cfg(test)]
use crate::tic_tac_toe::{self, minimax};

/// Evaluates many independent positions in one model invocation.
pub trait BatchEvaluator<G: Game> {
    type Error;

    /// Writes row-major policy logits and one value per input position.
    fn evaluate_batch(
        &mut self,
        games: &[G],
        policy_logits: &mut [f32],
        values: &mut [f32],
    ) -> Result<(), Self::Error>;
}

/// Reports the cumulative number of positions evaluated.
pub trait EvaluationCounter {
    fn evaluations(&self) -> u64;
}

/// Starts model work separately from collecting its packed policy/value output.
pub trait PipelinedEvaluator<G: Game> {
    type Error;
    type Pending;

    fn start_batch(&mut self, games: &[G]) -> Self::Pending;

    /// Returns rows packed as `ACTION_COUNT` policy logits followed by one value.
    fn finish_batch(&mut self, pending: Self::Pending) -> Result<Vec<f32>, Self::Error>;
}

/// Reusable contiguous storage for one bounded inference batch.
pub(crate) struct InferenceBatch<G: Game, T> {
    maximum: usize,
    tags: Vec<T>,
    positions: Vec<G>,
    policy_logits: Vec<f32>,
    values: Vec<f32>,
}

impl<G: Game, T> InferenceBatch<G, T> {
    pub(crate) fn new(maximum: usize) -> Self {
        assert!(maximum > 0, "inference batch size must be positive");
        Self {
            maximum,
            tags: Vec::with_capacity(maximum),
            positions: Vec::with_capacity(maximum),
            policy_logits: Vec::with_capacity(maximum * G::ACTION_COUNT),
            values: Vec::with_capacity(maximum),
        }
    }

    pub(crate) fn push(&mut self, tag: T, position: G) {
        assert!(!self.is_full(), "inference batch is full");
        self.tags.push(tag);
        self.positions.push(position);
    }

    pub(crate) fn clear(&mut self) {
        self.tags.clear();
        self.positions.clear();
    }

    pub(crate) fn len(&self) -> usize {
        self.tags.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.tags.is_empty()
    }

    pub(crate) fn is_full(&self) -> bool {
        self.tags.len() == self.maximum
    }

    pub(crate) const fn capacity(&self) -> usize {
        self.maximum
    }

    pub(crate) fn tags(&self) -> impl DoubleEndedIterator<Item = &T> + ExactSizeIterator {
        self.tags.iter()
    }

    pub(crate) fn positions(&self) -> &[G] {
        &self.positions
    }

    pub(crate) fn evaluate_batch<E: BatchEvaluator<G>>(
        &mut self,
        evaluator: &mut E,
    ) -> Result<(), E::Error> {
        self.policy_logits
            .resize(self.positions.len() * G::ACTION_COUNT, 0.0);
        self.values.resize(self.positions.len(), 0.0);
        evaluator.evaluate_batch(&self.positions, &mut self.policy_logits, &mut self.values)
    }

    pub(crate) fn result(&self, index: usize) -> (&T, &[f32], f32) {
        let start = index * G::ACTION_COUNT;
        (
            &self.tags[index],
            &self.policy_logits[start..start + G::ACTION_COUNT],
            self.values[index],
        )
    }

    pub(crate) fn pad_positions_to_capacity(&mut self) {
        let filler = *self
            .positions
            .last()
            .expect("cannot pad an empty inference batch");
        self.positions.resize(self.maximum, filler);
    }

    pub(crate) fn set_packed_results(&mut self, output: &[f32]) {
        self.policy_logits
            .resize(self.positions.len() * G::ACTION_COUNT, 0.0);
        self.values.resize(self.positions.len(), 0.0);
        let width = G::ACTION_COUNT + 1;
        assert_eq!(output.len(), self.values.len() * width);
        for (index, row) in output.chunks_exact(width).enumerate() {
            let start = index * G::ACTION_COUNT;
            self.policy_logits[start..start + G::ACTION_COUNT]
                .copy_from_slice(&row[..G::ACTION_COUNT]);
            self.values[index] = row[G::ACTION_COUNT];
        }
    }
}

/// A deterministic baseline with equal logits and zero value.
#[derive(Clone, Copy, Debug, Default)]
pub struct UniformEvaluator;

impl<G: Game> BatchEvaluator<G> for UniformEvaluator {
    type Error = Infallible;

    fn evaluate_batch(
        &mut self,
        games: &[G],
        policy_logits: &mut [f32],
        values: &mut [f32],
    ) -> Result<(), Self::Error> {
        assert_eq!(policy_logits.len(), games.len() * G::ACTION_COUNT);
        assert_eq!(values.len(), games.len());
        policy_logits.fill(0.0);
        values.fill(0.0);
        Ok(())
    }
}

/// Exact tic-tac-toe values with logits biased toward optimal actions.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct TicTacToeMinimaxEvaluator;

#[cfg(test)]
impl BatchEvaluator<tic_tac_toe::Board> for TicTacToeMinimaxEvaluator {
    type Error = Infallible;

    fn evaluate_batch(
        &mut self,
        games: &[tic_tac_toe::Board],
        policy_logits: &mut [f32],
        values: &mut [f32],
    ) -> Result<(), Self::Error> {
        assert_eq!(
            policy_logits.len(),
            games.len() * tic_tac_toe::Board::ACTION_COUNT
        );
        assert_eq!(values.len(), games.len());
        for (index, game) in games.iter().enumerate() {
            let start = index * tic_tac_toe::Board::ACTION_COUNT;
            let policy = &mut policy_logits[start..start + tic_tac_toe::Board::ACTION_COUNT];
            policy.fill(0.0);
            for action in game.legal_actions() {
                let mut child = *game;
                child.play_unchecked(action);
                policy[action.index()] = 4.0 * minimax(&child).reversed().value();
            }
            values[index] = minimax(game).value();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inference_batch_padding_preserves_logical_length() {
        let mut batch = InferenceBatch::new(4);
        batch.push(7, tic_tac_toe::Board::default());

        batch.pad_positions_to_capacity();

        assert_eq!(batch.len(), 1);
        assert_eq!(batch.positions().len(), 4);
        assert_eq!(batch.tags().copied().collect::<Vec<_>>(), vec![7]);
    }
}
