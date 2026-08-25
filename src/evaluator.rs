//! Position evaluation independent of any tensor backend.

use std::convert::Infallible;

use candle_core::{Device, Tensor};

use crate::encoding::{OthelloEncodingV1, StateEncoder, TicTacToeEncodingV1};
use crate::game::Game;
use crate::model::{OthelloNetwork, TicTacToeNetwork};
use crate::othello;
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

/// Reusable contiguous storage for one bounded inference batch.
pub struct InferenceBatch<G: Game, T> {
    maximum: usize,
    tags: Vec<T>,
    positions: Vec<G>,
    policy_logits: Vec<f32>,
    values: Vec<f32>,
}

impl<G: Game, T> InferenceBatch<G, T> {
    pub fn new(maximum: usize) -> Self {
        assert!(maximum > 0, "inference batch size must be positive");
        Self {
            maximum,
            tags: Vec::with_capacity(maximum),
            positions: Vec::with_capacity(maximum),
            policy_logits: Vec::with_capacity(maximum * G::ACTION_COUNT),
            values: Vec::with_capacity(maximum),
        }
    }

    pub fn push(&mut self, tag: T, position: G) -> bool {
        if self.is_full() {
            return false;
        }
        self.tags.push(tag);
        self.positions.push(position);
        true
    }

    pub fn clear(&mut self) {
        self.tags.clear();
        self.positions.clear();
    }

    pub fn len(&self) -> usize {
        self.tags.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tags.is_empty()
    }

    pub fn is_full(&self) -> bool {
        self.tags.len() == self.maximum
    }

    pub fn tags(&self) -> impl DoubleEndedIterator<Item = &T> + ExactSizeIterator {
        self.tags.iter()
    }

    pub fn evaluate_batch<E: BatchEvaluator<G>>(
        &mut self,
        evaluator: &mut E,
    ) -> Result<(), E::Error> {
        self.policy_logits
            .resize(self.positions.len() * G::ACTION_COUNT, 0.0);
        self.values.resize(self.positions.len(), 0.0);
        evaluator.evaluate_batch(&self.positions, &mut self.policy_logits, &mut self.values)
    }

    pub fn result(&self, index: usize) -> (&T, &[f32], f32) {
        let start = index * G::ACTION_COUNT;
        (
            &self.tags[index],
            &self.policy_logits[start..start + G::ACTION_COUNT],
            self.values[index],
        )
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
#[derive(Clone, Copy, Debug, Default)]
pub struct TicTacToeMinimaxEvaluator;

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

/// Batched Candle evaluation for tic-tac-toe tests.
pub struct TicTacToeCandleEvaluator {
    network: TicTacToeNetwork,
    device: Device,
    input: Vec<f32>,
    evaluations: u64,
    batches: u64,
}

impl TicTacToeCandleEvaluator {
    pub fn new(device: Device, seed: u64) -> candle_core::Result<Self> {
        let network = TicTacToeNetwork::new(&device, seed)?;
        Ok(Self::from_network(device, network))
    }

    pub fn from_network(device: Device, network: TicTacToeNetwork) -> Self {
        Self {
            network,
            device,
            input: vec![0.0; 18],
            evaluations: 0,
            batches: 0,
        }
    }

    pub const fn evaluations(&self) -> u64 {
        self.evaluations
    }

    pub const fn batches(&self) -> u64 {
        self.batches
    }
}

impl BatchEvaluator<tic_tac_toe::Board> for TicTacToeCandleEvaluator {
    type Error = candle_core::Error;

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
        if games.is_empty() {
            return Ok(());
        }

        self.input
            .resize(games.len() * TicTacToeEncodingV1::encoded_len(), 0.0);
        TicTacToeEncodingV1.encode_batch(games, &mut self.input);
        let input = Tensor::from_slice(&self.input, (games.len(), 2, 3, 3), &self.device)?;
        let (policy, value) = self.network.forward(&input)?;
        policy_logits.copy_from_slice(&policy.flatten_all()?.to_vec1::<f32>()?);
        values.copy_from_slice(&value.flatten_all()?.to_vec1::<f32>()?);
        self.evaluations += games.len() as u64;
        self.batches += 1;
        Ok(())
    }
}

/// Batched Candle evaluation for Othello search.
pub struct OthelloCandleEvaluator {
    network: OthelloNetwork,
    device: Device,
    input: Vec<f32>,
    evaluations: u64,
    batches: u64,
}

impl OthelloCandleEvaluator {
    pub fn new(device: Device, seed: u64) -> candle_core::Result<Self> {
        let network = OthelloNetwork::new(&device, seed)?;
        Ok(Self::from_network(device, &network))
    }

    pub fn from_network(device: Device, network: &OthelloNetwork) -> Self {
        Self {
            network: network.detached(),
            device,
            input: vec![0.0; OthelloEncodingV1::encoded_len()],
            evaluations: 0,
            batches: 0,
        }
    }

    pub const fn evaluations(&self) -> u64 {
        self.evaluations
    }

    pub const fn batches(&self) -> u64 {
        self.batches
    }
}

impl BatchEvaluator<othello::Board> for OthelloCandleEvaluator {
    type Error = candle_core::Error;

    fn evaluate_batch(
        &mut self,
        games: &[othello::Board],
        policy_logits: &mut [f32],
        values: &mut [f32],
    ) -> Result<(), Self::Error> {
        assert_eq!(
            policy_logits.len(),
            games.len() * othello::Board::ACTION_COUNT
        );
        assert_eq!(values.len(), games.len());
        if games.is_empty() {
            return Ok(());
        }

        self.input
            .resize(games.len() * OthelloEncodingV1::encoded_len(), 0.0);
        OthelloEncodingV1.encode_batch(games, &mut self.input);
        let input = Tensor::from_slice(&self.input, (games.len(), 2, 8, 8), &self.device)?;
        let (policy, value) = self.network.forward(&input)?;
        // One packed readback avoids synchronizing the accelerator separately
        // for policy and value tensors.
        let output = Tensor::cat(&[&policy, &value], 1)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let (rows, remainder) = output.as_chunks::<{ othello::Board::ACTION_COUNT + 1 }>();
        debug_assert!(remainder.is_empty());
        for (index, row) in rows.iter().enumerate() {
            let start = index * othello::Board::ACTION_COUNT;
            policy_logits[start..start + othello::Board::ACTION_COUNT]
                .copy_from_slice(&row[..othello::Board::ACTION_COUNT]);
            values[index] = row[othello::Board::ACTION_COUNT];
        }
        self.evaluations += games.len() as u64;
        self.batches += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn othello_evaluator_batches_policy_and_value_outputs() {
        let games = [othello::Board::default(); 8];
        let mut evaluator = OthelloCandleEvaluator::new(Device::Cpu, 7).unwrap();
        let mut policies = vec![0.0; games.len() * othello::Board::ACTION_COUNT];
        let mut values = vec![0.0; games.len()];

        evaluator
            .evaluate_batch(&games, &mut policies, &mut values)
            .unwrap();

        assert_eq!(evaluator.evaluations(), 8);
        assert_eq!(evaluator.batches(), 1);
        assert!(policies.into_iter().all(f32::is_finite));
        assert!(
            values
                .into_iter()
                .all(|value| value.is_finite() && (-1.0..=1.0).contains(&value))
        );
    }
}
