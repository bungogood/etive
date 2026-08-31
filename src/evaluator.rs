//! Position evaluation independent of any tensor backend.

use std::convert::Infallible;
#[cfg(feature = "cuda")]
use std::sync::OnceLock;

use burn::tensor::{Device, FloatDType, Tensor, TensorData, TensorReadError};

use crate::game::Game;
use crate::othello::{self, OthelloEncoding, OthelloNetwork};
#[cfg(test)]
use crate::tic_tac_toe::{self, minimax};

#[cfg(feature = "cuda")]
const INFERENCE_DTYPE: FloatDType = FloatDType::F16;
#[cfg(not(feature = "cuda"))]
const INFERENCE_DTYPE: FloatDType = FloatDType::F32;

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
}

impl<T> InferenceBatch<othello::Board, T> {
    pub(crate) fn pad_positions_to_capacity(&mut self) {
        let filler = *self
            .positions
            .last()
            .expect("cannot pad an empty inference batch");
        self.positions.resize(self.maximum, filler);
    }

    pub(crate) fn set_packed_results(&mut self, output: &[f32]) {
        self.policy_logits
            .resize(self.positions.len() * othello::Board::ACTION_COUNT, 0.0);
        self.values.resize(self.positions.len(), 0.0);
        unpack_othello_output(output, &mut self.policy_logits, &mut self.values);
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
struct EvaluationStream(cubecl_environment::stream::Stream);

#[cfg(feature = "cuda")]
impl EvaluationStream {
    fn shared() -> [Self; 3] {
        // CubeCL retains each stream's memory pool for the process lifetime.
        static STREAMS: OnceLock<[cubecl_environment::stream::Stream; 3]> = OnceLock::new();
        (*STREAMS
            .get_or_init(|| std::array::from_fn(|_| cubecl_environment::stream::Stream::new())))
        .map(Self)
    }

    fn enter<R>(&self, operation: impl FnOnce() -> R) -> R {
        self.0.enter(operation)
    }
}

#[cfg(not(feature = "cuda"))]
#[derive(Clone, Copy)]
struct EvaluationStream;

#[cfg(not(feature = "cuda"))]
impl EvaluationStream {
    const fn shared() -> [Self; 3] {
        [Self; 3]
    }

    fn enter<R>(&self, operation: impl FnOnce() -> R) -> R {
        operation()
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

/// Batched Burn evaluation for Othello search.
pub struct OthelloBurnEvaluator {
    network: OthelloNetwork,
    device: Device,
    inference_dtype: FloatDType,
    input: Vec<f32>,
    evaluations: u64,
    batches: u64,
    compute_streams: [EvaluationStream; 2],
    next_compute_stream: usize,
    upload_stream: EvaluationStream,
}

pub(crate) struct PendingOthelloInference {
    output: Tensor<2>,
    batch_size: usize,
    stream: EvaluationStream,
}

impl OthelloBurnEvaluator {
    pub fn new(device: Device, seed: u64) -> Self {
        let network = OthelloNetwork::new(&device, seed);
        Self::from_network(device, &network)
    }

    pub fn from_network(device: Device, network: &OthelloNetwork) -> Self {
        Self::from_network_with_dtype(device, network, INFERENCE_DTYPE)
    }

    pub fn from_network_with_dtype(
        device: Device,
        network: &OthelloNetwork,
        dtype: FloatDType,
    ) -> Self {
        let [compute_0, compute_1, upload] = EvaluationStream::shared();
        Self {
            network: network.detached().cast_float(dtype),
            device,
            inference_dtype: dtype,
            input: vec![0.0; OthelloEncoding::LEN],
            evaluations: 0,
            batches: 0,
            compute_streams: [compute_0, compute_1],
            next_compute_stream: 0,
            upload_stream: upload,
        }
    }

    pub const fn evaluations(&self) -> u64 {
        self.evaluations
    }

    pub const fn batches(&self) -> u64 {
        self.batches
    }

    pub(crate) fn start_batch(&mut self, games: &[othello::Board]) -> PendingOthelloInference {
        assert!(!games.is_empty(), "inference batch must not be empty");
        self.input.resize(games.len() * OthelloEncoding::LEN, 0.0);
        OthelloEncoding::encode_batch(games, &mut self.input);
        let batch_size = games.len();
        let input = self.upload_stream.enter(|| {
            let input =
                Tensor::<1>::from_data(TensorData::from(self.input.as_slice()), &self.device)
                    .reshape([batch_size, 2, 8, 8])
                    .cast(self.inference_dtype);
            self.device.flush();
            input
        });
        let stream = self.compute_streams[self.next_compute_stream];
        self.next_compute_stream = (self.next_compute_stream + 1) % self.compute_streams.len();
        let output = stream.enter(|| {
            let (policy, value) = self.network.forward(input);
            let output = Tensor::cat(vec![policy, value], 1);
            self.device.flush();
            output
        });
        PendingOthelloInference {
            output,
            batch_size,
            stream,
        }
    }

    pub(crate) fn finish_batch(
        &mut self,
        pending: PendingOthelloInference,
    ) -> Result<Vec<f32>, TensorReadError> {
        let data = pending
            .stream
            .enter(|| burn::tensor::read_sync(pending.output.into_data_async()))?;
        let output = data.try_into_vec_as::<f32>()?;
        self.evaluations += pending.batch_size as u64;
        self.batches += 1;
        Ok(output)
    }
}

impl BatchEvaluator<othello::Board> for OthelloBurnEvaluator {
    type Error = TensorReadError;

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

        let pending = self.start_batch(games);
        let output = self.finish_batch(pending)?;
        unpack_othello_output(&output, policy_logits, values);
        Ok(())
    }
}

fn unpack_othello_output(output: &[f32], policy_logits: &mut [f32], values: &mut [f32]) {
    let width = othello::Board::ACTION_COUNT + 1;
    assert_eq!(output.len(), values.len() * width);
    for (index, row) in output.chunks_exact(width).enumerate() {
        let start = index * othello::Board::ACTION_COUNT;
        policy_logits[start..start + othello::Board::ACTION_COUNT]
            .copy_from_slice(&row[..othello::Board::ACTION_COUNT]);
        values[index] = row[othello::Board::ACTION_COUNT];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "cuda")]
    use crate::othello::OthelloModelConfig;

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_evaluation_streams_are_reused() {
        let first = EvaluationStream::shared();
        let second = EvaluationStream::shared();

        assert_eq!(
            first.map(|stream| stream.0.id()),
            second.map(|stream| stream.0.id())
        );
        assert_ne!(first[0].0.id(), first[1].0.id());
        assert_ne!(first[0].0.id(), first[2].0.id());
        assert_ne!(first[1].0.id(), first[2].0.id());
    }

    #[test]
    fn inference_batch_padding_preserves_logical_length() {
        let mut batch = InferenceBatch::new(4);
        batch.push(7, othello::Board::default());

        batch.pad_positions_to_capacity();

        assert_eq!(batch.len(), 1);
        assert_eq!(batch.positions().len(), 4);
        assert_eq!(batch.tags().copied().collect::<Vec<_>>(), vec![7]);
    }

    #[test]
    fn othello_evaluator_batches_policy_and_value_outputs() {
        let games = [othello::Board::default(); 8];
        let mut evaluator = OthelloBurnEvaluator::new(Device::flex(), 7);
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

    #[cfg(feature = "cuda")]
    #[test]
    fn othello_cuda_evaluator_large_batch_outputs_are_finite() {
        let games = [othello::Board::default(); 1024];
        let device = Device::cuda(0);
        let network = OthelloNetwork::new(&device, 7);
        let mut evaluator = OthelloBurnEvaluator::from_network(device, &network);
        let mut policies = vec![0.0; games.len() * othello::Board::ACTION_COUNT];
        let mut values = vec![0.0; games.len()];

        evaluator
            .evaluate_batch(&games, &mut policies, &mut values)
            .unwrap();

        assert!(policies.into_iter().all(f32::is_finite));
        assert!(values.into_iter().all(f32::is_finite));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn othello_cuda_evaluator_pipelined_weekend_batches_are_finite() {
        let games = [othello::Board::default(); 1024];
        let device = Device::cuda(0);
        let network = OthelloNetwork::new_with_config(&device, 7, OthelloModelConfig::WEEKEND);
        let mut evaluator = OthelloBurnEvaluator::from_network(device, &network);

        for iteration in 0..16 {
            let first = evaluator.start_batch(&games);
            let second = evaluator.start_batch(&games);

            for (batch, pending) in [("first", first), ("second", second)] {
                let output = evaluator.finish_batch(pending).unwrap();
                assert!(
                    output.iter().all(|value| value.is_finite()),
                    "{batch} batch in iteration {iteration} contained a non-finite value",
                );
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn othello_cuda_evaluator_serial_weekend_batches_are_finite() {
        let games = [othello::Board::default(); 1024];
        let device = Device::cuda(0);
        let network = OthelloNetwork::new_with_config(&device, 7, OthelloModelConfig::WEEKEND);
        let mut evaluator = OthelloBurnEvaluator::from_network(device, &network);

        for iteration in 0..16 {
            let pending = evaluator.start_batch(&games);
            let output = evaluator.finish_batch(pending).unwrap();
            assert!(
                output.iter().all(|value| value.is_finite()),
                "batch in iteration {iteration} contained a non-finite value",
            );
        }
    }
}
