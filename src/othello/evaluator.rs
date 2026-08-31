use burn::tensor::{Device, FloatDType, Tensor, TensorData, TensorReadError};
#[cfg(feature = "cuda")]
use std::sync::OnceLock;

use super::{Board, OthelloEncoding, OthelloNetwork};
use crate::evaluator::BatchEvaluator;
use crate::game::Game;

#[cfg(feature = "cuda")]
const INFERENCE_DTYPE: FloatDType = FloatDType::F16;
#[cfg(not(feature = "cuda"))]
const INFERENCE_DTYPE: FloatDType = FloatDType::F32;

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

    pub(crate) fn start_batch(&mut self, games: &[Board]) -> PendingOthelloInference {
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

impl BatchEvaluator<Board> for OthelloBurnEvaluator {
    type Error = TensorReadError;

    fn evaluate_batch(
        &mut self,
        games: &[Board],
        policy_logits: &mut [f32],
        values: &mut [f32],
    ) -> Result<(), Self::Error> {
        assert_eq!(policy_logits.len(), games.len() * Board::ACTION_COUNT);
        assert_eq!(values.len(), games.len());
        if games.is_empty() {
            return Ok(());
        }

        let pending = self.start_batch(games);
        let output = self.finish_batch(pending)?;
        let width = Board::ACTION_COUNT + 1;
        assert_eq!(output.len(), values.len() * width);
        for (index, row) in output.chunks_exact(width).enumerate() {
            let start = index * Board::ACTION_COUNT;
            policy_logits[start..start + Board::ACTION_COUNT]
                .copy_from_slice(&row[..Board::ACTION_COUNT]);
            values[index] = row[Board::ACTION_COUNT];
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "cuda")]
    use super::super::OthelloModelConfig;
    use super::*;

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
    fn batches_policy_and_value_outputs() {
        let games = [Board::default(); 8];
        let mut evaluator = OthelloBurnEvaluator::new(Device::flex(), 7);
        let mut policies = vec![0.0; games.len() * Board::ACTION_COUNT];
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
    fn large_batch_outputs_are_finite() {
        let games = [Board::default(); 1024];
        let device = Device::cuda(0);
        let network = OthelloNetwork::new(&device, 7);
        let mut evaluator = OthelloBurnEvaluator::from_network(device, &network);
        let mut policies = vec![0.0; games.len() * Board::ACTION_COUNT];
        let mut values = vec![0.0; games.len()];
        evaluator
            .evaluate_batch(&games, &mut policies, &mut values)
            .unwrap();
        assert!(policies.into_iter().all(f32::is_finite));
        assert!(values.into_iter().all(f32::is_finite));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn pipelined_weekend_batches_are_finite() {
        let games = [Board::default(); 1024];
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
    fn serial_weekend_batches_are_finite() {
        let games = [Board::default(); 1024];
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
