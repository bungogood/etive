//! Othello compatibility adapter for generic self-play execution.

use burn::tensor::TensorReadError;

pub use super::replay::SelfPlaySample;
use super::{Board, OthelloBurnEvaluator};
use crate::self_play;

pub type ActorConfig = self_play::Config;
pub type ActorError = self_play::Error<TensorReadError>;

#[derive(Debug)]
pub struct ActorRun {
    pub evaluations: u64,
    pub inference_batches: u64,
    pub unique_games: usize,
    pub samples: Vec<SelfPlaySample>,
}

pub fn run(evaluator: OthelloBurnEvaluator, config: ActorConfig) -> Result<ActorRun, ActorError> {
    let result = self_play::run::<Board, _>(evaluator, config)?;
    Ok(ActorRun {
        evaluations: result.evaluations,
        inference_batches: result.inference_batches,
        unique_games: result.unique_games,
        samples: result
            .samples
            .into_iter()
            .map(|sample| SelfPlaySample {
                position: sample.position,
                policy: sample.policy,
                outcome: sample.outcome,
                game: sample.game,
            })
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use burn::tensor::Device;

    use super::*;
    use crate::game::Game;

    #[test]
    fn actor_workers_complete_games_through_one_inference_owner() {
        let evaluator = OthelloBurnEvaluator::new(Device::flex(), 7);
        let result = run(
            evaluator,
            ActorConfig {
                games: 2,
                simulations: 2,
                workers: 1,
                inference_batch_size: 8,
                seed: 11,
                dirichlet_alpha: 0.3,
                dirichlet_fraction: 0.25,
                temperature_moves: 20,
            },
        )
        .unwrap();

        assert!(result.evaluations > 0);
        assert!(result.unique_games > 1);
        assert!(!result.samples.is_empty());
        for sample in result.samples {
            assert!((sample.policy.iter().sum::<f32>() - 1.0).abs() < 1e-5);
            for (index, probability) in sample.policy.into_iter().enumerate() {
                if probability > 0.0 {
                    let action = Board::action_from_index(index).unwrap();
                    assert!(sample.position.is_legal(action));
                }
            }
        }
    }
}
