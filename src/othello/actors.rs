//! Persistent self-play actors feeding one batched inference owner.

use std::error::Error;
use std::fmt;
use std::thread;
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender, bounded};

use super::Board;
use crate::evaluator::{BatchEvaluator, OthelloCandleEvaluator};
use crate::game::{Game, Outcome};
use crate::mcts::{EvaluationRequest, Mcts, MctsConfig, MctsError, Selection};

#[derive(Clone, Copy, Debug)]
pub struct ActorConfig {
    pub games: usize,
    pub simulations: u32,
    pub workers: usize,
    pub inference_batch_size: usize,
}

#[derive(Debug)]
pub struct ActorRun {
    pub first_game_actions: Vec<usize>,
    pub draws: usize,
    pub evaluations: u64,
    pub batches: u64,
}

#[derive(Debug)]
pub enum ActorError {
    InvalidConfig,
    Evaluator(candle_core::Error),
    Mcts(MctsError),
    InferenceStopped,
    WorkerPanicked,
}

impl fmt::Display for ActorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig => formatter.write_str("actor configuration must be positive"),
            Self::Evaluator(error) => write!(formatter, "evaluator failed: {error}"),
            Self::Mcts(error) => error.fmt(formatter),
            Self::InferenceStopped => formatter.write_str("inference thread stopped"),
            Self::WorkerPanicked => formatter.write_str("actor thread panicked"),
        }
    }
}

impl Error for ActorError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Evaluator(error) => Some(error),
            Self::Mcts(error) => Some(error),
            _ => None,
        }
    }
}

struct InferenceRequest {
    worker: usize,
    tree: usize,
    request: EvaluationRequest,
    position: Board,
}

struct InferenceResponse {
    tree: usize,
    request: EvaluationRequest,
    policy: [f32; 65],
    value: f32,
}

struct ActorGame {
    tree: Mcts<Board>,
    simulations: u32,
}

struct WorkerResult {
    first_game_actions: Vec<usize>,
    draws: usize,
}

pub fn run(evaluator: OthelloCandleEvaluator, config: ActorConfig) -> Result<ActorRun, ActorError> {
    if config.games == 0
        || config.simulations == 0
        || config.workers == 0
        || config.inference_batch_size == 0
    {
        return Err(ActorError::InvalidConfig);
    }

    let worker_count = config.workers.min(config.games);
    let (request_sender, request_receiver) = bounded(config.inference_batch_size.saturating_mul(2));
    let mut response_senders = Vec::with_capacity(worker_count);
    let mut response_receivers = Vec::with_capacity(worker_count);
    for worker in 0..worker_count {
        let game_count = shard_len(config.games, worker_count, worker);
        let (sender, receiver) = bounded(game_count.max(1));
        response_senders.push(sender);
        response_receivers.push(receiver);
    }

    let inference = thread::spawn(move || {
        run_inference(
            evaluator,
            config.inference_batch_size,
            request_receiver,
            response_senders,
        )
    });

    let mut workers = Vec::with_capacity(worker_count);
    let mut first_game = 0;
    for (worker, responses) in response_receivers.into_iter().enumerate() {
        let game_count = shard_len(config.games, worker_count, worker);
        let owns_first_game = first_game == 0;
        first_game += game_count;
        let requests = request_sender.clone();
        workers.push(thread::spawn(move || {
            run_worker(
                worker,
                game_count,
                owns_first_game,
                config.simulations,
                requests,
                responses,
            )
        }));
    }
    drop(request_sender);

    let mut first_game_actions = Vec::new();
    let mut draws = 0;
    let mut worker_error = None;
    for worker in workers {
        match worker.join() {
            Ok(Ok(result)) => {
                if !result.first_game_actions.is_empty() {
                    first_game_actions = result.first_game_actions;
                }
                draws += result.draws;
            }
            Ok(Err(error)) => {
                worker_error.get_or_insert(error);
            }
            Err(_) => {
                worker_error.get_or_insert(ActorError::WorkerPanicked);
            }
        }
    }
    let inference = inference.join().map_err(|_| ActorError::WorkerPanicked)?;
    let (evaluations, batches) = inference?;
    if let Some(error) = worker_error {
        return Err(error);
    }

    Ok(ActorRun {
        first_game_actions,
        draws,
        evaluations,
        batches,
    })
}

fn run_inference(
    mut evaluator: OthelloCandleEvaluator,
    batch_size: usize,
    requests: Receiver<InferenceRequest>,
    responses: Vec<Sender<InferenceResponse>>,
) -> Result<(u64, u64), ActorError> {
    let mut batch = Vec::with_capacity(batch_size);
    let mut positions = Vec::with_capacity(batch_size);
    let mut policies = Vec::with_capacity(batch_size * Board::ACTION_COUNT);
    let mut values = Vec::with_capacity(batch_size);

    while let Ok(first) = requests.recv() {
        batch.clear();
        positions.clear();
        batch.push(first);
        while batch.len() < batch_size {
            match requests.recv_timeout(Duration::from_micros(100)) {
                Ok(request) => batch.push(request),
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => break,
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
            }
        }
        positions.extend(batch.iter().map(|request| request.position));
        policies.resize(batch.len() * Board::ACTION_COUNT, 0.0);
        values.resize(batch.len(), 0.0);
        evaluator
            .evaluate_batch(&positions, &mut policies, &mut values)
            .map_err(ActorError::Evaluator)?;

        for (index, request) in batch.iter().enumerate() {
            let start = index * Board::ACTION_COUNT;
            let mut policy = [0.0; 65];
            policy.copy_from_slice(&policies[start..start + Board::ACTION_COUNT]);
            let _ = responses[request.worker].send(InferenceResponse {
                tree: request.tree,
                request: request.request,
                policy,
                value: values[index],
            });
        }
    }

    Ok((evaluator.evaluations(), evaluator.batches()))
}

fn run_worker(
    worker: usize,
    game_count: usize,
    owns_first_game: bool,
    simulations: u32,
    requests: Sender<InferenceRequest>,
    responses: Receiver<InferenceResponse>,
) -> Result<WorkerResult, ActorError> {
    let mut games = (0..game_count)
        .map(|_| ActorGame {
            tree: Mcts::new(Board::default(), MctsConfig::default()),
            simulations: 0,
        })
        .collect::<Vec<_>>();
    let mut first_game_actions = Vec::new();

    loop {
        let mut progressed = false;
        while let Ok(response) = responses.try_recv() {
            let game = &mut games[response.tree];
            game.tree
                .complete(response.request, &response.policy, response.value)
                .map_err(ActorError::Mcts)?;
            game.simulations += 1;
            progressed = true;
        }

        for (tree_index, game) in games.iter_mut().enumerate() {
            if game.tree.root_position().outcome().is_some() {
                continue;
            }
            if game.simulations == simulations {
                let action = game
                    .tree
                    .best_action()
                    .ok_or(ActorError::Mcts(MctsError::NoLegalActions))?;
                if owns_first_game && tree_index == 0 {
                    first_game_actions.push(Board::action_index(action));
                }
                assert!(game.tree.advance(action));
                game.simulations = 0;
                progressed = true;
                if game.tree.root_position().outcome().is_some() {
                    continue;
                }
            }
            if game.tree.is_pending() {
                continue;
            }
            match game.tree.select().map_err(ActorError::Mcts)? {
                Selection::Terminal => {
                    game.simulations += 1;
                    progressed = true;
                }
                Selection::Evaluate { request, position } => {
                    requests
                        .send(InferenceRequest {
                            worker,
                            tree: tree_index,
                            request,
                            position: *position,
                        })
                        .map_err(|_| ActorError::InferenceStopped)?;
                    progressed = true;
                }
            }
        }

        if games
            .iter()
            .all(|game| game.tree.root_position().outcome().is_some())
        {
            break;
        }
        if !progressed {
            let response = responses.recv().map_err(|_| ActorError::InferenceStopped)?;
            let game = &mut games[response.tree];
            game.tree
                .complete(response.request, &response.policy, response.value)
                .map_err(ActorError::Mcts)?;
            game.simulations += 1;
        }
    }

    let draws = games
        .iter()
        .filter(|game| game.tree.root_position().outcome() == Some(Outcome::Draw))
        .count();
    Ok(WorkerResult {
        first_game_actions,
        draws,
    })
}

fn shard_len(games: usize, workers: usize, worker: usize) -> usize {
    games / workers + usize::from(worker < games % workers)
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;

    #[test]
    fn actor_workers_complete_games_through_one_inference_owner() {
        let evaluator = OthelloCandleEvaluator::new(Device::Cpu, 7).unwrap();
        let result = run(
            evaluator,
            ActorConfig {
                games: 8,
                simulations: 16,
                workers: 2,
                inference_batch_size: 8,
            },
        )
        .unwrap();

        assert!(!result.first_game_actions.is_empty());
        assert!(result.draws <= 8);
        assert!(result.evaluations > 0);
        assert!(result.batches > 0);
    }
}
