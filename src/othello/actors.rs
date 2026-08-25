//! Persistent self-play actors feeding one batched inference owner.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;
use std::thread;
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender, bounded};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Gamma};

use super::{Board, Color, GameStatus};
use crate::evaluator::{InferenceBatch, OthelloCandleEvaluator};
use crate::game::{Game, Outcome};
use crate::mcts::{EvaluationRequest, Mcts, MctsConfig, MctsError, Selection};

#[derive(Clone, Copy, Debug)]
pub struct ActorConfig {
    pub games: usize,
    pub simulations: u32,
    pub workers: usize,
    pub inference_batch_size: usize,
    pub seed: u64,
    pub dirichlet_alpha: f64,
    pub dirichlet_fraction: f32,
    pub temperature_moves: usize,
}

#[derive(Debug)]
pub struct ActorRun {
    pub first_game_actions: Vec<usize>,
    pub draws: usize,
    pub evaluations: u64,
    pub batches: u64,
    pub unique_games: usize,
    pub samples: Vec<SelfPlaySample>,
}

#[derive(bincode::Decode, bincode::Encode, Clone, Debug)]
pub struct SelfPlaySample {
    pub position: Board,
    pub policy: [f32; 65],
    pub outcome: Outcome,
    pub game: u64,
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

struct InferenceTag {
    worker: usize,
    tree: usize,
    request: EvaluationRequest,
}

struct ActorGame {
    tree: Mcts<Board>,
    simulations: u32,
    root_noise_applied: bool,
    noise: Vec<f32>,
    records: Vec<PendingSample>,
    trajectory_hash: u64,
}

struct PendingSample {
    position: Board,
    policy: [f32; 65],
    player: Color,
}

struct WorkerResult {
    first_game_actions: Vec<usize>,
    draws: usize,
    samples: Vec<SelfPlaySample>,
    trajectory_hashes: Vec<u64>,
}

struct WorkerConfig {
    worker: usize,
    game_count: usize,
    owns_first_game: bool,
    simulations: u32,
    seed: u64,
    dirichlet_alpha: f64,
    dirichlet_fraction: f32,
    temperature_moves: usize,
}

pub fn run(
    evaluator: OthelloCandleEvaluator,
    config: ActorConfig,
) -> Result<(ActorRun, OthelloCandleEvaluator), ActorError> {
    if config.games == 0
        || config.simulations < 2
        || config.workers == 0
        || config.inference_batch_size == 0
        || !config.dirichlet_alpha.is_finite()
        || config.dirichlet_alpha <= 0.0
        || !(0.0..=1.0).contains(&config.dirichlet_fraction)
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
                WorkerConfig {
                    worker,
                    game_count,
                    owns_first_game,
                    simulations: config.simulations,
                    seed: config.seed.wrapping_add(worker as u64),
                    dirichlet_alpha: config.dirichlet_alpha,
                    dirichlet_fraction: config.dirichlet_fraction,
                    temperature_moves: config.temperature_moves,
                },
                requests,
                responses,
            )
        }));
    }
    drop(request_sender);

    let mut first_game_actions = Vec::new();
    let mut draws = 0;
    let mut worker_error = None;
    let mut samples = Vec::with_capacity(config.games.saturating_mul(60));
    let mut trajectory_hashes = Vec::with_capacity(config.games);
    for worker in workers {
        match worker.join() {
            Ok(Ok(result)) => {
                if !result.first_game_actions.is_empty() {
                    first_game_actions = result.first_game_actions;
                }
                draws += result.draws;
                samples.extend(result.samples);
                trajectory_hashes.extend(result.trajectory_hashes);
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
    let (evaluator, evaluations, batches) = inference?;
    if let Some(error) = worker_error {
        return Err(error);
    }

    Ok((
        ActorRun {
            first_game_actions,
            draws,
            evaluations,
            batches,
            unique_games: trajectory_hashes.into_iter().collect::<HashSet<_>>().len(),
            samples,
        },
        evaluator,
    ))
}

fn run_inference(
    mut evaluator: OthelloCandleEvaluator,
    batch_size: usize,
    requests: Receiver<InferenceRequest>,
    responses: Vec<Sender<InferenceResponse>>,
) -> Result<(OthelloCandleEvaluator, u64, u64), ActorError> {
    let mut batch = InferenceBatch::new(batch_size);

    while let Ok(first) = requests.recv() {
        batch.clear();
        push_inference_request(&mut batch, first);
        while !batch.is_full() {
            match requests.recv_timeout(Duration::from_micros(100)) {
                Ok(request) => push_inference_request(&mut batch, request),
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => break,
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
            }
        }
        batch
            .evaluate_batch(&mut evaluator)
            .map_err(ActorError::Evaluator)?;

        for index in 0..batch.len() {
            let (request, logits, value) = batch.result(index);
            let mut policy = [0.0; 65];
            policy.copy_from_slice(logits);
            let _ = responses[request.worker].send(InferenceResponse {
                tree: request.tree,
                request: request.request,
                policy,
                value,
            });
        }
    }

    let evaluations = evaluator.evaluations();
    let batches = evaluator.batches();
    Ok((evaluator, evaluations, batches))
}

fn push_inference_request(
    batch: &mut InferenceBatch<Board, InferenceTag>,
    request: InferenceRequest,
) {
    assert!(batch.push(
        InferenceTag {
            worker: request.worker,
            tree: request.tree,
            request: request.request,
        },
        request.position,
    ));
}

fn run_worker(
    config: WorkerConfig,
    requests: Sender<InferenceRequest>,
    responses: Receiver<InferenceResponse>,
) -> Result<WorkerResult, ActorError> {
    let mut random = StdRng::seed_from_u64(config.seed);
    let dirichlet = Gamma::new(config.dirichlet_alpha, 1.0).expect("validated Dirichlet alpha");
    let mut games = (0..config.game_count)
        .map(|_| ActorGame {
            tree: Mcts::new(Board::default(), MctsConfig::default()),
            simulations: 0,
            root_noise_applied: false,
            noise: Vec::new(),
            records: Vec::with_capacity(64),
            trajectory_hash: 0xcbf2_9ce4_8422_2325,
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
            if game.simulations == config.simulations {
                let policy = root_policy(&game.tree)?;
                let action = select_action(
                    &game.tree,
                    game.records.len() < config.temperature_moves,
                    &mut random,
                )?;
                if config.owns_first_game && tree_index == 0 {
                    first_game_actions.push(Board::action_index(action));
                }
                game.records.push(PendingSample {
                    position: *game.tree.root_position(),
                    policy,
                    player: game.tree.root_position().side_to_move(),
                });
                game.trajectory_hash = game.trajectory_hash.wrapping_mul(0x0000_0100_0000_01b3)
                    ^ Board::action_index(action) as u64;
                assert!(game.tree.advance(action));
                assert!(game.tree.rebase_root());
                game.simulations = 0;
                game.root_noise_applied = false;
                progressed = true;
                if game.tree.root_position().outcome().is_some() {
                    continue;
                }
            }
            if !game.root_noise_applied && game.tree.root_stats().len() > 0 {
                mix_root_noise(
                    &mut game.tree,
                    &mut game.noise,
                    &dirichlet,
                    config.dirichlet_fraction,
                    &mut random,
                );
                game.root_noise_applied = true;
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
                            worker: config.worker,
                            tree: tree_index,
                            request,
                            position: *position,
                        })
                        .map_err(|_| ActorError::InferenceStopped)?;
                    progressed = true;
                }
                Selection::Blocked => {}
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
    let mut samples = Vec::with_capacity(games.iter().map(|game| game.records.len()).sum());
    let mut trajectory_hashes = Vec::with_capacity(games.len());
    for game in &mut games {
        let terminal = *game.tree.root_position();
        samples.extend(game.records.drain(..).map(|record| SelfPlaySample {
            position: record.position,
            policy: record.policy,
            outcome: outcome_for(record.player, terminal),
            game: game.trajectory_hash,
        }));
        trajectory_hashes.push(game.trajectory_hash);
    }
    Ok(WorkerResult {
        first_game_actions,
        draws,
        samples,
        trajectory_hashes,
    })
}

fn root_policy(tree: &Mcts<Board>) -> Result<[f32; 65], ActorError> {
    let total = tree.root_stats().map(|stats| stats.visits).sum::<u32>();
    if total == 0 {
        return Err(ActorError::Mcts(MctsError::NoLegalActions));
    }
    let mut policy = [0.0; 65];
    for stats in tree.root_stats() {
        policy[Board::action_index(stats.action)] = stats.visits as f32 / total as f32;
    }
    Ok(policy)
}

fn select_action(
    tree: &Mcts<Board>,
    sample: bool,
    random: &mut StdRng,
) -> Result<super::Move, ActorError> {
    if !sample {
        return tree
            .best_action()
            .ok_or(ActorError::Mcts(MctsError::NoLegalActions));
    }
    let total = tree.root_stats().map(|stats| stats.visits).sum::<u32>();
    if total == 0 {
        return Err(ActorError::Mcts(MctsError::NoLegalActions));
    }
    let mut selected = random.random_range(0..total);
    for stats in tree.root_stats() {
        if selected < stats.visits {
            return Ok(stats.action);
        }
        selected -= stats.visits;
    }
    unreachable!("visit counts must contain the sampled action")
}

fn mix_root_noise(
    tree: &mut Mcts<Board>,
    noise: &mut Vec<f32>,
    dirichlet: &Gamma<f64>,
    fraction: f32,
    random: &mut StdRng,
) {
    noise.resize(tree.root_stats().len(), 0.0);
    let mut sum = 0.0;
    for value in noise.iter_mut() {
        *value = dirichlet.sample(random) as f32;
        sum += *value;
    }
    for value in noise.iter_mut() {
        *value /= sum;
    }
    assert!(tree.mix_root_priors(noise, fraction));
}

fn outcome_for(player: Color, terminal: Board) -> Outcome {
    match terminal.status() {
        GameStatus::Won(winner) if winner == player => Outcome::Win,
        GameStatus::Won(_) => Outcome::Loss,
        GameStatus::Drawn => Outcome::Draw,
        GameStatus::Ongoing => unreachable!("self-play game must be terminal"),
    }
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
        let (result, _) = run(
            evaluator,
            ActorConfig {
                games: 2,
                simulations: 2,
                workers: 1,
                inference_batch_size: 2,
                seed: 11,
                dirichlet_alpha: 0.3,
                dirichlet_fraction: 0.25,
                temperature_moves: 20,
            },
        )
        .unwrap();

        assert!(!result.first_game_actions.is_empty());
        assert!(result.draws <= 2);
        assert!(result.evaluations > 0);
        assert!(result.batches > 0);
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
