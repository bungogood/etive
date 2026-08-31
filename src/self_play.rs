//! Generic persistent self-play workers feeding one pipelined inference owner.

use std::collections::HashSet;
use std::error::Error as StdError;
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, sync_channel};
use std::thread;
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Gamma};
use tracing::{Span, info};

use crate::evaluator::{InferenceBatch, PipelinedEvaluator};
use crate::game::{Color, Game, Outcome};
use crate::mcts::{EvaluationRequest, Mcts, MctsError, Selection};

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub games: usize,
    pub simulations: u32,
    pub workers: usize,
    pub inference_batch_size: usize,
    pub seed: u64,
    pub dirichlet_alpha: f64,
    pub dirichlet_fraction: f32,
    pub temperature_moves: usize,
}

impl Config {
    pub fn is_valid(self) -> bool {
        self.games > 0
            && self.simulations >= 2
            && self.workers > 0
            && self.inference_batch_size > 0
            && self.dirichlet_alpha.is_finite()
            && self.dirichlet_alpha > 0.0
            && self.dirichlet_fraction.is_finite()
            && (0.0..=1.0).contains(&self.dirichlet_fraction)
    }
}

#[derive(bincode::Decode, bincode::Encode, Clone)]
#[bincode(
    encode_bounds = "G: bincode::Encode, G::Policy: bincode::Encode",
    decode_bounds = "G: bincode::Decode<__Context>, G::Policy: bincode::Decode<__Context>",
    borrow_decode_bounds = "G: bincode::BorrowDecode<'__de, __Context>, G::Policy: bincode::BorrowDecode<'__de, __Context>"
)]
pub struct Sample<G: Game> {
    pub position: G,
    pub policy: G::Policy,
    pub outcome: Outcome,
    pub game: u64,
}

pub struct Run<G: Game> {
    pub evaluations: u64,
    pub inference_batches: u64,
    pub unique_games: usize,
    pub samples: Vec<Sample<G>>,
}

#[derive(Debug, thiserror::Error)]
pub enum Error<E: StdError + 'static> {
    #[error("self-play configuration is invalid")]
    InvalidConfig,
    #[error("evaluator failed: {0}")]
    Evaluator(#[source] E),
    #[error(transparent)]
    Mcts(#[from] MctsError),
    #[error("inference thread stopped")]
    InferenceStopped,
    #[error("self-play worker panicked")]
    WorkerPanicked,
}

struct InferenceRequest<G: Game> {
    worker: usize,
    tree: usize,
    request: EvaluationRequest,
    position: G,
}

struct InferenceResponse<G: Game> {
    tree: usize,
    request: EvaluationRequest,
    policy: G::Policy,
    value: f32,
}

struct InferenceTag {
    worker: usize,
    tree: usize,
    request: EvaluationRequest,
}

struct ActorGame<G: Game> {
    tree: Mcts<G>,
    simulations: u32,
    root_noise_applied: bool,
    noise: Vec<f32>,
    records: Vec<PendingSample<G>>,
    trajectory_hash: u64,
}

struct PendingSample<G: Game> {
    position: G,
    policy: G::Policy,
    player: Color,
}

struct InferenceRun {
    evaluations: u64,
    batches: u64,
}

struct WorkerConfig {
    worker: usize,
    game_count: usize,
    inference_share: usize,
    simulations: u32,
    seed: u64,
    dirichlet_alpha: f64,
    dirichlet_fraction: f32,
    temperature_moves: usize,
}

pub fn run<G, E>(evaluator: E, config: Config) -> Result<Run<G>, Error<E::Error>>
where
    G: Game + Default,
    E: PipelinedEvaluator<G> + Send + 'static,
    E::Error: StdError + Send + 'static,
    E::Pending: 'static,
{
    if !config.is_valid() {
        return Err(Error::InvalidConfig);
    }
    assert_eq!(G::zero_policy().as_ref().len(), G::ACTION_COUNT);

    let worker_count = config.workers.min(config.games);
    let (request_sender, request_receiver) =
        sync_channel(config.inference_batch_size.saturating_mul(2));
    let mut response_senders = Vec::with_capacity(worker_count);
    let mut response_receivers = Vec::with_capacity(worker_count);
    for worker in 0..worker_count {
        let game_count = shard_len(config.games, worker_count, worker);
        let (sender, receiver) = sync_channel(game_count.max(1));
        response_senders.push(sender);
        response_receivers.push(receiver);
    }

    let inference_span = Span::current();
    let pad_partial_batches = config.games >= config.inference_batch_size;
    let inference = thread::spawn(move || {
        let _inference_guard = inference_span.enter();
        run_inference(
            evaluator,
            config.inference_batch_size,
            pad_partial_batches,
            request_receiver,
            response_senders,
        )
    });

    let mut workers = Vec::with_capacity(worker_count);
    for (worker, responses) in response_receivers.into_iter().enumerate() {
        let game_count = shard_len(config.games, worker_count, worker);
        let requests = request_sender.clone();
        workers.push(thread::spawn(move || {
            run_worker(
                WorkerConfig {
                    worker,
                    game_count,
                    inference_share: config
                        .inference_batch_size
                        .saturating_mul(2)
                        .div_ceil(worker_count),
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

    let mut worker_error = None;
    let mut samples = Vec::new();
    for worker in workers {
        match worker.join() {
            Ok(Ok(worker_samples)) => samples.extend(worker_samples),
            Ok(Err(error)) => {
                worker_error.get_or_insert(error);
            }
            Err(_) => {
                worker_error.get_or_insert(Error::WorkerPanicked);
            }
        }
    }
    let inference = inference.join().map_err(|_| Error::WorkerPanicked)??;
    if let Some(error) = worker_error {
        return Err(error);
    }

    Ok(Run {
        evaluations: inference.evaluations,
        inference_batches: inference.batches,
        unique_games: samples
            .iter()
            .map(|sample| sample.game)
            .collect::<HashSet<_>>()
            .len(),
        samples,
    })
}

fn run_inference<G, E>(
    mut evaluator: E,
    batch_size: usize,
    pad_partial_batches: bool,
    requests: Receiver<InferenceRequest<G>>,
    responses: Vec<SyncSender<InferenceResponse<G>>>,
) -> Result<InferenceRun, Error<E::Error>>
where
    G: Game,
    E: PipelinedEvaluator<G>,
    E::Error: StdError + 'static,
{
    let (ready_sender, ready_receiver) = sync_channel(2);
    let (completed_sender, completed_receiver) = sync_channel(1);
    let (recycled_sender, recycled_receiver) = sync_channel(3);
    let batch_span = Span::current();
    let batcher = thread::spawn(move || {
        let _batch_guard = batch_span.enter();
        batch_inference(requests, ready_sender, recycled_receiver, batch_size)
    });
    let dispatch_span = Span::current();
    let dispatcher = thread::spawn(move || {
        let _dispatch_guard = dispatch_span.enter();
        dispatch_inference(completed_receiver, recycled_sender, responses)
    });
    let start = Instant::now();
    let mut last_progress = start;
    let mut last_evaluations = 0;
    let mut evaluations = 0;
    let mut batches = 0;

    let Ok(mut batch) = ready_receiver.recv() else {
        drop(completed_sender);
        batcher.join().map_err(|_| Error::WorkerPanicked)?;
        dispatcher.join().map_err(|_| Error::WorkerPanicked)?;
        return Ok(InferenceRun {
            evaluations: 0,
            batches: 0,
        });
    };
    if pad_partial_batches {
        batch.pad_positions_to_capacity();
    }
    let mut pending = evaluator.start_batch(batch.positions());

    loop {
        let next = match ready_receiver.recv_timeout(Duration::from_millis(4)) {
            Ok(mut next_batch) => {
                if pad_partial_batches {
                    next_batch.pad_positions_to_capacity();
                }
                let next_pending = evaluator.start_batch(next_batch.positions());
                Some((next_batch, next_pending))
            }
            Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => None,
        };
        let output = evaluator.finish_batch(pending).map_err(Error::Evaluator)?;
        batch.set_packed_results(&output);
        evaluations += batch.len() as u64;
        batches += 1;

        let interval = last_progress.elapsed();
        if interval >= Duration::from_secs(5) {
            info!(
                evaluations,
                batches,
                batch_size = batch.len(),
                average_batch_size = %format_args!("{:.1}", evaluations as f64 / batches as f64),
                evaluations_per_second = %format_args!(
                    "{:.0}",
                    (evaluations - last_evaluations) as f64 / interval.as_secs_f64()
                ),
                elapsed = %format_args!("{:.1}s", start.elapsed().as_secs_f64()),
                "self-play inference progress"
            );
            last_progress = Instant::now();
            last_evaluations = evaluations;
        }

        completed_sender
            .send(batch)
            .map_err(|_| Error::InferenceStopped)?;

        match next {
            Some((next_batch, next_pending)) => {
                batch = next_batch;
                pending = next_pending;
            }
            None => match ready_receiver.recv() {
                Ok(mut next_batch) => {
                    if pad_partial_batches {
                        next_batch.pad_positions_to_capacity();
                    }
                    pending = evaluator.start_batch(next_batch.positions());
                    batch = next_batch;
                }
                Err(_) => break,
            },
        }
    }

    drop(completed_sender);
    batcher.join().map_err(|_| Error::WorkerPanicked)?;
    dispatcher.join().map_err(|_| Error::WorkerPanicked)?;

    Ok(InferenceRun {
        evaluations,
        batches,
    })
}

fn batch_inference<G: Game>(
    requests: Receiver<InferenceRequest<G>>,
    ready: SyncSender<InferenceBatch<G, InferenceTag>>,
    recycled: Receiver<InferenceBatch<G, InferenceTag>>,
    batch_size: usize,
) {
    let mut available = vec![
        InferenceBatch::new(batch_size),
        InferenceBatch::new(batch_size),
        InferenceBatch::new(batch_size),
    ];
    loop {
        let mut batch = match available.pop() {
            Some(batch) => batch,
            None => match recycled.recv() {
                Ok(batch) => batch,
                Err(_) => return,
            },
        };
        let Ok(first) = requests.recv() else {
            return;
        };
        push_inference_request(&mut batch, first);
        while !batch.is_full() {
            match requests.recv_timeout(Duration::from_millis(5)) {
                Ok(request) => push_inference_request(&mut batch, request),
                Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => break,
            }
        }
        if ready.send(batch).is_err() {
            return;
        }
    }
}

fn dispatch_inference<G: Game>(
    completed: Receiver<InferenceBatch<G, InferenceTag>>,
    recycled: SyncSender<InferenceBatch<G, InferenceTag>>,
    responses: Vec<SyncSender<InferenceResponse<G>>>,
) {
    while let Ok(mut batch) = completed.recv() {
        for index in 0..batch.len() {
            let (request, logits, value) = batch.result(index);
            let mut policy = G::zero_policy();
            assert_eq!(policy.as_ref().len(), G::ACTION_COUNT);
            policy.as_mut().copy_from_slice(logits);
            let _ = responses[request.worker].send(InferenceResponse {
                tree: request.tree,
                request: request.request,
                policy,
                value,
            });
        }
        batch.clear();
        if recycled.send(batch).is_err() {
            break;
        }
    }
}

fn push_inference_request<G: Game>(
    batch: &mut InferenceBatch<G, InferenceTag>,
    request: InferenceRequest<G>,
) {
    batch.push(
        InferenceTag {
            worker: request.worker,
            tree: request.tree,
            request: request.request,
        },
        request.position,
    );
}

fn run_worker<G, E>(
    config: WorkerConfig,
    requests: SyncSender<InferenceRequest<G>>,
    responses: Receiver<InferenceResponse<G>>,
) -> Result<Vec<Sample<G>>, Error<E>>
where
    G: Game + Default,
    E: StdError + 'static,
{
    let mut random = StdRng::seed_from_u64(config.seed);
    let dirichlet = Gamma::new(config.dirichlet_alpha, 1.0).expect("validated Dirichlet alpha");
    let mut games = (0..config.game_count)
        .map(|_| ActorGame {
            tree: Mcts::new(G::default()),
            simulations: 0,
            root_noise_applied: false,
            noise: Vec::new(),
            records: Vec::new(),
            trajectory_hash: 0xcbf2_9ce4_8422_2325,
        })
        .collect::<Vec<_>>();
    loop {
        let mut progressed = false;
        while let Ok(response) = responses.try_recv() {
            let game = &mut games[response.tree];
            game.tree
                .complete(response.request, response.policy.as_ref(), response.value)
                .map_err(Error::Mcts)?;
            game.simulations += 1;
            progressed = true;
        }

        let active_games = games
            .iter()
            .filter(|game| game.tree.root_position().outcome().is_none())
            .count();
        let max_pending_per_game = config.inference_share.div_ceil(active_games.max(1));

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
                game.records.push(PendingSample {
                    position: *game.tree.root_position(),
                    policy,
                    player: game.tree.root_position().side_to_move(),
                });
                game.trajectory_hash = game.trajectory_hash.wrapping_mul(0x0000_0100_0000_01b3)
                    ^ G::action_index(action) as u64;
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
            while game.simulations + (game.tree.pending_count() as u32) < config.simulations
                && game.tree.pending_count() < max_pending_per_game
            {
                match game.tree.select().map_err(Error::Mcts)? {
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
                            .map_err(|_| Error::InferenceStopped)?;
                        progressed = true;
                    }
                    Selection::Blocked => break,
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
            let response = responses.recv().map_err(|_| Error::InferenceStopped)?;
            let game = &mut games[response.tree];
            game.tree
                .complete(response.request, response.policy.as_ref(), response.value)
                .map_err(Error::Mcts)?;
            game.simulations += 1;
        }
    }

    let mut samples = Vec::with_capacity(games.iter().map(|game| game.records.len()).sum());
    for game in &mut games {
        let terminal = *game.tree.root_position();
        samples.extend(game.records.drain(..).map(|record| Sample {
            position: record.position,
            policy: record.policy,
            outcome: outcome_for(record.player, terminal),
            game: game.trajectory_hash,
        }));
    }
    Ok(samples)
}

fn root_policy<G: Game>(tree: &Mcts<G>) -> Result<G::Policy, MctsError> {
    let total = tree.root_stats().map(|stats| stats.visits).sum::<u32>();
    if total == 0 {
        return Err(MctsError::NoLegalActions);
    }
    let mut policy = G::zero_policy();
    assert_eq!(policy.as_ref().len(), G::ACTION_COUNT);
    for stats in tree.root_stats() {
        policy.as_mut()[G::action_index(stats.action)] = stats.visits as f32 / total as f32;
    }
    Ok(policy)
}

fn select_action<G: Game>(
    tree: &Mcts<G>,
    sample: bool,
    random: &mut StdRng,
) -> Result<G::Action, MctsError> {
    if !sample {
        return tree.best_action().ok_or(MctsError::NoLegalActions);
    }
    let total = tree.root_stats().map(|stats| stats.visits).sum::<u32>();
    if total == 0 {
        return Err(MctsError::NoLegalActions);
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

fn mix_root_noise<G: Game>(
    tree: &mut Mcts<G>,
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

fn outcome_for<G: Game>(player: Color, terminal: G) -> Outcome {
    let outcome = terminal.outcome().expect("self-play game must be terminal");
    if player == terminal.side_to_move() {
        outcome
    } else {
        outcome.reversed()
    }
}

fn shard_len(games: usize, workers: usize, worker: usize) -> usize {
    games / workers + usize::from(worker < games % workers)
}

#[cfg(test)]
mod tests {
    use crate::game::Game;
    use crate::tic_tac_toe::{Board, Square};

    use super::*;

    fn terminal_board() -> Board {
        let mut board = Board::default();
        for index in [0, 3, 1, 4, 2] {
            board.play(Square::from_index(index).unwrap());
        }
        board
    }

    #[test]
    fn terminal_outcomes_are_side_relative() {
        let terminal = terminal_board();
        assert_eq!(terminal.outcome(), Some(Outcome::Loss));
        assert_eq!(outcome_for(Color::White, terminal), Outcome::Loss);
        assert_eq!(outcome_for(Color::Black, terminal), Outcome::Win);
    }
}
