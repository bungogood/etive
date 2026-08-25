//! Othello policy/value optimization and model comparison.

use std::collections::HashMap;
use std::error::Error;
use std::path::Path;
use std::time::{Duration, Instant};

use candle_core::{Device, Tensor, Var};
use candle_nn::{loss, ops};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::actors::SelfPlaySample;
use super::{Board, Color, GameStatus};
use crate::encoding::{OthelloEncodingV1, StateEncoder};
use crate::evaluator::OthelloCandleEvaluator;
use crate::game::Game;
use crate::mcts::{Mcts, MctsConfig, SearchWorkspace};
use crate::model::OthelloNetwork;

#[derive(Clone, Copy, Debug)]
pub struct TrainingConfig {
    pub duration: Duration,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub seed: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct TrainingReport {
    pub steps: usize,
    pub policy_loss: f32,
    pub value_loss: f32,
    pub elapsed: Duration,
}

#[derive(Clone, Copy, Debug)]
pub struct LossReport {
    pub policy_loss: f32,
    pub value_loss: f32,
}

pub struct TrainingSession {
    optimizer: RestorableAdamW,
    device: Device,
    random: StdRng,
    batch_size: usize,
    inputs: Vec<f32>,
    policies: Vec<f32>,
    outcomes: Vec<f32>,
}

pub struct TrainingSnapshot {
    step: usize,
    moments: Vec<(Tensor, Tensor)>,
}

struct AdamVariable {
    name: String,
    variable: Var,
    first_moment: Var,
    second_moment: Var,
}

struct RestorableAdamW {
    variables: Vec<AdamVariable>,
    step: usize,
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    weight_decay: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ArenaResult {
    pub trained_wins: usize,
    pub initial_wins: usize,
    pub draws: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct ArenaProgress {
    pub completed: usize,
    pub total: usize,
    pub moves: usize,
    pub evaluations: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct ArenaConfig {
    pub games: usize,
    pub simulations: u32,
    pub batch_size: usize,
    pub opening_plies: usize,
    pub seed: u64,
}

pub fn train(
    network: &OthelloNetwork,
    device: &Device,
    samples: &[SelfPlaySample],
    config: TrainingConfig,
) -> candle_core::Result<TrainingReport> {
    if samples.is_empty() || config.batch_size == 0 || config.learning_rate <= 0.0 {
        candle_core::bail!(
            "training data and batch size must be non-empty and learning rate positive"
        )
    }

    let mut session = TrainingSession::new(
        network,
        device.clone(),
        config.batch_size,
        config.learning_rate,
        config.seed,
    )?;
    let start = Instant::now();
    let mut steps = 0;
    let mut final_losses = None;

    while steps == 0 || start.elapsed() < config.duration {
        final_losses = Some(session.step(network, &[samples])?);
        steps += 1;
    }

    let (policy_loss, value_loss) = final_losses.expect("training always performs one step");
    Ok(TrainingReport {
        steps,
        policy_loss: policy_loss.to_scalar()?,
        value_loss: value_loss.to_scalar()?,
        elapsed: start.elapsed(),
    })
}

pub fn evaluate_loss(
    network: &OthelloNetwork,
    device: &Device,
    samples: &[SelfPlaySample],
    batch_size: usize,
) -> candle_core::Result<LossReport> {
    if samples.is_empty() || batch_size == 0 {
        candle_core::bail!("validation data and batch size must be non-empty")
    }
    let mut policy_total = 0.0;
    let mut value_total = 0.0;
    for batch in samples.chunks(batch_size) {
        let mut inputs = vec![0.0; batch.len() * OthelloEncodingV1::encoded_len()];
        let mut policies = vec![0.0; batch.len() * Board::ACTION_COUNT];
        let mut outcomes = vec![0.0; batch.len()];
        for (index, sample) in batch.iter().enumerate() {
            let input_start = index * OthelloEncodingV1::encoded_len();
            OthelloEncodingV1.encode(
                &sample.position,
                &mut inputs[input_start..input_start + OthelloEncodingV1::encoded_len()],
            );
            let policy_start = index * Board::ACTION_COUNT;
            policies[policy_start..policy_start + Board::ACTION_COUNT]
                .copy_from_slice(&sample.policy);
            outcomes[index] = sample.outcome.value();
        }
        let input = Tensor::from_slice(&inputs, (batch.len(), 2, 8, 8), device)?;
        let target_policy =
            Tensor::from_slice(&policies, (batch.len(), Board::ACTION_COUNT), device)?;
        let target_value = Tensor::from_slice(&outcomes, (batch.len(), 1), device)?;
        let (policy_logits, values) = network.forward(&input)?;
        let policy_loss = (&target_policy * &ops::log_softmax(&policy_logits, 1)?)?
            .sum_all()?
            .affine(-1.0 / batch.len() as f64, 0.0)?
            .to_scalar::<f32>()?;
        let value_loss = loss::mse(&values, &target_value)?.to_scalar::<f32>()?;
        policy_total += policy_loss * batch.len() as f32;
        value_total += value_loss * batch.len() as f32;
    }
    Ok(LossReport {
        policy_loss: policy_total / samples.len() as f32,
        value_loss: value_total / samples.len() as f32,
    })
}

impl TrainingSession {
    pub fn new(
        network: &OthelloNetwork,
        device: Device,
        batch_size: usize,
        learning_rate: f64,
        seed: u64,
    ) -> candle_core::Result<Self> {
        if batch_size == 0 || learning_rate <= 0.0 {
            candle_core::bail!("training batch size and learning rate must be positive")
        }
        let optimizer = RestorableAdamW::new(network.named_variables(), learning_rate)?;
        Ok(Self {
            optimizer,
            device,
            random: StdRng::seed_from_u64(seed),
            batch_size,
            inputs: vec![0.0; batch_size * OthelloEncodingV1::encoded_len()],
            policies: vec![0.0; batch_size * Board::ACTION_COUNT],
            outcomes: vec![0.0; batch_size],
        })
    }

    pub fn train_steps(
        &mut self,
        network: &OthelloNetwork,
        replay: &[&[SelfPlaySample]],
        steps: usize,
    ) -> candle_core::Result<TrainingReport> {
        let sample_count = replay.iter().map(|samples| samples.len()).sum::<usize>();
        if sample_count == 0 || steps == 0 {
            candle_core::bail!("replay data and training steps must be non-empty")
        }
        let start = Instant::now();
        let mut final_losses = None;
        for _ in 0..steps {
            final_losses = Some(self.step(network, replay)?);
        }
        let (policy_loss, value_loss) = final_losses.expect("positive training step count");
        Ok(TrainingReport {
            steps,
            policy_loss: policy_loss.to_scalar()?,
            value_loss: value_loss.to_scalar()?,
            elapsed: start.elapsed(),
        })
    }

    pub fn snapshot(&self) -> candle_core::Result<TrainingSnapshot> {
        self.optimizer.snapshot()
    }

    pub fn restore(&mut self, snapshot: &TrainingSnapshot) -> candle_core::Result<()> {
        self.optimizer.restore(snapshot)
    }

    pub fn set_learning_rate(&mut self, learning_rate: f64) {
        self.optimizer.learning_rate = learning_rate;
    }

    pub fn reseed(&mut self, seed: u64) {
        self.random = StdRng::seed_from_u64(seed);
    }

    pub fn save_optimizer(&self, path: impl AsRef<Path>) -> candle_core::Result<()> {
        self.optimizer.save(path)
    }

    pub fn load_optimizer(&mut self, path: impl AsRef<Path>) -> candle_core::Result<()> {
        self.optimizer.load(path)
    }

    fn step(
        &mut self,
        network: &OthelloNetwork,
        replay: &[&[SelfPlaySample]],
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let sample_count = replay.iter().map(|samples| samples.len()).sum::<usize>();
        for (batch_index, outcome) in self.outcomes.iter_mut().enumerate() {
            let mut sample_index = self.random.random_range(0..sample_count);
            let mut sample = None;
            for samples in replay {
                if sample_index < samples.len() {
                    sample = Some(&samples[sample_index]);
                    break;
                }
                sample_index -= samples.len();
            }
            let sample = sample.expect("sample index must fall within replay data");
            let input_start = batch_index * OthelloEncodingV1::encoded_len();
            OthelloEncodingV1.encode(
                &sample.position,
                &mut self.inputs[input_start..input_start + OthelloEncodingV1::encoded_len()],
            );
            let policy_start = batch_index * Board::ACTION_COUNT;
            self.policies[policy_start..policy_start + Board::ACTION_COUNT]
                .copy_from_slice(&sample.policy);
            apply_symmetry(
                &mut self.inputs[input_start..input_start + OthelloEncodingV1::encoded_len()],
                &mut self.policies[policy_start..policy_start + Board::ACTION_COUNT],
                self.random.random_range(0..8),
            );
            *outcome = sample.outcome.value();
        }

        let input = Tensor::from_slice(&self.inputs, (self.batch_size, 2, 8, 8), &self.device)?;
        let target_policy = Tensor::from_slice(
            &self.policies,
            (self.batch_size, Board::ACTION_COUNT),
            &self.device,
        )?;
        let target_value = Tensor::from_slice(&self.outcomes, (self.batch_size, 1), &self.device)?;
        let (policy_logits, values) = network.forward(&input)?;
        let log_policy = ops::log_softmax(&policy_logits, 1)?;
        let policy_loss = (&target_policy * &log_policy)?
            .sum_all()?
            .affine(-1.0 / self.batch_size as f64, 0.0)?;
        let value_loss = loss::mse(&values, &target_value)?;
        let total_loss = (&policy_loss + &value_loss)?;
        self.optimizer.backward_step(&total_loss)?;
        Ok((policy_loss, value_loss))
    }
}

impl RestorableAdamW {
    fn new(variables: Vec<(String, Var)>, learning_rate: f64) -> candle_core::Result<Self> {
        let variables = variables
            .into_iter()
            .filter(|(_, variable)| variable.dtype().is_float())
            .map(|(name, variable)| {
                let first_moment =
                    Var::zeros(variable.shape(), variable.dtype(), variable.device())?;
                let second_moment =
                    Var::zeros(variable.shape(), variable.dtype(), variable.device())?;
                Ok(AdamVariable {
                    name,
                    variable,
                    first_moment,
                    second_moment,
                })
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Self {
            variables,
            step: 0,
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
        })
    }

    fn backward_step(&mut self, loss: &Tensor) -> candle_core::Result<()> {
        let gradients = loss.backward()?;
        self.step += 1;
        let first_scale = 1.0 / (1.0 - self.beta1.powi(self.step as i32));
        let second_scale = 1.0 / (1.0 - self.beta2.powi(self.step as i32));
        for state in &self.variables {
            let Some(gradient) = gradients.get(&state.variable) else {
                continue;
            };
            let first = ((state.first_moment.as_tensor() * self.beta1)?
                + (gradient * (1.0 - self.beta1))?)?;
            let second = ((state.second_moment.as_tensor() * self.beta2)?
                + (gradient.sqr()? * (1.0 - self.beta2))?)?;
            let adjusted =
                ((&first * first_scale)? / ((&second * second_scale)?.sqrt()? + self.epsilon)?)?;
            let decayed =
                (state.variable.as_tensor() * (1.0 - self.learning_rate * self.weight_decay))?;
            state
                .variable
                .set(&(decayed - (adjusted * self.learning_rate)?)?)?;
            state.first_moment.set(&first)?;
            state.second_moment.set(&second)?;
        }
        Ok(())
    }

    fn snapshot(&self) -> candle_core::Result<TrainingSnapshot> {
        let moments = self
            .variables
            .iter()
            .map(|state| {
                Ok((
                    state.first_moment.as_tensor().copy()?,
                    state.second_moment.as_tensor().copy()?,
                ))
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(TrainingSnapshot {
            step: self.step,
            moments,
        })
    }

    fn restore(&mut self, snapshot: &TrainingSnapshot) -> candle_core::Result<()> {
        if snapshot.moments.len() != self.variables.len() {
            candle_core::bail!("optimizer snapshot does not match model variables")
        }
        self.step = snapshot.step;
        for (state, (first, second)) in self.variables.iter().zip(&snapshot.moments) {
            state.first_moment.set(first)?;
            state.second_moment.set(second)?;
        }
        Ok(())
    }

    fn save(&self, path: impl AsRef<Path>) -> candle_core::Result<()> {
        let mut tensors = HashMap::with_capacity(self.variables.len() * 2 + 2);
        for state in &self.variables {
            tensors.insert(
                format!("first.{}", state.name),
                state.first_moment.as_tensor().clone(),
            );
            tensors.insert(
                format!("second.{}", state.name),
                state.second_moment.as_tensor().clone(),
            );
        }
        tensors.insert(
            "step".to_owned(),
            Tensor::new(self.step as u32, &Device::Cpu)?,
        );
        tensors.insert(
            "learning_rate".to_owned(),
            Tensor::new(self.learning_rate, &Device::Cpu)?,
        );
        candle_core::safetensors::save(&tensors, path)
    }

    fn load(&mut self, path: impl AsRef<Path>) -> candle_core::Result<()> {
        let tensors = candle_core::safetensors::load(path, self.variables[0].variable.device())?;
        self.step = tensors
            .get("step")
            .ok_or_else(|| candle_core::Error::Msg("optimizer step missing".to_owned()))?
            .to_scalar::<u32>()? as usize;
        self.learning_rate = tensors
            .get("learning_rate")
            .ok_or_else(|| candle_core::Error::Msg("optimizer learning rate missing".to_owned()))?
            .to_scalar::<f64>()?;
        for state in &self.variables {
            let first = tensors
                .get(&format!("first.{}", state.name))
                .ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "optimizer first moment missing for {}",
                        state.name
                    ))
                })?;
            let second = tensors
                .get(&format!("second.{}", state.name))
                .ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "optimizer second moment missing for {}",
                        state.name
                    ))
                })?;
            state.first_moment.set(first)?;
            state.second_moment.set(second)?;
        }
        Ok(())
    }
}

fn apply_symmetry(input: &mut [f32], policy: &mut [f32], symmetry: usize) {
    debug_assert_eq!(input.len(), 128);
    debug_assert_eq!(policy.len(), 65);
    let original_input: [f32; 128] = input.try_into().expect("fixed Othello input size");
    let original_policy: [f32; 65] = policy.try_into().expect("fixed Othello policy size");
    for plane in 0..2 {
        for source in 0..64 {
            input[plane * 64 + transform_square(source, symmetry)] =
                original_input[plane * 64 + source];
        }
    }
    for source in 0..64 {
        policy[transform_square(source, symmetry)] = original_policy[source];
    }
    policy[64] = original_policy[64];
}

fn transform_square(index: usize, symmetry: usize) -> usize {
    let mut row = index / 8;
    let mut column = index % 8;
    for _ in 0..symmetry % 4 {
        (row, column) = (column, 7 - row);
    }
    if symmetry >= 4 {
        column = 7 - column;
    }
    row * 8 + column
}

pub fn arena(
    trained: &mut OthelloCandleEvaluator,
    initial: &mut OthelloCandleEvaluator,
    games: usize,
    simulations: u32,
) -> Result<ArenaResult, Box<dyn Error>> {
    arena_with_progress(
        trained,
        initial,
        ArenaConfig {
            games,
            simulations,
            batch_size: games,
            opening_plies: 0,
            seed: 0,
        },
        |_| {},
    )
}

pub fn arena_with_progress(
    trained: &mut OthelloCandleEvaluator,
    initial: &mut OthelloCandleEvaluator,
    config: ArenaConfig,
    mut report_progress: impl FnMut(ArenaProgress),
) -> Result<ArenaResult, Box<dyn Error>> {
    let ArenaConfig {
        games,
        simulations,
        batch_size,
        opening_plies,
        seed,
    } = config;
    if games == 0 || !games.is_multiple_of(2) || simulations < 2 || batch_size == 0 {
        return Err(
            "arena games must be positive and even, simulations at least two, and batch size positive"
                .into(),
        );
    }

    let mut result = ArenaResult::default();
    let initial_evaluations = initial.evaluations();
    let trained_evaluations = trained.evaluations();
    let mut moves = 0;
    let mut boards = arena_openings(games, opening_plies, seed);
    let mut workspace = SearchWorkspace::new(batch_size);

    while result.trained_wins + result.initial_wins + result.draws < games {
        let mut initial_turn = Vec::with_capacity(games / 2);
        let mut trained_turn = Vec::with_capacity(games / 2);
        for (index, (board, initial_color)) in boards.iter().enumerate() {
            if board.outcome().is_some() {
                continue;
            }
            if board.side_to_move() == *initial_color {
                initial_turn.push(index);
            } else {
                trained_turn.push(index);
            }
        }

        search_moves(
            &mut workspace,
            initial,
            &mut boards,
            &initial_turn,
            simulations,
        )?;
        search_moves(
            &mut workspace,
            trained,
            &mut boards,
            &trained_turn,
            simulations,
        )?;
        moves += initial_turn.len() + trained_turn.len();

        for index in initial_turn.into_iter().chain(trained_turn) {
            let (board, initial_color) = boards[index];
            match board.status() {
                GameStatus::Drawn => result.draws += 1,
                GameStatus::Won(winner) if winner == initial_color => result.initial_wins += 1,
                GameStatus::Won(_) => result.trained_wins += 1,
                GameStatus::Ongoing => {}
            }
        }
        report_progress(ArenaProgress {
            completed: result.trained_wins + result.initial_wins + result.draws,
            total: games,
            moves,
            evaluations: initial.evaluations().saturating_sub(initial_evaluations)
                + trained.evaluations().saturating_sub(trained_evaluations),
        });
    }
    Ok(result)
}

fn arena_openings(games: usize, opening_plies: usize, seed: u64) -> Vec<(Board, Color)> {
    let mut random = StdRng::seed_from_u64(seed);
    let mut boards = Vec::with_capacity(games);
    for _ in 0..games / 2 {
        let mut board = Board::default();
        for _ in 0..opening_plies {
            if board.outcome().is_some() {
                break;
            }
            let actions = board.legal_actions().collect::<Vec<_>>();
            let action = actions[random.random_range(0..actions.len())];
            board.apply(action);
        }
        boards.push((board, Color::Black));
        boards.push((board, Color::White));
    }
    boards
}

fn search_moves(
    workspace: &mut SearchWorkspace<Board>,
    evaluator: &mut OthelloCandleEvaluator,
    boards: &mut [(Board, Color)],
    game_indices: &[usize],
    simulations: u32,
) -> Result<(), Box<dyn Error>> {
    let mut searches = game_indices
        .iter()
        .map(|&index| Mcts::new(boards[index].0, MctsConfig::default()))
        .collect::<Vec<_>>();
    workspace.run_batched(&mut searches, evaluator, simulations)?;
    for (&game_index, search) in game_indices.iter().zip(searches) {
        let action = search.best_action().ok_or("arena search found no action")?;
        boards[game_index].0.apply(action);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::game::Outcome;

    use super::*;

    #[test]
    fn optimizer_consumes_soft_policy_and_value_targets() {
        let device = Device::Cpu;
        let network = OthelloNetwork::new(&device, 7).unwrap();
        let mut policy = [0.0; 65];
        for action in Board::default().legal_actions() {
            policy[Board::action_index(action)] = 0.25;
        }
        let sample = SelfPlaySample {
            position: Board::default(),
            policy,
            outcome: Outcome::Draw,
            game: 1,
        };

        let report = train(
            &network,
            &device,
            &[sample],
            TrainingConfig {
                duration: Duration::ZERO,
                batch_size: 2,
                learning_rate: 0.001,
                seed: 11,
            },
        )
        .unwrap();

        assert_eq!(report.steps, 1);
        assert!(report.policy_loss.is_finite());
        assert!(report.value_loss.is_finite());
    }

    #[test]
    fn arena_openings_are_diverse_and_color_paired() {
        let openings = arena_openings(10, 8, 7);

        assert_eq!(openings.len(), 10);
        for pair in openings.as_chunks::<2>().0 {
            assert_eq!(pair[0].0, pair[1].0);
            assert_eq!(pair[0].1, Color::Black);
            assert_eq!(pair[1].1, Color::White);
        }
        assert!(
            openings[2..]
                .iter()
                .any(|opening| opening.0 != openings[0].0)
        );
    }

    #[test]
    fn symmetry_transforms_state_and_policy_together() {
        let mut input = [0.0; 128];
        let mut policy = [0.0; 65];
        input[8] = 1.0;
        input[64 + 8] = 2.0;
        policy[8] = 0.75;
        policy[64] = 0.25;

        apply_symmetry(&mut input, &mut policy, 1);

        assert_eq!(input[6], 1.0);
        assert_eq!(input[64 + 6], 2.0);
        assert_eq!(policy[6], 0.75);
        assert_eq!(policy[64], 0.25);
        assert_eq!(policy.iter().sum::<f32>(), 1.0);
    }
}
