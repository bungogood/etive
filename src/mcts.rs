//! Arena-backed PUCT Monte Carlo tree search with split-phase evaluation.

use std::error::Error;
use std::fmt;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::evaluator::{BatchEvaluator, Evaluator};
use crate::game::Game;

static NEXT_TREE_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug)]
pub struct MctsConfig {
    pub exploration: f32,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self { exploration: 1.5 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActionStats<A> {
    pub action: A,
    pub prior: f32,
    pub visits: u32,
    pub value: f32,
}

#[derive(Debug)]
pub enum MctsError {
    EvaluationPending,
    NoEvaluationPending,
    StaleRequest,
    InvalidPolicy,
    InvalidValue(f32),
    NoLegalActions,
}

impl fmt::Display for MctsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EvaluationPending => f.write_str("an evaluation is already pending"),
            Self::NoEvaluationPending => f.write_str("no evaluation is pending"),
            Self::StaleRequest => f.write_str("evaluation request is stale"),
            Self::InvalidPolicy => f.write_str("evaluator returned invalid policy logits"),
            Self::InvalidValue(value) => write!(f, "evaluator returned invalid value {value}"),
            Self::NoLegalActions => f.write_str("non-terminal position has no legal actions"),
        }
    }
}

impl Error for MctsError {}

#[derive(Debug)]
pub enum SearchError<E> {
    Evaluator(E),
    Mcts(MctsError),
}

impl<E: fmt::Display> fmt::Display for SearchError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Evaluator(error) => write!(f, "evaluator failed: {error}"),
            Self::Mcts(error) => error.fmt(f),
        }
    }
}

impl<E: Error + 'static> Error for SearchError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Evaluator(error) => Some(error),
            Self::Mcts(error) => Some(error),
        }
    }
}

/// Opaque identity for one selected leaf awaiting evaluation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct EvaluationRequest {
    tree: u64,
    id: u64,
}

impl EvaluationRequest {
    pub const fn tree(self) -> u64 {
        self.tree
    }

    pub const fn id(self) -> u64 {
        self.id
    }
}

/// Result of one MCTS selection phase.
pub enum Selection<'a, G: Game> {
    /// A terminal leaf was backed up immediately without inference.
    Terminal,
    /// The selected position must be evaluated before this tree can continue.
    Evaluate {
        request: EvaluationRequest,
        position: &'a G,
    },
}

struct PendingEvaluation {
    request: EvaluationRequest,
    node: usize,
}

/// Runs equal simulation counts across independent trees using bounded batches.
pub fn run_batched<G, E>(
    trees: &mut [Mcts<G>],
    evaluator: &mut E,
    simulations: u32,
    max_batch_size: usize,
) -> Result<(), SearchError<E::Error>>
where
    G: Game,
    E: BatchEvaluator<G>,
{
    assert!(max_batch_size > 0, "maximum batch size must be positive");
    let mut positions: Vec<G> = Vec::with_capacity(trees.len());
    let mut requests: Vec<(usize, EvaluationRequest)> = Vec::with_capacity(trees.len());
    let mut policy_logits = Vec::new();
    let mut values = Vec::new();

    for _ in 0..simulations {
        positions.clear();
        requests.clear();
        for tree_index in 0..trees.len() {
            if trees[tree_index].root_position().outcome().is_some() {
                continue;
            }
            let selection = match trees[tree_index].select() {
                Ok(selection) => selection,
                Err(error) => {
                    for &(selected_tree, request) in &requests {
                        trees[selected_tree].cancel(request);
                    }
                    return Err(SearchError::Mcts(error));
                }
            };
            match selection {
                Selection::Terminal => {}
                Selection::Evaluate { request, position } => {
                    positions.push(*position);
                    requests.push((tree_index, request));
                }
            }
        }
        if positions.is_empty() {
            continue;
        }

        policy_logits.resize(positions.len() * G::ACTION_COUNT, 0.0);
        values.resize(positions.len(), 0.0);
        for start in (0..positions.len()).step_by(max_batch_size) {
            let end = (start + max_batch_size).min(positions.len());
            if let Err(error) = evaluator.evaluate_batch(
                &positions[start..end],
                &mut policy_logits[start * G::ACTION_COUNT..end * G::ACTION_COUNT],
                &mut values[start..end],
            ) {
                for &(tree_index, request) in &requests {
                    trees[tree_index].cancel(request);
                }
                return Err(SearchError::Evaluator(error));
            }
        }

        for (index, &(tree_index, request)) in requests.iter().enumerate() {
            let policy_start = index * G::ACTION_COUNT;
            if let Err(error) = trees[tree_index].complete(
                request,
                &policy_logits[policy_start..policy_start + G::ACTION_COUNT],
                values[index],
            ) {
                for &(waiting_tree, waiting_request) in &requests[index + 1..] {
                    trees[waiting_tree].cancel(waiting_request);
                }
                return Err(SearchError::Mcts(error));
            }
        }
    }
    Ok(())
}

pub struct Mcts<G: Game> {
    config: MctsConfig,
    nodes: Vec<Node<G>>,
    edges: Vec<Edge<G::Action>>,
    root: usize,
    policy_logits: Vec<f32>,
    path: Vec<(usize, usize)>,
    pending: Option<PendingEvaluation>,
    tree_id: u64,
    next_request: u64,
}

impl<G: Game> Mcts<G> {
    pub fn new(position: G, config: MctsConfig) -> Self {
        assert!(config.exploration.is_finite() && config.exploration >= 0.0);
        Self {
            config,
            nodes: vec![Node::new(position)],
            edges: Vec::new(),
            root: 0,
            policy_logits: vec![0.0; G::ACTION_COUNT],
            path: Vec::new(),
            pending: None,
            tree_id: NEXT_TREE_ID.fetch_add(1, Ordering::Relaxed),
            next_request: 0,
        }
    }

    pub fn run<E: Evaluator<G>>(
        &mut self,
        evaluator: &mut E,
        simulations: u32,
    ) -> Result<(), SearchError<E::Error>> {
        let mut policy_logits = std::mem::take(&mut self.policy_logits);
        let result = (|| {
            for _ in 0..simulations {
                match self.select().map_err(SearchError::Mcts)? {
                    Selection::Terminal => {}
                    Selection::Evaluate { request, position } => {
                        let value = match evaluator.evaluate(position, &mut policy_logits) {
                            Ok(value) => value,
                            Err(error) => {
                                self.cancel(request);
                                return Err(SearchError::Evaluator(error));
                            }
                        };
                        self.complete(request, &policy_logits, value)
                            .map_err(SearchError::Mcts)?;
                    }
                }
            }
            Ok(())
        })();
        self.policy_logits = policy_logits;
        result
    }

    /// Selects one leaf, immediately backing up terminal positions.
    pub fn select(&mut self) -> Result<Selection<'_, G>, MctsError> {
        if self.pending.is_some() {
            return Err(MctsError::EvaluationPending);
        }
        self.path.clear();
        let mut node_index = self.root;

        loop {
            match self.nodes[node_index].state {
                NodeState::Unexpanded => {
                    if let Some(outcome) = self.nodes[node_index].position.outcome() {
                        let value = outcome.value();
                        self.nodes[node_index].state = NodeState::Terminal(value);
                        self.backup(node_index, value);
                        self.path.clear();
                        return Ok(Selection::Terminal);
                    }
                    let request = EvaluationRequest {
                        tree: self.tree_id,
                        id: self.next_request,
                    };
                    self.next_request = self.next_request.wrapping_add(1);
                    self.pending = Some(PendingEvaluation {
                        request,
                        node: node_index,
                    });
                    return Ok(Selection::Evaluate {
                        request,
                        position: &self.nodes[node_index].position,
                    });
                }
                NodeState::Terminal(value) => {
                    self.backup(node_index, value);
                    self.path.clear();
                    return Ok(Selection::Terminal);
                }
                NodeState::Expanded { start, count } => {
                    let edge_index = self.select_edge(node_index, start..start + count);
                    let child = self.materialize_child(node_index, edge_index);
                    self.path.push((node_index, edge_index));
                    node_index = child;
                }
            }
        }
    }

    /// Completes a pending leaf evaluation, then expands and backs it up.
    pub fn complete(
        &mut self,
        request: EvaluationRequest,
        policy_logits: &[f32],
        value: f32,
    ) -> Result<(), MctsError> {
        let pending = self
            .pending
            .as_ref()
            .ok_or(MctsError::NoEvaluationPending)?;
        if pending.request != request {
            return Err(MctsError::StaleRequest);
        }
        let node_index = pending.node;
        let result = self.expand(node_index, policy_logits, value);
        self.pending = None;
        if result.is_ok() {
            self.backup(node_index, value);
        }
        self.path.clear();
        result
    }

    /// Cancels the matching pending evaluation without changing tree statistics.
    pub fn cancel(&mut self, request: EvaluationRequest) -> bool {
        if self.pending.as_ref().map(|pending| pending.request) != Some(request) {
            return false;
        }
        self.pending = None;
        self.path.clear();
        true
    }

    pub const fn is_pending(&self) -> bool {
        self.pending.is_some()
    }

    pub fn root_position(&self) -> &G {
        &self.nodes[self.root].position
    }

    pub fn root_value(&self) -> f32 {
        let root = &self.nodes[self.root];
        if root.visits == 0 {
            0.0
        } else {
            root.value_sum / root.visits as f32
        }
    }

    pub fn root_stats(&self) -> impl ExactSizeIterator<Item = ActionStats<G::Action>> + '_ {
        let range = self.nodes[self.root].edge_range().unwrap_or(0..0);
        self.edges[range].iter().map(|edge| ActionStats {
            action: edge.action,
            prior: edge.prior,
            visits: edge.visits,
            value: edge.mean_value(),
        })
    }

    pub fn best_action(&self) -> Option<G::Action> {
        self.root_stats()
            .reduce(|best, candidate| {
                if candidate.visits > best.visits
                    || (candidate.visits == best.visits && candidate.prior > best.prior)
                {
                    candidate
                } else {
                    best
                }
            })
            .map(|stats| stats.action)
    }

    /// Advances to a legal child, retaining its subtree and reclaiming siblings.
    pub fn advance(&mut self, action: G::Action) -> bool {
        if self.pending.is_some() {
            return false;
        }
        let Some(range) = self.nodes[self.root].edge_range() else {
            return false;
        };
        let Some(edge_index) = range
            .into_iter()
            .find(|&index| self.edges[index].action == action)
        else {
            return false;
        };
        let child = self.materialize_child(self.root, edge_index);
        self.root = child;
        self.compact();
        true
    }

    fn compact(&mut self) {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        copy_subtree(&self.nodes, &self.edges, self.root, &mut nodes, &mut edges);
        self.nodes = nodes;
        self.edges = edges;
        self.root = 0;
        self.path.clear();
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    fn expand(
        &mut self,
        node_index: usize,
        policy_logits: &[f32],
        value: f32,
    ) -> Result<(), MctsError> {
        if policy_logits.len() != G::ACTION_COUNT {
            return Err(MctsError::InvalidPolicy);
        }
        if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
            return Err(MctsError::InvalidValue(value));
        }

        let mut max_logit = f32::NEG_INFINITY;
        let mut legal_count = 0;
        for action in self.nodes[node_index].position.legal_actions() {
            let logit = policy_logits[G::action_index(action)];
            if !logit.is_finite() {
                return Err(MctsError::InvalidPolicy);
            }
            max_logit = max_logit.max(logit);
            legal_count += 1;
        }
        if legal_count == 0 {
            return Err(MctsError::NoLegalActions);
        }

        let mut policy_sum = 0.0;
        for action in self.nodes[node_index].position.legal_actions() {
            policy_sum += (policy_logits[G::action_index(action)] - max_logit).exp();
        }
        if !policy_sum.is_finite() || policy_sum <= 0.0 {
            return Err(MctsError::InvalidPolicy);
        }

        let start = self.edges.len();
        let position = &self.nodes[node_index].position;
        let logits = policy_logits;
        let edges = &mut self.edges;
        for action in position.legal_actions() {
            let prior = (logits[G::action_index(action)] - max_logit).exp() / policy_sum;
            edges.push(Edge::new(action, prior));
        }
        self.nodes[node_index].state = NodeState::Expanded {
            start,
            count: legal_count,
        };
        Ok(())
    }

    fn select_edge(&self, node_index: usize, range: Range<usize>) -> usize {
        let parent_scale = (self.nodes[node_index].visits.max(1) as f32).sqrt();
        let mut selected = range.start;
        let mut selected_score = f32::NEG_INFINITY;
        for edge_index in range {
            let edge = &self.edges[edge_index];
            let exploration =
                self.config.exploration * edge.prior * parent_scale / (1 + edge.visits) as f32;
            let score = edge.mean_value() + exploration;
            if score > selected_score {
                selected = edge_index;
                selected_score = score;
            }
        }
        selected
    }

    fn materialize_child(&mut self, node_index: usize, edge_index: usize) -> usize {
        if let Some(child) = self.edges[edge_index].child {
            return child;
        }
        let mut position = self.nodes[node_index].position;
        position.apply(self.edges[edge_index].action);
        let child = self.nodes.len();
        self.nodes.push(Node::new(position));
        self.edges[edge_index].child = Some(child);
        child
    }

    fn backup(&mut self, leaf: usize, mut value: f32) {
        self.nodes[leaf].record(value);
        for &(node_index, edge_index) in self.path.iter().rev() {
            value = -value;
            self.edges[edge_index].record(value);
            self.nodes[node_index].record(value);
        }
    }
}

struct Node<G> {
    position: G,
    state: NodeState,
    visits: u32,
    value_sum: f32,
}

impl<G> Node<G> {
    fn new(position: G) -> Self {
        Self {
            position,
            state: NodeState::Unexpanded,
            visits: 0,
            value_sum: 0.0,
        }
    }

    fn edge_range(&self) -> Option<Range<usize>> {
        match self.state {
            NodeState::Expanded { start, count } => Some(start..start + count),
            _ => None,
        }
    }

    fn record(&mut self, value: f32) {
        self.visits += 1;
        self.value_sum += value;
    }
}

#[derive(Clone, Copy)]
enum NodeState {
    Unexpanded,
    Expanded { start: usize, count: usize },
    Terminal(f32),
}

struct Edge<A> {
    action: A,
    prior: f32,
    visits: u32,
    value_sum: f32,
    child: Option<usize>,
}

impl<A> Edge<A> {
    fn new(action: A, prior: f32) -> Self {
        Self {
            action,
            prior,
            visits: 0,
            value_sum: 0.0,
            child: None,
        }
    }

    fn mean_value(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }

    fn record(&mut self, value: f32) {
        self.visits += 1;
        self.value_sum += value;
    }
}

fn copy_subtree<G: Game>(
    source_nodes: &[Node<G>],
    source_edges: &[Edge<G::Action>],
    source_index: usize,
    nodes: &mut Vec<Node<G>>,
    edges: &mut Vec<Edge<G::Action>>,
) -> usize {
    let source = &source_nodes[source_index];
    let target_index = nodes.len();
    nodes.push(Node {
        position: source.position,
        state: source.state,
        visits: source.visits,
        value_sum: source.value_sum,
    });

    let Some(source_range) = source.edge_range() else {
        return target_index;
    };
    let target_start = edges.len();
    for source_edge in &source_edges[source_range.clone()] {
        edges.push(Edge {
            action: source_edge.action,
            prior: source_edge.prior,
            visits: source_edge.visits,
            value_sum: source_edge.value_sum,
            child: None,
        });
    }
    nodes[target_index].state = NodeState::Expanded {
        start: target_start,
        count: source_range.len(),
    };

    for (offset, source_edge) in source_edges[source_range].iter().enumerate() {
        if let Some(source_child) = source_edge.child {
            let target_child = copy_subtree(source_nodes, source_edges, source_child, nodes, edges);
            edges[target_start + offset].child = Some(target_child);
        }
    }
    target_index
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use super::*;
    use candle_core::Device;

    use crate::evaluator::{
        BatchEvaluator, TicTacToeCandleEvaluator, TicTacToeMinimaxEvaluator, UniformEvaluator,
    };
    use crate::tic_tac_toe::{Board, Square};

    fn square(index: usize) -> Square {
        Square::from_index(index).unwrap()
    }

    fn position(actions: &[usize]) -> Board {
        let mut board = Board::default();
        for &action in actions {
            board.play(square(action));
        }
        board
    }

    #[test]
    fn expansion_appends_one_contiguous_edge_range() {
        let mut search = Mcts::new(Board::default(), MctsConfig::default());
        search.run(&mut UniformEvaluator, 1).unwrap();

        assert_eq!(search.node_count(), 1);
        assert_eq!(search.edge_count(), 9);
        assert_eq!(search.nodes[0].edge_range(), Some(0..9));
        assert!((search.root_stats().map(|stats| stats.prior).sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn split_phase_allows_one_pending_leaf_per_tree() {
        let mut search = Mcts::new(Board::default(), MctsConfig::default());
        let request = match search.select().unwrap() {
            Selection::Evaluate { request, position } => {
                assert_eq!(position, &Board::default());
                request
            }
            Selection::Terminal => panic!("initial position is not terminal"),
        };

        assert!(search.is_pending());
        assert!(matches!(search.select(), Err(MctsError::EvaluationPending)));
        assert!(!search.advance(square(0)));

        search.complete(request, &[0.0; 9], 0.0).unwrap();
        assert!(!search.is_pending());
        assert_eq!(search.edge_count(), 9);
        assert_eq!(search.nodes[search.root].visits, 1);
    }

    #[test]
    fn independent_trees_can_fill_an_inference_batch() {
        let mut trees = (0..32)
            .map(|_| Mcts::new(Board::default(), MctsConfig::default()))
            .collect::<Vec<_>>();
        let requests = trees
            .iter_mut()
            .map(|tree| match tree.select().unwrap() {
                Selection::Evaluate { request, .. } => request,
                Selection::Terminal => unreachable!(),
            })
            .collect::<Vec<_>>();

        assert!(trees.iter().all(Mcts::is_pending));
        for (tree, request) in trees.iter_mut().zip(requests) {
            tree.complete(request, &[0.0; 9], 0.0).unwrap();
        }
        assert!(trees.iter().all(|tree| !tree.is_pending()));
        assert!(trees.iter().all(|tree| tree.edge_count() == 9));
    }

    #[test]
    fn candle_batch_completes_independent_pending_trees() {
        let mut trees = (0..32)
            .map(|_| Mcts::new(Board::default(), MctsConfig::default()))
            .collect::<Vec<_>>();
        let mut positions = Vec::with_capacity(trees.len());
        let requests = trees
            .iter_mut()
            .map(|tree| match tree.select().unwrap() {
                Selection::Evaluate { request, position } => {
                    positions.push(*position);
                    request
                }
                Selection::Terminal => unreachable!(),
            })
            .collect::<Vec<_>>();
        let mut evaluator = TicTacToeCandleEvaluator::new(Device::Cpu, 7).unwrap();
        let mut policies = vec![0.0; positions.len() * Board::ACTION_COUNT];
        let mut values = vec![0.0; positions.len()];

        evaluator
            .evaluate_batch(&positions, &mut policies, &mut values)
            .unwrap();
        for (index, (tree, request)) in trees.iter_mut().zip(requests).enumerate() {
            let start = index * Board::ACTION_COUNT;
            tree.complete(
                request,
                &policies[start..start + Board::ACTION_COUNT],
                values[index],
            )
            .unwrap();
        }

        assert_eq!(evaluator.batches(), 1);
        assert_eq!(evaluator.evaluations(), 32);
        assert!(trees.iter().all(|tree| !tree.is_pending()));
        assert!(trees.iter().all(|tree| tree.edge_count() == 9));
    }

    #[test]
    fn batched_scheduler_matches_synchronous_candle_search() {
        let mut synchronous_evaluator = TicTacToeCandleEvaluator::new(Device::Cpu, 7).unwrap();
        let mut synchronous = Mcts::new(Board::default(), MctsConfig::default());
        synchronous.run(&mut synchronous_evaluator, 128).unwrap();

        let mut batched_evaluator = TicTacToeCandleEvaluator::new(Device::Cpu, 7).unwrap();
        let mut batched = (0..32)
            .map(|_| Mcts::new(Board::default(), MctsConfig::default()))
            .collect::<Vec<_>>();
        run_batched(&mut batched, &mut batched_evaluator, 128, 16).unwrap();

        let expected = batched[0].root_stats().collect::<Vec<_>>();
        assert!(
            batched
                .iter()
                .all(|tree| tree.root_stats().collect::<Vec<_>>() == expected)
        );
        assert_eq!(batched[0].best_action(), synchronous.best_action());
        assert_eq!(expected.iter().map(|stats| stats.visits).sum::<u32>(), 127);
        assert_eq!(
            batched_evaluator.evaluations(),
            batched_evaluator.batches() * 16
        );
    }

    #[test]
    fn stale_completion_does_not_disturb_the_live_request() {
        let mut search = Mcts::new(Board::default(), MctsConfig::default());
        let first = match search.select().unwrap() {
            Selection::Evaluate { request, .. } => request,
            Selection::Terminal => unreachable!(),
        };
        assert!(search.cancel(first));
        let second = match search.select().unwrap() {
            Selection::Evaluate { request, .. } => request,
            Selection::Terminal => unreachable!(),
        };

        assert!(matches!(
            search.complete(first, &[0.0; 9], 0.0),
            Err(MctsError::StaleRequest)
        ));
        assert!(search.is_pending());
        search.complete(second, &[0.0; 9], 0.0).unwrap();
        assert!(!search.is_pending());
    }

    #[test]
    fn requests_are_scoped_to_one_tree() {
        let mut first = Mcts::new(Board::default(), MctsConfig::default());
        let mut second = Mcts::new(Board::default(), MctsConfig::default());
        let first_request = match first.select().unwrap() {
            Selection::Evaluate { request, .. } => request,
            Selection::Terminal => unreachable!(),
        };
        let second_request = match second.select().unwrap() {
            Selection::Evaluate { request, .. } => request,
            Selection::Terminal => unreachable!(),
        };

        assert_ne!(first_request.tree(), second_request.tree());
        assert!(matches!(
            second.complete(first_request, &[0.0; 9], 0.0),
            Err(MctsError::StaleRequest)
        ));
        assert!(second.is_pending());
        second.complete(second_request, &[0.0; 9], 0.0).unwrap();
    }

    #[test]
    fn split_and_synchronous_search_produce_identical_statistics() {
        let mut synchronous = Mcts::new(Board::default(), MctsConfig::default());
        synchronous.run(&mut UniformEvaluator, 128).unwrap();

        let mut split = Mcts::new(Board::default(), MctsConfig::default());
        for _ in 0..128 {
            match split.select().unwrap() {
                Selection::Terminal => {}
                Selection::Evaluate { request, .. } => {
                    split.complete(request, &[0.0; 9], 0.0).unwrap();
                }
            }
        }

        assert_eq!(split.root_value(), synchronous.root_value());
        assert_eq!(
            split.root_stats().collect::<Vec<_>>(),
            synchronous.root_stats().collect::<Vec<_>>()
        );
        assert_eq!(split.node_count(), synchronous.node_count());
        assert_eq!(split.edge_count(), synchronous.edge_count());
    }

    #[test]
    fn evaluator_failure_releases_the_pending_leaf() {
        struct FailingEvaluator;

        impl Evaluator<Board> for FailingEvaluator {
            type Error = &'static str;

            fn evaluate(
                &mut self,
                _game: &Board,
                _policy_logits: &mut [f32],
            ) -> Result<f32, Self::Error> {
                Err("failed")
            }
        }

        let mut search = Mcts::new(Board::default(), MctsConfig::default());
        assert!(matches!(
            search.run(&mut FailingEvaluator, 1),
            Err(SearchError::Evaluator("failed"))
        ));
        assert!(!search.is_pending());
        assert!(matches!(search.select(), Ok(Selection::Evaluate { .. })));
    }

    #[test]
    fn search_finds_an_immediate_win_and_backs_it_up_positively() {
        let board = position(&[0, 3, 1, 4]);
        let mut search = Mcts::new(board, MctsConfig::default());
        search.run(&mut UniformEvaluator, 128).unwrap();

        assert_eq!(search.best_action(), Some(square(2)));
        let winning = search
            .root_stats()
            .find(|stats| stats.action == square(2))
            .unwrap();
        assert!(winning.value > 0.9);
        assert!(winning.visits > 0);
    }

    #[test]
    fn minimax_policy_blocks_a_forced_loss() {
        let board = position(&[0, 4, 1]);
        let mut search = Mcts::new(board, MctsConfig::default());
        search.run(&mut TicTacToeMinimaxEvaluator, 64).unwrap();

        assert_eq!(search.best_action(), Some(square(2)));
        assert!(search.root_value().abs() < 0.1);
    }

    #[test]
    fn terminal_nodes_bypass_the_evaluator() {
        struct CountingEvaluator {
            calls: usize,
        }

        impl Evaluator<Board> for CountingEvaluator {
            type Error = Infallible;

            fn evaluate(
                &mut self,
                _game: &Board,
                policy_logits: &mut [f32],
            ) -> Result<f32, Self::Error> {
                self.calls += 1;
                policy_logits.fill(0.0);
                Ok(0.0)
            }
        }

        let board = position(&[0, 3, 1, 4, 2]);
        let mut evaluator = CountingEvaluator { calls: 0 };
        let mut search = Mcts::new(board, MctsConfig::default());
        search.run(&mut evaluator, 8).unwrap();

        assert_eq!(evaluator.calls, 0);
        assert_eq!(search.node_count(), 1);
        assert_eq!(search.edge_count(), 0);
        assert_eq!(search.root_value(), -1.0);
    }

    #[test]
    fn random_candle_search_backs_up_exact_terminal_values() {
        let board = position(&[0, 3, 1, 4]);
        let mut evaluator = TicTacToeCandleEvaluator::new(Device::Cpu, 7).unwrap();
        let mut search = Mcts::new(board, MctsConfig::default());
        search.run(&mut evaluator, 256).unwrap();

        let winning = search
            .root_stats()
            .find(|stats| stats.action == square(2))
            .unwrap();
        assert_eq!(winning.value, 1.0);
        assert_eq!(search.best_action(), Some(square(2)));
        assert!(evaluator.evaluations() > 0);
    }

    #[test]
    fn terminal_root_bypasses_the_candle_network() {
        let board = position(&[0, 3, 1, 4, 2]);
        let mut evaluator = TicTacToeCandleEvaluator::new(Device::Cpu, 7).unwrap();
        let mut search = Mcts::new(board, MctsConfig::default());
        search.run(&mut evaluator, 8).unwrap();

        assert_eq!(evaluator.evaluations(), 0);
        assert_eq!(search.root_value(), -1.0);
    }

    #[test]
    fn advance_retains_the_chosen_subtree_and_reclaims_siblings() {
        let mut search = Mcts::new(Board::default(), MctsConfig::default());
        search.run(&mut TicTacToeMinimaxEvaluator, 256).unwrap();
        let action = search.best_action().unwrap();
        let mut expected = *search.root_position();
        expected.play(action);
        let nodes_before = search.node_count();
        let edges_before = search.edge_count();
        let root_range = search.nodes[search.root].edge_range().unwrap();
        let selected_edge = root_range
            .into_iter()
            .find(|&index| search.edges[index].action == action)
            .unwrap();
        let selected_child = search.edges[selected_edge].child.unwrap();
        let retained_visits = search.nodes[selected_child].visits;
        let retained_value = search.nodes[selected_child].value_sum;

        assert!(search.advance(action));

        assert_eq!(search.root_position(), &expected);
        assert_eq!(search.root, 0);
        assert_eq!(search.nodes[0].visits, retained_visits);
        assert_eq!(search.nodes[0].value_sum, retained_value);
        assert!(search.node_count() < nodes_before);
        assert!(search.edge_count() < edges_before);
    }
}
