//! Arena-backed PUCT Monte Carlo tree search with split-phase evaluation.

use std::error::Error;
use std::fmt;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::evaluator::{BatchEvaluator, Evaluator, InferenceBatch};
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
    /// Selection reached a leaf already reserved by another request.
    Blocked,
}

struct PendingEvaluation {
    request: EvaluationRequest,
    node: usize,
    path: Vec<(usize, usize)>,
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
    SearchWorkspace::new(max_batch_size).run_batched(trees, evaluator, simulations)
}

/// Runs leaf-parallel search across one or more trees using bounded batches.
pub fn run_parallel<G, E>(
    trees: &mut [Mcts<G>],
    evaluator: &mut E,
    simulations: u32,
    max_batch_size: usize,
) -> Result<(), SearchError<E::Error>>
where
    G: Game,
    E: BatchEvaluator<G>,
{
    SearchWorkspace::new(max_batch_size).run_parallel(trees, evaluator, simulations)
}

/// Reusable scheduling and inference storage for batched MCTS calls.
pub struct SearchWorkspace<G: Game> {
    maximum: usize,
    completed: Vec<u32>,
    scheduled: Vec<u32>,
    batch: InferenceBatch<G, (usize, EvaluationRequest)>,
}

impl<G: Game> SearchWorkspace<G> {
    pub fn new(maximum: usize) -> Self {
        Self {
            maximum,
            completed: Vec::new(),
            scheduled: Vec::new(),
            batch: InferenceBatch::new(maximum),
        }
    }

    pub fn run_batched<E: BatchEvaluator<G>>(
        &mut self,
        trees: &mut [Mcts<G>],
        evaluator: &mut E,
        simulations: u32,
    ) -> Result<(), SearchError<E::Error>> {
        self.run(trees, evaluator, simulations, 1)
    }

    pub fn run_parallel<E: BatchEvaluator<G>>(
        &mut self,
        trees: &mut [Mcts<G>],
        evaluator: &mut E,
        simulations: u32,
    ) -> Result<(), SearchError<E::Error>> {
        self.run(trees, evaluator, simulations, self.maximum)
    }

    fn run<E: BatchEvaluator<G>>(
        &mut self,
        trees: &mut [Mcts<G>],
        evaluator: &mut E,
        simulations: u32,
        max_pending_per_tree: usize,
    ) -> Result<(), SearchError<E::Error>> {
        if trees.iter().any(Mcts::is_pending) {
            return Err(SearchError::Mcts(MctsError::EvaluationPending));
        }
        self.completed.resize(trees.len(), 0);
        self.completed.fill(0);
        self.scheduled.resize(trees.len(), 0);

        while self.completed.iter().any(|&count| count < simulations) {
            self.batch.clear();
            self.scheduled.fill(0);
            for tree_index in 0..trees.len() {
                while self.completed[tree_index] + self.scheduled[tree_index] < simulations
                    && (self.scheduled[tree_index] as usize) < max_pending_per_tree
                    && !self.batch.is_full()
                {
                    let selection = match trees[tree_index].select() {
                        Ok(selection) => selection,
                        Err(error) => {
                            for &(selected_tree, request) in self.batch.tags() {
                                trees[selected_tree].cancel(request);
                            }
                            return Err(SearchError::Mcts(error));
                        }
                    };
                    match selection {
                        Selection::Terminal => self.completed[tree_index] += 1,
                        Selection::Evaluate { request, position } => {
                            assert!(self.batch.push((tree_index, request), *position));
                            self.scheduled[tree_index] += 1;
                        }
                        Selection::Blocked => break,
                    }
                }
                if self.batch.is_full() {
                    break;
                }
            }
            if self.batch.is_empty() {
                debug_assert!(self.completed.iter().all(|&count| count == simulations));
                break;
            }

            if let Err(error) = self.batch.evaluate(evaluator) {
                for &(tree_index, request) in self.batch.tags() {
                    trees[tree_index].cancel(request);
                }
                return Err(SearchError::Evaluator(error));
            }

            for index in (0..self.batch.len()).rev() {
                let (&(tree_index, request), policy, value) = self.batch.result(index);
                if let Err(error) = trees[tree_index].complete(request, policy, value) {
                    for &(waiting_tree, waiting_request) in self.batch.tags().take(index) {
                        trees[waiting_tree].cancel(waiting_request);
                    }
                    return Err(SearchError::Mcts(error));
                }
                self.completed[tree_index] += 1;
            }
        }
        Ok(())
    }
}

pub struct Mcts<G: Game> {
    config: MctsConfig,
    nodes: Vec<Node<G>>,
    edges: Vec<Edge<G::Action>>,
    root: usize,
    policy_logits: Vec<f32>,
    pending: Vec<PendingEvaluation>,
    free_paths: Vec<Vec<(usize, usize)>>,
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
            pending: Vec::new(),
            free_paths: Vec::new(),
            tree_id: NEXT_TREE_ID.fetch_add(1, Ordering::Relaxed),
            next_request: 0,
        }
    }

    pub fn run<E: Evaluator<G>>(
        &mut self,
        evaluator: &mut E,
        simulations: u32,
    ) -> Result<(), SearchError<E::Error>> {
        if self.is_pending() {
            return Err(SearchError::Mcts(MctsError::EvaluationPending));
        }
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
                    Selection::Blocked => unreachable!("synchronous search has no pending request"),
                }
            }
            Ok(())
        })();
        self.policy_logits = policy_logits;
        result
    }

    /// Selects one leaf, immediately backing up terminal positions.
    pub fn select(&mut self) -> Result<Selection<'_, G>, MctsError> {
        let mut path = self.free_paths.pop().unwrap_or_default();
        path.clear();
        let mut node_index = self.root;

        loop {
            match self.nodes[node_index].state {
                NodeState::Unexpanded => {
                    if let Some(outcome) = self.nodes[node_index].position.outcome() {
                        let value = outcome.value();
                        self.nodes[node_index].state = NodeState::Terminal(value);
                        self.backup(node_index, &path, value);
                        self.free_paths.push(path);
                        return Ok(Selection::Terminal);
                    }
                    let request = EvaluationRequest {
                        tree: self.tree_id,
                        id: self.next_request,
                    };
                    self.next_request = self.next_request.wrapping_add(1);
                    self.nodes[node_index].state = NodeState::Pending(request.id);
                    self.reserve(node_index, &path);
                    self.pending.push(PendingEvaluation {
                        request,
                        node: node_index,
                        path,
                    });
                    return Ok(Selection::Evaluate {
                        request,
                        position: &self.nodes[node_index].position,
                    });
                }
                NodeState::Terminal(value) => {
                    self.backup(node_index, &path, value);
                    self.free_paths.push(path);
                    return Ok(Selection::Terminal);
                }
                NodeState::Pending(_) => {
                    self.free_paths.push(path);
                    return Ok(Selection::Blocked);
                }
                NodeState::Expanded { start, count } => {
                    let edge_index = self.select_edge(node_index, start..start + count);
                    let child = self.materialize_child(node_index, edge_index);
                    path.push((node_index, edge_index));
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
        let pending_index = self.pending_index(request)?;
        let mut pending = self.pending.swap_remove(pending_index);
        let node_index = pending.node;
        debug_assert!(
            matches!(self.nodes[node_index].state, NodeState::Pending(id) if id == request.id)
        );
        self.release(node_index, &pending.path);
        self.nodes[node_index].state = NodeState::Unexpanded;
        let result = self.expand(node_index, policy_logits, value);
        if result.is_ok() {
            self.backup(node_index, &pending.path, value);
        }
        pending.path.clear();
        self.free_paths.push(pending.path);
        result
    }

    /// Cancels the matching pending evaluation without changing tree statistics.
    pub fn cancel(&mut self, request: EvaluationRequest) -> bool {
        let Ok(index) = self.pending_index(request) else {
            return false;
        };
        let mut pending = self.pending.swap_remove(index);
        debug_assert!(
            matches!(self.nodes[pending.node].state, NodeState::Pending(id) if id == request.id)
        );
        self.release(pending.node, &pending.path);
        self.nodes[pending.node].state = NodeState::Unexpanded;
        pending.path.clear();
        self.free_paths.push(pending.path);
        true
    }

    pub fn is_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    fn pending_index(&self, request: EvaluationRequest) -> Result<usize, MctsError> {
        if self.pending.is_empty() {
            return Err(MctsError::NoEvaluationPending);
        }
        if self.pending.last().map(|pending| pending.request) == Some(request) {
            return Ok(self.pending.len() - 1);
        }
        self.pending
            .iter()
            .position(|pending| pending.request == request)
            .ok_or(MctsError::StaleRequest)
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

    /// Mixes normalized exploration noise into the current root priors.
    pub fn mix_root_priors(&mut self, noise: &[f32], fraction: f32) -> bool {
        if self.is_pending() || !(0.0..=1.0).contains(&fraction) {
            return false;
        }
        let Some(range) = self.nodes[self.root].edge_range() else {
            return false;
        };
        if noise.len() != range.len()
            || noise.iter().any(|value| !value.is_finite() || *value < 0.0)
            || (noise.iter().sum::<f32>() - 1.0).abs() > 1e-4
        {
            return false;
        }
        for (edge, &noise) in self.edges[range].iter_mut().zip(noise) {
            edge.prior = (1.0 - fraction) * edge.prior + fraction * noise;
        }
        true
    }

    /// Advances to a legal child, retaining its subtree and reclaiming siblings.
    pub fn advance(&mut self, action: G::Action) -> bool {
        if self.is_pending() {
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

    /// Clears decision statistics at the root while retaining its descendants.
    pub fn rebase_root(&mut self) -> bool {
        if self.is_pending() {
            return false;
        }
        let range = self.nodes[self.root].edge_range();
        self.nodes[self.root].visits = 0;
        self.nodes[self.root].value_sum = 0.0;
        if let Some(range) = range {
            for edge in &mut self.edges[range] {
                edge.visits = 0;
                edge.value_sum = 0.0;
            }
        }
        true
    }

    fn compact(&mut self) {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        copy_subtree(&self.nodes, &self.edges, self.root, &mut nodes, &mut edges);
        self.nodes = nodes;
        self.edges = edges;
        self.root = 0;
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
        let node = &self.nodes[node_index];
        if node.reservations == 0 {
            let parent_scale = (node.visits.max(1) as f32).sqrt();
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
            return selected;
        }

        let parent_scale = ((node.visits + node.reservations).max(1) as f32).sqrt();
        let mut selected = range.start;
        let mut selected_score = f32::NEG_INFINITY;
        for edge_index in range {
            let edge = &self.edges[edge_index];
            let exploration = self.config.exploration * edge.prior * parent_scale
                / (1 + edge.visits + edge.reservations) as f32;
            let score = edge.reserved_mean_value() + exploration;
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

    fn reserve(&mut self, leaf: usize, path: &[(usize, usize)]) {
        self.nodes[leaf].reservations += 1;
        for &(node, edge) in path {
            self.nodes[node].reservations += 1;
            self.edges[edge].reservations += 1;
        }
    }

    fn release(&mut self, leaf: usize, path: &[(usize, usize)]) {
        self.nodes[leaf].reservations -= 1;
        for &(node, edge) in path {
            self.nodes[node].reservations -= 1;
            self.edges[edge].reservations -= 1;
        }
    }

    fn backup(&mut self, leaf: usize, path: &[(usize, usize)], mut value: f32) {
        self.nodes[leaf].record(value);
        for &(node_index, edge_index) in path.iter().rev() {
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
    reservations: u32,
}

impl<G> Node<G> {
    fn new(position: G) -> Self {
        Self {
            position,
            state: NodeState::Unexpanded,
            visits: 0,
            value_sum: 0.0,
            reservations: 0,
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
    Pending(u64),
    Expanded { start: usize, count: usize },
    Terminal(f32),
}

struct Edge<A> {
    action: A,
    prior: f32,
    visits: u32,
    value_sum: f32,
    child: Option<usize>,
    reservations: u32,
}

impl<A> Edge<A> {
    fn new(action: A, prior: f32) -> Self {
        Self {
            action,
            prior,
            visits: 0,
            value_sum: 0.0,
            child: None,
            reservations: 0,
        }
    }

    fn mean_value(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }

    fn reserved_mean_value(&self) -> f32 {
        let count = self.visits + self.reservations;
        if count == 0 {
            0.0
        } else {
            (self.value_sum - self.reservations as f32) / count as f32
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
        reservations: 0,
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
            reservations: 0,
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
    fn a_pending_root_blocks_colliding_selection_without_changing_stats() {
        let mut search = Mcts::new(Board::default(), MctsConfig::default());
        let request = match search.select().unwrap() {
            Selection::Evaluate { request, position } => {
                assert_eq!(position, &Board::default());
                request
            }
            Selection::Terminal => panic!("initial position is not terminal"),
            Selection::Blocked => unreachable!(),
        };

        assert!(search.is_pending());
        assert_eq!(search.pending_count(), 1);
        assert!(matches!(search.select().unwrap(), Selection::Blocked));
        assert_eq!(search.root_value(), 0.0);
        assert_eq!(search.nodes[search.root].visits, 0);
        assert!(!search.advance(square(0)));

        search.complete(request, &[0.0; 9], 0.0).unwrap();
        assert!(!search.is_pending());
        assert_eq!(search.edge_count(), 9);
        assert_eq!(search.nodes[search.root].visits, 1);
    }

    fn select_request(search: &mut Mcts<Board>) -> EvaluationRequest {
        match search.select().unwrap() {
            Selection::Evaluate { request, .. } => request,
            Selection::Terminal | Selection::Blocked => panic!("expected an evaluation request"),
        }
    }

    #[test]
    fn distinct_leaves_can_be_pending_and_completed_out_of_order() {
        let mut search = Mcts::new(Board::default(), MctsConfig::default());
        search.run(&mut UniformEvaluator, 1).unwrap();

        let first = select_request(&mut search);
        let second = select_request(&mut search);

        assert_eq!(search.pending_count(), 2);
        assert_ne!(search.pending[0].node, search.pending[1].node);
        assert!(search.root_stats().all(|stats| stats.visits == 0));

        search.complete(second, &[0.0; 9], 0.25).unwrap();
        assert_eq!(search.pending_count(), 1);
        search.complete(first, &[0.0; 9], -0.5).unwrap();

        assert_eq!(search.pending_count(), 0);
        assert_eq!(search.nodes[search.root].visits, 3);
        assert_eq!(
            search.root_stats().map(|stats| stats.visits).sum::<u32>(),
            2
        );
        assert!(search.nodes.iter().all(|node| node.reservations == 0));
        assert!(search.edges.iter().all(|edge| edge.reservations == 0));
    }

    #[test]
    fn cancelling_one_request_does_not_release_another() {
        let mut search = Mcts::new(Board::default(), MctsConfig::default());
        search.run(&mut UniformEvaluator, 1).unwrap();
        let first = select_request(&mut search);
        let second = select_request(&mut search);

        assert!(search.cancel(first));
        assert_eq!(search.pending_count(), 1);
        assert!(matches!(
            search.complete(first, &[0.0; 9], 0.0),
            Err(MctsError::StaleRequest)
        ));
        assert_eq!(search.pending_count(), 1);
        search.complete(second, &[0.0; 9], 0.0).unwrap();

        assert_eq!(search.pending_count(), 0);
        assert!(search.nodes.iter().all(|node| node.reservations == 0));
        assert!(search.edges.iter().all(|edge| edge.reservations == 0));
    }

    #[test]
    fn pending_requests_guard_advance_and_root_prior_mixing() {
        let mut search = Mcts::new(Board::default(), MctsConfig::default());
        search.run(&mut UniformEvaluator, 1).unwrap();
        let request = select_request(&mut search);

        assert!(!search.advance(square(0)));
        assert!(!search.mix_root_priors(&[1.0 / 9.0; 9], 0.25));

        assert!(search.cancel(request));
        assert!(search.mix_root_priors(&[1.0 / 9.0; 9], 0.25));
    }

    struct RecordingBatchEvaluator {
        batches: Vec<usize>,
        invalid: bool,
        fail: bool,
    }

    impl BatchEvaluator<Board> for RecordingBatchEvaluator {
        type Error = &'static str;

        fn evaluate_batch(
            &mut self,
            games: &[Board],
            policy_logits: &mut [f32],
            values: &mut [f32],
        ) -> Result<(), Self::Error> {
            self.batches.push(games.len());
            if self.fail {
                return Err("failed");
            }
            policy_logits.fill(0.0);
            values.fill(if self.invalid { 2.0 } else { 0.0 });
            Ok(())
        }
    }

    #[test]
    fn one_tree_fills_batches_larger_than_one_and_adds_exactly_the_target() {
        let mut trees = [Mcts::new(Board::default(), MctsConfig::default())];
        trees[0].run(&mut UniformEvaluator, 3).unwrap();
        let mut evaluator = RecordingBatchEvaluator {
            batches: Vec::new(),
            invalid: false,
            fail: false,
        };

        run_parallel(&mut trees, &mut evaluator, 12, 4).unwrap();

        assert!(evaluator.batches.iter().any(|&size| size > 1));
        assert!(evaluator.batches.iter().all(|&size| size <= 4));
        assert_eq!(trees[0].nodes[trees[0].root].visits, 15);
        assert_eq!(trees[0].pending_count(), 0);
    }

    #[test]
    fn batched_errors_cancel_every_unresolved_request() {
        for (invalid, fail) in [(false, true), (true, false)] {
            let mut trees = [Mcts::new(Board::default(), MctsConfig::default())];
            trees[0].run(&mut UniformEvaluator, 1).unwrap();
            let mut evaluator = RecordingBatchEvaluator {
                batches: Vec::new(),
                invalid,
                fail,
            };

            assert!(run_parallel(&mut trees, &mut evaluator, 4, 4).is_err());
            assert_eq!(trees[0].pending_count(), 0);
            assert!(trees[0].nodes.iter().all(|node| node.reservations == 0));
            assert!(trees[0].edges.iter().all(|edge| edge.reservations == 0));
        }
    }

    #[test]
    fn terminal_root_sync_and_batch_complete_the_same_number_of_simulations() {
        let board = position(&[0, 3, 1, 4, 2]);
        let mut synchronous = Mcts::new(board, MctsConfig::default());
        synchronous.run(&mut UniformEvaluator, 11).unwrap();
        let mut batched = [Mcts::new(board, MctsConfig::default())];
        let mut evaluator = RecordingBatchEvaluator {
            batches: Vec::new(),
            invalid: false,
            fail: false,
        };
        run_batched(&mut batched, &mut evaluator, 11, 4).unwrap();

        assert!(evaluator.batches.is_empty());
        assert_eq!(synchronous.nodes[synchronous.root].visits, 11);
        assert_eq!(batched[0].nodes[batched[0].root].visits, 11);
        assert_eq!(batched[0].root_value(), synchronous.root_value());
        assert_eq!(
            batched[0].root_stats().collect::<Vec<_>>(),
            synchronous.root_stats().collect::<Vec<_>>()
        );
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
                Selection::Blocked => unreachable!(),
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
                Selection::Blocked => unreachable!(),
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
        assert!(batched_evaluator.evaluations() <= batched_evaluator.batches() * 16);
    }

    #[test]
    fn stale_completion_does_not_disturb_the_live_request() {
        let mut search = Mcts::new(Board::default(), MctsConfig::default());
        let first = match search.select().unwrap() {
            Selection::Evaluate { request, .. } => request,
            Selection::Terminal => unreachable!(),
            Selection::Blocked => unreachable!(),
        };
        assert!(search.cancel(first));
        let second = match search.select().unwrap() {
            Selection::Evaluate { request, .. } => request,
            Selection::Terminal => unreachable!(),
            Selection::Blocked => unreachable!(),
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
            Selection::Blocked => unreachable!(),
        };
        let second_request = match second.select().unwrap() {
            Selection::Evaluate { request, .. } => request,
            Selection::Terminal => unreachable!(),
            Selection::Blocked => unreachable!(),
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
                Selection::Blocked => unreachable!(),
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

    #[test]
    fn rebasing_clears_root_statistics_but_retains_the_subtree() {
        let mut search = Mcts::new(Board::default(), MctsConfig::default());
        search.run(&mut UniformEvaluator, 64).unwrap();
        let action = search.best_action().unwrap();
        assert!(search.advance(action));
        let nodes = search.node_count();
        let edges = search.edge_count();
        assert!(search.nodes[search.root].visits > 0);

        assert!(search.rebase_root());

        assert_eq!(search.node_count(), nodes);
        assert_eq!(search.edge_count(), edges);
        assert_eq!(search.root_value(), 0.0);
        assert!(search.root_stats().all(|stats| stats.visits == 0));
    }
}
