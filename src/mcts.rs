//! Arena-backed PUCT Monte Carlo tree search with split-phase evaluation.

use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::evaluator::BatchEvaluator;
use crate::game::Game;

mod workspace;

pub use workspace::SearchWorkspace;

static NEXT_TREE_ID: AtomicU64 = AtomicU64::new(1);
const EXPLORATION: f32 = 1.5;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActionStats<A> {
    pub action: A,
    pub prior: f32,
    pub visits: u32,
    pub value: f32,
}

#[derive(Debug, thiserror::Error)]
pub enum MctsError {
    #[error("an evaluation is already pending")]
    EvaluationPending,
    #[error("no evaluation is pending")]
    NoEvaluationPending,
    #[error("evaluation request is stale")]
    StaleRequest,
    #[error("evaluator returned invalid policy logits")]
    InvalidPolicy,
    #[error("evaluator returned invalid value {0}")]
    InvalidValue(f32),
    #[error("non-terminal position has no legal actions")]
    NoLegalActions,
}

#[derive(Debug, thiserror::Error)]
pub enum SearchError<E> {
    #[error("evaluator failed: {0}")]
    Evaluator(#[source] E),
    #[error(transparent)]
    Mcts(#[from] MctsError),
}

/// Opaque identity for one selected leaf awaiting evaluation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct EvaluationRequest {
    tree: u64,
    id: u64,
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

pub struct Mcts<G: Game> {
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
    pub fn new(position: G) -> Self {
        Self {
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

    pub fn run<E: BatchEvaluator<G>>(
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
                        let mut value = [0.0];
                        if let Err(error) = evaluator.evaluate_batch(
                            std::slice::from_ref(position),
                            &mut policy_logits,
                            &mut value,
                        ) {
                            self.cancel(request);
                            return Err(SearchError::Evaluator(error));
                        }
                        self.complete(request, &policy_logits, value[0])
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

    #[cfg(test)]
    pub(crate) fn node_count(&self) -> usize {
        self.nodes.len()
    }

    #[cfg(test)]
    pub(crate) fn edge_count(&self) -> usize {
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
        let parent_scale = ((node.visits + node.reservations).max(1) as f32).sqrt();
        let mut selected = range.start;
        let mut selected_score = f32::NEG_INFINITY;
        for edge_index in range {
            let edge = &self.edges[edge_index];
            let exploration = EXPLORATION * edge.prior * parent_scale
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
        position.play_unchecked(self.edges[edge_index].action);
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
mod tests;
