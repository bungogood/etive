//! Shared policy/value metrics and CSV output.

use std::ops::Sub;

/// Aggregate policy and value losses with KL derived from its source values.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PolicyValueMetrics<T> {
    pub policy_cross_entropy: T,
    pub policy_target_entropy: T,
    pub value_mse: T,
}

impl<T: Copy + Sub<Output = T>> PolicyValueMetrics<T> {
    pub fn new(policy_cross_entropy: T, policy_target_entropy: T, value_mse: T) -> Self {
        Self {
            policy_cross_entropy,
            policy_target_entropy,
            value_mse,
        }
    }

    pub fn policy_kl(self) -> T {
        self.policy_cross_entropy - self.policy_target_entropy
    }
}
