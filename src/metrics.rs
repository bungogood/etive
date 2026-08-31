//! Shared policy/value metrics and CSV output.

use std::io::Write;
use std::ops::Sub;

use serde::Serialize;

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

    pub fn map<U: Copy + Sub<Output = U>>(
        self,
        mut map: impl FnMut(T) -> U,
    ) -> PolicyValueMetrics<U> {
        PolicyValueMetrics::new(
            map(self.policy_cross_entropy),
            map(self.policy_target_entropy),
            map(self.value_mse),
        )
    }
}

impl<T: Copy + Sub<Output = T>> PolicyValueMetrics<T> {
    pub fn policy_kl(self) -> T {
        self.policy_cross_entropy - self.policy_target_entropy
    }
}

impl From<PolicyValueMetrics<f32>> for PolicyValueMetrics<f64> {
    fn from(metrics: PolicyValueMetrics<f32>) -> Self {
        metrics.map(f64::from)
    }
}

/// Writes one typed CSV record, including its serde-derived header.
pub fn write_csv<W: Write, T: Serialize>(writer: W, value: &T) -> csv::Result<()> {
    let mut writer = csv::Writer::from_writer(writer);
    writer.serialize(value)?;
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derives_kl_and_maps_precision() {
        let metrics = PolicyValueMetrics::new(2.0_f32, 1.25, 0.5);

        assert_eq!(metrics.policy_kl(), 0.75);
        assert_eq!(PolicyValueMetrics::<f64>::from(metrics).policy_kl(), 0.75);
        assert_eq!(metrics.map(|value| value * 2.0).value_mse, 1.0);
    }
}
