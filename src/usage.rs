//! Usage metrics for agent responses and streams.

use serde::{Deserialize, Serialize};

/// Token counts, timing, and cost data for an agent call.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct UsageMetrics {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    pub estimated_cost: Option<f64>,
    pub processing_time_ms: Option<u128>,
    /// Time-to-first-token from start of generation; only populated by streaming backends.
    pub ttft_ms: Option<u128>,
}

impl UsageMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_tokens(mut self, input: u64, output: u64) -> Self {
        self.input_tokens = Some(input);
        self.output_tokens = Some(output);
        self.total_tokens = Some(input + output);
        self
    }

    pub fn with_processing_time_ms(mut self, ms: u128) -> Self {
        self.processing_time_ms = Some(ms);
        self
    }

    pub fn with_ttft_ms(mut self, ms: u128) -> Self {
        self.ttft_ms = Some(ms);
        self
    }
}
