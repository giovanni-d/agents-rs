//! Type-driven structured output. Declare a JSON schema for the expected
//! response shape and parse the agent's final message into a typed value.

use serde::Deserialize;

use crate::agent::{Agent, Response};
use crate::error::{AgentError, Result};
use crate::message::Context;

/// A type the agent emits as its final message and that we parse back into `Self`.
pub trait StructuredOutput: for<'de> Deserialize<'de> + Send + Sync + 'static {
    /// JSON Schema describing the expected output. Use this in your agent's prompt
    /// or for validation.
    fn schema() -> serde_json::Value;
}

/// Runs the agent and parses its final message as `T`.
pub async fn run_structured<T: StructuredOutput>(
    agent: &dyn Agent,
    ctx: Context,
) -> Result<T> {
    match agent.run(ctx).await? {
        Response::Message(s) => serde_json::from_str(&s).map_err(Into::into),
        Response::ToolCalls(_) => Err(AgentError::Other(
            "expected final message, got tool calls".into(),
        )),
    }
}
