//! Tracing wrapper for any [`Agent`]. Emits `tracing` events on every call —
//! start, completion (with response shape), and errors — so callers control
//! verbosity via a [`tracing::Subscriber`]. No prints, no levels on the type.

use std::time::Instant;

use crate::{Agent, BoxFuture, Context, Response, Result};

/// Wraps an [`Agent`] and emits lifecycle events through `tracing`.
///
/// ```ignore
/// let agent = LoggingAgent::new("local", inner);
/// // requires a `tracing_subscriber` installed by the caller
/// ```
pub struct LoggingAgent<A> {
    inner: A,
    label: String,
}

impl<A> LoggingAgent<A> {
    pub fn new(label: impl Into<String>, inner: A) -> Self {
        Self {
            inner,
            label: label.into(),
        }
    }
}

impl<A: Agent + 'static> Agent for LoggingAgent<A> {
    fn run<'a>(&'a self, ctx: Context) -> BoxFuture<'a, Result<Response>> {
        let msg_count = ctx.messages.len();
        let label = self.label.clone();
        let fut = self.inner.run(ctx);
        Box::pin(async move {
            tracing::info!(agent = %label, messages = msg_count, "agent.run start");
            let start = Instant::now();
            let result = fut.await;
            let elapsed = start.elapsed();
            match &result {
                Ok(Response::Message(s)) => {
                    tracing::info!(
                        agent = %label,
                        elapsed_ms = elapsed.as_millis() as u64,
                        chars = s.len(),
                        "agent.run -> message"
                    );
                    tracing::debug!(agent = %label, body = %s, "message body");
                }
                Ok(Response::ToolCalls(calls)) => {
                    let names: Vec<&str> = calls.iter().map(|c| c.name.as_str()).collect();
                    tracing::info!(
                        agent = %label,
                        elapsed_ms = elapsed.as_millis() as u64,
                        calls = ?names,
                        "agent.run -> tool_calls"
                    );
                    for c in calls {
                        tracing::debug!(
                            agent = %label,
                            tool = %c.name,
                            args = %c.arguments,
                            "tool call args"
                        );
                    }
                }
                Err(e) => tracing::error!(
                    agent = %label,
                    elapsed_ms = elapsed.as_millis() as u64,
                    error = %e,
                    "agent.run -> error"
                ),
            }
            result
        })
    }
}
