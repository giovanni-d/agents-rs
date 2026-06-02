//! Tracing wrapper for any [`Agent`] (and [`crate::StreamingAgent`]) that emits lifecycle events.

use std::pin::Pin;
use std::task::{Context as TaskContext, Poll};
use std::time::Instant;

use tokio_stream::Stream;

use crate::stream::{Chunk, StreamingAgent, ToolEvent};
use crate::{Agent, BoxFuture, Context, Response, Result};

/// Wraps an [`Agent`] and emits lifecycle events through `tracing`.
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

impl<A: StreamingAgent + 'static> StreamingAgent for LoggingAgent<A>
where
    A::Stream: Send + Unpin + 'static,
{
    type Stream = LoggingStream<A::Stream>;

    fn run_stream(&self, ctx: Context) -> Self::Stream {
        let messages = ctx.messages.len();
        let label = self.label.clone();
        tracing::info!(agent = %label, messages, "agent.run_stream start");
        LoggingStream {
            inner: self.inner.run_stream(ctx),
            label,
            start: Instant::now(),
            text_chars: 0,
            tools_started: 0,
            done_logged: false,
        }
    }
}

/// Stream produced by [`LoggingAgent::run_stream`]; forwards chunks verbatim and emits tracing events.
pub struct LoggingStream<S> {
    inner: S,
    label: String,
    start: Instant,
    text_chars: usize,
    tools_started: usize,
    done_logged: bool,
}

impl<S> Stream for LoggingStream<S>
where
    S: Stream<Item = Result<Chunk>> + Unpin,
{
    type Item = Result<Chunk>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.inner).poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                match &chunk {
                    Chunk::Text(t) => {
                        self.text_chars += t.chars().count();
                    }
                    Chunk::Tool(ToolEvent::Started { id, name }) => {
                        self.tools_started += 1;
                        tracing::info!(
                            agent = %self.label,
                            tool = %name,
                            %id,
                            "stream tool started"
                        );
                    }
                    _ => {}
                }
                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(Some(Err(e))) => {
                tracing::error!(
                    agent = %self.label,
                    error = %e,
                    elapsed_ms = self.start.elapsed().as_millis() as u64,
                    "agent.run_stream -> error"
                );
                Poll::Ready(Some(Err(e)))
            }
            Poll::Ready(None) => {
                if !self.done_logged {
                    self.done_logged = true;
                    tracing::info!(
                        agent = %self.label,
                        elapsed_ms = self.start.elapsed().as_millis() as u64,
                        text_chars = self.text_chars,
                        tools_started = self.tools_started,
                        "agent.run_stream -> done"
                    );
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}
