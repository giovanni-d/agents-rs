//! [`StreamingAgent`] — agents that emit a [`Stream`] of [`Chunk`]s.

use std::pin::Pin;

use tokio_stream::Stream;

use crate::{AgentError, Context};

use super::Chunk;

/// An agent that yields its response as a stream of [`Chunk`]s.
pub trait StreamingAgent: Send + Sync {
    type Stream: Stream<Item = Result<Chunk, AgentError>> + Send;

    fn run_stream(&self, ctx: Context) -> Self::Stream;
}

impl<S: StreamingAgent + ?Sized> StreamingAgent for Box<S> {
    type Stream = S::Stream;
    fn run_stream(&self, ctx: Context) -> Self::Stream {
        (**self).run_stream(ctx)
    }
}

impl<S: StreamingAgent + ?Sized> StreamingAgent for std::sync::Arc<S> {
    type Stream = S::Stream;
    fn run_stream(&self, ctx: Context) -> Self::Stream {
        (**self).run_stream(ctx)
    }
}

/// Type-erased streaming agent for dynamic dispatch.
pub type BoxStreamingAgent = Box<
    dyn StreamingAgent<Stream = Pin<Box<dyn Stream<Item = Result<Chunk, AgentError>> + Send>>>
        + Send
        + Sync,
>;

/// Sugar for `StreamCollector::new(streaming_agent)`.
pub trait StreamingAgentExt: StreamingAgent + Sized {
    fn collect(self) -> super::StreamCollector<Self> {
        super::StreamCollector::new(self)
    }
}

impl<S: StreamingAgent> StreamingAgentExt for S {}

/// Streams a fixed string one character at a time, then [`Chunk::End`].
#[derive(Clone, Debug)]
pub struct CharStreamAgent {
    pub text: String,
}

impl CharStreamAgent {
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

impl StreamingAgent for CharStreamAgent {
    type Stream = Pin<Box<dyn Stream<Item = Result<Chunk, AgentError>> + Send>>;

    fn run_stream(&self, _ctx: Context) -> Self::Stream {
        let chars: Vec<char> = self.text.chars().collect();
        Box::pin(tokio_stream::iter(
            chars
                .into_iter()
                .map(|c| Ok(Chunk::text(c.to_string())))
                .chain(std::iter::once(Ok(Chunk::end()))),
        ))
    }
}

/// Echoes the last user message one character at a time.
#[derive(Clone, Copy, Debug)]
pub struct StreamingEchoAgent;

impl StreamingAgent for StreamingEchoAgent {
    type Stream = Pin<Box<dyn Stream<Item = Result<Chunk, AgentError>> + Send>>;

    fn run_stream(&self, ctx: Context) -> Self::Stream {
        let text = ctx.last_user_message().unwrap_or("").to_string();
        let chars: Vec<char> = text.chars().collect();
        Box::pin(tokio_stream::iter(
            chars
                .into_iter()
                .map(|c| Ok(Chunk::text(c.to_string())))
                .chain(std::iter::once(Ok(Chunk::end()))),
        ))
    }
}

/// Streams the given text word-by-word.
#[derive(Clone, Debug)]
pub struct WordStreamAgent {
    text: String,
}

impl WordStreamAgent {
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

impl StreamingAgent for WordStreamAgent {
    type Stream = Pin<Box<dyn Stream<Item = Result<Chunk, AgentError>> + Send>>;

    fn run_stream(&self, _ctx: Context) -> Self::Stream {
        let words: Vec<String> = self.text.split_whitespace().map(String::from).collect();
        Box::pin(tokio_stream::iter(
            words
                .into_iter()
                .enumerate()
                .map(|(i, w)| {
                    if i == 0 {
                        Ok(Chunk::text(w))
                    } else {
                        Ok(Chunk::text(format!(" {w}")))
                    }
                })
                .chain(std::iter::once(Ok(Chunk::end()))),
        ))
    }
}

/// Always yields a single `Err` then closes.
#[derive(Clone, Debug)]
pub struct FailingStreamAgent {
    error: String,
}

impl FailingStreamAgent {
    pub fn new(error: impl Into<String>) -> Self {
        Self {
            error: error.into(),
        }
    }
}

impl StreamingAgent for FailingStreamAgent {
    type Stream = Pin<Box<dyn Stream<Item = Result<Chunk, AgentError>> + Send>>;

    fn run_stream(&self, _ctx: Context) -> Self::Stream {
        let error = AgentError::Other(self.error.clone());
        Box::pin(tokio_stream::iter(std::iter::once(Err(error))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_stream::StreamExt;

    async fn collect_chunks(
        mut s: Pin<Box<dyn Stream<Item = Result<Chunk, AgentError>> + Send>>,
    ) -> Vec<Chunk> {
        let mut out = Vec::new();
        while let Some(item) = s.next().await {
            out.push(item.unwrap());
        }
        out
    }

    #[tokio::test]
    async fn char_stream_agent() {
        let chunks = collect_chunks(CharStreamAgent::new("Hi").run_stream(Context::new())).await;
        assert_eq!(
            chunks,
            vec![Chunk::text("H"), Chunk::text("i"), Chunk::end()]
        );
    }

    #[tokio::test]
    async fn streaming_echo_agent() {
        let chunks =
            collect_chunks(StreamingEchoAgent.run_stream(Context::new().with_user("AB"))).await;
        assert_eq!(
            chunks,
            vec![Chunk::text("A"), Chunk::text("B"), Chunk::end()]
        );
    }

    #[tokio::test]
    async fn word_stream_agent() {
        let chunks =
            collect_chunks(WordStreamAgent::new("Hello world").run_stream(Context::new())).await;
        assert_eq!(
            chunks,
            vec![Chunk::text("Hello"), Chunk::text(" world"), Chunk::end()]
        );
    }

    #[tokio::test]
    async fn failing_stream_agent() {
        let mut s = FailingStreamAgent::new("boom").run_stream(Context::new());
        let first = s.next().await.unwrap();
        assert!(first.is_err());
    }

    #[tokio::test]
    async fn empty_text_stream() {
        let chunks = collect_chunks(CharStreamAgent::new("").run_stream(Context::new())).await;
        assert_eq!(chunks, vec![Chunk::end()]);
    }
}
