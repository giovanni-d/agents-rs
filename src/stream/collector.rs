//! [`StreamCollector`] adapts a [`StreamingAgent`] into a regular
//! [`Agent`] by concatenating `Chunk::Text` into a [`Response::Message`].
//! `Tool` and `Usage` chunks are dropped.

use tokio_stream::StreamExt;

use crate::{Agent, BoxFuture, Context, Response, Result};

use super::{Chunk, StreamingAgent};

#[derive(Clone)]
pub struct StreamCollector<S> {
    inner: S,
}

impl<S> StreamCollector<S> {
    pub fn new(inner: S) -> Self {
        Self { inner }
    }
    pub fn inner(&self) -> &S {
        &self.inner
    }
    pub fn into_inner(self) -> S {
        self.inner
    }
}

impl<S> Agent for StreamCollector<S>
where
    S: StreamingAgent + Send + Sync + 'static,
    S::Stream: Send + Unpin + 'static,
{
    fn run<'a>(&'a self, ctx: Context) -> BoxFuture<'a, Result<Response>> {
        let stream = self.inner.run_stream(ctx);
        Box::pin(async move {
            let mut s = stream;
            let mut text = String::new();
            while let Some(item) = s.next().await {
                match item? {
                    Chunk::Text(t) => text.push_str(&t),
                    Chunk::Tool(_) | Chunk::Usage(_) | Chunk::End => {}
                }
            }
            Ok(Response::Message(text))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::{CharStreamAgent, StreamingAgentExt, WordStreamAgent};

    #[tokio::test]
    async fn collects_char_stream_into_message() {
        let agent = CharStreamAgent::new("Test").collect();
        let r = agent.run(Context::new()).await.unwrap();
        match r {
            Response::Message(s) => assert_eq!(s, "Test"),
            _ => panic!("expected Message"),
        }
    }

    #[tokio::test]
    async fn collects_word_stream_with_spaces() {
        let agent = WordStreamAgent::new("Hello world").collect();
        let r = agent.run(Context::new()).await.unwrap();
        match r {
            Response::Message(s) => assert_eq!(s, "Hello world"),
            _ => panic!("expected Message"),
        }
    }

    #[tokio::test]
    async fn empty_stream_yields_empty_message() {
        let agent = CharStreamAgent::new("").collect();
        let r = agent.run(Context::new()).await.unwrap();
        match r {
            Response::Message(s) => assert_eq!(s, ""),
            _ => panic!("expected Message"),
        }
    }
}
