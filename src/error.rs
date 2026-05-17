//! Error type and shared async aliases.

use std::future::Future;
use std::pin::Pin;

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("{0}")]
    Other(String),
    #[error("unknown tool: {0}")]
    UnknownTool(String),
    #[error("tool loop exceeded {0} iterations")]
    MaxIterations(usize),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, AgentError>;
