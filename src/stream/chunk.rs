//! Streaming chunk types: one emission on a streaming agent's channel.

use serde::{Deserialize, Serialize};

use crate::UsageMetrics;

/// One piece of a streaming response.
#[derive(Clone, Debug, PartialEq)]
pub enum Chunk {
    Text(String),
    Tool(ToolEvent),
    /// Usage / timing update. May be emitted multiple times during a run.
    Usage(UsageMetrics),
    /// Terminal marker — no further chunks follow.
    End,
}

impl Chunk {
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text(s.into())
    }
    pub fn end() -> Self {
        Self::End
    }
    pub fn is_text(&self) -> bool {
        matches!(self, Self::Text(_))
    }
    pub fn is_end(&self) -> bool {
        matches!(self, Self::End)
    }
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(s) => Some(s),
            _ => None,
        }
    }
}

/// Streaming tool-call events. The backend emits `Started` when the
/// model opens a tool call, one or more `Arguments` chunks as the
/// arguments JSON streams in, and `Finished` when the envelope closes.
/// Consumers rely on this ordering: `Finished` is the dispatch trigger.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ToolEvent {
    Started { id: String, name: String },
    Arguments { id: String, fragment: String },
    Finished { id: String },
}
