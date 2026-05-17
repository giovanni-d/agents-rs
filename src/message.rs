//! Conversation messages and the [`Context`] passed to every agent call.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    System { content: String },
    User { content: String },
    Assistant { content: String },
    Tool { name: String, content: String },
}

impl Message {
    pub fn system(s: impl Into<String>) -> Self {
        Self::System { content: s.into() }
    }
    pub fn user(s: impl Into<String>) -> Self {
        Self::User { content: s.into() }
    }
    pub fn assistant(s: impl Into<String>) -> Self {
        Self::Assistant { content: s.into() }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Context {
    pub messages: Vec<Message>,
}

impl Context {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_message(mut self, m: Message) -> Self {
        self.messages.push(m);
        self
    }

    pub fn with_user(self, s: impl Into<String>) -> Self {
        self.with_message(Message::user(s))
    }

    pub fn with_system(self, s: impl Into<String>) -> Self {
        self.with_message(Message::system(s))
    }

    pub fn push(&mut self, m: Message) {
        self.messages.push(m);
    }

    pub fn last_user_message(&self) -> Option<&str> {
        self.messages.iter().rev().find_map(|m| match m {
            Message::User { content } => Some(content.as_str()),
            _ => None,
        })
    }
}
