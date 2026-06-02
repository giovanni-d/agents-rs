//! `ChatTemplateKind` enum dispatching to per-family template impls.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::Message;

use super::chatml::{ChatMLTemplate, ThinkingMode};
use super::gemma4::Gemma4Template;
use super::llama3::Llama3Template;
use super::mistral::MistralTemplate;
use super::template::ChatTemplate;

/// Strongly-typed chat template identifier.
///
/// Accepted string forms (for serde and for `.parse()`):
/// - `"llama3"`
/// - `"gemma4"` → `Gemma4Template { thinking: false }`
/// - `"gemma4-thinking"` → `Gemma4Template { thinking: true }`
/// - `"chatml"` → `ThinkingMode::None` (non-reasoning models)
/// - `"chatml-suppressed"` → `ThinkingMode::Suppressed`
/// - `"chatml-thinking"` → `ThinkingMode::Enabled`
/// - `"mistral"`
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(try_from = "String", into = "String")]
pub enum ChatTemplateKind {
    Llama3(Llama3Template),
    Gemma4(Gemma4Template),
    ChatML(ChatMLTemplate),
    Mistral(MistralTemplate),
}

impl Default for ChatTemplateKind {
    fn default() -> Self {
        Self::ChatML(ChatMLTemplate::default())
    }
}

impl FromStr for ChatTemplateKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "llama3" => Ok(Self::Llama3(Llama3Template)),
            "gemma4" => Ok(Self::Gemma4(Gemma4Template { thinking: false })),
            "gemma4-thinking" => {
                Ok(Self::Gemma4(Gemma4Template { thinking: true }))
            }
            "chatml" => Ok(Self::ChatML(ChatMLTemplate {
                thinking: ThinkingMode::None,
            })),
            "chatml-suppressed" => Ok(Self::ChatML(ChatMLTemplate {
                thinking: ThinkingMode::Suppressed,
            })),
            "chatml-thinking" => Ok(Self::ChatML(ChatMLTemplate {
                thinking: ThinkingMode::Enabled,
            })),
            "mistral" => Ok(Self::Mistral(MistralTemplate)),
            _ => Err(format!(
                "unknown chat template: '{s}' (expected: llama3, \
                 gemma4, gemma4-thinking, chatml, chatml-suppressed, \
                 chatml-thinking, mistral)"
            )),
        }
    }
}

impl fmt::Display for ChatTemplateKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Llama3(_) => write!(f, "llama3"),
            Self::Gemma4(t) if t.thinking => write!(f, "gemma4-thinking"),
            Self::Gemma4(_) => write!(f, "gemma4"),
            Self::ChatML(t) => match t.thinking {
                ThinkingMode::None => write!(f, "chatml"),
                ThinkingMode::Suppressed => {
                    write!(f, "chatml-suppressed")
                }
                ThinkingMode::Enabled => write!(f, "chatml-thinking"),
            },
            Self::Mistral(_) => write!(f, "mistral"),
        }
    }
}

impl TryFrom<String> for ChatTemplateKind {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        value.parse()
    }
}

impl From<ChatTemplateKind> for String {
    fn from(kind: ChatTemplateKind) -> Self {
        kind.to_string()
    }
}

// ============================================================================
// Dispatch impl
// ============================================================================

impl ChatTemplate for ChatTemplateKind {
    fn format(&self, messages: &[Message]) -> String {
        match self {
            Self::Llama3(t) => t.format(messages),
            Self::Gemma4(t) => t.format(messages),
            Self::ChatML(t) => t.format(messages),
            Self::Mistral(t) => t.format(messages),
        }
    }

    fn stop_tokens(&self) -> &[&str] {
        match self {
            Self::Llama3(t) => t.stop_tokens(),
            Self::Gemma4(t) => t.stop_tokens(),
            Self::ChatML(t) => t.stop_tokens(),
            Self::Mistral(t) => t.stop_tokens(),
        }
    }

    fn clean_response(&self, text: &str) -> String {
        match self {
            Self::Gemma4(t) => t.clean_response(text),
            _ => text.to_string(),
        }
    }

    fn format_system_prefix(&self, system: &str) -> Option<String> {
        match self {
            Self::Llama3(t) => t.format_system_prefix(system),
            Self::Gemma4(t) => t.format_system_prefix(system),
            Self::ChatML(t) => t.format_system_prefix(system),
            Self::Mistral(t) => t.format_system_prefix(system),
        }
    }
}
