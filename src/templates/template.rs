//! `ChatTemplate` trait: formats messages into a model-specific prompt.

use crate::Message;

// ============================================================================
// Trait
// ============================================================================

/// Format conversation messages into a model-specific prompt string.
pub trait ChatTemplate: Send + Sync {
    fn format(&self, messages: &[Message]) -> String;
    fn stop_tokens(&self) -> &[&str];

    /// Post-process raw model output. Default returns text unchanged;
    /// templates whose models emit reasoning markers (e.g. Gemma 4's
    /// `<|channel>thought...<channel|>`) override this to strip them.
    fn clean_response(&self, text: &str) -> String {
        text.to_string()
    }

    /// Stable byte prefix of `format()` output when the only message is
    /// `System(system)` and tools are absent. Used to prime the
    /// llama.cpp KV cache. Returns `None` when the template inlines the
    /// system message into a later turn (Mistral).
    fn format_system_prefix(&self, _system: &str) -> Option<String> {
        None
    }
}
