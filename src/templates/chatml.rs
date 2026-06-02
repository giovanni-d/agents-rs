//! ChatML template (Qwen, Nemotron).

use crate::{Message, Role};

use super::helpers::{collect_text_content, push_text_content, push_tool_result_content};
use super::template::ChatTemplate;

// ============================================================================
// ThinkingMode
// ============================================================================

/// Controls prefill after the final `<|im_start|>assistant\n` marker.
/// Reasoning knob for Qwen 3 / 3.5; non-reasoning ChatML models use
/// [`None`](ThinkingMode::None).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ThinkingMode {
    /// No prefill. Correct default for non-reasoning ChatML models
    /// (Qwen 2.5, Nemotron) — avoids ~12 tokens of inert prefill.
    #[default]
    None,
    /// Prefill an empty `<think>\n\n</think>\n\n` block to **suppress**
    /// chain-of-thought in Qwen 3 / 3.5, which was trained to always
    /// emit a `<think>` section.
    Suppressed,
    /// Prefill an open `<think>\n` tag to **enable** Qwen 3 / 3.5
    /// chain-of-thought reasoning.
    Enabled,
}

// ============================================================================
// ChatMLTemplate
// ============================================================================

/// ChatML template.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChatMLTemplate {
    pub thinking: ThinkingMode,
}

impl ChatTemplate for ChatMLTemplate {
    fn format(&self, messages: &[Message]) -> String {
        let mut prompt = String::new();

        for msg in messages {
            match &msg.role {
                Role::Tool => {
                    prompt.push_str("<|im_start|>user\n<tool_response>\n");
                    push_tool_result_content(&mut prompt, msg);
                    prompt.push_str("\n</tool_response><|im_end|>\n");
                }
                Role::Assistant => {
                    prompt.push_str("<|im_start|>assistant\n");
                    let text = collect_text_content(msg);
                    prompt.push_str(&strip_think_blocks(&text));
                    prompt.push_str("<|im_end|>\n");
                }
                _ => {
                    let role = match &msg.role {
                        Role::System => "system",
                        Role::User => "user",
                        Role::Custom(r) => r.as_str(),
                        _ => unreachable!(),
                    };
                    prompt.push_str("<|im_start|>");
                    prompt.push_str(role);
                    prompt.push('\n');
                    push_text_content(&mut prompt, msg);
                    prompt.push_str("<|im_end|>\n");
                }
            }
        }

        prompt.push_str("<|im_start|>assistant\n");
        match self.thinking {
            ThinkingMode::None => {}
            ThinkingMode::Suppressed => {
                prompt.push_str("<think>\n\n</think>\n\n");
            }
            ThinkingMode::Enabled => {
                prompt.push_str("<think>\n");
            }
        }
        prompt
    }

    fn stop_tokens(&self) -> &[&str] {
        &["<|im_end|>"]
    }

    fn format_system_prefix(&self, system: &str) -> Option<String> {
        let mut prompt = String::from("<|im_start|>system\n");
        prompt.push_str(system);
        prompt.push_str("<|im_end|>\n");
        Some(prompt)
    }
}

fn strip_think_blocks(text: &str) -> String {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";

    let mut result = String::new();
    let mut remaining = text;

    while let Some(start) = remaining.find(OPEN) {
        result.push_str(&remaining[..start]);
        match remaining[start + OPEN.len()..].find(CLOSE) {
            Some(end) => {
                remaining =
                    &remaining[start + OPEN.len() + end + CLOSE.len()..];
            }
            None => {
                return result;
            }
        }
    }

    result.push_str(remaining);
    result
}
