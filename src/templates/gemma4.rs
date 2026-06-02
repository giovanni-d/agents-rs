//! Gemma 4 template.

use crate::{Message, Role};

use super::helpers::{collect_text_content, push_text_content, push_tool_result_content};
use super::template::ChatTemplate;

/// Gemma 4 template. `thinking: true` injects `<|think|>` in the first
/// system turn to enable reasoning.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Gemma4Template {
    pub thinking: bool,
}

impl ChatTemplate for Gemma4Template {
    fn format(&self, messages: &[Message]) -> String {
        let mut prompt = String::from("<bos>");
        let mut first_system = true;

        for msg in messages {
            match &msg.role {
                Role::System => {
                    prompt.push_str("<|turn>system\n");
                    if self.thinking && first_system {
                        prompt.push_str("<|think|>\n");
                    }
                    first_system = false;
                    push_text_content(&mut prompt, msg);
                    prompt.push_str("<turn|>\n");
                }
                Role::User => {
                    prompt.push_str("<|turn>user\n");
                    push_text_content(&mut prompt, msg);
                    prompt.push_str("<turn|>\n");
                }
                Role::Assistant => {
                    prompt.push_str("<|turn>model\n");
                    let text = collect_text_content(msg);
                    prompt.push_str(&strip_gemma_think_channels(&text));
                    prompt.push_str("<turn|>\n");
                }
                Role::Tool | Role::Custom(_) => {
                    // Without the `<tool_response>` envelope, small
                    // Gemma variants (E2B) read a bare numeric result
                    // as a fresh user request and re-call the tool.
                    prompt.push_str("<|turn>user\n<tool_response>\n");
                    push_tool_result_content(&mut prompt, msg);
                    prompt.push_str("\n</tool_response><turn|>\n");
                }
            }
        }

        prompt.push_str("<|turn>model\n");
        prompt
    }

    fn stop_tokens(&self) -> &[&str] {
        &["<turn|>"]
    }

    fn clean_response(&self, text: &str) -> String {
        strip_gemma_think_channels(text)
    }

    fn format_system_prefix(&self, system: &str) -> Option<String> {
        let mut prompt = String::from("<bos><|turn>system\n");
        if self.thinking {
            prompt.push_str("<|think|>\n");
        }
        prompt.push_str(system);
        prompt.push_str("<turn|>\n");
        Some(prompt)
    }
}

/// Strip Gemma 4 `<|channel>thought...<channel|>` reasoning blocks.
pub(crate) fn strip_gemma_think_channels(text: &str) -> String {
    const OPEN: &str = "<|channel>thought";
    const CLOSE: &str = "<channel|>";

    let mut result = String::new();
    let mut remaining = text;

    while let Some(start) = remaining.find(OPEN) {
        result.push_str(&remaining[..start]);
        match remaining[start..].find(CLOSE) {
            Some(end) => {
                remaining = &remaining[start + end + CLOSE.len()..];
            }
            None => {
                return result;
            }
        }
    }

    result.push_str(remaining);
    result
}
