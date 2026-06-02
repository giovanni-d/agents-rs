//! Chat templates and prompt formatting for local LLM backends. Owns
//! the [`ChatTemplate`] trait, per-family impls, and [`format_prompt`]
//! / [`build_tools_summary`] for injecting the JSON tool-call contract.

mod helpers;

pub mod chatml;
pub mod gemma4;
pub mod grammar;
pub mod kind;
pub mod llama3;
pub mod mistral;
pub mod template;

pub use chatml::{ChatMLTemplate, ThinkingMode};
pub use gemma4::Gemma4Template;
pub use grammar::GbnfToolCallGrammar;
pub use kind::ChatTemplateKind;
pub use llama3::Llama3Template;
pub use mistral::MistralTemplate;
pub use template::ChatTemplate;

use std::sync::atomic::{AtomicU64, Ordering};

use crate::{Message, Role, ToolCall, ToolRegistry};

fn next_call_id() -> String {
    static N: AtomicU64 = AtomicU64::new(0);
    format!("call_{}", N.fetch_add(1, Ordering::Relaxed))
}

/// Build compact tool list (names + descriptions only).
pub fn build_tools_summary(registry: &ToolRegistry) -> Option<String> {
    let tools: Vec<String> = registry
        .tools()
        .map(|tool| {
            let def = tool.definition();
            if def.description.is_empty() {
                format!("- {}", def.name)
            } else {
                format!("- {}: {}", def.name, def.description)
            }
        })
        .collect();

    if tools.is_empty() {
        None
    } else {
        Some(format!(
            "Tools (call with JSON {{\"tool\": \"name\", \"args\": {{...}}}}):\n{}",
            tools.join("\n")
        ))
    }
}

/// Format a prompt, injecting a tools summary into the system message
/// when a registry is supplied.
pub fn format_prompt(
    messages: &[Message],
    tools: Option<&ToolRegistry>,
    template: &ChatTemplateKind,
) -> String {
    let tools_summary = tools.and_then(build_tools_summary);

    let messages: Vec<Message> = if let Some(ref summary) = tools_summary {
        messages
            .iter()
            .map(|msg| {
                if msg.role == Role::System {
                    let text = msg.as_text().unwrap_or("");
                    Message::system(&format!("{text}\n\n{summary}"))
                } else {
                    msg.clone()
                }
            })
            .collect()
    } else {
        messages.to_vec()
    };

    let messages = if tools_summary.is_some()
        && !messages.iter().any(|m| m.role == Role::System)
    {
        let mut with_system =
            vec![Message::system(tools_summary.as_deref().unwrap())];
        with_system.extend(messages);
        with_system
    } else {
        messages
    };

    template.format(&messages)
}

/// Extract `{"tool": "<name>", "args": {...}}` calls from raw model
/// output. Blocks that don't match the shape are silently skipped, so
/// the scanner works on both grammar-constrained and free output.
pub fn parse_tool_calls(model_output: &str) -> Vec<ToolCall> {
    #[derive(serde::Deserialize)]
    struct ToolCallShape {
        tool: String,
        args: serde_json::Value,
    }

    let mut calls = Vec::new();
    for candidate in extract_balanced_objects(model_output) {
        if let Ok(parsed) = serde_json::from_str::<ToolCallShape>(&candidate) {
            calls.push(ToolCall {
                id: next_call_id(),
                name: parsed.tool,
                arguments: parsed.args,
            });
        }
    }
    calls
}

/// Return every top-level balanced `{...}` block. Tracks string state
/// so braces inside strings don't affect depth.
fn extract_balanced_objects(input: &str) -> Vec<String> {
    let bytes = input.as_bytes();
    let mut objects = Vec::new();
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] != b'{' {
            i += 1;
            continue;
        }

        let start = i;
        let mut depth: i32 = 0;
        let mut in_string = false;
        let mut escape = false;
        let mut closed = false;

        while i < bytes.len() {
            let c = bytes[i];
            if escape {
                escape = false;
            } else if in_string {
                match c {
                    b'\\' => escape = true,
                    b'"' => in_string = false,
                    _ => {}
                }
            } else {
                match c {
                    b'"' => in_string = true,
                    b'{' => depth += 1,
                    b'}' => {
                        depth -= 1;
                        if depth == 0 {
                            i += 1;
                            closed = true;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            i += 1;
        }

        if closed {
            // Safe: slice boundaries are single-byte ASCII braces.
            objects.push(input[start..i].to_string());
        } else {
            break;
        }
    }

    objects
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::gemma4::strip_gemma_think_channels;
    use super::*;

    fn sample_messages() -> Vec<Message> {
        vec![
            Message::system("you decompose intents"),
            Message::user("pin this"),
        ]
    }

    #[test]
    fn chatml_none_has_no_prefill() {
        let template = ChatMLTemplate {
            thinking: ThinkingMode::None,
        };
        let prompt = template.format(&sample_messages());
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
        assert!(!prompt.contains("<think>"));
    }

    #[test]
    fn chatml_suppressed_prefills_empty_think_block() {
        let template = ChatMLTemplate {
            thinking: ThinkingMode::Suppressed,
        };
        let prompt = template.format(&sample_messages());
        assert!(prompt.ends_with("<think>\n\n</think>\n\n"));
    }

    #[test]
    fn chatml_enabled_prefills_open_think_tag() {
        let template = ChatMLTemplate {
            thinking: ThinkingMode::Enabled,
        };
        let prompt = template.format(&sample_messages());
        assert!(prompt.ends_with("<think>\n"));
    }

    /// `format_system_prefix` must lead the full `format()` output —
    /// the KV prefix-cache invariant.
    #[test]
    fn chatml_prefix_leads_full_format() {
        let template = ChatMLTemplate {
            thinking: ThinkingMode::None,
        };
        let prefix = template.format_system_prefix("you decompose intents").unwrap();
        let full = template.format(&sample_messages());
        assert!(full.starts_with(&prefix), "prefix={prefix:?}\nfull={full:?}");
    }

    #[test]
    fn llama3_prefix_leads_full_format() {
        let template = Llama3Template;
        let prefix = template.format_system_prefix("you decompose intents").unwrap();
        let full = template.format(&sample_messages());
        assert!(full.starts_with(&prefix), "prefix={prefix:?}\nfull={full:?}");
    }

    #[test]
    fn gemma4_prefix_leads_full_format() {
        let template = Gemma4Template { thinking: false };
        let prefix = template.format_system_prefix("you decompose intents").unwrap();
        let full = template.format(&sample_messages());
        assert!(full.starts_with(&prefix), "prefix={prefix:?}\nfull={full:?}");
    }

    #[test]
    fn mistral_has_no_stable_prefix() {
        // Mistral inlines system into the first user [INST] block.
        let template = MistralTemplate;
        assert!(template.format_system_prefix("anything").is_none());
    }

    #[test]
    fn gemma_strips_think_channel_blocks() {
        let raw = "before<|channel>thought reasoning here<channel|>after";
        assert_eq!(strip_gemma_think_channels(raw), "beforeafter");
    }

    #[test]
    fn gemma_clean_response_strips_channels() {
        let template = Gemma4Template { thinking: true };
        let cleaned = template
            .clean_response("hello<|channel>thought xyz<channel|> world");
        assert_eq!(cleaned, "hello world");
    }

    #[test]
    fn kind_roundtrip_through_string() {
        for spec in [
            "llama3",
            "gemma4",
            "gemma4-thinking",
            "chatml",
            "chatml-suppressed",
            "chatml-thinking",
            "mistral",
        ] {
            let parsed: ChatTemplateKind = spec.parse().unwrap();
            assert_eq!(parsed.to_string(), spec);
        }
    }

    #[test]
    fn kind_unknown_spec_errors() {
        assert!("bogus".parse::<ChatTemplateKind>().is_err());
    }

    #[test]
    fn kind_default_is_chatml_none() {
        assert_eq!(ChatTemplateKind::default().to_string(), "chatml");
    }

    #[test]
    fn parse_tool_calls_extracts_single_call() {
        let output = r#"{"tool": "add", "args": {"a": 2, "b": 3}}"#;
        let calls = parse_tool_calls(output);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "add");
        assert_eq!(calls[0].arguments["a"], 2);
        assert_eq!(calls[0].arguments["b"], 3);
    }

    #[test]
    fn parse_tool_calls_finds_call_after_text() {
        let output = "Sure, let me add those.\n{\"tool\": \"add\", \"args\": {\"a\": 1, \"b\": 2}}";
        let calls = parse_tool_calls(output);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "add");
    }

    #[test]
    fn parse_tool_calls_returns_empty_on_plain_text() {
        let calls = parse_tool_calls("just a chat response, nothing structured");
        assert!(calls.is_empty());
    }

    #[test]
    fn parse_tool_calls_skips_non_tool_objects() {
        let output = r#"Here is some JSON: {"foo": 1} but no tool call."#;
        let calls = parse_tool_calls(output);
        assert!(calls.is_empty());
    }

    #[test]
    fn parse_tool_calls_handles_nested_args() {
        let output = r#"{"tool": "search", "args": {"filters": {"type": "code"}}}"#;
        let calls = parse_tool_calls(output);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["filters"]["type"], "code");
    }

    #[test]
    fn parse_tool_calls_extracts_multiple_calls() {
        let output =
            r#"{"tool": "a", "args": {}} then {"tool": "b", "args": {"x": 1}}"#;
        let calls = parse_tool_calls(output);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].name, "b");
    }

    #[test]
    fn parse_tool_calls_ignores_braces_in_strings() {
        let output = r#"{"tool": "echo", "args": {"text": "hello {world}"}}"#;
        let calls = parse_tool_calls(output);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["text"], "hello {world}");
    }

    #[test]
    fn parse_tool_calls_assigns_unique_ids() {
        let output =
            r#"{"tool": "a", "args": {}} {"tool": "b", "args": {}}"#;
        let calls = parse_tool_calls(output);
        assert_eq!(calls.len(), 2);
        assert_ne!(calls[0].id, calls[1].id);
    }
}
