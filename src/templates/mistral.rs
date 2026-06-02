//! Mistral Instruct template (v3+, Ministral-3, etc.).

use crate::{Message, Role};

use super::helpers::{collect_text_content, push_text_content, push_tool_result_content};
use super::template::ChatTemplate;

/// Mistral Instruct template.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MistralTemplate;

impl ChatTemplate for MistralTemplate {
    fn format(&self, messages: &[Message]) -> String {
        let mut prompt = String::from("<s>");
        let mut system_text: Option<String> = None;
        let mut first_user = true;

        for msg in messages {
            match &msg.role {
                Role::System => {
                    system_text = Some(collect_text_content(msg));
                }
                Role::User => {
                    prompt.push_str("[INST] ");
                    if first_user {
                        if let Some(sys) = system_text.take() {
                            prompt.push_str(&sys);
                            prompt.push_str("\n\n");
                        }
                        first_user = false;
                    }
                    push_text_content(&mut prompt, msg);
                    prompt.push_str(" [/INST]");
                }
                Role::Assistant => {
                    push_text_content(&mut prompt, msg);
                    prompt.push_str("</s>");
                }
                Role::Tool | Role::Custom(_) => {
                    prompt.push_str("[INST] ");
                    push_tool_result_content(&mut prompt, msg);
                    prompt.push_str(" [/INST]");
                }
            }
        }

        prompt
    }

    fn stop_tokens(&self) -> &[&str] {
        &["</s>"]
    }
}
