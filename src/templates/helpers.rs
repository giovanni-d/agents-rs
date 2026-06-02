//! Internal helpers shared by per-family template impls.

use crate::{ContentPart, Message, ToolResult};

pub(crate) fn push_text_content(prompt: &mut String, msg: &Message) {
    if let Some(text) = msg.as_text() {
        prompt.push_str(text);
    } else {
        prompt.push_str(&msg.content.to_text());
    }
}

pub(crate) fn collect_text_content(msg: &Message) -> String {
    if let Some(text) = msg.as_text() {
        text.to_string()
    } else {
        msg.content.to_text()
    }
}

pub(crate) fn push_tool_result_content(prompt: &mut String, msg: &Message) {
    for part in msg.content.iter() {
        if let ContentPart::ToolResponse { response } = part {
            match &response.result {
                ToolResult::Json(value) => {
                    if let Ok(json) = serde_json::to_string(value) {
                        prompt.push_str(&json);
                    }
                }
                ToolResult::Text(text) => prompt.push_str(text),
                ToolResult::Error(err) => {
                    prompt.push_str("Error: ");
                    prompt.push_str(err);
                }
                ToolResult::Binary { content_type, .. } => {
                    prompt.push_str("[binary: ");
                    prompt.push_str(content_type);
                    prompt.push(']');
                }
            }
        }
    }
}
