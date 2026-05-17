//! Extract [`ToolCall`]s from free-form model output.
//!
//! Backends that drive a model via plain text (no native tool-calling API)
//! instruct the model to emit JSON envelopes and parse them back here. The
//! scanner finds every top-level balanced JSON value in the output; objects
//! shaped like `{"tool": "<name>", "args": {...}}` become calls, arrays of
//! such objects unfold into multiple calls, and anything else is ignored.

use serde_json::Value;

use crate::agent::ToolCall;

/// Scan `text` and return every tool-call envelope found, in order.
///
/// Accepted shapes (per the system-prompt contract this library uses):
/// - `{"tool": "<name>", "args": {...}}` — one call.
/// - `[{"tool": ..., "args": ...}, ...]` — many calls.
///
/// The model may interleave prose with JSON; non-envelope JSON is dropped.
/// Each returned call's `id` is its position in the result vector.
pub fn extract_tool_calls(text: &str) -> Vec<ToolCall> {
    let bytes = text.as_bytes();
    let mut calls: Vec<ToolCall> = Vec::new();
    let mut cursor = 0;
    while cursor < bytes.len() {
        let Some(rel) = bytes[cursor..].iter().position(|&b| b == b'{' || b == b'[') else {
            break;
        };
        let start = cursor + rel;
        let Some(end) = balanced_end(bytes, start) else {
            break;
        };
        if let Ok(value) = serde_json::from_str::<Value>(&text[start..end]) {
            match value {
                Value::Array(items) => {
                    for item in items {
                        if let Some(call) = call_from_value(&item, calls.len()) {
                            calls.push(call);
                        }
                    }
                }
                Value::Object(_) => {
                    if let Some(call) = call_from_value(&value, calls.len()) {
                        calls.push(call);
                    }
                }
                _ => {}
            }
        }
        cursor = end;
    }
    calls
}

fn balanced_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            if b == b'\\' {
                escaped = true;
                continue;
            }
            if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' | b'[' => depth += 1,
            b'}' | b']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
    }
    None
}

fn call_from_value(v: &Value, index: usize) -> Option<ToolCall> {
    let name = v.get("tool")?.as_str()?.to_string();
    let args = v
        .get("args")
        .cloned()
        .unwrap_or(Value::Object(Default::default()));
    Some(ToolCall {
        id: index.to_string(),
        name,
        arguments: args,
    })
}

#[cfg(test)]
mod tests {
    use super::extract_tool_calls;
    use serde_json::json;

    #[test]
    fn parses_single_envelope() {
        let calls = extract_tool_calls(r#"{"tool": "add", "args": {"a": 1, "b": 2}}"#);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "add");
        assert_eq!(calls[0].arguments, json!({"a": 1, "b": 2}));
    }

    #[test]
    fn parses_envelope_with_surrounding_text() {
        let calls = extract_tool_calls(r#"Sure! {"tool":"echo","args":{"v":3}} done."#);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "echo");
    }

    #[test]
    fn parses_array_of_calls() {
        let calls = extract_tool_calls(
            r#"[{"tool":"a","args":{"x":1}},{"tool":"b","args":{"y":2}}]"#,
        );
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[0].id, "0");
        assert_eq!(calls[1].name, "b");
        assert_eq!(calls[1].id, "1");
    }

    #[test]
    fn parses_two_concatenated_objects() {
        let calls = extract_tool_calls(
            r#"{"tool":"a","args":{}} and then {"tool":"b","args":{"k":"v"}}"#,
        );
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].name, "b");
    }

    #[test]
    fn ignores_non_envelope_json() {
        assert!(extract_tool_calls(r#"{"x": 1}"#).is_empty());
    }

    #[test]
    fn ignores_plain_text() {
        assert!(extract_tool_calls("hello there").is_empty());
    }
}
