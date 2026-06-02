//! Tool response types returned to the conversation.

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Response from a tool execution.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolResponse {
    pub call_id: String,
    pub name: String,
    pub result: ToolResult,
}

impl ToolResponse {
    pub fn new(call_id: impl Into<String>, name: impl Into<String>, result: ToolResult) -> Self {
        Self {
            call_id: call_id.into(),
            name: name.into(),
            result,
        }
    }
}

/// Result of a tool execution.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResult {
    Json(Value),
    Text(String),
    Binary {
        #[serde(with = "bytes_serde")]
        data: Bytes,
        content_type: String,
    },
    Error(String),
}

impl ToolResult {
    pub fn json(value: impl Serialize) -> Self {
        Self::Json(serde_json::to_value(value).unwrap_or(Value::Null))
    }
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }
    pub fn error(message: impl Into<String>) -> Self {
        Self::Error(message.into())
    }
    pub fn binary(data: impl Into<Bytes>, content_type: impl Into<String>) -> Self {
        Self::Binary {
            data: data.into(),
            content_type: content_type.into(),
        }
    }
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }
    pub fn error_message(&self) -> Option<&str> {
        match self {
            Self::Error(msg) => Some(msg),
            _ => None,
        }
    }
}

mod bytes_serde {
    use bytes::Bytes;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(bytes: &Bytes, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use base64::Engine;
        let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
        encoded.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Bytes, D::Error>
    where
        D: Deserializer<'de>,
    {
        use base64::Engine;
        let encoded = String::deserialize(deserializer)?;
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(&encoded)
            .map_err(serde::de::Error::custom)?;
        Ok(Bytes::from(decoded))
    }
}
