//! Conversation messages and the [`Context`] passed to every agent call.

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::tool_response::ToolResponse;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: MessageContent,
}

impl Message {
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: MessageContent::text(text),
        }
    }

    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::text(text),
        }
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: MessageContent::text(text),
        }
    }

    pub fn tool(
        call_id: impl Into<String>,
        name: impl Into<String>,
        result: crate::tool_response::ToolResult,
    ) -> Self {
        Self {
            role: Role::Tool,
            content: MessageContent(vec![ContentPart::ToolResponse {
                response: ToolResponse {
                    call_id: call_id.into(),
                    name: name.into(),
                    result,
                },
            }]),
        }
    }

    pub fn as_text(&self) -> Option<&str> {
        self.content.as_text()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
    #[serde(untagged)]
    Custom(String),
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MessageContent(pub Vec<ContentPart>);

impl MessageContent {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    pub fn text(text: impl Into<String>) -> Self {
        Self(vec![ContentPart::Text { text: text.into() }])
    }

    pub fn parts(parts: Vec<ContentPart>) -> Self {
        Self(parts)
    }

    pub fn push(&mut self, part: ContentPart) {
        self.0.push(part);
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn as_text(&self) -> Option<&str> {
        if self.0.len() == 1
            && let ContentPart::Text { text } = &self.0[0]
        {
            return Some(text);
        }
        None
    }

    pub fn to_text(&self) -> String {
        self.0
            .iter()
            .filter_map(|p| {
                if let ContentPart::Text { text } = p {
                    Some(text.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("")
    }

    pub fn iter(&self) -> impl Iterator<Item = &ContentPart> {
        self.0.iter()
    }
}

impl From<String> for MessageContent {
    fn from(text: String) -> Self {
        Self::text(text)
    }
}

impl From<&str> for MessageContent {
    fn from(text: &str) -> Self {
        Self::text(text)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
    },
    Image {
        #[serde(with = "bytes_serde")]
        data: Bytes,
        media_type: String,
    },
    Audio {
        #[serde(with = "bytes_serde")]
        data: Bytes,
        format: String,
    },
    Document {
        #[serde(with = "bytes_serde")]
        data: Bytes,
        name: String,
    },
    ToolResponse {
        response: ToolResponse,
    },
    Json {
        value: Value,
    },
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

#[derive(Debug, Clone, Default)]
pub struct Context {
    pub messages: Vec<Message>,
    output_schema_request: Option<crate::structured::OutputSchemaRequest>,
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
        self.messages.iter().rev().find_map(|m| match m.role {
            Role::User => m.as_text(),
            _ => None,
        })
    }

    /// Attach a structured-output schema request to this context.
    pub fn with_output_schema_request(
        mut self,
        request: crate::structured::OutputSchemaRequest,
    ) -> Self {
        self.output_schema_request = Some(request);
        self
    }

    pub fn output_schema_request(&self) -> Option<&crate::structured::OutputSchemaRequest> {
        self.output_schema_request.as_ref()
    }
}
