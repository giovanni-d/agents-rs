//! [`OutputSchema`] for types usable as structured LLM output, and
//! [`OutputSchemaRequest`] which carries schema info through the agent pipeline.

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;

use crate::structured::OutputFormat;
use crate::{GrammarSource, SchemaKind};

/// A type usable as structured LLM output: declares a [`SchemaKind`] that the
/// LLM is constrained to.
pub trait OutputSchema: DeserializeOwned + Send + Sync + 'static {
    fn schema() -> SchemaKind;

    fn description() -> Option<&'static str> {
        None
    }

    fn example() -> Option<Value> {
        None
    }
}

/// Schema request attached to Context for LLM agents to interpret.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OutputSchemaRequest {
    pub grammar: GrammarSource,
    pub type_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub example: Option<Value>,
    /// Hint: enforce strict adherence (no extra fields, etc.)
    #[serde(default = "default_strict")]
    pub strict: bool,
}

fn default_strict() -> bool {
    true
}

impl OutputSchemaRequest {
    pub fn new(schema: SchemaKind, type_name: impl Into<String>) -> Self {
        Self {
            grammar: GrammarSource::Schema(schema),
            type_name: type_name.into(),
            description: None,
            example: None,
            strict: true,
        }
    }

    /// Create a request from a raw [`GrammarSource`] — use this when the grammar
    /// is not a JSON schema (e.g. a pre-built GBNF for a custom DSL).
    pub fn from_grammar(
        grammar: GrammarSource,
        type_name: impl Into<String>,
    ) -> Self {
        Self {
            grammar,
            type_name: type_name.into(),
            description: None,
            example: None,
            strict: true,
        }
    }

    /// Create a request from any [`OutputFormat`] type.
    pub fn from_type<T: OutputFormat>() -> Self {
        Self {
            grammar: T::grammar(),
            type_name: std::any::type_name::<T>().into(),
            description: T::description().map(str::to_string),
            example: T::example(),
            strict: true,
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    pub fn with_example(mut self, example: Value) -> Self {
        self.example = Some(example);
        self
    }

    pub fn strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, PartialEq)]
    struct TestOutput {
        message: String,
        count: i32,
    }

    impl OutputSchema for TestOutput {
        fn schema() -> SchemaKind {
            SchemaKind::object()
                .field("message", SchemaKind::string())
                .field("count", SchemaKind::integer())
                .build()
        }

        fn description() -> Option<&'static str> {
            Some("A test output with message and count")
        }
    }

    #[test]
    fn test_output_schema_trait() {
        let schema = TestOutput::schema();

        match &schema {
            SchemaKind::Object { fields } => {
                assert_eq!(fields.len(), 2);
                assert_eq!(fields[0].name, "message");
                assert_eq!(fields[1].name, "count");
            }
            _ => panic!("Expected Object schema"),
        }

        assert_eq!(
            <TestOutput as OutputSchema>::description(),
            Some("A test output with message and count")
        );
    }

    #[test]
    fn test_output_schema_request_new() {
        let request = OutputSchemaRequest::new(SchemaKind::string(), "MyType");

        assert_eq!(request.type_name, "MyType");
        assert!(request.strict);
        assert!(request.description.is_none());
        assert!(request.example.is_none());
    }

    #[test]
    fn test_output_schema_request_from_type() {
        let request = OutputSchemaRequest::from_type::<TestOutput>();

        assert!(request.type_name.contains("TestOutput"));
        assert_eq!(
            request.description,
            Some("A test output with message and count".into())
        );
        assert!(request.strict);
    }

    #[test]
    fn test_output_schema_request_builder() {
        let request = OutputSchemaRequest::new(SchemaKind::integer(), "Score")
            .with_description("A score from 1-100")
            .with_example(serde_json::json!(75))
            .strict(false);

        assert_eq!(request.description, Some("A score from 1-100".into()));
        assert_eq!(request.example, Some(serde_json::json!(75)));
        assert!(!request.strict);
    }

    #[test]
    fn test_output_schema_request_serialization() {
        let request = OutputSchemaRequest::from_type::<TestOutput>();

        let json = serde_json::to_string(&request).unwrap();
        let parsed: OutputSchemaRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.type_name, request.type_name);
        assert_eq!(parsed.description, request.description);
        assert_eq!(parsed.strict, request.strict);
    }

    #[derive(Debug, Deserialize)]
    struct ComplexOutput {
        items: Vec<String>,
        metadata: Option<serde_json::Value>,
    }

    impl OutputSchema for ComplexOutput {
        fn schema() -> SchemaKind {
            SchemaKind::object()
                .field("items", SchemaKind::array(SchemaKind::string()))
                .optional_field("metadata", SchemaKind::any())
                .build()
        }

        fn example() -> Option<Value> {
            Some(serde_json::json!({
                "items": ["a", "b", "c"],
                "metadata": {"key": "value"}
            }))
        }
    }

    #[test]
    fn test_complex_output_schema() {
        let schema = ComplexOutput::schema();

        if let SchemaKind::Object { fields } = &schema {
            assert_eq!(fields.len(), 2);
            assert!(fields[0].required);
            assert!(!fields[1].required);
        } else {
            panic!("Expected Object schema");
        }

        let example = <ComplexOutput as OutputSchema>::example().unwrap();
        assert!(example["items"].is_array());
    }

    #[test]
    fn test_output_schema_request_from_complex_type() {
        let request = OutputSchemaRequest::from_type::<ComplexOutput>();

        assert!(request.example.is_some());
        let example = request.example.unwrap();
        assert_eq!(example["items"].as_array().unwrap().len(), 3);
    }
}
