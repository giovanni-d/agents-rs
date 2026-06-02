//! [`OutputFormat`] — a type that knows how to constrain the LLM's generation
//! and parse its raw response. JSON-shaped types get a blanket impl via
//! [`OutputSchema`]; non-JSON formats implement [`OutputFormat`] directly.
//!
//! [`OutputSchema`]: crate::OutputSchema

use serde_json::Value;

use crate::GrammarSource;
use crate::structured::OutputSchema;

/// A type produced by constrained LLM output: declares a [`GrammarSource`] and
/// parses the raw response back into `Self`. Every [`OutputSchema`] auto-implements
/// this via a blanket impl; custom formats implement it directly.
pub trait OutputFormat: Sized + Send + Sync + 'static {
    type Error: std::error::Error + Send + Sync + 'static;

    fn grammar() -> GrammarSource;

    fn parse(raw: &str) -> Result<Self, Self::Error>;

    fn description() -> Option<&'static str> {
        None
    }

    fn example() -> Option<Value> {
        None
    }
}

impl<T> OutputFormat for T
where
    T: OutputSchema,
{
    type Error = serde_json::Error;

    fn grammar() -> GrammarSource {
        GrammarSource::Schema(<T as OutputSchema>::schema())
    }

    fn parse(raw: &str) -> Result<Self, Self::Error> {
        serde_json::from_str(raw)
    }

    fn description() -> Option<&'static str> {
        <T as OutputSchema>::description()
    }

    fn example() -> Option<Value> {
        <T as OutputSchema>::example()
    }
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;

    use super::*;
    use crate::SchemaKind;

    #[derive(Debug, Deserialize, PartialEq)]
    struct JsonType {
        message: String,
        count: i32,
    }

    impl OutputSchema for JsonType {
        fn schema() -> SchemaKind {
            SchemaKind::object()
                .field("message", SchemaKind::string())
                .field("count", SchemaKind::integer())
                .build()
        }

        fn description() -> Option<&'static str> {
            Some("a test JSON type")
        }
    }

    #[test]
    fn output_schema_types_auto_impl_output_format() {
        // Compilation is the test — if the blanket impl doesn't fire,
        // these calls don't type-check.
        let grammar = <JsonType as OutputFormat>::grammar();
        assert!(matches!(grammar, GrammarSource::Schema(_)));
        assert_eq!(
            <JsonType as OutputFormat>::description(),
            Some("a test JSON type"),
        );
    }

    #[test]
    fn blanket_parse_defers_to_serde_json() {
        let parsed =
            <JsonType as OutputFormat>::parse(r#"{"message":"hi","count":7}"#)
                .expect("valid JSON must parse");
        assert_eq!(
            parsed,
            JsonType {
                message: "hi".into(),
                count: 7,
            },
        );

        let error =
            <JsonType as OutputFormat>::parse(r#"{"message":"hi"}"#);
        assert!(error.is_err(), "missing field must fail");
    }

    struct CompactPlan(String);

    #[derive(Debug)]
    struct CompactParseError(String);

    impl std::fmt::Display for CompactParseError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    impl std::error::Error for CompactParseError {}

    impl OutputFormat for CompactPlan {
        type Error = CompactParseError;

        fn grammar() -> GrammarSource {
            GrammarSource::Gbnf(
                "root ::= \"window_pin(\" [@] \",\" (\"true\"|\"false\") \")\""
                    .into(),
            )
        }

        fn parse(raw: &str) -> Result<Self, Self::Error> {
            if raw.starts_with("window_pin(") {
                Ok(Self(raw.to_string()))
            } else {
                Err(CompactParseError(format!("not a compact plan: {raw}")))
            }
        }
    }

    #[test]
    fn non_json_output_format_uses_gbnf_variant() {
        let grammar = <CompactPlan as OutputFormat>::grammar();
        match grammar {
            GrammarSource::Gbnf(text) => assert!(text.contains("window_pin")),
            other => panic!("expected Gbnf variant, got {other:?}"),
        }
    }

    #[test]
    fn non_json_output_format_uses_its_own_parser() {
        let parsed = <CompactPlan as OutputFormat>::parse("window_pin(@, true)")
            .expect("valid compact must parse");
        assert_eq!(parsed.0, "window_pin(@, true)");

        let error = <CompactPlan as OutputFormat>::parse("foo(@, true)");
        assert!(error.is_err(), "non-compact input must fail");
    }
}
