//! [`AgentData`] — types that carry their own [`SchemaKind`] for runtime validation.

use serde::{Serialize, de::DeserializeOwned};

use super::SchemaKind;

/// A type that knows its own [`SchemaKind`].
pub trait AgentData: Serialize + DeserializeOwned + Send + Sync + 'static {
    fn schema() -> SchemaKind;

    fn description() -> Option<&'static str> {
        None
    }
}

impl AgentData for String {
    fn schema() -> SchemaKind {
        SchemaKind::String
    }
    fn description() -> Option<&'static str> {
        Some("A text string")
    }
}

impl AgentData for i32 {
    fn schema() -> SchemaKind {
        SchemaKind::Integer
    }
}

impl AgentData for i64 {
    fn schema() -> SchemaKind {
        SchemaKind::Integer
    }
}

impl AgentData for f64 {
    fn schema() -> SchemaKind {
        SchemaKind::Number
    }
}

impl AgentData for bool {
    fn schema() -> SchemaKind {
        SchemaKind::Boolean
    }
}

impl AgentData for serde_json::Value {
    fn schema() -> SchemaKind {
        SchemaKind::Any
    }
    fn description() -> Option<&'static str> {
        Some("Any JSON value")
    }
}

impl<T: AgentData> AgentData for Vec<T> {
    fn schema() -> SchemaKind {
        SchemaKind::Array {
            items: Box::new(T::schema()),
        }
    }
}

impl<T: AgentData> AgentData for Option<T> {
    fn schema() -> SchemaKind {
        SchemaKind::Optional {
            inner: Box::new(T::schema()),
        }
    }
}

impl AgentData for () {
    fn schema() -> SchemaKind {
        SchemaKind::Object { fields: vec![] }
    }
    fn description() -> Option<&'static str> {
        Some("No data")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    struct TestData {
        name: String,
        count: i32,
    }

    impl AgentData for TestData {
        fn schema() -> SchemaKind {
            SchemaKind::object()
                .field("name", SchemaKind::string())
                .field("count", SchemaKind::integer())
                .build()
        }
        fn description() -> Option<&'static str> {
            Some("Test data structure")
        }
    }

    #[test]
    fn user_type_schema_has_declared_fields() {
        let SchemaKind::Object { fields } = TestData::schema() else {
            panic!("expected object");
        };
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].name, "name");
        assert_eq!(fields[1].name, "count");
    }

    #[test]
    fn primitive_schemas() {
        assert_eq!(String::schema(), SchemaKind::String);
        assert_eq!(i32::schema(), SchemaKind::Integer);
        assert_eq!(i64::schema(), SchemaKind::Integer);
        assert_eq!(f64::schema(), SchemaKind::Number);
        assert_eq!(bool::schema(), SchemaKind::Boolean);
    }

    #[test]
    fn vec_and_option_wrap_inner_schema() {
        let SchemaKind::Array { items } = Vec::<String>::schema() else {
            panic!("expected array");
        };
        assert_eq!(*items, SchemaKind::String);

        let SchemaKind::Optional { inner } = Option::<i32>::schema() else {
            panic!("expected optional");
        };
        assert_eq!(*inner, SchemaKind::Integer);
    }

    #[test]
    fn unit_schema_is_empty_object() {
        let SchemaKind::Object { fields } = <()>::schema() else {
            panic!("expected empty object");
        };
        assert!(fields.is_empty());
    }
}
