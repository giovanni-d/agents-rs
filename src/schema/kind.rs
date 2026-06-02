//! [`SchemaKind`] — universal schema convertible to JSON Schema, GBNF, or regex,
//! and usable to validate a `serde_json::Value`.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::SchemaValidationError;

/// The shape of a schema value — recursive to support nested types.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SchemaKind {
    String,
    Number,
    Integer,
    Boolean,
    /// Any JSON value; no validation.
    Any,
    Object {
        fields: Vec<SchemaField>,
    },
    Array {
        items: Box<SchemaKind>,
    },
    Optional {
        inner: Box<SchemaKind>,
    },
    /// Tagged union. Unit variants serialize to a plain string; data
    /// variants serialize to `{"type": "<name>", ...fields}` (or `"data"`
    /// when the variant's data isn't an object).
    Enum {
        variants: Vec<SchemaVariant>,
    },
    Literal {
        value: Value,
    },
    /// Object with string keys and uniformly typed values.
    Map {
        values: Box<SchemaKind>,
    },
}

/// A field within an [`SchemaKind::Object`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SchemaField {
    pub name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    pub schema: SchemaKind,
    #[serde(default = "default_true")]
    pub required: bool,
}

/// A variant within an [`SchemaKind::Enum`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SchemaVariant {
    pub name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// `None` for unit variants, `Some` for variants with associated data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<SchemaKind>,
}

fn default_true() -> bool {
    true
}

impl SchemaKind {
    pub fn string() -> Self {
        Self::String
    }
    pub fn number() -> Self {
        Self::Number
    }
    pub fn integer() -> Self {
        Self::Integer
    }
    pub fn boolean() -> Self {
        Self::Boolean
    }
    pub fn any() -> Self {
        Self::Any
    }

    pub fn array(items: impl Into<SchemaKind>) -> Self {
        Self::Array {
            items: Box::new(items.into()),
        }
    }

    pub fn optional(inner: impl Into<SchemaKind>) -> Self {
        Self::Optional {
            inner: Box::new(inner.into()),
        }
    }

    pub fn map(values: impl Into<SchemaKind>) -> Self {
        Self::Map {
            values: Box::new(values.into()),
        }
    }

    pub fn literal(value: impl Into<Value>) -> Self {
        Self::Literal {
            value: value.into(),
        }
    }

    pub fn object() -> ObjectBuilder {
        ObjectBuilder::new()
    }

    pub fn enumeration() -> EnumBuilder {
        EnumBuilder::new()
    }
}

impl SchemaKind {
    /// JSON Schema representation.
    pub fn to_json_schema(&self) -> Value {
        match self {
            Self::String => serde_json::json!({"type": "string"}),
            Self::Number => serde_json::json!({"type": "number"}),
            Self::Integer => serde_json::json!({"type": "integer"}),
            Self::Boolean => serde_json::json!({"type": "boolean"}),
            Self::Any => serde_json::json!({}),
            Self::Literal { value } => serde_json::json!({"const": value}),
            Self::Array { items } => serde_json::json!({
                "type": "array",
                "items": items.to_json_schema(),
            }),
            Self::Optional { inner } => {
                let mut schema = inner.to_json_schema();
                if let Some(obj) = schema.as_object_mut()
                    && let Some(type_val) = obj.get("type").cloned()
                {
                    obj.insert("type".into(), serde_json::json!([type_val, "null"]));
                }
                schema
            }
            Self::Map { values } => serde_json::json!({
                "type": "object",
                "additionalProperties": values.to_json_schema(),
            }),
            Self::Object { fields } => object_json_schema(fields),
            Self::Enum { variants } => enum_json_schema(variants),
        }
    }
}

fn object_json_schema(fields: &[SchemaField]) -> Value {
    let properties: serde_json::Map<String, Value> = fields
        .iter()
        .map(|f| (f.name.clone(), field_json_schema(f)))
        .collect();
    let required: Vec<&str> = fields
        .iter()
        .filter(|f| f.required)
        .map(|f| f.name.as_str())
        .collect();
    serde_json::json!({
        "type": "object",
        "properties": properties,
        "required": required,
    })
}

fn field_json_schema(field: &SchemaField) -> Value {
    let mut schema = field.schema.to_json_schema();
    if !field.description.is_empty()
        && let Some(obj) = schema.as_object_mut()
    {
        obj.insert("description".into(), field.description.clone().into());
    }
    schema
}

fn enum_json_schema(variants: &[SchemaVariant]) -> Value {
    if variants.iter().all(|v| v.data.is_none()) {
        let values: Vec<&str> = variants.iter().map(|v| v.name.as_str()).collect();
        return serde_json::json!({"type": "string", "enum": values});
    }

    let one_of: Vec<Value> = variants
        .iter()
        .map(|v| match &v.data {
            None => serde_json::json!({
                "type": "object",
                "properties": {"type": {"const": &v.name}},
                "required": ["type"],
            }),
            Some(SchemaKind::Object { fields }) => {
                let mut props = serde_json::Map::new();
                props.insert("type".into(), serde_json::json!({"const": &v.name}));
                for field in fields {
                    props.insert(field.name.clone(), field_json_schema(field));
                }
                let mut required = vec!["type"];
                required.extend(fields.iter().filter(|f| f.required).map(|f| f.name.as_str()));
                serde_json::json!({
                    "type": "object",
                    "properties": props,
                    "required": required,
                })
            }
            Some(data) => serde_json::json!({
                "type": "object",
                "properties": {
                    "type": {"const": &v.name},
                    "data": data.to_json_schema(),
                },
                "required": ["type", "data"],
            }),
        })
        .collect();

    serde_json::json!({"oneOf": one_of})
}

impl SchemaKind {
    /// GBNF grammar fragment for llama.cpp constrained decoding.
    ///
    /// Primitives reference the standard rule names (`string`, `number`,
    /// `integer`, `value`, `ws`) — wrap the result in a full grammar that
    /// defines them before feeding it to llama.cpp.
    pub fn to_gbnf(&self) -> String {
        match self {
            Self::String => "string".to_string(),
            Self::Number => "number".to_string(),
            Self::Integer => "integer".to_string(),
            Self::Boolean => "(\"true\" | \"false\")".to_string(),
            Self::Any => "value".to_string(),
            Self::Literal { value } => {
                format!("\"{}\"", value.to_string().replace('\"', "\\\""))
            }
            Self::Array { items } => {
                let item = items.to_gbnf();
                format!("\"[\" ws ({item} (\",\" ws {item})*)? ws \"]\"")
            }
            Self::Optional { inner } => format!("({} | \"null\")", inner.to_gbnf()),
            Self::Map { values } => {
                let v = values.to_gbnf();
                format!(
                    "\"{{\" ws (string \":\" ws {v} (\",\" ws string \":\" ws {v})*)? ws \"}}\""
                )
            }
            Self::Object { fields } => object_gbnf(fields),
            Self::Enum { variants } => enum_gbnf(variants),
        }
    }

    /// If this is a non-empty Object, returns the GBNF field rules without
    /// surrounding braces — used to inline data fields next to the discriminator.
    fn to_gbnf_object_fields(&self) -> Option<String> {
        match self {
            Self::Object { fields } if !fields.is_empty() => Some(gbnf_field_rules(fields)),
            _ => None,
        }
    }
}

fn gbnf_field_rules(fields: &[SchemaField]) -> String {
    fields
        .iter()
        .map(|f| format!("\"\\\"{}\\\":\" ws {}", f.name, f.schema.to_gbnf()))
        .collect::<Vec<_>>()
        .join(" \",\" ws ")
}

fn object_gbnf(fields: &[SchemaField]) -> String {
    if fields.is_empty() {
        return "\"{}\"".to_string();
    }
    format!("\"{{\" ws {} ws \"}}\"", gbnf_field_rules(fields))
}

fn enum_gbnf(variants: &[SchemaVariant]) -> String {
    if variants.iter().all(|v| v.data.is_none()) {
        let options: Vec<String> = variants
            .iter()
            .map(|v| format!("\"\\\"{}\\\"\"", v.name))
            .collect();
        return format!("({})", options.join(" | "));
    }

    let options: Vec<String> = variants
        .iter()
        .map(|v| match &v.data {
            None => format!("\"{{\\\"type\\\":\\\"{}\\\"}}\"", v.name),
            Some(data) => match data.to_gbnf_object_fields() {
                Some(fields) => format!(
                    "\"{{\\\"type\\\":\\\"{}\\\"\" \",\" ws {} ws \"}}\"",
                    v.name, fields
                ),
                None if matches!(data, SchemaKind::Object { .. }) => {
                    format!("\"{{\\\"type\\\":\\\"{}\\\"}}\"", v.name)
                }
                None => format!(
                    "\"{{\\\"type\\\":\\\"{}\\\"\" \",\" ws \"\\\"data\\\":\" ws {} ws \"}}\"",
                    v.name,
                    data.to_gbnf()
                ),
            },
        })
        .collect();
    format!("({})", options.join(" | "))
}

impl SchemaKind {
    /// Regex pattern matching this schema's JSON encoding (vLLM, Outlines).
    pub fn to_regex(&self) -> String {
        match self {
            Self::String => r#""[^"]*""#.to_string(),
            Self::Number => r#"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?"#.to_string(),
            Self::Integer => r#"-?(?:0|[1-9]\d*)"#.to_string(),
            Self::Boolean => r#"(?:true|false)"#.to_string(),
            Self::Any => r#"(?:null|true|false|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?|"[^"]*"|\[[^\]]*\]|\{[^}]*\})"#.to_string(),
            Self::Literal { value } => regex::escape(&value.to_string()),
            Self::Array { items } => {
                let item = items.to_regex();
                format!(r#"\[\s*(?:{item}(?:\s*,\s*{item})*)?\s*\]"#)
            }
            Self::Optional { inner } => format!(r#"(?:null|{})"#, inner.to_regex()),
            Self::Map { values } => {
                let v = values.to_regex();
                format!(
                    r#"\{{\s*(?:"[^"]*"\s*:\s*{v}(?:\s*,\s*"[^"]*"\s*:\s*{v})*)?\s*\}}"#
                )
            }
            Self::Object { fields } => object_regex(fields),
            Self::Enum { variants } => enum_regex(variants),
        }
    }
}

fn field_regex_pair(field: &SchemaField) -> String {
    format!(
        r#""{name}"\s*:\s*{value}"#,
        name = regex::escape(&field.name),
        value = field.schema.to_regex()
    )
}

fn object_regex(fields: &[SchemaField]) -> String {
    if fields.is_empty() {
        return r#"\{\s*\}"#.to_string();
    }
    let pairs: Vec<String> = fields.iter().map(field_regex_pair).collect();
    format!(r#"\{{\s*{}\s*\}}"#, pairs.join(r#"\s*,\s*"#))
}

fn enum_regex(variants: &[SchemaVariant]) -> String {
    if variants.iter().all(|v| v.data.is_none()) {
        let options: Vec<String> = variants
            .iter()
            .map(|v| format!(r#""{}""#, regex::escape(&v.name)))
            .collect();
        return format!("(?:{})", options.join("|"));
    }

    let options: Vec<String> = variants
        .iter()
        .map(|v| {
            let tag = regex::escape(&v.name);
            match &v.data {
                None => format!(r#"\{{\s*"type"\s*:\s*"{tag}"\s*\}}"#),
                Some(SchemaKind::Object { fields }) if fields.is_empty() => {
                    format!(r#"\{{\s*"type"\s*:\s*"{tag}"\s*\}}"#)
                }
                Some(SchemaKind::Object { fields }) => {
                    let pairs: Vec<String> = fields.iter().map(field_regex_pair).collect();
                    format!(
                        r#"\{{\s*"type"\s*:\s*"{tag}"\s*,\s*{}\s*\}}"#,
                        pairs.join(r#"\s*,\s*"#)
                    )
                }
                Some(data) => format!(
                    r#"\{{\s*"type"\s*:\s*"{tag}"\s*,\s*"data"\s*:\s*{}\s*\}}"#,
                    data.to_regex()
                ),
            }
        })
        .collect();
    format!("(?:{})", options.join("|"))
}

impl SchemaKind {
    /// Validate a JSON value against this schema. Returns the first mismatch
    /// found, with context (field path, array index, variant data).
    pub fn validate(&self, value: &Value) -> Result<(), SchemaValidationError> {
        match self {
            Self::String => check_type(value.is_string(), "string", value),
            Self::Number => check_type(value.is_number(), "number", value),
            Self::Integer => validate_integer(value),
            Self::Boolean => check_type(value.is_boolean(), "boolean", value),
            Self::Any => Ok(()),
            Self::Literal { value: expected } => {
                if value == expected {
                    Ok(())
                } else {
                    Err(SchemaValidationError::InvalidLiteral {
                        expected: expected.to_string(),
                        actual: value.to_string(),
                    })
                }
            }
            Self::Array { items } => validate_array(items, value),
            Self::Optional { inner } => {
                if value.is_null() {
                    Ok(())
                } else {
                    inner.validate(value)
                }
            }
            Self::Map { values } => validate_map(values, value),
            Self::Object { fields } => validate_object(fields, value),
            Self::Enum { variants } => validate_enum(variants, value),
        }
    }
}

fn check_type(matches: bool, expected: &str, value: &Value) -> Result<(), SchemaValidationError> {
    if matches {
        Ok(())
    } else {
        Err(SchemaValidationError::TypeMismatch {
            expected: expected.into(),
            actual: json_type_name(value).into(),
        })
    }
}

fn validate_integer(value: &Value) -> Result<(), SchemaValidationError> {
    if value.is_i64() || value.is_u64() {
        return Ok(());
    }
    if let Some(n) = value.as_f64() {
        if n.fract() == 0.0 {
            return Ok(());
        }
        return Err(SchemaValidationError::TypeMismatch {
            expected: "integer".into(),
            actual: "float".into(),
        });
    }
    Err(SchemaValidationError::TypeMismatch {
        expected: "integer".into(),
        actual: json_type_name(value).into(),
    })
}

fn validate_array(items: &SchemaKind, value: &Value) -> Result<(), SchemaValidationError> {
    let arr = value
        .as_array()
        .ok_or_else(|| SchemaValidationError::TypeMismatch {
            expected: "array".into(),
            actual: json_type_name(value).into(),
        })?;
    for (index, item) in arr.iter().enumerate() {
        items
            .validate(item)
            .map_err(|error| SchemaValidationError::InvalidArrayElement {
                index,
                error: Box::new(error),
            })?;
    }
    Ok(())
}

fn validate_map(values: &SchemaKind, value: &Value) -> Result<(), SchemaValidationError> {
    let obj = value
        .as_object()
        .ok_or_else(|| SchemaValidationError::TypeMismatch {
            expected: "object".into(),
            actual: json_type_name(value).into(),
        })?;
    for (key, val) in obj {
        values
            .validate(val)
            .map_err(|error| SchemaValidationError::InvalidMapValue {
                key: key.clone(),
                error: Box::new(error),
            })?;
    }
    Ok(())
}

fn validate_object(fields: &[SchemaField], value: &Value) -> Result<(), SchemaValidationError> {
    let obj = value
        .as_object()
        .ok_or_else(|| SchemaValidationError::TypeMismatch {
            expected: "object".into(),
            actual: json_type_name(value).into(),
        })?;
    for field in fields {
        match obj.get(&field.name) {
            Some(v) => {
                field
                    .schema
                    .validate(v)
                    .map_err(|error| SchemaValidationError::InvalidField {
                        field: field.name.clone(),
                        error: Box::new(error),
                    })?;
            }
            None if field.required => {
                return Err(SchemaValidationError::MissingField {
                    field: field.name.clone(),
                });
            }
            None => {}
        }
    }
    Ok(())
}

fn validate_enum(variants: &[SchemaVariant], value: &Value) -> Result<(), SchemaValidationError> {
    if variants.iter().all(|v| v.data.is_none()) {
        let s = value
            .as_str()
            .ok_or_else(|| SchemaValidationError::TypeMismatch {
                expected: "string (enum)".into(),
                actual: json_type_name(value).into(),
            })?;
        return if variants.iter().any(|v| v.name == s) {
            Ok(())
        } else {
            Err(SchemaValidationError::InvalidVariant {
                variant: s.into(),
            })
        };
    }

    let obj = value
        .as_object()
        .ok_or_else(|| SchemaValidationError::TypeMismatch {
            expected: "object (tagged union)".into(),
            actual: json_type_name(value).into(),
        })?;
    let type_val = obj
        .get("type")
        .ok_or_else(|| SchemaValidationError::MissingField {
            field: "type".into(),
        })?;
    let variant_name = type_val
        .as_str()
        .ok_or_else(|| SchemaValidationError::TypeMismatch {
            expected: "string".into(),
            actual: json_type_name(type_val).into(),
        })?;
    let variant = variants
        .iter()
        .find(|v| v.name == variant_name)
        .ok_or_else(|| SchemaValidationError::InvalidVariant {
            variant: variant_name.into(),
        })?;

    match &variant.data {
        None => Ok(()),
        Some(SchemaKind::Object { fields }) => {
            for field in fields {
                match obj.get(&field.name) {
                    Some(v) => {
                        field.schema.validate(v).map_err(|error| {
                            SchemaValidationError::InvalidVariantData {
                                variant: variant_name.into(),
                                error: Box::new(error),
                            }
                        })?;
                    }
                    None if field.required => {
                        return Err(SchemaValidationError::InvalidVariantData {
                            variant: variant_name.into(),
                            error: Box::new(SchemaValidationError::MissingField {
                                field: field.name.clone(),
                            }),
                        });
                    }
                    None => {}
                }
            }
            Ok(())
        }
        Some(data_schema) => {
            let data_value = obj
                .get("data")
                .ok_or_else(|| SchemaValidationError::InvalidVariantData {
                    variant: variant_name.into(),
                    error: Box::new(SchemaValidationError::MissingField {
                        field: "data".into(),
                    }),
                })?;
            data_schema
                .validate(data_value)
                .map_err(|error| SchemaValidationError::InvalidVariantData {
                    variant: variant_name.into(),
                    error: Box::new(error),
                })
        }
    }
}

fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(n) => {
            if n.is_i64() || n.is_u64() {
                "integer"
            } else {
                "number"
            }
        }
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Fluent builder for [`SchemaKind::Object`]. Fields default to required;
/// use `optional_field*` for nullable ones.
#[derive(Default)]
pub struct ObjectBuilder {
    fields: Vec<SchemaField>,
}

impl ObjectBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn field(mut self, name: impl Into<String>, schema: impl Into<SchemaKind>) -> Self {
        self.fields.push(SchemaField {
            name: name.into(),
            description: String::new(),
            schema: schema.into(),
            required: true,
        });
        self
    }

    pub fn field_with_desc(
        mut self,
        name: impl Into<String>,
        schema: impl Into<SchemaKind>,
        description: impl Into<String>,
    ) -> Self {
        self.fields.push(SchemaField {
            name: name.into(),
            description: description.into(),
            schema: schema.into(),
            required: true,
        });
        self
    }

    pub fn optional_field(
        mut self,
        name: impl Into<String>,
        schema: impl Into<SchemaKind>,
    ) -> Self {
        self.fields.push(SchemaField {
            name: name.into(),
            description: String::new(),
            schema: schema.into(),
            required: false,
        });
        self
    }

    pub fn optional_field_with_desc(
        mut self,
        name: impl Into<String>,
        schema: impl Into<SchemaKind>,
        description: impl Into<String>,
    ) -> Self {
        self.fields.push(SchemaField {
            name: name.into(),
            description: description.into(),
            schema: schema.into(),
            required: false,
        });
        self
    }

    pub fn build(self) -> SchemaKind {
        SchemaKind::Object {
            fields: self.fields,
        }
    }
}

impl From<ObjectBuilder> for SchemaKind {
    fn from(b: ObjectBuilder) -> Self {
        b.build()
    }
}

/// Fluent builder for [`SchemaKind::Enum`]. Mix unit and data variants freely;
/// a tagged-union JSON Schema is emitted only when at least one variant carries data.
#[derive(Default)]
pub struct EnumBuilder {
    variants: Vec<SchemaVariant>,
}

impl EnumBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn unit(mut self, name: impl Into<String>) -> Self {
        self.variants.push(SchemaVariant {
            name: name.into(),
            description: String::new(),
            data: None,
        });
        self
    }

    pub fn unit_with_desc(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        self.variants.push(SchemaVariant {
            name: name.into(),
            description: description.into(),
            data: None,
        });
        self
    }

    pub fn variant(mut self, name: impl Into<String>, data: impl Into<SchemaKind>) -> Self {
        self.variants.push(SchemaVariant {
            name: name.into(),
            description: String::new(),
            data: Some(data.into()),
        });
        self
    }

    pub fn variant_with_desc(
        mut self,
        name: impl Into<String>,
        data: impl Into<SchemaKind>,
        description: impl Into<String>,
    ) -> Self {
        self.variants.push(SchemaVariant {
            name: name.into(),
            description: description.into(),
            data: Some(data.into()),
        });
        self
    }

    pub fn build(self) -> SchemaKind {
        SchemaKind::Enum {
            variants: self.variants,
        }
    }
}

impl From<EnumBuilder> for SchemaKind {
    fn from(b: EnumBuilder) -> Self {
        b.build()
    }
}

/// Source of the grammar that constrains an LLM's structured output.
///
/// `Schema` is convertible to every backend's format. `Gbnf` carries a pre-built
/// grammar for non-JSON formats and is opaque to JSON-Schema / regex backends.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "source", content = "value", rename_all = "snake_case")]
pub enum GrammarSource {
    /// A `SchemaKind` — convertible to every backend's format.
    Schema(SchemaKind),
    /// A pre-built GBNF grammar, opaque to non-llama.cpp backends.
    Gbnf(String),
}

impl GrammarSource {
    /// GBNF grammar string. Always available.
    pub fn to_gbnf(&self) -> String {
        match self {
            Self::Schema(s) => s.to_gbnf(),
            Self::Gbnf(g) => g.clone(),
        }
    }

    /// JSON Schema, or `None` for non-JSON sources.
    pub fn to_json_schema(&self) -> Option<Value> {
        match self {
            Self::Schema(s) => Some(s.to_json_schema()),
            Self::Gbnf(_) => None,
        }
    }

    /// Regex pattern, or `None` for non-JSON sources.
    pub fn to_regex(&self) -> Option<String> {
        match self {
            Self::Schema(s) => Some(s.to_regex()),
            Self::Gbnf(_) => None,
        }
    }
}

impl From<SchemaKind> for GrammarSource {
    fn from(s: SchemaKind) -> Self {
        Self::Schema(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primitives_match_constructors() {
        assert_eq!(SchemaKind::string(), SchemaKind::String);
        assert_eq!(SchemaKind::number(), SchemaKind::Number);
        assert_eq!(SchemaKind::integer(), SchemaKind::Integer);
        assert_eq!(SchemaKind::boolean(), SchemaKind::Boolean);
        assert_eq!(SchemaKind::any(), SchemaKind::Any);
    }

    #[test]
    fn object_builder_records_required_and_descriptions() {
        let schema = SchemaKind::object()
            .field("name", SchemaKind::string())
            .field_with_desc("age", SchemaKind::integer(), "User's age")
            .optional_field("email", SchemaKind::string())
            .build();
        let SchemaKind::Object { fields } = schema else {
            panic!("expected object");
        };
        assert_eq!(fields.len(), 3);
        assert!(fields[0].required);
        assert_eq!(fields[1].description, "User's age");
        assert!(!fields[2].required);
    }

    #[test]
    fn enum_builder_supports_units_and_data_variants() {
        let schema = SchemaKind::enumeration()
            .unit("low")
            .unit_with_desc("medium", "Medium priority")
            .variant(
                "custom",
                SchemaKind::object().field("value", SchemaKind::integer()),
            )
            .build();
        let SchemaKind::Enum { variants } = schema else {
            panic!("expected enum");
        };
        assert_eq!(variants.len(), 3);
        assert!(variants[0].data.is_none());
        assert_eq!(variants[1].description, "Medium priority");
        assert!(variants[2].data.is_some());
    }

    #[test]
    fn json_schema_object() {
        let schema = SchemaKind::object()
            .field("name", SchemaKind::string())
            .field("age", SchemaKind::integer())
            .build();
        let json = schema.to_json_schema();
        assert_eq!(json["type"], "object");
        assert_eq!(json["properties"]["name"]["type"], "string");
        assert_eq!(json["properties"]["age"]["type"], "integer");
    }

    #[test]
    fn json_schema_array() {
        let json = SchemaKind::array(SchemaKind::string()).to_json_schema();
        assert_eq!(json["type"], "array");
        assert_eq!(json["items"]["type"], "string");
    }

    #[test]
    fn json_schema_simple_enum() {
        let schema = SchemaKind::enumeration()
            .unit("a")
            .unit("b")
            .unit("c")
            .build();
        let json = schema.to_json_schema();
        assert_eq!(json["type"], "string");
        assert!(
            json["enum"]
                .as_array()
                .unwrap()
                .contains(&serde_json::json!("a"))
        );
    }

    #[test]
    fn validate_primitives() {
        assert!(SchemaKind::string().validate(&serde_json::json!("hi")).is_ok());
        assert!(SchemaKind::string().validate(&serde_json::json!(1)).is_err());

        assert!(SchemaKind::number().validate(&serde_json::json!(3.14)).is_ok());
        assert!(SchemaKind::number().validate(&serde_json::json!("3")).is_err());

        assert!(SchemaKind::integer().validate(&serde_json::json!(42)).is_ok());
        assert!(SchemaKind::integer().validate(&serde_json::json!(42.0)).is_ok());
        assert!(SchemaKind::integer().validate(&serde_json::json!(3.14)).is_err());

        assert!(SchemaKind::boolean().validate(&serde_json::json!(true)).is_ok());
        assert!(SchemaKind::boolean().validate(&serde_json::json!(1)).is_err());

        assert!(SchemaKind::any().validate(&serde_json::json!(null)).is_ok());
    }

    #[test]
    fn validate_literal() {
        let s = SchemaKind::literal("hello");
        assert!(s.validate(&serde_json::json!("hello")).is_ok());
        assert!(s.validate(&serde_json::json!("world")).is_err());
    }

    #[test]
    fn validate_array_and_optional_and_map() {
        let arr = SchemaKind::array(SchemaKind::string());
        assert!(arr.validate(&serde_json::json!(["a", "b"])).is_ok());
        assert!(arr.validate(&serde_json::json!([1])).is_err());

        let opt = SchemaKind::optional(SchemaKind::string());
        assert!(opt.validate(&serde_json::json!(null)).is_ok());
        assert!(opt.validate(&serde_json::json!("x")).is_ok());
        assert!(opt.validate(&serde_json::json!(1)).is_err());

        let map = SchemaKind::map(SchemaKind::integer());
        assert!(map.validate(&serde_json::json!({"a": 1, "b": 2})).is_ok());
        assert!(map.validate(&serde_json::json!({"a": "x"})).is_err());
    }

    #[test]
    fn validate_object_required_and_optional() {
        let schema = SchemaKind::object()
            .field("name", SchemaKind::string())
            .optional_field("nickname", SchemaKind::string())
            .build();
        assert!(schema.validate(&serde_json::json!({"name": "A"})).is_ok());
        assert!(
            schema
                .validate(&serde_json::json!({"name": "A", "nickname": "B"}))
                .is_ok()
        );
        assert!(schema.validate(&serde_json::json!({})).is_err());
        assert!(
            schema
                .validate(&serde_json::json!({"name": "A", "nickname": 1}))
                .is_err()
        );
    }

    #[test]
    fn validate_simple_enum() {
        let schema = SchemaKind::enumeration()
            .unit("low")
            .unit("high")
            .build();
        assert!(schema.validate(&serde_json::json!("low")).is_ok());
        assert!(schema.validate(&serde_json::json!("nope")).is_err());
        assert!(schema.validate(&serde_json::json!(1)).is_err());
    }

    #[test]
    fn validate_tagged_union() {
        let schema = SchemaKind::enumeration()
            .variant(
                "read",
                SchemaKind::object().field("path", SchemaKind::string()),
            )
            .variant(
                "write",
                SchemaKind::object()
                    .field("path", SchemaKind::string())
                    .field("content", SchemaKind::string()),
            )
            .unit("list")
            .build();

        assert!(
            schema
                .validate(&serde_json::json!({"type": "read", "path": "/x"}))
                .is_ok()
        );
        assert!(
            schema
                .validate(
                    &serde_json::json!({"type": "write", "path": "/x", "content": "y"})
                )
                .is_ok()
        );
        assert!(schema.validate(&serde_json::json!({"type": "list"})).is_ok());
        assert!(schema.validate(&serde_json::json!({"type": "read"})).is_err());
        assert!(schema.validate(&serde_json::json!({"type": "nope"})).is_err());
    }

    #[test]
    fn regex_primitives_match_canonical_values() {
        let s = regex::Regex::new(&SchemaKind::string().to_regex()).unwrap();
        assert!(s.is_match(r#""hello""#));
        let i = regex::Regex::new(&SchemaKind::integer().to_regex()).unwrap();
        assert!(i.is_match("42"));
        assert!(i.is_match("-1"));
        let b = regex::Regex::new(&SchemaKind::boolean().to_regex()).unwrap();
        assert!(b.is_match("true"));
        assert!(b.is_match("false"));
    }

    #[test]
    fn regex_compiles_for_all_basic_shapes() {
        let schemas = [
            SchemaKind::string(),
            SchemaKind::number(),
            SchemaKind::integer(),
            SchemaKind::boolean(),
            SchemaKind::any(),
            SchemaKind::array(SchemaKind::string()),
            SchemaKind::optional(SchemaKind::string()),
            SchemaKind::map(SchemaKind::integer()),
            SchemaKind::object()
                .field("name", SchemaKind::string())
                .build(),
            SchemaKind::enumeration().unit("a").unit("b").build(),
        ];
        for schema in schemas {
            let pattern = schema.to_regex();
            regex::Regex::new(&pattern).unwrap_or_else(|e| {
                panic!("regex compile failed for {schema:?}: {e}\n{pattern}")
            });
        }
    }

    #[test]
    fn empty_object_round_trips() {
        let schema = SchemaKind::object().build();
        assert_eq!(schema.to_gbnf(), "\"{}\"");
        let json = schema.to_json_schema();
        assert!(json["properties"].as_object().unwrap().is_empty());
        let re = regex::Regex::new(&schema.to_regex()).unwrap();
        assert!(re.is_match("{}"));
        assert!(schema.validate(&serde_json::json!({})).is_ok());
    }

    #[test]
    fn tagged_enum_inlines_object_fields_in_gbnf_and_regex() {
        let schema = SchemaKind::enumeration()
            .unit("stop")
            .variant(
                "move_to",
                SchemaKind::object()
                    .field("x", SchemaKind::number())
                    .field("y", SchemaKind::number()),
            )
            .build();
        let gbnf = schema.to_gbnf();
        // No nested opening braces — fields inline next to the type tag.
        assert!(!gbnf.contains("\"{{\""), "gbnf nested braces: {gbnf}");
        assert!(gbnf.contains('x') && gbnf.contains('y'));

        let re = regex::Regex::new(&schema.to_regex()).unwrap();
        assert!(re.is_match(r#"{"type":"stop"}"#));
        assert!(re.is_match(r#"{"type":"move_to","x":1,"y":2}"#));
        assert!(!re.is_match(r#"{"type":"move_to",{"x":1,"y":2}}"#));
    }

    #[test]
    fn tagged_enum_wraps_non_object_data_under_data_key() {
        let schema = SchemaKind::enumeration()
            .unit("none")
            .variant("names", SchemaKind::array(SchemaKind::string()))
            .build();

        let gbnf = schema.to_gbnf();
        assert!(gbnf.contains("data"));

        let re = regex::Regex::new(&schema.to_regex()).unwrap();
        assert!(re.is_match(r#"{"type":"names","data":["a","b"]}"#));
        assert!(!re.is_match(r#"{"type":"names",["a","b"]}"#));

        let json = schema.to_json_schema();
        assert!(json["oneOf"][1]["properties"]["data"].is_object());
    }

    #[test]
    fn grammar_source_routes_per_backend() {
        let schema = SchemaKind::object()
            .field("answer", SchemaKind::string())
            .build();
        let from_schema: GrammarSource = schema.clone().into();
        assert!(from_schema.to_json_schema().is_some());
        assert!(from_schema.to_regex().is_some());
        assert!(!from_schema.to_gbnf().is_empty());

        let raw = GrammarSource::Gbnf("root ::= \"x\"\n".into());
        assert!(raw.to_json_schema().is_none());
        assert!(raw.to_regex().is_none());
        assert_eq!(raw.to_gbnf(), "root ::= \"x\"\n");
    }

    #[test]
    fn schema_field_required_defaults_to_true_on_deserialize() {
        let f: SchemaField =
            serde_json::from_value(serde_json::json!({"name": "x", "schema": {"kind": "string"}}))
                .unwrap();
        assert!(f.required);
    }
}
