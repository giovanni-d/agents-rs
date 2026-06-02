//! Schema validation error types.

use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum SchemaValidationError {
    #[error("Type mismatch: expected {expected}, got {actual}")]
    TypeMismatch { expected: String, actual: String },

    #[error("Missing required field: {field}")]
    MissingField { field: String },

    #[error("Unknown field: {field}")]
    UnknownField { field: String },

    #[error("Invalid enum variant: {variant}")]
    InvalidVariant { variant: String },

    #[error("Invalid literal: expected {expected}, got {actual}")]
    InvalidLiteral { expected: String, actual: String },

    #[error("Invalid array element at index {index}: {error}")]
    InvalidArrayElement {
        index: usize,
        #[source]
        error: Box<SchemaValidationError>,
    },

    #[error("Invalid field '{field}': {error}")]
    InvalidField {
        field: String,
        #[source]
        error: Box<SchemaValidationError>,
    },

    #[error("Invalid map value for key '{key}': {error}")]
    InvalidMapValue {
        key: String,
        #[source]
        error: Box<SchemaValidationError>,
    },

    #[error("Invalid variant data for '{variant}': {error}")]
    InvalidVariantData {
        variant: String,
        #[source]
        error: Box<SchemaValidationError>,
    },
}
