//! Schema types. [`SchemaKind`] is a universal schema convertible to JSON Schema,
//! GBNF, or regex; [`TypedAgent`] uses the same schemas for compile-time typed I/O.

mod agent_data;
mod kind;
mod typed_agent;
mod validation;

pub use agent_data::AgentData;
pub use kind::{
    EnumBuilder, GrammarSource, ObjectBuilder, SchemaField, SchemaKind, SchemaVariant,
};
pub use typed_agent::{TypedAgent, TypedAgentAdapter, TypedAgentExt};
pub use validation::SchemaValidationError;
