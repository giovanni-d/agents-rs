//! Minimal agent runtime.
//!
//! Three pieces:
//! - [`Agent`]: takes a [`Context`], returns a [`Response`] (a message or tool calls).
//! - [`Tool`] / [`ToolRegistry`] / [`run_with_tools`]: register callable tools and drive
//!   a tool-call loop until the agent emits a plain message.
//! - [`StructuredOutput`] / [`run_structured`]: parse the agent's final message into a typed value.
//!
//! Optional pieces: [`LoggingAgent`] wraps any agent and prints lifecycle events
//! to stderr. Enable the `llama-cpp` feature for [`LocalAgent`], a local-model
//! agent backed by llama.cpp.

pub mod agent;
pub mod error;
pub mod logging;
pub mod message;
pub mod schema;
pub mod stream;
pub mod structured;
pub mod tool;
pub mod tool_loop;
pub mod tool_parser;
pub mod tool_response;
pub mod usage;

#[cfg(feature = "llama-cpp")]
pub mod llama;

pub use agent::{Agent, FnAgent, Response, ToolCall, fn_agent};
pub use error::{AgentError, BoxFuture, Result};
pub use logging::LoggingAgent;
pub use message::{ContentPart, Context, Message, MessageContent, Role};
pub use stream::{
    BoxStreamingAgent, CharStreamAgent, Chunk, FailingStreamAgent, LoopEvent, StreamCollector,
    StreamingAgent, StreamingAgentExt, StreamingEchoAgent, ToolDispatchingAgent, ToolEvent,
    WordStreamAgent, run_with_tools_streaming,
};
pub use schema::{
    AgentData, EnumBuilder, GrammarSource, ObjectBuilder, SchemaField, SchemaKind,
    SchemaValidationError, SchemaVariant, TypedAgent, TypedAgentAdapter, TypedAgentExt,
};
pub use structured::{
    OutputFormat, OutputSchema, OutputSchemaRequest, StructuredOutput, run_structured,
};
pub use tool_response::{ToolResponse, ToolResult};
pub use tool::{
    DynTool, FnTool, Tool, ToolDefinition, ToolDefinitionBuilder, ToolOutput, ToolRegistry,
    fn_tool,
};
pub use tool_loop::{run_with_tools, run_with_tools_structured};
pub use tool_parser::extract_tool_calls;
pub use usage::UsageMetrics;

#[cfg(feature = "llama-cpp")]
pub use llama::{LocalAgent, LocalConfig};
