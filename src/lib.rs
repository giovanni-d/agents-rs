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
pub mod structured;
pub mod tool;
pub mod tool_loop;
pub mod tool_parser;

#[cfg(feature = "llama-cpp")]
pub mod llama;

pub use agent::{Agent, FnAgent, Response, ToolCall, fn_agent};
pub use error::{AgentError, BoxFuture, Result};
pub use logging::LoggingAgent;
pub use message::{ContentPart, Context, Message, MessageContent, Role};
pub use structured::{StructuredOutput, run_structured};
pub use tool::{FnTool, Tool, ToolRegistry};
pub use tool_loop::run_with_tools;
pub use tool_parser::extract_tool_calls;

#[cfg(feature = "llama-cpp")]
pub use llama::{LocalAgent, LocalConfig};
