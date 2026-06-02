//! Streaming agents: emit [`Chunk`]s as they're generated.

mod chunk;
mod collector;
mod loop_event;
mod streaming_agent;
mod tool_dispatch;

pub use chunk::{Chunk, ToolEvent};
pub use collector::StreamCollector;
pub use loop_event::{LoopEvent, run_with_tools_streaming};
pub use streaming_agent::{
    BoxStreamingAgent, CharStreamAgent, FailingStreamAgent, StreamingAgent, StreamingAgentExt,
    StreamingEchoAgent, WordStreamAgent,
};
pub use tool_dispatch::ToolDispatchingAgent;
