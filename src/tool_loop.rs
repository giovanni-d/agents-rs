//! Tool-call driver. Re-invokes the agent until it returns a plain message,
//! executing any tool calls in between and feeding the results — alongside the
//! assistant's own request — back into the context. Emits `tracing` events at
//! every step; callers install a subscriber to surface them.

use crate::agent::{Agent, Response, ToolCall};
use crate::error::{AgentError, Result};
use crate::message::{Context, Message};
use crate::tool::ToolRegistry;

/// Drive an agent in a tool-call loop. While the agent returns tool calls,
/// execute them against `registry` and feed the results back as
/// [`Message::Tool`] entries. Returns the first plain-message response or
/// [`AgentError::MaxIterations`].
pub async fn run_with_tools(
    agent: &dyn Agent,
    registry: &ToolRegistry,
    mut ctx: Context,
    max_iters: usize,
) -> Result<String> {
    for i in 0..max_iters {
        tracing::info!(iter = i, messages = ctx.messages.len(), "tool_loop iter");
        match agent.run(ctx.clone()).await? {
            Response::Message(s) => {
                tracing::info!(iter = i, chars = s.len(), "tool_loop final message");
                return Ok(s);
            }
            Response::ToolCalls(calls) => {
                let names: Vec<&str> = calls.iter().map(|c| c.name.as_str()).collect();
                tracing::info!(iter = i, calls = ?names, "tool_loop dispatching");
                ctx.push(Message::Assistant {
                    content: render_tool_envelope(&calls),
                });
                for call in calls {
                    tracing::debug!(tool = %call.name, args = %call.arguments, "tool exec");
                    let tool = registry
                        .get(&call.name)
                        .ok_or_else(|| AgentError::UnknownTool(call.name.clone()))?;
                    let result = tool.call(call.arguments).await?;
                    tracing::debug!(tool = %call.name, result = %result, "tool result");
                    ctx.push(Message::Tool {
                        name: call.name,
                        content: result.to_string(),
                    });
                }
            }
        }
    }
    Err(AgentError::MaxIterations(max_iters))
}

/// Reconstructs the JSON envelope an agent was instructed to emit, so the
/// next turn's prompt includes the assistant's own request alongside the
/// tool result. Single call → object; many → array.
fn render_tool_envelope(calls: &[ToolCall]) -> String {
    if calls.len() == 1 {
        serde_json::json!({
            "tool": calls[0].name,
            "args": calls[0].arguments,
        })
        .to_string()
    } else {
        let arr: Vec<serde_json::Value> = calls
            .iter()
            .map(|c| serde_json::json!({ "tool": c.name, "args": c.arguments }))
            .collect();
        serde_json::Value::Array(arr).to_string()
    }
}
