//! Tool-call driver: loops an agent, executing its tool calls until it returns a plain message.

use crate::agent::{Agent, Response, ToolCall};
use crate::error::{AgentError, Result};
use crate::message::{Context, Message};
use crate::structured::OutputSchemaRequest;
use crate::tool::ToolRegistry;
use crate::tool_response::ToolResult;

/// Drive an agent in a tool-call loop until it returns a plain message.
///
/// # Errors
/// Returns [`AgentError::MaxIterations`] if `max_iters` elapses without a plain message.
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
                ctx.push(Message::assistant(render_tool_envelope(&calls)));
                for call in calls {
                    tracing::debug!(tool = %call.name, args = %call.arguments, "tool exec");
                    if !registry.contains(&call.name) {
                        return Err(AgentError::UnknownTool(call.name));
                    }
                    let result = registry.execute(&call.name, call.arguments.clone()).await?;
                    tracing::debug!(tool = %call.name, result = %result, "tool result");
                    ctx.push(Message::tool(call.id, call.name, ToolResult::Json(result)));
                }
            }
        }
    }
    Err(AgentError::MaxIterations(max_iters))
}

/// Like [`run_with_tools`], but appends a schema-constrained final turn that returns structured output.
///
/// The tool phase exits early on a plain message, a repeated identical tool call, or `max_iters`.
pub async fn run_with_tools_structured(
    agent: &dyn Agent,
    registry: &ToolRegistry,
    mut ctx: Context,
    max_iters: usize,
    final_schema: OutputSchemaRequest,
) -> Result<String> {
    let mut last_call: Option<(String, serde_json::Value)> = None;
    for i in 0..max_iters {
        tracing::info!(iter = i, messages = ctx.messages.len(), "tool_loop iter");
        match agent.run(ctx.clone()).await? {
            Response::Message(s) => {
                // Skip the schema turn when the model's message already validates.
                if validates_against_schema(&s, &final_schema) {
                    tracing::info!(
                        iter = i,
                        chars = s.len(),
                        "tool_loop natural termination — schema-valid, returning verbatim"
                    );
                    return Ok(s);
                }
                tracing::info!(
                    iter = i,
                    chars = s.len(),
                    "tool_loop natural termination — non-conforming, running schema turn"
                );
                // Keep the off-shape answer in context so the schema turn can canonicalize it.
                ctx.push(Message::assistant(s));
                break;
            }
            Response::ToolCalls(calls) => {
                if calls.len() == 1 {
                    let curr = (calls[0].name.clone(), calls[0].arguments.clone());
                    if last_call.as_ref() == Some(&curr) {
                        tracing::info!(
                            iter = i,
                            tool = %curr.0,
                            "tool_loop detected repeat — switching to schema turn"
                        );
                        break;
                    }
                    last_call = Some(curr);
                }
                let names: Vec<&str> = calls.iter().map(|c| c.name.as_str()).collect();
                tracing::info!(iter = i, calls = ?names, "tool_loop dispatching");
                ctx.push(Message::assistant(render_tool_envelope(&calls)));
                for call in calls {
                    tracing::debug!(tool = %call.name, args = %call.arguments, "tool exec");
                    if !registry.contains(&call.name) {
                        return Err(AgentError::UnknownTool(call.name));
                    }
                    let result = registry.execute(&call.name, call.arguments.clone()).await?;
                    tracing::debug!(tool = %call.name, result = %result, "tool result");
                    ctx.push(Message::tool(call.id, call.name, ToolResult::Json(result)));
                }
            }
        }
    }
    tracing::info!("tool_loop schema turn");
    let schema_for_check = final_schema.clone();
    let ctx = ctx.with_output_schema_request(final_schema);
    match agent.run(ctx).await? {
        Response::Message(s) => {
            if !validates_against_schema(&s, &schema_for_check) {
                return Err(AgentError::Other(format!(
                    "schema turn produced non-conforming output for {}:\n{}",
                    schema_for_check.type_name, s
                )));
            }
            Ok(s)
        }
        Response::ToolCalls(_) => Err(AgentError::Other(
            "schema turn unexpectedly returned tool calls".into(),
        )),
    }
}

/// True iff `text` is JSON satisfying the request's schema; pre-built GBNF grammars pass on JSON parse alone.
fn validates_against_schema(text: &str, req: &OutputSchemaRequest) -> bool {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(text) else {
        return false;
    };
    match &req.grammar {
        crate::schema::GrammarSource::Schema(s) => s.validate(&value).is_ok(),
        crate::schema::GrammarSource::Gbnf(_) => true,
    }
}

/// Reconstructs the JSON envelope the agent was instructed to emit (single call → object, many → array).
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
