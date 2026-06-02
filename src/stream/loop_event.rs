//! [`run_with_tools_streaming`] drives a [`StreamingAgent`] through a
//! tool-call loop and emits [`LoopEvent`]s (text, tool lifecycle, final
//! `Done`).

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;
use tokio_stream::{Stream, StreamExt};

use crate::{
    AgentError, Context, Message, OutputSchemaRequest, Result, ToolRegistry, ToolResult,
};

use super::{Chunk, StreamingAgent, ToolEvent};

/// One emission from [`run_with_tools_streaming`].
///
/// Lifecycle of a single tool call, as observed by a consumer:
///
/// ```text
///   ToolStarted { id, name }
///   ToolArgumentsFragment { id, fragment }   (one or more, as the
///   ToolArgumentsFragment { id, fragment }    model streams args)
///   ...
///   ToolDispatched { id, name, args, result }
/// ```
///
/// Text fragments from the assistant interleave with these on the same
/// stream.
#[derive(Clone, Debug)]
pub enum LoopEvent {
    Text(String),
    ToolStarted { id: String, name: String },
    /// Tool arguments are streaming in. Concatenating every fragment
    /// for a given `id` yields the full arguments JSON.
    ToolArgumentsFragment { id: String, fragment: String },
    /// A tool call completed: arguments parsed, executed, result attached.
    ToolDispatched {
        id: String,
        name: String,
        args: Value,
        result: Value,
    },
    /// Loop finished; payload is the assistant's final message text.
    Done(String),
}

/// Drive a [`StreamingAgent`] through a tool-call loop and stream
/// [`LoopEvent`]s. Optional `final_schema` runs a canonicalization turn
/// if the model's natural-termination output doesn't validate.
pub fn run_with_tools_streaming<S>(
    agent: Arc<S>,
    registry: ToolRegistry,
    ctx: Context,
    max_iters: usize,
    final_schema: Option<OutputSchemaRequest>,
) -> impl Stream<Item = Result<LoopEvent>>
where
    S: StreamingAgent + 'static,
    S::Stream: Send + Unpin + 'static,
{
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<LoopEvent>>(64);
    tokio::spawn(async move {
        if let Err(e) = drive(agent, registry, ctx, max_iters, final_schema, &tx).await {
            let _ = tx.send(Err(e)).await;
        }
    });
    tokio_stream::wrappers::ReceiverStream::new(rx)
}

async fn drive<S>(
    agent: Arc<S>,
    registry: ToolRegistry,
    mut ctx: Context,
    max_iters: usize,
    final_schema: Option<OutputSchemaRequest>,
    tx: &tokio::sync::mpsc::Sender<Result<LoopEvent>>,
) -> Result<()>
where
    S: StreamingAgent + 'static,
    S::Stream: Send + Unpin + 'static,
{
    for _ in 0..max_iters {
        let outcome = run_one_turn(&agent, &registry, &mut ctx, tx).await?;
        match outcome {
            TurnOutcome::Finished(text) => {
                if let Some(req) = &final_schema
                    && !validates_against_schema(&text, req)
                {
                    ctx.push(Message::assistant(text));
                    break;
                }
                let _ = tx.send(Ok(LoopEvent::Done(text))).await;
                return Ok(());
            }
            TurnOutcome::ContinueAfterTools => continue,
        }
    }
    if let Some(req) = final_schema {
        let ctx = ctx.with_output_schema_request(req);
        let mut stream = agent.run_stream(ctx);
        let mut text = String::new();
        while let Some(item) = stream.next().await {
            match item? {
                Chunk::Text(t) => {
                    text.push_str(&t);
                    if tx.send(Ok(LoopEvent::Text(t.clone()))).await.is_err() {
                        return Ok(());
                    }
                }
                Chunk::End => break,
                _ => {}
            }
        }
        let _ = tx.send(Ok(LoopEvent::Done(text))).await;
        Ok(())
    } else {
        Err(AgentError::MaxIterations(max_iters))
    }
}

enum TurnOutcome {
    Finished(String),
    ContinueAfterTools,
}

async fn run_one_turn<S>(
    agent: &Arc<S>,
    registry: &ToolRegistry,
    ctx: &mut Context,
    tx: &tokio::sync::mpsc::Sender<Result<LoopEvent>>,
) -> Result<TurnOutcome>
where
    S: StreamingAgent + 'static,
    S::Stream: Send + Unpin + 'static,
{
    let mut stream = agent.run_stream(ctx.clone());
    let mut text_buf = String::new();
    let mut tools: HashMap<String, ToolBuilder> = HashMap::new();
    let mut order: Vec<String> = Vec::new();

    while let Some(item) = stream.next().await {
        match item? {
            Chunk::Text(t) => {
                text_buf.push_str(&t);
                if tx.send(Ok(LoopEvent::Text(t))).await.is_err() {
                    // Consumer dropped — bail.
                    return Ok(TurnOutcome::Finished(text_buf));
                }
            }
            Chunk::Tool(ToolEvent::Started { id, name }) => {
                if !tools.contains_key(&id) {
                    order.push(id.clone());
                    if tx
                        .send(Ok(LoopEvent::ToolStarted {
                            id: id.clone(),
                            name: name.clone(),
                        }))
                        .await
                        .is_err()
                    {
                        return Ok(TurnOutcome::Finished(text_buf));
                    }
                }
                tools.entry(id).or_insert_with(|| ToolBuilder {
                    name,
                    args: String::new(),
                    result: None,
                });
            }
            Chunk::Tool(ToolEvent::Arguments { id, fragment }) => {
                if let Some(b) = tools.get_mut(&id) {
                    b.args.push_str(&fragment);
                }
                if tx
                    .send(Ok(LoopEvent::ToolArgumentsFragment { id, fragment }))
                    .await
                    .is_err()
                {
                    return Ok(TurnOutcome::Finished(text_buf));
                }
            }
            Chunk::Tool(ToolEvent::Finished { id }) => {
                // Dispatch this tool the moment its envelope closes —
                // the consumer sees `ToolDispatched` live, before the
                // next tool starts (or the stream ends).
                if let Some(b) = tools.get_mut(&id)
                    && b.result.is_none()
                {
                    let args: Value = serde_json::from_str(&b.args)
                        .unwrap_or_else(|_| Value::Object(Default::default()));
                    let result = registry.execute(&b.name, args.clone()).await?;
                    b.result = Some(result.clone());
                    if tx
                        .send(Ok(LoopEvent::ToolDispatched {
                            id: id.clone(),
                            name: b.name.clone(),
                            args,
                            result,
                        }))
                        .await
                        .is_err()
                    {
                        return Ok(TurnOutcome::Finished(text_buf));
                    }
                }
            }
            Chunk::Usage(_) => {}
            Chunk::End => break,
        }
    }

    if tools.is_empty() {
        return Ok(TurnOutcome::Finished(text_buf));
    }

    // Tool dispatches already fired live; build the assistant tool-call
    // envelope + tool-result messages for the next iteration.
    ctx.push(Message::assistant(render_envelope(&order, &tools)));
    for id in &order {
        let b = tools
            .get(id)
            .expect("tool builder present for ordered id");
        let result = b
            .result
            .clone()
            .unwrap_or(Value::Object(Default::default()));
        ctx.push(Message::tool(
            id.clone(),
            b.name.clone(),
            ToolResult::Json(result),
        ));
    }
    Ok(TurnOutcome::ContinueAfterTools)
}

struct ToolBuilder {
    name: String,
    args: String,
    result: Option<Value>,
}

fn render_envelope(order: &[String], tools: &HashMap<String, ToolBuilder>) -> String {
    let calls: Vec<Value> = order
        .iter()
        .filter_map(|id| {
            let b = tools.get(id)?;
            let args: Value = serde_json::from_str(&b.args).ok()?;
            Some(serde_json::json!({ "tool": b.name, "args": args }))
        })
        .collect();
    if calls.len() == 1 {
        calls.into_iter().next().unwrap().to_string()
    } else {
        Value::Array(calls).to_string()
    }
}

fn validates_against_schema(text: &str, req: &OutputSchemaRequest) -> bool {
    let Ok(value) = serde_json::from_str::<Value>(text) else {
        return false;
    };
    match &req.grammar {
        crate::schema::GrammarSource::Schema(s) => s.validate(&value).is_ok(),
        crate::schema::GrammarSource::Gbnf(_) => true,
    }
}
