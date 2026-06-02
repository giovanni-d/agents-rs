//! [`ToolDispatchingAgent`] wraps a [`StreamingAgent`], intercepting
//! tool calls and looping until the model emits a tool-free reply.
//! Consumers only see `Chunk::Text` for the final answer plus `Chunk::End`.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;
use tokio_stream::{Stream, StreamExt};

use crate::{
    AgentError, Context, Message, OutputSchemaRequest, Result, ToolRegistry, ToolResult,
};

use super::{Chunk, StreamingAgent, ToolEvent};

/// Wraps a [`StreamingAgent`], dispatching tool calls as they stream in
/// and looping until the model emits a tool-free message.
pub struct ToolDispatchingAgent<S> {
    inner: Arc<S>,
    registry: ToolRegistry,
    max_iters: usize,
    final_schema: Option<OutputSchemaRequest>,
}

impl<S> ToolDispatchingAgent<S> {
    pub fn new(inner: S, registry: ToolRegistry) -> Self {
        Self {
            inner: Arc::new(inner),
            registry,
            max_iters: 10,
            final_schema: None,
        }
    }

    pub fn with_max_iters(mut self, n: usize) -> Self {
        self.max_iters = n;
        self
    }

    /// Attach a schema request to canonicalize the final answer; the
    /// schema turn is skipped if natural-termination text already validates.
    pub fn with_final_schema(mut self, schema: OutputSchemaRequest) -> Self {
        self.final_schema = Some(schema);
        self
    }
}

impl<S> StreamingAgent for ToolDispatchingAgent<S>
where
    S: StreamingAgent + 'static,
    S::Stream: Send + Unpin + 'static,
{
    type Stream = Pin<Box<dyn Stream<Item = Result<Chunk>> + Send>>;

    fn run_stream(&self, ctx: Context) -> Self::Stream {
        let agent = Arc::clone(&self.inner);
        let registry = self.registry.clone();
        let max_iters = self.max_iters;
        let final_schema = self.final_schema.clone();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Chunk>>(64);
        tokio::spawn(async move {
            if let Err(e) = drive(agent, registry, ctx, max_iters, final_schema, &tx).await {
                let _ = tx.send(Err(e)).await;
            }
            let _ = tx.send(Ok(Chunk::End)).await;
        });
        Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
    }
}

async fn drive<S>(
    agent: Arc<S>,
    registry: ToolRegistry,
    mut ctx: Context,
    max_iters: usize,
    final_schema: Option<OutputSchemaRequest>,
    tx: &tokio::sync::mpsc::Sender<Result<Chunk>>,
) -> Result<()>
where
    S: StreamingAgent + 'static,
    S::Stream: Send + Unpin + 'static,
{
    for iter in 0..max_iters {
        let mut stream = agent.run_stream(ctx.clone());
        // Text is BUFFERED, not forwarded. The wrapper's contract is
        // that consumers only see the final-answer text — intermediate
        // reasoning prose from iterations that don't validate against
        // the schema would otherwise pollute the consumer's output and
        // get re-canonicalized on the schema turn anyway.
        let mut text_buf = String::new();
        let mut tools: HashMap<String, ToolBuilder> = HashMap::new();
        let mut order: Vec<String> = Vec::new();

        while let Some(item) = stream.next().await {
            match item? {
                Chunk::Text(t) => {
                    text_buf.push_str(&t);
                }
                Chunk::Tool(ToolEvent::Started { id, name }) => {
                    if !tools.contains_key(&id) {
                        order.push(id.clone());
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
                }
                Chunk::Tool(ToolEvent::Finished { id }) => {
                    // Dispatch as soon as the tool's envelope closes —
                    // matches `run_with_tools_streaming`.
                    if let Some(b) = tools.get_mut(&id)
                        && b.result.is_none()
                    {
                        let args: Value = serde_json::from_str(&b.args)
                            .unwrap_or_else(|_| Value::Object(Default::default()));
                        let result = registry.execute(&b.name, args).await?;
                        b.result = Some(result);
                    }
                }
                Chunk::Usage(_) => {}
                Chunk::End => break,
            }
        }

        let calls_repr: Vec<String> = order
            .iter()
            .filter_map(|id| tools.get(id))
            .map(|b| {
                let result = b
                    .result
                    .as_ref()
                    .map(|r| serde_json::to_string(r).unwrap_or_default())
                    .unwrap_or_else(|| "<pending>".into());
                format!("{}({}) → {}", b.name, b.args.trim(), result)
            })
            .collect();
        tracing::debug!(
            iter = iter + 1,
            tool_calls = order.len(),
            calls = ?calls_repr,
            text_bytes = text_buf.len(),
            "dispatch iter"
        );

        if tools.is_empty() {
            let valid = match &final_schema {
                Some(req) => validates_against_schema(&text_buf, req),
                None => true, // no schema requested → any text counts
            };
            if valid {
                // Flush buffered text as one chunk so the consumer
                // sees the final answer.
                if !text_buf.is_empty() {
                    let _ = tx.send(Ok(Chunk::Text(text_buf))).await;
                }
                return Ok(());
            }
            // Non-conforming text — fall through to schema turn.
            ctx.push(Message::assistant(text_buf));
            break;
        }

        ctx.push(Message::assistant(render_envelope(&order, &tools)));
        for id in &order {
            let b = tools.get(id).expect("tool builder present");
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
    }

    if let Some(req) = final_schema {
        let ctx = ctx.with_output_schema_request(req);
        let mut stream = agent.run_stream(ctx);
        while let Some(item) = stream.next().await {
            match item? {
                Chunk::Text(t) => {
                    if tx.send(Ok(Chunk::Text(t))).await.is_err() {
                        return Ok(());
                    }
                }
                Chunk::End => break,
                _ => {}
            }
        }
        Ok(())
    } else {
        Err(AgentError::MaxIterations(max_iters))
    }
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
