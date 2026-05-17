use agents_rs::{
    AgentError, Context, FnTool, Response, StructuredOutput, ToolCall, ToolRegistry, fn_agent,
    run_structured, run_with_tools,
};
use serde::Deserialize;
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};

#[tokio::test]
async fn tool_loop_runs_tool_then_returns_message() {
    let echo = FnTool::new(
        "echo",
        "Echo back the input.",
        json!({ "type": "object" }),
        |args| async move { Ok(args) },
    );
    let registry = ToolRegistry::new().register(echo);

    let step = AtomicUsize::new(0);
    let agent = fn_agent(move |_ctx| {
        let n = step.fetch_add(1, Ordering::SeqCst);
        async move {
            if n == 0 {
                Ok(Response::tool_calls(vec![ToolCall {
                    id: "1".into(),
                    name: "echo".into(),
                    arguments: json!({ "v": 42 }),
                }]))
            } else {
                Ok(Response::message("done"))
            }
        }
    });

    let out = run_with_tools(&agent, &registry, Context::new(), 4).await.unwrap();
    assert_eq!(out, "done");
}

#[tokio::test]
async fn unknown_tool_errors() {
    let registry = ToolRegistry::new();
    let agent = fn_agent(|_ctx| async {
        Ok(Response::tool_calls(vec![ToolCall {
            id: "1".into(),
            name: "missing".into(),
            arguments: json!({}),
        }]))
    });
    let err = run_with_tools(&agent, &registry, Context::new(), 2).await.unwrap_err();
    assert!(matches!(err, AgentError::UnknownTool(_)));
}

#[derive(Debug, Deserialize, PartialEq)]
struct Point {
    x: i32,
    y: i32,
}

impl StructuredOutput for Point {
    fn schema() -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "x": { "type": "integer" },
                "y": { "type": "integer" }
            },
            "required": ["x", "y"]
        })
    }
}

#[tokio::test]
async fn structured_output_parses() {
    let agent = fn_agent(|_ctx| async {
        Ok(Response::message(json!({ "x": 1, "y": 2 }).to_string()))
    });
    let p: Point = run_structured(&agent, Context::new()).await.unwrap();
    assert_eq!(p, Point { x: 1, y: 2 });
}
