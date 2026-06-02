//! Multi-tool calculator demo: a local GGUF model chains arithmetic tool calls to evaluate an expression.
//!
//! Run with:
//!
//! ```sh
//! RUST_LOG=agents_rs=info cargo run --example calculator --features cuda --release
//! ```

#[cfg(not(feature = "llama-cpp"))]
fn main() {
    eprintln!("Build with --features llama-cpp (or cuda / metal) to run this example.");
}

#[cfg(feature = "llama-cpp")]
#[path = "calc_common.rs"]
mod calc_common;

#[cfg(feature = "llama-cpp")]
#[tokio::main(flavor = "current_thread")]
async fn main() -> agents_rs::Result<()> {
    use std::io::Write;

    use agents_rs::{
        AgentError, Chunk, Context, LocalAgent, OutputSchemaRequest, StreamingAgent,
        ToolDispatchingAgent,
    };
    use serde_json::Value;
    use tokio_stream::StreamExt;

    use crate::calc_common::{
        CALC_USER_PROMPT, FinalAnswer, build_calc_config, build_calc_registry,
        build_calc_system_prompt, final_answer_schema, init_tracing, model_path_from_env,
    };

    init_tracing();

    let final_schema = final_answer_schema();
    let registry = build_calc_registry();

    let model_path = model_path_from_env();
    let local = LocalAgent::from_config(build_calc_config(&model_path))?.with_tools(&registry);

    let system = build_calc_system_prompt(&final_schema);

    let schema_request = OutputSchemaRequest::new(final_schema.clone(), "FinalAnswer");
    let agent = ToolDispatchingAgent::new(local, registry)
        .with_max_iters(10)
        .with_final_schema(schema_request);

    let ctx = Context::new()
        .with_system(&system)
        .with_user(CALC_USER_PROMPT);

    let mut stream = agent.run_stream(ctx);
    let mut raw = String::new();
    print!("answer: ");
    std::io::stdout().flush().ok();
    while let Some(item) = stream.next().await {
        match item? {
            Chunk::Text(t) => {
                print!("{t}");
                std::io::stdout().flush().ok();
                raw.push_str(&t);
            }
            Chunk::End => {
                println!();
                break;
            }
            _ => {}
        }
    }

    let value: Value = serde_json::from_str(&raw)
        .map_err(|e| AgentError::Other(format!("model did not return JSON: {e}\n{raw}")))?;
    final_schema
        .validate(&value)
        .map_err(|e| AgentError::Other(format!("schema validation failed: {e}\n{raw}")))?;
    let parsed: FinalAnswer = serde_json::from_value(value)?;
    println!("Final answer: {}", parsed.answer);
    Ok(())
}
