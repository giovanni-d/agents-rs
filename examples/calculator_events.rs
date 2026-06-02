//! Multi-tool calculator demo exposing the raw [`LoopEvent`] stream from [`run_with_tools_streaming`].
//!
//! Run with:
//!
//! ```sh
//! cargo run --example calculator_events --features metal --release
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
    use std::sync::Arc;

    use agents_rs::{
        Context, LocalAgent, LoopEvent, OutputSchemaRequest, run_with_tools_streaming,
    };
    use tokio_stream::StreamExt;

    use crate::calc_common::{
        CALC_USER_PROMPT, build_calc_config, build_calc_registry, build_calc_system_prompt,
        final_answer_schema, init_tracing, model_path_from_env,
    };

    init_tracing();

    let final_schema = final_answer_schema();
    let registry = build_calc_registry();

    let model_path = model_path_from_env();
    let local = LocalAgent::from_config(build_calc_config(&model_path))?.with_tools(&registry);

    let system = build_calc_system_prompt(&final_schema);
    let ctx = Context::new()
        .with_system(&system)
        .with_user(CALC_USER_PROMPT);

    let schema_request = OutputSchemaRequest::new(final_schema, "FinalAnswer");
    let mut stream =
        run_with_tools_streaming(Arc::new(local), registry, ctx, 10, Some(schema_request));

    while let Some(item) = stream.next().await {
        match item? {
            LoopEvent::Text(t) => {
                print!("{t}");
                std::io::stdout().flush().ok();
            }
            LoopEvent::ToolStarted { name, .. } => {
                print!("\n  {name}(");
                std::io::stdout().flush().ok();
            }
            LoopEvent::ToolArgumentsFragment { fragment, .. } => {
                print!("{fragment}");
                std::io::stdout().flush().ok();
            }
            LoopEvent::ToolDispatched { result, .. } => {
                println!(") → {result}");
            }
            LoopEvent::Done(answer) => {
                println!("\n[done] {answer}");
                break;
            }
        }
    }
    Ok(())
}
